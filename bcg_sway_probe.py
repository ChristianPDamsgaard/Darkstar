"""
bcg_sway_probe.py
-----------------
FFT-based band-energy analysis to investigate whether the STANDING heart-rate
failures come from a low-frequency, in-band oscillation (consistent with
postural sway) rather than 50 Hz mains.

The preprocess_bcg function is reproduced UNCHANGED from thesis Appendix A.5.
Everything else (FFT spectra, band-energy ratio, matching seated/standing
files by subject) is new analysis scaffolding -- a diagnostic, NOT thesis code.

The argument it supports, in two parts:
  1. RAW vs FILTERED spectrum: the 50 Hz spike in the raw signal is removed by
     the 0.7-12 Hz bandpass, so 50 Hz cannot be what the ACF locks onto.
  2. BAND ENERGY: how much of the surviving in-band energy sits in a low
     "sway band" vs a "cardiac band". If the sway-band share rises going from
     seated to standing -- especially for the subjects whose standing estimate
     failed -- that points to a low-frequency oscillation, not mains noise.

NOTE: the band boundaries are a reasonable choice, not a law. Sway and a slow
heartbeat overlap in frequency, so a high sway-band share is CONSISTENT WITH
sway but does not prove it; only a direct sway sensor (accelerometer) settles
that. What the spectra DO prove cleanly is "not 50 Hz".

Examples:
  python bcg_sway_probe.py --subject "Christian Crone"   # seated vs standing spectra
  python bcg_sway_probe.py --subject Simon --out simon.png
  python bcg_sway_probe.py --list                        # matchable subjects
  python bcg_sway_probe.py --all                         # band-energy table, all subjects
Folders default to BCG_Sitting and BCG_Standing in the current directory;
override with --sit-dir / --stand-dir.
"""

import argparse
import glob
import os
import re
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt


# ============================================================================
# Reproduced verbatim from thesis Appendix A.5.
# ============================================================================

def preprocess_bcg(voltage: np.ndarray, fs: float) -> np.ndarray:
    v = signal.detrend(np.asarray(voltage, dtype=float), type="linear")
    nyq = fs / 2.0
    hi = min(12.0, 0.45 * nyq)
    sos = butter(4, [0.7, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, v)


# ============================================================================
# New diagnostic scaffolding (NOT from the thesis).
# ============================================================================

# Frequency bands (Hz). Sway sits below the slowest plausible heart rate.
SWAY_BAND = (0.7, 1.0)      # below ~60 bpm
CARDIAC_BAND = (1.0, 2.5)   # ~60-150 bpm
PASSBAND = (0.7, 12.0)      # what the bandpass keeps


def load_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    cols = data.dtype.names
    if cols and "time" in cols and "voltage" in cols:
        return data["time"], data["voltage"]
    raw = np.genfromtxt(path, delimiter=",", skip_header=1)
    return raw[:, 0], raw[:, 1]


def clean_name(filename):
    """Pull a comparable subject key out of a filename."""
    base = os.path.basename(filename)
    for tag in ("_standing_BCG", "_sitting_BCG", "_BCG"):
        i = base.find(tag)
        if i != -1:
            base = base[:i]
            break
    base = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), base)
    return base.replace("_", " ").strip()


def index_folder(folder):
    """Map subject-key -> filepath for a folder of CSVs."""
    out = {}
    if folder and os.path.isdir(folder):
        for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            out[clean_name(f).lower()] = f
    return out


def fs_of(t):
    dt = np.diff(t); dt = dt[dt > 0]
    return 1.0 / np.median(dt)


def amp_spectrum(sig, fs):
    """One-sided amplitude spectrum."""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(sig)) / n
    return freqs, mag


def band_fraction(x, fs, band):
    """Fraction of in-passband POWER that falls in `band`."""
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    p = np.abs(np.fft.rfft(x)) ** 2
    inband = (freqs >= PASSBAND[0]) & (freqs <= PASSBAND[1])
    sel = (freqs >= band[0]) & (freqs < band[1])
    denom = p[inband].sum()
    return p[sel].sum() / denom if denom > 0 else float("nan")


def analyse(path):
    t, v = load_csv(path)
    fs = fs_of(t)
    x = preprocess_bcg(v, fs)
    sway = band_fraction(x, fs, SWAY_BAND)
    card = band_fraction(x, fs, CARDIAC_BAND)
    return t, v, x, fs, sway, card


def find(index, query):
    q = query.lower()
    return [k for k in index if q in k]


def main():
    ap = argparse.ArgumentParser(description="BCG sway probe (FFT): raw-vs-filtered spectra + band energy.")
    ap.add_argument("--subject", help="subject name or substring, e.g. 'Christian Crone' or 'Simon'")
    ap.add_argument("--sit-dir", default="BCG_Sitting")
    ap.add_argument("--stand-dir", default="BCG_Standing")
    ap.add_argument("--out", default=None, help="save the figure to this path (default: show on screen)")
    ap.add_argument("--list", action="store_true", help="list subjects matchable in both folders")
    ap.add_argument("--all", action="store_true", help="print band-energy table for all matched subjects")
    args = ap.parse_args()

    sit = index_folder(args.sit_dir)
    stand = index_folder(args.stand_dir)

    if args.list:
        both = sorted(set(sit) & set(stand))
        print(f"Subjects present in BOTH folders ({len(both)}):")
        for k in both:
            print("  ", clean_name(sit[k]))
        only_stand = sorted(set(stand) - set(sit))
        if only_stand:
            print(f"\nStanding-only ({len(only_stand)}):")
            for k in only_stand:
                print("  ", clean_name(stand[k]))
        return

    if args.all:
        both = sorted(set(sit) & set(stand))
        if not both:
            print("No subjects found in both folders. Check --sit-dir / --stand-dir.")
            return
        print(f"{'Subject':<26}{'sit sway%':>10}{'stand sway%':>12}{'change':>9}")
        print("-" * 57)
        dsum = []
        for k in both:
            _, _, _, _, ss, _ = analyse(sit[k])
            _, _, _, _, ts, _ = analyse(stand[k])
            d = ts - ss
            dsum.append(d)
            print(f"{clean_name(sit[k]):<26}{ss*100:>9.1f}{ts*100:>11.1f}{d*100:>+9.1f}")
        print("-" * 57)
        print(f"Mean change in sway-band share (sit->stand): {np.mean(dsum)*100:>+.1f} pts")
        print("SWAY_BAND = %.1f-%.1f Hz (below slowest plausible HR)." % SWAY_BAND)
        print("Higher standing share is consistent with sway, not proof; see header note.")
        return

    if not args.subject:
        ap.error("give --subject NAME, or use --list / --all")

    sit_hits = find(sit, args.subject)
    stand_hits = find(stand, args.subject)
    if not stand_hits and not sit_hits:
        print(f"No match for '{args.subject}'. Try --list to see available names.")
        return

    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if sit_hits:
        panels.append(("Seated", sit[sit_hits[0]]))
    if stand_hits:
        panels.append(("Standing", stand[stand_hits[0]]))

    # one column per posture; two rows: raw spectrum (with 50 Hz) and filtered (in-band)
    ncol = len(panels)
    fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 7), squeeze=False)

    for col, (label, path) in enumerate(panels):
        t, v, x, fs, sway, card = analyse(path)

        # row 0: RAW spectrum, full range, 50 Hz marked
        raw = np.asarray(v, dtype=float) - np.mean(v)
        fr, mr = amp_spectrum(raw, fs)
        ax0 = axes[0][col]
        ax0.plot(fr, mr, lw=0.6, color="C0")
        ax0.axvline(50, color="red", ls="--", lw=0.8, label="50 Hz mains")
        ax0.set_xlim(0, 80)
        ax0.set_title(f"{clean_name(path)} — {label}\nRAW spectrum (50 Hz present)", fontsize=10)
        ax0.set_ylabel("Amplitude"); ax0.legend(fontsize=8)

        # row 1: FILTERED spectrum, zoomed to low end, bands shaded
        ff, mf = amp_spectrum(x, fs)
        ax1 = axes[1][col]
        ax1.plot(ff, mf, lw=0.7, color="C3")
        ax1.axvspan(*SWAY_BAND, color="orange", alpha=0.25, label=f"sway band {SWAY_BAND[0]}-{SWAY_BAND[1]} Hz")
        ax1.axvspan(*CARDIAC_BAND, color="green", alpha=0.15, label=f"cardiac band {CARDIAC_BAND[0]}-{CARDIAC_BAND[1]} Hz")
        ax1.set_xlim(0, 5)
        ax1.set_title(f"FILTERED (ACF input)\nsway {sway*100:.0f}%  |  cardiac {card*100:.0f}%", fontsize=10)
        ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Amplitude")
        ax1.legend(fontsize=7)

    fig.suptitle("Raw vs filtered FFT — 50 Hz removed; where the surviving energy sits", fontsize=11)
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=120)
        print("saved", args.out)
    else:
        plt.show()

    print()
    for label, path in panels:
        _, _, _, _, sway, card = analyse(path)
        print(f"{label:<10} sway-band {sway*100:5.1f}%   cardiac-band {card*100:5.1f}%")


if __name__ == "__main__":
    main()