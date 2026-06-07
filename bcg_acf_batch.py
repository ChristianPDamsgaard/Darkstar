"""
bcg_acf_batch.py
----------------
Runs the thesis autocorrelation HR estimator over a folder of BCG CSV files.

The four core functions (preprocess_bcg, autocorrelation, acf_bpm, estimate_hr)
are reproduced UNCHANGED from Appendix A.5 of the "Heart and Sole" thesis.
The CSV loading, the 20-second windowing, and the table printing are new
scaffolding written to apply those functions to real recordings.

Each CSV is expected to have two columns with a header: time,voltage
(time in seconds, voltage in volts), as produced by the Arduino logger.

Usage:
    python bcg_acf_batch.py /path/to/folder
    python bcg_acf_batch.py /path/to/folder --windows      # add 20 s window analysis
    python bcg_acf_batch.py /path/to/folder --plot-dir figs # save an ACF plot per file
    python bcg_acf_batch.py /path/to/folder --fft fftfigs   # save a raw-vs-filtered FFT per file
"""

import argparse
import glob
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfiltfilt


# ============================================================================
# Reproduced verbatim from thesis Appendix A.5 (Listing A.1).
# ============================================================================

LAG_LO, LAG_HI = 0.45, 1.15


def preprocess_bcg(voltage: np.ndarray, fs: float) -> np.ndarray:
    v = signal.detrend(np.asarray(voltage, dtype=float), type="linear")
    nyq = fs / 2.0
    hi = min(12.0, 0.45 * nyq)
    sos = butter(4, [0.7, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, v)


def autocorrelation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    nfft = 1 << (2 * n - 2).bit_length()
    spectrum = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:n].real
    if ac[0] != 0:
        ac /= ac[0]
    return ac


def acf_bpm(x: np.ndarray, fs: float) -> float:
    ac = autocorrelation(x)
    lags = np.arange(len(ac)) / fs
    mask = (lags >= LAG_LO) & (lags <= LAG_HI)
    if not np.any(mask):
        return float("nan")
    k = np.argmax(ac[mask])
    tau = lags[mask][k]
    return 60.0 / tau


def estimate_hr(time_s: np.ndarray, voltage: np.ndarray) -> float:
    dt = np.diff(time_s)
    fs = 1.0 / np.median(dt[dt > 0])
    x = preprocess_bcg(voltage, fs)
    return acf_bpm(x, fs)


# ============================================================================
# New scaffolding (NOT from the thesis): load CSVs, window, report.
# ============================================================================

def load_csv(path: str):
    """Load a two-column time,voltage CSV. Tolerant of a header line."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    # genfromtxt with names=True reads the header; columns are 'time','voltage'
    cols = data.dtype.names
    if cols is None or "time" not in cols or "voltage" not in cols:
        # fall back: assume first two numeric columns, skip one header row
        raw = np.genfromtxt(path, delimiter=",", skip_header=1)
        return raw[:, 0], raw[:, 1]
    return data["time"], data["voltage"]


def subject_name(path: str) -> str:
    base = os.path.basename(path)
    # strip the "_standing_BCG_..." tail to get a readable name
    cut = base.find("_standing_BCG")
    if cut == -1:
        cut = base.find("_BCG")
    name = base[:cut] if cut != -1 else base
    # some zip tools mangle non-ASCII as #UXXXX; decode those back
    import re
    name = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), name)
    return name.replace("_", " ")


def window_analysis(t, v, fs, win_s=20.0):
    """Split into consecutive win_s windows, estimate HR per window
    (same preprocessing applied per window, as in thesis section 6.3.1)."""
    n_per = int(round(win_s * fs))
    bpms = []
    for start in range(0, len(v) - n_per + 1, n_per):
        seg = v[start:start + n_per]
        tseg = t[start:start + n_per]
        bpms.append(estimate_hr(tseg, seg))
    return bpms


def save_fft_plot(t, v, fs, name, out_path):
    """Save a raw-vs-filtered amplitude spectrum. Shows the 50 Hz mains spike
    in the raw signal and its removal by the 0.7-12 Hz bandpass that the ACF
    actually receives. NOT from the thesis -- argument/illustration helper."""
    import matplotlib.pyplot as plt

    raw = np.asarray(v, dtype=float) - np.mean(v)
    filt = preprocess_bcg(v, fs)

    def amp_spectrum(sig):
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(np.fft.rfft(sig)) / n
        return freqs, mag

    fr, mr = amp_spectrum(raw)
    ff, mf = amp_spectrum(filt)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(fr, mr, lw=0.6, color="C0")
    a1.axvline(50, color="red", ls="--", lw=0.8, label="50 Hz mains")
    a1.set_title(f"{name} - RAW spectrum (50 Hz spike present)")
    a1.set_ylabel("Amplitude"); a1.legend(fontsize=8); a1.set_xlim(0, 80)

    a2.plot(ff, mf, lw=0.6, color="C3")
    a2.axvline(50, color="red", ls="--", lw=0.8, label="50 Hz mains")
    a2.set_title("AFTER 0.7-12 Hz bandpass (what the ACF receives) - 50 Hz removed")
    a2.set_ylabel("Amplitude"); a2.set_xlabel("Frequency (Hz)")
    a2.legend(fontsize=8); a2.set_xlim(0, 80)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Batch autocorrelation HR over a folder of BCG CSVs.")
    ap.add_argument("folder", help="folder containing *.csv files")
    ap.add_argument("--windows", action="store_true", help="also run 20 s window analysis")
    ap.add_argument("--win", type=float, default=20.0, help="window length in seconds")
    ap.add_argument("--plot-dir", type=str, default=None, help="save one ACF plot per file here")
    ap.add_argument("--fft", type=str, default=None, help="save one raw-vs-filtered FFT plot per file here")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "**", "*.csv"), recursive=True))
    if not files:
        print(f"No CSV files found under {args.folder}")
        return

    if args.plot_dir or args.fft:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
    if args.fft:
        os.makedirs(args.fft, exist_ok=True)

    print(f"{'Subject':<34}{'fs':>6}{'dur(s)':>8}{'HR(bpm)':>9}")
    print("-" * 57)

    results = []
    for f in files:
        t, v = load_csv(f)
        dt = np.diff(t); dt = dt[dt > 0]
        fs = 1.0 / np.median(dt)
        hr = estimate_hr(t, v)
        results.append((subject_name(f), hr))
        print(f"{subject_name(f):<34}{fs:>6.0f}{t[-1]-t[0]:>8.1f}{hr:>9.2f}")

        if args.windows:
            wb = window_analysis(t, v, fs, args.win)
            wb_str = "  ".join(f"{b:5.1f}" for b in wb)
            print(f"    {int(args.win)}s windows: {wb_str}")

        if args.plot_dir:
            x = preprocess_bcg(v, fs)
            ac = autocorrelation(x)
            lags = np.arange(len(ac)) / fs
            peak_lag = 60.0 / hr if hr == hr else 0
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
            sl = (t >= 10) & (t <= 20)
            a1.plot(t[sl], x[sl], lw=0.8)
            a1.set_title(f"{subject_name(f)} - preprocessed BCG (10-20 s)")
            a1.set_xlabel("Time (s)"); a1.set_ylabel("Amplitude")
            m = lags <= 2.0
            a2.plot(lags[m], ac[m], color="0.3")
            a2.axvspan(LAG_LO, LAG_HI, alpha=0.2, label="Lag window 0.45-1.15 s")
            if hr == hr:
                a2.axvline(peak_lag, color="green",
                           label=f"Peak {peak_lag:.2f}s -> {hr:.1f} bpm")
            a2.set_title("Autocorrelation"); a2.set_xlabel("Lag (s)")
            a2.set_ylabel("Normalized ACF"); a2.legend(fontsize=8)
            fig.tight_layout()
            out = os.path.join(args.plot_dir, subject_name(f).replace(" ", "_") + ".png")
            fig.savefig(out, dpi=110); plt.close(fig)

        if args.fft:
            out = os.path.join(args.fft, subject_name(f).replace(" ", "_") + "_fft.png")
            save_fft_plot(t, v, fs, subject_name(f), out)

    print("-" * 57)
    valid = [hr for _, hr in results if hr == hr]
    if valid:
        print(f"{'Mean HR across files':<34}{'':>6}{'':>8}{np.mean(valid):>9.2f}")


if __name__ == "__main__":
    main()
