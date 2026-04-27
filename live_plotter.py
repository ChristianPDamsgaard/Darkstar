#!/usr/bin/env python3
"""
Live 0–7 V plot from Arduino on A0 (direct serial read; CSV: voltage_a0).

arduino-cli monitor buffers heavily when stdout is not a TTY, so this uses
PySerial for the live stream. Optional: arduino-cli board list for --port.

Usage:
  python3 live_plot.py
  python3 live_plot.py --port /dev/cu.usbmodem101
  python3 live_plot.py --record
  python3 live_plot.py --csv ./run.csv --record-sec 60
  python3 live_plot.py --record --record-sec 300 --no-plot   # 5 min to default CSV, no GUI
  python3 live_plot.py --record --record-sec 300 --utc-iso  # + datetime_utc_iso8601 column (UTC)
  python3 live_plot.py --record --record-sec 300 --utc-iso --start-at-local 19:25:00  # Copenhagen default

Requires: matplotlib, pyserial. Optional: arduino-cli (auto port).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from datetime import time as dtime_cls
from typing import Deque, List, Optional, Tuple

SamplePt = Tuple[float, float]  # time_s, v_a0
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import serial

# Sketch line: ref0, ref7, v_a0 (3 fields) or legacy ref0, ref7, v_a0, v_a1 (4 fields)
LINE_RE3 = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
LINE_RE4 = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)

WINDOW_SEC = 10.0
Y_MIN = 0.0
Y_MAX = 7.0


def default_csv_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, f"a0_capture_{ts}.csv")


def utc_iso8601_from_epoch(epoch_s: float) -> str:
    """UTC wall time for the same instant as unix_time_s (host clock)."""
    dt = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def parse_local_clock(s: str) -> tuple[int, int, int]:
    """Parse HH:MM or HH:MM:SS (24 h)."""
    parts = s.strip().split(":")
    if len(parts) == 2:
        h, m = int(parts[0]), int(parts[1])
        return h, m, 0
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise SystemExit(f"Invalid --start-at-local (use HH:MM or HH:MM:SS): {s!r}")


def next_local_wall_time(tz_name: str, hour: int, minute: int, second: int) -> datetime:
    """Next occurrence of this local wall-clock time in tz (today if still ahead, else tomorrow)."""
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    t = dtime_cls(hour, minute, second)
    target = datetime.combine(now.date(), t, tzinfo=tz)
    if target <= now:
        target = datetime.combine(now.date() + timedelta(days=1), t, tzinfo=tz)
    return target


def sleep_until_utc(ts: float) -> None:
    """Sleep until time.time() >= ts (tight loop near the deadline)."""
    last_print = 0.0
    while True:
        now = time.time()
        left = ts - now
        if left <= 0:
            return
        if left > 2.0 and int(now) - int(last_print) >= 10:
            print(f"… {left:.0f}s until start", flush=True)
            last_print = now
        time.sleep(min(left, 0.02))


def detect_arduino_port() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["arduino-cli", "board", "list", "--format", "json"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    data = json.loads(out.decode())
    for entry in data.get("detected_ports", []):
        if entry.get("matching_boards"):
            return entry["port"]["address"]
    return None


def reader_serial(
    port: str,
    baud: int,
    samples: Deque[SamplePt],
    samples_lock: threading.Lock,
    stop: threading.Event,
    csv_path: Optional[str],
    record_sec: float,
    record_state: dict,
    utc_iso: bool = False,
) -> None:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.2
    ser.open()
    record_end: Optional[float] = None
    csv_f = None
    if csv_path:
        csv_f = open(csv_path, "w", newline="")
        if utc_iso:
            csv_f.write("unix_time_s,datetime_utc_iso8601,voltage_a0\n")
        else:
            csv_f.write("unix_time_s,voltage_a0\n")
        csv_f.flush()
        record_end = time.time() + record_sec
        record_state["path"] = csv_path
        record_state["end"] = record_end
        record_state["rows"] = 0
        print(f"Recording to {csv_path} for {record_sec:.0f} s (wall clock)", flush=True)
    try:
        ser.reset_input_buffer()
        while not stop.is_set():
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            v: float
            m4 = LINE_RE4.match(line)
            m3 = LINE_RE3.match(line)
            if m4:
                try:
                    v = float(m4.group(3))
                except ValueError:
                    continue
            elif m3:
                try:
                    v = float(m3.group(3))
                except ValueError:
                    continue
            else:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        v = float(parts[2].strip())
                    except ValueError:
                        continue
                else:
                    continue
            now = time.time()
            if csv_f is not None and record_end is not None:
                if now < record_end:
                    if utc_iso:
                        iso = utc_iso8601_from_epoch(now)
                        csv_f.write(f"{now:.6f},{iso},{v:.6f}\n")
                    else:
                        csv_f.write(f"{now:.6f},{v:.6f}\n")
                    record_state["rows"] = record_state.get("rows", 0) + 1
                    if record_state["rows"] % 500 == 0:
                        csv_f.flush()
                else:
                    csv_f.flush()
                    csv_f.close()
                    csv_f = None
                    record_state["end"] = None
                    record_state["finished"] = True
                    print(
                        f"Recording finished: {record_state.get('rows', 0)} rows → {record_state.get('path')}",
                        flush=True,
                    )
                    record_end = None
            with samples_lock:
                samples.append((now, v))
                cutoff = now - WINDOW_SEC
                while samples and samples[0][0] < cutoff:
                    samples.popleft()
    finally:
        if csv_f is not None:
            csv_f.flush()
            csv_f.close()
            record_state["end"] = None
            record_state["finished"] = True
            print(
                f"Recording stopped (window closed): {record_state.get('rows', 0)} rows → {record_state.get('path')}",
                flush=True,
            )
        ser.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live A0 voltage — last 10 s (0–7 V)")
    ap.add_argument("--port", help="Serial port (default: arduino-cli board list)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--record",
        action="store_true",
        help="Write first --record-sec seconds to CSV (default filename next to script)",
    )
    ap.add_argument(
        "--csv",
        metavar="FILE",
        help="CSV output path (implies recording; overrides --record default name)",
    )
    ap.add_argument(
        "--record-sec",
        type=float,
        default=60.0,
        help="Seconds of wall-clock time to record to CSV (default: 60)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Record only: no live window; exit when recording finishes (use with --record or --csv)",
    )
    ap.add_argument(
        "--utc-iso",
        action="store_true",
        help="Add datetime_utc_iso8601 column (UTC Z) next to unix_time_s when recording",
    )
    ap.add_argument(
        "--start-at-local",
        metavar="HH:MM[:SS]",
        default=None,
        help="Wait until this local wall-clock time (see --tz), then open serial and record",
    )
    ap.add_argument(
        "--tz",
        default="Europe/Copenhagen",
        help="IANA timezone for --start-at-local (default: Europe/Copenhagen)",
    )
    args = ap.parse_args()

    csv_path: Optional[str] = args.csv
    defer_default_csv = False
    if csv_path is None and args.record:
        if args.start_at_local:
            defer_default_csv = True
        else:
            csv_path = default_csv_path()
    if args.no_plot and csv_path is None:
        print("--no-plot requires --record or --csv", file=sys.stderr)
        sys.exit(1)

    port = args.port or detect_arduino_port()
    if not port:
        print(
            "No serial port. Connect the board or pass --port /dev/cu.usbmodemXXX",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.start_at_local:
        h, m, s = parse_local_clock(args.start_at_local)
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
            print("Invalid hour/minute/second for --start-at-local (use 24 h clock).", file=sys.stderr)
            sys.exit(1)
        target_local = next_local_wall_time(args.tz, h, m, s)
        target_ts = target_local.timestamp()
        rec_note = ""
        if args.record or args.csv:
            rec_note = f" — then record {args.record_sec:.0f} s to CSV"
        print(
            f"Waiting until {target_local.isoformat()} ({args.tz}){rec_note}, then live plot…",
            flush=True,
        )
        sleep_until_utc(target_ts)
        print(f"Start time reached ({args.tz}). Opening serial…", flush=True)
        if defer_default_csv:
            csv_path = default_csv_path()

    samples_lock = threading.Lock()
    samples: Deque[SamplePt] = deque()
    stop = threading.Event()
    record_state: dict = {}

    t = threading.Thread(
        target=reader_serial,
        args=(
            port,
            args.baud,
            samples,
            samples_lock,
            stop,
            csv_path,
            float(args.record_sec),
            record_state,
            bool(args.utc_iso),
        ),
        daemon=True,
    )
    t.start()

    if args.no_plot:
        deadline = time.time() + float(args.record_sec) + 120.0
        try:
            while not record_state.get("finished"):
                if time.time() > deadline:
                    print("Recording timeout (no data or serial issue).", file=sys.stderr)
                    stop.set()
                    t.join(timeout=3.0)
                    sys.exit(1)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Stopped.", flush=True)
        finally:
            stop.set()
            t.join(timeout=5.0)
        sys.exit(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        mgr.set_window_title("A0 live — last 10 s @ 0–7 V")

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-WINDOW_SEC, 0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{port} — rolling {WINDOW_SEC:.0f} s window")
    ax.grid(True, alpha=0.3)
    (line0,) = ax.plot([], [], "b-", lw=1.2, label="A0")
    ax.legend(loc="upper right")

    hud = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    def animate(_frame: int) -> None:
        now = time.time()
        with samples_lock:
            pts: List[SamplePt] = list(samples)
        if not pts:
            line0.set_data([], [])
            hud.set_text("Waiting for serial…\n(open plot after upload; baud 115200)")
            return
        xs = [p[0] - now for p in pts]
        ys0 = [p[1] for p in pts]
        line0.set_data(xs, ys0)
        ax.set_xlim(-WINDOW_SEC, 0)
        last0 = pts[-1][1]
        t0 = pts[0][0]
        span = now - t0
        finite = [y for y in ys0 if y == y]
        vmin = min(finite) if finite else 0.0
        vmax = max(finite) if finite else 1.0
        rec_note = ""
        if csv_path and record_state.get("end") is not None:
            left = max(0.0, record_state["end"] - now)
            rec_note = f"\nCSV: {left:4.0f}s left → {os.path.basename(csv_path)}"
        elif csv_path and record_state.get("finished"):
            rec_note = f"\nCSV: saved ({record_state.get('rows', 0)} rows) {os.path.basename(csv_path)}"
        hud.set_text(
            f"A0: {last0:6.3f} V\n"
            f"Last {span:4.1f} s in window  |  min {vmin:.3f}  max {vmax:.3f} V"
            f"{rec_note}"
        )

    _ = animation.FuncAnimation(
        fig,
        animate,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    def on_close(_event) -> None:
        stop.set()

    fig.canvas.mpl_connect("close_event", on_close)

    try:
        plt.tight_layout()
        plt.show()
    finally:
        stop.set()
        t.join(timeout=2.0)


if __name__ == "__main__":
    main()
