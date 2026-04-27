"""
BCG live recorder (Windows + Arduino Uno R3/R4)
-----------------------------------------------
Reads CSV lines from an Arduino, shows a scrolling live plot of the BCG
channel (A0 only), and saves all samples to a timestamped CSV file.
Stop recording by closing the plot window or pressing Ctrl-C.

First-time setup (Windows, in PowerShell or CMD):
    py -m pip install pyserial matplotlib numpy

Find your Arduino's COM port:
    py record_bcg.py --list

Record a session:
    py record_bcg.py --port COM4 --label standing
    py record_bcg.py --port COM4 --label sitting

Flags:
    --port    COM port of the Arduino (required unless --list)
    --label   optional string appended to the saved filename
    --outdir  directory for CSV files (default: current dir)
    --r4      use this flag if your Arduino R4 is running 14-bit ADC
              (matches USE_R4_HIGH_RES = 1 in the sketch)
"""

import argparse
import csv
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports

# ------------ Config (match the Arduino sketch) ------------
SAMPLE_RATE_HZ  = 200
PLOT_WINDOW_S   = 10
PLOT_BUFFER_N   = SAMPLE_RATE_HZ * PLOT_WINDOW_S
VREF            = 5.0
PLOT_REFRESH_HZ = 30
# -----------------------------------------------------------


def list_ports_and_exit():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Is the Arduino plugged in?")
    else:
        print("Available serial ports:")
        for p in ports:
            print(f"  {p.device}   {p.description}")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Record BCG from Arduino A0")
    parser.add_argument("--port",   help="Serial port, e.g. COM4")
    parser.add_argument("--baud",   type=int, default=115200)
    parser.add_argument("--list",   action="store_true", help="List serial ports and exit")
    parser.add_argument("--outdir", default=".", help="Where to save CSV recordings")
    parser.add_argument("--label",  default="", help="Label added to filename, e.g. 'standing'")
    parser.add_argument("--r4",     action="store_true",
                        help="Arduino Uno R4 with 14-bit ADC (0..16383). Omit for R3 (0..1023).")
    args = parser.parse_args()

    if args.list:
        list_ports_and_exit()
    if not args.port:
        print("Error: --port is required. Run with --list to see available ports.")
        sys.exit(1)

    adc_max = 16383 if args.r4 else 1023

    # --- Open serial port ---
    print(f"Opening {args.port} @ {args.baud} baud ...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open {args.port}: {e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    # --- Prepare output CSV ---
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.label}" if args.label else ""
    csv_path = Path(args.outdir) / f"recording_{stamp}{suffix}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["t_seconds", "bcg_volts"])
    print(f"Recording to {csv_path}")
    print("Close the plot window (or press Ctrl-C) to stop.\n")

    # --- Ring buffers ---
    t_buf   = deque(maxlen=PLOT_BUFFER_N)
    bcg_buf = deque(maxlen=PLOT_BUFFER_N)

    # --- Live plot (single panel) ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.suptitle("Live BCG   (close window to stop)")
    (line_bcg,) = ax.plot([], [], lw=0.9)
    ax.set_ylabel("BCG (V)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    running = {"go": True}
    fig.canvas.mpl_connect("close_event", lambda evt: running.update(go=False))

    t0_us           = None
    sample_count    = 0
    last_plot_time  = time.monotonic()
    last_status_time = last_plot_time
    plot_period     = 1.0 / PLOT_REFRESH_HZ

    try:
        while running["go"]:
            # Drain all waiting serial lines
            while ser.in_waiting and running["go"]:
                raw = ser.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                # Accept lines with 2 values (t, bcg) or 3 values (t, bcg, ecg — ignore ecg)
                if len(parts) < 2:
                    continue
                try:
                    t_us    = int(parts[0])
                    bcg_raw = int(parts[1])
                except ValueError:
                    continue

                if t0_us is None:
                    t0_us = t_us
                t_s   = (t_us - t0_us) / 1_000_000.0
                bcg_v = (bcg_raw / adc_max) * VREF

                writer.writerow([f"{t_s:.6f}", f"{bcg_v:.5f}"])

                t_buf.append(t_s)
                bcg_buf.append(bcg_v)
                sample_count += 1

            # Redraw at capped rate
            now = time.monotonic()
            if now - last_plot_time >= plot_period and len(t_buf) > 2:
                line_bcg.set_data(t_buf, bcg_buf)

                t_right = t_buf[-1]
                t_left  = max(0.0, t_right - PLOT_WINDOW_S)
                ax.set_xlim(t_left, t_right)

                lo, hi = min(bcg_buf), max(bcg_buf)
                if hi == lo:
                    hi = lo + 1e-3
                pad = 0.1 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)

                fig.canvas.draw_idle()
                last_plot_time = now

            # Terminal status line
            if now - last_status_time >= 2.0:
                elapsed = t_buf[-1] if t_buf else 0.0
                print(f"  t = {elapsed:6.1f} s   samples = {sample_count}", end="\r")
                last_status_time = now

            plt.pause(0.005)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        csv_file.close()
        plt.ioff()
        plt.close("all")
        print(f"\nSaved {sample_count} samples to {csv_path}")


if __name__ == "__main__":
    main()