#!/usr/bin/env python
"""
main2 — Watch PPG PREPROCESSING inspector.

Shows, per channel, a figure with every preprocessing stage stacked so you can
see exactly what each step does to the signal and tune it:

  0 Raw (file → 256 Hz)   1 Inverted   2 FFT upsample ×4 → 1024 Hz + trim
  3 HP baseline (drift removed)   4 High-passed   5 Bandpass 0.5–4 Hz (final)

Plus one accelerometer / movement figure (raw axes + jerk energy + threshold).

The stage math lives in respiration_rr/ppg/preprocess_stages.py and is a thin
wrapper over the real pipeline blocks — what you see is what analyze_ppg uses.
Tune parameters in respiration_rr/settings.py (PPG.*) and re-run.

Usage
-----
    py Code/main2.py                                   # recording 001, all channels, full length
    py Code/main2.py --recording "Data/Exp1/recordings data/003"
    py Code/main2.py --channels IR Artifact            # only these channels
    py Code/main2.py --window 30 50                    # zoom all panels to 30–50 s (see beats)
    py Code/main2.py --watch path/to/rt_flow.csv
    py Code/main2.py --no-show                         # build only, don't open windows
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:                       # make Unicode prints safe on cp1252 Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from respiration_rr.settings import PPG
from respiration_rr.io.csv_reader import read_watch_auto
from respiration_rr.ppg.preprocess_stages import (
    compute_channel_stages, compute_beat_stages, compute_movement_stage,
)
from respiration_rr import viz

DEFAULT_RECORDING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data", "Exp1", "recordings data", "001")

# rt_flow column -> (display name, colour)
CHANNELS = {
    "Green":    ("ppg",       "#22c55e"),
    "Red":      ("red",       "#ef4444"),
    "IR":       ("infra_red", "#a855f7"),
    "Artifact": ("artifact",  "#38bdf8"),
}


def _find_watch(recording_dir):
    c = glob.glob(os.path.join(recording_dir, "rt_flow*.csv")) or \
        glob.glob(os.path.join(recording_dir, "*.csv"))
    return c[0] if c else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Watch PPG preprocessing inspector")
    ap.add_argument("--recording", default=None, help="folder with an rt_flow*.csv")
    ap.add_argument("--watch", default=None, help="explicit watch CSV path")
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS),
                    help="channels to show (Green Red IR Artifact)")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    default=None, help="zoom all panels to this time span (s)")
    ap.add_argument("--no-beats", action="store_true", help="skip the beat/peak-finding figure")
    ap.add_argument("--no-movement", action="store_true", help="skip the accelerometer figure")
    ap.add_argument("--no-show", action="store_true", help="build figures but don't open windows")
    args = ap.parse_args(argv)

    csv_path = args.watch
    if csv_path is None:
        rec = args.recording or DEFAULT_RECORDING
        csv_path = _find_watch(rec)
        print(f"Recording: {rec}")
    if not csv_path:
        ap.error("No watch CSV found. Pass --recording DIR or --watch FILE.")

    print(f"[Watch preprocessing]  {os.path.basename(csv_path)}")
    watch = read_watch_auto(csv_path)          # real-time (rt_flow) or monitor CSV
    print(f"  file rate {watch['fs']:.1f} Hz, {watch['time'][-1]:.0f}s | "
          f"invert={PPG.invert_ppg}  band {PPG.ppg_hp_hz}-{PPG.ppg_lp_hz} Hz order {PPG.ppg_filter_order} | "
          f"upsample x{PPG.upsample_factor} -> {PPG.target_fs:.0f} Hz")
    window = tuple(args.window) if args.window else None

    # Movement first — its regions gate the LPF-derivative beat detector (as in the HTML).
    mv = compute_movement_stage(watch, PPG)
    move_regions = mv["regions"] if mv else None
    if mv is not None:
        cov = sum(e - s for s, e in move_regions)
        print(f"  movement   thr {mv['threshold']:.0f} g/s | {len(move_regions)} region(s) "
              f"covering {cov:.0f}s of {mv['t'][-1] - mv['t'][0]:.0f}s")

    n_figs = 0
    for name in args.channels:
        if name not in CHANNELS:
            print(f"  ! unknown channel '{name}' (choose from {list(CHANNELS)})")
            continue
        col, color = CHANNELS[name]
        if col not in watch:
            print(f"  ! channel '{name}' (column '{col}') not in file — skipped")
            continue
        stages = compute_channel_stages(watch[col], watch["fs"], PPG,
                                        invert=PPG.invert_ppg, base_color=color)
        viz.plot_preprocess_stages(name, stages, window=window, base_color=color)
        n_figs += 1
        msg = f"  {name:<9} {len(stages)} preprocessing stages"

        if not args.no_beats:
            filtered = stages[-1]["traces"][0]["y"]
            t = stages[-1]["traces"][0]["t"]
            bstages = compute_beat_stages(filtered, t, PPG, base_color=color,
                                          move_regions=move_regions)
            viz.plot_preprocess_stages(f"{name} — beat finding", bstages,
                                       window=window, base_color=color)
            n_figs += 1
            beats_stage = next((s for s in bstages if s["key"] == "beats"), None)
            n_beats = beats_stage["markers"][1]["t"].size if beats_stage else 0
            method = "LPF-deriv" if PPG.use_lpf_derivative_beats else "troughs"
            msg += f" + {len(bstages)} beat-finding stages ({n_beats} beats, {method})"
        print(msg)

    if not args.no_movement and mv is not None:
        viz.plot_movement_preprocess(mv, window=window)
        n_figs += 1

    print(f"\nBuilt {n_figs} figures.")
    if not args.no_show:
        print("Opening figures — close the windows to exit. "
              "Zoom/pan with the toolbar; panels share the x-axis.")
        viz.show_all()


if __name__ == "__main__":
    main()
