#!/usr/bin/env python
"""
main3 — per-signal respiration figure suite (consolidates main + main2).

For EACH watch PPG signal (Green / Red / IR / Artifact) it builds the full
picture, from raw trace to final respiration rate, in six figure groups:

  1. Raw signal                          (original file trace + upsampled channel)
  2. Baseline-corrected signal           (with SS / MSD / SE fiducials marked)
  3. Per-parameter feature series        (one panel each: RSA, RIIV/AMP, AUC, LP)
  4. Respiration envelopes + BS peaks    (breath-start detection on each envelope)
  5. Spline vs linear+BP RR              (RSA/RIIV/AUC: RR from each method's peaks)
  6. Per-parameter spectrograms          (STFT + respiration ridge, one each)
  7. Final respiration rate (RR)         (bpm from each parameter, overlaid)

It reuses the exact same pipeline as main.py (analyze_ppg) so what you see is
what the analyzer computes. Algorithm knobs live in respiration_rr/settings.py.

Usage
-----
    py main3.py                                    # recording 001, all channels
    py main3.py --recording "Data/Exp1/recordings data/003"
    py main3.py --channels IR Artifact             # only these channels
    py main3.py --window 30 60                      # zoom every panel to 30-60 s
    py main3.py --watch path/to/rt_flow.csv
    py main3.py --save figs_001                     # also save every figure as PNG
    py main3.py --no-show                           # build (+optionally save) only
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

import numpy as np
import matplotlib.pyplot as plt

from respiration_rr.settings import PPG
from respiration_rr.io.csv_reader import read_watch_auto
from respiration_rr.ppg.preprocess import prepare_watch
from respiration_rr.ppg.respiration import analyze_ppg
from respiration_rr import viz

DEFAULT_RECORDING = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data", "Exp1", "recordings data", "001")

# analyzer channel name -> display colour (matches viz.figures palette)
CHANNEL_COLORS = {"Green": "#22c55e", "Red": "#ef4444",
                  "IR": "#a855f7", "Artifact": "#38bdf8"}

# analyzer channel name -> raw rt_flow CSV column (for the untouched original trace)
CHANNEL_COLUMN = {"Green": "ppg", "Red": "red", "IR": "infra_red", "Artifact": "artifact"}

# the four respiration parameters, with human-facing labels + colours
PARAMS = ("RSA", "RIIV", "AUC", "LP")
PARAM_LABEL = {
    "RSA":  "RSA — RR-interval (ms)",
    "RIIV": "RIIV / AMP — per-beat amplitude",
    "AUC":  "AUC — per-beat area",
    "LP":   "LP — band-passed trace",
}
PARAM_COLORS = {"RSA": "#22c55e", "RIIV": "#38bdf8", "AUC": "#f472b6", "LP": "#f97316"}


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _shade_moves(ax, regions, color="#ef4444", alpha=0.08):
    for (s, e) in regions or []:
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


def _apply_window(axes, window):
    if window:
        axes[-1].set_xlim(*window)


# ----------------------------------------------------------------------
# Figure 1+2 — raw signal + baseline-corrected signal with SS/MSD/SE
# ----------------------------------------------------------------------
def plot_signal(name, raw_t, raw_y, res, orig_t=None, orig_y=None,
                move_regions=None, window=None):
    """Stacked panels: (0) original untouched file trace, (1) upsampled channel
    entering analysis, (2) baseline-corrected + SS/MSD/SE fiducials."""
    color = CHANNEL_COLORS.get(name, "#333")
    has_orig = orig_t is not None and orig_y is not None and np.size(orig_y)
    n = 3 if has_orig else 2
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.0 * n), sharex=True)
    fig.suptitle(f"{name} — original raw → upsampled → baseline-corrected fiducials",
                 fontsize=13, fontweight="bold", color=color)
    axes = list(axes)
    it = iter(axes)

    # (0) ORIGINAL raw signal — straight from the CSV, before we touched it
    if has_orig:
        ax = next(it)
        ax.plot(orig_t, orig_y, color="#64748b", lw=0.5,
                label="original raw (file rate, untouched)")
        ax.set_ylabel("orig (a.u.)", fontsize=8)
        ax.set_title("0 · Original raw signal (before any processing)",
                     fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.15)

    # (1) upsampled channel that actually enters the analysis
    ax = next(it)
    ax.plot(raw_t, raw_y, color=color, lw=0.6, label="upsampled channel (→1024 Hz, trimmed)")
    _shade_moves(ax, move_regions)
    ax.set_ylabel("raw (a.u.)", fontsize=8)
    ax.set_title("1 · Upsampled channel (input to analysis)", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.15)

    # (2) baseline-corrected + SS / MSD / SE
    ax = next(it)
    t = raw_t  # bc lives on the same 1024 Hz timeline as the prepared channel
    bc = res.bc
    ax.plot(t, bc, color="#2563eb", lw=0.6, label="baseline-corrected")
    for idx, mk_color, mk, lbl in (
        (res.ss_idx, "#f59e0b", "o", "SS (systolic start)"),
        (res.msd_idx, "#10b981", "^", "MSD (max sys. deriv.)"),
        (res.se_idx, "#ef4444", "v", "SE (systolic peak)"),
    ):
        idx = np.asarray([i for i in idx if 0 <= i < bc.size], dtype=int)
        if idx.size:
            ax.scatter(t[idx], bc[idx], s=18, c=mk_color, marker=mk,
                       edgecolors="none", zorder=5, label=lbl)
    _shade_moves(ax, move_regions)
    ax.set_ylabel("BC (a.u.)", fontsize=8)
    ax.set_title(f"2 · Baseline-corrected + fiducials "
                 f"({len(res.ss_idx)} beats)", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    _apply_window(axes, window)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ----------------------------------------------------------------------
# Figure 3 — per-parameter feature series
# ----------------------------------------------------------------------
def plot_param_series(name, res, window=None):
    """One panel per parameter: the raw per-beat feature series feeding RR."""
    color = CHANNEL_COLORS.get(name, "#333")
    params = [p for p in PARAMS if p in res.params]
    n = len(params)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{name} — respiration parameters (per-beat feature series)",
                 fontsize=13, fontweight="bold", color=color)

    for ax, pname in zip(axes, params):
        pr = res.params[pname]
        col = PARAM_COLORS.get(pname, "#333")
        if pname == "LP":                     # LP is a full-rate trace, not a scatter
            ax.plot(pr.series_x, pr.series_y, color=col, lw=0.6)
        elif pr.series_x.size:
            ax.plot(pr.series_x, pr.series_y, "o-", color=col, ms=2.5, lw=0.7)
        ax.set_ylabel(pname, fontsize=8)
        ax.set_title(PARAM_LABEL[pname], fontsize=9, loc="left")
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    _apply_window(axes, window)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----------------------------------------------------------------------
# Figure 6 — final respiration rate from each parameter
# ----------------------------------------------------------------------
def plot_final_rr(name, res, window=None):
    """Final RR (bpm) from every parameter, overlaid. For each parameter, two
    lines: solid = RR from breath-start peak spacing, dashed = RR from that
    parameter's spectrogram ridge (dominant respiration frequency over time)."""
    color = CHANNEL_COLORS.get(name, "#333")
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"{name} — final respiration rate (RR) by parameter "
                 f"(solid = peaks, dashed = spectrogram ridge)",
                 fontsize=13, fontweight="bold", color=color)

    for pname in PARAMS:
        pr = res.params.get(pname)
        if pr is None:
            continue
        col = PARAM_COLORS.get(pname, "#333")
        # peak-detection RR
        if pr.rr_bpm.size:
            ax.plot(pr.rr_time, pr.rr_bpm, "o-", ms=3, lw=0.8, color=col,
                    label=f"{pname} peaks  (mean {np.nanmean(pr.rr_bpm):.1f} bpm)")
        # spectrogram-ridge RR (same colour, dashed)
        if pr.ridge_rr is not None and np.isfinite(pr.ridge_rr).any():
            ax.plot(pr.ridge_time, pr.ridge_rr, "--", lw=1.1, color=col, alpha=0.75,
                    label=f"{pname} ridge  (mean {np.nanmean(pr.ridge_rr):.1f} bpm)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RR (bpm)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)
    if window:
        ax.set_xlim(*window)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def _find_watch(recording_dir):
    c = glob.glob(os.path.join(recording_dir, "rt_flow*.csv")) or \
        glob.glob(os.path.join(recording_dir, "*.csv"))
    return c[0] if c else None


def _save(fig, save_dir, stem):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, stem + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Per-signal respiration figure suite (main + main2 consolidated)")
    ap.add_argument("--recording", default=None, help="folder with an rt_flow*.csv")
    ap.add_argument("--watch", default=None, help="explicit watch CSV path")
    ap.add_argument("--channels", nargs="+", default=list(CHANNEL_COLORS),
                    help="channels to show (Green Red IR Artifact)")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    default=None, help="zoom every panel to this time span (s)")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="also save every figure as PNG into DIR")
    ap.add_argument("--no-show", action="store_true",
                    help="build (+optionally save) figures but don't open windows")
    args = ap.parse_args(argv)

    csv_path = args.watch
    if csv_path is None:
        rec = args.recording or DEFAULT_RECORDING
        csv_path = _find_watch(rec)
        print(f"Recording: {rec}")
    if not csv_path:
        ap.error("No watch CSV found. Pass --recording DIR or --watch FILE.")

    print(f"[main3]  {os.path.basename(csv_path)}")
    watch = read_watch_auto(csv_path)          # real-time (rt_flow) or monitor CSV
    sig = prepare_watch(watch)
    results = analyze_ppg(sig.channels, sig.time, sig.fs, move_regions=sig.move_regions)
    print(f"  {watch['fs']:.0f} Hz -> {PPG.target_fs:.0f} Hz, "
          f"{sig.time[-1] - sig.time[0]:.0f}s | channels {list(results)} | "
          f"movement regions {len(sig.move_regions)}")
    window = tuple(args.window) if args.window else None

    n_figs = 0
    for name in args.channels:
        res = results.get(name)
        if res is None:
            print(f"  ! channel '{name}' not present / empty — skipped")
            continue
        raw = sig.channels.get(name)
        col = CHANNEL_COLUMN.get(name)
        orig_t = watch.get("time") if col and col in watch else None
        orig_y = watch.get(col) if col and col in watch else None
        print(f"\n  {name}: {len(res.ss_idx)} beats | "
              + " ".join(f"{p} {res.mean_rr(p):.1f}bpm" for p in PARAMS if p in res.params))

        figs = [
            ("signal",       plot_signal(name, sig.time, raw, res,
                                         orig_t=orig_t, orig_y=orig_y,
                                         move_regions=sig.move_regions, window=window)),
            ("params",       plot_param_series(name, res, window=window)),
            ("envelopes_bs", viz.plot_ppg_channel(res)),           # item 4 (existing)
            ("spline_vs_bp", viz.plot_spline_comparison(res, window=window)),  # item 5 (spline compare)
            ("spectrograms", viz.plot_ppg_spectrograms(res)),      # item 6 (existing)
            ("final_rr",     plot_final_rr(name, res, window=window)),
        ]
        for stem, fig in figs:
            if fig is None:
                continue
            n_figs += 1
            if args.save:
                print(f"    saved {_save(fig, args.save, f'{name}_{stem}')}")

    print(f"\nBuilt {n_figs} figures across {len(results)} signal(s).")
    if not args.no_show:
        print("Opening figures — close the windows to exit. "
              "Panels within a figure share the x-axis; zoom/pan with the toolbar.")
        viz.show_all()


if __name__ == "__main__":
    main()
