#!/usr/bin/env python
"""
avg_rr_compare.py — averaged RR (watch) vs averaged RR (reference).

"Average RR" = a sliding-window median of a per-breath RR series over time. Per
channel (wavelength) it is applied to every per-beat series individually, so you
get all THREE parameters in all THREE methods = 9 averaged curves per channel:

    parameter : RSA (pulse) · RIIV (amplitude) · AUC (area)
    method    : source (linear+BP) · spline (cubic) · ssp (smoothing spline)

One figure per channel, 3 panels (one per parameter); each panel overlays the 3
method-averaged curves and the reference average (averaged the SAME way). The
watch is shifted onto the REMbo clock by the IR-PPG sync offset. Each curve is
scored by MAE (bpm) vs the reference over the overlap.

LP (BW), BWlegacy and BWbank are per-signal traces, not per-beat, and are excluded.

Usage
-----
    py avg_rr_compare.py                                  # first recording, opens windows
    py avg_rr_compare.py --rec-id Exp2/002                # by id under --data-root
    py avg_rr_compare.py --recording "<dir with edf+csv>" # a specific folder
    py avg_rr_compare.py --edf a.edf --watch b.csv        # explicit files
    py avg_rr_compare.py --channels IR Green              # only these channels
    py avg_rr_compare.py --window 20 --stride 1 --min-pts 2 --method median
    py avg_rr_compare.py --save out_dir --no-show         # save PNGs instead of opening
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


DEFAULT_DATA_ROOT = (r"C:\Users\RachelMizrahi\CardiacSense"
                     r"\shares - BBB Respiration Rate\Data")
DEFAULT_OUT = r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\avg_rr"

# exclude Artifact for now
DEFAULT_CHANNELS = ["Green", "Red", "IR"]
CHANNEL_TITLE_COLOR = {"Green": "#22c55e", "Red": "#ef4444", "IR": "#a855f7", "Artifact": "#38bdf8"}

# per-method line colour (same across every panel/channel)
METHOD_COLOR = {"": "#2563eb", "_spline": "#f59e0b", "_ssp": "#10b981"}


def _find_inputs(folder):
    edfs = glob.glob(os.path.join(folder, "*.edf"))
    csvs = (glob.glob(os.path.join(folder, "rt_flow*.csv"))
            or glob.glob(os.path.join(folder, "*.csv")))
    return (edfs[0] if edfs else None), (csvs[0] if csvs else None)


def discover(data_root):
    out = []
    for exp in sorted(glob.glob(os.path.join(data_root, "Exp*"))):
        rec_root = os.path.join(exp, "recordings data")
        if not os.path.isdir(rec_root):
            continue
        for folder in sorted(glob.glob(os.path.join(rec_root, "*"))):
            edf, csv = _find_inputs(folder)
            if edf and csv:
                out.append((f"{os.path.basename(exp)}/{os.path.basename(folder)}", edf, csv))
    return out


def _resolve_inputs(args):
    if args.edf and args.watch:
        return "custom", args.edf, args.watch
    if args.recording:
        edf, csv = _find_inputs(args.recording)
        if not (edf and csv):
            sys.exit(f"No .edf + .csv found in {args.recording}")
        return os.path.basename(args.recording.rstrip("/\\")), edf, csv
    recs = discover(args.data_root)
    if not recs:
        sys.exit(f"No recordings found under {args.data_root}")
    if args.rec_id:
        recs = [r for r in recs if r[0] == args.rec_id]
        if not recs:
            sys.exit(f"Recording id '{args.rec_id}' not found under {args.data_root}")
    return recs[0]


def _plot_channel(rid, ch, ref_pts, ref_avg, ch_avgs, offset, cfg, plt,
                  PER_BEAT_BASE, METHOD_SUFFIXES, PARAM_LABEL, METHOD_LABEL):
    """One figure for a channel: a panel per base param, 3 method curves + reference."""
    bases = [b for b in PER_BEAT_BASE if b in ch_avgs]
    if not bases:
        return None
    fig, axes = plt.subplots(len(bases), 1, figsize=(14, 3.0 * len(bases)), sharex=True)
    if len(bases) == 1:
        axes = [axes]
    fig.suptitle(f"{ch} — averaged RR by parameter × method vs reference   |   {rid}   |   "
                 f"window {cfg.rr_avg_window_sec:.0f}s · stride {cfg.rr_avg_step_sec:.0f}s · "
                 f"min_pts {cfg.rr_avg_min_pts} · {cfg.rr_avg_method}",
                 fontweight="bold", color=CHANNEL_TITLE_COLOR.get(ch, "#111827"))

    rt, rr = ref_pts
    at, ar = ref_avg
    for ax, base in zip(axes, bases):
        if rt.size:
            ax.plot(rt, rr, ".", color="#cbd5e1", ms=3, alpha=0.55,
                    label="reference per-breath", zorder=1)
        if at.size:
            ax.plot(at, ar, "-", color="#111827", lw=2.4, label="reference avg", zorder=4)
        for suf in METHOD_SUFFIXES:
            if suf not in ch_avgs[base]:
                continue
            t, r, mae, n = ch_avgs[base][suf]
            col = METHOD_COLOR.get(suf, "#333")
            lbl = METHOD_LABEL[suf] + (f"  (MAE {mae:.2f}, n={n})" if np.isfinite(mae) else "")
            ax.plot(t, r, "-", color=col, lw=1.5, alpha=0.9, label=lbl, zorder=3)
        ax.set_ylabel("RR (bpm)", fontsize=9)
        ax.set_title(PARAM_LABEL[base], fontsize=10, loc="left")
        ax.grid(True, alpha=0.15)
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    axes[-1].set_xlabel("REMbo-clock time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Averaged watch RR (param × method) vs averaged reference RR")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None, help="recording id, e.g. Exp2/002")
    ap.add_argument("--recording", default=None, help="folder with one .edf + one .csv")
    ap.add_argument("--edf", default=None)
    ap.add_argument("--watch", default=None)
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--window", type=float, default=None, help="override window (s)")
    ap.add_argument("--stride", type=float, default=None, help="override stride (s)")
    ap.add_argument("--min-pts", type=int, default=None, help="override min points/window")
    ap.add_argument("--method", choices=("median", "mean"), default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--save", default=None, metavar="DIR", help="also save PNGs into DIR")
    ap.add_argument("--no-show", action="store_true", help="do not open windows (save only)")
    args = ap.parse_args(argv)

    rid, edf_path, csv_path = _resolve_inputs(args)
    print(f"Recording: {rid}\n  EDF   : {edf_path}\n  watch : {csv_path}")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    from respiration_rr.settings import PPG
    from main import run_reference, run_ppg, run_sync       # reuse the exact pipeline wiring
    from respiration_rr.rr_average import (
        PER_BEAT_BASE, METHOD_SUFFIXES, PARAM_LABEL, METHOD_LABEL,
        average_watch_channel, reference_average_rr, reference_rr_points, mae_over_overlap)

    if args.window is not None:
        PPG.rr_avg_window_sec = args.window
    if args.stride is not None:
        PPG.rr_avg_step_sec = args.stride
    if args.min_pts is not None:
        PPG.rr_avg_min_pts = args.min_pts
    if args.method is not None:
        PPG.rr_avg_method = args.method

    ref = run_reference(edf_path)
    ppg_results, sig = run_ppg(csv_path)
    offset = run_sync(edf_path, ppg_results, sig)

    ref_pts = reference_rr_points(ref)                      # REMbo clock
    ref_avg_t, ref_avg_r = reference_average_rr(ref)

    print(f"\n[Averaged RR]  window {PPG.rr_avg_window_sec:.0f}s · "
          f"stride {PPG.rr_avg_step_sec:.0f}s · min_pts {PPG.rr_avg_min_pts} · "
          f"{PPG.rr_avg_method}   (sync offset {offset:+.3f}s)")
    print(f"  reference: {ref_pts[0].size} per-breath pts -> "
          f"{np.isfinite(ref_avg_r).sum()} averaged samples")
    print(f"  {'channel':<9} {'param':<6} {'method':<7} {'MAE(bpm)':>9} {'n':>6}")

    figs, out_dir = [], (args.save or (None if not args.no_show else args.out))
    for ch in args.channels:
        res = ppg_results.get(ch)
        if res is None:
            print(f"  {ch:<9} -- not present --")
            continue
        raw = average_watch_channel(res)                   # {base: {suf: (t, r)}}
        ch_avgs = {}
        for base in PER_BEAT_BASE:
            if base not in raw:
                continue
            ch_avgs[base] = {}
            for suf in METHOD_SUFFIXES:
                if suf not in raw[base]:
                    continue
                t, r = raw[base][suf]
                t = t + offset                             # watch clock -> REMbo clock
                mae, n = mae_over_overlap(t, r, ref_avg_t, ref_avg_r)
                ch_avgs[base][suf] = (t, r, mae, n)
                print(f"  {ch:<9} {base:<6} {METHOD_LABEL[suf]:<7} {mae:>9.2f} {n:>6d}")
        fig = _plot_channel(rid, ch, ref_pts, (ref_avg_t, ref_avg_r), ch_avgs, offset,
                            PPG, plt, PER_BEAT_BASE, METHOD_SUFFIXES, PARAM_LABEL, METHOD_LABEL)
        if fig is None:
            continue
        figs.append(fig)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"{rid.replace('/', '_')}_{ch}_avg_rr.png")
            fig.savefig(png, dpi=120, bbox_inches="tight")
            print(f"  saved -> {png}")

    if args.no_show:
        for f in figs:
            plt.close(f)
    else:
        print(f"\nOpening {len(figs)} figure(s) — close the windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
