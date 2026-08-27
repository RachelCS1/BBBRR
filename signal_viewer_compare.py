#!/usr/bin/env python
"""
signal_viewer_compare.py — per recording, show ALL THREE chosen methods stacked
(Butter / MAPAS / Legacy), each in its best-tuned, honest-footing config, so you
can compare where each works/fails on the same recording at a glance.

Each recording -> one figure with 3 method rows. Every row: reference RR (black)
+ that method's averaged RR (red) over the RR-category bands, plus a thin
category-agreement strip (green=match, red=mismatch, gray=no data) and the
"% correctly categorised" in the row title.

Methods & configs (all ridge ceiling 0.9 Hz, windowed-median averaged RR):
    Butter  — detrend baseline 5,  argmax ridge
    MAPAS   — detrend baseline 15, argmax ridge
    Legacy  — envelope detrend built in, fundamental ridge

Usage
-----
    py signal_viewer_compare.py --all                      # every recording
    py signal_viewer_compare.py --rec-id Exp2/004
    py signal_viewer_compare.py --all --recordings Exp2/004 Exp1/001
    py signal_viewer_compare.py --all --save DIR --no-show
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import bw_bp_sweep as S
import bw_category_map as CM      # RR_EDGES / HR_EDGES / _labels

REF_LEDS = ("Green", "Red", "IR")

# (name, detrend post-step?, detrend baseline, fundamental ridge?)
METHOD_SPECS = [
    ("Butter", True, 5.0, False),
    ("MAPAS", True, 15.0, False),
    ("Legacy", False, 5.0, True),   # Legacy IS the detrend; fundamental ridge
]
RR_BAND_HIGH = 0.9                  # honest footing (cut the 0.9-1.0 Hz junk)


def _method_rr(name, detrend_on, baseline, fundamental, x, refs, fs, low, high,
               specB_window, PPG, offset, deps):
    """Extract BW for one method, return its averaged RR curve (rr_t+offset, rr_v)."""
    (average_series, METHODS, REF_METHODS, legacy_detrend, spectro_average) = deps
    if name in REF_METHODS:
        bw = REF_METHODS[name](x, refs, fs, low, high)
    elif name == "MA":
        bw = METHODS["MA"](x, fs, low, high, PPG.bw_filter_order)
    else:
        bw = METHODS[name](x, fs, low, high)
    if detrend_on and name != "Legacy":
        bw = legacy_detrend(bw, fs, low, high, baseline_sec=baseline)
    avg_fn = lambda tt, rr: average_series(tt, rr, PPG)
    out = spectro_average(bw, fs, specB_window, PPG, avg_fn, fundamental=fundamental)
    if out is None:
        return np.zeros(0), np.zeros(0)
    st, sr, at, ar, _prom = out
    return np.asarray(at) + offset, np.asarray(ar)     # averaged RR curve


def _build_figure(rid, edf_path, csv_path, args, PPG, deps, plt):
    """One figure with the three methods stacked for a single recording."""
    (run_reference, run_ppg, run_sync, reference_average_rr, average_series,
     METHODS, REF_METHODS, legacy_detrend, spectro_average) = deps
    try:
        ref = run_reference(edf_path)
        ppg_results, sig = run_ppg(csv_path)
        offset = run_sync(edf_path, ppg_results, sig)
    except Exception as e:
        print(f"  [skip] {rid}: {e}")
        return None
    truth_at, truth_ar = reference_average_rr(ref)
    if args.channel not in sig.channels:
        print(f"  [skip] {rid}: channel '{args.channel}' not in {list(sig.channels)}")
        return None
    x = np.asarray(sig.channels[args.channel], np.float64)
    fs = float(sig.fs)
    low, high = PPG.bw_band_low_hz, PPG.bw_band_high_hz
    refs = [np.asarray(sig.channels[c], np.float64)
            for c in REF_LEDS if c in sig.channels and c != args.channel]
    m_deps = (average_series, METHODS, REF_METHODS, legacy_detrend, spectro_average)

    rr_labels = CM._labels(CM.RR_EDGES, "RR")
    edges = [0.0] + list(CM.RR_EDGES) + [55.0]
    band_cols = ["#93c5fd", "#86efac", "#fca5a5", "#fcd34d"]
    step = getattr(PPG, "rr_avg_step_sec", 4.0)

    n = len(METHOD_SPECS)
    fig, axes = plt.subplots(2 * n, 1, figsize=(15, 3.1 * n), sharex=True,
                             gridspec_kw={"height_ratios": [6, 1] * n})

    for mi, (name, det_on, baseline, fund) in enumerate(METHOD_SPECS):
        ax, axm = axes[2 * mi], axes[2 * mi + 1]
        rr_t, rr_v = _method_rr(name, det_on, baseline, fund, x, refs, fs, low, high,
                                args.specB_window, PPG, offset, m_deps)
        our_on_grid = (np.interp(truth_at, rr_t, rr_v) if rr_t.size
                       else np.full(truth_at.size, np.nan))
        ref_bin = np.digitize(truth_ar, CM.RR_EDGES)
        our_bin = np.digitize(our_on_grid, CM.RR_EDGES)
        valid = (np.isfinite(truth_ar) & np.isfinite(our_on_grid)
                 & (truth_ar >= 4) & (truth_ar <= 55))
        match = valid & (ref_bin == our_bin)
        pct = 100.0 * match[valid].mean() if valid.any() else float("nan")

        for k in range(len(edges) - 1):
            ax.axhspan(edges[k], edges[k + 1], color=band_cols[k % len(band_cols)], alpha=0.18, lw=0)
            ax.text(0.004, (edges[k] + edges[k + 1]) / 2, rr_labels[k],
                    transform=ax.get_yaxis_transform(), fontsize=8, va="center", ha="left", color="#444")
        if truth_at.size:
            ax.plot(truth_at, truth_ar, color="#111827", lw=2.4, label="reference", zorder=5)
        ax.plot(rr_t, rr_v, "-", color="#ef4444", lw=1.2, alpha=0.9, zorder=4, label=f"{name} avg RR")
        ax.plot(truth_at, our_on_grid, ".", color="#7f1d1d", ms=3.5, alpha=0.7, zorder=4.5)
        ax.set_ylabel("RR (bpm)")
        ax.set_ylim(0, 55)
        det = f"+detrend{baseline:g}" if det_on else "+detrend(built-in)"
        ax.set_title(f"{name}{det}  —  correctly categorised: {pct:.0f}%",
                     fontweight="bold", fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.12)

        for i in range(truth_at.size):
            col = "#9ca3af" if not valid[i] else ("#22c55e" if match[i] else "#ef4444")
            axm.axvspan(truth_at[i] - step / 2, truth_at[i] + step / 2, color=col, alpha=0.85, lw=0)
        axm.set_yticks([])
        axm.set_ylabel("match", fontsize=7, rotation=0, ha="right", va="center")

    axes[-1].set_xlabel("REMbo-clock time (s)")
    fig.suptitle(f"{rid} · {args.channel} · 3-method compare · ridge<={RR_BAND_HIGH:g}Hz",
                 fontweight="bold", fontsize=12)
    try:
        fig.canvas.manager.set_window_title(rid)
    except Exception:
        pass
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per recording, compare Butter/MAPAS/Legacy RR vs reference")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--all", action="store_true", help="loop ALL recordings (one figure each)")
    ap.add_argument("--recordings", nargs="+", default=None, help="with --all: restrict to these rec-ids")
    ap.add_argument("--edf", default=None)
    ap.add_argument("--watch", default=None)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--specB-window", type=float, default=32.0)
    ap.add_argument("--out", default=r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\viewer_compare")
    ap.add_argument("--save", default=None, metavar="DIR")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    import dataclasses
    from respiration_rr.settings import PPG as _PPG
    PPG = dataclasses.replace(_PPG, rr_band_high_hz=RR_BAND_HIGH)      # honest ridge ceiling
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr, average_series
    from respiration_rr.ppg.bw_methods import METHODS, REF_METHODS, legacy_detrend
    from spectro_avg_lab import spectro_average
    deps = (run_reference, run_ppg, run_sync, reference_average_rr, average_series,
            METHODS, REF_METHODS, legacy_detrend, spectro_average)

    if args.all or (args.rec_id is None and args.recording is None):
        recs = S.discover(args.data_root)
        if args.recordings:
            recs = [r for r in recs if r[0] in args.recordings]
        if not recs:
            sys.exit("No recordings found.")
        print(f"Comparing 3 methods over {len(recs)} recordings · ridge<={RR_BAND_HIGH:g}Hz")
    else:
        recs = [S._resolve_inputs(args)]

    n_shown = 0
    for rid, edf_path, csv_path in recs:
        print(f"Recording: {rid}")
        fig = _build_figure(rid, edf_path, csv_path, args, PPG, deps, plt)
        if fig is None:
            continue
        n_shown += 1
        if args.save or args.no_show:
            out_dir = args.save or args.out
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"{rid.replace('/', '_')}_{args.channel}_compare.png")
            fig.savefig(png, dpi=110, bbox_inches="tight")
            print(f"  saved -> {png}")
            plt.close(fig)

    if not (args.save or args.no_show) and n_shown:
        plt.show()


if __name__ == "__main__":
    main()
