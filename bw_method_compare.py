#!/usr/bin/env python
"""
bw_method_compare.py — compare BW-extraction methods on one channel.

For each method (MA incumbent, Butterworth, Wavelet, MAPAS) the raw channel is
turned into a BW signal, then the SAME breath detector + RR logic is applied, so
the comparison isolates the extractor. Scored by MAE (bpm) of the breath-by-breath
RR vs the reference (time domain — metric A).

Produces one figure: a BW-trace panel per method (with detected breaths) plus an
RR-vs-reference overlay, and prints the MAE table.

Usage
-----
    py bw_method_compare.py                        # first recording, Artifact
    py bw_method_compare.py --rec-id Exp2/004
    py bw_method_compare.py --channel IR
    py bw_method_compare.py --methods MA Butter Wavelet MAPAS
    py bw_method_compare.py --save DIR --no-show
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

import bw_bp_sweep as S      # reuse discover / _resolve_inputs

METHOD_COLOR = {"MA": "#2563eb", "Butter": "#ef4444", "Wavelet": "#10b981",
                "MAPAS": "#f59e0b", "MAPASref": "#8b5cf6", "MAPASnar": "#ec4899"}
REF_LEDS = ("Green", "Red", "IR")     # reference channels for faithful MAPAS


def _znorm(y):
    y = np.asarray(y, np.float64)
    s = y.std()
    return (y - y.mean()) / s if s > 0 else y - y.mean()


def _within_band(t_cand, r_cand, t_ref, r_ref, tol):
    """Percent of candidate samples within +/- tol bpm of the reference (interp)."""
    t_cand = np.asarray(t_cand, np.float64); r_cand = np.asarray(r_cand, np.float64)
    m = np.isfinite(t_cand) & np.isfinite(r_cand)
    t_cand, r_cand = t_cand[m], r_cand[m]
    if t_cand.size == 0 or t_ref.size < 2:
        return np.nan
    inside = (t_cand >= t_ref[0]) & (t_cand <= t_ref[-1])
    if inside.sum() == 0:
        return np.nan
    ref_at = np.interp(t_cand[inside], t_ref, r_ref)
    return 100.0 * np.mean(np.abs(r_cand[inside] - ref_at) <= tol)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare BW-extraction methods (time-domain RR vs reference)")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--edf", default=None)
    ap.add_argument("--watch", default=None)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--methods", nargs="+", default=["MA", "Butter", "MAPAS", "MAPASref", "MAPASnar"])
    ap.add_argument("--specB-window", type=float, default=16.0, help="FFT window (s) for metric B")
    ap.add_argument("--tol", type=float, default=4.0, help="allowed +/- error band (bpm)")
    ap.add_argument("--no-avg", action="store_true",
                    help="use the raw spectrogram ridge (skip windowed-median), to match category map --no-avg")
    ap.add_argument("--out", default=r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\bw_methods")
    ap.add_argument("--save", default=None, metavar="DIR")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    rid, edf_path, csv_path = S._resolve_inputs(args)
    print(f"Recording: {rid}\n  EDF   : {edf_path}\n  watch : {csv_path}")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    from respiration_rr.settings import PPG
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import Candidate, score_candidate, reference_rr_series
    from respiration_rr.ppg.bw_methods import METHODS, REF_METHODS
    from respiration_rr.ppg.respiration import (_breath_starts_raw, _rr_from_starts, _rr_gate_args)
    from respiration_rr.rr_average import reference_average_rr, average_series, mae_over_overlap
    from spectro_avg_lab import spectro_average

    ref = run_reference(edf_path)
    ppg_results, sig = run_ppg(csv_path)
    offset = run_sync(edf_path, ppg_results, sig)
    ref_t, ref_r = reference_rr_series(ref)               # per-breath (metric A)
    truth_at, truth_ar = reference_average_rr(ref)        # breath-counted average (metric B truth)
    avg_fn = lambda tt, rr: average_series(tt, rr, PPG)

    if args.channel not in sig.channels:
        sys.exit(f"Channel '{args.channel}' not in {list(sig.channels)}")
    x = np.asarray(sig.channels[args.channel], np.float64)
    t = np.asarray(sig.time, np.float64)
    fs = float(sig.fs)
    mr = sig.move_regions
    low, high, order = PPG.bw_band_low_hz, PPG.bw_band_high_hz, PPG.bw_filter_order
    bw_prom = getattr(PPG, "bw_prominence", None) or PPG.breath_start_prominence

    print(f"\n[BW method compare]  channel={args.channel}  band={low}-{high}Hz  "
          f"offset={offset:+.3f}s  reference breaths={ref_t.size}  specB window={args.specB_window:.0f}s  "
          f"tol=+/-{args.tol:.0f}bpm")
    print(f"  {'method':<9} | {'A:MAE':>7} {'breaths':>7} | {'B:MAE':>7} {'promin':>7} {'stab(IQR)':>9} {'in-band':>8}")

    results = {}
    for name in args.methods:
        if name in REF_METHODS:                       # MAPAS variants: LEDs -> ART
            refs = [np.asarray(sig.channels[c], np.float64)
                    for c in REF_LEDS if c in sig.channels and c != args.channel]
            if not refs:
                print(f"  {name:<9}  (no reference LED channels, skipped)")
                continue
            bw = REF_METHODS[name](x, refs, fs, low, high)
        else:
            fn = METHODS.get(name)
            if fn is None:
                print(f"  {name:<9}  (unknown method, skipped)")
                continue
            bw = fn(x, fs, low, high) if name != "MA" else fn(x, fs, low, high, order)
        # --- metric A: breath-by-breath RR vs reference (time domain) ---
        bs = _breath_starts_raw(bw, fs, bw_prom, t=t, cfg=PPG, move_regions=mr)
        bs_t = t[bs] if bs.size else np.zeros(0)
        rr_t, rr = _rr_from_starts(bs_t, *_rr_gate_args(PPG, mr))
        cand = Candidate(label=name, channel="", param="LP", t=rr_t, rr=rr)
        a_mae, a_n = score_candidate(cand, ref_t, ref_r, offset)
        # --- metric B: spectrogram-average (correctness + confidence) ---
        out = spectro_average(bw, fs, args.specB_window, PPG, avg_fn)
        if out is None:
            b_mae = prom = stab = pass_pct = np.nan
            b_at = b_ar = np.zeros(0)
        else:
            st, sr, av_at, av_ar, prom = out
            b_at, b_ar = (st, sr) if args.no_avg else (av_at, av_ar)   # raw ridge vs windowed-median
            b_mae, _ = mae_over_overlap(b_at + offset, b_ar, truth_at, truth_ar)
            stab = (float(np.nanpercentile(sr, 75) - np.nanpercentile(sr, 25))
                    if np.isfinite(sr).any() else np.nan)
            pass_pct = _within_band(b_at + offset, b_ar, truth_at, truth_ar, args.tol)
        results[name] = dict(bw=bw, bs=bs, rr_t=rr_t, rr=rr, a_mae=a_mae,
                             b_mae=b_mae, prom=prom, stab=stab, b_at=b_at, b_ar=b_ar,
                             pass_pct=pass_pct)
        print(f"  {name:<9} | {a_mae:7.2f} {bs.size:7d} | {b_mae:7.2f} {prom:7.1f} "
              f"{stab:9.2f} {pass_pct:7.0f}%")

    names = [m for m in args.methods if m in results]
    if not names:
        sys.exit("No methods produced output.")
    best = min(names, key=lambda m: results[m]["b_mae"] if np.isfinite(results[m]["b_mae"]) else np.inf)
    print(f"\nBEST (metric B — spectral avg MAE): {best}  ({results[best]['b_mae']:.2f} bpm, "
          f"prominence {results[best]['prom']:.1f})")

    # ---- figure: one trace panel per method + an RR-vs-reference overlay ----
    nrow = len(names) + 1
    fig, axes = plt.subplots(nrow, 1, figsize=(15, 2.2 * nrow), sharex=True)
    fig.suptitle(f"BW-extraction methods — {args.channel}   |   {rid}   |   band {low}-{high} Hz",
                 fontweight="bold")
    for ax, name in zip(axes[:-1], names):
        r = results[name]
        col = METHOD_COLOR.get(name, "#333")
        ax.plot(t, _znorm(r["bw"]), "-", color=col, lw=0.8, alpha=0.9)
        if r["bs"].size:
            ax.plot(t[r["bs"]], _znorm(r["bw"])[r["bs"]], "v", color="#111827", ms=5)
        ax.set_title(f"{name}  —  A:MAE {r['a_mae']:.2f} ({r['bs'].size} breaths)  ·  "
                     f"B:MAE {r['b_mae']:.2f} · prom {r['prom']:.1f} · stab {r['stab']:.2f}"
                     + ("   <= best B" if name == best else ""),
                     fontsize=9, loc="left", color=col)
        ax.set_ylabel("BW (z)", fontsize=8)
        ax.grid(True, alpha=0.15)

    axrr = axes[-1]
    if truth_at.size:
        axrr.fill_between(truth_at, truth_ar - args.tol, truth_ar + args.tol,
                          color="#9ca3af", alpha=0.30, label=f"+/-{args.tol:.0f} bpm band", zorder=1)
        axrr.plot(truth_at, truth_ar, "-", color="#111827", lw=2.4, label="reference avg", zorder=5)
    for name in names:
        r = results[name]
        col = METHOD_COLOR.get(name, "#333")
        if r["b_at"].size:
            axrr.plot(r["b_at"] + offset, r["b_ar"], "-", color=col, lw=1.5, alpha=0.9,
                      label=f"{name} ({r['pass_pct']:.0f}% in-band)")
    axrr.set_title(f"metric B — spectrogram average RR vs reference (window {args.specB_window:.0f}s, "
                   f"+/-{args.tol:.0f} bpm band, watch shifted to REMbo clock)", fontsize=10, loc="left")
    axrr.set_ylabel("RR (bpm)", fontsize=8)
    axrr.set_xlabel("REMbo-clock time (s)")
    axrr.grid(True, alpha=0.15)
    axrr.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = args.save or (args.out if args.no_show else None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f"{rid.replace('/', '_')}_{args.channel}_bw_methods.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"  saved -> {png}")

    if args.no_show:
        plt.close(fig)
    else:
        print("\nOpening figure — close the window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
