#!/usr/bin/env python
"""
bw_bp_sweep.py — sweep the BW band-pass cutoffs and score the LP breath-by-breath
RR against the reference, to pick the best (low, high) band.

Target metric (this stage): MAE (bpm) between the LP RR of ONE channel (default
Artifact) and the reference RR, over their time overlap. We look for the (low,
high) band that MINIMISES it. Time-domain only for now (frequency-based scoring
comes later).

Only the BW path is recomputed per grid point — bandpass_filter -> _breath_starts_raw
-> _rr_from_starts, exactly the production functions — so the sweep is faithful and
fast (the recording, reference and sync offset are computed once).

Usage
-----
    py bw_bp_sweep.py                                   # first recording, Artifact, default grid
    py bw_bp_sweep.py --rec-id Exp2/002
    py bw_bp_sweep.py --recording "<dir with edf+csv>"
    py bw_bp_sweep.py --channel IR
    py bw_bp_sweep.py --low 0.05 0.1 0.15 --high 0.7 0.9 1.1
    py bw_bp_sweep.py --order 2 --save out_dir --no-show
"""

import argparse
import dataclasses
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
DEFAULT_OUT = r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\bw_bp_sweep"

DEFAULT_LOW = [0.05, 0.08, 0.10, 0.12, 0.15]
DEFAULT_HIGH = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20]


def _find_inputs(folder):
    edfs = glob.glob(os.path.join(folder, "*.edf"))
    csvs = (glob.glob(os.path.join(folder, "rt_flow*.csv"))
            or glob.glob(os.path.join(folder, "*.csv")))
    return (edfs[0] if edfs else None), (csvs[0] if csvs else None)


def discover(data_root):
    out = []
    for exp in sorted(glob.glob(os.path.join(data_root, "Exp*"))):
        for sub in ("recordings data", "Test"):          # Test/: watch-13 (Exp3) lives here
            rec_root = os.path.join(exp, sub)
            if not os.path.isdir(rec_root):
                continue
            for folder in sorted(glob.glob(os.path.join(rec_root, "*"))):
                edf, csv = _find_inputs(folder)
                if edf and csv:
                    out.append((f"{os.path.basename(exp)}/{os.path.basename(folder)}", edf, csv))
    rf = os.environ.get("REC_FILTER")          # e.g. REC_FILTER=Exp3 or "Exp3/003,Exp3/004"
    if rf:
        subs = [s.strip() for s in rf.split(",") if s.strip()]
        out = [r for r in out if any(s in r[0] for s in subs)]
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


def score_band(x, t, fs, mr, cfg, low, high, order, ref_t, ref_r, offset,
               Candidate, score_candidate):
    """Recompute the LP path for one (low, high) band and score it vs reference.

    Returns (mae, n_overlap, n_breaths, n_rr)."""
    from respiration_rr.ppg.dsp import bandpass_filter
    from respiration_rr.ppg.respiration import (_breath_starts_raw, _rr_from_starts,
                                                _rr_gate_args)
    cfg_b = dataclasses.replace(cfg, bw_band_low_hz=low, bw_band_high_hz=high,
                                bw_filter_order=order)
    lp = bandpass_filter(x, fs, low, high, order)["filtered"]
    bw_prom = getattr(cfg_b, "bw_prominence", None)
    if bw_prom is None:
        bw_prom = cfg_b.breath_start_prominence
    bs = _breath_starts_raw(lp, fs, bw_prom, t=t, cfg=cfg_b, move_regions=mr)
    bs_t = t[bs] if bs.size else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs_t, *_rr_gate_args(cfg_b, mr))
    cand = Candidate(label="LP", channel="", param="LP", t=rr_t, rr=rr)
    mae, n = score_candidate(cand, ref_t, ref_r, offset)
    return mae, n, int(bs.size), int(rr_t.size)


def _heatmap(rid, ch, lows, highs, M, best, plt):
    fig, ax = plt.subplots(figsize=(1.4 * len(highs) + 3, 1.0 * len(lows) + 2.5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(highs)))
    ax.set_xticklabels([f"{h:.2f}" for h in highs])
    ax.set_yticks(range(len(lows)))
    ax.set_yticklabels([f"{l:.2f}" for l in lows])
    ax.set_xlabel("high cutoff (Hz)")
    ax.set_ylabel("low cutoff (Hz)")
    ax.set_title(f"{ch} — LP RR vs reference  MAE (bpm)   |   {rid}\n"
                 f"best: low={best[0]:.2f} high={best[1]:.2f}  MAE={best[2]:.2f} (n={best[3]})",
                 fontsize=10, fontweight="bold")
    for i in range(len(lows)):
        for j in range(len(highs)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white", fontsize=8)
    bi, bj = lows.index(best[0]), highs.index(best[1])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="#ef4444", lw=2.5))
    fig.colorbar(im, ax=ax, label="MAE (bpm)")
    fig.tight_layout()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sweep BW band-pass cutoffs; score LP RR vs reference")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None, help="recording id, e.g. Exp2/002")
    ap.add_argument("--recording", default=None, help="folder with one .edf + one .csv")
    ap.add_argument("--edf", default=None)
    ap.add_argument("--watch", default=None)
    ap.add_argument("--channel", default="Artifact", help="channel to score (default Artifact)")
    ap.add_argument("--low", type=float, nargs="+", default=DEFAULT_LOW, help="low cutoffs (Hz)")
    ap.add_argument("--high", type=float, nargs="+", default=DEFAULT_HIGH, help="high cutoffs (Hz)")
    ap.add_argument("--order", type=int, default=None, help="BW filter order (default = settings)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--save", default=None, metavar="DIR", help="save PNG + CSV into DIR")
    ap.add_argument("--no-show", action="store_true", help="do not open windows")
    args = ap.parse_args(argv)

    rid, edf_path, csv_path = _resolve_inputs(args)
    print(f"Recording: {rid}\n  EDF   : {edf_path}\n  watch : {csv_path}")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    from respiration_rr.settings import PPG
    from main import run_reference, run_ppg, run_sync            # exact pipeline wiring
    from respiration_rr.compare.compare import (Candidate, score_candidate,
                                                reference_rr_series)

    order = args.order if args.order is not None else PPG.bw_filter_order

    ref = run_reference(edf_path)
    ppg_results, sig = run_ppg(csv_path)                         # baseline run (current settings)
    offset = run_sync(edf_path, ppg_results, sig)
    ref_t, ref_r = reference_rr_series(ref)

    if args.channel not in sig.channels:
        sys.exit(f"Channel '{args.channel}' not in {list(sig.channels)}")
    x = np.asarray(sig.channels[args.channel], np.float64)
    t = np.asarray(sig.time, np.float64)
    fs = float(sig.fs)
    mr = sig.move_regions

    lows = list(args.low)
    highs = list(args.high)
    print(f"\n[BW BP sweep]  channel={args.channel}  order={order}  "
          f"offset={offset:+.3f}s  reference breaths={ref_t.size}")
    print(f"  grid: low={lows}  high={highs}")
    print(f"  {'low':>5} {'high':>5} {'MAE(bpm)':>9} {'n':>5} {'breaths':>8} {'rr':>5}")

    M = np.full((len(lows), len(highs)), np.nan)
    rows = []
    best = (None, None, np.inf, 0)
    for i, low in enumerate(lows):
        for j, high in enumerate(highs):
            if low >= high:
                continue
            mae, n, nb, nrr = score_band(x, t, fs, mr, PPG, low, high, order,
                                         ref_t, ref_r, offset, Candidate, score_candidate)
            M[i, j] = mae
            rows.append((low, high, mae, n, nb, nrr))
            flag = ""
            if np.isfinite(mae) and mae < best[2]:
                best = (low, high, mae, n)
                flag = "  <= best"
            print(f"  {low:5.2f} {high:5.2f} {mae:9.2f} {n:5d} {nb:8d} {nrr:5d}{flag}")

    if best[0] is None:
        sys.exit("No scorable band (check overlap / reference).")
    print(f"\nBEST: low={best[0]:.2f} high={best[1]:.2f}  MAE={best[2]:.2f} bpm (n={best[3]})")

    fig = _heatmap(rid, args.channel, lows, highs, M, best, plt)
    out_dir = args.save or (args.out if args.no_show else None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem = f"{rid.replace('/', '_')}_{args.channel}_bw_bp_sweep"
        png = os.path.join(out_dir, stem + ".png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        csvp = os.path.join(out_dir, stem + ".csv")
        with open(csvp, "w", encoding="utf-8") as f:
            f.write("low_hz,high_hz,mae_bpm,n_overlap,n_breaths,n_rr\n")
            for r in rows:
                f.write("%.3f,%.3f,%.4f,%d,%d,%d\n" % r)
        print(f"  saved -> {png}\n  saved -> {csvp}")

    if args.no_show:
        plt.close(fig)
    else:
        print("\nOpening heatmap — close the window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
