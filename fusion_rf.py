#!/usr/bin/env python
"""
fusion_rf.py — fuse the per-CATEGORY BBBRR of every param x channel into one BBBRR, scored per
breath against the reference (REMbo), leave-one-recording-out. Standalone lab tool; reads the
per-category analysis from per_category_lab (step 3) and does not modify it.

Features (all on a 1 s common grid, REMbo clock):
  * 3 per-beat params (RSA/RIIV/AUC), ssp, AFTER per-category analysis, each channel   (per_category_series, detrend)
  * LP (global BW) of the optical channels (Green/Red/IR/Yellow)
  * Artifact BW hybrid (per_category_lab)
  * context: HR, category (RR bin from the reference split — see leakage note), spread/agreement,
    n_present, prev_rr (previous grid point's consensus — temporal memory)
Label y = reference RR on the grid. groups = recording.

Model: HistGradientBoostingRegressor (native NaN — a missing estimator is just NaN) or RF.
Validation: LeaveOneGroupOut (recording). Metrics: per-breath MAE (primary; model prediction
sampled at each reference breath time), per-category MAE (high-rate emphasis), grid MAE — all vs
baseline = median of the estimators. Permutation importance on the all-data model (trend only).

FEASIBILITY STUDY: few recordings -> read numbers as trend, not absolute. `category` is derived
from the reference (oracle) — a partial leak; in deployment it must be an ESTIMATED category.

Usage
-----
    py fusion_rf.py
    py fusion_rf.py --recs Exp3/003 Exp3/004 Exp3/005 Exp3/006 Exp3/007 Exp3/008
    py fusion_rf.py --model rf
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
import bw_category_map as CM
import per_category_lab as PCL

DEFAULT_RECS = [f"Exp3/{i:03d}" for i in range(1, 9)]
GRID_SEC = 1.0
BASES = ("RSA", "RIIV", "AUC")
METHOD = "ssp"
ALL_CH = ("Green", "Red", "IR", "Yellow", "Artifact")
OPTICAL = ("Green", "Red", "IR", "Yellow")
RR_LABELS = ("<15", "15-22", "22-30", "30+")


def _rr_on_grid(rr_time_watch, rr_bpm, offset, grid):
    """Interpolate an RR series (watch clock) onto the REMbo grid; NaN outside its own span."""
    rt = np.asarray(rr_time_watch, float) + offset
    rv = np.asarray(rr_bpm, float)
    if rt.size < 2:
        return np.full(grid.shape, np.nan)
    o = np.argsort(rt)
    return np.interp(grid, rt[o], rv[o], left=np.nan, right=np.nan)


def _best_lag(est_t, est_r, ref_bt, ref_br, max_lag=5.0, step=0.5):
    """Per-graph time alignment: the shift (s) ADDED to est times that best correlates this
    estimator's RR curve with the reference RR curve. 0 if too little overlap."""
    lo = max(est_t[0], ref_bt[0]); hi = min(est_t[-1], ref_bt[-1])
    if hi - lo < 10:
        return 0.0
    g = np.arange(lo, hi, step)
    b = np.interp(g, ref_bt, ref_br); b = b - b.mean()
    if b.std() < 1e-6:
        return 0.0
    best, bl, d = -1e9, 0.0, -max_lag
    while d <= max_lag + 1e-9:
        a = np.interp(g, est_t + d, est_r); a = a - a.mean()
        if a.std() > 1e-6:
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best:
                best, bl = c, d
        d += step
    return bl


def _match_to_breaths(est_t_watch, est_r, offset, ref_bt, ref_br):
    """Align this estimator (its own breath points) to the reference, then for each reference
    breath take the estimator's NEAREST breath value within +/- half that breath's period.
    No breath in the window -> NaN (the estimator missed this breath). Not interpolation."""
    et = np.asarray(est_t_watch, float) + offset
    er = np.asarray(est_r, float)
    out = np.full(np.shape(ref_bt), np.nan)
    if et.size < 2:
        return out
    o = np.argsort(et); et, er = et[o], er[o]
    et = et + _best_lag(et, er, ref_bt, ref_br)             # per-graph alignment
    for i in range(ref_bt.size):
        tol = 0.5 * 60.0 / max(ref_br[i], 1e-6)            # half the breath's own period
        j = int(np.argmin(np.abs(et - ref_bt[i])))
        if abs(et[j] - ref_bt[i]) <= tol:
            out[i] = er[j]
    return out


def build_recording(rid, edf, csv, channels, route="ref", rows="breaths"):
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import reference_rr_series
    from respiration_rr.rr_average import reference_average_rr, reference_rr_points
    import ref_split_viewer as V

    ref = run_reference(edf)
    results, sig = run_ppg(csv)
    offset = run_sync(edf, results, sig)
    pt, pr = reference_rr_points(ref)                       # the actual reference breaths (time, rate)
    pt, pr = np.asarray(pt, float), np.asarray(pr, float)
    hr_t, hr_v = CM.hr_series_from_beats(results, sig, offset)
    # category split: "ref" = reference RR (oracle), "ours" = Artifact-BW spectrogram ridge (own signal)
    if route == "ours":
        at, ar = PCL._our_rr_curve(sig)
        at = at + offset                                    # watch clock -> REMbo clock (match HR + reference + runs)
    else:
        at, ar = reference_average_rr(ref)
    runs = V._segment_by_crossing(at, ar, hr_t, hr_v, PCL.MIN_SUBSEG_SEC,
                                  PCL.RR_EDGES, CM.HR_EDGES, fill=True)

    # rows: "breaths" = one row per reference breath (nearest-match features, per-graph aligned);
    #       "grid"    = a 1 s grid (interpolated features)
    if rows == "breaths":
        grid, y = pt, pr
        feat_of = lambda et, er: _match_to_breaths(et, er, offset, pt, pr)
    else:
        rt, rv = reference_rr_series(ref)
        grid = np.arange(rt[0], rt[-1], GRID_SEC)
        y = np.interp(grid, rt, rv, left=np.nan, right=np.nan)
        feat_of = lambda et, er: _rr_on_grid(et, er, offset, grid)

    cols = {}
    # 3 per-beat params, per-category analysis (ssp), each channel
    for ch in [c for c in ALL_CH if c in results and c in channels]:
        mr = results[ch].move_regions
        for base in BASES:
            gp = results[ch].params.get(base + "_" + METHOD)
            if gp is None or not np.size(getattr(gp, "series_y", [])):
                continue
            sx, sy = np.asarray(gp.series_x), np.asarray(gp.series_y)
            try:
                _, _, _, rr_t, rr = PCL.per_category_series(base, METHOD, sx, sy, mr, runs,
                                                            offset, detrend=True)
                if np.size(rr) >= 2:
                    cols[f"{ch}:{base}"] = feat_of(rr_t, rr)
            except Exception as e:
                print(f"  [warn] {rid} {ch}:{base} per-cat skipped ({e})")
    # LP (global BW) of the optical channels
    for ch in [c for c in OPTICAL if c in results and c in channels]:
        lp = results[ch].params.get("LP")
        if lp is not None and np.size(getattr(lp, "rr_bpm", [])) >= 2:
            cols[f"{ch}:LP"] = feat_of(lp.rr_time, lp.rr_bpm)
    # Artifact BW hybrid
    if "Artifact" in results and "Artifact" in channels:
        try:
            _, _, _, ht, hr_ = PCL.bw_artifact_hybrid_series(
                sig.channels["Artifact"], sig.time, sig.fs,
                results["Artifact"].move_regions, runs, offset)
            if np.size(hr_) >= 2:
                cols["Artifact:BW_hybrid"] = feat_of(ht, hr_)
        except Exception as e:
            print(f"  [warn] {rid} Artifact:BW_hybrid skipped ({e})")

    # context features
    hr = (np.interp(grid, np.asarray(hr_t, float), np.asarray(hr_v, float), left=np.nan, right=np.nan)
          if np.size(hr_t) else np.full(grid.shape, np.nan))
    idx_of, rb_of = PCL._run_indexer(runs)
    ridx = idx_of(grid)                                     # grid is REMbo; runs are REMbo
    cat = np.where(ridx >= 0, rb_of[np.clip(ridx, 0, len(runs) - 1)], -1).astype(float)
    return dict(rid=rid, grid=grid, y=y, cols=cols, hr=hr, cat=cat, ref_t=pt, ref_r=pr)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fuse per-category BBBRR (LOSO), per-breath scored")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--recs", nargs="+", default=None,
                    help="recording ids; default = ALL Exp3 recordings discover finds (new ones auto-included)")
    ap.add_argument("--channels", nargs="+", default=list(ALL_CH),
                    help="channels to include (default all 5); e.g. --channels Green Red IR Yellow")
    ap.add_argument("--model", choices=("hgb", "rf"), default="hgb")
    ap.add_argument("--route", choices=("ref", "ours"), default="ref",
                    help="category split: ref = reference RR (oracle); ours = Artifact-BW spectrogram ridge")
    ap.add_argument("--rows", choices=("breaths", "grid"), default="breaths",
                    help="breaths = one row per reference breath, per-graph aligned nearest-match features "
                         "(default); grid = 1 s grid with interpolated features")
    args = ap.parse_args(argv)

    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.inspection import permutation_importance
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor as Model
        mk = lambda: Model(random_state=42)
    else:
        from sklearn.ensemble import RandomForestRegressor as Model
        mk = lambda: Model(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=42)

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    disc = S.discover(args.data_root)
    if args.recs:
        found = {r[0]: r for r in disc}
        recs = [found[r] for r in args.recs if r in found]
    else:                                                    # default: every Exp3 recording found
        recs = sorted((r for r in disc if r[0].startswith("Exp3/")), key=lambda r: r[0])
    if not recs:
        sys.exit("no recordings found")
    print(f"  recordings: {[r[0] for r in recs]}")

    data = []
    for rid, edf, csv in recs:
        try:
            d = build_recording(rid, edf, csv, args.channels, args.route, args.rows)
            data.append(d)
            print(f"  built {rid}: {len(d['cols'])} method columns, {d['grid'].size} grid pts")
        except Exception as e:
            print(f"  [ERR] {rid}: {e}")
    if not data:
        sys.exit("no recordings built")

    names = sorted(set().union(*[d["cols"].keys() for d in data]))
    print("\n  method feature columns:")
    for c in sorted(set(n.split(":")[0] for n in names)):
        print(f"     {c:9}: {', '.join(k.split(':', 1)[1] for k in names if k.startswith(c + ':'))}")
    print("     + context: HR, category, spread, n_present, prev_rr")

    # assemble per-recording blocks (prev_rr computed WITHIN a recording -> no cross-rec leak)
    Xparts, yparts, gparts, gridparts = [], [], [], []
    base_parts = []                                          # median-of-estimators baseline
    for d in data:
        n = d["grid"].size
        M = np.column_stack([d["cols"].get(c, np.full(n, np.nan)) for c in names])
        consensus = np.nanmedian(M, axis=1)
        spread = np.nan_to_num(np.nanstd(M, axis=1), nan=0.0)
        n_present = np.sum(~np.isnan(M), axis=1).astype(float)
        prev_rr = np.concatenate([[np.nan], consensus[:-1]])
        ctx = np.column_stack([d["hr"], d["cat"], spread, n_present, prev_rr])
        Xparts.append(np.column_stack([M, ctx]))
        yparts.append(d["y"]); gparts.append(np.array([d["rid"]] * n)); gridparts.append(d["grid"])
        base_parts.append(consensus)
    X = np.vstack(Xparts); y = np.concatenate(yparts)
    groups = np.concatenate(gparts); grid = np.concatenate(gridparts)
    baseline = np.concatenate(base_parts)
    feat_names = list(names) + ["HR", "category", "spread", "n_present", "prev_rr"]

    any_present = np.sum(~np.isnan(X[:, :len(names)]), axis=1)
    keep = np.isfinite(y) & (any_present > 0)
    X, y, groups, grid, baseline = X[keep], y[keep], groups[keep], grid[keep], baseline[keep]
    if args.model == "rf":                                   # RF can't take NaN; HGB can
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
    print(f"\n  matrix: {X.shape[0]} rows x {X.shape[1]} features  (model={args.model})\n")

    # ---- LOSO ----
    logo = LeaveOneGroupOut()
    preds = np.full(y.shape, np.nan)
    for tr, te in logo.split(X, y, groups):
        preds[te] = mk().fit(X[tr], y[tr]).predict(X[te])

    def mae(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else np.nan

    def rmsd(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else np.nan

    # ---- per-breath metric: sample the LOSO prediction at each reference breath time ----
    pb_err, pb_cat = [], []
    for d in data:
        te = groups == d["rid"]
        if not te.any():
            continue
        gt, pr = grid[te], preds[te]
        o = np.argsort(gt)
        tb, rb = np.asarray(d["ref_t"], float), np.asarray(d["ref_r"], float)
        inb = (tb >= gt[o][0]) & (tb <= gt[o][-1])
        pr_at = np.interp(tb[inb], gt[o], pr[o])
        pb_err.append(np.abs(pr_at - rb[inb]))
        pb_cat.append(np.digitize(rb[inb], PCL.RR_EDGES))
    pb_err = np.concatenate(pb_err) if pb_err else np.zeros(0)
    pb_cat = np.concatenate(pb_cat) if pb_cat else np.zeros(0)

    pb_rmsd = float(np.sqrt(np.mean(pb_err ** 2))) if pb_err.size else np.nan
    print("=" * 64)
    print(f"  per-breath MAE  (fusion)  = {np.mean(pb_err):.2f} bpm   (n={pb_err.size})")
    print(f"  per-breath RMSD (fusion)  = {pb_rmsd:.2f} bpm   (pooled over all recordings)")
    print(f"  grid MAE  (fusion)        = {mae(preds, y):.2f} bpm")
    print(f"  grid RMSD (fusion)        = {rmsd(preds, y):.2f} bpm")
    print(f"  grid MAE  (baseline=median)= {mae(baseline, y):.2f} bpm")
    print(f"  grid RMSD (baseline=median)= {rmsd(baseline, y):.2f} bpm")
    print("=" * 64)
    print("  per-breath MAE by reference category:")
    for b, lab in enumerate(RR_LABELS):
        m = pb_cat == b
        if m.any():
            print(f"     {lab:6}: {np.mean(pb_err[m]):.2f} bpm  (n={int(m.sum())})")
    print("\n  per-recording per-breath MAE:")
    for d, e in zip(data, pb_err_split(data, groups, grid, preds)):
        print(f"     {d['rid']}: {e:.2f} bpm")

    # ---- permutation importance (all-data model, trend only) ----
    full = mk().fit(X, y)
    pi = permutation_importance(full, X, y, n_repeats=8, random_state=42, n_jobs=-1)
    order = np.argsort(pi.importances_mean)[::-1]
    print("\n  feature importance (permutation, all-data — TREND ONLY, small N):")
    for j in order[:15]:
        print(f"     {feat_names[j]:20} {pi.importances_mean[j]:.3f}")

    print("\n  NOTE: few recordings -> treat all numbers as TREND, not absolute. 'category' is "
          "reference-derived (oracle); a real deployment needs an estimated category.")

    # ---- plots: fused vs reference, per recording ----
    for d in data:
        te = groups == d["rid"]
        if not te.any():
            continue
        gt, pr = grid[te], preds[te]
        o = np.argsort(gt)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(d["ref_t"], d["ref_r"], "-", color="#0f172a", lw=1.8, label="reference BBBRR")
        ax.plot(gt[o], pr[o], "-", color="#dc2626", lw=1.3, alpha=0.9, label="fused BBBRR")
        ax.set_title(f"{d['rid']} — fused ({args.model}) vs reference", fontsize=11)
        ax.set_xlabel("time (s, REMbo clock)"); ax.set_ylabel("RR (bpm)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        fig.tight_layout()

    # ---- feature-importance chart ----
    top = order[:20][::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(top)), pi.importances_mean[top], color="#2563eb")
    ax.set_yticks(range(len(top))); ax.set_yticklabels([feat_names[j] for j in top], fontsize=8)
    ax.set_title("Permutation importance (all-data, trend only)", fontsize=11)
    ax.set_xlabel("importance"); ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()

    print("\n  showing figures — close the windows to exit.")
    plt.show()


def pb_err_split(data, groups, grid, preds):
    """Per-recording per-breath MAE (helper for the printout)."""
    out = []
    for d in data:
        te = groups == d["rid"]
        if not te.any():
            out.append(float("nan")); continue
        gt, pr = grid[te], preds[te]
        o = np.argsort(gt)
        tb, rb = np.asarray(d["ref_t"], float), np.asarray(d["ref_r"], float)
        inb = (tb >= gt[o][0]) & (tb <= gt[o][-1])
        if not inb.any():
            out.append(float("nan")); continue
        pr_at = np.interp(tb[inb], gt[o], pr[o])
        out.append(float(np.mean(np.abs(pr_at - rb[inb]))))
    return out


if __name__ == "__main__":
    main()
