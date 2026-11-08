#!/usr/bin/env python
"""
fusion_rf.py — fuse the per-CATEGORY BBBRR of every param x channel into one BBBRR, scored per
breath against the reference (REMbo), leave-one-recording-out. Standalone lab tool; reads the
per-category analysis from per_category_lab (step 3) and does not modify it.

Features (all on a 1 s common grid, REMbo clock):
  * 3 per-beat params (RSA/AMP/AUC), ssp, AFTER per-category analysis, each channel   (per_category_series, detrend)
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
import json
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
import noise_zones as NZ

DEFAULT_RECS = [f"Exp3/{i:03d}" for i in range(1, 9)]
GRID_SEC = 1.0
BASES = ("RSA", "AMP", "AUC")
METHOD = "ssp"
ALL_CH = ("Green", "Red", "IR", "Yellow", "Artifact")
OPTICAL = ("Green", "Red", "IR", "Yellow")
RR_LABELS = ("<15", "15-22", "22-30", "30+")


def _edges_to_labels(edges):
    """('<e0', 'e0-e1', ..., 'eN+') labels for a list of RR bin edges (bpm)."""
    e = [("%g" % x) for x in edges]
    return tuple(["<" + e[0]] + ["%s-%s" % (e[i], e[i + 1]) for i in range(len(e) - 1)] + [e[-1] + "+"])


# ----------------------------------------------------------------------------
# Alternative fusion models (besides HGB/RF), sklearn-compatible (fit/predict/score).
# ----------------------------------------------------------------------------
def _r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2:
        return 0.0
    ss_res = float(np.sum((y[m] - p[m]) ** 2)); ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class RobustMean:
    """(A) No-training fusion: Tukey-biweight robust location across the per-estimator RR columns
    (the first n_est feature columns = the Channel:param detected-RR curves). Ignores y; downweights
    an outlier estimator per sample, so one bad channel/param doesn't drag the fused value."""
    _estimator_type = "regressor"

    def __init__(self, n_est=0, c=6.0):
        self.n_est = n_est; self.c = c

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        a = np.asarray(X, float)[:, :self.n_est]
        med = np.nanmedian(a, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(a - med), axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = (a - med) / (self.c * mad + 1e-12)
            w = (1.0 - u ** 2) ** 2
        w[np.abs(u) >= 1] = 0.0
        w[~np.isfinite(a)] = 0.0
        num = np.nansum(w * np.nan_to_num(a), axis=1)
        den = np.nansum(w, axis=1)
        loc = np.where(den > 0, num / np.where(den > 0, den, 1.0), med[:, 0])
        return np.where(np.isfinite(loc), loc, np.nan_to_num(med[:, 0]))

    def score(self, X, y):
        return _r2(np.asarray(y, float), self.predict(X))


class CategoryMoE:
    """(C) Mixture-of-experts: one regularized-linear expert (Ridge, imputed + scaled) per RR
    category, routed by the 'category' feature column (cat_idx). A category with too few training
    samples, or an unknown/invalid category at predict, falls back to a global expert. Gives each
    channel/param weight WITHIN the band where it is reliable instead of one global compromise."""
    _estimator_type = "regressor"

    def __init__(self, cat_idx=None, n_bins=4, alpha=1.0, min_n=8):
        self.cat_idx = cat_idx; self.n_bins = n_bins; self.alpha = alpha; self.min_n = min_n

    def _mk(self):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("rg", Ridge(alpha=self.alpha))])

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        cat = np.rint(X[:, self.cat_idx])
        self.global_ = self._mk().fit(X, y)
        self.experts_ = {}
        for b in range(self.n_bins):
            m = cat == b
            if int(m.sum()) >= self.min_n:
                self.experts_[b] = self._mk().fit(X[m], y[m])
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        cat = np.rint(X[:, self.cat_idx])
        out = self.global_.predict(X)
        for b, ex in self.experts_.items():
            m = cat == b
            if m.any():
                out[m] = ex.predict(X[m])
        return out

    def score(self, X, y):
        return _r2(np.asarray(y, float), self.predict(X))


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


def _agree3_rr(bs_list, tol=0.5, min_votes=2):
    """Intra-channel agreement across a channel's param breath-time arrays (RSA/AMP/AUC, watch
    clock). Pool all breath times, cluster by gaps <= tol; keep a breath only when >= min_votes
    DISTINCT params place one in the cluster, at the MEDIAN of the voters (so a slightly mistimed
    param is pulled to the majority; a lone false breath is dropped; a breath 2 params agree on
    survives even if the third missed it). Returns (times, rr=60/interval)."""
    pooled = []
    for src, arr in enumerate(bs_list):
        for t in np.asarray(arr, float).ravel():
            pooled.append((float(t), src))
    if len(pooled) < 2:
        return np.zeros(0), np.zeros(0)
    pooled.sort()
    clusters = [[pooled[0]]]
    for t, s in pooled[1:]:
        if t - clusters[-1][-1][0] <= tol:
            clusters[-1].append((t, s))
        else:
            clusters.append([(t, s)])
    cons = np.array(sorted(float(np.median([t for t, _ in cl])) for cl in clusters
                           if len({s for _, s in cl}) >= min_votes))
    if cons.size < 2:
        return np.zeros(0), np.zeros(0)
    d = np.diff(cons); ok = d > 1e-6
    return cons[1:][ok], (60.0 / d)[ok]


def _gy_combine(gx, gy, yx, yy):
    """Signal-level Green+Yellow combine for a per-beat param: z-score each channel's series, put
    both on a common uniform grid (finer median beat-spacing) over their overlap, average. Boosts
    the breathing SNR (signal adds, noise partly cancels) before per-category detection. Average,
    not sum, so the stronger channel can't dominate. Returns (common_times, combined_series)."""
    gx, gy, yx, yy = map(lambda a: np.asarray(a, float), (gx, gy, yx, yy))
    t0, t1 = max(gx[0], yx[0]), min(gx[-1], yx[-1])
    if t1 - t0 < 10 or gx.size < 4 or yx.size < 4:
        return np.zeros(0), np.zeros(0)
    dt = max(0.1, min(np.median(np.diff(gx)), np.median(np.diff(yx))))
    cg = np.arange(t0, t1, dt)
    a = np.interp(cg, gx, gy); b = np.interp(cg, yx, yy)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return cg, 0.5 * (a + b)


def build_recording(rid, edf, csv, channels, route="ref", rows="breaths", agree3=False, gy=False, cat_ridge=None):
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import reference_rr_series
    from respiration_rr.rr_average import reference_average_rr, reference_rr_points
    import ref_split_viewer as V

    ref = run_reference(edf)
    results, sig = run_ppg(csv)
    offset = run_sync(edf, results, sig)
    pt, pr = reference_rr_points(ref)                       # the actual reference breaths (time, rate)
    pt, pr = np.asarray(pt, float), np.asarray(pr, float)
    tav_ref, rav_ref = reference_average_rr(ref)            # smoothed (20s windowed-median) reference curve
    pr_smooth = np.interp(pt, np.asarray(tav_ref, float), np.asarray(rav_ref, float), left=np.nan, right=np.nan)
    hr_t, hr_v = CM.hr_series_from_beats(results, sig, offset)
    # category split: "ref" = reference RR (oracle), "ours" = Artifact-BW spectrogram ridge (own signal)
    if route == "ours":
        if cat_ridge is not None and rid in cat_ridge:      # precomputed ML categorization ridge (LOSO)
            at, ar = np.asarray(cat_ridge[rid][0], float), np.asarray(cat_ridge[rid][1], float)
        else:
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
        bsb = []                                            # per-param breath times (for agree3)
        for base in BASES:
            gp = results[ch].params.get(base + "_" + METHOD)
            if gp is None or not np.size(getattr(gp, "series_y", [])):
                continue
            sx, sy = np.asarray(gp.series_x), np.asarray(gp.series_y)
            try:
                _, _, bs, rr_t, rr = PCL.per_category_series(base, METHOD, sx, sy, mr, runs,
                                                             offset, detrend=True)
                if np.size(rr) >= 2:
                    cols[f"{ch}:{base}"] = feat_of(rr_t, rr)
                if np.size(bs) >= 2:
                    bsb.append(np.asarray(bs, float))
            except Exception as e:
                print(f"  [warn] {rid} {ch}:{base} per-cat skipped ({e})")
        if agree3 and len(bsb) >= 2:                         # intra-channel 3-param agreement -> feature
            ct, cr = _agree3_rr(bsb)
            if ct.size >= 2:
                cols[f"{ch}:agree3"] = feat_of(ct, cr)
    # Green+Yellow signal-level combine per per-beat param (the two trusted channels) -> GY:<base>
    if gy and all(c in results and c in channels for c in ("Green", "Yellow")):
        mrG = results["Green"].move_regions
        for base in BASES:
            gpG = results["Green"].params.get(base + "_" + METHOD)
            gpY = results["Yellow"].params.get(base + "_" + METHOD)
            if gpG is None or gpY is None:
                continue
            cg, comb = _gy_combine(gpG.series_x, gpG.series_y, gpY.series_x, gpY.series_y)
            if cg.size < 4:
                continue
            try:
                _, _, _, rr_t, rr = PCL.per_category_series(base, METHOD, cg, comb, mrG, runs,
                                                            offset, detrend=True)
                if np.size(rr) >= 2:
                    cols[f"GY:{base}"] = feat_of(rr_t, rr)
            except Exception as e:
                print(f"  [warn] {rid} GY:{base} skipped ({e})")
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
    # our division curve (for the plot): 'at' is already on the REMbo clock above
    return dict(rid=rid, grid=grid, y=y, cols=cols, hr=hr, cat=cat, ref_t=pt, ref_r=pr,
                ref_r_smooth=pr_smooth, our_t=np.asarray(at, float), our_r=np.asarray(ar, float))


RR_EDGES = [15.0, 22.0, 30.0]                               # category boundaries (bpm); --rr-edges overrides
CAT_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]   # <15 / 15-22 / 22-30 / 30+


def _under_test(r):
    """True if a discovered recording lives under a 'Test' folder -> excluded from default runs.
    Pass explicit --recs to include a Test recording on purpose."""
    return any(part.lower() == "test" for p in (r[1], r[2]) if p
               for part in os.path.normpath(p).split(os.sep))


def _exclude_red(d, red):
    """Drop every sample/breath inside a red zone from a built recording dict (REMbo clock), so NO
    stage scores or trains on it and no curve is interpolated across it. Each array is masked once,
    routed by its ORIGINAL length to the matching clock: grid-aligned (grid/y/hr/cat/cols) by `grid`,
    breath arrays (ref_t/ref_r/ref_r_smooth, and cat in the cat-only path) by `ref_t`, the division
    curve (our_t/our_r) by `our_t`. Length-matched so it is a no-op on anything it can't place."""
    if not red:
        return d
    masks = {}
    if np.size(d.get("grid", [])):
        masks["grid"] = (len(d["grid"]), ~NZ.in_red(d["grid"], red))
    if np.size(d.get("ref_t", [])):
        masks["ref"] = (len(d["ref_t"]), ~NZ.in_red(d["ref_t"], red))
    if np.size(d.get("our_t", [])):
        masks["our"] = (len(d["our_t"]), ~NZ.in_red(d["our_t"], red))

    def pick(n):                                             # mask whose clock has this length
        for key in ("grid", "ref", "our"):
            if key in masks and masks[key][0] == n:
                return masks[key][1]
        return None

    for k in ("grid", "y", "hr", "cat", "ref_t", "ref_r", "ref_r_smooth", "our_t", "our_r"):
        if k in d and d[k] is not None and np.size(d[k]):
            m = pick(len(np.asarray(d[k])))
            if m is not None:
                d[k] = np.asarray(d[k])[m]
    if "cols" in d:
        d["cols"] = {k: (np.asarray(v)[pick(len(np.asarray(v)))]
                         if pick(len(np.asarray(v))) is not None else v) for k, v in d["cols"].items()}
    return d


def _load_red(data, noise_dir):
    """Attach d['red'] (merged zones) to each recording and return (stats, span_by_rid).
    stats: total noise sec, total span sec, pct, + per-recording. Span = reference-breath time range."""
    tot_noise = tot_span = 0.0
    by_type_tot = {}
    per_rec = []
    for d in data:
        red, by_type = NZ.manual_zones(d["rid"], noise_dir)
        d["red"] = red
        t = np.asarray(d.get("ref_t", d.get("grid", [])), float)
        span = float(t.max() - t.min()) if t.size else 0.0
        nsec = NZ.total_seconds(red)
        tot_noise += nsec; tot_span += span
        types = {k: NZ.total_seconds(v) for k, v in by_type.items()}
        for k, v in types.items():
            by_type_tot[k] = by_type_tot.get(k, 0.0) + v
        per_rec.append({"rid": d["rid"], "noise_sec": nsec, "span_sec": span,
                        "pct": (100.0 * nsec / span) if span > 0 else 0.0, "types": types})
    pct = (100.0 * tot_noise / tot_span) if tot_span > 0 else 0.0
    return {"noise_sec": tot_noise, "span_sec": tot_span, "pct": pct,
            "by_type_sec": by_type_tot, "n_recs": len(data), "per_rec": per_rec}


def _fmt_dur(s):
    """'190s (3.2 min)'"""
    return f"{s:.0f}s ({s/60:.1f} min)"


def build_categorization(rid, edf, csv, route="ref", cat_ridge=None):
    """Light path (--stage cat): category split only, no per-category feature building, no model.
    Returns our predicted category at each reference breath + the division curve, for diagnostics."""
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr, reference_rr_points
    import ref_split_viewer as V
    ref = run_reference(edf); results, sig = run_ppg(csv); offset = run_sync(edf, results, sig)
    pt, pr = reference_rr_points(ref); pt, pr = np.asarray(pt, float), np.asarray(pr, float)
    tav_ref, rav_ref = reference_average_rr(ref)
    pr_smooth = np.interp(pt, np.asarray(tav_ref, float), np.asarray(rav_ref, float), left=np.nan, right=np.nan)
    hr_t, hr_v = CM.hr_series_from_beats(results, sig, offset)
    if route == "ours":
        if cat_ridge is not None and rid in cat_ridge:                         # precomputed ML ridge (LOSO)
            at, ar = np.asarray(cat_ridge[rid][0], float) + offset, np.asarray(cat_ridge[rid][1], float)
        else:
            at, ar = PCL._our_rr_curve(sig); at = np.asarray(at, float) + offset   # watch -> REMbo
    else:
        at, ar = reference_average_rr(ref); at = np.asarray(at, float)
    ar = np.asarray(ar, float)
    runs = V._segment_by_crossing(at, ar, hr_t, hr_v, PCL.MIN_SUBSEG_SEC,
                                  PCL.RR_EDGES, CM.HR_EDGES, fill=True)
    idx_of, rb_of = PCL._run_indexer(runs)
    ridx = idx_of(pt)
    cat = np.where(ridx >= 0, rb_of[np.clip(ridx, 0, len(runs) - 1)], -1).astype(float)
    return dict(rid=rid, ref_t=pt, ref_r=pr, ref_r_smooth=pr_smooth, cat=cat, our_t=at, our_r=ar,
                offset=offset, hr_t=np.asarray(hr_t, float), hr_v=np.asarray(hr_v, float))


def _merged_cells(ref_t, cat, ref_r):
    """Merge consecutive breaths that share BOTH the predicted category AND correctness into one box.
    A run of correct breaths in the same category -> a single green box; a change in category OR in
    correctness starts a new box. Returns [[t0, t1, pcat, correct(0/1)], ...] (still per-breath truth,
    so it stays consistent with the per-breath accuracy)."""
    bt = np.asarray(ref_t, float); bp = np.asarray(cat).astype(int)
    rr = np.asarray(ref_r, float)
    btc = np.where(np.isfinite(rr), np.digitize(rr, RR_EDGES), -1).astype(int)
    n = len(bt); dt = float(np.median(np.diff(bt))) if n > 1 else 1.0
    cells = []; i = 0
    while i < n:
        if bp[i] < 0:
            i += 1; continue
        ok = int(bp[i] == btc[i]); j = i
        while j < n and bp[j] == bp[i] and int(bp[j] == btc[j]) == ok:
            j += 1
        t1 = float(bt[j]) if j < n else float(bt[j - 1] + dt)    # right edge = next run's start
        cells.append([float(bt[i]), t1, int(bp[i]), ok])
        i = j
    return cells


def _overlap_err(t_ref, v_ref, t_cand, v_cand, min_overlap=20.0, red=None):
    """Errors of a candidate averaged-RR curve vs the reference averaged-RR curve, sampled at the
    reference grid times over their time overlap. Both are expected from average_series (may hold NaN).
    `red` (merged red zones) excludes any grid time inside a noise/apnea region from the error."""
    tr = np.asarray(t_ref, float); vr = np.asarray(v_ref, float)
    tc = np.asarray(t_cand, float); vc = np.asarray(v_cand, float)
    mr = np.isfinite(tr) & np.isfinite(vr); tr, vr = tr[mr], vr[mr]
    mc = np.isfinite(tc) & np.isfinite(vc); tc, vc = tc[mc], vc[mc]
    if tr.size < 2 or tc.size < 2:
        return np.zeros(0)
    lo = max(tr[0], tc[0]); hi = min(tr[-1], tc[-1])
    if hi - lo < min_overlap:
        return np.zeros(0)
    m = (tr >= lo) & (tr <= hi)
    tt = tr[m]
    e = np.interp(tt, tc, vc) - vr[m]
    if red:                                              # drop samples inside red zones (clean-only stats)
        e = e[~NZ.in_red(tt, red)]
    return e[np.isfinite(e)]


def _avg_rr_stats(err_list):
    """Pool per-recording error arrays -> MAE and ARMS (RMS of errors)."""
    e = np.concatenate(err_list) if err_list else np.zeros(0)
    if not e.size:
        return {"mae": None, "arms": None, "n": 0}
    return {"mae": float(np.mean(np.abs(e))), "arms": float(np.sqrt(np.mean(e ** 2))), "n": int(e.size)}


WITHIN_TOLS = (1.0, 2.0, 3.0, 5.0)


def _within(abs_err):
    """% of breaths whose |error| is within each tolerance band (±1/±2/±3/±5 bpm)."""
    e = np.asarray(abs_err, float)
    return {str(int(t)): (100.0 * float(np.mean(e <= t)) if e.size else None) for t in WITHIN_TOLS}


def _categorization_diag(recs):
    """Pool per-breath predicted vs reference category across recordings. MAIN truth = segment-median
    (smoothed reference, ref_r_smooth); a SECONDARY accuracy vs the raw per-breath reference is also
    reported. Confusion / per-bin / severity / leverage all use the MAIN (smoothed) truth."""
    conf = [[0] * 4 for _ in range(4)]                       # conf[true][pred]
    borderline = gross = adjacent = far = 0
    per_rec = []
    tot = ok = ok_raw = 0
    for d in recs:
        pcat = np.asarray(d["cat"]); rr = np.asarray(d["ref_r"], float)
        rs = np.asarray(d.get("ref_r_smooth", d["ref_r"]), float)   # smoothed truth (fallback raw)
        m = (pcat >= 0) & np.isfinite(rr) & np.isfinite(rs)
        pc = pcat[m].astype(int)
        tc = np.digitize(rs[m], RR_EDGES).astype(int)              # MAIN: segment-median truth
        tcr = np.digitize(rr[m], RR_EDGES).astype(int)            # secondary: raw per-breath truth
        n = pc.size; good = int(np.sum(pc == tc)); good_raw = int(np.sum(pc == tcr))
        tot += n; ok += good; ok_raw += good_raw
        rb = {"rid": d["rid"], "n": int(n), "wrong": int(n - good),
              "acc": (100.0 * good / n) if n else float("nan"),
              "acc_raw": (100.0 * good_raw / n) if n else float("nan"),
              "borderline": 0, "gross": 0, "adjacent": 0, "far": 0}
        for t, p, rv in zip(tc, pc, rs[m]):
            conf[t][p] += 1
            if t == p:
                continue
            d_edge = float(np.min(np.abs(rv - np.array(RR_EDGES))))   # bpm distance to nearest boundary
            if d_edge <= 2.0:
                borderline += 1; rb["borderline"] += 1
            else:
                gross += 1; rb["gross"] += 1
            if abs(t - p) == 1:
                adjacent += 1; rb["adjacent"] += 1
            else:
                far += 1; rb["far"] += 1
        per_rec.append(rb)
    # per-bin recall/precision
    perbin = []
    for b in range(4):
        row = sum(conf[b]); col = sum(conf[r][b] for r in range(4))
        perbin.append({"label": RR_LABELS[b], "n": row,
                       "recall": (100.0 * conf[b][b] / row) if row else float("nan"),
                       "precision": (100.0 * conf[b][b] / col) if col else float("nan")})
    # leverage: overall-accuracy points recoverable if this recording were perfect
    for p in per_rec:
        acc_frac = (p["acc"] / 100.0) if p["acc"] == p["acc"] else 0.0
        wrong = p["n"] * (1.0 - acc_frac)
        p["leverage"] = (100.0 * wrong / tot) if tot else 0.0
    per_rec.sort(key=lambda r: -r["leverage"])
    return {"confusion": conf, "perbin": perbin, "acc": (100.0 * ok / tot) if tot else float("nan"),
            "acc_raw": (100.0 * ok_raw / tot) if tot else float("nan"),
            "n": tot, "severity": {"borderline": borderline, "gross": gross,
                                   "adjacent": adjacent, "far": far}, "per_rec": per_rec}


def _report_html(payload):
    """Self-contained interactive report: per-recording division plot + all error diagnostics."""
    data = json.dumps(payload).replace("</", "<\\/")
    return r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BBBRR report</title>
<style>
 body{margin:0;background:#0b1020;color:#e6edf6;font:14px/1.5 system-ui,Segoe UI,Arial}
 .wrap{max-width:1160px;margin:0 auto;padding:18px}
 h1{font-size:19px;margin:0 0 2px} h2{font-size:15px;margin:22px 0 8px;color:#9fb4d0}
 .sub{color:#8ea2bd;font-size:12px}
 .cards{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}
 .card{background:#121a30;border:1px solid #223154;border-radius:10px;padding:10px 14px;min-width:120px}
 .card .k{font-size:11px;color:#8ea2bd} .card .v{font-size:20px;font-weight:600}
 select{background:#121a30;color:#e6edf6;border:1px solid #2a3c66;border-radius:7px;padding:6px 10px;font-size:14px}
 canvas{width:100%;height:360px;background:#070b16;border:1px solid #223154;border-radius:10px;margin-top:8px}
 table{border-collapse:collapse;font-size:13px;margin-top:6px} td,th{border:1px solid #223154;padding:5px 9px;text-align:center}
 th{background:#121a30;color:#9fb4d0;font-weight:500} .rid{text-align:left}
 .bar{height:13px;background:#2563eb;border-radius:3px;display:inline-block;vertical-align:middle}
 .legend span{display:inline-block;margin-right:14px;font-size:12px}
 .sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:4px}
 .flex{display:flex;flex-wrap:wrap;gap:30px;align-items:flex-start}
 .section{border:1px solid #223154;border-radius:12px;padding:4px 16px 16px;margin-top:22px}
 .shead{font-size:15px;font-weight:600;color:#cfe0f5;margin:12px 0 4px;padding-bottom:6px;border-bottom:1px solid #2a3c66}
 .scap{color:#8ea2bd;font-size:12px;margin-bottom:10px}
 h3{font-size:13px;margin:16px 0 4px;color:#9fb4d0;font-weight:500}
 .metrics{max-width:380px;margin:10px 0 4px}
 .metrics .row{display:flex;justify-content:space-between;padding:5px 2px;border-bottom:1px dashed #223154}
 .metrics .row .mk{color:#b9c8df} .metrics .row .mv{font-weight:600;font-variant-numeric:tabular-nums}
</style></head><body><div class="wrap">
<h1>BBBRR — results &amp; error report</h1>
<div class="sub" id="meta"></div>

<h2>Per-recording division vs reference</h2>
<div><select id="sel"></select> <span class="sub" id="recmeta"></span></div>
<div class="legend" style="margin-top:6px">
 <span><span class="sw" style="background:#111;border:1px solid #888"></span>reference BBBRR</span>
 <span><span class="sw" style="background:#e879f9"></span>fused BBBRR</span>
 <span><span class="sw" style="background:#38bdf8"></span>our division curve</span>
 <span><span class="sw" style="border:2px solid #16a34a"></span>category match</span>
 <span><span class="sw" style="border:2px solid #ef4444"></span>category miss</span>
</div>
<canvas id="cv"></canvas>

<div class="section">
 <div class="shead">1 &middot; Categorization errors &mdash; which bin, and how wrong</div>
 <div class="scap">truth = segment-median of the reference RR (smoothed); the raw per-breath accuracy is shown beneath it. confusion / severity / leverage use the segment-median truth.</div>
 <div class="metrics" id="catmetrics"></div>
 <div class="flex">
  <div><h3>Confusion matrix</h3><div id="conf"></div>
    <div class="sub">rows = reference category, columns = our category</div></div>
  <div><h3>Per-bin recall / precision</h3><div id="perbin"></div></div>
 </div>
 <h3>Per-recording ranking (leverage)</h3><div id="leverage"></div>
 <div class="sub">leverage = overall accuracy points recoverable if this recording were perfect</div>
</div>

<div class="section">
 <div class="shead">2 &middot; BBBRR errors &mdash; per-breath respiration rate</div>
 <div class="scap">all metrics below are on the final fused BBBRR (bpm) sampled at each reference breath</div>
 <div class="metrics" id="bbmetrics"></div>
 <div class="flex">
  <div><h3>Per-category breath error</h3><div id="percat"></div></div>
  <div><h3>Feature importance (strong &rarr; weak)</h3><div id="feat"></div></div>
 </div>
 <h3>BBBRR error by category confusion</h3><div id="attrib"></div>
 <div class="sub">total BBBRR error (bpm) attributed to each true&rarr;our-category cell; off-diagonal = error caused by misclassification. cell shows total / mean (n).</div>
</div>

<script>
const D=""" + data + r""";
const LAB=""" + json.dumps(list(RR_LABELS)) + r""", COL=""" + json.dumps(CAT_COLORS) + r""";
const EDG=""" + json.dumps(RR_EDGES) + r""", BAND=""" + json.dumps([0.0] + list(RR_EDGES) + [55.0]) + r""", YMAX=55;
const f2=x=>(x==null||x!==x)?"&mdash;":x.toFixed(2);
const f1=x=>(x==null||x!==x)?"&mdash;":x.toFixed(1);

document.getElementById("meta").innerHTML =
  "route <b>"+D.route+"</b> &middot; model <b>"+D.model+"</b> &middot; stage <b>"+D.stage+"</b> &middot; "+D.n_recs+" recordings"
  +(D.noise?(" &middot; <span style='color:#f0a0a0'>noise excluded <b>"+D.noise.pct.toFixed(1)+"%</b> ("
    +D.noise.noise_sec.toFixed(0)+"s of "+D.noise.span_sec.toFixed(0)+"s"
    +(D.noise.by_type?"; "+Object.keys(D.noise.by_type).sort().map(k=>k+" "+D.noise.by_type[k].toFixed(0)+"s").join(", "):"")
    +"), errors on clean only</span>"):"");
function rows(id,arr){document.getElementById(id).innerHTML = arr.map(c=>
  "<div class='row'><span class='mk'>"+c[0]+"</span><span class='mv'>"+c[1]+"</span></div>").join("");}
const o=D.overall, s=D.cat.severity;
const correct=[0,1,2,3].reduce((a,i)=>a+D.cat.confusion[i][i],0);
rows("catmetrics",[
  ["category accuracy (vs segment-median)", f1(D.cat.acc)+"%"],
  ["&nbsp;&nbsp;vs raw per-breath", f1(D.cat.acc_raw)+"%"],
  ["breaths evaluated", D.cat.n],
  ["misclassified breaths", (D.cat.n-correct)],
  ["&nbsp;&nbsp;borderline (&le;2 bpm from edge)", s.borderline],
  ["&nbsp;&nbsp;gross (&gt;2 bpm from edge)", s.gross],
  ["&nbsp;&nbsp;adjacent bin (&plusmn;1)", s.adjacent],
  ["&nbsp;&nbsp;far (&ge;2 bins apart)", s.far]]);
rows("bbmetrics",[
  ["per-breath MAE", f2(o.pb_mae)+" bpm"],
  ["per-breath RMSE", f2(o.pb_rmsd)+" bpm"],
  ["within &plusmn;1 brpm", o.within?f1(o.within["1"])+"%":"&mdash;"],
  ["within &plusmn;2 brpm", o.within?f1(o.within["2"])+"%":"&mdash;"],
  ["within &plusmn;3 brpm", o.within?f1(o.within["3"])+"%":"&mdash;"],
  ["within &plusmn;5 brpm", o.within?f1(o.within["5"])+"%":"&mdash;"],
  ["grid MAE", f2(o.grid_mae)+" bpm"],
  ["grid RMSE", f2(o.grid_rmsd)+" bpm"],
  ["baseline MAE (median of estimators)", f2(o.base_mae)+" bpm"],
  ["&mdash; average RR (windowed) vs reference average &mdash;", ""],
  ["average-RR MAE &mdash; fused BBBRR", f2(o.avgrr_bb_mae)+" bpm"],
  ["average-RR ARMS &mdash; fused BBBRR", f2(o.avgrr_bb_arms)+" bpm"],
  ["average-RR MAE &mdash; division curve (baseline)", f2(o.avgrr_rg_mae)+" bpm"],
  ["average-RR ARMS &mdash; division curve (baseline)", f2(o.avgrr_rg_arms)+" bpm"]]);

// ---- recording selector + canvas ----
const sel=document.getElementById("sel");
D.recordings.forEach((r,i)=>{const op=document.createElement("option");op.value=i;op.textContent=r.rid+"  ("+f1(r.cat_acc)+"%)";sel.appendChild(op);});
const cv=document.getElementById("cv"), cx=cv.getContext("2d");
function draw(i){
  const r=D.recordings[i], dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=360;
  cv.width=W*dpr; cv.height=H*dpr; cx.setTransform(dpr,0,0,dpr,0,0); cx.clearRect(0,0,W,H);
  const m={l:46,r:14,t:10,b:26}, x0=m.l,x1=W-m.r,y0=H-m.b,y1=m.t;
  let ts=r.ref_t.concat(r.fused_t||[],r.our_t||[]); const t0=Math.min(...ts),t1=Math.max(...ts);
  const TX=v=>x0+(x1-x0)*(v-t0)/((t1-t0)||1), BY=v=>y0+(y1-y0)*(v/YMAX);
  // red zones (hand-annotated noise/apnea): shaded, excluded from analysis
  (r.red||[]).forEach(z=>{const a=TX(z[0]),b=TX(z[1]),w=Math.max(2,b-a);
    cx.fillStyle="rgba(130,140,160,.30)";cx.fillRect(a,y1,w,y0-y1);
    cx.strokeStyle="rgba(230,120,120,.55)";cx.setLineDash([3,3]);cx.strokeRect(a,y1,w,y0-y1);cx.setLineDash([]);});
  // clip an interval [a,b] to the parts NOT inside any red zone (so nothing is drawn there)
  function clipOut(a,b){let segs=[[a,b]];(r.red||[]).forEach(z=>{const out=[];segs.forEach(([s,e])=>{
    if(z[1]<=s||z[0]>=e)out.push([s,e]);else{if(z[0]>s)out.push([s,z[0]]);if(z[1]<e)out.push([z[1],e]);}});segs=out;});return segs;}
  // merged category boxes: a run of same category + same correctness = one box.
  // fill = our predicted category; border green=correct / red=wrong. Boxes are clipped out of red zones.
  (r.cells||[]).forEach(c=>{const[a,b,pc,ok]=c; const yA=BY(BAND[pc+1]), h=BY(BAND[pc])-BY(BAND[pc+1]);
    clipOut(a,b).forEach(([s,e])=>{const w=Math.max(2,TX(e)-TX(s));
      cx.fillStyle=COL[pc]+"33";cx.fillRect(TX(s),yA,w,h);
      cx.strokeStyle=ok?"#16a34a":"#ef4444";cx.lineWidth=1.4;cx.strokeRect(TX(s),yA,w,h);});});
  // grid + boundary lines
  cx.strokeStyle="#1e2a44";cx.fillStyle="#8ea2bd";cx.font="11px system-ui";
  EDG.forEach(e=>{cx.beginPath();cx.setLineDash([4,4]);cx.moveTo(x0,BY(e));cx.lineTo(x1,BY(e));cx.stroke();cx.setLineDash([]);cx.fillText(e,6,BY(e)+3);});
  const span=t1-t0,step=span>500?120:(span>250?60:30);
  for(let tt=Math.ceil(t0/step)*step;tt<=t1;tt+=step)cx.fillText(tt+"s",TX(tt)-8,y0+16);
  function segRed(a,b){return (r.red||[]).some(z=>z[0]<b&&z[1]>a);}   // does [a,b] cross a red zone?
  function line(T,Rv,color,w,dash){if(!T||!T.length)return;cx.strokeStyle=color;cx.lineWidth=w;cx.setLineDash(dash||[]);cx.beginPath();let pen=false,pt=null;
    for(let k=0;k<T.length;k++){const v=Rv[k];if(v==null||v!==v){pen=false;pt=null;continue;}const X=TX(T[k]),Y=BY(v);
      if(!pen||(pt!=null&&segRed(pt,T[k]))){cx.moveTo(X,Y);pen=true;}else cx.lineTo(X,Y);pt=T[k];}cx.stroke();cx.setLineDash([]);}
  line(r.our_t,r.our_r,"#38bdf8",1.2,[5,4]);
  line(r.ref_t,r.ref_r,"#f8fafc",1.8);
  line(r.fused_t,r.fused_r,"#e879f9",1.6);
  document.getElementById("recmeta").innerHTML="per-breath MAE "+f2(r.pb_mae)+" &middot; RMSE "+f2(r.pb_rmsd)+" &middot; cat "+f1(r.cat_acc)+"%  (n="+r.n+")";
}
sel.onchange=()=>draw(+sel.value); window.onresize=()=>draw(+sel.value); draw(0);

// ---- confusion matrix ----
let ch="<table><tr><th></th>"+LAB.map((l,j)=>"<th>"+l+"<div class='sub'>pred</div></th>").join("")+"</tr>";
const C=D.cat.confusion, mx=Math.max(1,...C.flat());
for(let t=0;t<4;t++){ch+="<tr><th>"+LAB[t]+"</th>";for(let p=0;p<4;p++){const v=C[t][p];
  const bg=(t===p)?"rgba(34,197,94,":"rgba(239,68,68,";const a=v?(0.12+0.5*v/mx):0;
  ch+="<td style='background:"+bg+a+")'>"+v+"</td>";}ch+="</tr>";}
document.getElementById("conf").innerHTML=ch+"</table>";

// ---- per-bin ----
let pb="<table><tr><th>cat</th><th>n</th><th>recall</th><th>precision</th></tr>";
D.cat.perbin.forEach(b=>pb+="<tr><td>"+b.label+"</td><td>"+b.n+"</td><td>"+f1(b.recall)+"%</td><td>"+f1(b.precision)+"%</td></tr>");
document.getElementById("perbin").innerHTML=pb+"</table>";

// ---- per-category breath error ----
if(D.per_cat_breath&&D.per_cat_breath.length){let pc="<table><tr><th>cat</th><th>n</th><th>MAE</th><th>RMSE</th><th>&plusmn;2</th><th>&plusmn;3</th></tr>";
 D.per_cat_breath.forEach(c=>pc+="<tr><td>"+c.label+"</td><td>"+c.n+"</td><td>"+f2(c.mae)+"</td><td>"+f2(c.rmsd)+"</td><td>"+(c.within?f1(c.within["2"])+"%":"&mdash;")+"</td><td>"+(c.within?f1(c.within["3"])+"%":"&mdash;")+"</td></tr>");
 document.getElementById("percat").innerHTML=pc+"</table>";}
else document.getElementById("percat").innerHTML="<div class='sub'>run --stage full for breath-level BBBRR errors</div>";

// ---- BBBRR error by category confusion (true row x our-pred col); total error attributed per cell ----
if(D.attribution){const AT=D.attribution.total, AN=D.attribution.count;
 let amx=0;for(let t=0;t<4;t++)for(let p=0;p<4;p++)if(t!==p)amx=Math.max(amx,AT[t][p]);
 let ah="<table><tr><th>true &darr; / pred &rarr;</th>"+LAB.map(l=>"<th>"+l+"</th>").join("")+"</tr>";
 for(let t=0;t<4;t++){ah+="<tr><th>"+LAB[t]+"</th>";for(let p=0;p<4;p++){const tot=AT[t][p],n=AN[t][p];
   const diag=(t===p); const a=(!diag&&amx>0)?(0.12+0.55*tot/amx):0;
   const cell=n?("<b>"+tot.toFixed(0)+"</b><br><span class='mut' style='font-size:11px'>"+(tot/n).toFixed(1)+" (n="+n+")</span>"):"&mdash;";
   ah+="<td style='background:rgba(239,68,68,"+(diag?0:a)+")'>"+cell+"</td>";}ah+="</tr>";}
 ah+="</table>";
 const off=[];for(let t=0;t<4;t++)for(let p=0;p<4;p++)if(t!==p&&AN[t][p])off.push([t,p,AT[t][p],AN[t][p]]);
 off.sort((a,b)=>b[2]-a[2]);
 if(off.length)ah+="<div class='sub' style='margin-top:6px'>worst confusion: <b>"+LAB[off[0][0]]+" &rarr; "+LAB[off[0][1]]+"</b> caused "+off[0][2].toFixed(0)+" bpm total BBBRR error (mean "+(off[0][2]/off[0][3]).toFixed(1)+", n="+off[0][3]+")</div>";
 document.getElementById("attrib").innerHTML=ah;}
else document.getElementById("attrib").innerHTML="<div class='sub'>run --stage full</div>";

// ---- leverage + per-recording error spread (concentration) ----
const PR=D.cat.per_rec;
function share(key){const tot=PR.reduce((a,p)=>a+(p[key]||0),0);if(!tot)return null;
  const top3=[...PR].sort((a,b)=>(b[key]||0)-(a[key]||0)).slice(0,3);
  return {tot:tot,pct:100*top3.reduce((a,p)=>a+(p[key]||0),0)/tot,rids:top3.filter(p=>p[key]>0).map(p=>p.rid)};}
const sg=share("gross"), sf=share("far");
let lv="<table><tr><th>recording</th><th>n</th><th>wrong</th><th>gross</th><th>far</th><th>cat acc</th><th>leverage</th></tr>";
PR.forEach(p=>lv+="<tr><td class='rid'>"+p.rid+"</td><td>"+p.n+"</td><td>"+p.wrong+"</td><td>"+p.gross+"</td><td>"+p.far+"</td><td>"+f1(p.acc)+"%</td><td>"+f1(p.leverage)+" pts</td></tr>");
lv+="</table>";
if(sg)lv+="<div class='sub' style='margin-top:6px'>gross errors ("+sg.tot+"): top-3 recordings hold "+sg.pct.toFixed(0)+"% &nbsp;["+sg.rids.join(", ")+"]</div>";
if(sf)lv+="<div class='sub'>far errors ("+sf.tot+"): top-3 recordings hold "+sf.pct.toFixed(0)+"% &nbsp;["+sf.rids.join(", ")+"]</div>";
document.getElementById("leverage").innerHTML=lv;

// ---- feature importance ----
if(D.features&&D.features.length){const mxf=Math.max(...D.features.map(f=>f.imp));
 let ff="<table>";D.features.forEach(f=>ff+="<tr><td class='rid'>"+f.name+"</td><td style='text-align:left'><span class='bar' style='width:"+(Math.max(1,240*f.imp/mxf))+"px'></span> "+f.imp.toFixed(3)+"</td></tr>");
 document.getElementById("feat").innerHTML=ff+"</table>";}
else document.getElementById("feat").innerHTML="<div class='sub'>run --stage full for feature importance</div>";
</script></div></body></html>"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fuse per-category BBBRR (LOSO), per-breath scored")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--recs", nargs="+", default=None,
                    help="recording ids; default = ALL Exp3 recordings discover finds (new ones auto-included)")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="recording ids to leave OUT (e.g. --exclude Exp3/011); applied after --recs")
    ap.add_argument("--channels", nargs="+", default=list(ALL_CH),
                    help="channels to include (default all 5); e.g. --channels Green Red IR Yellow")
    ap.add_argument("--model", choices=("hgb", "rf", "enet", "robust", "moe"), default="hgb",
                    help="fusion model: hgb/rf = trees; enet = Elastic-Net (regularized linear); "
                         "robust = no-train Tukey-biweight mean of estimators; moe = per-category Ridge experts")
    ap.add_argument("--route", choices=("ref", "ours"), default="ref",
                    help="category split: ref = reference RR (oracle); ours = Artifact-BW spectrogram ridge")
    ap.add_argument("--rows", choices=("breaths", "grid"), default="breaths",
                    help="breaths = one row per reference breath, per-graph aligned nearest-match features "
                         "(default); grid = 1 s grid with interpolated features")
    ap.add_argument("--stage", choices=("full", "cat"), default="full",
                    help="full = whole pipeline incl. BBBRR fusion model (slow); "
                         "cat = category split only, no model (fast iteration)")
    ap.add_argument("--method", choices=("ssp", "spline"), default=None,
                    help="per-beat envelope method for RSA/AMP/AUC extraction (default: ssp). "
                         "NOTE: per-category overrides are ssp-tuned; spline uses its own defaults.")
    ap.add_argument("--cat", choices=("artifact", "ml"), default="artifact",
                    help="route=ours categorization ridge: 'artifact' = Artifact BW only (default); "
                         "'ml' = LOSO multi-channel ML ridge, computed in-process (one command, no pre-step).")
    ap.add_argument("--cat-ridge", default=None,
                    help="advanced: pickle {rid:(t_watch, rr)} of a precomputed categorization ridge, used "
                         "instead of --cat. route=ours uses it in place of the Artifact-only ridge.")
    ap.add_argument("--noise-dir", default=None,
                    help="directory with manual annotation JSONs (annotations_<rec>.json from "
                         "noise_annotator.py). Those time regions are EXCLUDED everywhere (categorization, "
                         "BBBRR, average, training) and never interpolated across; errors are computed on "
                         "clean spans only and the report shows what %% of time was noise.")
    ap.add_argument("--out", default=None, help="HTML report path (default: scratchpad/fusion_report_<route>.html)")
    ap.add_argument("--no-html", action="store_true", help="skip the HTML report")
    ap.add_argument("--show", action="store_true", help="also open the old blocking matplotlib windows")
    ap.add_argument("--drop-feats", nargs="+", default=None,
                    help="feature columns to EXCLUDE from the model. Channel:param, e.g. "
                         "Red:RSA IR:AUC Green:LP; OR context features: "
                         "category HR prev_rr spread n_present HR/2 nyq_margin_bw above_nyq_bw nyq_margin_cons. "
                         "e.g. --drop-feats category   (run the fusion without the category feature)")
    ap.add_argument("--agree3", action="store_true",
                    help="add a per-channel <ch>:agree3 feature = consensus breaths across its 3 "
                         "params (>=2 of RSA/AMP/AUC agree within 0.5 s, median time)")
    ap.add_argument("--gy", action="store_true",
                    help="add GY:<base> features = Green+Yellow signal-level average (z-scored) of "
                         "each per-beat param, then per-category detection")
    ap.add_argument("--cache", default=None,
                    help="pickle path for the built per-recording data: build once (with the flags "
                         "that make the SUPERSET of features), then reuse across runs that only vary "
                         "--drop-feats. Loading ignores --channels/--agree3/--gy (use --drop-feats).")
    ap.add_argument("--rr-edges", type=float, nargs="+", default=None,
                    help="category RR-bin edges in bpm (default 15 22 30 -> <15/15-22/22-30/30+); "
                         "e.g. --rr-edges 12 20 30. Applied to BOTH this report and the per_category_lab pipeline.")
    ap.add_argument("--merge-mid", choices=("low", "high"), default=None,
                    help="collapse the two middle bins into one (4 categories -> 3, e.g. <15/15-30/30+). "
                         "low/high = which mid detection profile the merged middle inherits (matters only "
                         "for --stage full; in --stage cat both give the same 3-way split). After --rr-edges.")
    args = ap.parse_args(argv)

    cat_ridge = None
    if args.cat_ridge:                                          # precomputed categorization ridge (ML, LOSO)
        import pickle
        with open(args.cat_ridge, "rb") as fh:
            cat_ridge = pickle.load(fh)
        print(f"  loaded cat-ridge for {len(cat_ridge)} recordings from {args.cat_ridge}")

    global METHOD
    if args.method and args.method != METHOD:                   # per-beat extraction method override
        METHOD = args.method
        print(f"  per-beat method -> {METHOD}")

    if args.rr_edges or args.merge_mid:                         # manual category boundaries
        global RR_EDGES, RR_LABELS
        if args.rr_edges:
            PCL.RR_EDGES = sorted(float(x) for x in args.rr_edges)
        if args.merge_mid:
            PCL.merge_middle_bins(args.merge_mid)               # merges PCL.RR_EDGES + remaps PCL overrides
        RR_EDGES = list(PCL.RR_EDGES)                           # keep report truth-binning in sync with the pipeline
        RR_LABELS = _edges_to_labels(RR_EDGES)
        print(f"  RR edges -> {RR_EDGES}  ({' / '.join(RR_LABELS)})"
              + (f"  [merge-mid={args.merge_mid}]" if args.merge_mid else ""))

    import tempfile
    # OneDrive project dir can't be written by Python -> default to a stable, always-writable temp dir
    out_html = args.out or os.path.join(tempfile.gettempdir(), "fusion_report_%s.html" % args.route)
    out_html = os.path.abspath(out_html)

    # ---- fast path: category split only, no feature building, no model ----
    if args.stage == "cat":
        disc = S.discover(args.data_root)
        if args.recs:
            found = {r[0]: r for r in disc}; recs = [found[r] for r in args.recs if r in found]
        else:                                                    # default: Exp3, excluding the Test folder
            recs = sorted((r for r in disc if r[0].startswith("Exp3/") and not _under_test(r)), key=lambda r: r[0])
        if args.exclude:
            recs = [r for r in recs if r[0] not in set(args.exclude)]
        if not recs:
            sys.exit("no recordings found")
        print(f"  [stage=cat] recordings: {[r[0] for r in recs]}")
        if cat_ridge is None and args.cat == "ml" and args.route == "ours":
            import ml_categorizer
            cat_ridge = ml_categorizer.compute_loso_ridge(recs)   # train on all cols (clean-only tested worse)
        cat_recs = []; ns = None
        for rid, edf, csv in recs:
            try:
                cat_recs.append(build_categorization(rid, edf, csv, args.route, cat_ridge=cat_ridge))
            except Exception as e:
                print(f"  [ERR] {rid}: {e}")
        if args.noise_dir:                                   # stats from FULL span, then exclude red zones
            ns = _load_red(cat_recs, args.noise_dir)
            for c in cat_recs:
                _exclude_red(c, c.get("red", []))
            print("\n  " + "=" * 58)
            print(f"  NOISE COVERAGE ({ns['n_recs']} recordings)")
            print(f"    total recording time : {_fmt_dur(ns['span_sec'])}")
            print(f"    total noise  excluded: {_fmt_dur(ns['noise_sec'])}  = {ns['pct']:.1f}%")
            for k, v in sorted(ns["by_type_sec"].items()):
                print(f"       - {k:6}: {_fmt_dur(v)}")
            print(f"    clean time  analysed : {_fmt_dur(ns['span_sec'] - ns['noise_sec'])}")
            print("  " + "=" * 58)
        for c in cat_recs:                                   # categorization accuracy on clean spans
            pcat = np.asarray(c["cat"]); rs = np.asarray(c["ref_r_smooth"], float)
            mm = (pcat >= 0) & np.isfinite(rs)
            c["acc"] = (100.0 * np.mean(pcat[mm].astype(int) == np.digitize(rs[mm], RR_EDGES))) if mm.any() else float("nan")
            print(f"  built {c['rid']}: cat acc {c['acc']:.0f}%")
        diag = _categorization_diag(cat_recs)
        from respiration_rr.rr_average import average_series
        rg_errs = []
        for c in cat_recs:
            t_ref, v_ref = average_series(np.asarray(c["ref_t"], float), np.asarray(c["ref_r"], float))
            t_rg, v_rg = average_series(np.asarray(c["our_t"], float), np.asarray(c["our_r"], float))
            rg_errs.append(_overlap_err(t_ref, v_ref, t_rg, v_rg, red=c.get("red")))
        avg_rg = _avg_rr_stats(rg_errs)
        payload = {"route": args.route, "model": "-", "stage": "cat", "n_recs": len(cat_recs),
                   "overall": {"pb_mae": None, "pb_rmsd": None, "grid_mae": None, "grid_rmsd": None, "base_mae": None,
                               "avgrr_bb_mae": None, "avgrr_bb_arms": None,
                               "avgrr_rg_mae": avg_rg["mae"], "avgrr_rg_arms": avg_rg["arms"]},
                   "cat": diag, "per_cat_breath": [], "features": [],
                   "recordings": [{"rid": c["rid"], "pb_mae": None, "pb_rmsd": None, "cat_acc": c["acc"],
                                   "n": int(np.sum(np.asarray(c["cat"]) >= 0)),
                                   "ref_t": c["ref_t"].tolist(), "ref_r": c["ref_r"].tolist(),
                                   "our_t": c["our_t"].tolist(), "our_r": c["our_r"].tolist(),
                                   "fused_t": [], "fused_r": [],
                                   "red": [[float(s), float(e)] for s, e in c.get("red", [])],
                                   "cells": _merged_cells(c["ref_t"], c["cat"], c["ref_r_smooth"])}
                                  for c in cat_recs],
                   "noise": ({"pct": ns["pct"], "noise_sec": ns["noise_sec"],
                              "span_sec": ns["span_sec"], "by_type": ns["by_type_sec"]} if ns else None)}
        if not args.no_html:
            open(out_html, "w", encoding="utf-8").write(_report_html(payload))
            print(f"\n  report -> {out_html}")
        return

    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.inspection import permutation_importance
    if args.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor as Model
        mk = lambda: Model(random_state=42)
    elif args.model == "rf":
        from sklearn.ensemble import RandomForestRegressor as Model
        mk = lambda: Model(n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=42)
    elif args.model == "enet":                               # (B) regularized linear (correlated-group friendly)
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import ElasticNetCV
        mk = lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler()),
                               ("en", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1.0], n_alphas=80,
                                                   cv=5, max_iter=5000, random_state=42))])
    elif args.model == "robust":                             # (A) no-training Tukey-biweight fusion of estimators
        mk = lambda: RobustMean(n_est=len(names))
    else:                                                    # (C) moe: per-category Ridge experts
        mk = lambda: CategoryMoE(cat_idx=feat_names.index("category"), n_bins=len(RR_EDGES) + 1)

    disc = S.discover(args.data_root)
    if args.recs:
        found = {r[0]: r for r in disc}
        recs = [found[r] for r in args.recs if r in found]
    else:                                                    # default: every Exp3 recording, excluding the Test folder
        recs = sorted((r for r in disc if r[0].startswith("Exp3/") and not _under_test(r)), key=lambda r: r[0])
    if args.exclude:
        recs = [r for r in recs if r[0] not in set(args.exclude)]
    if not recs:
        sys.exit("no recordings found")
    print(f"  recordings: {[r[0] for r in recs]}")
    if cat_ridge is None and args.cat == "ml" and args.route == "ours":
        import ml_categorizer
        cat_ridge = ml_categorizer.compute_loso_ridge(recs)   # train on all cols (clean-only tested worse)

    data = []
    if args.cache and os.path.exists(args.cache):
        import pickle
        with open(args.cache, "rb") as fh:
            data = pickle.load(fh)
        print(f"  loaded {len(data)} recordings from cache {args.cache}")
    else:
        for rid, edf, csv in recs:
            try:
                d = build_recording(rid, edf, csv, args.channels, args.route, args.rows, args.agree3, args.gy,
                                    cat_ridge=cat_ridge)
                data.append(d)
                print(f"  built {rid}: {len(d['cols'])} method columns, {d['grid'].size} grid pts")
            except Exception as e:
                print(f"  [ERR] {rid}: {e}")
        if args.cache and data:
            import pickle
            with open(args.cache, "wb") as fh:
                pickle.dump(data, fh)
            print(f"  cached {len(data)} recordings -> {args.cache}")
    if not data:
        sys.exit("no recordings built")

    noise_stats = None
    if args.noise_dir:                                       # exclude hand-annotated red zones everywhere
        noise_stats = _load_red(data, args.noise_dir)
        for d in data:
            _exclude_red(d, d.get("red", []))
        print("\n  " + "=" * 58)
        print(f"  NOISE COVERAGE ({noise_stats['n_recs']} recordings)")
        print(f"    total recording time : {_fmt_dur(noise_stats['span_sec'])}")
        print(f"    total noise  excluded: {_fmt_dur(noise_stats['noise_sec'])}  = {noise_stats['pct']:.1f}%")
        for k, v in sorted(noise_stats["by_type_sec"].items()):
            print(f"       - {k:6}: {_fmt_dur(v)}")
        print(f"    clean time  analysed : {_fmt_dur(noise_stats['span_sec'] - noise_stats['noise_sec'])}")
        print("  " + "=" * 58)
        for p in noise_stats["per_rec"]:
            if p["noise_sec"] > 0:
                ty = " ".join(f"{k}={v:.0f}s" for k, v in p["types"].items())
                print(f"     {p['rid']:9} {p['pct']:4.1f}%  ({p['noise_sec']:.0f}s)  {ty}")

    names = sorted(set().union(*[d["cols"].keys() for d in data]))
    CTX_FEATS = ["HR", "category", "spread", "n_present", "prev_rr",
                 "HR/2", "nyq_margin_bw", "above_nyq_bw", "nyq_margin_cons"]  # non-channel context features
    if args.drop_feats:
        drop = set(args.drop_feats)
        missing = drop - set(names) - set(CTX_FEATS)      # context names (e.g. category) are droppable too
        if missing:
            print(f"  [warn] --drop-feats names not found (ignored): {sorted(missing)}")
        names = [n for n in names if n not in drop]       # channel:param cols (also excluded from consensus)
        print(f"  dropped feature(s): {sorted(drop - missing)}")
    print("\n  method feature columns:")
    for c in sorted(set(n.split(":")[0] for n in names)):
        print(f"     {c:9}: {', '.join(k.split(':', 1)[1] for k in names if k.startswith(c + ':'))}")
    print("     + context: HR, category, spread, n_present, prev_rr, HR/2, nyq_margin_bw, above_nyq_bw, nyq_margin_cons")

    # assemble per-recording blocks (prev_rr computed WITHIN a recording -> no cross-rec leak)
    Xparts, yparts, gparts, gridparts = [], [], [], []
    base_parts = []                                          # median-of-estimators baseline
    bw_idx = names.index("Artifact:BW_hybrid") if "Artifact:BW_hybrid" in names else None
    for d in data:
        n = d["grid"].size
        M = np.column_stack([d["cols"].get(c, np.full(n, np.nan)) for c in names])
        consensus = np.nanmedian(M, axis=1)
        spread = np.nan_to_num(np.nanstd(M, axis=1), nan=0.0)
        n_present = np.sum(~np.isnan(M), axis=1).astype(float)
        prev_rr = np.concatenate([[np.nan], consensus[:-1]])
        # ---- Nyquist / anti-alias context (option 3): per-beat params alias above HR/2 ----
        hr = np.asarray(d["hr"], float); hr_half = hr / 2.0
        r_bw = M[:, bw_idx] if bw_idx is not None else np.full(n, np.nan)   # continuous (non-aliased) rate
        nyq_bw = r_bw - hr_half                                            # >0 -> per-beat params aliasing
        above_bw = np.where(np.isfinite(r_bw) & np.isfinite(hr_half), (r_bw > hr_half).astype(float), np.nan)
        nyq_cons = consensus - hr_half
        ctx = np.column_stack([hr, d["cat"], spread, n_present, prev_rr, hr_half, nyq_bw, above_bw, nyq_cons])
        Xparts.append(np.column_stack([M, ctx]))
        yparts.append(d["y"]); gparts.append(np.array([d["rid"]] * n)); gridparts.append(d["grid"])
        base_parts.append(consensus)
    X = np.vstack(Xparts); y = np.concatenate(yparts)
    groups = np.concatenate(gparts); grid = np.concatenate(gridparts)
    baseline = np.concatenate(base_parts)
    feat_names = list(names) + CTX_FEATS
    if args.drop_feats:                                      # also drop context features (e.g. category)
        keepcols = [i for i, fn in enumerate(feat_names) if fn not in drop]
        if len(keepcols) != len(feat_names):
            X = X[:, keepcols]; feat_names = [feat_names[i] for i in keepcols]
    if args.model == "moe" and "category" not in feat_names:
        sys.exit("--model moe needs the 'category' feature (don't drop it).")

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
    _w = _within(pb_err)
    print("  within  " + "  ".join(f"±{t}:{_w[str(int(t))]:.0f}%" for t in WITHIN_TOLS))
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
    pi = permutation_importance(full, X, y, n_repeats=8, random_state=42, n_jobs=1)  # n_jobs=1: loky BrokenProcessPool on Windows/OneDrive
    order = np.argsort(pi.importances_mean)[::-1]
    print("\n  feature importance (permutation, all-data — TREND ONLY, small N):")
    for j in order[:15]:
        print(f"     {feat_names[j]:20} {pi.importances_mean[j]:.3f}")

    print("\n  NOTE: few recordings -> treat all numbers as TREND, not absolute. 'category' is "
          "reference-derived (oracle); a real deployment needs an estimated category.")

    # ---- categorization diagnostics (predicted vs reference category, per breath) ----
    diag = _categorization_diag([{"rid": d["rid"], "cat": d["cat"], "ref_r": d["ref_r"],
                                   "ref_r_smooth": d["ref_r_smooth"]} for d in data])

    # ---- per-category breath error (MAE + RMSE) ----
    per_cat_breath = []
    for b, lab in enumerate(RR_LABELS):
        mcat = pb_cat == b
        if mcat.any():
            e = pb_err[mcat]
            per_cat_breath.append({"label": lab, "n": int(mcat.sum()),
                                   "mae": float(np.mean(e)), "rmsd": float(np.sqrt(np.mean(e ** 2))),
                                   "within": _within(e)})

    # ---- per-recording fused curve + metrics for the plot ----
    from respiration_rr.rr_average import average_series
    rec_payload = []
    acc_by_rid = {p["rid"]: p["acc"] for p in diag["per_rec"]}
    avg_bb_errs, avg_rg_errs = [], []            # average-RR errors (vs reference average curve)
    attr_tot = np.zeros((4, 4)); attr_cnt = np.zeros((4, 4), int)   # BBBRR error by (true,pred) category
    for d in data:
        te = groups == d["rid"]
        gt, prd = grid[te], preds[te]; o = np.argsort(gt)
        tb, rb = np.asarray(d["ref_t"], float), np.asarray(d["ref_r"], float)
        rs = np.asarray(d.get("ref_r_smooth", d["ref_r"]), float)      # smoothed (main) category truth
        inb = (tb >= gt[o][0]) & (tb <= gt[o][-1]) if gt.size else np.zeros(tb.shape, bool)
        pr_at = np.interp(tb[inb], gt[o], prd[o]) if inb.any() else np.zeros(0)
        err = np.abs(pr_at - rb[inb]) if inb.any() else np.zeros(0)
        # attribute each breath's BBBRR error to its (true category, our predicted category) cell
        if err.size:
            tc = np.digitize(rs[inb], RR_EDGES).astype(int)
            pc = np.asarray(d["cat"])[inb].astype(int)
            for ti, pj, e in zip(tc, pc, err):
                if pj < 0:
                    continue
                attr_tot[ti][pj] += e; attr_cnt[ti][pj] += 1
        # average-RR: window BOTH our estimate and the reference the SAME way (average_series), compare
        t_ref, v_ref = average_series(tb, rb)                         # reference average-RR curve
        t_bb, v_bb = average_series(gt[o], prd[o])                    # fused BBBRR, averaged
        t_rg, v_rg = average_series(np.asarray(d["our_t"], float), np.asarray(d["our_r"], float))  # division curve
        e_bb = _overlap_err(t_ref, v_ref, t_bb, v_bb, red=d.get("red")); avg_bb_errs.append(e_bb)
        e_rg = _overlap_err(t_ref, v_ref, t_rg, v_rg, red=d.get("red")); avg_rg_errs.append(e_rg)
        rec_payload.append({
            "rid": d["rid"], "n": int(np.sum(np.asarray(d["cat"]) >= 0)),
            "pb_mae": float(np.mean(err)) if err.size else None,
            "pb_rmsd": float(np.sqrt(np.mean(err ** 2))) if err.size else None,
            "avg_bb_mae": float(np.mean(np.abs(e_bb))) if e_bb.size else None,
            "avg_rg_mae": float(np.mean(np.abs(e_rg))) if e_rg.size else None,
            "cat_acc": acc_by_rid.get(d["rid"], float("nan")),
            "ref_t": tb.tolist(), "ref_r": rb.tolist(),
            "fused_t": gt[o].tolist(), "fused_r": prd[o].tolist(),
            "our_t": d["our_t"].tolist(), "our_r": d["our_r"].tolist(),
            "red": [[float(s), float(e)] for s, e in d.get("red", [])],
            "cells": _merged_cells(tb, d["cat"], rs)})
    avg_bb = _avg_rr_stats(avg_bb_errs); avg_rg = _avg_rr_stats(avg_rg_errs)
    print("\n  average-RR (windowed) vs reference average curve:")
    print(f"     fused BBBRR     MAE={avg_bb['mae']}  ARMS={avg_bb['arms']}")
    print(f"     division curve  MAE={avg_rg['mae']}  ARMS={avg_rg['arms']}")

    # which category-confusion caused the most BBBRR error? (total error attributed to each off-diagonal)
    attribution = {"total": attr_tot.tolist(), "count": attr_cnt.tolist()}
    off = [(t, p, attr_tot[t][p], attr_cnt[t][p]) for t in range(4) for p in range(4) if t != p and attr_cnt[t][p]]
    off.sort(key=lambda z: -z[2])
    print("\n  BBBRR error by category confusion (true -> our pred), worst first:")
    for t, p, tot, n in off[:5]:
        print(f"     {RR_LABELS[t]:6} -> {RR_LABELS[p]:6}  total={tot:6.1f} bpm  mean={tot/n:4.1f}  (n={n})")

    features = [{"name": feat_names[j], "imp": float(pi.importances_mean[j])} for j in order]
    payload = {"route": args.route, "model": args.model, "stage": "full", "n_recs": len(data),
               "overall": {"pb_mae": float(np.mean(pb_err)) if pb_err.size else None, "pb_rmsd": pb_rmsd,
                           "within": _within(pb_err),
                           "grid_mae": mae(preds, y), "grid_rmsd": rmsd(preds, y),
                           "base_mae": mae(baseline, y),
                           "avgrr_bb_mae": avg_bb["mae"], "avgrr_bb_arms": avg_bb["arms"],
                           "avgrr_rg_mae": avg_rg["mae"], "avgrr_rg_arms": avg_rg["arms"]},
               "cat": diag, "per_cat_breath": per_cat_breath, "features": features,
               "attribution": attribution, "recordings": rec_payload,
               "noise": ({"pct": noise_stats["pct"], "noise_sec": noise_stats["noise_sec"],
                          "span_sec": noise_stats["span_sec"], "by_type": noise_stats["by_type_sec"]}
                         if noise_stats else None)}
    if not args.no_html:
        open(out_html, "w", encoding="utf-8").write(_report_html(payload))
        print(f"\n  report -> {out_html}")

    # ---- optional: old blocking matplotlib windows ----
    if args.show:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        for d in data:
            te = groups == d["rid"]
            if not te.any():
                continue
            gt, prd = grid[te], preds[te]; o = np.argsort(gt)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(d["ref_t"], d["ref_r"], "-", color="#0f172a", lw=1.8, label="reference BBBRR")
            ax.plot(gt[o], prd[o], "-", color="#dc2626", lw=1.3, alpha=0.9, label="fused BBBRR")
            ax.set_title(f"{d['rid']} — fused ({args.model}) vs reference", fontsize=11)
            ax.set_xlabel("time (s, REMbo clock)"); ax.set_ylabel("RR (bpm)")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.2); fig.tight_layout()
        top = order[:20][::-1]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.barh(range(len(top)), pi.importances_mean[top], color="#2563eb")
        ax.set_yticks(range(len(top))); ax.set_yticklabels([feat_names[j] for j in top], fontsize=8)
        ax.set_title("Permutation importance (all-data, trend only)", fontsize=11)
        ax.set_xlabel("importance"); ax.grid(True, axis="x", alpha=0.2); fig.tight_layout()
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
