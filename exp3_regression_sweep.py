#!/usr/bin/env python
"""exp3_regression_sweep.py — regression-then-bin RR categorization on Exp3 (watch-13)
recordings only. Sweeps (segment length x STFT window) jointly, LOSO by recording.

Features per segment: ridge RR of every channel x {raw, BW, RSA, AMP, AUC}, prominence
of each ridge, HR, and consensus spread/agreement. Target: reference segment-median RR
(REMbo clock). Model: HistGradientBoostingRegressor (NaN-aware) by default, or
RandomForest (MODEL=rf, with median imputation). Trained on MAE, scored on category
accuracy after binning [15,22,30]. Reports accuracy overall + per bin (>30 emphasis),
MAE, baseline (median of estimators), and feature importance for the best cell.

    py exp3_regression_sweep.py            # HistGBR
    MODEL=rf py exp3_regression_sweep.py   # RandomForest (bagging; wins on this small data)
"""
import os, sys, numpy as np
sys.path.insert(0, r"C:\Users\RachelMizrahi\OneDrive - CardiacSense\Documents\GitHub\BBBRR")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import bw_bp_sweep as S
from param_category_sweep import stft_power, ridge_rr, envelope_on_grid, GRID_FS

CHANNELS = ("Green", "Red", "IR", "Yellow", "Artifact")
PARAMS = (("RSA", "RSA"), ("AMP", "AMP"), ("AUC", "AUC"))
LO, HI = 0.10, 0.80
EDGES = [15.0, 22.0, 30.0]; LAB = ["<15", "15-22", "22-30", ">30"]
SEGS = [10.0, 15.0, 20.0, 30.0]
WINS = [12.0, 16.0, 24.0, 32.0]
HOP = 2.5


def col_feats(sigarr, fs, win):
    t, fr, P = stft_power(sigarr, fs, win, HOP)
    if t.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    b = np.where((fr >= LO) & (fr <= HI))[0]
    Pb = P[b, :]; fb = fr[b]
    pk = Pb.max(0); med = np.median(Pb, 0)
    prom = np.divide(pk, med, out=np.full_like(pk, np.nan), where=med > 0)
    ridge = fb[np.argmax(Pb, 0)] * 60.0; ridge[pk <= 0] = np.nan
    return t, ridge, prom


def build(recs_data, seg):
    """Aggregate precomputed per-column features into a segment table for this seg length.
    REF_MODE env: segmed (default)=segment-median of per-breath | basic=per-breath @ center
                  | avg20=reference_average_rr (20s windowed-median) @ center."""
    REF_MODE = os.environ.get("REF_MODE", "segmed")
    rows, y, groups = [], [], []
    keys = None
    for rid, dur, offset, pt, pr, tav, rav, hrfun, SC in recs_data:
        s = 0.0
        while s < dur:
            e = s + seg
            m = (pt >= s + offset) & (pt < e + offset)
            if REF_MODE == "avg20":
                ref = float(np.interp((s + e) / 2 + offset, tav, rav, left=np.nan, right=np.nan))
                have = np.isfinite(ref)
            elif REF_MODE == "basic":
                ref = float(np.interp((s + e) / 2 + offset, pt, pr, left=np.nan, right=np.nan))
                have = np.isfinite(ref)
            else:
                ref = float(np.median(pr[m])) if m.sum() >= 1 else np.nan
                have = m.sum() >= 1
            if have:
                feat = {}
                ridges = []
                for k in SC:  # k -> (t, ridge, prom)
                    tt, rg, pm = SC[k]
                    mm = (tt >= s) & (tt < e)
                    rv = rg[mm]; rv = rv[np.isfinite(rv)]
                    pv = pm[mm]; pv = pv[np.isfinite(pv)]
                    rr = float(np.median(rv)) if rv.size else np.nan
                    feat[k + "_rr"] = rr
                    feat[k + "_prom"] = float(np.median(pv)) if pv.size else np.nan
                    if np.isfinite(rr):
                        ridges.append(rr)
                ridges = np.array(ridges, float)
                feat["hr"] = hrfun(s, e)
                feat["cons_med"] = float(np.median(ridges)) if ridges.size else np.nan
                feat["cons_iqr"] = float(np.percentile(ridges, 75) - np.percentile(ridges, 25)) if ridges.size else np.nan
                feat["cons_nagree"] = int(np.sum(np.abs(ridges - feat["cons_med"]) <= 3.0)) if ridges.size else 0
                if keys is None:
                    keys = sorted(feat.keys())
                rows.append([feat.get(k, np.nan) for k in keys]); y.append(ref); groups.append(rid)
            s = e
    return np.array(rows, float), np.array(y, float), np.array(groups), keys


def main():
    from scipy.signal import butter, sosfiltfilt
    from respiration_rr.settings import PPG as ppg
    from main import run_ppg, run_reference, run_sync
    from respiration_rr.rr_average import reference_rr_points, reference_average_rr
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.inspection import permutation_importance
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    MODEL = os.environ.get("MODEL", "hgb").lower()        # hgb (default) | rf
    def make_model():
        if MODEL == "rf":                                # RF needs NaN imputation (train-fold only)
            return Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("rf", RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                                          random_state=42, n_jobs=-1))])
        return HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                             l2_regularization=1.0, random_state=42)
    print("MODEL =", MODEL)

    recs = S.discover(S.DEFAULT_DATA_ROOT)
    seen = set()
    recs = [r for r in recs if r[0].startswith("Exp3/") and "Test" not in r[1]
            and not (r[0] in seen or seen.add(r[0]))]
    print("Exp3 recordings:", [r[0] for r in recs])

    # precompute per-column features per window (reused across segment lengths)
    per_win = {w: [] for w in WINS}
    for rid, edf, csv in recs:
        ppg_res, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg_res, sig)
        pt, pr = reference_rr_points(ref); tav, rav = reference_average_rr(ref)
        fs = float(sig.fs); n = int(np.asarray(sig.channels["Artifact"]).size); dur = n / fs
        green = np.asarray(sig.channels.get("Green", np.zeros(n)), float)
        def hrfun(s, e, green=green, fs=fs):
            a = int(max(0, s * fs)); b = int(min(green.size, e * fs)); x = green[a:b]
            if x.size < int(4 * fs): return np.nan
            f = np.fft.rfftfreq(x.size, 1.0 / fs); Pw = np.abs(np.fft.rfft(x - x.mean())) ** 2
            band = (f >= 0.7) & (f <= 3.0)
            return float(f[band][np.argmax(Pw[band])] * 60.0) if band.any() else np.nan
        g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
        ny = fs / 2.0; sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
        envs = {}  # (key -> signal array, fs) for spectrogram
        for c in CHANNELS:
            if c not in sig.channels: continue
            x = np.asarray(sig.channels[c], float)
            envs["raw_" + c] = (x, fs); envs["BW_" + c] = (sosfiltfilt(sos, x), fs)
            res = ppg_res.get(c)
            for pn, pl in PARAMS:
                prm = res.params.get(pn) if res and getattr(res, "params", None) else None
                bx = np.asarray(getattr(prm, "series_x", []), float) if prm else np.zeros(0)
                by = np.asarray(getattr(prm, "series_y", []), float) if prm else np.zeros(0)
                if bx.size >= 4:
                    envs[pl + "_" + c] = (envelope_on_grid(bx, by, "spl", g_t, ppg), GRID_FS)
        for w in WINS:
            SC = {k: col_feats(a, f, w) for k, (a, f) in envs.items()}
            per_win[w].append((rid, dur, offset, pt, pr, tav, rav, hrfun, SC))
        print("  built cols:", rid)

    def cat(v): return np.digitize(v, EDGES)
    logo = LeaveOneGroupOut()
    grid_acc = np.full((len(SEGS), len(WINS)), np.nan)
    grid_mae = np.full((len(SEGS), len(WINS)), np.nan)
    best = (None, None, -1, None)
    for i, seg in enumerate(SEGS):
        for jw, w in enumerate(WINS):
            X, y, g, keys = build(per_win[w], seg)
            if len(y) < 30: continue
            pred = np.full(len(y), np.nan)
            for tr, te in logo.split(X, y, g):
                m = make_model().fit(X[tr], y[tr])
                pred[te] = m.predict(X[te])
            a = 100.0 * np.mean(cat(pred) == cat(y)); mae = np.mean(np.abs(pred - y))
            grid_acc[i, jw] = a; grid_mae[i, jw] = mae
            if a > best[2]:
                best = (seg, w, a, (X, y, g, keys, pred))
    print("\n=== category accuracy (%)  rows=segment, cols=window ===")
    print("  seg\\win " + "".join("%8.0f" % w for w in WINS))
    for i, seg in enumerate(SEGS):
        print("  %5.0fs  " % seg + "".join("%8.1f" % grid_acc[i, j] for j in range(len(WINS))))
    print("\n=== MAE (bpm) ===")
    print("  seg\\win " + "".join("%8.0f" % w for w in WINS))
    for i, seg in enumerate(SEGS):
        print("  %5.0fs  " % seg + "".join("%8.2f" % grid_mae[i, j] for j in range(len(WINS))))

    seg, w, a, (X, y, g, keys, pred) = best
    print("\n=== BEST: segment %.0fs, window %.0fs -> accuracy %.1f%% ===" % (seg, w, a))
    tc, pc = np.digitize(y, EDGES), np.digitize(pred, EDGES)
    for c in range(4):
        mm = tc == c
        print("  %-6s (n=%3d): %.1f%%" % (LAB[c], mm.sum(), 100 * np.mean(pc[mm] == tc[mm])))
    print("  MAE = %.2f bpm" % np.mean(np.abs(pred - y)))
    # baseline: median of the *_rr estimator columns
    rr_idx = [k for k, key in enumerate(keys) if key.endswith("_rr")]
    base = np.nanmedian(X[:, rr_idx], axis=1)
    print("  baseline (median of estimators): accuracy %.1f%%  MAE %.2f" %
          (100 * np.mean(np.digitize(base, EDGES) == tc), np.nanmean(np.abs(base - y))))
    # feature importance (full-fit permutation)
    reg = make_model().fit(X, y)
    imp = permutation_importance(reg, X, y, n_repeats=8, random_state=42, scoring="neg_mean_absolute_error")
    order = np.argsort(imp.importances_mean)[::-1]
    print("\n=== TOP 15 features (permutation importance) ===")
    for k in order[:15]:
        print("  %-22s %.3f" % (keys[k], imp.importances_mean[k]))


if __name__ == "__main__":
    main()
