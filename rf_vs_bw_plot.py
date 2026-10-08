#!/usr/bin/env python
"""Per-recording eyeball comparison on Exp3, at seg=20s / win=12s (RF's best):
  fig 1 = RandomForest regression-then-bin (LOSO) predicted category vs reference RR
  fig 2 = BW_Artifact ridge binned      predicted category vs reference RR

REF_MODE env: segmed (default) = reference truth is the segment-median of per-breath RR;
              basic            = raw per-breath reference interpolated at the segment centre
                                 (the reference "as is", unmodified).
"""
import os, sys, numpy as np
sys.path.insert(0, r"C:\Users\RachelMizrahi\OneDrive - CardiacSense\Documents\GitHub\BBBRR")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import bw_bp_sweep as S
from param_category_sweep import stft_power, ridge_rr, envelope_on_grid, GRID_FS

CH = ("Green", "Red", "IR", "Yellow", "Artifact")
PARAMS = (("RSA", "RSA"), ("RIIV", "AMP"), ("AUC", "AUC"))
LO, HI = 0.10, 0.80
E = [15.0, 22.0, 30.0]; LAB = ["<15", "15-22", "22-30", ">30"]
SEG, WIN, HOP = 15.0, 16.0, 2.5
YMAX = 55.0; BIN_COL = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]; band = [0.0] + E + [YMAX]


def col_feats(x, fs, win):
    t, fr, P = stft_power(x, fs, win, HOP)
    if t.size == 0: return np.zeros(0), np.zeros(0), np.zeros(0)
    b = np.where((fr >= LO) & (fr <= HI))[0]; Pb = P[b]; fb = fr[b]
    pk = Pb.max(0); med = np.median(Pb, 0)
    prom = np.divide(pk, med, out=np.full_like(pk, np.nan), where=med > 0)
    rg = fb[np.argmax(Pb, 0)] * 60.0; rg[pk <= 0] = np.nan
    return t, rg, prom


def main():
    from scipy.signal import butter, sosfiltfilt
    from respiration_rr.settings import PPG as ppg
    from main import run_ppg, run_reference, run_sync
    from respiration_rr.rr_average import reference_rr_points, reference_average_rr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    recs = S.discover(S.DEFAULT_DATA_ROOT); seen = set()
    recs = [r for r in recs if r[0].startswith("Exp3/") and "Test" not in r[1] and not (r[0] in seen or seen.add(r[0]))]

    rows, meta, keys, fullref = [], [], None, {}
    for rid, edf, csv in recs:
        ppg_res, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg_res, sig)
        pt, pr = reference_rr_points(ref); tav, rav = reference_average_rr(ref); fs = float(sig.fs)
        fullref[rid] = (np.asarray(tav) - offset, np.asarray(rav))   # full 20s-avg curve on watch clock (spectro white line)
        n = int(np.asarray(sig.channels["Artifact"]).size); dur = n / fs
        green = np.asarray(sig.channels.get("Green", np.zeros(n)), float)
        g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
        ny = fs / 2.0; sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
        SC = {}
        for c in CH:
            if c not in sig.channels: continue
            x = np.asarray(sig.channels[c], float)
            SC["raw_" + c] = col_feats(x, fs, WIN); SC["BW_" + c] = col_feats(sosfiltfilt(sos, x), fs, WIN)
            res = ppg_res.get(c)
            for pn, pl in PARAMS:
                prm = res.params.get(pn) if res and getattr(res, "params", None) else None
                bx = np.asarray(getattr(prm, "series_x", []), float) if prm else np.zeros(0)
                by = np.asarray(getattr(prm, "series_y", []), float) if prm else np.zeros(0)
                if bx.size >= 4:
                    SC[pl + "_" + c] = col_feats(envelope_on_grid(bx, by, "spl", g_t, ppg), GRID_FS, WIN)
        REF_MODE = os.environ.get("REF_MODE", "segmed")   # segmed = segment-median | basic = per-breath @ center | avg20 = 20s windowed-median @ center
        s = 0.0
        while s < dur:
            e = s + SEG; m = (pt >= s + offset) & (pt < e + offset)
            if REF_MODE == "basic":
                ref_rr = float(np.interp((s + e) / 2 + offset, pt, pr, left=np.nan, right=np.nan))
                have = np.isfinite(ref_rr)
            elif REF_MODE == "avg20":
                ref_rr = float(np.interp((s + e) / 2 + offset, tav, rav, left=np.nan, right=np.nan))
                have = np.isfinite(ref_rr)
            else:
                ref_rr = float(np.median(pr[m])) if m.sum() >= 1 else np.nan
                have = m.sum() >= 1
            if have:
                feat = {}; ridges = []
                for k in SC:
                    tt, rg, pm = SC[k]; mm = (tt >= s) & (tt < e)
                    rv = rg[mm][np.isfinite(rg[mm])]; pv = pm[mm][np.isfinite(pm[mm])]
                    rr = float(np.median(rv)) if rv.size else np.nan
                    feat[k + "_rr"] = rr; feat[k + "_prom"] = float(np.median(pv)) if pv.size else np.nan
                    if np.isfinite(rr): ridges.append(rr)
                a = int(max(0, s * fs)); b2 = int(min(green.size, e * fs)); xg = green[a:b2]
                if xg.size >= int(4 * fs):
                    f = np.fft.rfftfreq(xg.size, 1.0 / fs); Pw = np.abs(np.fft.rfft(xg - xg.mean())) ** 2
                    bd = (f >= 0.7) & (f <= 3.0); feat["hr"] = float(f[bd][np.argmax(Pw[bd])] * 60.0) if bd.any() else np.nan
                else: feat["hr"] = np.nan
                ridges = np.array(ridges, float)
                feat["cons_med"] = float(np.median(ridges)) if ridges.size else np.nan
                feat["cons_iqr"] = float(np.percentile(ridges, 75) - np.percentile(ridges, 25)) if ridges.size else np.nan
                feat["cons_nagree"] = int(np.sum(np.abs(ridges - feat["cons_med"]) <= 3.0)) if ridges.size else 0
                if keys is None: keys = sorted(feat.keys())
                rows.append([feat.get(k, np.nan) for k in keys])
                meta.append((rid, s, e, (s + e) / 2, ref_rr, feat.get("BW_Artifact_rr", np.nan)))
            s = e
    X = np.array(rows, float)
    rid_arr = np.array([m[0] for m in meta])
    s_arr = np.array([m[1] for m in meta]); e_arr = np.array([m[2] for m in meta])
    tarr = np.array([m[3] for m in meta]); yrr = np.array([m[4] for m in meta]); bwart = np.array([m[5] for m in meta])
    yc = np.digitize(yrr, E)

    # RF LOSO
    pred = np.full(len(X), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, yrr, rid_arr):
        mdl = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                                     random_state=42, n_jobs=-1))]).fit(X[tr], yrr[tr])
        pred[te] = mdl.predict(X[te])
    rf_c = np.digitize(pred, E)
    bw_c = np.where(np.isfinite(bwart), np.digitize(bwart, E), -1)   # NaN BW -> -1 (not counted)
    RM = os.environ.get("REF_MODE", "segmed")
    recids = sorted(set(rid_arr))

    def seg_pred_at(rid, t, pcat):
        mrec = rid_arr == rid
        idx = np.where((s_arr[mrec] <= t) & (t < e_arr[mrec]))[0]
        return int(pcat[mrec][idx[0]]) if idx.size else -1

    def dense_acc(pcat):
        """Per-segment prediction vs the FULL reference curve (every ~4 s point of reference_average_rr)."""
        tot = ok = 0; per = {}
        for rid in recids:
            tw, rv = fullref[rid]
            mrec = rid_arr == rid; t0 = s_arr[mrec].min(); t1 = e_arr[mrec].max()
            pt2 = po = 0
            for ti, ri in zip(tw, rv):
                if not np.isfinite(ri) or ri < 4 or ri > 55 or ti < t0 or ti >= t1: continue
                p = seg_pred_at(rid, ti, pcat)
                if p < 0 or p > 3: continue
                pt2 += 1; po += (p == int(np.digitize(ri, E)))
            tot += pt2; ok += po; per[rid] = (100.0 * po / pt2) if pt2 else np.nan
        return (100.0 * ok / tot if tot else np.nan), per

    accRF, perRF = dense_acc(rf_c); accBW, perBW = dense_acc(bw_c)
    print("REF_MODE=%s  seg=%.0f win=%.0f  DENSE vs full curve:  RF acc=%.1f%%  BW_Artifact acc=%.1f%%"
          % (RM, SEG, WIN, accRF, accBW))

    def figure(pcat, per, title, fname):
        nrec = len(recids); nrows = (nrec + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(15, 3 * nrows), squeeze=False); axes = axes.ravel()
        for ax, rid in zip(axes, recids):
            mrec = rid_arr == rid; pc = pcat[mrec]; rc = yc[mrec]; ss = s_arr[mrec]; ee = e_arr[mrec]
            tw, rv = fullref[rid]
            for k in range(4): ax.axhspan(band[k], band[k + 1], color=BIN_COL[k], alpha=0.06)
            for y in E: ax.axhline(y, color="#9ca3af", lw=0.7, ls=":")
            for si, ei, p, r in zip(ss, ee, pc, rc):     # predicted category band per segment; edge = centre-match
                if p < 0 or p > 3: continue
                ax.add_patch(Rectangle((si, band[p]), ei - si, band[p + 1] - band[p],
                                       color=BIN_COL[p], alpha=0.40, ec=("#16a34a" if p == r else "#ef4444"), lw=1.6))
            ax.plot(tw, rv, "-", color="#111827", lw=1.8)   # FULL reference_average_rr curve (spectro white line)
            ax.set_ylim(0, YMAX); ax.set_xlim(ss.min() - SEG, ee.max() + SEG)
            ax.set_title("%s  acc=%.0f%%" % (rid, per.get(rid, np.nan)), fontsize=10, fontweight="bold", loc="left")
            ax.set_ylabel("RR (bpm)")
        for ax in axes[nrec:]: ax.axis("off")
        fig.suptitle(title, fontsize=13, fontweight="bold"); fig.tight_layout(rect=[0, 0, 1, 0.97])
        outdir = os.environ.get("OUT_DIR",  # OneDrive project dir can't create files -> default to scratchpad
            r"C:\Users\RACHEL~1\AppData\Local\Temp\claude\C--Users-RachelMizrahi-OneDrive---CardiacSense-Documents-GitHub-BBBRR\3bbdb805-5d0d-4f52-b50c-def0402d0def\scratchpad")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved ->", out)

    figure(rf_c, perRF, "Exp3 seg=%.0fs win=%.0fs [ref=avg20 FULL curve] — RandomForest (LOSO)  [dense %.1f%%]" % (SEG, WIN, accRF),
           "rfvsbw_RF_fullref.png")
    figure(bw_c, perBW, "Exp3 seg=%.0fs win=%.0fs [ref=avg20 FULL curve] — BW_Artifact binned  [dense %.1f%%]" % (SEG, WIN, accBW),
           "rfvsbw_BW_fullref.png")


if __name__ == "__main__":
    main()
