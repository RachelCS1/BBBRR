#!/usr/bin/env python
"""
consensus_fusion_sweep.py — the user's consensus/disagreement idea for categorization.

Per 30 s segment, pool many RR estimates: the 3 per-beat params (RSA/RIIV/AUC) from
EVERY channel (Green/Red/IR/Yellow/Artifact), spline reconstruction, PLUS the Artifact
BW. Then:
  * slow/mid regime -> take the CONSENSUS (median of the pool = the most-agreed value);
  * fast regime     -> detected by DISAGREEMENT (high spread of the pool), because
    per-beat params alias fast breathing (>~HR/2) and are expected to scatter.

CRUX first: does disagreement actually flag fast? We print the pool SPREAD (MAD)
stratified by the TRUE reference category. If spread really is higher for the >30
category, the fast-by-disagreement idea is valid. Then we compare methods by BOTH
categorization accuracy (total + per category) AND MAE (numeric RR error), to avoid
being fooled by near-boundary luck.

Config = the winner: exact Hann window 48 s, hop 5 s, argmax ridge 0.10-0.80 Hz.

    py consensus_fusion_sweep.py                # all recordings
    py consensus_fusion_sweep.py --limit 4
    REC_FILTER=Exp3 py consensus_fusion_sweep.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import bw_bp_sweep as S
from param_category_sweep import (stft_power, ridge_rr, envelope_on_grid,
                                  RR_EDGES, CAT_LABELS, SEG_SEC, HOP_SEC, WIN_SEC, GRID_FS)
from param_fusion_sweep import seg_medians

CHANNELS = ("Green", "Red", "IR", "Yellow", "Artifact")
PARAMS = ("RSA", "RIIV", "AUC")
RECON = "spl"
ROUTE_TAU = 22.0                 # route_fast threshold (reference method)
SPREAD_TAUS = [3.0, 5.0, 7.0, 10.0]   # MAD (bpm) thresholds for the disagreement router


def cat(v):
    return int(np.digitize(v, RR_EDGES)) if np.isfinite(v) else -1


def agree_consensus(vals, tol=3.0):
    """Majority-agreement: the CENTER of the largest cluster of values agreeing within
    +-tol bpm (isolated / disagreeing values are dropped). Stronger than the median when
    some params are off, because a lone outlier never enters the winning cluster."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan
    if v.size <= 2:
        return float(np.median(v))
    best, bn = None, -1
    for c in v:
        cl = v[np.abs(v - c) <= tol]
        if cl.size > bn:
            bn, best = cl.size, cl
    return float(np.median(best))


def ridge_prom(freqs, P):
    """Per-column spectral prominence in the RR band = peak / median (a clean peak
    is high). Used to drop weak (unreliable) per-beat param estimates before fusing."""
    from param_category_sweep import F_LO, F_HI
    bidx = np.where((freqs >= F_LO) & (freqs <= F_HI))[0]
    if bidx.size == 0 or P.size == 0:
        return np.full(P.shape[1] if P.ndim == 2 else 0, np.nan)
    Pb = P[bidx, :]; pk = Pb.max(0); med = np.median(Pb, 0)
    return np.divide(pk, med, out=np.full_like(pk, np.nan), where=med > 0)


PROM_THS = [2.0, 3.0, 5.0]   # prominence gates to try for prominence-filtered agreement


def hr_from_beats(ppg_res, sig):
    """Per-beat HR (bpm) on the SIGNAL clock, from the cleanest available LED."""
    for ch in ("IR", "Green", "Red", "Yellow"):
        res = ppg_res.get(ch)
        idx = getattr(res, "ss_idx", None) if res is not None else None
        if idx is not None and len(idx) > 3:
            bt = np.asarray(sig.time)[np.asarray(idx, int)]
            dt = np.diff(bt); ok = dt > 0
            hr = 60.0 / dt[ok]; ht = ((bt[:-1] + bt[1:]) / 2)[ok]
            good = (hr >= 35) & (hr <= 200)
            return ht[good], hr[good]
    return np.zeros(0), np.zeros(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from respiration_rr.settings import PPG as ppg
    from scipy.signal import butter, sosfiltfilt
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr

    recs = S.discover(args.data_root)
    if args.limit:
        recs = recs[:args.limit]
    print("Recordings: %d  channels=%s  params=%s recon=%s\n" % (len(recs), CHANNELS, PARAMS, RECON))

    # per-segment records across all recordings
    R = []   # each: dict(refcat, refval, bw, cons, spread, irmed, npool)
    for rid, edf, csv in recs:
        try:
            ppg_res, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg_res, sig)
            tat, tar = reference_average_rr(ref)
            fs = float(sig.fs); n = int(np.asarray(sig.channels["Artifact"]).size)
        except Exception as e:
            print("  %-14s SKIP: %s" % (rid, e)); continue
        ny = fs / 2.0
        sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
        xbw = sosfiltfilt(sos, np.asarray(sig.channels["Artifact"], np.float64))
        tb, fb, Pb = stft_power(xbw, fs, WIN_SEC, HOP_SEC)
        if tb.size == 0:
            continue
        segs = np.arange(tb[0], tb[-1], SEG_SEC)
        bw_seg = seg_medians(tb, ridge_rr(fb, Pb), segs)
        ht, hv = hr_from_beats(ppg_res, sig)
        hr_seg = seg_medians(ht, hv, segs) if ht.size else np.full(segs.size, np.nan)

        g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
        src = {"Artifact-BW": bw_seg}
        prom_list = []                               # per-param per-seg prominence (params only, in add order)
        ir_rows = []
        for c in CHANNELS:
            res = ppg_res.get(c)
            for p in PARAMS:
                pr = res.params.get(p) if res and getattr(res, "params", None) else None
                bx = np.asarray(getattr(pr, "series_x", []), np.float64) if pr else np.zeros(0)
                by = np.asarray(getattr(pr, "series_y", []), np.float64) if pr else np.zeros(0)
                if bx.size < 4:
                    continue
                env = envelope_on_grid(bx, by, RECON, g_t, ppg)
                tt, ff, PP = stft_power(env, GRID_FS, WIN_SEC, HOP_SEC)
                src[f"{c}.{p}"] = seg_medians(tt, ridge_rr(ff, PP), segs)
                prom_list.append(seg_medians(tt, ridge_prom(ff, PP), segs))
                if c == "IR":
                    ir_rows.append(src[f"{c}.{p}"])
        names = list(src.keys())
        M = np.vstack([src[k] for k in names])       # (n_sources, n_seg); row 0 = Artifact-BW
        pmask = np.array([k != "Artifact-BW" for k in names])   # per-beat params only
        Mprom = np.vstack(prom_list) if prom_list else np.full((1, segs.size), np.nan)  # aligned to M[pmask]
        ir = np.vstack(ir_rows) if ir_rows else None

        for i, s in enumerate(segs):
            vals = M[:, i]; vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            cons = float(np.median(vals))
            spread = float(np.median(np.abs(vals - cons)))     # MAD
            refv = np.interp(s + SEG_SEC / 2 + offset, tat, tar, left=np.nan, right=np.nan)
            if not np.isfinite(refv):
                continue
            pv_all = M[pmask, i]; pp_all = Mprom[:, i]          # params + their prominence
            fin = np.isfinite(pv_all)
            pv = pv_all[fin]; pp = pp_all[fin]
            pmed = float(np.median(pv)) if pv.size else np.nan
            bwv = float(bw_seg[i])
            gap = (bwv - pmed) if (np.isfinite(bwv) and np.isfinite(pmed)) else np.nan  # Artifact - params
            irmed = np.nan
            if ir is not None:
                iv = ir[:, i]; iv = iv[np.isfinite(iv)]
                if iv.size:
                    irmed = float(np.median(iv))
            R.append(dict(refcat=cat(refv), refval=float(refv), bw=bwv,
                          cons=cons, spread=spread, pmed=pmed, gap=gap,
                          pvals=pv.tolist(), pproms=pp.tolist(),
                          irmed=irmed, hr=float(hr_seg[i]), npool=int(vals.size)))
        print("  %-14s done (%d sources)" % (rid, M.shape[0]))

    if not R:
        sys.exit("no segments scored.")

    # ---- CRUX diagnostic: pool spread stratified by TRUE category ----
    print("\n=== disagreement diagnostic: pool spread (MAD, bpm) by TRUE category ===")
    print("  %-8s %6s %10s %10s" % ("cat", "n", "mean-spread", "median-spread"))
    for c in range(4):
        sp = [r["spread"] for r in R if r["refcat"] == c]
        if sp:
            print("  %-8s %6d %10.1f %10.1f" % (CAT_LABELS[c], len(sp), np.mean(sp), np.median(sp)))
    print("  (if >30 spread is clearly higher, 'fast = disagreement' is valid)")

    # separation: % of each category ABOVE a spread threshold (can we detect fast by spread?)
    taus = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    print("\n=== separation: %% of segments with spread >= tau, by TRUE category ===")
    print("  %-6s" % "tau" + "".join("%9s" % CAT_LABELS[c] for c in range(4)))
    for tau in taus:
        row = "  %-6.1f" % tau
        for c in range(4):
            sp = [r["spread"] for r in R if r["refcat"] == c]
            row += ("%8.0f%%" % (100 * np.mean([s >= tau for s in sp]))) if sp else "%9s" % "--"
        print(row)
    print("  (fast detectable if >30 stays high while <15/15-22 drop to ~0)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        data = [[r["spread"] for r in R if r["refcat"] == c] for c in range(4)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, labels=CAT_LABELS, showfliers=True, whis=(5, 95))
        ax.set_ylabel("pool spread (MAD, bpm)")
        ax.set_title("disagreement (pool spread) by TRUE category — %d segments" % len(R))
        ax.grid(alpha=.3, axis="y")
        out = os.path.join(os.environ.get("TEMP", "."), "spread_by_category.png")
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print("  boxplot saved -> %s" % out)
    except Exception as e:
        print("  (boxplot skipped: %s)" % e)

    # ---- Artifact-vs-params GAP: directional fast detector (bw reads high, params alias low) ----
    print("\n=== Artifact-BW  minus  params_median  GAP (bpm) by TRUE category ===")
    print("  %-8s %6s %10s %10s" % ("cat", "n", "mean-gap", "median-gap"))
    for c in range(4):
        g = [r["gap"] for r in R if r["refcat"] == c and np.isfinite(r["gap"])]
        if g:
            print("  %-8s %6d %10.1f %10.1f" % (CAT_LABELS[c], len(g), np.mean(g), np.median(g)))
    print("  %% of segments with GAP >= tau (Artifact that much higher than params):")
    print("  %-6s" % "tau" + "".join("%9s" % CAT_LABELS[c] for c in range(4)))
    for tau in [3.0, 5.0, 8.0, 12.0]:
        row = "  %-6.0f" % tau
        for c in range(4):
            g = [r["gap"] for r in R if r["refcat"] == c and np.isfinite(r["gap"])]
            row += ("%8.0f%%" % (100 * np.mean([x >= tau for x in g]))) if g else "%9s" % "--"
        print(row)
    print("  (a >30-specific detector needs >30 high while 22-30 AND slow stay low)")

    # ---- HR premise: are the per-beat params even VALID for fast segments? ----
    print("\n=== HR premise: does fast breathing come WITH high HR? (params valid iff HR/2 >= trueRR) ===")
    print("  %-8s %6s %8s %8s %15s" % ("cat", "n", "mean-HR", "mean-HR/2", "%HR/2>=trueRR"))
    for c in range(4):
        sub = [r for r in R if r["refcat"] == c and np.isfinite(r["hr"])]
        if sub:
            hrs = [r["hr"] for r in sub]
            cap = 100 * np.mean([(r["hr"] / 2.0) >= r["refval"] for r in sub])
            print("  %-8s %6d %8.0f %8.0f %14.0f%%" % (CAT_LABELS[c], len(sub), np.mean(hrs), np.mean(hrs) / 2, cap))
    print("  (>30 with high %HR/2>=trueRR => params CAN capture fast there => hr-route should rescue it)")

    # is the "valid 40%%" of >30 actually captured by the params? (potential vs realized)
    hi = [r for r in R if r["refcat"] == 3 and np.isfinite(r["hr"])]
    valid = [r for r in hi if r["hr"] / 2.0 >= r["refval"]]
    invalid = [r for r in hi if r["hr"] / 2.0 < r["refval"]]
    def catacc(sub, key):
        v = [cat(r[key]) == 3 for r in sub if np.isfinite(r[key])]
        return 100 * np.mean(v) if v else float("nan")
    print("\n=== >30 split by params-validity — is the valid subset actually captured? ===")
    print("  valid   (HR/2>=RR) n=%2d:  bw=%3.0f%%  params=%3.0f%%  consensus=%3.0f%%"
          % (len(valid), catacc(valid, "bw"), catacc(valid, "pmed"), catacc(valid, "cons")))
    print("  invalid (HR/2< RR) n=%2d:  bw=%3.0f%%  params=%3.0f%%  consensus=%3.0f%%"
          % (len(invalid), catacc(invalid, "bw"), catacc(invalid, "pmed"), catacc(invalid, "cons")))
    print("  (if params >> bw on the VALID subset, the 40%% is a REAL opportunity worth a better router)")

    # ---- method comparison: category accuracy (total + per cat) + MAE ----
    def pred_bw(r):        return r["bw"]
    def pred_cons(r):      return r["cons"]
    def pred_route(r):     return r["bw"] if (np.isfinite(r["bw"]) and r["bw"] >= ROUTE_TAU) else (r["irmed"] if np.isfinite(r["irmed"]) else r["bw"])
    def make_spread(tau):
        def f(r):          return r["bw"] if r["spread"] >= tau else r["cons"]
        return f

    def make_gap(tau):
        def f(r):          return r["bw"] if (np.isfinite(r["gap"]) and r["gap"] >= tau) else r["cons"]
        return f

    def make_hr(frac):
        # HR-aware route: params are valid up to ~HR/2, so if BW says we are within
        # that range use the (cleaner) params, else fall to BW. frac=0.5 -> HR/2.
        def f(r):
            thr = frac * r["hr"] if np.isfinite(r["hr"]) else ROUTE_TAU
            use_p = np.isfinite(r["bw"]) and np.isfinite(r["pmed"]) and r["bw"] <= thr
            return r["pmed"] if use_p else r["bw"]
        return f

    def pred_route_agree(r):   # route_fast but the slow branch uses majority-agreement, not median
        if np.isfinite(r["bw"]) and r["bw"] >= ROUTE_TAU:
            return r["bw"]
        a = agree_consensus(r["pvals"])
        return a if np.isfinite(a) else r["bw"]

    def make_promagree(th):    # majority-agreement over only the STRONG params (prominence >= th)
        def f(r):
            if np.isfinite(r["bw"]) and r["bw"] >= ROUTE_TAU:
                return r["bw"]
            pv = np.asarray(r["pvals"], float); pp = np.asarray(r["pproms"], float)
            strong = pv[np.isfinite(pp) & (pp >= th)]
            a = agree_consensus(strong if strong.size >= 2 else pv)
            return a if np.isfinite(a) else r["bw"]
        return f

    methods = [("bw", pred_bw), ("consensus", pred_cons), ("route_fast", pred_route),
               ("route_agree", pred_route_agree)]
    for th in PROM_THS:
        methods.append(("promagree>=%g" % th, make_promagree(th)))
    for tau in SPREAD_TAUS:
        methods.append(("spread>=%g" % tau, make_spread(tau)))
    for tau in (3.0, 5.0, 8.0):
        methods.append(("gap>=%g" % tau, make_gap(tau)))
    for name, fr in (("hr/2", 0.5), ("hr*0.45", 0.45)):
        methods.append((name, make_hr(fr)))

    def acc(pred, sub=None):
        rr = R if sub is None else [r for r in R if r["refcat"] == sub]
        v = [cat(pred(r)) == r["refcat"] for r in rr if np.isfinite(pred(r))]
        return 100.0 * np.mean(v) if v else np.nan
    def mae(pred):
        v = [abs(pred(r) - r["refval"]) for r in R if np.isfinite(pred(r))]
        return float(np.mean(v)) if v else np.nan

    print("\n=== methods: categorization accuracy + MAE  (%d segments) ===" % len(R))
    print("  %-12s %7s %7s %7s %7s %7s %8s" % ("method", "total", "<15", "15-22", "22-30", ">30", "MAE"))
    for name, f in methods:
        print("  %-12s %6.1f %7.1f %7.1f %7.1f %7.1f %8.2f" %
              (name, acc(f), acc(f, 0), acc(f, 1), acc(f, 2), acc(f, 3), mae(f)))


if __name__ == "__main__":
    main()
