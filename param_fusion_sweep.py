#!/usr/bin/env python
"""
param_fusion_sweep.py — workstream C, fusion: can COMBINING parameters (and the
BW) beat any single one for categorization?

Per 30 s segment we get a median RR from each source, then test fusion rules and
score categorization accuracy (total + the weak >30 cell) vs the Artifact-BW
baseline. Sources: Artifact BW (continuous, sees fast) + IR RSA/RIIV/AUC (spline
reconstruction; per-beat -> blind above ~HR/2). Config = the winner: exact Hann
window 48 s, hop 5 s, ridge ceiling 0.80 Hz.

Fusion rules:
  bw            baseline (Artifact BW only)
  par_mean      mean of the 3 IR params
  par_median    median of the 3 IR params
  all_mean      mean of BW + 3 params
  all_median    median of BW + 3 params
  agree_gate    Karlen-style: if SD(3 params) < 4 bpm use their mean, else BW
  route_fast    complementary: if BW >= 28 bpm use BW (fast regime), else median(params)

    py param_fusion_sweep.py                # all recordings
    py param_fusion_sweep.py --limit 4
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

PARAM_CH = "IR"
RECON = "spl"
PARAMS = ("RSA", "RIIV", "AUC")
FAST_TH = 28.0                     # route_fast: BW>=this -> use BW


def seg_medians(times, rr, seg_starts):
    """Median RR per segment (NaN if none)."""
    out = np.full(seg_starts.size, np.nan)
    for i, s in enumerate(seg_starts):
        m = (times >= s) & (times < s + SEG_SEC)
        v = rr[m]; v = v[np.isfinite(v)]
        if v.size:
            out[i] = np.median(v)
    return out


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
    print("Recordings: %d  param_ch=%s recon=%s  window=%gs ceiling=0.80Hz\n" % (len(recs), PARAM_CH, RECON, WIN_SEC))

    RULES = ["bw", "par_mean", "par_median", "all_mean", "all_median", "agree_gate", "route_fast"]
    pairs = {r: [] for r in RULES}

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
            print("  %-14s SKIP (short)" % rid); continue
        seg_starts = np.arange(tb[0], tb[-1], SEG_SEC)
        bw_seg = seg_medians(tb, ridge_rr(fb, Pb), seg_starts)

        g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
        res = ppg_res.get(PARAM_CH)
        par_seg = []
        for p in PARAMS:
            pr = res.params.get(p) if res and getattr(res, "params", None) else None
            bx = np.asarray(getattr(pr, "series_x", []), np.float64) if pr else np.zeros(0)
            by = np.asarray(getattr(pr, "series_y", []), np.float64) if pr else np.zeros(0)
            if bx.size < 4:
                par_seg.append(np.full(seg_starts.size, np.nan)); continue
            env = envelope_on_grid(bx, by, RECON, g_t, ppg)
            tt, ff, PP = stft_power(env, GRID_FS, WIN_SEC, HOP_SEC)
            par_seg.append(seg_medians(tt, ridge_rr(ff, PP), seg_starts))
        par_seg = np.array(par_seg)                     # (3, nseg)

        for i, s in enumerate(seg_starts):
            ref = np.interp(s + SEG_SEC / 2 + offset, tat, tar, left=np.nan, right=np.nan)
            if not np.isfinite(ref):
                continue
            rc = int(np.digitize(ref, RR_EDGES))
            bw = bw_seg[i]
            pv = par_seg[:, i]; pv = pv[np.isfinite(pv)]
            fused = {}
            fused["bw"] = bw
            if pv.size:
                fused["par_mean"] = np.mean(pv)
                fused["par_median"] = np.median(pv)
            if np.isfinite(bw):
                allv = np.concatenate([[bw], pv]) if pv.size else np.array([bw])
                fused["all_mean"] = np.mean(allv)
                fused["all_median"] = np.median(allv)
                fused["agree_gate"] = np.mean(pv) if (pv.size >= 2 and np.std(pv) < 4.0) else bw
                fused["route_fast"] = bw if bw >= FAST_TH else (np.median(pv) if pv.size else bw)
            for r in RULES:
                v = fused.get(r, np.nan)
                if np.isfinite(v):
                    pairs[r].append((rc, int(np.digitize(v, RR_EDGES))))
        print("  %-14s done" % rid)

    def acc(pl):
        return 100.0 * np.mean([a == b for a, b in pl]) if pl else np.nan
    def cat_acc(pl, c):
        sub = [(a, b) for a, b in pl if a == c]
        return 100.0 * np.mean([a == b for a, b in sub]) if len(sub) >= 5 else np.nan

    print("\n%-12s %8s %8s %8s %8s %8s" % ("rule", "total", "<15", "15-22", "22-30", ">30"))
    for r in RULES:
        pl = pairs[r]
        print("%-12s %8.1f %8.1f %8.1f %8.1f %8.1f" %
              (r, acc(pl), cat_acc(pl, 0), cat_acc(pl, 1), cat_acc(pl, 2), cat_acc(pl, 3)))


if __name__ == "__main__":
    main()
