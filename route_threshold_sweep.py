#!/usr/bin/env python
"""
route_threshold_sweep.py — tune the route_fast threshold.

route_fast: per 30 s segment, if the BW estimate >= TAU use the BW (fast regime it
alone can see), else use median(IR RSA/RIIV/AUC spline) (slow regime, where the
per-beat params are clean and valid). The 22-30 dip in the fixed-28 version was
because ~27-30 bpm sits at the params' HR/2 edge -> routing there to params hurt.
Sweep TAU to find the balance. Config = winner (BW BP0.1-1.0 full rate, window 48 s,
hop 5 s, ceiling 0.80 Hz). Metric: categorization accuracy, total + per category.

    py route_threshold_sweep.py
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
                                  RR_EDGES, SEG_SEC, HOP_SEC, WIN_SEC, GRID_FS)
from param_fusion_sweep import seg_medians

PARAM_CH, RECON, PARAMS = "IR", "spl", ("RSA", "RIIV", "AUC")
TAUS = [0.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 999.0]   # 0=all BW, 999=all params


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
    print("Recordings: %d  param_ch=%s recon=%s\n" % (len(recs), PARAM_CH, RECON))

    pairs = {tau: [] for tau in TAUS}
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
        seg_starts = np.arange(tb[0], tb[-1], SEG_SEC)
        bw_seg = seg_medians(tb, ridge_rr(fb, Pb), seg_starts)

        g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
        res = ppg_res.get(PARAM_CH)
        par = []
        for p in PARAMS:
            pr = res.params.get(p) if res and getattr(res, "params", None) else None
            bx = np.asarray(getattr(pr, "series_x", []), np.float64) if pr else np.zeros(0)
            by = np.asarray(getattr(pr, "series_y", []), np.float64) if pr else np.zeros(0)
            if bx.size < 4:
                par.append(np.full(seg_starts.size, np.nan)); continue
            env = envelope_on_grid(bx, by, RECON, g_t, ppg)
            tt, ff, PP = stft_power(env, GRID_FS, WIN_SEC, HOP_SEC)
            par.append(seg_medians(tt, ridge_rr(ff, PP), seg_starts))
        par = np.array(par)

        for i, s in enumerate(seg_starts):
            refv = np.interp(s + SEG_SEC / 2 + offset, tat, tar, left=np.nan, right=np.nan)
            if not np.isfinite(refv):
                continue
            rc = int(np.digitize(refv, RR_EDGES))
            bw = bw_seg[i]
            pv = par[:, i]; pv = pv[np.isfinite(pv)]
            pmed = np.median(pv) if pv.size else np.nan
            for tau in TAUS:
                if np.isfinite(bw) and bw >= tau:
                    v = bw
                elif np.isfinite(pmed):
                    v = pmed
                else:
                    v = bw
                if np.isfinite(v):
                    pairs[tau].append((rc, int(np.digitize(v, RR_EDGES))))
        print("  %-14s done" % rid)

    def acc(pl):
        return 100.0 * np.mean([a == b for a, b in pl]) if pl else np.nan
    def cat(pl, c):
        sub = [(a, b) for a, b in pl if a == c]
        return 100.0 * np.mean([a == b for a, b in sub]) if len(sub) >= 5 else np.nan

    print("\n%-8s %8s %8s %8s %8s %8s" % ("tau", "total", "<15", "15-22", "22-30", ">30"))
    best = (None, -1)
    for tau in TAUS:
        pl = pairs[tau]
        t = acc(pl)
        lab = "all-BW" if tau == 0 else ("all-par" if tau == 999 else "%.0f" % tau)
        print("%-8s %8.1f %8.1f %8.1f %8.1f %8.1f" % (lab, t, cat(pl, 0), cat(pl, 1), cat(pl, 2), cat(pl, 3)))
        if np.isfinite(t) and t > best[1]:
            best = (lab, t)
    print("\nBEST total: tau=%s -> %.1f%%" % best)


if __name__ == "__main__":
    main()
