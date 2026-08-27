#!/usr/bin/env python
"""
fence_sweep.py — find the best envelope-distance FENCE parameters for the
Butter+detrend average-RR extraction, scored by category-routing accuracy over
all recordings.

Two knobs are swept:
  --env-gate  (P2P fraction)  : keep a peak only if prominence >= gate * P2P-ref
  --env-win   (adapt window s): 0 = global median P2P, >0 = LOCAL moving reference

For every recording it loads once (reference, beats, Butter), then for each
(gate, win) combo recomputes only detrend -> spectrogram -> ridge and scores the
30 s segment-majority RR x HR routing (same metric as bw_category_map, ridge
argmax in [0.1,0.9] Hz). gate=0 is the no-fence control (plain Butter+detrend).

Reports per combo: worst-case cell %, the RR<12 and RR12-30 (x HR60-85) cells,
and total — so you can see the low-RR vs mid/high-RR balance.

    py fence_sweep.py
    py fence_sweep.py --gates 0 0.3 0.5 0.7 --wins 0 10 20 40
    py fence_sweep.py --recordings Exp2/002 Exp2/pilot01
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bw_bp_sweep as S
import bw_category_map as CM

SEG = 30.0; SEG_MIN = 3; MIN_SAMPLES = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--recordings", nargs="+", default=None)
    ap.add_argument("--gates", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    ap.add_argument("--wins", type=float, nargs="+", default=[0.0, 10.0, 20.0, 40.0])
    ap.add_argument("--specB-window", type=float, default=32.0)
    ap.add_argument("--detrend-baseline", type=float, default=5.0)
    args = ap.parse_args()

    import dataclasses
    from respiration_rr.settings import PPG as _PPG
    cfg = dataclasses.replace(_PPG, rr_band_high_hz=0.9)
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr
    from respiration_rr.ppg.bw_methods import butterworth, legacy_detrend
    from respiration_rr.ppg.spectrogram import compute_spectrogram, ridge_rr

    # combos: gate=0 -> single no-fence control; gate>0 -> gate x win grid
    combos = [(0.0, 0.0)]
    for g in args.gates:
        if g > 0:
            for w in args.wins:
                combos.append((g, w))

    recs = S.discover(args.data_root)
    if args.recordings:
        recs = [r for r in recs if r[0] in args.recordings]
    n_rr, n_hr = len(CM.RR_EDGES) + 1, len(CM.HR_EDGES) + 1
    tot = {c: np.zeros((n_rr, n_hr)) for c in combos}
    inb = {c: np.zeros((n_rr, n_hr)) for c in combos}

    low, high = cfg.bw_band_low_hz, cfg.bw_band_high_hz
    for rid, edf, csv in recs:
        try:
            ppg, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg, sig)
        except Exception as e:
            print(f"  [skip] {rid}: {e}"); continue
        truth_at, truth_ar = reference_average_rr(ref)
        hr_t, hr_v = CM.hr_series_from_beats(ppg, sig, offset)
        if truth_at.size < 2 or hr_t.size < 2 or args.channel not in sig.channels:
            print(f"  [skip] {rid}: no ref/HR/channel"); continue
        x = np.asarray(sig.channels[args.channel], np.float64); fs = float(sig.fs)
        bp = butterworth(x, fs, low, high)
        print(f"  {rid}: scoring {len(combos)} combos")
        for (g, w) in combos:
            det = legacy_detrend(bp, fs, low, high, baseline_sec=args.detrend_baseline,
                                 env_gate_frac=(g if g > 0 else None), env_gate_win=w)
            spect = compute_spectrogram(det - det.mean(), fs, args.specB_window, cfg.rr_band_high_hz + 0.2)
            if spect is None:
                continue
            st, rr = ridge_rr(spect, cfg.rr_band_low_hz, cfg.rr_band_high_hz)
            bt = np.asarray(st) + offset; rr = np.asarray(rr)
            t0 = max(truth_at[0], hr_t[0]); t1 = min(truth_at[-1], hr_t[-1]); seg = t0
            while seg + SEG <= t1 + 1e-9:
                a0, a1 = seg, seg + SEG; seg += SEG
                rr_ref = truth_ar[(truth_at >= a0) & (truth_at < a1) & np.isfinite(truth_ar)]
                rr_ref = rr_ref[(rr_ref >= 4) & (rr_ref <= 55)]
                rr_our = rr[(bt >= a0) & (bt < a1) & np.isfinite(rr)]
                hseg = hr_v[(hr_t >= a0) & (hr_t < a1)]
                if rr_ref.size < SEG_MIN or rr_our.size < SEG_MIN or hseg.size < 1:
                    continue
                rb = CM._mode_bin(rr_ref, n_rr, CM.RR_EDGES)
                ob = CM._mode_bin(rr_our, n_rr, CM.RR_EDGES)
                hb = int(np.digitize(np.median(hseg), CM.HR_EDGES))
                tot[(g, w)][rb, hb] += 1
                inb[(g, w)][rb, hb] += int(ob == rb)

    RRL = CM._labels(CM.RR_EDGES, "RR"); HRL = CM._labels(CM.HR_EDGES, "HR")
    def cellpct(c, ri, hi):
        t = tot[c][ri, hi]; return (100.0 * inb[c][ri, hi] / t) if t else np.nan, int(t)
    print("\n================  FENCE SWEEP (Butter+detrend, ridge<=0.9)  ================")
    print(f"{'gate':>5} {'win':>4} | {'worst':>6} | {'RR<12xHR60-85':>14} {'RR12-30xHR60-85':>16} {'RR30+xHR60-85':>14} | {'total':>7}")
    rows = []
    for c in combos:
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 100.0 * inb[c] / tot[c]
        p[tot[c] < MIN_SAMPLES] = np.nan
        worst = np.nanmin(p) if np.isfinite(p).any() else np.nan
        tt = tot[c].sum(); ii = inb[c].sum(); total = 100.0 * ii / tt if tt else np.nan
        low_p, low_n = cellpct(c, 0, 1); mid_p, mid_n = cellpct(c, 1, 1); hi_p, hi_n = cellpct(c, 2, 1)
        tag = "  (no fence)" if c == (0.0, 0.0) else ""
        print(f"{c[0]:>5.2f} {c[1]:>4.0f} | {worst:>6.0f} | "
              f"{low_p:>6.0f}% ({low_n:>3}) {mid_p:>7.0f}% ({mid_n:>3}) {hi_p:>6.0f}% ({hi_n:>3}) | {total:>6.0f}%{tag}")
        rows.append((worst if np.isfinite(worst) else -1, total, c))
    best = max(rows, key=lambda r: (r[0], r[1]))
    print(f"\nBEST worst-case: gate={best[2][0]}, win={best[2][1]}  (worst {best[0]:.0f}%, total {best[1]:.0f}%)")
    ctrl = next(r for r in rows if r[2] == (0.0, 0.0))
    print(f"control (no fence): worst {ctrl[0]:.0f}%, total {ctrl[1]:.0f}%")


if __name__ == "__main__":
    main()
