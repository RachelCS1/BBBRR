#!/usr/bin/env python
"""
param_category_sweep.py — workstream C: does a per-beat parameter spectrogram beat
the Artifact BW for CATEGORIZATION, and which (channel x param x reconstruction)?

For every channel (Green/Red/IR/Artifact), every per-beat parameter (RSA/RIIV/AUC),
every reconstruction (MA / spline / ssp), we build the envelope exactly as the main
uniform analysis does, run the SAME winning spectrogram config as the BW baseline
(exact-length Hann window 48 s, hop 5 s, ridge ceiling 0.80 Hz), and score
categorization accuracy (per 30 s segment majority vs reference, 4 bins). The
Artifact BW (BP 0.1-1.0, full rate) is the baseline to beat, with a focus on the
weak >30 bpm category.

    py param_category_sweep.py                # all recordings
    py param_category_sweep.py --limit 4
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import bw_bp_sweep as S

RR_EDGES = [15.0, 22.0, 30.0]
CAT_LABELS = ["<15", "15-22", "22-30", ">30"]
SEG_SEC, HOP_SEC, F_LO, F_HI, WIN_SEC = 30.0, 5.0, 0.10, 0.80, 48.0
GRID_FS = 10.0
CHANNELS = ("Green", "Red", "IR", "Artifact")
PARAMS = ("RSA", "RIIV", "AUC")
RECON = ("ma", "spl", "ssp")


def stft_power(sig, fs, win_sec, hop_sec):
    sig = np.asarray(sig, np.float64)
    L = int(round(win_sec * fs))
    if sig.size < L:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    nfft = 1 << int(np.ceil(np.log2(L)))
    hop = max(1, int(round(hop_sec * fs)))
    han = np.hanning(L)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    starts = list(range(0, sig.size - L + 1, hop))
    P = np.empty((freqs.size, len(starts))); times = np.empty(len(starts))
    for i, a in enumerate(starts):
        seg = sig[a:a + L]
        P[:, i] = np.abs(np.fft.rfft((seg - seg.mean()) * han, n=nfft)) ** 2
        times[i] = (a + L / 2) / fs
    return times, freqs, P


def ridge_rr(freqs, P):
    bidx = np.where((freqs >= F_LO) & (freqs <= F_HI))[0]
    if bidx.size == 0 or P.size == 0:
        return np.full(P.shape[1] if P.ndim == 2 else 0, np.nan)
    Pb = P[bidx, :]
    rr = freqs[bidx][np.argmax(Pb, axis=0)] * 60.0
    rr[Pb.max(axis=0) <= 0] = np.nan
    return rr


def score_segments(times, rr, tat, tar, offset):
    if times.size == 0:
        return []
    out, s, t1 = [], times[0], times[-1]
    while s < t1:
        e = s + SEG_SEC
        m = (times >= s) & (times < e)
        rseg = rr[m]; rseg = rseg[np.isfinite(rseg)]
        if rseg.size:
            pred = int(np.bincount(np.digitize(rseg, RR_EDGES), minlength=4).argmax())
            ref = np.interp((s + e) / 2 + offset, tat, tar, left=np.nan, right=np.nan)
            if np.isfinite(ref):
                out.append((int(np.digitize(ref, RR_EDGES)), pred))
        s = e
    return out


def envelope_on_grid(x, y, recon, g_t, ppg):
    from respiration_rr.ppg.respiration import (bp_rr_series, cubic_spline_rr_series,
                                                smoothing_spline_rr_series)
    if recon == "ma":
        gx, gy = bp_rr_series(x, y, ppg.rr_band_low_hz, ppg.rr_band_high_hz,
                              ppg.rr_filter_order, resample_fs=GRID_FS)
    elif recon == "spl":
        gx, gy = cubic_spline_rr_series(x, y, GRID_FS)
    else:
        cut = getattr(ppg, "rr_ssp_cutoff_hz", 1.0) or 1.0
        gx, gy = smoothing_spline_rr_series(x, y, GRID_FS, cutoff_hz=cut)
    if gx is None or np.asarray(gx).size < 2:
        return np.zeros(g_t.size)
    return np.interp(g_t, gx, gy)


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
    print("Recordings: %d  window=%gs hop=%gs ceiling=%.2fHz\n" % (len(recs), WIN_SEC, HOP_SEC, F_HI))

    keys = ["Artifact-BW"] + [(c, p, r) for c in CHANNELS for p in PARAMS for r in RECON]
    pairs = {k: [] for k in keys}

    for rid, edf, csv in recs:
        try:
            ppg_res, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg_res, sig)
            tat, tar = reference_average_rr(ref)
            fs = float(sig.fs); n = int(np.asarray(sig.channels["Artifact"]).size)
        except Exception as e:
            print("  %-14s SKIP: %s" % (rid, e)); continue
        # Artifact BW baseline (BP 0.1-1.0, full rate)
        ny = fs / 2.0
        sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
        xbw = sosfiltfilt(sos, np.asarray(sig.channels["Artifact"], np.float64))
        t, fr, P = stft_power(xbw, fs, WIN_SEC, HOP_SEC)
        pairs["Artifact-BW"].extend(score_segments(t, ridge_rr(fr, P), tat, tar, offset))
        # param envelopes per channel (10 Hz grid)
        g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
        for c in CHANNELS:
            res = ppg_res.get(c)
            for p in PARAMS:
                pr = res.params.get(p) if res and getattr(res, "params", None) else None
                bx = np.asarray(getattr(pr, "series_x", []), np.float64) if pr else np.zeros(0)
                by = np.asarray(getattr(pr, "series_y", []), np.float64) if pr else np.zeros(0)
                for r in RECON:
                    if bx.size < 4:
                        continue
                    env = envelope_on_grid(bx, by, r, g_t, ppg)
                    tt, ff, PP = stft_power(env, GRID_FS, WIN_SEC, HOP_SEC)
                    pairs[(c, p, r)].extend(score_segments(tt, ridge_rr(ff, PP), tat, tar, offset))
        print("  %-14s done" % rid)

    def acc(pl):
        return 100.0 * np.mean([r == q for r, q in pl]) if pl else np.nan
    def cat_acc(pl, cc):
        sub = [(r, q) for r, q in pl if r == cc]
        return 100.0 * np.mean([r == q for r, q in sub]) if len(sub) >= 5 else np.nan

    base_t, base_hi = acc(pairs["Artifact-BW"]), cat_acc(pairs["Artifact-BW"], 3)
    print("\n=== BASELINE  Artifact-BW: total=%.1f%%  >30=%.1f%% ===" % (base_t, base_hi))
    print("\n%-9s %-6s %-6s %8s %8s %8s   %s" % ("channel", "param", "recon", "total", ">30", "vs base", "flag"))
    rows = []
    for c in CHANNELS:
        for p in PARAMS:
            for r in RECON:
                pl = pairs[(c, p, r)]
                if not pl:
                    continue
                tt, hh = acc(pl), cat_acc(pl, 3)
                rows.append((c, p, r, tt, hh))
    # print grouped, flag winners (beat baseline total or >30)
    for c, p, r, tt, hh in rows:
        flag = ""
        if np.isfinite(tt) and tt > base_t:
            flag += "TOTAL+ "
        if np.isfinite(hh) and np.isfinite(base_hi) and hh > base_hi:
            flag += ">30+ "
        print("%-9s %-6s %-6s %8.1f %8.1f %8s   %s" %
              (c, p, r, tt, hh, ("+%.1f" % (tt - base_t)) if np.isfinite(tt) else "--", flag))

    # best by total and best by >30
    valid = [(c, p, r, tt, hh) for c, p, r, tt, hh in rows if np.isfinite(tt)]
    if valid:
        bt = max(valid, key=lambda z: z[3])
        bh = max([v for v in valid if np.isfinite(v[4])] or valid, key=lambda z: (z[4] if np.isfinite(z[4]) else -1))
        print("\nBEST total : %s %s %s -> %.1f%% (>30 %.1f%%)" % (bt[0], bt[1], bt[2], bt[3], bt[4]))
        print("BEST >30   : %s %s %s -> >30 %.1f%% (total %.1f%%)" % (bh[0], bh[1], bh[2], bh[4], bh[3]))


if __name__ == "__main__":
    main()
