#!/usr/bin/env python
"""
ridge_ceiling_sweep.py — does raising the ridge SEARCH ceiling above 0.7 Hz help
the fast-breathing (>30 bpm) category?  (follow-up to filter_window_sweep)

The A+B sweep showed the >30 category is the weak cell. Hypothesis: real breaths
at 42-51 bpm (0.70-0.85 Hz) fall just above a 0.7 Hz search cap, so the argmax is
forced onto a lower peak -> under-read. Test: fix the two winning filters and a few
windows, and sweep the ridge search ceiling {0.70, 0.80, 0.85, 0.90}, reporting
BOTH total accuracy AND per-category accuracy (so we see the >30 cell and any
over-read of slow breaths).

The STFT power is computed once per (filter, window); only the argmax band changes
with the ceiling, so the sweep is cheap. Exact-length Hann window, zero-pad to pow2.

    py ridge_ceiling_sweep.py                 # all recordings
    py ridge_ceiling_sweep.py --limit 4
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
SEG_SEC = 30.0
HOP_SEC = 5.0
F_LO = 0.10
FILTERS = ["raw", "BP0.1-1.0"]
WINDOWS = [32.0, 48.0, 64.0]
FHIS = [0.70, 0.80, 0.85, 0.90]


def apply_filter(x, fs, name):
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, np.float64)
    if name == "raw":
        return x
    ny = fs / 2.0
    sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def stft_power(sig, fs, win_sec, hop_sec):
    """Exact-length Hann STFT (zero-pad to pow2). Returns (times, freqs, P) with
    P shape (nfreq, ncol)."""
    sig = np.asarray(sig, np.float64)
    L = int(round(win_sec * fs))
    if sig.size < L:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    nfft = 1 << int(np.ceil(np.log2(L)))
    hop = max(1, int(round(hop_sec * fs)))
    han = np.hanning(L)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    starts = list(range(0, sig.size - L + 1, hop))
    P = np.empty((freqs.size, len(starts)))
    times = np.empty(len(starts))
    for i, a in enumerate(starts):
        seg = sig[a:a + L]
        P[:, i] = np.abs(np.fft.rfft((seg - seg.mean()) * han, n=nfft)) ** 2
        times[i] = (a + L / 2) / fs
    return times, freqs, P


def ridge_rr(freqs, P, f_hi):
    bidx = np.where((freqs >= F_LO) & (freqs <= f_hi))[0]
    if bidx.size == 0:
        return np.full(P.shape[1], np.nan)
    Pb = P[bidx, :]
    rr = freqs[bidx][np.argmax(Pb, axis=0)] * 60.0
    rr[Pb.max(axis=0) <= 0] = np.nan
    return rr


def score_segments(times, rr, tat, tar, offset):
    if times.size == 0:
        return []
    t0, t1 = times[0], times[-1]
    out, s = [], times[0]
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr

    recs = S.discover(args.data_root)
    if args.limit:
        recs = recs[:args.limit]
    print("Recordings: %d  channel=%s  ceilings=%s\n" % (len(recs), args.channel, FHIS))

    pairs = {(f, w, fh): [] for f in FILTERS for w in WINDOWS for fh in FHIS}
    for rid, edf, csv in recs:
        try:
            ppg, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg, sig)
            tat, tar = reference_average_rr(ref)
            if args.channel not in sig.channels:
                print("  %-14s SKIP" % rid); continue
            x = np.asarray(sig.channels[args.channel], np.float64); fs = float(sig.fs)
        except Exception as e:
            print("  %-14s SKIP: %s" % (rid, e)); continue
        for fname in FILTERS:
            xf = apply_filter(x, fs, fname)
            for w in WINDOWS:
                t, fr, P = stft_power(xf, fs, w, HOP_SEC)
                for fh in FHIS:
                    pairs[(fname, w, fh)].extend(score_segments(t, ridge_rr(fr, P, fh), tat, tar, offset))
        print("  %-14s done" % rid)

    def acc(pl):
        return 100.0 * np.mean([r == p for r, p in pl]) if pl else np.nan
    def cat_acc(pl, c):
        sub = [(r, p) for r, p in pl if r == c]
        return 100.0 * np.mean([r == p for r, p in sub]) if len(sub) >= 5 else np.nan

    for fname in FILTERS:
        print("\n===== filter = %s =====" % fname)
        print("  total accuracy (%) — window x ceiling")
        print("    %-8s" % "" + "".join("%9s" % ("%.2fHz" % fh) for fh in FHIS))
        for w in WINDOWS:
            print("    %-8s" % ("%gs" % w) + "".join("%9.1f" % acc(pairs[(fname, w, fh)]) for fh in FHIS))
        print("  >30 bpm category accuracy (%) — window x ceiling")
        print("    %-8s" % "" + "".join("%9s" % ("%.2fHz" % fh) for fh in FHIS))
        for w in WINDOWS:
            print("    %-8s" % ("%gs" % w) + "".join("%9.1f" % cat_acc(pairs[(fname, w, fh)], 3) for fh in FHIS))

    # per-category detail for the best (filter,window,ceiling) by total
    best = max(pairs, key=lambda k: (acc(pairs[k]) if pairs[k] else -1))
    print("\nBEST total: filter=%s window=%gs ceiling=%.2fHz -> %.1f%%" %
          (best[0], best[1], best[2], acc(pairs[best])))
    print("  per-category: " + "  ".join("%s=%.0f%%" % (CAT_LABELS[c], cat_acc(pairs[best], c)) for c in range(4)))


if __name__ == "__main__":
    main()
