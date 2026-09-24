#!/usr/bin/env python
"""
filter_window_sweep.py — workstreams A + B of the categorization plan.

For the Artifact BW spectrogram, sweep (filter x window) and score CATEGORIZATION
accuracy against the reference, over all recordings. Category = per 30 s segment
majority of the ridge RR, binned into 4 cells (<15 / 15-22 / 22-30 / >30 bpm).
Target respiration 6-42 bpm -> ridge search band 0.1-0.7 Hz.

KEY: the STFT uses the EXACT window length (Hann) and zero-pads to the next power
of two only for the FFT, so 40 s and 48 s are honored (unlike the production
compute_spectrogram which forces window=fft_size=pow2 and would round them to 64).

Filters (all preserve 6-42 bpm; order 2, zero-phase):
  raw · LP-only 0.7 · BP 0.05-0.7 · BP 0.1-0.7 · BP 0.1-1.0
Windows: 24 32 40 48 64 s.  Hop fixed 5 s (main sweep).

    py filter_window_sweep.py                 # all recordings, Artifact
    py filter_window_sweep.py --limit 4       # quick
    py filter_window_sweep.py --save DIR
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import bw_bp_sweep as S

RR_EDGES = [15.0, 22.0, 30.0]          # 4 category bins
SEG_SEC = float(os.environ.get("SEG_SEC", 30.0))   # categorization segment (SEG_SEC=15 to override)
HOP_SEC = 5.0
F_LO, F_HI = 0.10, 0.80                 # ridge search band (6-48 bpm; winning ceiling)
WINDOWS = [24.0, 32.0, 40.0, 48.0, 64.0]
FILTERS = ["raw", "LP0.7", "BP0.05-0.7", "BP0.1-0.7", "BP0.1-1.0"]


def apply_filter(x, fs, name):
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, np.float64)
    if name == "raw":
        return x
    ny = fs / 2.0
    if name == "LP0.7":
        sos = butter(2, 0.7 / ny, btype="low", output="sos")
    elif name == "BP0.05-0.7":
        sos = butter(2, [0.05 / ny, 0.7 / ny], btype="band", output="sos")
    elif name == "BP0.1-0.7":
        sos = butter(2, [0.10 / ny, 0.7 / ny], btype="band", output="sos")
    elif name == "BP0.1-1.0":
        sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
    else:
        return x
    return sosfiltfilt(sos, x)


def stft_ridge(sig, fs, win_sec, hop_sec):
    """Exact-length Hann STFT, zero-pad to pow2 for the FFT; argmax ridge in band.
    Returns (times_s, rr_bpm)."""
    sig = np.asarray(sig, np.float64)
    L = int(round(win_sec * fs))
    if sig.size < L:
        return np.zeros(0), np.zeros(0)
    nfft = 1 << int(np.ceil(np.log2(L)))          # zero-pad only for the FFT
    hop = max(1, int(round(hop_sec * fs)))
    han = np.hanning(L)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    bidx = np.where((freqs >= F_LO) & (freqs <= F_HI))[0]
    starts = range(0, sig.size - L + 1, hop)
    times, rr = [], []
    for a in starts:
        seg = sig[a:a + L]
        S_ = np.abs(np.fft.rfft((seg - seg.mean()) * han, n=nfft)) ** 2
        times.append((a + L / 2) / fs)
        if bidx.size and S_[bidx].max() > 0:
            rr.append(freqs[bidx][int(np.argmax(S_[bidx]))] * 60.0)
        else:
            rr.append(np.nan)
    return np.asarray(times), np.asarray(rr)


def _cat(rr):
    return np.digitize(rr, RR_EDGES)              # 0..3


def score_segments(times, rr, ref_pt, ref_pr, offset):
    """Per segment: predicted category (majority of ridge) vs reference category.
    Reference (ground truth) = MEDIAN of all reference breaths inside the segment
    (average rate), not a center-point sample. Returns list of (ref_cat, pred_cat)."""
    if times.size == 0:
        return []
    t0, t1 = times[0], times[-1]
    out = []
    s = t0
    while s < t1:
        e = s + SEG_SEC
        m = (times >= s) & (times < e)
        rseg = rr[m]
        rseg = rseg[np.isfinite(rseg)]
        if rseg.size == 0:
            s = e; continue
        pred = int(np.bincount(_cat(rseg), minlength=4).argmax())
        rmask = (ref_pt >= s + offset) & (ref_pt < e + offset)
        if rmask.any():
            ref = float(np.median(ref_pr[rmask]))
            out.append((int(np.digitize(ref, RR_EDGES)), pred))
        s = e
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Filter x window sweep, categorization accuracy")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save", default=None, metavar="DIR")
    args = ap.parse_args(argv)

    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_rr_points

    recs = S.discover(args.data_root)
    if args.limit:
        recs = recs[:args.limit]
    print("Recordings: %d  channel=%s  band=%.2f-%.2f Hz  seg=%.0fs  hop=%.0fs  ref=segment-median\n"
          % (len(recs), args.channel, F_LO, F_HI, SEG_SEC, HOP_SEC))

    # accumulate confusion per (filter, window): list of (ref,pred) across recordings
    pairs = {(f, w): [] for f in FILTERS for w in WINDOWS}
    for rid, edf, csv in recs:
        try:
            ppg, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg, sig)
            ref_pt, ref_pr = reference_rr_points(ref)
            if args.channel not in sig.channels:
                print("  %-14s SKIP (no channel)" % rid); continue
            x = np.asarray(sig.channels[args.channel], np.float64); fs = float(sig.fs)
        except Exception as e:
            print("  %-14s SKIP: %s" % (rid, e)); continue
        for fname in FILTERS:
            xf = apply_filter(x, fs, fname)
            for w in WINDOWS:
                t, rr = stft_ridge(xf, fs, w, HOP_SEC)
                pairs[(fname, w)].extend(score_segments(t, rr, ref_pt, ref_pr, offset))
        print("  %-14s done" % rid)

    # accuracy grid (total) + worst reference-category cell
    def acc(pl):
        return 100.0 * np.mean([r == p for r, p in pl]) if pl else np.nan
    def worst_cell(pl):
        best = 999.0
        for c in range(4):
            sub = [(r, p) for r, p in pl if r == c]
            if len(sub) >= 5:
                best = min(best, 100.0 * np.mean([r == p for r, p in sub]))
        return best if best < 999 else np.nan

    print("\n=== categorization accuracy (total %) — filter x window ===")
    hdr = "  %-12s" % "" + "".join("%8s" % ("%gs" % w) for w in WINDOWS)
    print(hdr)
    grid = np.full((len(FILTERS), len(WINDOWS)), np.nan)
    for i, f in enumerate(FILTERS):
        row = "  %-12s" % f
        for j, w in enumerate(WINDOWS):
            a = acc(pairs[(f, w)]); grid[i, j] = a
            row += "%7.1f " % a
        print(row)

    print("\n=== worst-cell accuracy (min over the 4 ref categories) ===")
    print(hdr)
    gridw = np.full((len(FILTERS), len(WINDOWS)), np.nan)
    for i, f in enumerate(FILTERS):
        row = "  %-12s" % f
        for j, w in enumerate(WINDOWS):
            a = worst_cell(pairs[(f, w)]); gridw[i, j] = a
            row += "%7.1f " % a
        print(row)

    bi = np.unravel_index(np.nanargmax(grid), grid.shape)
    print("\nBEST total: %s @ %gs = %.1f%% (worst-cell %.1f%%)" %
          (FILTERS[bi[0]], WINDOWS[bi[1]], grid[bi], gridw[bi]))

    out_dir = args.save
    if out_dir:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, G, ttl in ((axes[0], grid, "total accuracy %"), (axes[1], gridw, "worst-cell accuracy %")):
            im = ax.imshow(G, aspect="auto", cmap="viridis", vmin=np.nanmin(gridw), vmax=100)
            ax.set_xticks(range(len(WINDOWS))); ax.set_xticklabels(["%gs" % w for w in WINDOWS])
            ax.set_yticks(range(len(FILTERS))); ax.set_yticklabels(FILTERS)
            ax.set_title(ttl, fontsize=11, fontweight="bold")
            for i in range(len(FILTERS)):
                for j in range(len(WINDOWS)):
                    if np.isfinite(G[i, j]):
                        ax.text(j, i, "%.0f" % G[i, j], ha="center", va="center", color="white", fontsize=9)
            fig.colorbar(im, ax=ax)
        fig.suptitle("Filter x window — categorization accuracy · %d recordings · %s" % (len(recs), args.channel),
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, "filter_window_sweep_%s.png" % args.channel)
        fig.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
        print("saved ->", png)


if __name__ == "__main__":
    main()
