#!/usr/bin/env python
"""
bw_detector_eval.py — score the AUTOMATIC BW peak detector against HAND-PICKED peaks.

Reads the ground-truth peaks you saved from bw_peak_picker.py (bw_handpicks/*.json),
runs an automatic detector on the SAME channel/band, and reports how well the auto
peaks match your hand-picks: recall (of your peaks, how many were found), precision
(of the auto peaks, how many are real), and F1 — matched within +/- TOL seconds.

With --sweep it grid-searches prominence x min-distance and prints the F1 table, so
you can see which detector settings best reproduce your hand-picks. Nothing here
touches the pipeline; the detector is a local scipy.find_peaks with the same
band-pass as the picker.

Usage
-----
    py bw_detector_eval.py                          # all saved hand-picks, current defaults
    py bw_detector_eval.py --rec-id Exp3/001        # one recording
    py bw_detector_eval.py --sweep                  # grid-search prom x dist, per file
    py bw_detector_eval.py --prom 0.01 --dist 1.0 --hp 0.4   # try one explicit config
"""

import argparse
import glob
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
from respiration_rr.settings import PPG

import tempfile
# hand-picks may live in the script dir, CWD, or temp (bw_peak_picker falls back if OneDrive
# blocks new-dir creation) — search all of them.
HANDPICK_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "bw_handpicks"),
                 os.path.join(os.getcwd(), "bw_handpicks"),
                 os.path.join(tempfile.gettempdir(), "bw_handpicks")]
TOL_SEC = 0.5                 # a hand-pick and an auto peak match if within +/- this
LOCAL_WIN_SEC = 4.0          # find_peaks wlen (local prominence window); None = global


def _highpass(y, fs, cutoff, order=2):
    y = np.asarray(y, float)
    if y.size < 9 or cutoff <= 0:
        return y - (y.mean() if y.size else 0.0)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, y)
    except Exception:
        return y - y.mean()


def detect_peaks(tr, fs, prom, dist_s, wlen_sec=LOCAL_WIN_SEC):
    """Non-greedy peaks: prom*robust-range floor, min-distance dist_s, local prominence
    window wlen_sec. Returns sample indices into tr."""
    from scipy.signal import find_peaks
    tr = np.asarray(tr, float)
    if tr.size < 3:
        return np.zeros(0, int)
    rng = float(np.percentile(tr, 95) - np.percentile(tr, 5)) or 1.0
    dist = max(1, int(round(fs * dist_s)))
    wlen = max(3, int(round(fs * wlen_sec))) if wlen_sec else None
    pk, _ = find_peaks(tr, prominence=prom * rng, distance=dist, wlen=wlen)
    return np.asarray(pk, int)


def match(truth_t, auto_t, tol):
    """Greedy nearest match within tol. Returns (tp, fp, fn, matched_pairs)."""
    truth = np.sort(np.asarray(truth_t, float))
    auto = np.sort(np.asarray(auto_t, float))
    used = np.zeros(auto.size, bool)
    tp = 0
    for tt in truth:
        if auto.size == 0:
            break
        # nearest UNUSED auto peak within tol
        order = np.argsort(np.abs(auto - tt))
        j = next((k for k in order if not used[k] and abs(auto[k] - tt) <= tol), None)
        if j is not None:
            used[j] = True
            tp += 1
    fn = truth.size - tp
    fp = int((~used).sum())
    return tp, fp, fn


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _tallest_localmax(t, tr, t0, t1):
    """Time of the tallest local maximum of tr strictly within (t0, t1), or None."""
    m = (t > t0) & (t < t1)
    if m.sum() < 3:
        return None
    x, y = t[m], tr[m]
    loc = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    if loc.size == 0:
        return None
    return float(x[loc[np.argmax(y[loc])]])


def rate_gate(times, t, tr, imin, imax):
    """Rate-prior spacing gate on peak TIMES (a fixed 30-60 bpm range -> [imin, imax] s).
    REMOVE peaks closer than imin (keep the taller on tr); ADD the tallest local max into any
    gap longer than imax. Returns gated, sorted times."""
    pk = np.sort(np.asarray(times, float))
    if pk.size == 0:
        return pk
    val = lambda tt: float(np.interp(tt, t, tr))
    kept = [pk[0]]                                          # REMOVE too-close, keep the taller
    for x in pk[1:]:
        if x - kept[-1] < imin:
            if val(x) > val(kept[-1]):
                kept[-1] = x
        else:
            kept.append(x)
    out = [kept[0]]                                         # ADD into gaps longer than imax
    for x in kept[1:]:
        a = out[-1]; guard = 0
        while x - a > imax and guard < 10:
            cand = _tallest_localmax(t, tr, a + imin * 0.5, x - imin * 0.5)
            if cand is None or cand - a < imin or x - cand < imin:
                break
            out.append(cand); a = cand; guard += 1
        out.append(x)
    return np.array(sorted(out), float)


def _load_channel(rec_id, channel, data_root):
    """Return (tr_raw_channel, t, fs) for the recording/channel via the pipeline loaders."""
    from main import run_ppg
    recs = [r for r in S.discover(data_root) if r[0] == rec_id]
    if not recs:
        raise SystemExit(f"recording {rec_id} not found")
    _, _, csv = recs[0]
    results, sig = run_ppg(csv)
    if channel not in results:
        raise SystemExit(f"channel {channel} not in {list(results)}")
    return np.asarray(sig.channels[channel], float), np.asarray(sig.time, float), sig.fs


def _bandpass(raw, fs, lo, hi, hp):
    from respiration_rr.ppg.dsp import bandpass_filter
    tr = bandpass_filter(raw, fs, lo, hi, PPG.bw_filter_order)["filtered"]
    return _highpass(tr, fs, hp) if hp else tr


def evaluate_file(path, args):
    hp_data = json.load(open(path))
    rec_id = hp_data["rec_id"]; channel = hp_data["channel"]
    truth_watch = np.asarray(hp_data["peaks_watch"], float)
    lo, hi = args.band if args.band else hp_data.get("band", [PPG.bw_band_low_hz, PPG.bw_band_high_hz])
    if truth_watch.size < 2:
        print(f"  {rec_id}/{channel}: <2 hand-picks, skipping"); return None

    raw, t, fs = _load_channel(rec_id, channel, args.data_root)
    tr = _bandpass(raw, fs, lo, hi, args.hp)
    truth_t = truth_watch                                   # both auto & truth on the watch clock
    # ONLY evaluate inside the marked span (auto peaks outside it have no ground truth). Either the
    # explicit --win (REMbo clock, as seen on the picker x-axis) or the hand-picks' own span.
    if args.win:
        off = hp_data.get("offset", 0.0)
        w0, w1 = args.win[0] - off, args.win[1] - off      # REMbo -> watch clock
        truth_t = truth_t[(truth_t >= w0) & (truth_t <= w1)]
        if truth_t.size < 2:
            print(f"  {rec_id}/{channel}: <2 hand-picks inside --win, skipping"); return None
    else:
        w0, w1 = truth_t.min() - args.tol, truth_t.max() + args.tol

    def auto_in_window(prom, dist):
        at = t[detect_peaks(tr, fs, prom, dist, args.wlen)]
        if args.gate:
            at = rate_gate(at, t, tr, 60.0 / args.rate_hi, 60.0 / args.rate_lo)
        return at[(at >= w0) & (at <= w1)]

    def auto_win(prom, dist, wlen):
        at = t[detect_peaks(tr, fs, prom, dist, wlen)]
        if args.gate:
            at = rate_gate(at, t, tr, 60.0 / args.rate_hi, 60.0 / args.rate_lo)
        return at[(at >= w0) & (at <= w1)]

    if args.sweep:
        # dist barely matters here; sweep prominence x local-window (wlen), the real levers
        print(f"\n=== {rec_id}/{channel}  band {lo}-{hi} Hz  hp={args.hp}  dist={args.dist}s  "
              f"window {w0:.1f}-{w1:.1f}s  (F1, tol {args.tol}s) ===")
        prom_grid = [0.0, 0.001, 0.003, 0.005, 0.01]
        wlen_grid = [1.5, 2.0, 3.0, 4.0, 6.0]
        print("prom\\wlen  " + "  ".join(f"{w:>5.1f}" for w in wlen_grid))
        best = (None, -1)
        for p in prom_grid:
            row = [f"{p:>6.3f}   "]
            for wl in wlen_grid:
                _, _, f1 = prf(*match(truth_t, auto_win(p, args.dist, wl), args.tol))
                row.append(f"{f1:>5.2f}")
                if f1 > best[1]:
                    best = ((p, wl), f1)
            print("  ".join(row))
        print(f"  best: prom={best[0][0]} wlen={best[0][1]}s (dist={args.dist}s) -> F1={best[1]:.2f}")
        return None

    auto_t = auto_in_window(args.prom, args.dist)
    tp, fp, fn = match(truth_t, auto_t, args.tol)
    prec, rec, f1 = prf(tp, fp, fn)
    print(f"  {rec_id}/{channel:8}  window {w0:.0f}-{w1:.0f}s (watch)  hand={truth_t.size:3d} "
          f"auto={auto_t.size:3d} | TP={tp:3d} FP={fp:3d} FN={fn:3d} | P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from main import run_reference
        from respiration_rr.compare.compare import reference_rr_series
        off = hp_data.get("offset", 0.0)
        edf = next(r[1] for r in S.discover(args.data_root) if r[0] == rec_id)
        ref_t, ref_r = reference_rr_series(run_reference(edf))

        def rr_from(pk_t):                                   # per-breath RR (bpm) on REMbo clock
            pk = np.sort(np.asarray(pk_t, float))
            if pk.size < 2:
                return np.zeros(0), np.zeros(0)
            d = np.diff(pk); ok = d > 1e-6
            return ((pk[:-1] + pk[1:]) / 2 + off)[ok], (60.0 / d)[ok]

        fig, (axT, axR) = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        show = (t >= w0 - 2) & (t <= w1 + 2)
        axT.plot(t[show] + off, tr[show], "-", color="#1d4ed8", lw=0.9,
                 label=f"Artifact BW {lo}-{hi} Hz, detrend hp={args.hp}")
        if auto_t.size:
            axT.plot(auto_t + off, np.interp(auto_t, t, tr), "v", ms=8, color="#16a34a",
                     label=f"auto ({auto_t.size})")
        if truth_t.size:
            axT.plot(truth_t + off, np.interp(truth_t, t, tr), "o", ms=11, mfc="none",
                     mec="#dc2626", mew=1.8, label=f"hand ({truth_t.size})")
        axT.set_title(f"{rec_id}/{channel}  prom={args.prom} dist={args.dist}s hp={args.hp}  "
                      f"| P={prec:.2f} R={rec:.2f} F1={f1:.2f}  (green on every red = ideal)", fontsize=10)
        axT.set_ylabel("BW (detrended)"); axT.legend(fontsize=8, loc="upper right")
        axT.grid(True, alpha=0.2)

        # bottom: RR (per-breath) from auto peaks vs reference
        axR.plot(ref_t, ref_r, "-", color="#0f172a", lw=1.6, label="reference RR")
        art, arr = rr_from(auto_t)
        if art.size:
            axR.plot(art, arr, ".-", color="#16a34a", ms=5, lw=1.0, label="RR from auto peaks")
        hrt, hrr = rr_from(truth_t)
        if hrt.size:
            axR.plot(hrt, hrr, ".-", color="#dc2626", ms=4, lw=0.6, alpha=0.4, label="RR from hand")
        axR.set_xlim((w0 + off), (w1 + off))
        axR.set_title("RR per breath vs reference", fontsize=10)
        axR.set_xlabel("time (s, REMbo clock)"); axR.set_ylabel("RR (bpm)")
        axR.legend(fontsize=8, loc="upper right"); axR.grid(True, alpha=0.2)
        fig.tight_layout()
        plt.show()
    return dict(tp=tp, fp=fp, fn=fn)


def view_only(args):
    """Detect-only view (no hand-picks): run the detector on a channel and plot the detrended
    trace + auto peaks and the per-breath RR vs reference. For checking what the detector finds."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import reference_rr_series

    ch = args.channel or "Artifact"
    recs = [r for r in S.discover(args.data_root) if r[0] == args.rec_id]
    if not recs:
        sys.exit(f"recording {args.rec_id} not found")
    rid, edf, csv = recs[0]
    results, sig = run_ppg(csv)
    if ch not in results:
        sys.exit(f"channel {ch} not in {list(results)}")
    raw = np.asarray(sig.channels[ch], float)
    t = np.asarray(sig.time, float); fs = sig.fs
    offset = run_sync(edf, results, sig)
    ref_t, ref_r = reference_rr_series(run_reference(edf))

    lo, hi = args.band if args.band else (PPG.bw_band_low_hz, PPG.bw_band_high_hz)
    tr = _bandpass(raw, fs, lo, hi, args.hp)
    auto_t = t[detect_peaks(tr, fs, args.prom, args.dist, args.wlen)]
    if args.gate:
        auto_t = rate_gate(auto_t, t, tr, 60.0 / args.rate_hi, 60.0 / args.rate_lo)
    w0, w1 = (args.win[0] - offset, args.win[1] - offset) if args.win else (t[0], t[-1])
    auto_t = auto_t[(auto_t >= w0) & (auto_t <= w1)]
    print(f"  {rid}/{ch}  band {lo}-{hi} Hz hp={args.hp} prom={args.prom} dist={args.dist}s "
          f"-> {auto_t.size} auto peaks in {w0+offset:.0f}-{w1+offset:.0f}s")

    def rr_from(pk_t):
        pk = np.sort(np.asarray(pk_t, float))
        if pk.size < 2:
            return np.zeros(0), np.zeros(0)
        d = np.diff(pk); ok = d > 1e-6
        return ((pk[:-1] + pk[1:]) / 2 + offset)[ok], (60.0 / d)[ok]

    fig, (axT, axR) = plt.subplots(2, 1, figsize=(15, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    show = (t >= w0) & (t <= w1)
    axT.plot(t[show] + offset, tr[show], "-", color="#1d4ed8", lw=0.9,
             label=f"{ch} BW {lo}-{hi} Hz, detrend hp={args.hp}")
    if auto_t.size:
        axT.plot(auto_t + offset, np.interp(auto_t, t, tr), "v", ms=7, color="#16a34a",
                 label=f"auto ({auto_t.size})")
    axT.set_title(f"{rid}/{ch}  prom={args.prom} dist={args.dist}s hp={args.hp}  (detect-only view)",
                  fontsize=10)
    axT.set_ylabel("BW (detrended)"); axT.legend(fontsize=8, loc="upper right"); axT.grid(True, alpha=0.2)

    axR.plot(ref_t, ref_r, "-", color="#0f172a", lw=1.6, label="reference RR")
    art, arr = rr_from(auto_t)
    if art.size:
        axR.plot(art, arr, ".-", color="#16a34a", ms=5, lw=1.0, label="RR from auto peaks")
    axR.set_xlim(w0 + offset, w1 + offset)
    axR.set_title("RR per breath vs reference", fontsize=10)
    axR.set_xlabel("time (s, REMbo clock)"); axR.set_ylabel("RR (bpm)")
    axR.legend(fontsize=8, loc="upper right"); axR.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the auto BW detector vs hand-picked peaks")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None, help="one recording (default: all saved hand-picks)")
    ap.add_argument("--channel", default=None, help="filter to this channel")
    ap.add_argument("--band", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
                    help="override band (default = the band saved with the hand-picks)")
    ap.add_argument("--hp", type=float, default=None, help="detrend high-pass cutoff (Hz)")
    ap.add_argument("--prom", type=float, default=0.01, help="prominence floor (fraction of range)")
    ap.add_argument("--dist", type=float, default=1.0, help="min seconds between peaks")
    ap.add_argument("--wlen", type=float, default=LOCAL_WIN_SEC,
                    help=f"local-prominence window (s); smaller catches breaths on a shoulder "
                         f"(default {LOCAL_WIN_SEC}; 0 = global)")
    ap.add_argument("--sweep", action="store_true", help="grid-search prom x dist and print F1 table")
    ap.add_argument("--plot", action="store_true",
                    help="show the detrended trace with auto (v) and hand (o) peaks — see it work")
    ap.add_argument("--view", action="store_true",
                    help="detect-only view (no hand-picks needed): plot auto peaks + RR vs reference")
    ap.add_argument("--gate", action="store_true",
                    help="apply the rate gate: remove peaks faster than --rate-hi, fill gaps slower than --rate-lo")
    ap.add_argument("--rate-lo", type=float, default=30.0, help="slow bound bpm (gap-fill), default 30")
    ap.add_argument("--rate-hi", type=float, default=60.0, help="fast bound bpm (remove), default 60")
    ap.add_argument("--win", nargs=2, type=float, default=None, metavar=("T0", "T1"),
                    help="restrict evaluation to this REMbo-clock window (s), e.g. the ~1 min you marked")
    ap.add_argument("--tol", type=float, default=TOL_SEC,
                    help=f"match tolerance (s) — hand-picks are approximate; default {TOL_SEC}")
    args = ap.parse_args(argv)

    if args.view:
        if not args.rec_id:
            sys.exit("--view needs --rec-id")
        view_only(args)
        return

    seen, files = set(), []
    for d in HANDPICK_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            b = os.path.basename(f)
            if b not in seen:                                # script dir wins over CWD/temp
                seen.add(b); files.append(f)
    if args.rec_id:
        stem = args.rec_id.replace("/", "_")
        files = [f for f in files if os.path.basename(f).startswith(stem)]
    if args.channel:
        files = [f for f in files if f"_{args.channel}.json" in os.path.basename(f)]
    if not files:
        sys.exit("no hand-pick files found in " + " | ".join(HANDPICK_DIRS)
                 + "\n(save some with 's' in bw_peak_picker.py)")

    print(f"config: prom={args.prom} dist={args.dist}s hp={args.hp} tol={args.tol}s")
    tot = dict(tp=0, fp=0, fn=0)
    for f in files:
        r = evaluate_file(f, args)
        if r:
            for k in tot:
                tot[k] += r[k]
    if not args.sweep and (tot["tp"] + tot["fp"] + tot["fn"]):
        prec, rec, f1 = prf(tot["tp"], tot["fp"], tot["fn"])
        print(f"\nAGGREGATE  TP={tot['tp']} FP={tot['fp']} FN={tot['fn']} "
              f"| P={prec:.2f} R={rec:.2f} F1={f1:.2f}")


if __name__ == "__main__":
    main()
