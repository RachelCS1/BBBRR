#!/usr/bin/env python
"""
envelope_lab — a standalone lab for the RESPIRATION-ENVELOPE step.

Does NOT touch the shipped pipeline. It runs the real analysis (analyze_ppg) only
to obtain each parameter's raw per-beat feature series (RSA / RIIV / AUC), then
REPLACES the envelope-construction step (normally bp_rr_series' moving-average
band-pass) with a set of alternative methods, and re-runs breath-start (BS) peak
detection on each — so you can compare which envelope method yields the best RR.

The BS peak-detection prominence is adjustable from the command line so you can
sweep it and judge each method fairly.

Methods (all operate on the per-beat series resampled to 10 Hz):
  ma_bandpass  current pipeline — moving-average-cascade band-pass (baseline)
  savgol       Savitzky-Golay smoothing
  butter       Butterworth IIR band-pass (zero-phase, filtfilt)
  gaussian     Gaussian smoothing
  wavelet      wavelet soft-threshold denoise (db4)
  ssa          Singular Spectrum Analysis (leading components)
  spline       smoothing cubic spline (available, not in default set)

Lomb-Scargle is intentionally excluded: it is a spectral estimator (dominant
frequency), not a time-domain envelope with detectable peaks.

Per recording it builds, for each channel × parameter, one figure:
  row 0        the raw per-beat series (input, "before manipulation")
  rows 1..N    each method's envelope ("after manipulation") + detected BS peaks
Each method row's title reports mean RR (bpm) and the BS count.

Usage
-----
    py envelope_lab.py                                  # recording 001, default 5 methods
    py envelope_lab.py --recording "Data/Exp1/recordings data/003"
    py envelope_lab.py --watch path/to/rt_flow.csv
    py envelope_lab.py --root "Data/Exp1/recordings data"   # batch: every sub-folder
    py envelope_lab.py --channels Green --params RIIV        # narrow down
    py envelope_lab.py --methods ma_bandpass savgol ssa      # pick methods
    py envelope_lab.py --prominence 0.05                     # BS prominence (fraction of range)
    py envelope_lab.py --save figs_env --no-show             # save PNGs, no windows
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt

from respiration_rr.settings import PPG
from respiration_rr.io.csv_reader import read_watch_auto
from respiration_rr.ppg.preprocess import prepare_watch
from respiration_rr.ppg.respiration import analyze_ppg, bp_rr_series
from respiration_rr.ppg.beats import filter_peaks_by_prominence

DEFAULT_RECORDING = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data", "Exp1", "recordings data", "001")

CHANNEL_COLORS = {"Green": "#22c55e", "Red": "#ef4444",
                  "IR": "#a855f7", "Artifact": "#38bdf8"}
ENV_PARAMS = ("RSA", "RIIV", "AUC")     # the per-beat-series envelope params

# per-method tuning constants (edit here to explore; prominence is on the CLI)
SAVGOL_SEC = 1.5        # Savitzky-Golay window length (s)
SAVGOL_POLY = 3
GAUSS_SEC = 0.4         # Gaussian smoothing sigma (s)
SPLINE_S_FACTOR = 0.5   # smoothing-spline strength (× N·var)
WAVELET = "db4"
WAVELET_MAX_LEVEL = 4
SSA_WINDOW_SEC = 6.0    # SSA embedding window (s)
SSA_COMPONENTS = 4      # SSA leading components to reconstruct


# ----------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------
def _resample_grid(x, y, fs):
    """Resample an irregular per-beat series onto a uniform fs grid (matches the
    grid bp_rr_series builds), with edge-hold interpolation."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 4:
        return np.zeros(0), np.zeros(0)
    dt = 1.0 / fs
    n = max(2, int(np.floor((x[-1] - x[0]) / dt)) + 1)
    t = x[0] + np.arange(n) * dt
    return t, np.interp(t, x, y)


def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def detect_bs(env_x, env_y, prom_frac, min_dist_sec, fs):
    """Breath-start peaks on an envelope — same recipe as the shipped detector
    (respiration._breath_starts_envelope) but with a caller-set prominence.

    Returns (bs_times, rr_time, rr_bpm, mean_rr, n_bs).
    """
    env_y = np.asarray(env_y, np.float64)
    if env_y.size < 4:
        return np.zeros(0), np.zeros(0), np.zeros(0), float("nan"), 0
    min_dist = max(1, int(round(min_dist_sec * fs)))
    raw_peaks, last = [], -min_dist
    for i in range(1, env_y.size - 1):
        if env_y[i] > env_y[i - 1] and env_y[i] > env_y[i + 1] and i - last >= min_dist:
            raw_peaks.append(i)
            last = i
    if not raw_peaks:
        return np.zeros(0), np.zeros(0), np.zeros(0), float("nan"), 0
    rng = env_y.max() - env_y.min()
    kept = filter_peaks_by_prominence(env_y, raw_peaks, rng * prom_frac)
    kept = np.asarray(kept, dtype=int)
    bs_t = np.asarray(env_x)[kept] if kept.size else np.zeros(0)
    # RR = 60 / Δt between consecutive breath-starts
    if bs_t.size < 2:
        return bs_t, np.zeros(0), np.zeros(0), float("nan"), int(bs_t.size)
    dt = np.diff(bs_t)
    ok = dt > 0
    rr = 60.0 / dt[ok]
    mid = ((bs_t[:-1] + bs_t[1:]) / 2)[ok]
    mean_rr = float(np.nanmean(rr)) if rr.size else float("nan")
    return bs_t, mid, rr, mean_rr, int(bs_t.size)


# ----------------------------------------------------------------------
# envelope methods : (per-beat x, y) -> (grid_t, envelope_y)
# ----------------------------------------------------------------------
def m_ma_bandpass(x, y, fs, cfg):
    """Current pipeline envelope: moving-average-cascade band-pass (baseline)."""
    return bp_rr_series(x, y, cfg.rr_band_low_hz, cfg.rr_band_high_hz,
                        cfg.rr_filter_order, resample_fs=fs)


def m_butter(x, y, fs, cfg):
    from scipy.signal import butter, filtfilt
    t, yg = _resample_grid(x, y, fs)
    if t.size < 12:
        return t, yg
    nyq = fs / 2.0
    lo, hi = cfg.rr_band_low_hz / nyq, cfg.rr_band_high_hz / nyq
    b, a = butter(2, [lo, hi], btype="band")
    yb = filtfilt(b, a, yg - yg.mean(), padlen=min(3 * max(len(a), len(b)), t.size - 1))
    return t, yb + yg.mean()


def m_savgol(x, y, fs, cfg):
    from scipy.signal import savgol_filter
    t, yg = _resample_grid(x, y, fs)
    win = _odd(round(fs * SAVGOL_SEC))
    if t.size <= win or win <= SAVGOL_POLY:
        return t, yg
    return t, savgol_filter(yg, win, SAVGOL_POLY)


def m_gaussian(x, y, fs, cfg):
    from scipy.ndimage import gaussian_filter1d
    t, yg = _resample_grid(x, y, fs)
    if t.size < 4:
        return t, yg
    return t, gaussian_filter1d(yg, max(0.5, fs * GAUSS_SEC))


def m_spline(x, y, fs, cfg):
    from scipy.interpolate import UnivariateSpline
    t, yg = _resample_grid(x, y, fs)
    if t.size < 8:
        return t, yg
    s = yg.size * np.var(yg) * SPLINE_S_FACTOR
    try:
        spl = UnivariateSpline(t, yg, s=s, k=3)
        return t, spl(t)
    except Exception:
        return t, yg


def m_wavelet(x, y, fs, cfg):
    import pywt
    t, yg = _resample_grid(x, y, fs)
    if t.size < 8:
        return t, yg
    w = pywt.Wavelet(WAVELET)
    level = min(WAVELET_MAX_LEVEL, pywt.dwt_max_level(yg.size, w.dec_len))
    if level < 1:
        return t, yg
    coeffs = pywt.wavedec(yg, w, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthr = sigma * np.sqrt(2 * np.log(yg.size)) if sigma > 0 else 0.0
    coeffs = [coeffs[0]] + [pywt.threshold(c, uthr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, w)[:yg.size]
    return t, rec


def _hankelize(X):
    """Diagonal averaging of an L×K matrix back to a length-(L+K-1) series."""
    L, K = X.shape
    n = L + K - 1
    out = np.zeros(n)
    cnt = np.zeros(n)
    for i in range(L):
        out[i:i + K] += X[i, :]
        cnt[i:i + K] += 1
    return out / cnt


def m_ssa(x, y, fs, cfg):
    t, yg = _resample_grid(x, y, fs)
    n = yg.size
    L = min(int(round(SSA_WINDOW_SEC * fs)), n // 2)
    if L < 2:
        return t, yg
    K = n - L + 1
    traj = np.column_stack([yg[i:i + L] for i in range(K)])   # L × K
    U, S, Vt = np.linalg.svd(traj, full_matrices=False)
    r = min(SSA_COMPONENTS, S.size)
    approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
    return t, _hankelize(approx)


METHODS = {
    "ma_bandpass": ("current (MA band-pass)", m_ma_bandpass, "#94a3b8"),
    "savgol":      ("Savitzky-Golay",         m_savgol,      "#22c55e"),
    "butter":      ("Butterworth",            m_butter,      "#38bdf8"),
    "gaussian":    ("Gaussian",               m_gaussian,    "#f472b6"),
    "wavelet":     ("Wavelet (db4)",          m_wavelet,     "#f97316"),
    "ssa":         ("SSA",                    m_ssa,         "#a855f7"),
    "spline":      ("Smoothing spline",       m_spline,      "#eab308"),
}
# current MA band-pass is included by default as the baseline, plus the 5 best
DEFAULT_METHODS = ["ma_bandpass", "savgol", "butter", "gaussian", "wavelet", "ssa"]


# ----------------------------------------------------------------------
# compute every method once, then draw from the shared results
# ----------------------------------------------------------------------
def compute_method_results(series_x, series_y, methods, prom_frac, min_dist_sec, fs):
    """Run each method's envelope + BS detection once. Returns a list of dicts."""
    out = []
    for key in methods:
        label, func, col = METHODS[key]
        rec = {"key": key, "label": label, "color": col, "error": None,
               "env_x": None, "env_y": None, "bs_t": None,
               "rr_time": None, "rr_bpm": None, "mean_rr": float("nan"), "n_bs": 0}
        try:
            env_x, env_y = func(series_x, series_y, fs, PPG)
            bs_t, rr_t, rr, mean_rr, n_bs = detect_bs(env_x, env_y, prom_frac,
                                                      min_dist_sec, fs)
            rec.update(env_x=env_x, env_y=env_y, bs_t=bs_t, rr_time=rr_t,
                       rr_bpm=rr, mean_rr=mean_rr, n_bs=n_bs)
        except Exception as e:                 # keep going; note the failure
            rec["error"] = str(e)
        out.append(rec)
    return out


def plot_envelopes(channel, pname, series_x, series_y, results, prom_frac, window=None):
    """Input row + one row per method (envelope 'after manipulation' + BS peaks)."""
    ch_color = CHANNEL_COLORS.get(channel, "#333")
    n = 1 + len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{channel} — {pname} — envelope methods "
                 f"(BS prominence = {prom_frac:g} of range)",
                 fontsize=13, fontweight="bold", color=ch_color)

    ax = axes[0]
    if np.size(series_x):
        ax.plot(series_x, series_y, "o-", color="#334155", ms=2.5, lw=0.6)
    ax.set_ylabel(f"{pname}\ninput", fontsize=8)
    ax.set_title("0 · raw per-beat series (input)", fontsize=9, loc="left")
    ax.grid(True, alpha=0.15)

    for ax, r in zip(axes[1:], results):
        if r["error"] is not None:
            ax.text(0.5, 0.5, f"{r['label']}: failed ({r['error']})",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#b91c1c", fontsize=8)
            continue
        if np.size(r["env_x"]):
            ax.plot(r["env_x"], r["env_y"], color=r["color"], lw=0.9,
                    label=f"{r['label']} envelope")
        if r["bs_t"] is not None and r["bs_t"].size:
            ymark = np.interp(r["bs_t"], r["env_x"], r["env_y"])
            ax.plot(r["bs_t"], ymark, "v", color="#fbbf24", ms=6, label="BS")
        ax.set_ylabel(f"{pname}\n{r['key']}", fontsize=8)
        ax.set_title(f"{r['label']} — mean RR {r['mean_rr']:.1f} bpm, {r['n_bs']} BS",
                     fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    if window:
        axes[-1].set_xlim(*window)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_rr(channel, pname, results, prom_frac, window=None):
    """Final respiration rate (bpm) over time from each method, overlaid."""
    ch_color = CHANNEL_COLORS.get(channel, "#333")
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(f"{channel} — {pname} — final RR by envelope method "
                 f"(BS prominence = {prom_frac:g})",
                 fontsize=13, fontweight="bold", color=ch_color)
    for r in results:
        if r["rr_bpm"] is None or np.size(r["rr_bpm"]) == 0:
            continue
        ax.plot(r["rr_time"], r["rr_bpm"], "o-", ms=3, lw=0.8, color=r["color"],
                label=f"{r['label']}  (mean {r['mean_rr']:.1f} bpm, {r['n_bs']} BS)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RR (bpm)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15)
    if window:
        ax.set_xlim(*window)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def _find_watch(recording_dir):
    c = glob.glob(os.path.join(recording_dir, "rt_flow*.csv")) or \
        glob.glob(os.path.join(recording_dir, "*.csv"))
    return c[0] if c else None


def _save(fig, save_dir, stem):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, stem + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def process_recording(csv_path, args, label=""):
    """Analyze one watch CSV and build the method-comparison figures."""
    print(f"\n[envelope_lab] {label or os.path.basename(csv_path)}")
    watch = read_watch_auto(csv_path)          # real-time (rt_flow) or monitor CSV
    sig = prepare_watch(watch)
    results = analyze_ppg(sig.channels, sig.time, sig.fs, move_regions=sig.move_regions)
    fs = PPG.rr_resample_fs
    window = tuple(args.window) if args.window else None
    save_dir = None
    if args.save:
        save_dir = os.path.join(args.save, label) if label else args.save

    n_figs = 0
    for ch in args.channels:
        res = results.get(ch)
        if res is None:
            continue
        for pname in args.params:
            pr = res.params.get(pname)
            if pr is None:
                continue
            method_results = compute_method_results(
                pr.series_x, pr.series_y, args.methods,
                args.prominence, args.min_dist_sec, fs)
            fig_env = plot_envelopes(ch, pname, pr.series_x, pr.series_y,
                                     method_results, args.prominence, window=window)
            fig_rr = plot_rr(ch, pname, method_results, args.prominence, window=window)
            n_figs += 2
            # console summary — reproduces your table, per channel × param
            print(f"  {ch}/{pname}:  " + "  ".join(
                f"{r['key']}={r['mean_rr']:.1f}bpm/{r['n_bs']}BS" for r in method_results))
            if save_dir:
                _save(fig_env, save_dir, f"{ch}_{pname}_methods")
                _save(fig_rr, save_dir, f"{ch}_{pname}_rr")
    print(f"  built {n_figs} figure(s)")
    return n_figs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Envelope-method lab for the respiration BS/RR step")
    ap.add_argument("--recording", default=None, help="folder with an rt_flow*.csv")
    ap.add_argument("--watch", default=None, help="explicit watch CSV path")
    ap.add_argument("--root", default=None,
                    help="batch: process every sub-folder that holds an rt_flow*.csv")
    ap.add_argument("--channels", nargs="+", default=list(CHANNEL_COLORS))
    ap.add_argument("--params", nargs="+", default=list(ENV_PARAMS),
                    help="which envelope params to test (RSA RIIV AUC)")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                    choices=list(METHODS), metavar="M",
                    help=f"methods to compare (default: {' '.join(DEFAULT_METHODS)}; "
                         f"choices: {', '.join(METHODS)})")
    ap.add_argument("--prominence", type=float, default=PPG.breath_start_prominence,
                    help="BS peak prominence as a fraction of envelope range "
                         f"(default {PPG.breath_start_prominence})")
    ap.add_argument("--min-dist-sec", type=float, default=1.0,
                    help="minimum spacing between breath-starts (s, default 1.0)")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    default=None, help="zoom every panel to this time span (s)")
    ap.add_argument("--save", default=None, metavar="DIR",
                    help="save every figure as PNG into DIR (per-recording sub-folders under --root)")
    ap.add_argument("--no-show", action="store_true",
                    help="build (+optionally save) figures but don't open windows")
    args = ap.parse_args(argv)

    print(f"methods: {args.methods} | params: {args.params} | "
          f"prominence: {args.prominence:g} | min-dist: {args.min_dist_sec:g}s")

    if args.root:
        subdirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                         if os.path.isdir(d))
        recs = [(d, _find_watch(d)) for d in subdirs]
        recs = [(d, c) for d, c in recs if c]
        if not recs:
            ap.error(f"No rt_flow*.csv found in any sub-folder of {args.root}")
        print(f"Batch: {len(recs)} recording(s) under {args.root}")
        for d, c in recs:
            process_recording(c, args, label=os.path.basename(d))
    else:
        csv_path = args.watch
        if csv_path is None:
            rec = args.recording or DEFAULT_RECORDING
            csv_path = _find_watch(rec)
            print(f"Recording: {rec}")
        if not csv_path:
            ap.error("No watch CSV found. Pass --recording DIR, --watch FILE, or --root DIR.")
        process_recording(csv_path, args)

    if not args.no_show:
        print("\nOpening figures — close the windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
