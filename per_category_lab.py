#!/usr/bin/env python
"""
per_category_lab.py — per-category BBBRR analysis lab (spline & ssp), read-only.

Compares the EXISTING per-beat analysis (global settings) against a PER-CATEGORY
variant of the SAME analysis, for the two spline-family methods (spline, ssp) on
the three per-beat params (RSA/RIIV/AUC), across all channels. The goal is to see
whether category-specific settings improve breath-by-breath RR (BBBRR).

Nothing here touches the pipeline. Category-specific settings are LOCAL copies via
`dataclasses.replace(PPG, ...)` (the shared PPG singleton is never mutated), and the
production envelope builders `_make_param_spline` / `_make_param_ssp` are called
directly — same faithful, non-invasive pattern as bw_bp_sweep.

Routing: the FINAL reference split (ref_split_viewer._segment_by_crossing — RR-bin
causal hysteresis, HR bin from the per-run median, min-subseg 8s). Each breath's
category = the run it falls in. Routing is by the REFERENCE for now (oracle); the
own-signal source swaps in later without changing this tool.

Two views (live matplotlib, one pair of figures per channel):
  Stage A — envelope + peak detection, global vs per-category (see the process).
  Stage B — final RR vs reference, global vs per-category, with per-category MAE.

Per-category settings live in PER_CATEGORY_OVERRIDES below, keyed by RR bin. Empty
{} => that bin uses the global PPG, so with everything empty the per-category run
reproduces the global run exactly (plumbing sanity check). Change one bin's knob,
re-run, compare.

Usage
-----
    py per_category_lab.py --rec-id Exp2/002
    py per_category_lab.py --rec-id Exp2/004 --channels Artifact IR
    py per_category_lab.py --rec-id Exp2/002 --methods spline
    py per_category_lab.py --rec-id Exp2/002 --no-show --save DIR
"""

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import bw_bp_sweep as S
import bw_category_map as CM
import ref_split_viewer as V

from respiration_rr.settings import PPG

# ----------------------------------------------------------------------------
# PER-CATEGORY OVERRIDES — keyed by RR bin (0 = <15 slow, 1 = 15-30 middle, 2 = 30+ fast).
# Empty {} => use global PPG for that bin. Each override replaces a settings.PPG field on a
# LOCAL copy (dataclasses.replace) — the shared PPG is never mutated. These affect the SPLINE
# and SSP envelope + peak detection (RSA/RIIV/AUC); BW has its own BW_CAT_BAND.
# ----------------------------------------------------------------------------
PER_CATEGORY_OVERRIDES = {
    # ssp SMOOTHING LEVEL per segment: cutoff just above the bin's max respiration
    # frequency (12 bpm=0.2 Hz, 30 bpm=0.5 Hz) — smooths faster jitter, keeps breathing.
    0: {"rr_ssp_cutoff_hz": 0.30},   # RR < 15  (slow): smooth harder
    1: {"rr_ssp_cutoff_hz": 0.55,    # RR 15-22  (mid-low; less smoothing: 0.50 -> 0.55)
        "rr_local_prom_frac": 0.08}, #           lower peak-detection floor (catch shallow mid beats); rate-gate rejects false peaks
    2: {"rr_ssp_cutoff_hz": 0.55,    # RR 22-30  (mid-high): SAME settings as bin 1 — differs ONLY in rate-gate spacing
        "rr_local_prom_frac": 0.08},
    3: {"rr_ssp_lam": 0.0,             # RR 30+  (fast): lam=0 -> interpolating spline, NO smoothing
        "rr_local_prom_frac": 0.04,    #           more sensitive peak detection (subtle fast breaths)
        "rr_local_prom_win_sec": 2.0}, #           tighter local window (~1 fast-breath period)
}

# Per-category BAND-PASS on the SPLINE envelope only. Empty => plain spline (reverted).
PER_CATEGORY_SPLINE_BP = {}

# Categorization RR-bin edges for THIS lab only (bw_category_map.CM.RR_EDGES stays [12,30]
# and is untouched). Override-dict keys map 0=<edge0, 1..=mids, last=>=edge[-1].
# Middle split at 22: bins 1 (15-22) and 2 (22-30) share settings, differ ONLY in rate-gate spacing.
RR_EDGES = [15.0, 22.0, 30.0]      # <15 slow / 15-22 mid-low / 22-30 mid-high / 30+ fast

# ---- Rate-gate (THIRD graph) — use the category as a RATE prior on peak SPACING ----
# For each breath, its category gives a rate range -> an inter-breath INTERVAL range.
# A peak spaced SHORTER than the fast bound = spurious -> REMOVE (drop the less prominent
# of the too-close pair). A gap LONGER than the slow bound = a missed breath -> ADD the
# deepest real local extremum inside the gap (never fabricate on a flat/noisy stretch).
# The range is WIDENED past the bin edges by RATE_GATE_MARGIN_BPM, so category errors and
# the non-uniform rate inside a bin (12 vs 24 bpm) don't over-constrain detection.
RATE_GATE_MARGIN_BPM = 2.0          # widen each bin's rate range by +/- this before gating
RATE_FLOOR_BPM = 4.0                # absolute physiological interval bounds
RATE_CEIL_BPM = 48.0
RATE_GATE = True                    # apply the per-category spacing gate to the DETREND (green) curve
RATE_GATE_LOCAL_WIN_SEC = 4.0       # window (s) for "prominence relative to surroundings" when
                                    # choosing which of two too-close peaks survives


def _bin_rate_range(rb):
    """Widened [lo, hi] bpm for RR bin rb (from RR_EDGES + margin, clamped)."""
    lo = RATE_FLOOR_BPM if rb == 0 else RR_EDGES[rb - 1]
    hi = RR_EDGES[rb] if rb < len(RR_EDGES) else RATE_CEIL_BPM
    return (max(RATE_FLOOR_BPM, lo - RATE_GATE_MARGIN_BPM),
            min(RATE_CEIL_BPM, hi + RATE_GATE_MARGIN_BPM))


# ---- Detrend (THIRD graph): remove the envelope midline before peak detection ----
# legacy_detrend (Detrend_peaks): FFT band-pass + peak-envelope detrend on the per-category
# envelope, then re-detect peaks. Centres a baseline-wandering envelope so shallow breaths
# aren't lost under a drifting midline. (The rate-gate idea above is kept but unused for now.)
DETREND_BASELINE_SEC = 5.0          # legacy_detrend envelope-gate baseline window (s)
# Envelope-distance FENCE (the exposed, scale-invariant P2P-style gate): a peak shapes the
# envelope only if its prominence >= frac * (median envelope distance), so small fast ripples
# don't fake high RR. None/0 => off. env_gate_win=0 => global median.
DETREND_ENV_GATE_FRAC = 0.3
DETREND_ENV_GATE_WIN = 0.0

# Which detrend feeds the THIRD graph:
#   "lower_env" (default) — fit a cubic spline through the envelope's local MINIMA (the
#                lower envelope) and subtract it, so each breath cycle stands up from ~0;
#                breaths are then the PEAKS of the detrended signal.
#   "legacy"    — legacy_detrend (FFT band-pass + peak-envelope midline + env-fence).
DETREND_METHOD = "hp_percat"        # "hp_percat" | "lower_env" | "legacy"
SHOW_GLOBAL = False                 # per-beat: show the whole-recording global curve
SHOW_PERCAT = False                 # per-beat: show the per-category (red) curve
SHOW_DETREND = True                 # per-beat: show the per-category detrend (valleys) curve
SHOW_DETREND_PEAKS = False          # per-beat: show the detrend+MAXIMA test curve
# Rate-aware per-category HIGH-PASS cutoff (Hz) for DETREND_METHOD="hp_percat": remove the
# baseline BELOW each bin's respiration band (so the slow swing goes, breathing survives).
# Set safely below the bin's MIN breathing freq (slow can be ~6 bpm=0.1 Hz -> 0.06).
DETREND_HP_CUTOFF = {0: None, 1: 0.15, 2: 0.15, 3: 0.30}   # <15 (no detrend) / 15-22 / 22-30 / 30+

# ---- BW (4th param): band-pass the RAW channel per category, then find peaks ----
# Each RR bin -> a respiration band (Hz) with a little OVERLAP into the neighbours, so a
# breath near a bin edge (or a mis-categorised segment) is not clipped. Peak detection is
# GLOBAL for now (PPG.bw_prominence etc.); per-category peak tuning comes later.
BW_CAT_BAND = {0: (0.08, 0.30), 1: (0.20, 0.55), 2: (0.20, 0.55), 3: (0.45, 0.90)}   # <15 / 15-22 / 22-30 / 30+
# Extra BW curve: same per-category band, but ADAPTIVE LOCAL peak detection added on top of
# the global floor (threshold = max(global, BW_LOCAL_PROM_FRAC x local range in +/-win s)).
BW_SHOW_LOCAL = True
BW_LOCAL_PROM_FRAC = 0.20
BW_LOCAL_PROM_WIN_SEC = 4.0
BW_SHOW_GLOBAL_BAND = True          # BW: show the global-band (0.1-1.0) curves
BW_SHOW_PERCAT_BAND = False         # BW: show the per-category-band curves (off for now)

# ---- Artifact BW, HIGH-rate bin ONLY: extra validated curve (added, not replacing) ----
# Chain validated in bw_detector_eval on the high segment: band 0.1-1.0 -> detrend hp -> non-greedy
# find_peaks (robust P95-P5 range) -> 30-60 bpm rate-gate. Drawn only for BW_HI_CHANNEL in the
# high bin's spans; every other bin/channel keeps the global/per-cat BW above.
BW_HI_CHANNEL = "Artifact"
BW_HI_BAND = (0.1, 1.0)
BW_HI_HP = 0.3                       # detrend high-pass (Hz)
BW_HI_PROM = 0.01                   # find_peaks prominence (fraction of robust range)
BW_HI_DIST = 0.6                    # min seconds between peaks
BW_HI_WLEN = 4.0                   # local-prominence window (s)
BW_HI_RATE_LO = 30.0               # rate-gate slow bound (bpm) -> gap-fill
BW_HI_RATE_HI = 60.0               # rate-gate fast bound (bpm) -> remove too-close


def _highpass(y, fs, cutoff, order=2):
    """Zero-phase Butterworth high-pass (remove baseline below `cutoff` Hz)."""
    y = np.asarray(y, float)
    if y.size < 9 or cutoff <= 0:
        return y - (y.mean() if y.size else 0.0)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, y)
    except Exception:
        return y - y.mean()


def _detrend_hp_percat(env_x, env_y, offset, runs):
    """Rate-aware detrend: high-pass the envelope with EACH bin's cutoff and keep the
    result inside that bin's spans (stitched). Removes the slow swing per category
    without touching the breathing band."""
    idx_of, rb_of = _run_indexer(runs)
    det = np.asarray(env_y, float).copy()
    ridx = idx_of(np.asarray(env_x) + offset)
    for rb in sorted(set(rb_of.tolist())):
        c = DETREND_HP_CUTOFF.get(int(rb))
        if c is None:
            continue
        hp = _highpass(env_y, PPG.rr_resample_fs, c)
        m = ridx >= 0
        m[m] = rb_of[ridx[m]] == rb
        det[m] = hp[m]
    return det


def _detrend_lower_env(x, y):
    """Subtract a cubic spline through the local MINIMA (lower envelope). Edges are
    padded flat (first/last minimum) so the spline doesn't wildly extrapolate."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if y.size < 4:
        return y - (y.min() if y.size else 0.0)
    mins = np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1
    if mins.size < 2:
        return y - y.min()
    from scipy.interpolate import CubicSpline
    xi = np.concatenate(([x[0]], x[mins], [x[-1]]))
    yi = np.concatenate(([y[mins[0]]], y[mins], [y[mins[-1]]]))
    keep = np.concatenate(([True], np.diff(xi) > 0))     # strictly increasing x
    base = CubicSpline(xi[keep], yi[keep])(x)
    return y - base

SUFFIX = {"spline": "_spline", "ssp": "_ssp"}
BASES = ("RSA", "RIIV", "AUC")
MIN_SUBSEG_SEC = 8.0                       # matches the final split decision
RR_BAND_COLORS = ["#dbeafe", "#dcfce7", "#fef9c3", "#fee2e2"]   # RR bin backgrounds (<15 / 15-22 / 22-30 / 30+)


def _overrides(rb):
    return PER_CATEGORY_OVERRIDES.get(int(rb), {})


def _build_spline_bp(base, sx, sy, cfg, mr, band):
    """Spline param with an optional band-pass on the ENVELOPE (per-category BP).

    Same as _make_param_spline, but when `band` is given the cubic-spline envelope is
    band-passed to (low, high) Hz before breath-start detection — the category's
    frequency prior. band=None reproduces the plain spline exactly."""
    from respiration_rr.ppg.respiration import (cubic_spline_rr_series,
                                                _breath_starts_envelope,
                                                _rr_from_starts, _rr_gate_args, RRParam)
    env_x, env_y = cubic_spline_rr_series(sx, sy, cfg.rr_resample_fs)
    if band is not None and np.size(env_y):
        from respiration_rr.ppg.dsp import bandpass_filter
        env_y = bandpass_filter(np.asarray(env_y, float), cfg.rr_resample_fs,
                                band[0], band[1], cfg.rr_filter_order)["filtered"]
    val = getattr(cfg, "rr_spline_use_valleys", False)
    prom = cfg.rr_spline_prominence
    if prom is None:
        prom = cfg.breath_start_prominence
    if np.size(env_y):
        detect_on = -env_y if val else env_y
        bs = _breath_starts_envelope(env_x, detect_on, prom, cfg=cfg, move_regions=mr)
    else:
        bs = np.array([], int)
    bs_t = env_x[bs] if np.size(bs) else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs_t, *_rr_gate_args(cfg, mr))
    return RRParam(base + "_spline", series_x=np.asarray(sx), series_y=np.asarray(sy),
                   env_x=env_x, env_y=env_y, bs_times=bs_t, rr_time=rr_t, rr_bpm=rr)


def _build(base, method, sx, sy, cfg, mr, rb=None):
    """Rebuild one spline-family param from its per-beat source with a given cfg.

    For the spline method, a per-category band-pass (PER_CATEGORY_SPLINE_BP[rb]) is
    applied to the envelope; ssp is untouched."""
    from respiration_rr.ppg.respiration import _make_param_spline, _make_param_ssp
    val = getattr(cfg, "rr_spline_use_valleys", False)
    if method == "spline":
        band = PER_CATEGORY_SPLINE_BP.get(int(rb)) if rb is not None else None
        return _build_spline_bp(base, sx, sy, cfg, mr, band)
    prom = cfg.rr_ssp_prominence                       # ssp: rr_ssp_prominence -> spline -> floor
    if prom is None:
        prom = cfg.rr_spline_prominence
    if prom is None:
        prom = cfg.breath_start_prominence
    lam = getattr(cfg, "rr_ssp_lam", None)
    cut = getattr(cfg, "rr_ssp_cutoff_hz", None)
    return _make_param_ssp(base + "_ssp", sx, sy, cfg, prom, val, lam, cut, mr)


def _run_indexer(runs):
    """Vectorised map: REMbo time -> run index (or -1 outside any run)."""
    starts = np.array([r["a0"] for r in runs], float)
    ends = np.array([r["a1"] for r in runs], float)
    rb_of = np.array([r["rb"] for r in runs], int)

    def idx_of(times):
        t = np.asarray(times, float)
        idx = np.searchsorted(starts, t, side="right") - 1
        res = np.full(t.shape, -1, int)
        ok = (idx >= 0) & (idx < len(runs))
        ok2 = ok.copy()
        ok2[ok] = t[ok] < ends[idx[ok]]
        res[ok2] = idx[ok2]
        return res

    return idx_of, rb_of


def _percat_prom(method, cfg):
    v = cfg.rr_ssp_prominence if method == "ssp" else cfg.rr_spline_prominence
    if v is None:
        v = cfg.rr_spline_prominence if method == "ssp" else None
    if v is None:
        v = cfg.breath_start_prominence
    return v


def per_category_series(base, method, sx, sy, mr, runs, offset, detrend=False, valleys=None):
    """Stitch a per-category envelope + breath-starts: for each RR bin present, build
    the FULL-recording envelope with that bin's cfg and keep only the part inside that
    bin's runs. With `detrend`, high-pass each bin's envelope (DETREND_HP_CUTOFF[rb],
    None = skip) BEFORE re-detecting peaks with that bin's own cfg. `valleys` overrides
    the breath-start polarity (None = settings default; True = minima, False = maxima) —
    forces explicit detection. RR from the stitched breath-starts. Watch clock."""
    from respiration_rr.ppg.respiration import (_breath_starts_envelope,
                                                _rr_from_starts, _rr_gate_args)
    idx_of, rb_of = _run_indexer(runs)
    val = valleys if valleys is not None else getattr(PPG, "rr_spline_use_valleys", True)
    explicit = detrend or (valleys is not None)          # re-detect explicitly (not p.bs_times)
    ex, ey, bsx = [], [], []
    for rb in sorted(set(rb_of.tolist())):
        cfg = dataclasses.replace(PPG, **_overrides(rb))
        p = _build(base, method, sx, sy, cfg, mr, rb=rb)
        env_x, env_y = np.asarray(p.env_x), np.asarray(p.env_y)
        if env_y.size == 0:
            continue
        in_rb = idx_of(env_x + offset) >= 0
        in_rb[in_rb] = rb_of[idx_of(env_x + offset)[in_rb]] == rb
        if not in_rb.any():
            continue
        if explicit:
            c = DETREND_HP_CUTOFF.get(int(rb)) if detrend else None
            disp = _highpass(env_y, PPG.rr_resample_fs, c) if (detrend and c is not None) else env_y
            detect_on = -disp if val else disp
            dbs = _breath_starts_envelope(env_x, detect_on, _percat_prom(method, cfg),
                                          cfg=cfg, move_regions=mr)
            bt = env_x[dbs] if np.size(dbs) else np.zeros(0)
        else:
            disp = env_y
            bt = np.asarray(p.bs_times)          # as the pipeline detected it
        ex.append(env_x[in_rb]); ey.append(disp[in_rb])
        if bt.size:
            kb = idx_of(bt + offset) >= 0
            kb[kb] = rb_of[idx_of(bt + offset)[kb]] == rb
            bsx.append(bt[kb])
    if ex:
        ex = np.concatenate(ex); ey = np.concatenate(ey)
        o = np.argsort(ex); ex, ey = ex[o], ey[o]
    else:
        ex, ey = np.zeros(0), np.zeros(0)
    bs = np.sort(np.concatenate(bsx)) if bsx else np.zeros(0)
    # Category-rate spacing gate: drop peaks closer than the bin's fast interval (keeping the
    # one most prominent vs its surroundings), fill gaps longer than the bin's slow interval.
    if RATE_GATE and explicit and bs.size:
        bs = _rate_gate(bs, ex, ey, offset, runs, use_valleys=val)
    rr_t, rr = _rr_from_starts(bs, *_rr_gate_args(PPG, mr))
    return ex, ey, bs, rr_t, rr


def bw_whole_band_series(raw, t, fs, mr, local_frac=None, local_win=None):
    """BW on the WHOLE global band (PPG.bw_band_low/high, 0.1-1.0) — no per-category split.
    Peak detection global by default; pass local_frac to add the adaptive local check.
    Returns (env_x, env_y, bs, rr_t, rr) on the watch clock."""
    from respiration_rr.ppg.dsp import bandpass_filter
    from respiration_rr.ppg.respiration import (_breath_starts_raw, _rr_from_starts,
                                                _rr_gate_args)
    t = np.asarray(t, np.float64); raw = np.asarray(raw, np.float64)
    tr = bandpass_filter(raw, fs, PPG.bw_band_low_hz, PPG.bw_band_high_hz,
                         PPG.bw_filter_order)["filtered"]
    bw_prom = getattr(PPG, "bw_prominence", None) or PPG.breath_start_prominence
    cfg = PPG if local_frac is None else dataclasses.replace(
        PPG, bw_local_prom_frac=local_frac,
        bw_local_prom_win_sec=(local_win if local_win is not None else PPG.bw_local_prom_win_sec))
    bs = _breath_starts_raw(tr, fs, bw_prom, t=t, cfg=cfg, move_regions=mr)
    bt = t[np.asarray(bs, int)] if np.size(bs) else np.zeros(0)
    rr_t, rr = _rr_from_starts(bt, *_rr_gate_args(PPG, mr))
    return t, tr, bt, rr_t, rr


def bw_per_category_series(raw, t, fs, mr, runs, offset, local_frac=None, local_win=None):
    """BW (4th param) per category: band-pass the RAW channel to each bin's BW_CAT_BAND
    and keep the part inside that bin's runs, then detect breath-starts. Peak detection is
    GLOBAL-floor by default; pass `local_frac` (+`local_win` s) to ALSO apply the adaptive
    local-relative check (threshold = max(global floor, local_frac x local range)).
    Returns (env_x, env_y, bs, rr_t, rr) on the watch clock (+offset -> REMbo)."""
    from respiration_rr.ppg.dsp import bandpass_filter
    from respiration_rr.ppg.respiration import (_breath_starts_raw, _rr_from_starts,
                                                _rr_gate_args)
    idx_of, rb_of = _run_indexer(runs)
    t = np.asarray(t, np.float64)
    raw = np.asarray(raw, np.float64)
    order = PPG.bw_filter_order
    bw_prom = getattr(PPG, "bw_prominence", None) or PPG.breath_start_prominence
    # cfg for detection: enable the local-relative (adaptive) check when local_frac is given
    cfg = PPG if local_frac is None else dataclasses.replace(
        PPG, bw_local_prom_frac=local_frac,
        bw_local_prom_win_sec=(local_win if local_win is not None else PPG.bw_local_prom_win_sec))
    ridx = idx_of(t + offset)
    ex, ey, bsx = [], [], []
    for rb in sorted(set(rb_of.tolist())):
        band = BW_CAT_BAND.get(int(rb))
        if band is None:
            continue
        tr = bandpass_filter(raw, fs, band[0], band[1], order)["filtered"]
        m = ridx >= 0
        m[m] = rb_of[ridx[m]] == rb
        ex.append(t[m]); ey.append(tr[m])
        bs = _breath_starts_raw(tr, fs, bw_prom, t=t, cfg=cfg, move_regions=mr)
        if np.size(bs):
            bt = t[np.asarray(bs, int)]
            kb = idx_of(bt + offset) >= 0
            kb[kb] = rb_of[idx_of(bt + offset)[kb]] == rb
            bsx.append(bt[kb])
    if ex:
        ex = np.concatenate(ex); ey = np.concatenate(ey)
        o = np.argsort(ex); ex, ey = ex[o], ey[o]
    else:
        ex, ey = np.zeros(0), np.zeros(0)
    bs = np.sort(np.concatenate(bsx)) if bsx else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs, *_rr_gate_args(PPG, mr))
    return ex, ey, bs, rr_t, rr


def _bw_hi_peaks(sig, fs):
    """Non-greedy find_peaks for the Artifact-BW-high curve (robust P95-P5 range floor)."""
    from scipy.signal import find_peaks
    sig = np.asarray(sig, np.float64)
    if sig.size < 3:
        return np.zeros(0, int)
    rng = float(np.percentile(sig, 95) - np.percentile(sig, 5)) or 1.0
    dist = max(1, int(round(fs * BW_HI_DIST)))
    wlen = max(3, int(round(fs * BW_HI_WLEN)))
    pk, _ = find_peaks(sig, prominence=BW_HI_PROM * rng, distance=dist, wlen=wlen)
    return np.asarray(pk, int)


def _bw_gate(times, t, tr, imin, imax):
    """Rate gate on peak TIMES (maxima on tr): remove peaks closer than imin s (keep the taller),
    fill gaps longer than imax s with the tallest local max. imin/imax = 60/fast, 60/slow bpm."""
    pk = np.sort(np.asarray(times, float))
    if pk.size == 0:
        return pk
    val = lambda tt: float(np.interp(tt, t, tr))
    kept = [pk[0]]
    for x in pk[1:]:
        if x - kept[-1] < imin:
            if val(x) > val(kept[-1]):
                kept[-1] = x
        else:
            kept.append(x)
    out = [kept[0]]
    for x in kept[1:]:
        a = out[-1]; guard = 0
        while x - a > imax and guard < 10:
            m = (t > a + imin * 0.5) & (t < x - imin * 0.5)
            if m.sum() < 3:
                break
            xs, ys = t[m], tr[m]
            loc = np.where((ys[1:-1] > ys[:-2]) & (ys[1:-1] > ys[2:]))[0] + 1
            if loc.size == 0:
                break
            cand = float(xs[loc[np.argmax(ys[loc])]])
            if cand - a < imin or x - cand < imin:
                break
            out.append(cand); a = cand; guard += 1
        out.append(x)
    return np.array(sorted(out), float)


def bw_artifact_high_series(raw, t, fs, mr, runs, offset):
    """Artifact BW on the HIGH bin ONLY (rb == last): band 0.1-1.0 -> detrend hp -> non-greedy
    find_peaks -> 30-60 gate; keep the part inside the high runs. Watch clock. Returns
    (env_x, env_y, bs, rr_t, rr) — empty arrays if there is no high-bin span."""
    from respiration_rr.ppg.dsp import bandpass_filter
    from respiration_rr.ppg.respiration import _rr_from_starts, _rr_gate_args
    t = np.asarray(t, np.float64); raw = np.asarray(raw, np.float64)
    idx_of, rb_of = _run_indexer(runs)
    hi_bin = len(RR_EDGES)                                  # highest bin index (30+)
    in_hi = idx_of(t + offset) >= 0
    in_hi[in_hi] = rb_of[idx_of(t + offset)[in_hi]] == hi_bin
    if not in_hi.any():
        z = np.zeros(0)
        return z, z, z, z, z
    tr = bandpass_filter(raw, fs, BW_HI_BAND[0], BW_HI_BAND[1], PPG.bw_filter_order)["filtered"]
    tr = _highpass(tr, fs, BW_HI_HP)
    bs = _bw_gate(t[_bw_hi_peaks(tr, fs)], t, tr, 60.0 / BW_HI_RATE_HI, 60.0 / BW_HI_RATE_LO)
    kb = idx_of(bs + offset) >= 0                           # keep peaks inside high spans
    kb[kb] = rb_of[idx_of(bs + offset)[kb]] == hi_bin
    bs = bs[kb]
    rr_t, rr = _rr_from_starts(bs, *_rr_gate_args(PPG, mr))
    return t[in_hi], tr[in_hi], bs, rr_t, rr


def bw_artifact_hybrid_series(raw, t, fs, mr, runs, offset):
    """WHOLE-recording Artifact BW: global/global (0.1-1.0 band-pass + pipeline peak detection)
    everywhere, EXCEPT the high bin (rb == last) where it uses the validated chain
    (detrend hp + non-greedy find_peaks + 30-60 gate). Returns a stitched trace + hybrid peaks."""
    from respiration_rr.ppg.dsp import bandpass_filter
    from respiration_rr.ppg.respiration import (_breath_starts_raw, _rr_from_starts, _rr_gate_args)
    t = np.asarray(t, np.float64); raw = np.asarray(raw, np.float64)
    idx_of, rb_of = _run_indexer(runs)
    hi_bin = len(RR_EDGES)                                  # highest bin index (30+)

    def bins_of(times):
        r = idx_of(np.asarray(times) + offset)
        out = np.full(np.shape(times), -1, int)
        ok = r >= 0
        out[ok] = rb_of[r[ok]]
        return out

    # global/global: standard whole-band trace + pipeline peak detection
    tr_g = bandpass_filter(raw, fs, PPG.bw_band_low_hz, PPG.bw_band_high_hz,
                           PPG.bw_filter_order)["filtered"]
    bw_prom = getattr(PPG, "bw_prominence", None) or PPG.breath_start_prominence
    bs_g = t[np.asarray(_breath_starts_raw(tr_g, fs, bw_prom, t=t, cfg=PPG, move_regions=mr), int)]
    # high chain: detrended band + non-greedy find_peaks + 30-60 gate
    tr_h = _highpass(bandpass_filter(raw, fs, BW_HI_BAND[0], BW_HI_BAND[1],
                                     PPG.bw_filter_order)["filtered"], fs, BW_HI_HP)
    bs_h = _bw_gate(t[_bw_hi_peaks(tr_h, fs)], t, tr_h, 60.0 / BW_HI_RATE_HI, 60.0 / BW_HI_RATE_LO)
    # stitched display trace: global band, swapped to the detrended band inside high spans
    in_hi = bins_of(t) == hi_bin
    ey = tr_g.copy(); ey[in_hi] = tr_h[in_hi]
    # hybrid peaks: high-chain peaks INSIDE high (already 30-60 gated); global peaks OUTSIDE high,
    # rate-gated PER CATEGORY (each non-high run gets its own _bin_rate_range bounds; gating is done
    # run-by-run so a gap-fill never bridges across a category boundary).
    keep = [bs_h[bins_of(bs_h) == hi_bin]] if bs_h.size else []
    for run in runs:
        rb = run["rb"]
        if rb == hi_bin:
            continue
        a0w, a1w = run["a0"] - offset, run["a1"] - offset      # REMbo span -> watch clock
        seg = bs_g[(bs_g >= a0w) & (bs_g < a1w)]
        if seg.size:
            lo_bpm, hi_bpm = _bin_rate_range(rb)
            seg = _bw_gate(seg, t, ey, 60.0 / hi_bpm, 60.0 / lo_bpm)
            seg = seg[(seg >= a0w) & (seg <= a1w)]            # keep fills inside the run
        if seg.size:
            keep.append(seg)
    bs = np.sort(np.concatenate(keep)) if keep else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs, *_rr_gate_args(PPG, mr))
    return t, ey, bs, rr_t, rr


def _deepest_extremum(env_x, env_y, t0, t1, use_valleys):
    """Time of the deepest interior local extremum (valley if use_valleys) of the
    envelope strictly within (t0, t1), or None."""
    m = (env_x > t0) & (env_x < t1)
    xs, ys = env_x[m], env_y[m]
    if xs.size < 3:
        return None
    sig = ys if use_valleys else -ys                 # minima of sig = the peaks we detect
    loc = np.where((sig[1:-1] < sig[:-2]) & (sig[1:-1] < sig[2:]))[0] + 1
    if loc.size == 0:
        return None
    return float(xs[loc[np.argmin(sig[loc])]])


def _rate_gate(bs, env_x, env_y, offset, runs, use_valleys=True):
    """Category-rate spacing gate on breath-starts (watch clock). REMOVE peaks closer
    than the category's fast interval bound (keep the more prominent), then ADD the
    deepest real extremum into any gap longer than the slow bound. Returns gated bs."""
    bs = np.sort(np.asarray(bs, float))
    if bs.size == 0 or np.size(env_x) == 0:
        return bs
    idx_of, rb_of = _run_indexer(runs)

    def score(t):                                    # higher = MORE PROMINENT vs surroundings -> keep
        # Prominence relative to the local neighbourhood (+/- RATE_GATE_LOCAL_WIN_SEC):
        # a valley's depth below the local ridge, or a peak's height above the local floor.
        # This drops shallow false peaks that barely stand out, keeping the ones that do.
        v = np.interp(t, env_x, env_y)
        w = RATE_GATE_LOCAL_WIN_SEC
        m = (env_x >= t - w) & (env_x <= t + w)
        if not m.any():
            return -v if use_valleys else v
        return float(env_y[m].max() - v) if use_valleys else float(v - env_y[m].min())

    def bounds(t):                                   # (imin, imax) seconds at time t
        ri = idx_of(np.array([t + offset]))[0]
        rb = int(rb_of[ri]) if ri >= 0 else 1
        lo, hi = _bin_rate_range(rb)
        return 60.0 / hi, 60.0 / lo

    # REMOVE too-close (greedy, keep the more prominent of the pair)
    kept = [bs[0]]
    for t in bs[1:]:
        imin, _ = bounds(t)
        if (t - kept[-1]) < imin:
            if score(t) > score(kept[-1]):
                kept[-1] = t
        else:
            kept.append(t)

    # ADD into gaps longer than the slow bound (insert deepest real extremum)
    out = [kept[0]]
    for t in kept[1:]:
        a = out[-1]
        _, imax = bounds((a + t) / 2.0)
        guard = 0
        while (t - a) > imax and guard < 8:
            cand = _deepest_extremum(env_x, env_y, a + 1e-3, t - 1e-3, use_valleys)
            if cand is None or (cand - a) < imax * 0.5 or (t - cand) < imax * 0.5:
                break
            out.append(cand); a = cand; guard += 1
        out.append(t)
    return np.array(sorted(out), float)


def _errors_by_category(rr_t_watch, rr, offset, ref_t, ref_r, runs):
    """Per-breath |RR - reference| with each breath's RR bin. Returns (err, rb) arrays
    (rb = -1 outside any run / reference span). Pool these across recordings for an
    honest aggregate (concatenate errors, not average-of-averages)."""
    if np.size(rr_t_watch) == 0:
        return np.zeros(0), np.zeros(0, int)
    idx_of, rb_of = _run_indexer(runs)
    t = np.asarray(rr_t_watch, float) + offset
    inside = (t >= ref_t[0]) & (t <= ref_t[-1])
    if not inside.any():
        return np.zeros(0), np.zeros(0, int)
    t, r = t[inside], np.asarray(rr)[inside]
    err = np.abs(r - np.interp(t, ref_t, ref_r))
    ridx = idx_of(t)
    rb = np.where(ridx >= 0, rb_of[np.clip(ridx, 0, len(runs) - 1)], -1)
    return err, rb


def _mae_by_category(rr_t_watch, rr, offset, ref_t, ref_r, runs):
    """(overall_mae, {rb: mae}, n) from the per-breath errors."""
    err, rb = _errors_by_category(rr_t_watch, rr, offset, ref_t, ref_r, runs)
    if err.size == 0:
        return np.nan, {}, 0
    per = {b: float(err[rb == b].mean()) for b in sorted(set(rb[rb >= 0].tolist()))}
    return float(err.mean()), per, int(err.size)


def _shade_runs(ax, runs, rr_labels):
    for r in runs:
        ax.axvspan(r["a0"], r["a1"], color=RR_BAND_COLORS[r["rb"]], zorder=0, alpha=0.6)


def _plot_channel(rid, ch, offset, runs, ref_t, ref_r, results, methods, bases,
                  rr_labels, plt, with_bw=False, raw=None, t=None, fs=None):
    """Two figures for one channel: Stage A (envelope+peaks) and Stage B (RR vs ref).
    with_bw adds a 4th row = the BW param (needs raw/t/fs)."""
    nrow, ncol = len(bases) + (1 if with_bw else 0), len(methods)
    figA, axesA = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.4 * nrow),
                               squeeze=False, sharex=True)
    figB, axesB = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.4 * nrow),
                               squeeze=False, sharex=True)
    mae_rows = []
    for i, base in enumerate(bases):
        for j, method in enumerate(methods):
            gp = results[ch].params.get(base + SUFFIX[method])
            axA, axB = axesA[i][j], axesB[i][j]
            _shade_runs(axA, runs, rr_labels); _shade_runs(axB, runs, rr_labels)
            if gp is None or np.size(gp.series_y) == 0:
                axA.set_title(f"{base}/{method}: n/a", fontsize=9)
                continue
            sx, sy = np.asarray(gp.series_x), np.asarray(gp.series_y)
            # global (as the pipeline produced it)
            gex, gey = np.asarray(gp.env_x) + offset, np.asarray(gp.env_y)
            gbs = np.asarray(gp.bs_times) + offset
            grt, grr = np.asarray(gp.rr_time) + offset, np.asarray(gp.rr_bpm)
            # per-category (this tool)
            mr = results[ch].move_regions
            cex, cey, cbs, crt_w, crr = per_category_series(base, method, sx, sy, mr, runs, offset)
            cex_p, cbs_p, crt_p = cex + offset, cbs + offset, crt_w + offset
            # THIRD graph (optional): per-category DETREND (rate-aware high-pass), re-detected
            # per bin with each bin's own cfg — a like-for-like comparison to the per-cat curve.
            det_x = det_y = gg_bs = gg_rt_w = gg_rr = np.zeros(0)
            if SHOW_DETREND:
                det_x, det_y, gg_bs, gg_rt_w, gg_rr = per_category_series(
                    base, method, sx, sy, mr, runs, offset, detrend=True)
            det_x_p, gg_bs_p, gg_rt_p = det_x + offset, gg_bs + offset, gg_rt_w + offset
            # TEST 4th graph: same path incl. detrend, but detect RR from PEAKS (maxima)
            pk_bs = pk_rt_w = pk_rr = np.zeros(0)
            if SHOW_DETREND_PEAKS:
                _, _, pk_bs, pk_rt_w, pk_rr = per_category_series(
                    base, method, sx, sy, mr, runs, offset, detrend=True, valleys=False)
            pk_bs_p, pk_rt_p = pk_bs + offset, pk_rt_w + offset

            # ---- Stage A: envelope + breath-starts ----
            if SHOW_GLOBAL:
                axA.plot(gex, gey, "-", color="#1d4ed8", lw=1.0, alpha=0.9, label="global env")
            if SHOW_PERCAT:
                axA.plot(cex_p, cey, "-", color="#dc2626", lw=1.0, alpha=0.7, label="per-cat env")
            # raw per-beat samples the spline is fit through (see where it invents a peak)
            axA.plot(sx + offset, sy, ".", ms=3.5, color="#111827", alpha=0.55,
                     zorder=5, label="beat samples")
            if SHOW_GLOBAL and gbs.size:
                axA.plot(gbs, np.interp(gbs, gex, gey) if gex.size else np.zeros(gbs.size),
                         "v", ms=5, color="#1d4ed8", label="global peaks")
            if SHOW_PERCAT and cbs_p.size:
                axA.plot(cbs_p, np.interp(cbs_p, cex_p, cey) if cex_p.size else np.zeros(cbs_p.size),
                         "^", ms=5, color="#dc2626", label="per-cat peaks")
            if np.size(det_y):
                axA.plot(det_x_p, det_y, "-", color="#16a34a", lw=1.0, alpha=0.8, label="detrended env")
            if gg_bs_p.size:
                axA.plot(gg_bs_p, np.interp(gg_bs_p, det_x_p, det_y) if det_x_p.size else np.zeros(gg_bs_p.size),
                         "o", ms=7, mfc="none", mec="#16a34a", mew=1.3, label="detrended peaks(valleys)")
            if SHOW_DETREND_PEAKS and pk_bs_p.size:
                axA.plot(pk_bs_p, np.interp(pk_bs_p, det_x_p, det_y) if det_x_p.size else np.zeros(pk_bs_p.size),
                         "x", ms=6, color="#a21caf", mew=1.4, label="detrend+MAXIMA")
            axA.set_title(f"{ch} · {base}/{method}", fontsize=9)
            allv = np.concatenate([v for v in (gey, cey, det_y, sy) if np.size(v)])   # robust y-limits:
            if allv.size:                                                  # ignore startup transients
                lo, hi = np.percentile(allv, [1, 99]); pad = 0.1 * (hi - lo + 1e-9)
                axA.set_ylim(lo - pad, hi + pad)
            if i == 0 and j == 0:
                axA.legend(fontsize=6, ncol=2, loc="upper right")

            # ---- Stage B: RR vs reference ----
            axB.plot(ref_t, ref_r, "-", color="#0f172a", lw=1.4, label="reference")
            if SHOW_GLOBAL:
                axB.plot(grt, grr, ".-", color="#1d4ed8", ms=3, lw=0.7, alpha=0.6, label="global")
            if SHOW_PERCAT:
                axB.plot(crt_p, crr, ".-", color="#dc2626", ms=3, lw=0.7, alpha=0.6, label="per-cat")
            g_mae, g_per, g_n = _mae_by_category(np.asarray(gp.rr_time), grr, offset, ref_t, ref_r, runs)
            c_mae, c_per, c_n = _mae_by_category(crt_w, crr, offset, ref_t, ref_r, runs)
            if SHOW_DETREND:
                axB.plot(gg_rt_p, gg_rr, ".-", color="#16a34a", ms=3, lw=0.9, alpha=0.9, label="detrended")
                gg_mae, gg_per, gg_n = _mae_by_category(gg_rt_w, gg_rr, offset, ref_t, ref_r, runs)
            else:
                gg_mae, gg_per = np.nan, {}
            if SHOW_DETREND_PEAKS:
                axB.plot(pk_rt_p, pk_rr, ".-", color="#a21caf", ms=3, lw=0.9, alpha=0.9, label="detrend+maxima")
                pk_mae, _, _ = _mae_by_category(pk_rt_w, pk_rr, offset, ref_t, ref_r, runs)
            else:
                pk_mae = np.nan
            g_str = f" g={g_mae:.2f}" if SHOW_GLOBAL else ""
            det_str = f" det={gg_mae:.2f}" if SHOW_DETREND else ""
            pk_str = f" pk={pk_mae:.2f}" if SHOW_DETREND_PEAKS else ""
            axB.set_title(f"{base}/{method}  MAE{g_str} c={c_mae:.2f}{det_str}{pk_str}", fontsize=9)
            if i == 0 and j == 0:
                axB.legend(fontsize=6, loc="upper right")
            mae_rows.append((base, method, g_mae, c_mae, gg_mae, g_per, c_per, gg_per))

    if with_bw:                                          # 4th row = BW param
        r = len(bases)
        _plot_bw_into(axesA[r][0], axesB[r][0], ch, raw, t, fs, results[ch].move_regions,
                      offset, runs, ref_t, ref_r, results, rr_labels)
        for jj in range(1, ncol):                        # blank extra columns in the BW row
            axesA[r][jj].axis("off"); axesB[r][jj].axis("off")

    figA.suptitle(f"{rid} · {ch} — Stage A: envelope + peaks (global / per-cat / detrended)",
                  fontweight="bold")
    figB.suptitle(f"{rid} · {ch} — Stage B: BBBRR vs reference (global / per-cat / detrended)",
                  fontweight="bold")
    figA.tight_layout(rect=(0, 0, 1, 0.96)); figB.tight_layout(rect=(0, 0, 1, 0.96))

    # console MAE table
    hdr = "global / per-cat / detrended" if SHOW_DETREND else "global / per-cat"
    print(f"\n  [{ch}] MAE (bpm)  {hdr}   [per RR bin: {rr_labels}]")
    for base, method, gm, cm, ggm, gper, cper, ggper in mae_rows:
        def _fmt(d):
            return " ".join(f"{rr_labels[k]}={d.get(k, float('nan')):.2f}" for k in range(len(rr_labels)))
        tail = f" / {ggm:5.2f}   det[{_fmt(ggper)}]" if SHOW_DETREND else f"   c[{_fmt(cper)}]"
        print(f"     {base:5}/{method:6}  overall {gm:5.2f} / {cm:5.2f}{tail}")
    return figA, figB


def _plot_bw_into(axA, axB, ch, raw, t, fs, mr, offset, runs, ref_t, ref_r, results, rr_labels):
    """Draw the BW (4th) param into the given axes: band (global/per-cat) x peaks
    (global/adaptive-local). axA = trace + breath-starts, axB = RR vs reference."""
    _shade_runs(axA, runs, rr_labels); _shade_runs(axB, runs, rr_labels)

    lp = results[ch].params.get("LP")                    # global BW (whole-recording band)
    gex = np.asarray(lp.env_x) + offset if lp is not None else np.zeros(0)
    gey = np.asarray(lp.env_y) if lp is not None else np.zeros(0)
    gbs = np.asarray(lp.bs_times) + offset if lp is not None else np.zeros(0)
    grt = np.asarray(lp.rr_time) + offset if lp is not None else np.zeros(0)
    grr = np.asarray(lp.rr_bpm) if lp is not None else np.zeros(0)
    cex, cey, cbs, crt_w, crr = bw_per_category_series(raw, t, fs, mr, runs, offset)
    cex_p, cbs_p, crt_p = cex + offset, cbs + offset, crt_w + offset
    # adaptive-local peak detection (same bands) — one extra curve per band
    glbs = glrt_w = glrr = np.zeros(0)     # global band + adaptive-local
    plbs = plrt_w = plrr = np.zeros(0)     # per-cat band + adaptive-local
    if BW_SHOW_LOCAL:
        _, _, glbs, glrt_w, glrr = bw_whole_band_series(raw, t, fs, mr,
            local_frac=BW_LOCAL_PROM_FRAC, local_win=BW_LOCAL_PROM_WIN_SEC)
        _, _, plbs, plrt_w, plrr = bw_per_category_series(raw, t, fs, mr, runs, offset,
            local_frac=BW_LOCAL_PROM_FRAC, local_win=BW_LOCAL_PROM_WIN_SEC)
    glbs_p, glrt_p = glbs + offset, glrt_w + offset
    plbs_p, plrt_p = plbs + offset, plrt_w + offset
    # Artifact BW hybrid — global/global everywhere, validated chain in the HIGH bin (whole recording)
    hi_ex = hi_ey = hi_bs = hi_rt_w = hi_rr = np.zeros(0)
    if ch == BW_HI_CHANNEL:
        hi_ex, hi_ey, hi_bs, hi_rt_w, hi_rr = bw_artifact_hybrid_series(raw, t, fs, mr, runs, offset)

    # ---- top: BW traces + breath-starts (global band = blue trace, per-cat band = red) ----
    if BW_SHOW_GLOBAL_BAND and gex.size:
        axA.plot(gex, gey, "-", color="#1d4ed8", lw=0.7, alpha=0.7, label="global band (0.1-1.0)")
        axA.plot(gbs, np.interp(gbs, gex, gey), "v", ms=5, color="#1d4ed8", label="global: global peaks")
        if BW_SHOW_LOCAL and glbs_p.size:
            axA.plot(glbs_p, np.interp(glbs_p, gex, gey), "s", ms=7, mfc="none", mec="#0891b2",
                     mew=1.3, label="global: adaptive-local")
    if BW_SHOW_PERCAT_BAND and cex_p.size:
        axA.plot(cex_p, cey, "-", color="#dc2626", lw=0.7, alpha=0.7, label="per-cat band")
        axA.plot(cbs_p, np.interp(cbs_p, cex_p, cey), "^", ms=5, color="#dc2626", label="per-cat: global peaks")
        if BW_SHOW_LOCAL and plbs_p.size:
            axA.plot(plbs_p, np.interp(plbs_p, cex_p, cey), "o", ms=7, mfc="none", mec="#16a34a",
                     mew=1.3, label="per-cat: adaptive-local")
    if ch == BW_HI_CHANNEL and hi_ex.size:
        axA.plot(hi_ex + offset, hi_ey, "-", color="#7c3aed", lw=0.9, alpha=0.85,
                 label="Artifact BW hybrid (global; HIGH=detrend+gate)")
        if hi_bs.size:
            axA.plot(hi_bs + offset, np.interp(hi_bs, hi_ex, hi_ey), "D", ms=6, mfc="none",
                     mec="#7c3aed", mew=1.6, label="hybrid peaks")
    allv = np.concatenate([v for v in (gey, cey, hi_ey) if np.size(v)])
    if allv.size:
        lo, hi = np.percentile(allv, [1, 99]); pad = 0.1 * (hi - lo + 1e-9)
        axA.set_ylim(lo - pad, hi + pad)
    axA.set_title(f"{ch} · BW (raw band-pass; global vs per-category band)", fontsize=10)
    axA.legend(fontsize=5.5, ncol=2, loc="upper right")

    # ---- bottom: RR vs reference ----
    axB.plot(ref_t, ref_r, "-", color="#0f172a", lw=1.4, label="reference")
    if BW_SHOW_GLOBAL_BAND and grt.size:
        axB.plot(grt, grr, ".-", color="#1d4ed8", ms=3, lw=0.6, alpha=0.6, label="global/global")
    if BW_SHOW_PERCAT_BAND:
        axB.plot(crt_p, crr, ".-", color="#dc2626", ms=3, lw=0.6, alpha=0.6, label="per-cat/global")
    g_mae, g_per, _ = _mae_by_category(np.asarray(lp.rr_time) if lp is not None else np.zeros(0),
                                       grr, offset, ref_t, ref_r, runs)
    c_mae, c_per, _ = _mae_by_category(crt_w, crr, offset, ref_t, ref_r, runs)
    gl_mae = pl_mae = np.nan
    if BW_SHOW_LOCAL and BW_SHOW_GLOBAL_BAND:
        axB.plot(glrt_p, glrr, ".-", color="#0891b2", ms=3, lw=0.9, alpha=0.9, label="global/adaptive")
        gl_mae, _, _ = _mae_by_category(glrt_w, glrr, offset, ref_t, ref_r, runs)
    if BW_SHOW_LOCAL and BW_SHOW_PERCAT_BAND:
        axB.plot(plrt_p, plrr, ".-", color="#16a34a", ms=3, lw=0.9, alpha=0.9, label="per-cat/adaptive")
        pl_mae, _, _ = _mae_by_category(plrt_w, plrr, offset, ref_t, ref_r, runs)
    hi_mae = np.nan
    if ch == BW_HI_CHANNEL and hi_rt_w.size:
        axB.plot(hi_rt_w + offset, hi_rr, ".-", color="#7c3aed", ms=4, lw=1.0, alpha=0.95,
                 label="Artifact BW hybrid")
        hi_mae, _, _ = _mae_by_category(hi_rt_w, hi_rr, offset, ref_t, ref_r, runs)
    loc_str = f"  glob+adapt={gl_mae:.2f}  cat+adapt={pl_mae:.2f}" if BW_SHOW_LOCAL else ""
    hi_str = f"  hybrid={hi_mae:.2f}" if (ch == BW_HI_CHANNEL and np.isfinite(hi_mae)) else ""
    axB.set_title(f"BW MAE  global={g_mae:.2f}  per-cat={c_mae:.2f}{loc_str}{hi_str}", fontsize=9)
    axB.legend(fontsize=5.5, ncol=2, loc="upper right")

    def _fmt(d):
        return " ".join(f"{rr_labels[k]}={d.get(k, float('nan')):.2f}" for k in range(len(rr_labels)))
    print(f"\n  [{ch}] BW MAE (bpm)  glob/glob {g_mae:.2f} / cat/glob {c_mae:.2f} / "
          f"glob/adapt {gl_mae:.2f} / cat/adapt {pl_mae:.2f}")
    return


def _plot_channel_bw(rid, ch, raw, t, fs, mr, offset, runs, ref_t, ref_r, results, rr_labels, plt):
    """Standalone BW figure (top trace+peaks, bottom RR vs ref)."""
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(13, 6.0), sharex=True)
    _plot_bw_into(axA, axB, ch, raw, t, fs, mr, offset, runs, ref_t, ref_r, results, rr_labels)
    fig.suptitle(f"{rid} · {ch} — BW (4th param): band (global/per-cat) x peaks (global/adaptive-local)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-category BBBRR lab (spline & ssp & BW)")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None, help="recording id, e.g. Exp2/002 (default: first)")
    ap.add_argument("--edf", default=None,
                    help="explicit reference EDF path (with --watch; bypasses discovery, "
                         "so a recording outside Data/Exp*/recordings data can be analysed)")
    ap.add_argument("--watch", default=None,
                    help="explicit watch CSV path (use together with --edf)")
    ap.add_argument("--all", action="store_true",
                    help="loop over ALL recordings and save PNGs (requires --save DIR)")
    ap.add_argument("--channels", nargs="+", default=None, help="default: all present")
    ap.add_argument("--methods", nargs="+", default=["spline", "ssp"],
                    choices=("spline", "ssp", "BW"),
                    help="spline/ssp = per-beat params; BW = raw-channel band-pass per category "
                         "(run BW on its own: py ... --methods BW)")
    ap.add_argument("--with-bw", action="store_true",
                    help="add the BW param as a 4th row to the spline/ssp figures "
                         "(all four params in one display; use with a single per-beat method)")
    ap.add_argument("--params", nargs="+", default=list(BASES))
    ap.add_argument("--min-subseg-sec", type=float, default=MIN_SUBSEG_SEC)
    ap.add_argument("--save", default=None, metavar="DIR")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    if args.edf and args.watch:
        recs = [(args.rec_id or "custom", args.edf, args.watch)]   # direct paths, skip discovery
    elif args.edf or args.watch:
        sys.exit("--edf and --watch must be given together.")
    else:
        recs = S.discover(args.data_root)
        if args.rec_id:
            recs = [r for r in recs if r[0] == args.rec_id]
        elif not args.all:
            recs = recs[:1]                              # default: first recording only
    if not recs:
        sys.exit("No recording found.")

    out_dir = args.save
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if args.all and not out_dir:
        sys.exit("--all needs --save DIR (would open too many windows otherwise).")
    headless = args.no_show or args.all                  # --all always saves, never shows

    import matplotlib
    matplotlib.use("Agg" if headless else "TkAgg")
    import matplotlib.pyplot as plt

    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import reference_rr_series
    from respiration_rr.rr_average import reference_average_rr

    active = {rb: ov for rb, ov in PER_CATEGORY_OVERRIDES.items() if ov}
    print(f"RR_EDGES={RR_EDGES} | methods {args.methods} | overrides: {active or 'none'}")
    bw_mode = args.methods == ["BW"]                     # BW is per-channel (not per-base)

    for rid, edf, csv in recs:
        try:
            ref = run_reference(edf)
            results, sig = run_ppg(csv)
            offset = run_sync(edf, results, sig)
            truth_at, truth_ar = reference_average_rr(ref)
            hr_t, hr_v = CM.hr_series_from_beats(results, sig, offset)
            runs = V._segment_by_crossing(truth_at, truth_ar, hr_t, hr_v, args.min_subseg_sec,
                                          RR_EDGES, CM.HR_EDGES, fill=True)
            ref_t, ref_r = reference_rr_series(ref)
        except Exception as e:
            print(f"  [ERR] {rid}: {e}"); continue
        rr_labels = CM._labels(RR_EDGES, "RR")
        channels = args.channels or [c for c in ("Green", "Red", "IR", "Yellow", "Artifact") if c in results]
        print(f"\nRecording {rid} | offset {offset:+.2f}s | {len(runs)} category runs | channels {channels}")
        for ch in channels:
            if ch not in results:
                print(f"  [skip] {ch}: not in results"); continue
            stem = f"pcl_{rid.replace('/', '_')}_{ch}"
            if bw_mode:
                fig = _plot_channel_bw(rid, ch, sig.channels[ch], sig.time, sig.fs,
                                       results[ch].move_regions, offset, runs, ref_t, ref_r,
                                       results, rr_labels, plt)
                figs = [fig]
                if out_dir:
                    fig.savefig(os.path.join(out_dir, stem + "_BW.png"), dpi=110, bbox_inches="tight")
            else:
                figA, figB = _plot_channel(rid, ch, offset, runs, ref_t, ref_r, results,
                                           args.methods, args.params, rr_labels, plt,
                                           with_bw=args.with_bw, raw=sig.channels[ch],
                                           t=sig.time, fs=sig.fs)
                figs = [figA, figB]
                if out_dir:
                    figA.savefig(os.path.join(out_dir, stem + "_A.png"), dpi=110, bbox_inches="tight")
                    figB.savefig(os.path.join(out_dir, stem + "_B.png"), dpi=110, bbox_inches="tight")
            if headless:
                for f in figs:
                    plt.close(f)

    if out_dir:
        print(f"\nSaved PNGs -> {out_dir}")
    if not headless:
        print("\nOpening figures — close the windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
