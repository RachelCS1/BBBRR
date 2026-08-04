"""
MATLAB filter-bank + stitching BBB — port of BBB_SignalCreation.m and
BBB_peakDetection.m from the legacy Breath_by_Breath (MATLAB) pipeline.

This is a SECOND, independent breath-by-breath method, added alongside the
Python-legacy `BWlegacy` (which ports RR_run.py). It is OFF by default
(settings.PPG.bwbank_enabled) so existing outputs are unchanged until enabled.

The idea (BBB_SignalCreation.m):
  Breathing rate drifts, so instead of one compromise band-pass, run a BANK of
  narrow band-passes — one per rate range — and, per instant, RIDE the one that
  best fits. "Best" = highest energy (Eng) + closest match to the wide-band
  signal (RMS2). The chosen level is smoothed (mode over N frames) so it does
  not flicker, and at each level TRANSITION the two filter outputs are STITCHED
  at the zero-crossing of their difference (where they are equal) so no fake
  edge is created. A trend state-machine (BBB_peakDetection.m) then counts
  breaths on the resulting clean trace.

Deviations from the MATLAB original (documented, not silent):
  * Works on the channel DECIMATED to `bwbank_work_fs` (~16 Hz) instead of
    64 Hz, so the narrow band-passes are cheap. Respiration (<0.7 Hz) is far
    below the 8 Hz Nyquist, so nothing of interest is lost.
  * Narrow bands use a zero-phase Butterworth (sos + sosfiltfilt) rather than
    the order-5000 FIR `filtfilt` of the MATLAB (which would be ~78 s of taps).
    Zero-phase preserves peak timing, which is what BBB needs.
  * The MATLAB noise DETECTOR is not ported (same choice as BWlegacy): movement
    regions from the new pipeline are converted to a per-sample noise flag.
  * Signal normalization is a single median/std over the valid samples rather
    than the per-segment normalization of BBB_signalNormalization.m; the bank
    score is relative across bands within a window, so this does not change the
    level choice.

scipy is imported at module top; respiration.py imports THIS module lazily
(only when the BWbank parameter is enabled), so the core pipeline never needs
scipy on account of this file.
"""

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

from .dsp import moving_average


# ----------------------------------------------------------------------
# Small DSP helpers
# ----------------------------------------------------------------------
def _bandpass(sig, fs, lo_hz, hi_hz, order):
    """Zero-phase Butterworth band-pass (sos). Stands in for the MATLAB FIR."""
    nyq = fs / 2.0
    lo = max(lo_hz / nyq, 1e-4)
    hi = min(hi_hz / nyq, 0.999)
    if lo >= hi:
        return np.zeros_like(np.asarray(sig, np.float64))
    sos = butter(order, [lo, hi], btype="band", output="sos")
    x = np.asarray(sig, np.float64)
    # sosfiltfilt needs length > 3*(max section order); guard short signals
    if x.size <= 3 * sos.shape[0] * 2:
        return np.zeros_like(x)
    return sosfiltfilt(sos, x)


def _fill_next(a):
    """Fill NaNs with the NEXT valid value (MATLAB fillmissing 'next'); trailing
    NaNs take the previous valid value; all-NaN -> zeros."""
    a = np.asarray(a, np.float64).copy()
    n = a.size
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(a[i]):
            a[i] = nxt
        else:
            nxt = a[i]
    prev = np.nan
    for i in range(n):
        if np.isnan(a[i]):
            a[i] = prev
        else:
            prev = a[i]
    if np.any(np.isnan(a)):
        a[np.isnan(a)] = 0.0
    return a


def _noise_on_grid(grid_t, move_regions):
    """Per-sample 0/1 noise flag on `grid_t` from (start, end) movement regions."""
    flag = np.zeros(grid_t.size, dtype=int)
    for (s, e) in move_regions or []:
        flag[(grid_t >= s) & (grid_t <= e)] = 1
    return flag


def _rank_ascending(v):
    """Rank 1..n; largest value -> largest rank (MATLAB double-sort score)."""
    order = np.argsort(v, kind="stable")
    rank = np.empty(v.size)
    rank[order] = np.arange(1, v.size + 1)
    return rank


def _mode_count(vals):
    """Most frequent value (NaN ignored), its count, and a tie flag."""
    v = vals[~np.isnan(vals)]
    if v.size == 0:
        return np.nan, 0, False
    u, c = np.unique(v, return_counts=True)
    m = int(c.max())
    cand = u[c == m]
    return cand[0], m, cand.size > 1


# ----------------------------------------------------------------------
# Filter bank + per-band scoring (BBB_SignalCreation.m @16-75)
# ----------------------------------------------------------------------
def _bank_and_scores(aligned, wide, noise, fs, bands, order, win_sec, ov_sec, min_valid_sec):
    nw = aligned.size
    nb = bands.shape[0]
    rr_bank = np.zeros((nw, nb))
    aligned_bank = np.zeros((nw, nb))
    for i in range(nb):
        rr_bank[:, i] = _bandpass(aligned, fs, bands[i, 0] / 60.0, bands[i, 1] / 60.0, order)
        # DC removal from the wide signal by this band's longest period
        aligned_bank[:, i] = wide - moving_average(wide, int(round(fs * 60.0 / bands[i, 0])))

    Win = int(round(win_sec * fs))
    ov = max(1, int(round(ov_sec * fs)))
    Eng = np.full((nw, nb), np.nan)
    RMS2 = np.full((nw, nb), np.nan)
    for j in range(Win, nw + 1, ov):
        sl = slice(j - Win, j)
        valid = noise[sl] != 1
        vN = int(np.count_nonzero(valid))
        if vN / fs < min_valid_sec:
            continue
        seg = rr_bank[sl][valid, :]                 # (vN, nb) band-filtered
        seg2 = aligned_bank[sl][valid, :]           # (vN, nb) wide-DC-by-band
        eng = np.sum(seg ** 2, axis=0) / vN
        rms2 = np.sum((seg - seg2) ** 2, axis=0) / vN
        a = max(0, j - ov)
        Eng[a:j, :] = eng
        RMS2[a:j, :] = rms2
    return rr_bank, Eng, RMS2


def _compatibility(Eng, RMS2):
    """Per-instant score = rank(Eng) + rank(1/RMS2). Higher = better fit."""
    nw, nb = Eng.shape
    comp = np.full((nw, nb), np.nan)
    scored = np.isfinite(Eng).all(axis=1) & np.isfinite(RMS2).all(axis=1)
    for r in np.where(scored)[0]:
        comp[r] = _rank_ascending(Eng[r]) + _rank_ascending(-RMS2[r])
    return comp


# ----------------------------------------------------------------------
# Level selection + smoothing (LevelSelection @199-242)
# ----------------------------------------------------------------------
def _select_level(comp, noise, Win, ov, smooth_n):
    nw, nb = comp.shape
    raw = np.full(nw, np.nan)
    scored = np.isfinite(comp).any(axis=1)
    for r in np.where(scored)[0]:
        raw[r] = int(np.nanargmax(comp[r]))
    raw[noise == 1] = np.nan

    sec = np.arange(Win - 1, nw, ov)                # ~ MATLAB Second_ind
    if sec.size == 0:
        return _fill_next(raw)
    ls = raw[sec]
    N = int(smooth_n)

    # forward majority (carry previous on ties)
    padf = np.concatenate([np.full(N - 1, ls[0]), ls])
    majf = np.full(ls.size, np.nan)
    Ff = np.zeros(ls.size)
    for i in range(N - 1, padf.size):
        M, F, tie = _mode_count(padf[i - N + 1:i + 1])
        Ff[i - N + 1] = F
        if not tie:
            majf[i - N + 1] = M
        elif (i - N + 1) > 0:
            majf[i - N + 1] = majf[i - N]
        else:
            majf[0] = padf[0]

    # backward majority
    padb = np.concatenate([ls, np.full(N - 1, ls[0])])
    majb = np.full(ls.size, np.nan)
    Fb = np.zeros(ls.size)
    for i in range(ls.size - 1, -1, -1):
        M, F, tie = _mode_count(padb[i:i + N])
        Fb[i] = F
        if not tie:
            majb[i] = M
        elif i < ls.size - 1:
            majb[i] = majb[i + 1]
        else:
            majb[i] = padb[-1]

    maj = np.where(Fb >= Ff, majb, majf)
    out = np.full(nw, np.nan)
    out[sec] = maj
    return _fill_next(out)


# ----------------------------------------------------------------------
# Assemble the chosen levels + STITCH transitions (BBB_SignalCreation.m @116-161)
# ----------------------------------------------------------------------
def _assemble_and_stitch(rr_bank, level, bands, fs):
    nw, nb = rr_bank.shape
    lev = np.clip(np.round(_fill_next(level)), 0, nb - 1).astype(int)
    stitched = rr_bank[np.arange(nw), lev].copy()

    transitions = np.where(np.diff(lev) != 0)[0]        # change between ti and ti+1
    for ti in transitions:
        before = lev[ti]
        after = lev[ti + 1]
        n = int(round(max(bands[before, 0], bands[after, 1]) * fs / 2.0))
        if n < 1:
            continue
        a = max(0, ti - n)
        b = min(ti + n, nw)
        sb = rr_bank[a:b, before]
        sa = rr_bank[a:b, after]
        dist = sa - sb
        if dist.size < 2:
            continue
        zc = np.where(np.sign(dist[:-1]) != np.sign(dist[1:]))[0]
        if zc.size == 0:
            continue
        local_trans = ti - a                            # ~ n, the nominal switch point
        icross = int(zc[np.argmin(np.abs(zc - local_trans))])
        # Move the switch to the crossing so before==after there (no jump).
        # Ranges are INCLUSIVE of the transition sample (MATLAB [Icross:n] / [n:Icross]);
        # an exclusive slice would leave the transition sample in the old band -> 1-sample spike.
        if icross < local_trans:
            stitched[a + icross:a + local_trans + 1] = sa[icross:local_trans + 1]
        else:
            stitched[a + local_trans:a + icross + 1] = sb[local_trans:icross + 1]
    return stitched, lev


# ----------------------------------------------------------------------
# Trend state-machine breath detector (BBB_peakDetection.m)
# ----------------------------------------------------------------------
def _peak_detect(sig, noise, fs, lev, bands, down_tol_sec, up_th_frac):
    TH = fs * 60.0 / bands[lev, 0]                      # samples/breath at the low rate
    rr_sig = moving_average(sig, 3)
    trend = np.concatenate([[0.0], np.diff(rr_sig)])
    up_th = np.round(up_th_frac * TH)
    down_tol = down_tol_sec * fs

    up = 0
    down = 0
    suspect = -1
    suspect_flag = False
    down_flag = False
    peaks = []
    for i in range(sig.size):
        if noise[i] != 0:
            continue
        if trend[i] >= 0:
            up += 1
            if down_flag and not suspect_flag:
                suspect = i - 1
                suspect_flag = True
            down = 0
        else:
            down += 1
            if down > down_tol:
                up = 0
                down_flag = True
        if suspect_flag and down_flag and up > up_th[i]:
            peaks.append(suspect)
            down_flag = False
            suspect_flag = False
    return np.asarray([p for p in peaks if p >= 0], dtype=int)


def _rr_from_peaks(peaks, grid_t, noise, valid_bpm):
    if peaks.size < 2:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    bs_times = grid_t[peaks]
    dt = np.diff(bs_times)
    ok = dt > 0
    rr = np.where(ok, 60.0 / np.where(ok, dt, 1.0), np.nan)
    lo, hi = valid_bpm
    keep = ok & (rr >= lo) & (rr <= hi)
    for i in range(peaks.size - 1):
        if keep[i] and np.any(noise[peaks[i]:peaks[i + 1]] == 1):
            keep[i] = False
    mid = (bs_times[:-1] + bs_times[1:]) / 2.0
    return bs_times, mid[keep], rr[keep]


# ----------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------
@dataclass
class BankResult:
    grid_t: np.ndarray          # working-rate time vector (s)
    stitched: np.ndarray        # reconstructed respiration trace (working rate)
    sig_on_t: np.ndarray        # stitched interpolated back onto the input t grid
    level: np.ndarray           # chosen band index per working-rate sample
    bs_times: np.ndarray        # breath-start times (s)
    rr_time: np.ndarray         # per-breath RR timestamps (s)
    rr_bpm: np.ndarray          # per-breath RR (bpm)


def analyze_filterbank_bbb(x, t, fs, cfg, move_regions=None):
    """Run the MATLAB filter-bank + stitching BBB method on one channel.

    x  : channel samples on the input timeline (e.g. 1024 Hz)
    t  : matching time vector (s)
    fs : input sample rate
    cfg: settings.PPG (reads the bwbank_* fields)
    Returns a BankResult (all-empty-but-valid if the signal is too short).
    """
    x = np.asarray(x, np.float64)
    t = np.asarray(t, np.float64)
    bands = np.asarray(cfg.bwbank_rate_bands_bpm, np.float64)   # (nb, 2) bpm
    work_fs = float(cfg.bwbank_work_fs)
    order = int(cfg.bwbank_bp_order)

    # 1) decimate to the working rate
    g = gcd(int(round(work_fs)), int(round(fs)))
    xw = resample_poly(x, int(round(work_fs)) // g, int(round(fs)) // g)
    nw = xw.size
    t0 = float(t[0]) if t.size else 0.0
    grid_t = t0 + np.arange(nw) / work_fs

    empty = BankResult(grid_t, np.zeros(nw), np.zeros(t.size), np.zeros(nw),
                       np.zeros(0), np.zeros(0), np.zeros(0))
    if nw < int(round(cfg.bwbank_score_win_sec * work_fs)) + 2:
        return empty

    noise = _noise_on_grid(grid_t, move_regions)

    # 2) build the two working signals: aligned (narrow, DC-removed) + wide band
    wide = _bandpass(xw, work_fs, cfg.bwbank_wide_band_hz[0], cfg.bwbank_wide_band_hz[1], order)
    aligned = xw - moving_average(xw, int(round(cfg.bwbank_align_win_sec * work_fs)))

    def _norm(s):
        v = s[noise != 1]
        if v.size < 2:
            return s
        return (s - np.median(v)) / (np.std(v) + 1e-12)

    aligned = _norm(aligned)
    wide = _norm(wide)

    # 3) filter bank + per-window Eng/RMS2 scores
    rr_bank, Eng, RMS2 = _bank_and_scores(
        aligned, wide, noise, work_fs, bands, order,
        cfg.bwbank_score_win_sec, cfg.bwbank_score_overlap_sec, cfg.bwbank_min_valid_sec)

    # 4) pick the best level per instant + smooth the choice
    comp = _compatibility(Eng, RMS2)
    Win = int(round(cfg.bwbank_score_win_sec * work_fs))
    ov = max(1, int(round(cfg.bwbank_score_overlap_sec * work_fs)))
    level = _select_level(comp, noise, Win, ov, cfg.bwbank_level_smooth_n)

    # 5) assemble + stitch transitions -> clean respiration trace
    stitched, lev = _assemble_and_stitch(rr_bank, level, bands, work_fs)

    # 6) detect breaths + per-breath RR
    peaks = _peak_detect(stitched, noise, work_fs, lev, bands,
                         cfg.bwbank_peak_down_tol_sec, cfg.bwbank_peak_up_th_frac)
    bs_times, rr_time, rr_bpm = _rr_from_peaks(peaks, grid_t, noise, cfg.bwbank_valid_bpm)

    sig_on_t = np.interp(t, grid_t, stitched) if t.size else np.zeros(0)
    return BankResult(grid_t, stitched, sig_on_t, lev.astype(float), bs_times, rr_time, rr_bpm)
