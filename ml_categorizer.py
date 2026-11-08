#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conditional multi-channel ML categorization ridge (LOSO).

For each recording, per STFT column we build features from every channel's BW spectrogram (its
excess-ridge RR + peak prominence + best fast-peak RR + its prominence) plus cross-channel
consensus/spread and HR / Nyquist context. A HistGradientBoosting regressor predicts the per-column
RR, trained Leave-One-Recording-Out so a recording is never predicted by a model that saw it. The
predicted RR is temporal-cleaned and returned as a per-recording ridge, ready to drop into the
existing categorization chain (segment-by-crossing) in place of the Artifact-only ridge.

`compute_loso_ridge(recs)` -> {rid: (t_watch, ridge)}  (watch clock, no sync offset).

Rationale: naive spectral fusion of channels HURTS (optical low-freq / per-beat aliasing pollute a
shared ridge); a learned conditional combiner uses each channel only where it helps. See the memory
note cat-multichannel-fusion-rejected.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter
import per_category_lab as PCL
import bw_category_map as CM
import noise_zones as NZ
from param_category_sweep import stft_power

CH5 = ["Artifact", "Red", "IR", "Green", "Yellow"]
BASES = ["RSA", "AMP", "AUC"]
FAST_PROM = 0.10            # a channel "sees fast" if its fast-peak prominence >= this
AGREE_BPM = 3.0            # a channel "agrees" if its ridge RR is within this of the consensus
WIN_PRIMARY = 24.0         # main STFT window (defines the column grid)
WIN_SHORT = 16.0           # multi-resolution: short window, better time res for fast/transitions
HOP = 5.0


def _chan_feats(sig_ch, fs, win=WIN_PRIMARY):
    """per-column [rr_all, prom_all, rr_fast, prom_fast] from a channel's BW excess spectrogram."""
    ny = fs / 2
    xbw = sosfiltfilt(butter(2, [0.10 / ny, 0.9 / ny], btype="band", output="sos"), np.asarray(sig_ch, float))
    t, fr, P = stft_power(PCL._agc(xbw, fs), fs, float(win), HOP)
    fb = fr * 60.; df = (fb[1] - fb[0]) if fb.size > 1 else 2.5
    K = max(2, int(round(9. / df)))
    E = np.clip(P - median_filter(P, size=(2 * K + 1, 1), mode="nearest"), 0, None)
    n = P.shape[1]; cols = np.arange(n)
    inb = (fb >= 6) & (fb <= 54); colMax = np.where(inb[:, None], P, 0).max(0) + 1e-30
    Eb = np.where(inb[:, None], E, -1.0); kA = np.argmax(Eb, 0)
    rr_all = fb[kA]; prom_all = Eb[kA, cols] / colMax
    fast = (fb >= 30) & (fb <= 54); Ef = np.where(fast[:, None], E, -1.0); kF = np.argmax(Ef, 0)
    rr_fast = fb[kF]; prom_fast = Ef[kF, cols] / colMax
    return t, np.column_stack([rr_all, prom_all, rr_fast, prom_fast])


COMBO_CH = ["Green", "Yellow"]     # per-beat-param channels summed into one combined modulation


def _excess_rr_prom(x, fs, t_grid, win=WIN_PRIMARY):
    """[rr, prom] per column from a 1-D signal's excess spectrogram, aligned to t_grid (same
    excess-ridge recipe as _chan_feats, on an already-band-limited envelope)."""
    t, fr, P = stft_power(PCL._agc(np.asarray(x, float), fs), fs, float(win), HOP)
    fb = fr * 60.; df = (fb[1] - fb[0]) if fb.size > 1 else 2.5
    K = max(2, int(round(9. / df)))
    E = np.clip(P - median_filter(P, size=(2 * K + 1, 1), mode="nearest"), 0, None)
    cols = np.arange(P.shape[1])
    inb = (fb >= 6) & (fb <= 54); colMax = np.where(inb[:, None], P, 0).max(0) + 1e-30
    Eb = np.where(inb[:, None], E, -1.0); kA = np.argmax(Eb, 0)
    rr = fb[kA]; prom = Eb[kA, cols] / colMax
    return (np.interp(t_grid, t, rr, left=np.nan, right=np.nan),
            np.interp(t_grid, t, prom, left=np.nan, right=np.nan))


def _combo_param_feats(results, t):
    """For each per-beat-param base (AUC/AMP/RSA): z-score-normalize the band-passed envelopes of
    COMBO_CH (Green+Yellow) so neither LED dominates, sum them on a common grid, and extract
    [rr, prom] aligned to column times t. Summing two weak-but-coherent LED modulations boosts the
    respiration SNR where either channel alone is low -> better fast/mid categorization evidence."""
    out = {}
    for base in BASES:
        series = []
        for c in COMBO_CH:
            if c in results:
                p = results[c].params.get(base)
                if p is not None and np.size(getattr(p, "env_y", [])) >= 8:
                    ex = np.asarray(p.env_x, float); ey = np.asarray(p.env_y, float)
                    s = np.nanstd(ey)
                    if np.isfinite(s) and s > 0:
                        series.append((ex, (ey - np.nanmean(ey)) / s))
        if not series:
            out[base] = (np.full(len(t), np.nan), np.full(len(t), np.nan)); continue
        lo = max(s[0][0] for s in series); hi = min(s[0][-1] for s in series)
        dt = np.median(np.diff(series[0][0]))
        if not (hi > lo) or not (dt > 0):
            out[base] = (np.full(len(t), np.nan), np.full(len(t), np.nan)); continue
        grid = np.arange(lo, hi, dt)
        summed = np.sum([np.interp(grid, ex, ey) for ex, ey in series], axis=0)
        out[base] = _excess_rr_prom(summed, 1.0 / dt, t)
    return out


FEAT_NAMES = (
    [f"{c}:{k}" for c in CH5 for k in ("rr", "prom", "rr_fast", "prom_fast")]
    + ["cons", "spread", "hr", "hr/2", "cons-hr/2"]
    + ["max_prom", "max_prom_fast", "n_fast", "fast_cons", "fast_cons-hr/2",
       "n_agree", "param_cons", "param-bw", "above_nyq", "prev_cons"]
    + [f"gy_{b}_{k}" for b in BASES for k in ("rr", "prom")])   # summed Green+Yellow per-beat params
# (multi-resolution 16s features were tested and REVERTED: they raised BBBRR MAE 2.33 -> 2.40, overfit.)


def _features_for(sig, results, offset, ref):
    """Per-column feature matrix X, column times t (watch clock), and target y (ref segment-median RR).
    Columns are FEAT_NAMES: per-channel ridge/prominence (BW), consensus/HR context, plus richer
    cross-channel fast-evidence & agreement, per-beat-param consensus RR, Nyquist flags, temporal prev."""
    from respiration_rr.rr_average import reference_average_rr
    fs = float(sig.fs); t = None; feats = {}
    for c in CH5:
        if c in sig.channels:
            tc, f = _chan_feats(sig.channels[c], fs, WIN_PRIMARY)
            if t is None:
                t = tc
            feats[c] = f
    nrow = t.shape[0]
    col = lambda j: np.column_stack([feats[c][:, j] if c in feats else np.full(nrow, np.nan) for c in CH5])
    RA, PA, RF, PF = col(0), col(1), col(2), col(3)                 # rr_all, prom_all, rr_fast, prom_fast
    with np.errstate(invalid="ignore", all="ignore"):
        cons = np.nanmedian(RA, axis=1); spread = np.nan_to_num(np.nanstd(RA, axis=1))
        max_prom = np.nanmax(np.where(np.isfinite(PA), PA, np.nan), axis=1)
        max_prom_fast = np.nanmax(np.where(np.isfinite(PF), PF, np.nan), axis=1)
    n_fast = np.sum(np.nan_to_num(PF) >= FAST_PROM, axis=1).astype(float)
    n_agree = np.sum(np.abs(RA - cons[:, None]) <= AGREE_BPM, axis=1).astype(float)
    # fast consensus = rr_fast of the channel with the strongest fast peak
    fast_cons = np.full(nrow, np.nan); valid = np.isfinite(PF).any(axis=1)
    if valid.any():
        idx = np.nanargmax(np.where(np.isfinite(PF), PF, -np.inf)[valid], axis=1)
        fast_cons[valid] = RF[valid][np.arange(int(valid.sum())), idx]
    hr_t, hr_v = CM.hr_series_from_beats(results, sig, offset)
    hr = (np.interp(t + offset, np.asarray(hr_t, float), np.asarray(hr_v, float), left=np.nan, right=np.nan)
          if np.size(hr_t) else np.full(nrow, np.nan))
    # per-beat param consensus RR (model may down-weight; aliases above HR/2)
    prch = []
    for c in CH5:
        vals = []
        if c in results:
            for b in BASES:
                p = results[c].params.get(b)
                if p is not None and np.size(getattr(p, "rr_bpm", [])) >= 2:
                    vals.append(np.interp(t, np.asarray(p.rr_time, float), np.asarray(p.rr_bpm, float),
                                          left=np.nan, right=np.nan))
        prch.append(np.nanmedian(np.column_stack(vals), axis=1) if vals else np.full(nrow, np.nan))
    with np.errstate(invalid="ignore", all="ignore"):
        param_cons = np.nanmedian(np.column_stack(prch), axis=1)
    above_nyq = np.where(np.isfinite(cons) & np.isfinite(hr), (cons > hr / 2.0).astype(float), np.nan)
    prev_cons = np.concatenate([[np.nan], cons[:-1]])
    combo = _combo_param_feats(results, t)                       # summed Green+Yellow per-beat params
    combo_cols = [combo[b][j] for b in BASES for j in (0, 1)]    # [rr, prom] per base, FEAT_NAMES order
    blocks = [feats[c] if c in feats else np.full((nrow, 4), np.nan) for c in CH5]
    X = np.column_stack(blocks + [cons, spread, hr, hr / 2.0, cons - hr / 2.0,
                                  max_prom, max_prom_fast, n_fast, fast_cons, fast_cons - hr / 2.0,
                                  n_agree, param_cons, param_cons - cons, above_nyq, prev_cons]
                        + combo_cols)
    tav, rav = reference_average_rr(ref)
    y = np.interp(t + offset, np.asarray(tav, float), np.asarray(rav, float), left=np.nan, right=np.nan)
    return X, t, y


def compute_loso_ridge(recs, verbose=True, noise_dir=None):
    """recs: list of (rid, edf, csv). Returns {rid: (t_watch, temporal_cleaned_ridge)} via LOSO.
    noise_dir: if given, hand-annotated red-zone columns are EXCLUDED FROM TRAINING (their reference
    target is untrustworthy) — but the ridge is still PREDICTED for every column (routing needs it)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    from main import run_ppg, run_reference, run_sync
    data = []
    for rid, edf, csv in recs:
        try:
            results, sig = run_ppg(csv); ref = run_reference(edf); off = run_sync(edf, results, sig)
            X, t, y = _features_for(sig, results, off, ref)
            clean = np.ones(t.shape[0], bool)
            if noise_dir:
                red, _ = NZ.manual_zones(rid, noise_dir)
                clean = ~NZ.in_red(t + off, red)                    # red zones are on the REMbo clock
            data.append((rid, t, X, y, clean))
        except Exception as e:
            if verbose:
                print(f"  [ml-cat warn] {rid}: {e}")
    if not data:
        return {}
    Xall = np.vstack([d[2] for d in data]); yall = np.concatenate([d[3] for d in data])
    clean_all = np.concatenate([d[4] for d in data])
    groups = np.concatenate([[d[0]] * d[2].shape[0] for d in data])
    keep = np.isfinite(yall) & clean_all                            # train only on clean, finite-target columns
    pred = np.full(yall.shape, np.nan)
    for tr, te in LeaveOneGroupOut().split(Xall, yall, groups):
        trk = tr[keep[tr]]
        if trk.size:
            pred[te] = HistGradientBoostingRegressor(random_state=42).fit(Xall[trk], yall[trk]).predict(Xall[te])
    out = {}
    for rid, t, X, y, clean in data:
        rr = pred[groups == rid]
        out[rid] = (np.asarray(t, float), PCL._temporal_clean(np.asarray(t, float), rr))
    if verbose:
        print(f"  [ml-cat] trained LOSO multi-channel categorizer on {len(data)} recordings"
              + (f"; {int(keep.sum())}/{keep.size} clean train columns" if noise_dir else ""))
    return out
