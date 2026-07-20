"""
Systolic analysis — SS / SE / MSD fiducials, AUC and per-beat max-height.

Faithful numeric port of computeSystolicAnalysis(@5576). The HTML function also
builds Plotly annotations (SSTD/MSDTD/PI/CORR labels); those are dropped here —
we keep the numeric series the respiration and comparison stages consume.

MSD (max systolic derivative) priority, all >= minMsdMs after SS and past the
percentage-of-d1-max gate:
  1. 4th-derivative zero crossing (pos -> neg)
  2. 3rd-derivative zero crossing (neg -> pos)
  3. 1st-derivative peak
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SystolicResult:
    ss_idx: list          # systolic start (beat trough) indices
    se_idx: list          # systolic peak indices
    msd_idx: list         # max-systolic-derivative indices
    auc_x: np.ndarray     # per-beat AUC: time at SS
    auc_y: np.ndarray     # per-beat AUC value
    maxht_x: np.ndarray   # per-beat max-height: mid-beat time
    maxht_y: np.ndarray   # per-beat (max-min) amplitude
    msd_ss_ms: np.ndarray # SS->MSD time in ms (per beat)


def compute_systolic_analysis(bc, t, beat_indices, fs, min_msd_ms=40.0, msd_min_pct_d1=30.0):
    """Compute SS/SE/MSD, AUC and max-height per beat from a baseline-corrected trace."""
    bc = np.asarray(bc, np.float64)
    t = np.asarray(t, np.float64)
    n = bc.size
    ss_idx = list(beat_indices)

    # ---- SE: first local max after SS above the SS level ----
    se_idx = []
    for k in range(len(ss_idx) - 1):
        i0, i1 = ss_idx[k], ss_idx[k + 1]
        se = -1
        for i in range(i0 + 1, min(i1, n - 1)):
            if bc[i] >= bc[i - 1] and bc[i] >= bc[i + 1] and bc[i] > bc[i0]:
                se = i
                break
        if se < 0:
            seg = bc[i0:min(i1 + 1, n)]
            se = i0 + int(np.argmax(seg)) if seg.size else i0
        se_idx.append(se)

    # ---- derivatives ----
    d1 = np.zeros(n)
    d1[1:] = (bc[1:] - bc[:-1]) * fs
    d1[0] = d1[1] if n > 1 else 0.0
    d2 = np.zeros(n)
    d2[1:-1] = (d1[2:] - d1[:-2]) * fs / 2
    d3_raw = np.zeros(n)
    d3_raw[1:-1] = (d2[2:] - d2[:-2]) * fs / 2
    hw = int(round(fs * 0.012))  # +-12 ms smoothing
    d3 = _box_smooth(d3_raw, hw)
    d4_raw = np.zeros(n)
    d4_raw[1:-1] = (d3[2:] - d3[:-2]) * fs / 2
    d4 = _box_smooth(d4_raw, hw)

    # ---- MSD detection ----
    msd_idx = []
    min_msd_samples = int(round(fs * min_msd_ms / 1000))
    for k in range(len(se_idx)):
        srt_start = ss_idx[k]
        srt_end = se_idx[k]
        min_idx = srt_start + min_msd_samples

        d4_cross = -1
        for i in range(max(srt_start + 3, min_idx), min(srt_end, n - 3)):
            if d4[i - 1] > 0 and d4[i] <= 0:
                d4_cross = i
                break
        d3_cross = -1
        for i in range(max(srt_start + 2, min_idx), min(srt_end, n - 2)):
            if d3[i - 1] < 0 and d3[i] >= 0:
                d3_cross = i
                break

        next_ss = ss_idx[k + 1] if k + 1 < len(ss_idx) else n - 1
        seg = d1[srt_start:min(next_ss + 1, n)]
        d1_global_max_idx = srt_start + int(np.argmax(seg)) if seg.size else srt_start

        d1_time = max(d1_global_max_idx - srt_start, min_msd_samples)
        min_cross_samples = int(round(d1_time * msd_min_pct_d1 / 100))
        min_cross_idx = srt_start + max(min_cross_samples, min_msd_samples)

        d1_peak = -1
        for i in range(srt_start + 1, min(srt_end, n - 1)):
            if d1[i] > 0 and d1[i] >= d1[i - 1] and d1[i] >= d1[i + 1] and i >= min_cross_idx:
                d1_peak = i
                break
        if d1_peak < 0:
            d1_peak = d1_global_max_idx
        d1_valid = d1_peak >= 0 and (d1_peak - srt_start) >= min_msd_samples

        earliest = -1
        d1_past = d1_peak >= 0 and d1_peak >= min_cross_idx
        if d4_cross >= 0 and d4_cross >= min_cross_idx:
            if not d1_past or d4_cross < d1_peak:
                earliest = d4_cross
        if d3_cross >= 0 and d3_cross >= min_cross_idx:
            if not d1_past or d3_cross < d1_peak:
                if earliest < 0 or d3_cross < earliest:
                    earliest = d3_cross

        if earliest >= 0:
            found = earliest
        elif d1_valid:
            found = d1_peak
        else:
            if d3_cross >= 0 and d3_cross >= min_cross_idx:
                found = d3_cross
            elif d4_cross >= 0 and d4_cross >= min_cross_idx:
                found = d4_cross
            else:
                found = d1_global_max_idx
        msd_idx.append(found)

    # ---- AUC per beat (trapezoid of BC / fs) ----
    auc_x, auc_y = [], []
    for k in range(len(ss_idx) - 1):
        i0, i1 = ss_idx[k], ss_idx[k + 1]
        seg = bc[i0:min(i1, n)]
        if seg.size >= 2:
            auc = np.trapz(seg) / fs
        else:
            auc = 0.0
        auc_x.append(t[i0]); auc_y.append(auc)

    # ---- max-height per beat (RIIV / amplitude) ----
    maxht_x, maxht_y = [], []
    for k in range(len(ss_idx) - 1):
        i0, i1 = ss_idx[k], ss_idx[k + 1]
        if i0 >= i1 or i1 >= n:
            continue
        seg = bc[i0:i1 + 1]
        height = float(seg.max() - seg.min())
        if height > 0:
            maxht_x.append((t[i0] + t[i1]) / 2)
            maxht_y.append(height)

    # ---- SS->MSD time (ms) ----
    msd_ss_ms = []
    for k in range(min(len(ss_idx), len(msd_idx))):
        dt = t[msd_idx[k]] - t[ss_idx[k]]
        if dt > 0:
            msd_ss_ms.append(dt * 1000)

    return SystolicResult(
        ss_idx=ss_idx, se_idx=se_idx, msd_idx=msd_idx,
        auc_x=np.asarray(auc_x), auc_y=np.asarray(auc_y),
        maxht_x=np.asarray(maxht_x), maxht_y=np.asarray(maxht_y),
        msd_ss_ms=np.asarray(msd_ss_ms),
    )


def _box_smooth(x, hw):
    """Centered box smooth over +-hw samples with shrinking edges (matches JS loop)."""
    n = x.size
    if hw <= 0:
        return x.copy()
    cum = np.empty(n + 1); cum[0] = 0.0
    np.cumsum(x, out=cum[1:])
    i = np.arange(n)
    lo = np.maximum(0, i - hw)
    hi = np.minimum(n - 1, i + hw)
    return (cum[hi + 1] - cum[lo]) / (hi - lo + 1)
