"""
PPG beat detection and quality refinement (faithful ports).

detectBeats @4474, removeDCBeatAligned @4595, refineSSByDerivative @8162,
postCorrFilter @8469, filterPeaksByProminence @4790.
"""

import numpy as np

from ..settings import PPG
from .dsp import lowpass_filter


def is_in_noise(t, regions):
    """True if time t falls inside any (start, end) noise region."""
    if not regions:
        return False
    for (s, e) in regions:
        if s <= t <= e:
            return True
    return False


def detect_beats(signal, fs, min_pr, max_pr, return_stages=False, cfg=None):
    """Detect PPG beats = troughs of the inverted band-pass signal (detectBeats @4474).

    1. Local minima over a +-halfWin window (halfWin = minDist//4).
    2. Enforce a minimum inter-beat distance (60/max_pr), keeping the deeper.
    3. Dicrotic-notch rejection: for a beat whose neighbour interval is < 65% of
       the median, drop it if its valley is shallow (< 75% of the typical
       amplitude) or shallower than both neighbours.
    4. Upslope check: reject beats whose 80 ms post-trough gain < 40% of median.

    Returns a list of sample indices. If return_stages=True, returns
    (final_beats, stages) where stages has 'minima', 'after_dist', 'after_notch',
    'final' — the index list after each step (for the preprocessing inspector).
    """
    if cfg is None:
        cfg = PPG
    dicrotic_rr = cfg.dicrotic_rr_frac      # 0.65
    dicrotic_amp = cfg.dicrotic_amp_frac    # 0.75
    upslope_frac = cfg.upslope_frac         # 0.40
    upslope_win = cfg.upslope_window_sec    # 0.08 s

    x = np.asarray(signal, np.float64)
    n = x.size
    min_dist = int(np.floor(fs * 60 / max_pr))
    half_win = max(2, min_dist // 4)
    stages = {"minima": [], "after_dist": [], "after_notch": [], "final": []}

    def _ret(final):
        stages["final"] = list(final)
        return (list(final), stages) if return_stages else list(final)

    # ---- local minima ----
    all_minima = []
    for i in range(half_win, n - half_win):
        v = x[i]
        is_min = True
        for j in range(1, half_win + 1):
            if v > x[i - j] or v > x[i + j]:
                is_min = False
                break
        if is_min:
            all_minima.append(i)
    stages["minima"] = list(all_minima)
    if not all_minima:
        return _ret([])

    # ---- enforce minimum distance (keep deeper trough) ----
    kept = [all_minima[0]]
    for idx in all_minima[1:]:
        last = kept[-1]
        if idx - last >= min_dist:
            kept.append(idx)
        elif x[idx] < x[last]:
            kept[-1] = idx
    beats = kept
    stages["after_dist"] = list(beats)
    if len(beats) < 3:
        stages["after_notch"] = list(beats)
        return _ret(beats)

    # ---- dicrotic-notch rejection ----
    intervals = np.diff(beats)
    median_interval = np.sort(intervals)[len(intervals) // 2]
    keep = [True] * len(beats)
    for i in range(1, len(beats) - 1):
        prev_iv = beats[i] - beats[i - 1]
        next_iv = beats[i + 1] - beats[i]
        if prev_iv < median_interval * dicrotic_rr or next_iv < median_interval * dicrotic_rr:
            peak_before = x[beats[i - 1]:beats[i] + 1].max()
            peak_after = x[beats[i]:beats[i + 1] + 1].max()
            valley = x[beats[i]]
            higher_peak = max(peak_before, peak_after)
            beat_amp = higher_peak - valley
            deeper_neighbor = min(x[beats[i - 1]], x[beats[i + 1]])
            typical_amp = higher_peak - deeper_neighbor
            shallower_than_both = valley > x[beats[i - 1]] and valley > x[beats[i + 1]]
            if typical_amp > 0 and (beat_amp < typical_amp * dicrotic_amp or shallower_than_both):
                keep[i] = False
    after_notch = [b for b, k in zip(beats, keep) if k]
    stages["after_notch"] = list(after_notch)
    if len(after_notch) < 3:
        return _ret(after_notch)

    # ---- upslope (post-slope) verification ----
    gain_win = int(round(fs * upslope_win))
    gains = []
    for idx in after_notch:
        end_j = min(idx + gain_win, n - 1)
        seg = x[idx + 1:end_j + 1] - x[idx]
        gains.append(seg.max() if seg.size else 0.0)
    gains = np.asarray(gains)
    median_gain = np.sort(gains)[len(gains) // 2]
    gain_thresh = median_gain * upslope_frac
    final = [b for b, g in zip(after_notch, gains) if g >= gain_thresh]
    return _ret(final)


def detect_beats_lpf_derivative(bp_signal, fs, cfg=None, time=None, move_regions=None,
                                return_stages=False):
    """LPF-derivative beat detection — the method the shipped HTML actually uses
    (Override PPG/Red/IR beats, graph 2b @8115 / 15b @8211).

    1. 1st derivative of the band-passed signal.
    2. Low-pass the derivative at cfg.deriv_lpf_hz (order cfg.ppg_filter_order).
    3. Positive local maxima of that = steepest systolic upslopes.
    4. Snap each peak back to the nearest local minimum within +-beat_snap_window_sec
       -> the beat foot (SS).
    5. Keep in chronological order with a minimum inter-beat distance (60/hr_max),
       skipping any foot that falls inside a movement/noise region.

    Returns a list of sample indices (the beat feet). With return_stages=True,
    returns (beats, stages) where stages has 'deriv_lpf', 'deriv_peaks',
    'snapped' for the inspector.
    """
    if cfg is None:
        cfg = PPG
    x = np.asarray(bp_signal, np.float64)
    n = x.size
    stages = {"deriv_lpf": None, "deriv_peaks": [], "snapped": []}

    deriv = np.zeros(n)
    if n > 1:
        deriv[1:] = (x[1:] - x[:-1]) * fs
        deriv[0] = deriv[1]
    deriv_lpf = lowpass_filter(deriv, fs, cfg.deriv_lpf_hz, cfg.ppg_filter_order)
    stages["deriv_lpf"] = deriv_lpf

    # positive local maxima of the LP derivative
    peaks = []
    for i in range(1, n - 1):
        if deriv_lpf[i] > 0 and deriv_lpf[i] >= deriv_lpf[i - 1] and deriv_lpf[i] >= deriv_lpf[i + 1]:
            peaks.append(i)
    stages["deriv_peaks"] = list(peaks)

    sw = int(round(fs * cfg.beat_snap_window_sec))
    min_dist = int(round(fs * 60 / cfg.hr_max_bpm))
    beats = []
    for pk in peaks:
        lo = max(0, pk - sw)
        hi = min(n - 1, pk + sw)
        seg = x[lo:hi + 1]
        min_idx = lo + int(np.argmin(seg))
        t_at = time[min_idx] if time is not None else None
        if t_at is not None and is_in_noise(t_at, move_regions):
            continue
        if not beats or min_idx - beats[-1] >= min_dist:
            beats.append(min_idx)
    stages["snapped"] = list(beats)
    return (beats, stages) if return_stages else beats


def remove_dc_beat_aligned(signal, peaks):
    """Per-beat linear-baseline subtraction between troughs (removeDCBeatAligned @4595)."""
    x = np.asarray(signal, np.float64)
    out = np.zeros(x.size)
    if len(peaks) < 2:
        return out
    out[:peaks[0]] = x[:peaks[0]] - x[peaks[0]]
    for p in range(len(peaks) - 1):
        i0, i1 = peaks[p], peaks[p + 1]
        v0, v1 = x[i0], x[i1]
        idx = np.arange(i0, i1)
        frac = (idx - i0) / (i1 - i0)
        out[i0:i1] = x[i0:i1] - (v0 + frac * (v1 - v0))
    last = peaks[-1]
    out[last:] = x[last:] - x[last]
    return out


def refine_ss_by_derivative(beats, bp_signal, fs):
    """Refine each SS to the systolic-rise onset (refineSSByDerivative @8162)."""
    x = np.asarray(bp_signal, np.float64)
    n = x.size
    deriv = np.zeros(n)
    deriv[1:] = (x[1:] - x[:-1]) * fs
    deriv[0] = deriv[1]
    refined = []
    for b in range(len(beats)):
        ss = beats[b]
        next_ss = beats[b + 1] if b + 1 < len(beats) else n - 1
        seg = deriv[ss:next_ss] if next_ss > ss else deriv[ss:ss + 1]
        d1_max_idx = ss + int(np.argmax(seg)) if seg.size else ss
        d1_max_val = deriv[d1_max_idx]
        prev_beat = beats[b - 1] if b > 0 else max(0, d1_max_idx - int(round(fs * 0.5)))
        lo = max(0, max(prev_beat, d1_max_idx - int(round(fs * 0.5))))
        rise_thresh = d1_max_val * 0.05
        ref_idx = -1
        for i in range(d1_max_idx, lo, -1):
            if deriv[i] < rise_thresh:
                ref_idx = i
                break
        if ref_idx < 0:               # fallback: local minimum
            ref_idx = d1_max_idx
            min_val = x[d1_max_idx]
            for i in range(d1_max_idx, lo - 1, -1):
                if x[i] <= min_val:
                    min_val = x[i]
                    ref_idx = i
        refined.append(ref_idx)
    return refined


def post_corr_filter(beats, bp_signal, corr_thresh):
    """Pearson beat-quality gate (postCorrFilter @8469).

    NOTE: in the shipped tool the calls to this were commented out
    (POST_CORR_REMOVED_END @8530), so it is OFF by default. Enable by passing a
    positive corr_thresh. Rejects a beat whose interpolated 64-pt waveform
    correlates below `corr_thresh` with the last accepted beat.
    """
    if len(beats) < 3 or corr_thresh <= 0:
        return beats
    x = np.asarray(bp_signal, np.float64)
    n_pts = 64

    def interp_beat(i0, i1):
        length = i1 - i0
        if length < 3:
            return None
        frac = np.arange(n_pts) / (n_pts - 1)
        idx = i0 + frac * (length - 1)
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, x.size - 1)
        return x[lo] + (idx - lo) * (x[hi] - x[lo])

    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else 0.0

    kept = [beats[0]]
    last_valid = interp_beat(beats[0], beats[1])
    for k in range(1, len(beats) - 1):
        wf = interp_beat(beats[k], beats[k + 1])
        if wf is None or last_valid is None:
            kept.append(beats[k])
            if wf is not None:
                last_valid = wf
            continue
        if pearson(last_valid, wf) >= corr_thresh:
            kept.append(beats[k])
            last_valid = wf
    if len(beats) > 1:
        kept.append(beats[-1])
    return kept


def filter_peaks_by_prominence(signal, peaks, min_prom):
    """Iteratively drop the least-prominent peak until all exceed min_prom
    (filterPeaksByProminence @4790)."""
    x = np.asarray(signal, np.float64)
    kept = list(peaks)
    while kept:
        worst_k = -1
        worst_prom = np.inf
        for k in range(len(kept)):
            peak_val = x[kept[k]]
            left_bound = 0 if k == 0 else kept[k - 1]
            right_bound = x.size - 1 if k == len(kept) - 1 else kept[k + 1]
            left_valley = x[left_bound:kept[k] + 1].min()
            right_valley = x[kept[k]:right_bound + 1].min()
            prom = peak_val - max(left_valley, right_valley)
            if prom < worst_prom:
                worst_prom = prom
                worst_k = k
        if worst_prom >= min_prom:
            break
        kept.pop(worst_k)
    return kept


def _local_maxima(x):
    """Indices of strict interior local maxima (scipy.find_peaks default; pure numpy)."""
    x = np.asarray(x, np.float64)
    if x.size < 3:
        return np.array([], dtype=int)
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1


def peak_detection_zero_cross(signal, noise, fs):
    """Legacy breath-marker detector — port of Breath_by_Breath's
    peakDetectionZeroCross (RR_run.py, v5.0).

    On a respiratory-band signal (oscillating about zero), per contiguous
    non-noise segment it takes whichever polarity of peak is *rarer* (positive
    maxima vs negative minima) as the breath markers — the minority polarity
    tracks breath boundaries — then keeps only the most positive marker between
    consecutive zero crossings.

    signal : respiratory-band trace (e.g. the BW 0.1-1.0 Hz band-passed channel)
    noise  : per-sample flag, 1 = drop (movement / invalid), 0 = valid
    fs     : sample rate (kept for signature parity; not used numerically)

    Returns breath-marker sample indices (sorted, unique int array).

    NOTE: the MATLAB/JS original removed duplicate markers with a scalar-only
    expression; this port uses a set-based drop so it stays correct (and never
    raises) when >2 markers fall between one pair of zero crossings.
    """
    x = np.asarray(signal, np.float64)
    noise = np.asarray(noise).astype(int)
    n = x.size
    if n < 3:
        return np.array([], dtype=int)

    # 1) peaks (both polarities) on the noise-eroded valid signal
    noise1 = noise.astype(bool).copy()
    noise1[np.where(noise[1:] == 1)[0]] = True            # erode: 1-sample around noise
    noise1[np.where(noise[:-1] == 1)[0] + 1] = True
    valid_idx = np.where(~noise1)[0]
    if valid_idx.size < 3:
        return np.array([], dtype=int)
    valid_sig = x[valid_idx]

    pos = _local_maxima(valid_sig)
    pos = pos[valid_sig[pos] > 0]
    neg = _local_maxima(-valid_sig)
    neg = neg[valid_sig[neg] < 0]
    pos_idx = valid_idx[pos]
    neg_idx = valid_idx[neg]

    # 2) contiguous non-noise segment edges
    seg_start = np.where((noise[:-1] == 1) & (noise[1:] == 0))[0] + 1
    if noise[0] == 0:
        seg_start = np.concatenate(([0], seg_start))
    seg_end = np.where((noise[:-1] == 0) & (noise[1:] == 1))[0]
    if noise[-1] == 0:
        seg_end = np.concatenate((seg_end, [n - 1]))

    peaks = np.array([], dtype=int)
    for s, e in zip(seg_start, seg_end):
        cur_pos = pos_idx[(pos_idx >= s) & (pos_idx <= e)]
        cur_neg = neg_idx[(neg_idx >= s) & (neg_idx <= e)]
        # 3) rarer polarity = the breath markers
        chosen = cur_neg if cur_pos.size > cur_neg.size else cur_pos
        peaks = np.concatenate((peaks, chosen))

        # 4) keep only the highest marker between consecutive zero crossings
        seg = x[s:e]
        if seg.size < 2:
            continue
        zc = np.where(((seg[:-1] > 0) & (seg[1:] < 0)) |
                      ((seg[:-1] < 0) & (seg[1:] > 0)))[0] + s
        for j in range(1, zc.size):
            between = peaks[(peaks > zc[j - 1]) & (peaks < zc[j])]
            if between.size > 1:
                keep = between[int(np.argmax(x[between]))]
                peaks = peaks[~np.isin(peaks, between[between != keep])]

    return np.unique(peaks.astype(int))
