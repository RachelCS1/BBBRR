"""
Movement / activity energy and noise-region detection.

Faithful ports of computeMovementEnergy(@1415), computeActivityEnergy(@3278),
computeNoiseRegions(@1437), _otsuThreshold(@3287), autoActivityThreshold(@3336).
"""

import numpy as np
from .filters import moving_average


def compute_movement_energy(ax, ay, az, fs_acc, smooth_window_sec):
    """Accelerometer jerk-energy envelope (computeMovementEnergy @1415).

    jerk[i] = sqrt(dax^2 + day^2 + daz^2) * fs   (scaled to g/s),
    then smoothed with a moving average. energy[0] = energy[1].
    """
    ax = np.asarray(ax, np.float64); ay = np.asarray(ay, np.float64); az = np.asarray(az, np.float64)
    n = ax.size
    energy = np.zeros(n)
    dx = np.diff(ax); dy = np.diff(ay); dz = np.diff(az)
    energy[1:] = np.sqrt(dx * dx + dy * dy + dz * dz) * fs_acc
    if n > 1:
        energy[0] = energy[1]
    win = max(3, int(round(smooth_window_sec * fs_acc)))
    return moving_average(energy, win)


def compute_activity_energy(activity, fs_act, smooth_window_sec):
    """REMbo activity index -> smoothed energy (computeActivityEnergy @3278)."""
    win = max(3, int(round(smooth_window_sec * fs_act)))
    return moving_average(np.asarray(activity, np.float64), win)


def compute_noise_regions(move_time, move_energy, move_threshold, min_clean_sec):
    """Movement regions + bridge gaps + merged noise (computeNoiseRegions @1437).

    Returns dict: move_regions, bridge_regions, merged_noise
    (each a list of (start, end) tuples in seconds).
    """
    t = np.asarray(move_time, np.float64)
    e = np.asarray(move_energy, np.float64)

    # Step 1: contiguous stretches above threshold.
    move_regions = []
    in_region = False
    region_start = 0.0
    for i in range(t.size):
        if e[i] > move_threshold:
            if not in_region:
                region_start = t[i]; in_region = True
        else:
            if in_region:
                move_regions.append((region_start, t[i])); in_region = False
    if in_region:
        move_regions.append((region_start, t[-1]))

    # Step 2: bridge gaps — clean stretches shorter than min_clean_sec.
    bridge_regions = []
    for i in range(len(move_regions) - 1):
        gap_start = move_regions[i][1]
        gap_end = move_regions[i + 1][0]
        if gap_end - gap_start < min_clean_sec:
            bridge_regions.append((gap_start, gap_end))

    # Step 3: merge into non-overlapping noise intervals.
    merged = _merge_intervals(move_regions + bridge_regions)
    return {"move_regions": move_regions,
            "bridge_regions": bridge_regions,
            "merged_noise": merged}


def merge_intervals(intervals):
    return _merge_intervals(intervals)


def _merge_intervals(intervals):
    if not intervals:
        return []
    ivs = sorted(intervals, key=lambda r: r[0])
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def otsu_threshold(values):
    """Otsu on a 256-bin histogram (_otsuThreshold @3287)."""
    v = np.asarray(values, np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    mn, mx = v.min(), v.max()
    if not (mx > mn):
        return float(mn)
    NBINS = 256
    bin_w = (mx - mn) / NBINS
    b = np.floor((v - mn) / bin_w).astype(int)
    np.clip(b, 0, NBINS - 1, out=b)
    hist = np.bincount(b, minlength=NBINS).astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(NBINS), hist)

    w_b = 0.0; sum_b = 0.0; var_max = -1.0; best_bin = 0
    for bb in range(NBINS):
        w_b += hist[bb]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += bb * hist[bb]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > var_max:
            var_max = between; best_bin = bb
    return float(mn + (best_bin + 1) * bin_w)


def auto_activity_threshold(energy):
    """Two-pass Otsu valley for the REMbo activity gate (autoActivityThreshold @3336)."""
    e = np.asarray(energy, np.float64)
    arr = e[np.isfinite(e) & (e >= 0)]
    if arr.size < 100:
        return 3.0
    t1 = otsu_threshold(arr)
    tail = arr[arr >= t1]
    if tail.size < 50:
        return t1
    t2 = otsu_threshold(tail)
    return t2 if t2 > t1 else t1
