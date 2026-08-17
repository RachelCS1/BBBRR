"""
Reference analyzer — Tool 2 (REMbo EDF + Polysomnograph CSV).

Breath-by-breath respiration rate from a nasal-pressure airflow signal via
hysteresis zero-crossing detection, plus the four breath parameters
(rate, I:E ratio, RRV, and per-breath error vs the device reference) and
agreement statistics.

Faithful ports of:
  findBreathCrossings @1499, detectBreaths @1575, computeIeRatios @2344,
  plotRRVGraph (variance) @3089, computeBreathMetrics @2405, runAnalysis @4255.
"""

from dataclasses import dataclass, field
import numpy as np

from ..settings import REFERENCE
from ..preprocessing.filters import bandpass_ma_cascade
from ..preprocessing.movement import (
    compute_movement_energy, compute_activity_energy,
    compute_noise_regions, auto_activity_threshold, merge_intervals,
)


@dataclass
class Breath:
    start: float
    end: float
    center: float
    duration: float
    rate: float
    exhale_dur: float = float("nan")
    inhale_dur: float = float("nan")
    ie_ratio: float = float("nan")


@dataclass
class ReferenceResult:
    time: np.ndarray
    raw: np.ndarray
    filtered: np.ndarray
    baseline: np.ndarray
    high_passed: np.ndarray
    fs: float
    local_rms: np.ndarray
    crossings: np.ndarray            # crossing times (s)
    breaths: list                    # list[Breath]
    move_time: np.ndarray
    move_energy: np.ndarray
    move_threshold: float
    move_regions: list
    bridge_regions: list
    merged_noise: list
    rrv_time: np.ndarray
    rrv_value: np.ndarray
    metrics: dict = field(default=None)
    pairs: list = field(default_factory=list)          # {time, computed, device}
    avg_pairs: list = field(default_factory=list)      # 30-s averaged {computed, device}
    ref_time: np.ndarray = None
    ref_rate: np.ndarray = None


# ----------------------------------------------------------------------
# Breath detection
# ----------------------------------------------------------------------
def find_breath_crossings(time, signal, fs, rms_window_sec=None, hyst_frac=None):
    """Hysteresis / Schmitt-trigger positive-going zero crossings (findBreathCrossings @1499).

    Symmetric Schmitt trigger: a positive-going zero crossing is accepted as a
    breath boundary only when a full swing through BOTH rails is confirmed --
    the signal dipped below -(hyst_frac * local_RMS) (a real inhale) AND, after
    the crossing, rises above +(hyst_frac * local_RMS) (a real exhale) before
    it dips back below -thresh. A shallow mid-breath notch that briefly crosses
    zero but never reaches +thresh is discarded, so one breath is not split into
    two. Crossing time is linearly interpolated. Returns (crossing_times, local_rms).
    """
    if rms_window_sec is None:
        rms_window_sec = REFERENCE.rms_window_sec
    if hyst_frac is None:
        hyst_frac = REFERENCE.hyst_frac

    x = np.asarray(signal, np.float64)
    t = np.asarray(time, np.float64)
    n = x.size

    # Local RMS over a centered window (edge-shrinking), O(n) via cumsum of squares.
    win = int(round(rms_window_sec * fs))
    half = win // 2
    cum_sq = np.empty(n + 1); cum_sq[0] = 0.0
    np.cumsum(x * x, out=cum_sq[1:])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n - 1, idx + half)
    local_rms = np.sqrt((cum_sq[hi + 1] - cum_sq[lo]) / (hi - lo + 1))

    crossings = []
    state = "seek_neg"             # seek_neg -> seek_up -> confirm_pos -> seek_neg
    pending = np.nan               # tentative up-crossing time awaiting +thresh confirmation
    for i in range(1, n):
        thresh = local_rms[i] * hyst_frac
        if state == "seek_neg":
            # arm: require a real inhale (dip below the negative rail)
            if x[i] < -thresh:
                state = "seek_up"
        elif state == "seek_up":
            # look for a positive-going zero crossing; record it tentatively
            if x[i - 1] <= 0 and x[i] > 0:
                frac = abs(x[i - 1]) / abs(x[i] - x[i - 1])
                pending = t[i - 1] + frac * (t[i] - t[i - 1])
                state = "confirm_pos"
        else:  # confirm_pos: is the positive lobe real, or a mid-breath notch?
            if x[i] > thresh:
                # real exhale -> commit the boundary
                crossings.append(pending)
                pending = np.nan
                state = "seek_neg"
            elif x[i] < -thresh:
                # notch: fell back below the negative rail before a real exhale;
                # discard the tentative crossing and keep seeking the real one
                pending = np.nan
                state = "seek_up"
    return np.asarray(crossings, np.float64), local_rms


def detect_breaths(crossings, min_duration, max_duration, merged_noise):
    """Pair adjacent crossings into breaths, rejecting out-of-range / noisy ones
    (detectBreaths @1575). rate = 60/duration."""
    breaths = []
    noise = merged_noise or []

    def overlaps_noise(start, end):
        for (ns, ne) in noise:
            if ns > end:
                break
            if ne >= start:
                return True
        return False

    c = np.asarray(crossings, np.float64)
    for i in range(len(c) - 1):
        start, end = c[i], c[i + 1]
        dur = end - start
        if dur < min_duration or dur > max_duration:
            continue
        if noise and overlaps_noise(start, end):
            continue
        breaths.append(Breath(start=start, end=end, center=(start + end) / 2,
                              duration=dur, rate=60.0 / dur))
    return breaths


def compute_ie_ratios(breaths, time, filtered, fs):
    """Inhale/exhale split per breath (computeIeRatios @2344).

    Within a breath (pos-going crossing to next), the exhale lobe is positive
    and the inhale lobe negative; boundary = first downward zero crossing after
    the signal has been positive. Mutates each Breath in place.
    """
    if not breaths:
        return
    x = np.asarray(filtered, np.float64)
    t = np.asarray(time, np.float64)
    t0 = t[0]
    n = x.size
    for b in breaths:
        k_lo = max(0, int(round((b.start - t0) * fs)))
        k_hi = min(n - 1, int(round((b.end - t0) * fs)))
        if k_hi <= k_lo + 2:
            continue
        down_cross = -1
        was_positive = False
        for k in range(k_lo, k_hi + 1):
            v = x[k]
            if v > 0:
                was_positive = True
            if down_cross < 0 and was_positive and v <= 0:
                down_cross = k
                break
        if down_cross < 0 or not was_positive:
            continue
        exh = t[down_cross] - b.start
        inh = b.end - t[down_cross]
        if exh <= 0 or inh <= 0:
            continue
        b.exhale_dur = exh
        b.inhale_dur = inh
        b.ie_ratio = inh / exh


def compute_rrv(breaths, window_breaths=None, gap_thresh_sec=None):
    """Moving-variance respiration-rate variability (plotRRVGraph @3089).

    Sample variance (ddof=1) of `rate` over W consecutive valid breaths,
    plotted at the window's mid-time. A NaN is inserted where the gap between
    consecutive window centers exceeds gap_thresh_sec (line break across gaps).
    Returns (x, y) arrays.
    """
    if window_breaths is None:
        window_breaths = REFERENCE.rrv_window_breaths
    if gap_thresh_sec is None:
        gap_thresh_sec = REFERENCE.rrv_gap_thresh_sec
    W = max(2, int(window_breaths))

    cx = np.array([b.center for b in breaths if np.isfinite(b.rate) and np.isfinite(b.center)])
    cr = np.array([b.rate for b in breaths if np.isfinite(b.rate) and np.isfinite(b.center)])

    x, y = [], []
    if cr.size >= W:
        prev_x = None
        for i in range(W - 1, cr.size):
            lo = i - W + 1
            window = cr[lo:i + 1]
            var = float(np.var(window, ddof=1))
            x_mid = (cx[lo] + cx[i]) / 2
            if prev_x is not None and (x_mid - prev_x) > gap_thresh_sec:
                x.append((prev_x + x_mid) / 2); y.append(np.nan)
            x.append(x_mid); y.append(var)
            prev_x = x_mid
    return np.asarray(x), np.asarray(y)


# ----------------------------------------------------------------------
# Agreement vs device reference
# ----------------------------------------------------------------------
def compute_breath_metrics(breaths, ref_time, ref_rate, ref_fs, time_shift):
    """Breath-by-breath agreement vs device reference (computeBreathMetrics @2405).

    For each breath, look up the device rate at (center + time_shift) via
    nearest-neighbour. Skip stale (frozen >= stale_run_sec) and out-of-range
    device values. Also compute a 30-s sliding-window averaged MAE/RMSE.

    Returns dict {metrics, pairs, avg_pairs}. metrics is None if no reference.
    """
    if ref_time is None or ref_rate is None or len(ref_time) == 0 or len(ref_rate) == 0:
        return {"metrics": None, "pairs": [], "avg_pairs": []}

    ref_time = np.asarray(ref_time, np.float64)
    ref_rate = np.asarray(ref_rate, np.float64)
    stale_run = int(round(REFERENCE.stale_run_sec * ref_fs))

    # Stale mask: runs of identical value >= stale_run samples.
    stale_mask = np.zeros(ref_rate.size, dtype=bool)
    run_start = 0
    for i in range(1, ref_rate.size + 1):
        if i < ref_rate.size and ref_rate[i] == ref_rate[run_start]:
            continue
        if i - run_start >= stale_run:
            stale_mask[run_start:i] = True
        run_start = i

    gmin, gmax = REFERENCE.ref_gate_min_bpm, REFERENCE.ref_gate_max_bpm
    within = REFERENCE.within_bpm
    pairs = []
    errs = []
    cs, rs = [], []
    n_within = 0
    for b in breaths:
        t = b.center + (time_shift or 0.0)
        c = b.rate
        if t < ref_time[0] or t > ref_time[-1]:
            continue
        j = int(np.searchsorted(ref_time, t))
        if j >= ref_time.size:
            j = ref_time.size - 1
        if j > 0 and abs(ref_time[j - 1] - t) < abs(ref_time[j] - t):
            j -= 1
        r = ref_rate[j]
        if stale_mask[j]:
            continue
        if r > gmax or r < gmin:
            continue
        err = c - r
        errs.append(err)
        cs.append(c); rs.append(r)
        if abs(err) <= within:
            n_within += 1
        pairs.append({"time": b.center, "computed": c, "device": r})

    n = len(errs)
    if n == 0:
        return {"metrics": None, "pairs": pairs, "avg_pairs": []}
    errs = np.asarray(errs)
    mae_bbb = float(np.mean(np.abs(errs)))
    bbb_rmse = float(np.sqrt(np.mean(errs ** 2)))
    mean_c = float(np.mean(cs)); mean_r = float(np.mean(rs))
    pct_within = 100.0 * n_within / n

    # 30-s sliding-window averaged agreement.
    pairs.sort(key=lambda p: p["time"])
    ptime = np.array([p["time"] for p in pairs])
    pc = np.array([p["computed"] for p in pairs])
    pr = np.array([p["device"] for p in pairs])
    WIN = REFERENCE.avg_window_sec; STEP = REFERENCE.avg_step_sec
    avg_pairs = []
    win_errs = []
    if ptime.size:
        w = ptime[0]
        t_end = ptime[-1]
        while w <= t_end:
            lo = np.searchsorted(ptime, w - WIN / 2, side="left")
            hi = np.searchsorted(ptime, w + WIN / 2, side="left")
            if hi > lo:
                avg_c = float(pc[lo:hi].mean()); avg_r = float(pr[lo:hi].mean())
                win_errs.append(avg_c - avg_r)
                avg_pairs.append({"computed": avg_c, "device": avg_r})
            w += STEP
    if win_errs:
        win_errs = np.asarray(win_errs)
        mae30 = float(np.mean(np.abs(win_errs)))
        ave_rmse = float(np.sqrt(np.mean(win_errs ** 2)))
    else:
        mae30, ave_rmse = mae_bbb, bbb_rmse

    metrics = {"mae30": mae30, "maeBBB": mae_bbb, "bbbRmse": bbb_rmse,
               "aveRmse": ave_rmse, "pctWithin": pct_within, "n": n,
               "meanC": mean_c, "meanR": mean_r}
    return {"metrics": metrics, "pairs": pairs, "avg_pairs": avg_pairs}


# ----------------------------------------------------------------------
# Full pipeline (mirrors runAnalysis @4255)
# ----------------------------------------------------------------------
def analyze_reference(airflow_time, airflow_signal, fs,
                      move_time=None, move_energy=None,
                      accel=None, activity=None, activity_fs=None,
                      ref_time=None, ref_rate=None, ref_fs=25.0,
                      user_noise=None, cfg=REFERENCE):
    """Run the full reference pipeline on an airflow signal.

    Provide movement info in ONE of these ways:
      - move_time + move_energy (precomputed), or
      - accel = (ax, ay, az) with matching sample rate == activity_fs (CSV mode), or
      - activity + activity_fs (REMbo EDF activity channel), or
      - nothing (movement gating disabled).

    ref_time/ref_rate enable agreement metrics (CSV/Poly mode). In EDF mode
    they are absent and metrics come back None.
    """
    airflow_time = np.asarray(airflow_time, np.float64)
    airflow_signal = np.asarray(airflow_signal, np.float64)

    # ---- Movement energy / threshold ----
    move_threshold = cfg.move_thresh_gs
    if move_time is None or move_energy is None:
        if activity is not None:
            move_energy = compute_activity_energy(activity, activity_fs, cfg.move_smooth_sec)
            move_time = np.arange(move_energy.size) / activity_fs
            if cfg.activity_mode == "auto":
                move_threshold = auto_activity_threshold(move_energy)
            else:
                move_threshold = cfg.activity_thresh_manual
        elif accel is not None:
            ax, ay, az = accel
            move_energy = compute_movement_energy(ax, ay, az, activity_fs, cfg.move_smooth_sec)
            move_time = np.arange(move_energy.size) / activity_fs
            move_threshold = cfg.move_thresh_gs
        else:
            move_energy = np.zeros(airflow_time.size)
            move_time = airflow_time
            move_threshold = cfg.move_thresh_gs

    # ---- Bandpass ----
    filtered, baseline, high_passed = bandpass_ma_cascade(
        airflow_signal, fs, cfg.hp_cutoff_hz, cfg.lp_cutoff_hz)

    # ---- Noise regions (auto + user) ----
    nr = compute_noise_regions(move_time, move_energy, move_threshold, cfg.min_clean_sec)
    merged_noise = nr["merged_noise"]
    if user_noise:
        merged_noise = merge_intervals(list(merged_noise) + list(user_noise))

    # ---- Breath detection ----
    crossings, local_rms = find_breath_crossings(airflow_time, filtered, fs)
    min_breath = 60.0 / cfg.max_rate_bpm
    max_breath = 60.0 / cfg.min_rate_bpm
    breaths = detect_breaths(crossings, min_breath, max_breath, merged_noise)

    # ---- I:E ratio ----
    compute_ie_ratios(breaths, airflow_time, filtered, fs)

    # ---- RRV ----
    rrv_x, rrv_y = compute_rrv(breaths)

    # ---- Agreement ----
    m = compute_breath_metrics(breaths, ref_time, ref_rate, ref_fs, cfg.time_shift_sec)

    return ReferenceResult(
        time=airflow_time, raw=airflow_signal, filtered=filtered,
        baseline=baseline, high_passed=high_passed, fs=fs, local_rms=local_rms,
        crossings=crossings, breaths=breaths,
        move_time=move_time, move_energy=move_energy, move_threshold=move_threshold,
        move_regions=nr["move_regions"], bridge_regions=nr["bridge_regions"],
        merged_noise=merged_noise,
        rrv_time=rrv_x, rrv_value=rrv_y,
        metrics=m["metrics"], pairs=m["pairs"], avg_pairs=m["avg_pairs"],
        ref_time=ref_time, ref_rate=ref_rate,
    )
