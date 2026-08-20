"""
Averaged RR — a windowed-median smoothing of each per-breath RR series.

"Average RR" here means: slide a time window over a per-breath RR series and, at
each step, take the median of the RR points inside the window. It is a temporal
smoothing of ONE series — it does NOT pool different series together.

It is applied to every watch per-beat series individually, so per channel
(wavelength) you get all THREE parameters in all THREE methods = 9 averaged
curves:

    parameter : RSA (pulse) · RIIV (amplitude) · AUC (area)
    method    : source (linear+BP) · spline (cubic) · ssp (smoothing spline)

and to the REMbo reference's single per-breath RR series (one averaged curve),
using the SAME window/stride/min-points/method (settings.PPG.rr_avg_*), so watch
and reference are averaged identically and stay comparable. The watch lives on
its own clock and the reference on the REMbo clock; shift the watch average by
the IR-PPG sync offset before comparing (the runner does this).

LP (BW), BWlegacy and BWbank are per-signal traces, not per-beat, so they are
excluded. This is INDEPENDENT of the MAE30 aggregation in reference/airflow.py
(which averages the agreement ERROR for a summary metric; left untouched).
"""

import numpy as np

from .settings import PPG

# The three per-beat base parameters and the three envelope methods.
PER_BEAT_BASE = ("RSA", "RIIV", "AUC")
METHOD_SUFFIXES = ("", "_spline", "_ssp")

PARAM_LABEL = {"RSA": "Pulse (RSA)", "RIIV": "Amplitude (RIIV)", "AUC": "Area (AUC)"}
METHOD_LABEL = {"": "source", "_spline": "spline", "_ssp": "ssp"}


def param_methods(base):
    """The three method series names for a base param, e.g. RSA -> (RSA, RSA_spline, RSA_ssp)."""
    return tuple(base + s for s in METHOD_SUFFIXES)


def windowed_median(times, values, window_sec, step_sec, min_pts, method="median"):
    """Sliding-window average of a scattered (time, value) series.

    For each grid time t (from first to last point, step `step_sec`), collect all
    points with time in [t - window_sec/2, t + window_sec/2] and emit their median
    (or mean) when at least `min_pts` fall inside; otherwise NaN, so gaps break the
    line instead of being bridged. Returns (grid_t, averaged) — `averaged` may hold
    NaNs. Empty input returns two empty arrays.
    """
    t = np.asarray(times, np.float64)
    v = np.asarray(values, np.float64)
    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m]
    if t.size == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]

    half = window_sec / 2.0
    grid = np.arange(t[0], t[-1] + step_sec * 0.5, step_sec)   # inclusive of t[-1]
    if grid.size == 0:
        grid = np.array([t[0]], np.float64)

    lo = np.searchsorted(t, grid - half, side="left")
    hi = np.searchsorted(t, grid + half, side="right")
    reducer = np.mean if method == "mean" else np.median
    out = np.full(grid.size, np.nan)
    for i in range(grid.size):
        if hi[i] - lo[i] >= min_pts:
            out[i] = reducer(v[lo[i]:hi[i]])
    return grid, out


def average_series(times, values, cfg=PPG):
    """windowed_median with the shared settings knobs (rr_avg_*)."""
    return windowed_median(times, values, cfg.rr_avg_window_sec, cfg.rr_avg_step_sec,
                           cfg.rr_avg_min_pts, cfg.rr_avg_method)


def average_watch_param(channel_result, param_name, cfg=PPG):
    """Averaged RR (WATCH clock) for ONE per-beat series of a channel, e.g. 'RSA_ssp'.

    Returns (grid_t, rr_bpm); shift grid_t by the sync offset to reach the REMbo
    clock. Returns empty arrays if that series is absent/empty.
    """
    pr = channel_result.params.get(param_name)
    if pr is None or np.size(pr.rr_bpm) == 0:
        return np.zeros(0), np.zeros(0)
    return average_series(pr.rr_time, pr.rr_bpm, cfg)


def average_watch_channel(channel_result, cfg=PPG):
    """All 9 averaged curves for a channel.

    Returns {base: {suffix: (grid_t, rr_bpm)}} for base in PER_BEAT_BASE and
    suffix in METHOD_SUFFIXES (only series that exist and are non-empty).
    """
    out = {}
    for base in PER_BEAT_BASE:
        methods = {}
        for suf in METHOD_SUFFIXES:
            t, r = average_watch_param(channel_result, base + suf, cfg)
            if t.size:
                methods[suf] = (t, r)
        if methods:
            out[base] = methods
    return out


def reference_rr_points(ref_result):
    """(center_time, rate) of the accepted reference breaths, sorted by time."""
    from .compare.compare import reference_rr_series
    return reference_rr_series(ref_result)


def reference_average_rr(ref_result, cfg=PPG):
    """Averaged reference RR (REMbo clock), same method as the watch series."""
    t, r = reference_rr_points(ref_result)
    return average_series(t, r, cfg)


def mae_over_overlap(t_cand, v_cand, t_ref, v_ref, min_overlap_sec=20.0):
    """MAE (bpm) of a candidate averaged-RR curve vs a reference averaged-RR curve,
    on a shared clock, over the span where both are defined.

    Both inputs may carry NaNs; they are dropped first. The candidate is compared
    to the reference linearly interpolated at the candidate's times, restricted to
    the reference's time span. Returns (mae, n_overlap); (nan, 0) if too short.
    """
    tc = np.asarray(t_cand, np.float64)
    vc = np.asarray(v_cand, np.float64)
    tr = np.asarray(t_ref, np.float64)
    vr = np.asarray(v_ref, np.float64)
    mc = np.isfinite(tc) & np.isfinite(vc)
    mr = np.isfinite(tr) & np.isfinite(vr)
    tc, vc, tr, vr = tc[mc], vc[mc], tr[mr], vr[mr]
    if tc.size == 0 or tr.size < 2:
        return float("nan"), 0
    inside = (tc >= tr[0]) & (tc <= tr[-1])
    if inside.sum() < 2:
        return float("nan"), int(inside.sum())
    span = tc[inside].max() - tc[inside].min()
    if span < min_overlap_sec:
        return float("nan"), int(inside.sum())
    ref_at = np.interp(tc[inside], tr, vr)
    mae = float(np.mean(np.abs(vc[inside] - ref_at)))
    return mae, int(inside.sum())
