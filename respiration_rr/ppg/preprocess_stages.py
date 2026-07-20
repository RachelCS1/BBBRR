"""
Instrumented watch preprocessing — exposes every intermediate stage so the
preprocessing can be inspected and tuned.

This recomputes the same chain as ppg.preprocess.prepare_watch, but instead of
returning only the final channels it returns each stage as a labelled trace:

  0. Raw           — channel resampled from the file rate to FS_ORIG (256 Hz)
  1. Inverted      — sign flip (PPG upstrokes point up)              [if cfg.invert_ppg]
  2. Upsampled     — FFT super-resolution x4 -> 1024 Hz, edge-trimmed
  3. HP baseline   — the long-MA baseline (LP @ ppg_hp_hz) that gets subtracted
  4. High-passed   — upsampled - baseline
  5. Bandpass      — final 0.5-4 Hz band-passed trace (what beat detection sees)

Everything is a thin wrapper over the real building blocks in
preprocessing.resample and ppg.dsp, so what you see is exactly what the pipeline
does. Tune via settings.PPG (ppg_hp_hz, ppg_lp_hz, ppg_filter_order,
upsample_factor, trim_*), then re-run.
"""

import numpy as np

from ..settings import PPG
from ..preprocessing.resample import resample_linear, upsample_fft
from .dsp import lowpass_filter, bandpass_filter
from .beats import (
    detect_beats, detect_beats_lpf_derivative, refine_ss_by_derivative,
    remove_dc_beat_aligned,
)
from .systolic import compute_systolic_analysis


def compute_channel_stages(raw_signal, src_fs, cfg=PPG, invert=None, base_color="#22c55e"):
    """Return the ordered list of preprocessing stages for one channel.

    Each stage is a dict: {key, title, ylabel, traces:[{label, t, y, color, lw, alpha}]}.
    Traces carry their own time axis (256 Hz stages vs 1024 Hz stages differ).
    """
    if invert is None:
        invert = cfg.invert_ppg

    fs0 = cfg.fs_orig
    fs1 = cfg.target_fs

    # 0) raw @ 256 Hz
    raw = resample_linear(raw_signal, src_fs, fs0)
    t0 = np.arange(raw.size) / fs0

    # 1) inversion
    inv = -raw if invert else raw.copy()

    # 2) FFT upsample x4 -> 1024 Hz, then edge trim
    up_full = upsample_fft(inv, cfg.upsample_factor)
    n = up_full.size
    h = int(round(cfg.trim_head_sec * fs1))
    tl = int(round(cfg.trim_tail_sec * fs1))
    lo, hi = h, max(h, n - tl)
    up = up_full[lo:hi]
    t1 = np.arange(lo, hi) / fs1

    # 3-5) MA-cascade bandpass intermediates (matches dsp.bandpass_filter internals)
    baseline = lowpass_filter(up, fs1, cfg.ppg_hp_hz, cfg.ppg_filter_order)
    bp = bandpass_filter(up, fs1, cfg.ppg_hp_hz, cfg.ppg_lp_hz, cfg.ppg_filter_order)
    high_passed = bp["hp"]
    filtered = bp["filtered"]

    faint = base_color
    stages = [
        {"key": "raw", "title": f"0 · Raw (file → {fs0:.0f} Hz)",
         "ylabel": "ADC",
         "traces": [{"label": "raw", "t": t0, "y": raw, "color": "#8a93a6", "lw": 0.6, "alpha": 1}]},
        {"key": "inv", "title": "1 · Inverted (×−1)" if invert else "1 · (no inversion)",
         "ylabel": "ADC",
         "traces": [{"label": "inverted", "t": t0, "y": inv, "color": faint, "lw": 0.6, "alpha": 1}]},
        {"key": "up", "title": f"2 · FFT upsample ×{cfg.upsample_factor} → {fs1:.0f} Hz + trim "
                               f"({cfg.trim_head_sec:.0f}s/{cfg.trim_tail_sec:.0f}s)",
         "ylabel": "ADC",
         "traces": [{"label": "upsampled", "t": t1, "y": up, "color": faint, "lw": 0.5, "alpha": 1}]},
        {"key": "baseline", "title": f"3 · HP baseline (LP @ {cfg.ppg_hp_hz} Hz, order {cfg.ppg_filter_order}) — drift being removed",
         "ylabel": "ADC",
         "traces": [
             {"label": "upsampled", "t": t1, "y": up, "color": "#c2cad9", "lw": 0.5, "alpha": 0.8},
             {"label": "baseline", "t": t1, "y": baseline, "color": "#ef4444", "lw": 1.2, "alpha": 1}]},
        {"key": "hp", "title": f"4 · High-passed (signal − baseline)",
         "ylabel": "a.u.",
         "traces": [{"label": "high-passed", "t": t1, "y": high_passed, "color": "#f59e0b", "lw": 0.5, "alpha": 1}]},
        {"key": "filtered", "title": f"5 · Bandpass {cfg.ppg_hp_hz}–{cfg.ppg_lp_hz} Hz (final — beat detection input)",
         "ylabel": "a.u.",
         "traces": [{"label": "bandpass", "t": t1, "y": filtered, "color": faint, "lw": 0.7, "alpha": 1}]},
    ]
    return stages


def compute_channel_stages_full(raw_signal, src_fs, cfg=PPG, invert=None, base_color="#22c55e"):
    """Preprocessing stages + beat-finding stages for one channel.

    Returns (pre_stages, beat_stages). Convenience wrapper so callers can show
    both groups; the final band-passed trace from preprocessing feeds beats.
    """
    pre = compute_channel_stages(raw_signal, src_fs, cfg, invert=invert, base_color=base_color)
    filtered = pre[-1]["traces"][0]["y"]          # stage 5 (bandpass)
    t = pre[-1]["traces"][0]["t"]
    beats = compute_beat_stages(filtered, t, cfg, base_color=base_color)
    return pre, beats


def compute_beat_stages(filtered, t, cfg=PPG, base_color="#22c55e", move_regions=None):
    """Beat / peak-finding stages on the final band-passed trace.

    Shows BOTH beat-detection methods so their difference is visible:
      6. Trough method (detect_beats): candidates → kept (min-dist+dicrotic+upslope)
      7. LPF-derivative method: d/dt (LP) + its positive peaks
      8. Beats used + refined SS  (which method feeds downstream depends on
         cfg.use_lpf_derivative_beats)
      9. Baseline-corrected + SS / SE / MSD fiducials
    """
    fs = cfg.target_fs
    x = np.asarray(filtered, np.float64)
    t = np.asarray(t, np.float64)

    def _pts(idx, arr):
        idx = np.asarray(idx, int)
        idx = idx[(idx >= 0) & (idx < arr.size)]
        return t[idx], arr[idx]

    # --- trough method (detect_beats cascade) ---
    trough_final, dbg = detect_beats(x, fs, cfg.hr_min_bpm, cfg.hr_max_bpm,
                                    return_stages=True, cfg=cfg)
    minima = np.asarray(dbg["minima"], int)
    trough_final = np.asarray(trough_final, int)

    # --- LPF-derivative method (the HTML override) ---
    lpf_beats, ldbg = detect_beats_lpf_derivative(x, fs, cfg, time=t,
                                                  move_regions=move_regions, return_stages=True)
    lpf_beats = np.asarray(lpf_beats, int)
    deriv_lpf = ldbg["deriv_lpf"]
    deriv_peaks = np.asarray(ldbg["deriv_peaks"], int)

    active = "LPF-derivative" if cfg.use_lpf_derivative_beats else "trough"
    beats = lpf_beats if cfg.use_lpf_derivative_beats else trough_final
    refined = np.asarray(refine_ss_by_derivative(list(beats), x, fs), int) if beats.size > 2 else beats

    bc = remove_dc_beat_aligned(x, list(refined)) if refined.size else np.zeros_like(x)
    sa = compute_systolic_analysis(bc, t, list(refined), fs, cfg.msd_min_ms, cfg.msd_min_pct_d1) \
        if refined.size > 1 else None

    mt_all, my_all = _pts(minima, x)
    mt_fin, my_fin = _pts(trough_final, x)
    dp_t, dp_y = _pts(deriv_peaks, deriv_lpf)
    lb_t, lb_y = _pts(lpf_beats, x)
    rt, ry = _pts(refined, x)

    stages = [
        {"key": "troughs", "ylabel": "a.u.",
         "title": f"6 · Trough method (detect_beats) — {minima.size} candidates → "
                  f"{trough_final.size} kept" + ("" if active == "trough" else "  [not used]"),
         "traces": [{"label": "bandpass", "t": t, "y": x, "color": base_color, "lw": 0.6, "alpha": 0.9}],
         "markers": [
             {"label": f"candidates ({minima.size})", "t": mt_all, "y": my_all,
              "color": "#c2cad9", "marker": "o", "size": 12},
             {"label": f"kept ({trough_final.size})", "t": mt_fin, "y": my_fin,
              "color": "#ef4444", "marker": "v", "size": 30}]},
        {"key": "lpfderiv", "ylabel": "d/dt",
         "title": f"7 · LPF-derivative method — d/dt (LP {cfg.deriv_lpf_hz} Hz) + "
                  f"{deriv_peaks.size} positive peaks" + (" [ACTIVE]" if active != "trough" else ""),
         "traces": [{"label": f"d/dt (LP {cfg.deriv_lpf_hz} Hz)", "t": t, "y": deriv_lpf,
                     "color": "#f59e0b", "lw": 0.5, "alpha": 1}],
         "markers": [{"label": f"upslope peaks ({deriv_peaks.size})", "t": dp_t, "y": dp_y,
                      "color": "#2563eb", "marker": "o", "size": 16}]},
        {"key": "beats", "ylabel": "a.u.",
         "title": f"8 · Beats used ({active}, {beats.size}) snapped to trough + refined SS ({refined.size})",
         "traces": [{"label": "bandpass", "t": t, "y": x, "color": base_color, "lw": 0.6, "alpha": 0.7}],
         "markers": [
             {"label": f"beats ({active})", "t": lb_t if active != "trough" else mt_fin,
              "y": lb_y if active != "trough" else my_fin, "color": "#c2cad9", "marker": "v", "size": 20},
             {"label": "refined SS", "t": rt, "y": ry, "color": "#22c55e", "marker": "^", "size": 32}]},
    ]

    if sa is not None:
        ss_t, ss_y = _pts(sa.ss_idx, bc)
        se_t, se_y = _pts(sa.se_idx, bc)
        msd_t, msd_y = _pts(sa.msd_idx, bc)
        stages.append({
            "key": "systolic", "ylabel": "a.u.",
            "title": f"9 · Baseline-corrected + SS / SE / MSD  ({len(sa.ss_idx)} beats)",
            "traces": [{"label": "BC", "t": t, "y": bc, "color": base_color, "lw": 0.7, "alpha": 0.9}],
            "markers": [
                {"label": "SS", "t": ss_t, "y": ss_y, "color": "#22c55e", "marker": "^", "size": 28},
                {"label": "SE", "t": se_t, "y": se_y, "color": "#ef4444", "marker": "o", "size": 20},
                {"label": "MSD", "t": msd_t, "y": msd_y, "color": "#a855f7", "marker": "D", "size": 20}]})
    return stages


def compute_movement_stage(watch, cfg=PPG):
    """Accelerometer preprocessing (HTML-faithful): jerk energy on the 256 Hz
    accel, energy FFT-upsampled + smoothed at 1024 Hz, plus the movement regions.

    Returns dict {t, ax, ay, az, energy, threshold, regions} on the 1024 Hz
    trimmed timeline.
    """
    from .preprocess import _watch_movement_energy, build_move_regions
    src_fs = watch["fs"]
    if not all(c in watch for c in ("acc_x", "acc_y", "acc_z")):
        return None
    acc256 = {c: resample_linear(watch[c], src_fs, cfg.fs_orig)
              for c in ("acc_x", "acc_y", "acc_z")}
    acc_up = {c: upsample_fft(acc256[c], cfg.upsample_factor) for c in acc256}
    energy_full = _watch_movement_energy(acc256, cfg)

    n = min(min(v.size for v in acc_up.values()), energy_full.size)
    fs1 = cfg.target_fs
    h = int(round(cfg.trim_head_sec * fs1))
    tl = int(round(cfg.trim_tail_sec * fs1))
    lo, hi = h, max(h, n - tl)
    t = np.arange(lo, hi) / fs1
    energy = energy_full[lo:hi]
    regions = build_move_regions(t, energy, cfg.move_thresh_gs, cfg)
    return {"t": t, "ax": acc_up["acc_x"][lo:hi], "ay": acc_up["acc_y"][lo:hi],
            "az": acc_up["acc_z"][lo:hi], "energy": energy,
            "threshold": cfg.move_thresh_gs, "regions": regions}
