"""
PPG respiration estimation — the FOUR parameters + spectral ridge.

For each PPG-type channel (Green, Red, IR, Artifact) the tool derives respiration
rate from four independent per-beat modulation series. Each param also gets its
own STFT spectrogram + respiration ridge (four spectrograms per channel):

  1. RSA  — RR-interval (ms) between consecutive systolic starts   (RSA / RIFV)
  2. RIIV — per-beat max-height amplitude                          (RIIV / RIAV)
  3. AUC  — per-beat area under the baseline-corrected curve
  4. LP   — the raw channel band-passed directly at the RR band

Params 1-3 go through bpRRSeries (resample to 10 Hz -> band-pass -> DC-restore),
then breath-starts are the prominence-filtered envelope peaks. Param 4 detects
peaks directly on the full-rate band-passed raw trace. In every case
RR = 60 / Δt between consecutive breath-starts.

Pipeline ports: runAnalysis Green-RR @9337-9461, Artifact @11841-12060,
bpRRSeries @4755, filterPeaksByProminence @4790.
"""

from dataclasses import dataclass, field
import numpy as np

from ..settings import PPG
from .dsp import bandpass_filter
from .beats import (
    detect_beats, detect_beats_lpf_derivative, remove_dc_beat_aligned,
    refine_ss_by_derivative, post_corr_filter, filter_peaks_by_prominence,
    peak_detection_zero_cross,
)
from .systolic import compute_systolic_analysis
from .spectrogram import compute_spectrogram, ridge_rr


# ----------------------------------------------------------------------
# Core respiration primitives
# ----------------------------------------------------------------------
def bp_rr_series(x, y, f_hp, f_lp, order, resample_fs=None):
    """Resample an irregular (x,y) series to a uniform grid, band-pass it, and
    restore the DC mean (bpRRSeries @4755). Returns (grid_x, filtered_y)."""
    if resample_fs is None:
        resample_fs = PPG.rr_resample_fs
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 4:
        return np.zeros(0), np.zeros(0)
    t_start, t_end = x[0], x[-1]
    dt = 1.0 / resample_fs
    n_grid = max(2, int(np.floor((t_end - t_start) / dt)) + 1)
    grid_t = t_start + np.arange(n_grid) * dt
    # linear interp with edge hold (matches the JS clamp at both ends)
    y_grid = np.interp(grid_t, x, y)
    mean_y = y_grid.mean()
    bp = bandpass_filter(y_grid, resample_fs, f_hp, f_lp, order)["filtered"]
    return grid_t, bp + mean_y


def cubic_spline_rr_series(x, y, resample_fs=None):
    """Resample an irregular (x,y) per-beat series to a uniform grid with a
    CUBIC SPLINE and NO band-pass.

    Alternative to bp_rr_series: the spline replaces BOTH the linear
    interpolation AND the band-pass, so breath-starts are found directly on the
    smooth spline curve. Returns (grid_x, spline_y) on the same 10 Hz grid.
    """
    if resample_fs is None:
        resample_fs = PPG.rr_resample_fs
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 4:
        return np.zeros(0), np.zeros(0)
    # CubicSpline needs strictly-increasing, de-duplicated knots
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if x.size < 4:
        return np.zeros(0), np.zeros(0)
    t_start, t_end = x[0], x[-1]
    dt = 1.0 / resample_fs
    n_grid = max(2, int(np.floor((t_end - t_start) / dt)) + 1)
    grid_t = t_start + np.arange(n_grid) * dt
    from scipy.interpolate import CubicSpline   # lazy: only this path needs scipy
    y_grid = CubicSpline(x, y, extrapolate=False)(grid_t)
    if np.isnan(y_grid).any():                 # guard FP edge just past x[-1]
        nan = np.isnan(y_grid)
        y_grid[nan] = np.interp(grid_t[nan], x, y)
    return grid_t, y_grid


def smoothing_spline_rr_series(x, y, resample_fs=None, lam=None, cutoff_hz=None):
    """Resample an irregular (x,y) per-beat series to a uniform grid with a
    PENALIZED SMOOTHING SPLINE and NO band-pass.

    Unlike cubic_spline_rr_series (which interpolates every point and so keeps
    the beat-to-beat jitter), this fits a spline that trades data-fidelity for
    smoothness, denoising the jitter while preserving the respiratory shape —
    less information loss than the linear+BP band-pass.

    Smoothing is controlled by an effective -3 dB cutoff (cutoff_hz): the cubic
    smoothing spline has transfer H(f)=1/(1+lam*(2*pi*f)^4), so lam=1/(2*pi*fc)^4.
    The series is standardised (unit variance) before fitting so the cutoff↔lam
    mapping is scale-invariant across params/channels. A raw `lam` overrides the
    cutoff. Returns (grid_x, spline_y).
    """
    if resample_fs is None:
        resample_fs = PPG.rr_resample_fs
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size < 4:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(x, kind="stable")           # strictly-increasing, de-duped
    x, y = x[order], y[order]
    keep = np.concatenate(([True], np.diff(x) > 0))
    x, y = x[keep], y[keep]
    if x.size < 4:
        return np.zeros(0), np.zeros(0)
    t_start, t_end = x[0], x[-1]
    dt = 1.0 / resample_fs
    n_grid = max(2, int(np.floor((t_end - t_start) / dt)) + 1)
    grid_t = t_start + np.arange(n_grid) * dt
    if lam is None:                                # derive lam from the cutoff
        fc = cutoff_hz if cutoff_hz else 1.0
        lam = 1.0 / (2.0 * np.pi * fc) ** 4
    mu, sd = y.mean(), y.std()                      # standardise -> scale-invariant lam
    if sd == 0:
        sd = 1.0
    yn = (y - mu) / sd
    from scipy.interpolate import make_smoothing_spline   # lazy: only this path
    try:
        spl = make_smoothing_spline(x, yn, lam=lam)
    except Exception:
        return np.zeros(0), np.zeros(0)
    y_grid = np.asarray(spl(grid_t), np.float64) * sd + mu
    if np.isnan(y_grid).any():                     # guard FP edge / extrapolation
        nan = np.isnan(y_grid)
        y_grid[nan] = np.interp(grid_t[nan], x, y)
    return grid_t, y_grid


def _filter_peaks_prominence_combined(env_y, raw_peaks, global_thr, frac_local, local_win):
    """Prominence filter whose per-peak threshold must clear BOTH an (absolute)
    global floor and a local-relative check:

        thr_i = max(global_thr,  frac_local * local_range_i)

    where local_range_i = max-min in a +-local_win-sample window around peak i.
    Drops the largest-deficit peak iteratively until all survivors clear it."""
    x = np.asarray(env_y, np.float64)
    n = x.size
    thr = {}
    for p in raw_peaks:
        lo = max(0, p - local_win)
        hi = min(n, p + local_win + 1)
        seg = x[lo:hi]
        thr[p] = max(global_thr, frac_local * (seg.max() - seg.min()))
    kept = list(raw_peaks)
    while kept:
        worst_k, worst_deficit = -1, np.inf
        for k in range(len(kept)):
            peak_val = x[kept[k]]
            left_bound = 0 if k == 0 else kept[k - 1]
            right_bound = n - 1 if k == len(kept) - 1 else kept[k + 1]
            left_valley = x[left_bound:kept[k] + 1].min()
            right_valley = x[kept[k]:right_bound + 1].min()
            prom = peak_val - max(left_valley, right_valley)
            deficit = prom - thr[kept[k]]
            if deficit < worst_deficit:
                worst_deficit = deficit
                worst_k = k
        if worst_deficit >= 0:
            break
        kept.pop(worst_k)
    return kept


def _breath_starts_envelope(env_x, env_y, prom_frac, min_dist=10, cfg=None, move_regions=None):
    """Breath-starts = prominence-filtered local maxima on a 10 Hz envelope.

    The global-floor prominence is prom_frac * global range. When
    cfg.rr_local_prom_win_sec and rr_local_prom_frac are set, a peak must ALSO
    clear frac_local * local range (see _filter_peaks_prominence_combined).

    If cfg.rr_prominence_ignore_noise and move_regions are given, the global
    range is measured from noise-free samples only, so a high-amplitude noise
    burst does not inflate the floor and starve real-breath detection."""
    if env_y.size < 4:
        return np.array([], dtype=int)
    raw_peaks = []
    last = -min_dist
    for i in range(1, env_y.size - 1):
        if env_y[i] > env_y[i - 1] and env_y[i] > env_y[i + 1] and i - last >= min_dist:
            raw_peaks.append(i)
            last = i
    if not raw_peaks:
        return np.array([], dtype=int)
    x = np.asarray(env_y, np.float64)
    rng = _prominence_range(x, env_x, cfg, move_regions)
    global_thr = prom_frac * rng                        # manual per-variant floor

    # optional local-relative check
    local_win = local_frac = None
    if cfg is not None:
        w = getattr(cfg, "rr_local_prom_win_sec", None)
        local_win = int(round(w * cfg.rr_resample_fs)) if w else None
        local_frac = getattr(cfg, "rr_local_prom_frac", None)

    if local_win and local_win > 0 and local_frac:
        kept = _filter_peaks_prominence_combined(x, raw_peaks, global_thr, local_frac, int(local_win))
    else:
        kept = filter_peaks_by_prominence(x, raw_peaks, global_thr)
    return np.asarray(kept, dtype=int)


def _prominence_range(y, y_times, cfg, move_regions):
    """Amplitude range used to scale the breath-start prominence floor.

    Two independent, composable safeguards against a high-amplitude artefact
    inflating the floor and starving real-breath detection:

      1. rr_prominence_ignore_noise — drop samples inside a movement region
         before measuring, so a noise burst there does not count.
      2. rr_prominence_robust — measure a ROBUST spread (k * IQR) instead of
         max-min. IQR ignores the outer quartiles, so an edge/startup transient
         or a spline overshoot that survives step 1 (because it sits just
         OUTSIDE the noise window) still cannot inflate the floor. k maps IQR to
         peak-to-peak (√2 for a clean sinusoid).

    Both fall back to the plain global max-min when disabled or degenerate."""
    y = np.asarray(y, np.float64)
    seg = y
    if (move_regions and cfg is not None
            and getattr(cfg, "rr_prominence_ignore_noise", False)):
        clean = ~_noise_flag_from_regions(y_times, move_regions).astype(bool)
        if int(clean.sum()) >= 4:
            seg = y[clean]
    if cfg is not None and getattr(cfg, "rr_prominence_robust", False) and seg.size >= 4:
        q1, q3 = np.percentile(seg, [25.0, 75.0])
        rng = float(getattr(cfg, "rr_prominence_iqr_k", 1.4)) * float(q3 - q1)
        if rng > 0:
            return rng
    return float(seg.max() - seg.min())


def _breath_starts_raw(sig, fs, prom_frac, t=None, cfg=None, move_regions=None):
    """Breath-starts on a full-rate band-passed trace (LP param, @12016).

    When cfg.rr_prominence_ignore_noise and (t, move_regions) are given, the
    prominence-floor range is measured from noise-free samples only."""
    sig = np.asarray(sig, np.float64)
    min_dist = max(1, int(round(fs * 1.0)))
    raw_peaks = []
    last = -min_dist
    for i in range(1, sig.size - 1):
        if sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and i - last >= min_dist:
            raw_peaks.append(i)
            last = i
    if not raw_peaks:
        return np.array([], dtype=int)
    rng = _prominence_range(sig, t, cfg, move_regions) if t is not None else (sig.max() - sig.min())
    return np.asarray(filter_peaks_by_prominence(sig, raw_peaks, rng * prom_frac), dtype=int)


def _interval_hits_noise(t0, t1, move_regions):
    """True if the interval [t0, t1] overlaps any (start, end) movement region."""
    for (s, e) in move_regions or []:
        if t0 <= e and t1 >= s:
            return True
    return False


def _rr_from_starts(bs_times, move_regions=None, valid_bpm=None):
    """RR (bpm) at each mid-interval between consecutive breath-starts.

    move_regions : if given, any interval whose span [start, end] overlaps a
                   movement/noise region is dropped. The beats there were removed
                   as noise, so the envelope (linear/spline/ssp) merely
                   interpolated across the gap — a breath-start straddling it is
                   not a real breath. This matches the BWlegacy/BWbank gating.
    valid_bpm    : optional (lo, hi) clamp on the surviving RR values.

    NOTE: only the emitted RR is gated; bs_times (the detected breath-starts) are
    returned untouched by the caller, so they still plot inside the noise region.
    """
    bs_times = np.asarray(bs_times, np.float64)
    if bs_times.size < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(bs_times)
    ok = dt > 0
    rr = np.where(ok, 60.0 / np.where(ok, dt, 1.0), np.nan)
    if move_regions:
        for i in range(dt.size):
            if ok[i] and _interval_hits_noise(bs_times[i], bs_times[i + 1], move_regions):
                ok[i] = False
    if valid_bpm:
        lo, hi = valid_bpm
        ok &= (rr >= lo) & (rr <= hi)
    mid = (bs_times[:-1] + bs_times[1:]) / 2
    return mid[ok], rr[ok]


def _noise_flag_from_regions(t, move_regions):
    """Per-sample 0/1 noise flag on timeline `t` from (start, end) movement regions."""
    t = np.asarray(t, np.float64)
    flag = np.zeros(t.size, dtype=int)
    for (s, e) in move_regions or []:
        flag[(t >= s) & (t <= e)] = 1
    return flag


@dataclass
class RRParam:
    name: str                       # "RSA" | "RIIV" | "AUC" | "LP"
    series_x: np.ndarray            # per-beat feature time
    series_y: np.ndarray            # per-beat feature value (or full-rate trace for LP)
    env_x: np.ndarray               # band-passed envelope time
    env_y: np.ndarray
    bs_times: np.ndarray            # breath-start times
    rr_time: np.ndarray             # per-breath RR timestamps
    rr_bpm: np.ndarray              # per-breath RR values
    spect: dict = None              # STFT spectrogram of this param's envelope
    ridge_time: np.ndarray = None   # per-param ridge: dominant-RR timestamps
    ridge_rr: np.ndarray = None     # per-param ridge: dominant RR (bpm) over time


@dataclass
class PPGChannelResult:
    channel: str
    fs: float
    filtered: np.ndarray            # band-passed channel
    bc: np.ndarray                  # baseline-corrected
    ss_idx: list
    se_idx: list
    msd_idx: list
    params: dict = field(default_factory=dict)     # name -> RRParam
    ridge_time: np.ndarray = None
    ridge_rr: np.ndarray = None
    move_regions: list = field(default_factory=list)  # (start, end) noise regions (for viz + gating)

    def mean_rr(self, param):
        p = self.params.get(param)
        if p is None or p.rr_bpm.size == 0:
            return float("nan")
        return float(np.nanmean(p.rr_bpm))


# ----------------------------------------------------------------------
# Per-channel pipeline
# ----------------------------------------------------------------------
def analyze_ppg_channel(signal, t, fs, channel="Green", cfg=PPG,
                        min_msd_ms=None, invert=None, compute_ridge=True,
                        move_regions=None):
    """Full RR pipeline for one PPG-type channel.

    signal : channel samples on the 1024 Hz timeline (already upsampled)
    t      : matching time vector (s)
    move_regions : list of (start, end) motion intervals; beats inside are
                   dropped by the LPF-derivative detector (matches the HTML).
    Returns a PPGChannelResult with the four RR params + spectral ridge.
    """
    if min_msd_ms is None:
        min_msd_ms = cfg.msd_min_ms
    if invert is None:
        invert = cfg.invert_ppg
    x = np.asarray(signal, np.float64)
    if invert:
        x = -x

    # 1) band-pass the channel (0.5-4 Hz cardiac band)
    bp = bandpass_filter(x, fs, cfg.ppg_hp_hz, cfg.ppg_lp_hz, cfg.ppg_filter_order)["filtered"]

    # 2) detect + refine beats. The shipped HTML overrides detectBeats troughs
    #    with the LPF-derivative method (graph 2b/15b); toggle in settings.
    if cfg.use_lpf_derivative_beats:
        beats = detect_beats_lpf_derivative(bp, fs, cfg, time=t, move_regions=move_regions)
        if len(beats) <= 2:                       # fall back if the override underdetects
            beats = detect_beats(bp, fs, cfg.hr_min_bpm, cfg.hr_max_bpm, cfg=cfg)
    else:
        beats = detect_beats(bp, fs, cfg.hr_min_bpm, cfg.hr_max_bpm, cfg=cfg)
    if len(beats) > 2:
        beats = refine_ss_by_derivative(beats, bp, fs)
    corr_thresh = cfg.corr_green if channel == "Green" else cfg.corr_red_ir
    # post-corr is OFF by default (see beats.post_corr_filter docstring)
    beats = post_corr_filter(beats, bp, 0.0 if channel != "_enable" else corr_thresh)

    # 3) baseline-correct + systolic analysis
    bc = remove_dc_beat_aligned(bp, beats)
    sa = compute_systolic_analysis(bc, t, beats, fs, min_msd_ms, cfg.msd_min_pct_d1)

    prom = cfg.breath_start_prominence
    params = {}

    # ---- Param 1: RSA (RR-interval ms between consecutive SS) ----
    rr_min_ms = 60000.0 / cfg.hr_max_bpm
    rr_max_ms = 60000.0 / cfg.hr_min_bpm
    ss = sa.ss_idx
    rsa_x, rsa_y = [], []
    for k in range(len(ss) - 1):
        i0, i1 = ss[k], ss[k + 1]
        dt = t[i1] - t[i0]
        if dt <= 0:
            continue
        rr_ms = dt * 1000
        if rr_ms < rr_min_ms or rr_ms > rr_max_ms:
            continue
        rsa_x.append(t[i1]); rsa_y.append(rr_ms)
    params["RSA"] = _make_param("RSA", np.asarray(rsa_x), np.asarray(rsa_y), cfg, prom, move_regions)

    # ---- Param 2: RIIV (per-beat max height) ----
    params["RIIV"] = _make_param("RIIV", sa.maxht_x, sa.maxht_y, cfg, prom, move_regions)

    # ---- Param 3: AUC (per-beat area) ----
    params["AUC"] = _make_param("AUC", sa.auc_x, sa.auc_y, cfg, prom, move_regions)

    # ---- Spline variants of RSA/RIIV/AUC (comparison alternatives) ----
    # Same per-beat source series and same breath-start/RR logic as the linear
    # params; only the envelope construction differs. Additive only.
    prom_spl = getattr(cfg, "rr_spline_prominence", None)
    if prom_spl is None:                           # fall back to the shared prominence
        prom_spl = prom
    val = getattr(cfg, "rr_spline_use_valleys", False)

    # (a) interpolating cubic spline (passes through every point, NO band-pass)
    if getattr(cfg, "rr_spline_enabled", False):
        params["RSA_spline"] = _make_param_spline("RSA_spline", np.asarray(rsa_x), np.asarray(rsa_y), cfg, prom_spl, val, move_regions)
        params["RIIV_spline"] = _make_param_spline("RIIV_spline", sa.maxht_x, sa.maxht_y, cfg, prom_spl, val, move_regions)
        params["AUC_spline"] = _make_param_spline("AUC_spline", sa.auc_x, sa.auc_y, cfg, prom_spl, val, move_regions)

    # (b) smoothing spline (penalized fit: denoises the beat-to-beat jitter but
    #     keeps the respiratory shape — less information loss than linear+BP)
    if getattr(cfg, "rr_ssp_enabled", False):
        lam = getattr(cfg, "rr_ssp_lam", None)
        cut = getattr(cfg, "rr_ssp_cutoff_hz", 1.0)
        prom_ssp = getattr(cfg, "rr_ssp_prominence", None)
        if prom_ssp is None:                       # fall back to the spline prominence
            prom_ssp = prom_spl
        params["RSA_ssp"] = _make_param_ssp("RSA_ssp", np.asarray(rsa_x), np.asarray(rsa_y), cfg, prom_ssp, val, lam, cut, move_regions)
        params["RIIV_ssp"] = _make_param_ssp("RIIV_ssp", sa.maxht_x, sa.maxht_y, cfg, prom_ssp, val, lam, cut, move_regions)
        params["AUC_ssp"] = _make_param_ssp("AUC_ssp", sa.auc_x, sa.auc_y, cfg, prom_ssp, val, lam, cut, move_regions)

    # ---- Param 4: LP/BW (raw channel band-passed at its OWN band, full rate) ----
    lp_trace = bandpass_filter(x, fs, cfg.bw_band_low_hz, cfg.bw_band_high_hz,
                               cfg.bw_filter_order)["filtered"]
    bs_lp = _breath_starts_raw(lp_trace, fs, prom, t=t, cfg=cfg, move_regions=move_regions)
    bs_lp_t = t[bs_lp] if bs_lp.size else np.zeros(0)
    lp_rr_t, lp_rr = _rr_from_starts(bs_lp_t, *_rr_gate_args(cfg, move_regions))
    params["LP"] = RRParam("LP", series_x=t, series_y=lp_trace,
                           env_x=t, env_y=lp_trace,
                           bs_times=bs_lp_t, rr_time=lp_rr_t, rr_bpm=lp_rr)

    # ---- Param 5: BWlegacy (FULL legacy pipeline: their signal + their peaks) ----
    # Build the respiration signal the old Breath_by_Breath way (FFT band-pass
    # 0.1-0.7 Hz + peak-envelope detrend + spline gap-fill) from the RAW channel,
    # then detect breaths with the legacy zero-cross valley detector. Movement
    # regions -> per-sample noise flag (their noise *detector* is not ported).
    if getattr(cfg, "bwlegacy_enabled", True):
        from .legacy_bw import build_legacy_bw_signal, legacy_avg_rr   # lazy: only this path needs scipy
        noise_flag = _noise_flag_from_regions(t, move_regions)
        leg_sig, leg_noise = build_legacy_bw_signal(x, fs, noise_flag,
                                                    getattr(cfg, "legacy_bw_p2p_th", 20.0),
                                                    getattr(cfg, "legacy_bw_detrend_band", (0.1, 0.7)))

        # Legacy average-RR gate (old BBB_RR): validity mask + agreement + range.
        avg_ps = legacy_avg_rr(leg_sig, fs, cfg) if getattr(cfg, "legacy_bw_avg_gate", True) else None
        det_noise = leg_noise.copy()
        if avg_ps is not None:
            det_noise[np.isnan(avg_ps)] = 1          # no valid average here -> exclude

        bs_leg = peak_detection_zero_cross(leg_sig, det_noise, fs)
        bs_leg_t = t[bs_leg] if bs_leg.size else np.zeros(0)

        if bs_leg.size > 1:
            dt = np.diff(bs_leg_t)
            ok = dt > 0
            rr_all = np.where(ok, 60.0 / np.where(ok, dt, 1.0), np.nan)
            mid = (bs_leg_t[:-1] + bs_leg_t[1:]) / 2
            lo, hi = getattr(cfg, "legacy_bw_valid_bpm", (6.0, 40.0))
            keep = ok & (rr_all >= lo) & (rr_all <= hi)
            for i in range(bs_leg.size - 1):          # drop intervals spanning noise
                if keep[i] and np.any(det_noise[bs_leg[i]:bs_leg[i + 1]] == 1):
                    keep[i] = False
            if avg_ps is not None:                    # +-ratio agreement vs the average
                avg_at = avg_ps[bs_leg[1:]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = np.abs(rr_all - avg_at) / avg_at
                keep &= ~np.isnan(avg_at) & (ratio <= getattr(cfg, "legacy_bw_avg_ratio", 0.30))
            leg_rr_t, leg_rr = mid[keep], rr_all[keep]
        else:
            leg_rr_t, leg_rr = np.zeros(0), np.zeros(0)

        params["BWlegacy"] = RRParam("BWlegacy", series_x=t, series_y=leg_sig,
                                     env_x=t, env_y=leg_sig,
                                     bs_times=bs_leg_t, rr_time=leg_rr_t, rr_bpm=leg_rr)

    # ---- Param 6: BWbank (MATLAB filter-bank + stitching BBB) ----
    # Independent second BBB method: a bank of narrow band-passes, per-instant
    # best-fit level, zero-crossing stitching at transitions, trend-machine
    # breath detection. OFF by default (settings.PPG.bwbank_enabled).
    if getattr(cfg, "bwbank_enabled", False):
        from .filterbank_bbb import analyze_filterbank_bbb   # lazy: only this path needs scipy.signal.butter
        bank = analyze_filterbank_bbb(x, t, fs, cfg, move_regions=move_regions)
        params["BWbank"] = RRParam("BWbank", series_x=t, series_y=bank.sig_on_t,
                                   env_x=bank.grid_t, env_y=bank.stitched,
                                   bs_times=bank.bs_times, rr_time=bank.rr_time, rr_bpm=bank.rr_bpm)

    # ---- Per-parameter spectrogram + ridge (one STFT per RR parameter) ----
    # Each param's respiration-band envelope gets its own STFT so the dominant
    # respiration frequency can be tracked over time (4 spectrograms per channel).
    ridge_t = ridge_v = None
    if compute_ridge:
        for p in params.values():
            if p.name in ("BWlegacy", "BWbank"):   # full-rate BBB traces; skip param STFT
                continue
            _add_param_spectrogram(p, cfg)
        # channel-level ridge stays the RSA read for backward compatibility
        rsa = params.get("RSA")
        if rsa is not None:
            ridge_t, ridge_v = rsa.ridge_time, rsa.ridge_rr

    return PPGChannelResult(
        channel=channel, fs=fs, filtered=bp, bc=bc,
        ss_idx=ss, se_idx=sa.se_idx, msd_idx=sa.msd_idx, params=params,
        ridge_time=ridge_t, ridge_rr=ridge_v, move_regions=list(move_regions or []),
    )


def _rr_gate_args(cfg, move_regions):
    """Resolve the (move_regions, valid_bpm) gate arguments for _rr_from_starts
    from settings — shared by all envelope variants so the noise gate is applied
    identically to linear+BP, cubic spline, smoothing spline and LP."""
    mr = move_regions if getattr(cfg, "rr_reject_noise_spanning", False) else None
    vb = getattr(cfg, "rr_valid_bpm", None)
    return mr, vb


def _make_param(name, x, y, cfg, prom, move_regions=None):
    """Build an RRParam for the resampled-envelope path (RSA/RIIV/AUC)."""
    env_x, env_y = bp_rr_series(x, y, cfg.rr_band_low_hz, cfg.rr_band_high_hz, cfg.rr_filter_order)
    bs = _breath_starts_envelope(env_x, env_y, prom, cfg=cfg, move_regions=move_regions) if env_y.size else np.array([], int)
    bs_t = env_x[bs] if bs.size else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs_t, *_rr_gate_args(cfg, move_regions))
    return RRParam(name, series_x=np.asarray(x), series_y=np.asarray(y),
                   env_x=env_x, env_y=env_y, bs_times=bs_t, rr_time=rr_t, rr_bpm=rr)


def _make_param_spline(name, x, y, cfg, prom, use_valleys=False, move_regions=None):
    """Build an RRParam via the cubic-spline envelope path (RSA/RIIV/AUC variant).

    Identical to _make_param except the envelope is a cubic spline through the
    per-beat series (cubic_spline_rr_series) instead of linear-interp + band-pass.
    use_valleys=True detects VALLEYS (minima) instead of peaks by running the same
    detector on the inverted envelope; breath markers still reference env_y.
    """
    env_x, env_y = cubic_spline_rr_series(x, y, cfg.rr_resample_fs)
    if env_y.size:
        detect_on = -env_y if use_valleys else env_y
        bs = _breath_starts_envelope(env_x, detect_on, prom, cfg=cfg, move_regions=move_regions)
    else:
        bs = np.array([], int)
    bs_t = env_x[bs] if bs.size else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs_t, *_rr_gate_args(cfg, move_regions))
    return RRParam(name, series_x=np.asarray(x), series_y=np.asarray(y),
                   env_x=env_x, env_y=env_y, bs_times=bs_t, rr_time=rr_t, rr_bpm=rr)


def _make_param_ssp(name, x, y, cfg, prom, use_valleys=False, lam=None, cutoff_hz=None,
                    move_regions=None):
    """Build an RRParam via the SMOOTHING-spline envelope path (RSA/RIIV/AUC).

    Same breath-start + RR logic as _make_param_spline, but the envelope is a
    penalized smoothing spline (smoothing_spline_rr_series) so the beat-to-beat
    jitter is denoised while the respiratory shape is preserved.
    """
    env_x, env_y = smoothing_spline_rr_series(x, y, cfg.rr_resample_fs, lam, cutoff_hz)
    if env_y.size:
        detect_on = -env_y if use_valleys else env_y
        bs = _breath_starts_envelope(env_x, detect_on, prom, cfg=cfg, move_regions=move_regions)
    else:
        bs = np.array([], int)
    bs_t = env_x[bs] if bs.size else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs_t, *_rr_gate_args(cfg, move_regions))
    return RRParam(name, series_x=np.asarray(x), series_y=np.asarray(y),
                   env_x=env_x, env_y=env_y, bs_times=bs_t, rr_time=rr_t, rr_bpm=rr)


def _add_param_spectrogram(p, cfg):
    """Attach an STFT spectrogram + respiration ridge to one RRParam, in place.

    RSA/RIIV/AUC envelopes already live on the 10 Hz RR grid. LP is a full-rate
    band-passed trace, so it is resampled onto that same grid first for a
    comparable spectrogram. The ridge search spans the respiration band
    (rr_band_low_hz..rr_band_high_hz) — not the degenerate rr_spec_* pair.
    """
    env_x = np.asarray(p.env_x, np.float64)
    env_y = np.asarray(p.env_y, np.float64)
    if env_x.size < 4:
        return
    spec_fs = cfg.rr_resample_fs
    if p.name == "LP":                       # full-rate trace -> 10 Hz RR grid
        grid = np.arange(env_x[0], env_x[-1], 1.0 / spec_fs)
        sig = np.interp(grid, env_x, env_y)
    else:
        sig = env_y
    t_off = env_x[0]
    if sig.size <= int(cfg.rr_spec_window_sec * spec_fs):
        return
    spect = compute_spectrogram(sig - sig.mean(), spec_fs,
                                cfg.rr_spec_window_sec, cfg.rr_band_high_hz + 0.2)
    if spect is None:
        return
    rt, rv = ridge_rr(spect, cfg.rr_band_low_hz, cfg.rr_band_high_hz)
    spect = dict(spect)
    spect["times"] = spect["times"] + t_off
    if rt is not None:
        rt = rt + t_off
    p.spect = spect
    p.ridge_time = rt
    p.ridge_rr = rv


def analyze_ppg(channels, t, fs, cfg=PPG, which=("Green", "Red", "IR", "Artifact"),
                move_regions=None):
    """Run analyze_ppg_channel over several channels.

    channels : dict mapping channel name -> 1024 Hz signal array
    move_regions : motion intervals passed to each channel's beat detector.
    Returns dict name -> PPGChannelResult (only for channels present & non-empty).
    """
    out = {}
    for name in which:
        sig = channels.get(name)
        if sig is None or np.asarray(sig).size == 0 or not np.any(sig):
            continue
        min_msd = {"Green": cfg.msd_min_ms}.get(name, cfg.msd_min_ms)
        out[name] = analyze_ppg_channel(sig, t, fs, channel=name, cfg=cfg,
                                        min_msd_ms=min_msd, move_regions=move_regions)
    return out
