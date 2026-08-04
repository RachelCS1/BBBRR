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


def _breath_starts_envelope(env_x, env_y, prom_frac, min_dist=10):
    """Breath-starts = prominence-filtered local maxima on a 10 Hz envelope."""
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
    rng = env_y.max() - env_y.min()
    kept = filter_peaks_by_prominence(env_y, raw_peaks, rng * prom_frac)
    return np.asarray(kept, dtype=int)


def _breath_starts_raw(sig, fs, prom_frac):
    """Breath-starts on a full-rate band-passed trace (LP param, @12016)."""
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
    rng = sig.max() - sig.min()
    return np.asarray(filter_peaks_by_prominence(sig, raw_peaks, rng * prom_frac), dtype=int)


def _rr_from_starts(bs_times):
    """RR (bpm) at each mid-interval between consecutive breath-starts."""
    bs_times = np.asarray(bs_times, np.float64)
    if bs_times.size < 2:
        return np.zeros(0), np.zeros(0)
    dt = np.diff(bs_times)
    ok = dt > 0
    rr = np.where(ok, 60.0 / np.where(ok, dt, 1.0), np.nan)
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
    params["RSA"] = _make_param("RSA", np.asarray(rsa_x), np.asarray(rsa_y), cfg, prom)

    # ---- Param 2: RIIV (per-beat max height) ----
    params["RIIV"] = _make_param("RIIV", sa.maxht_x, sa.maxht_y, cfg, prom)

    # ---- Param 3: AUC (per-beat area) ----
    params["AUC"] = _make_param("AUC", sa.auc_x, sa.auc_y, cfg, prom)

    # ---- Param 4: LP/BW (raw channel band-passed at its OWN band, full rate) ----
    lp_trace = bandpass_filter(x, fs, cfg.bw_band_low_hz, cfg.bw_band_high_hz,
                               cfg.bw_filter_order)["filtered"]
    bs_lp = _breath_starts_raw(lp_trace, fs, prom)
    bs_lp_t = t[bs_lp] if bs_lp.size else np.zeros(0)
    lp_rr_t, lp_rr = _rr_from_starts(bs_lp_t)
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
        ridge_time=ridge_t, ridge_rr=ridge_v,
    )


def _make_param(name, x, y, cfg, prom):
    """Build an RRParam for the resampled-envelope path (RSA/RIIV/AUC)."""
    env_x, env_y = bp_rr_series(x, y, cfg.rr_band_low_hz, cfg.rr_band_high_hz, cfg.rr_filter_order)
    bs = _breath_starts_envelope(env_x, env_y, prom) if env_y.size else np.array([], int)
    bs_t = env_x[bs] if bs.size else np.zeros(0)
    rr_t, rr = _rr_from_starts(bs_t)
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
