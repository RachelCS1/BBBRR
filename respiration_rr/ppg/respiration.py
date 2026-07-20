"""
PPG respiration estimation — the FOUR parameters + spectral ridge.

For each PPG-type channel (Green, Red, IR, Artifact) the tool derives respiration
rate from four independent per-beat modulation series, plus a spectral ridge:

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

    # ---- Spectral ridge (on the RSA envelope; the "most direct" read) ----
    ridge_t = ridge_v = None
    if compute_ridge:
        env = params["RSA"].env_y
        if env.size > int(cfg.rr_spec_window_sec * cfg.rr_resample_fs):
            spect = compute_spectrogram(env - env.mean(), cfg.rr_resample_fs,
                                        cfg.rr_spec_window_sec, cfg.rr_band_high_hz + 0.2)
            ridge_t, ridge_v = ridge_rr(spect, cfg.rr_spec_low_hz, cfg.rr_spec_high_hz)
            if ridge_t is not None:
                ridge_t = ridge_t + params["RSA"].env_x[0]

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
