"""
Legacy Breath_by_Breath BW *signal* preprocessing — faithful port of the Python
RR_run.py chain that builds the trace the legacy zero-cross detector ran on.

This is the "get the signal the way they did it" half of the legacy BW method
(the peak detector itself lives in beats.peak_detection_zero_cross). Together
they reproduce the old pipeline end-to-end on a watch channel.

build_legacy_bw_signal(raw_channel, fs, noise):
  1. spline-fill noise gaps                 (NoiseCorrection 'basic')
  2. FFT band-pass 0.1-0.5 Hz               (FFTfilter)              <- their band
  3. peak-envelope detrend                  (Detrend_peaks)         <- upper/lower
     + amplitude (peak-to-peak) noise flag                            spline mean
  4. spline / hidden-4 Hz-sine gap fill     (NoiseCorrection 'custom')

Ports of Average RR Python/RR_run.py: FFTfilter, NoiseCorrection, SPLINE,
Detrend_peaks. The `noise` flag is supplied by the caller (the new pipeline's
movement regions) — the legacy noise *detector* is intentionally not ported, it
only served to mark noisy segments (which we already have).

NOTE: the amplitude peak-to-peak threshold (p2p_th) is sensor-scale dependent —
it was tuned for the old 64 Hz artifact ADC. On the watch optical channels the
scale differs, so it is exposed via settings.PPG.legacy_bw_p2p_th for tuning.
Set it to 0 to disable the amplitude gate.

scipy is imported lazily by the caller (only when BWlegacy runs), so importing
the core pipeline never requires scipy.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from .dsp import moving_average
from .spectrogram import compute_spectrogram, ridge_rr


def fft_bandpass(signal, fs, lo, hi):
    """FFT-zeroing band-pass (port of FFTfilter, type='bandpass')."""
    x = np.asarray(signal, np.float64)
    L = x.size
    if L < 4:
        return x.copy()
    fq = np.linspace(1.0 / (L / fs), fs + 1.0 / (L / fs), L, endpoint=False)
    F = np.fft.fft(x)
    F[(fq <= lo) | (fq >= hi)] = 0.0
    return 2.0 * np.real(np.fft.ifft(F))


def _noise_edges(noise):
    noise = np.asarray(noise).astype(int)
    start = np.where((noise[:-1] == 0) & (noise[1:] == 1))[0] + 1
    if noise[0] == 1:
        start = np.concatenate(([0], start))
    end = np.where((noise[:-1] == 1) & (noise[1:] == 0))[0]
    if noise[-1] == 1:
        end = np.concatenate((end, [noise.size - 1]))
    return start, end


def _fill_gap(seg_len, y0, y1, base_seg, fs, mode):
    """One gap fill (port of SPLINE): 'basic' -> linear; 'custom' -> short(<1s)
    linear, else a hidden 4 Hz sine whose amplitude follows the local signal."""
    if seg_len < 2:
        return np.full(max(seg_len, 0), y0)
    if mode == "basic" or seg_len / fs < 1.0:
        return np.linspace(y0, y1, seg_len)
    amp = (np.max(base_seg) - np.min(base_seg)) / 2.0
    t = np.arange(seg_len) / fs
    return amp * np.sin(2 * np.pi * 4.0 * t)     # 4 Hz: outside the RR band


def noise_correction(signal, noise, fs, mode="basic"):
    """Replace noise gaps with a spline / hidden sine (port of NoiseCorrection)."""
    x = np.asarray(signal, np.float64).copy()
    if noise is None or not np.any(noise):
        return x
    start, end = _noise_edges(noise)
    for s, e in zip(start, end):
        x[s:e + 1] = _fill_gap(e - s + 1, x[s], x[e], x[s:e + 1], fs, mode)
    return x


def _envelope(peak_idx, peak_val, grid):
    """Spline an envelope through kept peaks onto the full index grid."""
    if peak_idx.size >= 4:
        return CubicSpline(peak_idx, peak_val, extrapolate=True)(grid)
    if peak_idx.size >= 2:
        return np.interp(grid, peak_idx, peak_val)
    if peak_idx.size == 1:
        return np.full(grid.size, peak_val[0])
    return np.zeros(grid.size)


def detrend_peaks(signal, fs, noise, p2p_th, band=(0.1, 0.7)):
    """Peak-envelope detrend + amplitude noise flag (port of Detrend_peaks).

    `band` = FFT band-pass (Hz) applied before the envelope detrend; the legacy
    upper cutoff was 0.5, raised here to 0.7. Returns (detrended, noise_flag).
    """
    spl = noise_correction(signal, noise, fs, "basic")
    filt = fft_bandpass(spl, fs, band[0], band[1])
    mov = moving_average(filt, max(3, int(round(5 * fs))))
    grid = np.arange(filt.size)

    up, _ = find_peaks(filt)
    up = up[filt[up] >= mov[up]]                 # drop peaks below the mean trend
    above = _envelope(up, filt[up], grid)

    dn, _ = find_peaks(-filt)
    dn = dn[filt[dn] <= mov[dn]]                  # drop troughs above the mean trend
    below = _envelope(dn, filt[dn], grid)

    detrended = filt - (above + below) / 2.0
    p2p = (above - below) * 1.1                   # *1.1: filter shrinks amplitude

    noise2 = (np.asarray(noise).astype(int).copy()
              if noise is not None else np.zeros(filt.size, dtype=int))
    if p2p_th and p2p_th > 0:
        noise2[p2p < p2p_th] = 1                  # too-flat segments -> noise
    return detrended, noise2


def build_legacy_bw_signal(raw_channel, fs, noise, p2p_th=20.0, band=(0.1, 0.7)):
    """Full legacy BW signal (== RR_run.py 'Artifact_noDC_spline') + noise flag."""
    detrended, noise2 = detrend_peaks(raw_channel, fs, noise, p2p_th, band)
    sig = noise_correction(detrended, noise2, fs, "custom")
    return sig, noise2


# ----------------------------------------------------------------------
# Legacy average-RR gate (port of avg_RR/findRRinSpec + the BBB_RR cross-check)
# ----------------------------------------------------------------------
def _moving_median_nan(x, k):
    """Centered NaN-aware median over +-k samples (edge-shrinking)."""
    n = x.size
    out = np.full(n, np.nan)
    for i in range(n):
        seg = x[max(0, i - k):min(n, i + k + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _mark_short_island_nan(rr, min_len):
    """NaN out valid runs shorter than min_len samples (port of markShortIsland)."""
    valid = ~np.isnan(rr)
    n = rr.size
    i = 0
    while i < n:
        if valid[i]:
            j = i
            while j < n and valid[j]:
                j += 1
            if j - i < min_len:
                rr[i:j] = np.nan
            i = j
        else:
            i += 1
    return rr


def legacy_avg_rr(sig, fs, cfg):
    """Per-sample average RR (bpm) from a spectrogram ridge of `sig`, with the
    legacy quality gates (out-of-range / extreme-deviation / short-island -> NaN).

    Faithful-enough port of avg_RR + findRRinSpec: it yields an average with real
    NaNs so the BBB validity mask means something. Returns an array of length
    sig.size (NaN = no valid average there), or None if the signal is too short
    for the spectrogram window (caller then skips the gate rather than nuking all
    output).
    """
    grid_fs = cfg.rr_resample_fs
    if sig.size < 16:
        return None
    src_t = np.arange(sig.size) / fs
    n = int(np.floor(src_t[-1] * grid_fs)) + 1
    if n < 16:
        return None
    grid_t = np.arange(n) / grid_fs
    g = np.interp(grid_t, src_t, np.asarray(sig, np.float64))
    spect = compute_spectrogram(g - g.mean(), grid_fs,
                                getattr(cfg, "legacy_bw_avg_win_sec", 8.0),
                                cfg.rr_band_high_hz + 0.2)
    if spect is None:
        return None
    st, rr = ridge_rr(spect, cfg.rr_band_low_hz, cfg.rr_band_high_hz)   # rr already bpm
    rr = np.asarray(rr, np.float64).copy()
    lo, hi = cfg.legacy_bw_valid_bpm
    rr[(rr < lo) | (rr > hi)] = np.nan                                  # out of range
    med = _moving_median_nan(rr, 10)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr[np.abs((rr - med) / rr) > 0.2] = np.nan                      # extreme deviation
    if st.size >= 2:
        frame_dt = float(np.median(np.diff(st)))
        if frame_dt > 0:
            rr = _mark_short_island_nan(rr, max(1, int(round(20.0 / frame_dt))))
    if st.size == 0:
        return None
    # map frame-level rr onto every signal sample (nearest frame)
    gi = np.clip(np.searchsorted(st, src_t), 0, st.size - 1)
    left = np.clip(gi - 1, 0, st.size - 1)
    pick_left = np.abs(st[left] - src_t) < np.abs(st[gi] - src_t)
    gi[pick_left] = left[pick_left]
    avg_ps = rr[gi]
    if np.mean(np.isfinite(avg_ps)) < 0.2:        # too sparse to gate reliably -> skip gate
        return None
    return avg_ps
