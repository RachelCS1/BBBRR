"""
STFT spectrogram + respiratory ridge extraction.

Faithful port of computeSpectrogram(@3449): Hann-windowed STFT, hop = 1 sample,
fftSize = next pow2 of fs*fftSeconds, subsampled to ~2000 columns. The spectral
ridge (used as the "most direct" RR estimator, and for the Artifact channel) is
the per-column argmax frequency within the respiration band.
"""

import numpy as np


def compute_spectrogram(signal, fs, fft_seconds, max_freq_store):
    """Return dict with times, freqs, power_db, power_lin (freqs up to max_freq_store)."""
    x = np.asarray(signal, np.float64)
    fft_size = int(2 ** np.ceil(np.log2(fs * fft_seconds)))
    n_frames = x.size - fft_size + 1
    if n_frames <= 0:
        return None

    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(fft_size) / (fft_size - 1)))
    max_bin = min(int(np.floor(max_freq_store * fft_size / fs)), fft_size // 2)
    n_bins = max_bin + 1
    freqs = np.arange(n_bins) * fs / fft_size

    target_cols = 2000
    frame_step = max(1, n_frames // target_cols)
    frame_indices = np.arange(0, n_frames, frame_step)

    times = (frame_indices + fft_size / 2) / fs
    power_db = np.empty((n_bins, frame_indices.size))
    power_lin = np.empty((n_bins, frame_indices.size))

    for fi, f in enumerate(frame_indices):
        seg = x[f:f + fft_size] * hann
        spec = np.fft.rfft(seg, n=fft_size)[:n_bins]
        mag2 = spec.real ** 2 + spec.imag ** 2
        power_db[:, fi] = 10 * np.log10(mag2 + 1e-20)
        power_lin[:, fi] = np.sqrt(mag2)

    return {"times": times, "freqs": freqs,
            "power_db": power_db, "power_lin": power_lin,
            "fft_size": fft_size, "n_bins": n_bins}


def ridge_rr(spect, f_low, f_high):
    """Per-column dominant-frequency ridge within [f_low, f_high] -> RR in bpm.

    Returns (times, rr_bpm). Columns whose band has no positive power are NaN.
    """
    if spect is None:
        return np.zeros(0), np.zeros(0)
    freqs = spect["freqs"]
    band = (freqs >= f_low) & (freqs <= f_high)
    if not band.any():
        return spect["times"], np.full(spect["times"].size, np.nan)
    band_idx = np.where(band)[0]
    p = spect["power_lin"][band_idx, :]        # (nBandBins, nFrames)
    best = np.argmax(p, axis=0)
    ridge_hz = freqs[band_idx][best]
    rr = ridge_hz * 60.0
    rr[p.max(axis=0) <= 0] = np.nan
    return spect["times"], rr


def ridge_rr_fundamental(spect, f_low, f_high, ratio_th=0.85, energy_th=0.06):
    """Ridge with fundamental selection + per-column normalisation (port of
    findRRinSpec/RR_freq_func — steps 1 & 3 together, which are coupled).

    Per column: normalise the RR band to a pdf (sum=1) and zero bins below
    `energy_th` (kills spread-out drift/Mayer energy so it cannot be mistaken for
    a low fundamental); find local peaks; if more than one is within `ratio_th` of
    the tallest, pick the LOWEST-frequency one (the fundamental, not a harmonic);
    otherwise the tallest. The argmax bin is always a candidate so a fundamental
    on the band edge (which find_peaks cannot flag) is not lost. Returns (times, bpm).
    """
    from scipy.signal import find_peaks
    if spect is None:
        return np.zeros(0), np.zeros(0)
    freqs = spect["freqs"]
    band_idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    times = spect["times"]
    if band_idx.size == 0:
        return times, np.full(times.size, np.nan)
    P = spect["power_lin"][band_idx, :]        # (nBandBins, nFrames)
    bandfreqs = freqs[band_idx]
    n = P.shape[1]
    rr = np.full(n, np.nan)
    for i in range(n):
        col = P[:, i]
        s = col.sum()
        if s <= 0:
            continue
        col = col / s                          # column -> pdf (normalise)
        col = np.where(col < energy_th, 0.0, col)   # zero weak/spread energy
        if col.max() <= 0:
            continue
        locs, _ = find_peaks(col)
        cand = np.unique(np.append(locs, np.argmax(col))) if locs.size else np.array([np.argmax(col)])
        order = np.argsort(col[cand])[::-1]    # candidates, tallest first
        cand_s = cand[order]
        pks_s = col[cand_s]
        high = np.where(pks_s / pks_s[0] > ratio_th)[0]
        if high.size > 1:                      # harmonics present -> lowest freq
            chosen = cand_s[high[np.argmin(cand_s[high])]]
        else:
            chosen = cand_s[0]                 # single dominant peak
        rr[i] = bandfreqs[chosen] * 60.0
    return times, rr
