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
