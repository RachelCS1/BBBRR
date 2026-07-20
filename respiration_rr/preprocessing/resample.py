"""
Resampling — linear + FFT super-resolution upsampling.

Faithful ports of resampleLinear(@2431) and upsampleFFT(@3082) from the PPG
analyzer. upsampleFFT is spectral zero-padding with mirror-reflect edge padding
and overlap-save chunk cross-fading, exactly as the JS does it, so the 1024 Hz
timeline matches the tool.
"""

import numpy as np


def resample_linear(signal, src_fs, dst_fs):
    """Linear-interpolation resample (resampleLinear @2431)."""
    x = np.asarray(signal, np.float64)
    src_len = x.size
    duration = src_len / src_fs
    dst_len = int(round(duration * dst_fs))
    if dst_len <= 0:
        return np.zeros(0)
    src_idx = np.arange(dst_len) * src_fs / dst_fs
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, src_len - 1)
    frac = src_idx - lo
    return x[lo] * (1 - frac) + x[hi] * frac


def _upsample_chunk(chunk, nfft, factor):
    """Spectral zero-padding of one power-of-2 chunk (_upsampleChunk @3054).

    Returns the ifft'd real output (caller multiplies by `factor`).
    """
    real = np.zeros(nfft)
    real[:chunk.size] = chunk
    spec = np.fft.fft(real)  # length nfft, complex

    nfft2 = nfft * factor
    spec2 = np.zeros(nfft2, dtype=complex)
    half = nfft >> 1
    # positive frequencies [0, half)
    spec2[:half] = spec[:half]
    # split the Nyquist bin symmetrically
    spec2[half] = spec[half] * 0.5
    spec2[nfft2 - half] = spec[half] * 0.5
    # negative frequencies (half+1 .. nfft-1) mapped to the top of the padded spectrum
    spec2[nfft2 - nfft + half + 1:nfft2] = spec[half + 1:nfft]

    return np.fft.ifft(spec2).real


def upsample_fft(signal, factor):
    """FFT super-resolution upsample by integer `factor` (upsampleFFT @3082).

    Short signals processed in one shot; long signals via overlap-save with a
    cosine cross-fade over the overlap. Mirror-reflect padding avoids the
    zero-pad discontinuity that would ring across the output.
    """
    x = np.asarray(signal, np.float64)
    orig_len = x.size
    if orig_len == 0:
        return np.zeros(0)

    CHUNK = 8192
    OVERLAP = 512

    if orig_len <= CHUNK:
        nfft = 1
        while nfft < orig_len:
            nfft <<= 1
        buf = np.empty(nfft)
        buf[:orig_len] = x
        if nfft > orig_len:                    # mirror-reflect into pad
            i = np.arange(orig_len, nfft)
            j = np.maximum(0, 2 * (orig_len - 1) - i)
            buf[orig_len:] = x[j]
        up = _upsample_chunk(buf, nfft, factor)
        out_len = orig_len * factor
        return up[:out_len] * factor

    # Long signal: overlap-save with cross-fade.
    out_len = orig_len * factor
    result = np.zeros(out_len)
    step = CHUNK - OVERLAP
    nfft = 1
    while nfft < CHUNK:
        nfft <<= 1
    fade_len = OVERLAP * factor

    start = 0
    while start < orig_len:
        end = min(start + CHUNK, orig_len)
        chunk_len = end - start
        buf = np.empty(nfft)
        buf[:chunk_len] = x[start:end]
        if nfft > chunk_len:
            i = np.arange(chunk_len, nfft)
            j = np.maximum(0, 2 * (chunk_len - 1) - i)
            buf[chunk_len:] = x[start + j]

        up = _upsample_chunk(buf, nfft, factor) * factor
        out_start = start * factor
        out_chunk_len = chunk_len * factor

        if start == 0:
            n = min(out_chunk_len, out_len - out_start)
            result[out_start:out_start + n] = up[:n]
        else:
            # cross-fade the overlap region
            for i in range(fade_len):
                if out_start + i >= out_len:
                    break
                w = i / fade_len
                result[out_start + i] = result[out_start + i] * (1 - w) + up[i] * w
            lo = fade_len
            hi = min(out_chunk_len, out_len - out_start)
            if hi > lo:
                result[out_start + lo:out_start + hi] = up[lo:hi]
        start += step
    return result
