"""
PPG-analyzer DSP primitives (faithful ports).

NOTE: the PPG tool's moving average (@2567) differs from the reference tool's:
it computes a true centered W-sample average over the fully-windowed interior
[hw, n-1-hw] and CLAMPS the boundaries to the nearest interior value (rather
than shrinking the window). Cascading it `order` times gives the low-pass; the
band-pass subtracts a low-pass baseline then low-passes again.
"""

import numpy as np


def moving_average(signal, window_size):
    """Centered MA with edge-clamping (movingAverage @2567)."""
    x = np.asarray(signal, np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0)
    hw = window_size // 2
    W = 2 * hw + 1
    if n < W:                       # too short — global mean everywhere
        return np.full(n, x.mean())
    # interior centered sums via cumulative sum
    cum = np.empty(n + 1)
    cum[0] = 0.0
    np.cumsum(x, out=cum[1:])
    out = np.empty(n)
    i = np.arange(hw, n - hw)
    out[hw:n - hw] = (cum[i + hw + 1] - cum[i - hw]) / W
    out[:hw] = out[hw]              # clamp leading edge
    out[n - hw:] = out[n - 1 - hw]  # clamp trailing edge
    return out


def lowpass_filter(signal, fs, f_cutoff, order):
    """Cascaded moving-average low-pass (lowpassFilter @2601)."""
    window = max(3, int(round(fs / f_cutoff)))
    out = np.asarray(signal, np.float64).copy()
    for _ in range(order):
        out = moving_average(out, window)
    return out


def highpass_filter(signal, fs, f_cutoff):
    """Signal minus 1st-order LP (highpassFilter @2613)."""
    lp = lowpass_filter(signal, fs, f_cutoff, 1)
    return np.asarray(signal, np.float64) - lp


def bandpass_filter(signal, fs, f_hp, f_lp, order):
    """MA-cascade band-pass (bandpassFilter @2623).

    Returns dict {filtered, hp, lp}:
      hp       = signal - LP(signal, f_hp, order)
      filtered = LP(hp, f_lp, order)
      lp       = LP(signal, f_lp, order)   (kept for plots)
    """
    x = np.asarray(signal, np.float64)
    lp_for_hp = lowpass_filter(x, fs, f_hp, order)
    hp = x - lp_for_hp
    filtered = lowpass_filter(hp, fs, f_lp, order)
    lp = lowpass_filter(x, fs, f_lp, order)
    return {"filtered": filtered, "hp": hp, "lp": lp}


def notch_filter(signal, fs, f0, Q=30.0):
    """2nd-order IIR notch, direct-form-II transposed (notchFilter @2861)."""
    x = np.asarray(signal, np.float64)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * Q)
    b0, b1, b2 = 1.0, -2 * np.cos(w0), 1.0
    a0, a1, a2 = 1 + alpha, -2 * np.cos(w0), 1 - alpha
    nb0, nb1, nb2 = b0 / a0, b1 / a0, b2 / a0
    na1, na2 = a1 / a0, a2 / a0
    out = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(x.size):
        x0 = x[i]
        yi = nb0 * x0 + nb1 * x1 + nb2 * x2 - na1 * y1 - na2 * y2
        out[i] = yi
        x2, x1 = x1, x0
        y2, y1 = y1, yi
    return out


# db4 analysis/synthesis filters used by waveletDenoise (@2895).
_DB4_H = np.array([0.48296291314469025, 0.8365163037378079,
                   0.22414386804185735, -0.12940952255092145])
_DB4_G = np.array([-0.12940952255092145, -0.22414386804185735,
                   0.8365163037378079, -0.48296291314469025])
_DB4_HREC = np.array([-0.12940952255092145, 0.22414386804185735,
                      0.8365163037378079, 0.48296291314469025])
_DB4_GREC = np.array([-0.48296291314469025, -0.8365163037378079,
                      0.22414386804185735, 0.12940952255092145])


def _dwt_step(x):
    n = x.size
    half = n // 2
    approx = np.zeros(half)
    detail = np.zeros(half)
    for i in range(half):
        idx = (2 * i + np.arange(4)) % n
        approx[i] = np.dot(_DB4_H, x[idx])
        detail[i] = np.dot(_DB4_G, x[idx])
    return approx, detail


def _idwt_step(approx, detail):
    half = approx.size
    n = half * 2
    out = np.zeros(n)
    for i in range(half):
        idx = (2 * i + np.arange(4)) % n
        out[idx] += _DB4_HREC * approx[i] + _DB4_GREC * detail[i]
    return out


def wavelet_denoise(signal, levels=4):
    """db4 universal-soft-threshold denoise (waveletDenoise @2895)."""
    x = np.asarray(signal, np.float64)
    n = x.size
    if n == 0:
        return x.copy()
    pad_len = 1
    while pad_len < n:
        pad_len *= 2
    padded = np.zeros(pad_len)
    padded[:n] = x
    for i in range(n, pad_len):     # mirror padding
        j = 2 * n - 2 - i
        padded[i] = x[j] if 0 <= j < n else 0.0

    details = []
    current = padded
    for _ in range(levels):
        if current.size < 8:
            break
        approx, detail = _dwt_step(current)
        details.append(detail)
        current = approx
    final_approx = current

    if not details:
        return x.copy()

    finest = np.abs(details[0])
    median = np.sort(finest)[finest.size // 2]
    sigma = median / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(pad_len))

    for d in details:               # soft threshold
        mask = np.abs(d) <= threshold
        d[mask] = 0.0
        d[~mask] -= np.sign(d[~mask]) * threshold

    reconstructed = final_approx
    for d in reversed(details):
        reconstructed = _idwt_step(reconstructed, d)
    return reconstructed[:n].copy()
