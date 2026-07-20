"""
Filters — faithful ports of the reference tool's DSP primitives (@1319-1409).

The reference RR bandpass is deliberately NOT a Butterworth: it is a
moving-average cascade (HP = subtract a long MA baseline; LP = short MA).
The Butterworth is only used in the opt-in per-segment spectrogram stage.
"""

import numpy as np


def moving_average(signal, window_size):
    """Centered moving average with edge-shrinking window.

    Exact port of movingAverage(@1390): for sample i the window is
    [i-halfW, i+halfW] clamped to the array, halfW = floor(window/2).
    Uses a cumulative sum for O(n).
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n == 0:
        return x.copy()
    half = window_size // 2
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(x, out=cum[1:])
    i = np.arange(n)
    lo = np.maximum(0, i - half)
    hi = np.minimum(n - 1, i + half)
    return (cum[hi + 1] - cum[lo]) / (hi - lo + 1)


def bandpass_ma_cascade(signal, fs, f_low, f_high):
    """MA-cascade bandpass (bandpassFilter @1366).

    Step 1 (high-pass): subtract a moving-average baseline over a
            (1/f_low)-second window -> removes DC/drift.
    Step 2 (low-pass): short moving average over 1/(2*f_high) seconds
            (min 3 samples) -> removes HF noise, keeps breath shape.

    Returns (filtered, baseline, high_passed) so callers can plot intermediates.
    """
    x = np.asarray(signal, dtype=np.float64)
    hp_window = int(round((1.0 / f_low) * fs))
    baseline = moving_average(x, hp_window)
    high_passed = x - baseline

    lp_window = max(3, int(round((1.0 / (2.0 * f_high)) * fs)))
    filtered = moving_average(high_passed, lp_window)
    return filtered, baseline, high_passed


def butterworth_coeffs(fc, fs, kind):
    """2nd-order Butterworth via bilinear transform (butterworthCoeffs @1319)."""
    wc = np.tan(np.pi * fc / fs)
    wc2 = wc * wc
    sqrt2 = np.sqrt(2.0)
    k = 1.0 / (wc2 + sqrt2 * wc + 1.0)
    if kind == "low":
        b = np.array([wc2 * k, 2 * wc2 * k, wc2 * k])
    else:  # high
        b = np.array([k, -2 * k, k])
    a = np.array([1.0, 2 * (wc2 - 1) * k, (wc2 - sqrt2 * wc + 1) * k])
    return b, a


def _apply_iir(signal, b, a):
    """Direct-Form-II transposed, 2nd order (applyIIR @1338)."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.empty_like(x)
    w1 = w2 = 0.0
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    for i in range(x.size):
        xi = x[i]
        yi = b0 * xi + w1
        w1 = b1 * xi - a1 * yi + w2
        w2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


def filtfilt_iir(signal, b, a):
    """Zero-phase forward+reverse filtering (filtfilt @1353)."""
    fwd = _apply_iir(signal, b, a)
    rev = _apply_iir(fwd[::-1], b, a)
    return rev[::-1]


def butter_bandpass_filtfilt(signal, fs, f_low, f_high):
    """HP then LP 2nd-order Butterworth, each zero-phase (butterBandpassFiltfilt @1945)."""
    bhp, ahp = butterworth_coeffs(f_low, fs, "high")
    blp, alp = butterworth_coeffs(f_high, fs, "low")
    y = filtfilt_iir(signal, bhp, ahp)
    y = filtfilt_iir(y, blp, alp)
    return y
