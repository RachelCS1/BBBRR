"""
BW-extraction methods — raw continuous channel -> respiratory baseline-wander
signal, at the SAME sample rate as the input, so the shared breath detector
(_breath_starts_raw) runs identically on every method's output.

Four extractors, spanning the design space discussed:
  * ma_cascade  — the incumbent (moving-average cascade band-pass).
  * butterworth — zero-phase 2nd-order Butterworth band-pass (sharp, no smear).
  * wavelet     — undecimated (SWT) multiresolution; keep the components whose
                  dominant frequency falls in the respiration band.
  * mapas       — model-based cardiac removal: per window estimate the cardiac
                  frequency, fit a harmonic basis by least squares, subtract it;
                  the residual is the respiratory baseline. Survives HR~RR overlap.

Every function returns a float64 array the same length as `x`.

scipy / pywt are imported lazily inside each method so importing this module is
cheap and the core pipeline never depends on them.
"""

import numpy as np


def ma_cascade(x, fs, low, high, order=2):
    """Incumbent LP extractor (MA-cascade band-pass)."""
    from .dsp import bandpass_filter
    return np.asarray(bandpass_filter(x, fs, low, high, order)["filtered"], np.float64)


def butterworth(x, fs, low, high, order=2):
    """Zero-phase Butterworth band-pass.

    order == 2 uses the faithful hand-rolled biquad port (HP+LP, each filtfilt);
    any other order uses scipy's Nth-order bandpass design (lab order-sweeps).
    """
    if order == 2:
        from ..preprocessing.filters import butter_bandpass_filtfilt
        return np.asarray(butter_bandpass_filtfilt(x, fs, low, high), np.float64)
    # SOS form (not b,a): at our tiny normalised freqs (0.1 Hz / high Nyquist)
    # the transfer-function form is numerically unstable and collapses to drift.
    from scipy.signal import butter, sosfiltfilt
    ny = fs / 2.0
    sos = butter(order, [low / ny, high / ny], btype="band", output="sos")
    return np.asarray(sosfiltfilt(sos, np.asarray(x, np.float64)), np.float64)


def legacy_detrend(x, fs, low, high, baseline_sec=5.0, use_hamming=False,
                   env_gate_frac=None, env_gate_win=0.0):
    """Legacy Average-RR extraction: FFT band-pass + peak-envelope detrend
    (port of Detrend_peaks). No legacy noise handling (noise=None) — our own
    movement gating stays in charge; the P2P amplitude gate is disabled (p2p_th=0).
    `baseline_sec` is the envelope-gate baseline window (legacy 5 s; longer helps
    slow breaths). `use_hamming` reproduces the original FFTfilter's pre-FFT
    Hamming window. `env_gate_frac` fences the envelope by peak prominence
    (fraction of the median envelope distance) so small ripples don't fake high RR.
    Pair with the fundamental/argmax ridge as needed.
    """
    from .legacy_bw import detrend_peaks
    det, _ = detrend_peaks(np.asarray(x, np.float64), fs, None, 0.0, (low, high),
                           baseline_sec=baseline_sec, use_hamming=use_hamming,
                           env_gate_frac=env_gate_frac, env_gate_win=env_gate_win)
    return np.asarray(det, np.float64)


def _dominant_freq(sig, fs):
    """Energy-weighted dominant frequency of a 1-D signal (Hann-windowed FFT)."""
    n = sig.size
    if n < 4:
        return 0.0
    w = np.hanning(n)
    S = np.abs(np.fft.rfft((sig - sig.mean()) * w))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    if S.sum() <= 0:
        return 0.0
    return float(f[np.argmax(S)])


def wavelet(x, fs, low, high, name="db4", work_fs=16.0):
    """SWT multiresolution; sum the components whose dominant freq is in [low, high].

    The channel is decimated to `work_fs` (respiration is well below its Nyquist)
    so only a handful of SWT levels are needed, then the reconstructed baseline is
    interpolated back onto the original full-rate grid.
    """
    import pywt
    from scipy.signal import decimate
    x = np.asarray(x, np.float64)
    n = x.size
    if n < 16:
        return np.zeros(n)

    # ---- decimate to ~work_fs (chain factors so each stays FIR-friendly) ----
    q = max(1, int(round(fs / work_fs)))
    xd, fd = x, float(fs)
    rem = q
    while rem > 1:
        step = min(8, rem)
        if step > 1 and xd.size > 27:
            xd = decimate(xd, step, ftype="fir", zero_phase=True)
            fd /= step
        if step <= 1:
            break
        rem //= step
    m = xd.size

    # ---- SWT needs length divisible by 2**level; pad by reflection ----
    level = int(np.floor(np.log2(max(4, m))))
    level = max(2, min(level, int(np.floor(np.log2(fd / max(low, 1e-3)))) + 1))
    per = 1 << level
    pad = (-m) % per
    xp = np.pad(xd, (0, pad), mode="reflect") if pad else xd

    comps = pywt.mra(xp, name, level=level, transform="swt")   # sum(comps) == xp
    keep = np.zeros_like(xp)
    for c in comps:
        fdom = _dominant_freq(c[:m], fd)
        if low <= fdom <= high:
            keep = keep + c
    bwd = keep[:m]

    # ---- interpolate back onto the original full-rate time grid ----
    td = np.arange(m) / fd
    tf = np.arange(n) / fs
    return np.interp(tf, td, bwd).astype(np.float64)


def mapas(x, fs, low, high, win_sec=8.0, n_harm=4, hr_band=(0.7, 3.5), order=2):
    """Model-based cardiac removal, then band-limit.

    Sliding Hann windows (50% overlap, overlap-added): per window estimate the
    cardiac fundamental in `hr_band`, fit {sin,cos}*(k=1..n_harm) + linear trend by
    least squares, and subtract. The residual (cardiac + trend removed) is the
    respiratory baseline; a final zero-phase Butterworth keeps it in [low, high].
    """
    from ..preprocessing.filters import butter_bandpass_filtfilt
    x = np.asarray(x, np.float64)
    n = x.size
    win = max(16, int(round(win_sec * fs)))
    if n <= win:
        win = n
    hop = max(1, win // 2)
    resid = np.zeros(n)
    wsum = np.zeros(n)
    hann = np.hanning(win) if win > 1 else np.ones(win)

    a = 0
    while a < n:
        b = min(a + win, n)
        seg = x[a:b]
        L = seg.size
        if L < 8:
            resid[a:b] += seg * hann[:L]
            wsum[a:b] += hann[:L]
            a += hop
            continue
        # cardiac fundamental within hr_band
        w = np.hanning(L)
        S = np.abs(np.fft.rfft((seg - seg.mean()) * w))
        f = np.fft.rfftfreq(L, 1.0 / fs)
        m = (f >= hr_band[0]) & (f <= hr_band[1])
        if not m.any():
            r = seg - seg.mean()
        else:
            f_hr = f[m][np.argmax(S[m])]
            t = np.arange(L) / fs
            cols = [np.ones(L), t]                       # constant + linear trend
            for k in range(1, n_harm + 1):
                cols.append(np.sin(2 * np.pi * k * f_hr * t))
                cols.append(np.cos(2 * np.pi * k * f_hr * t))
            B = np.vstack(cols).T
            coef, *_ = np.linalg.lstsq(B, seg, rcond=None)
            r = seg - B @ coef                           # residual = respiration
        hh = hann[:L]
        resid[a:b] += r * hh
        wsum[a:b] += hh
        if b >= n:
            break
        a += hop

    wsum[wsum == 0] = 1.0
    bw = resid / wsum
    return np.asarray(butter_bandpass_filtfilt(bw, fs, low, high), np.float64)


def mapas_hrlock(x, fs, low, high, hr_t, hr_v, win_sec=8.0, n_harm=4, hr_band=(0.7, 3.5)):
    """MAPAS with the cardiac fundamental LOCKED to the measured beat HR.

    Identical sliding-window harmonic subtraction as `mapas`, but per window the
    fundamental `f_hr` is the MEDIAN measured HR (from the PPG beat detector) over
    the window's time span, instead of the in-band FFT peak of ART. This uses the
    HR we already know reliably, so the basis can't latch onto a respiration
    harmonic or noise. Falls back to the FFT peak when no beats land in a window.

    `hr_t` (s, SIGNAL clock — no sync offset) / `hr_v` (bpm) is the per-beat HR
    series; window absolute time is index/fs, matching the rest of the extraction.
    """
    from ..preprocessing.filters import butter_bandpass_filtfilt
    x = np.asarray(x, np.float64)
    hr_t = np.asarray(hr_t, np.float64)
    hr_v = np.asarray(hr_v, np.float64)
    n = x.size
    win = max(16, int(round(win_sec * fs)))
    if n <= win:
        win = n
    hop = max(1, win // 2)
    resid = np.zeros(n)
    wsum = np.zeros(n)
    hann = np.hanning(win) if win > 1 else np.ones(win)

    a = 0
    while a < n:
        b = min(a + win, n)
        seg = x[a:b]
        L = seg.size
        if L < 8:
            resid[a:b] += seg * hann[:L]
            wsum[a:b] += hann[:L]
            a += hop
            continue
        # fundamental from the measured HR over this window (signal clock)
        m = (hr_t >= a / fs) & (hr_t < b / fs)
        if m.any():
            f_hr = float(np.median(hr_v[m])) / 60.0            # bpm -> Hz
        else:                                                  # fallback: FFT peak
            w = np.hanning(L)
            S = np.abs(np.fft.rfft((seg - seg.mean()) * w))
            f = np.fft.rfftfreq(L, 1.0 / fs)
            mm = (f >= hr_band[0]) & (f <= hr_band[1])
            f_hr = float(f[mm][np.argmax(S[mm])]) if mm.any() else 0.0
        if f_hr <= 0:
            r = seg - seg.mean()
        else:
            t = np.arange(L) / fs
            cols = [np.ones(L), t]                             # constant + linear trend
            for k in range(1, n_harm + 1):
                cols.append(np.sin(2 * np.pi * k * f_hr * t))
                cols.append(np.cos(2 * np.pi * k * f_hr * t))
            B = np.vstack(cols).T
            coef, *_ = np.linalg.lstsq(B, seg, rcond=None)
            r = seg - B @ coef                                 # residual = respiration
        hh = hann[:L]
        resid[a:b] += r * hh
        wsum[a:b] += hh
        if b >= n:
            break
        a += hop

    wsum[wsum == 0] = 1.0
    bw = resid / wsum
    return np.asarray(butter_bandpass_filtfilt(bw, fs, low, high), np.float64)


def mapas_ref(art, refs, fs, low, high, win_sec=8.0, min_hr=0.625, max_hr=3.5, res=None):
    """Faithful MAPAS adapted to respiration (two-stage least-squares projection).

    Mirrors MAPAS.m with the substitution: accelerometer -> PPG LEDs (cardiac
    reference), PPG -> ART (target). Per sliding Hann window (50% overlap):

      stage 1  fit a sin/cos DICTIONARY over the HR band [min_hr:res:max_hr] to
               each reference LED  ->  reconstructs that LED's cardiac content
               (in-band only, so the LED's own respiration/RIIV is NOT dragged in);
      stage 2  fit the reconstructed cardiac set to ART                  ->
               cardiac-as-it-appears-on-ART; subtract it from ART.

    The residual is the respiratory baseline; a final zero-phase Butterworth keeps
    it in [low, high]. `refs` is a list of reference channel arrays (same length
    as `art`); `res` defaults to 1/win_sec (the source's FFT resolution).
    """
    from ..preprocessing.filters import butter_bandpass_filtfilt
    art = np.asarray(art, np.float64)
    refs = [np.asarray(r, np.float64) for r in refs]
    n = art.size
    win = max(16, int(round(win_sec * fs)))
    if n <= win:
        win = n
    hop = max(1, win // 2)
    if res is None:
        res = 1.0 / win_sec
    freqs = np.arange(min_hr, max_hr + 1e-9, res)
    out = np.zeros(n)
    wsum = np.zeros(n)
    hann = np.hanning(win) if win > 1 else np.ones(win)

    a = 0
    while a < n:
        b = min(a + win, n)
        L = b - a
        if L < 8 or not refs:
            out[a:b] += (art[a:b] - art[a:b].mean()) * hann[:L]
            wsum[a:b] += hann[:L]
            if b >= n:
                break
            a += hop
            continue
        t = np.arange(L) / fs
        base = np.outer(t, freqs)                                  # L x F
        w_h = np.hstack([np.cos(2 * np.pi * base), np.sin(2 * np.pi * base)])   # L x 2F
        # stage 1: reconstruct each reference LED's cardiac (HR-band dictionary)
        cols = []
        for r in refs:
            seg = r[a:b] - r[a:b].mean()
            c, *_ = np.linalg.lstsq(w_h, seg, rcond=None)
            cols.append(w_h @ c)
        w_a = np.vstack(cols).T                                    # L x n_ref
        # stage 2: project the cardiac references onto ART and subtract
        seg_art = art[a:b] - art[a:b].mean()
        c2, *_ = np.linalg.lstsq(w_a, seg_art, rcond=None)
        resid = seg_art - w_a @ c2                                 # residual = respiration
        hh = hann[:L]
        out[a:b] += resid * hh
        wsum[a:b] += hh
        if b >= n:
            break
        a += hop

    wsum[wsum == 0] = 1.0
    bw = out / wsum
    return np.asarray(butter_bandpass_filtfilt(bw, fs, low, high), np.float64)


def _peak_freq_in_band(seg, fs, lo, hi):
    """Hann-windowed FFT peak frequency of `seg` within [lo, hi] (or None)."""
    seg = np.asarray(seg, np.float64)
    L = seg.size
    if L < 8:
        return None
    w = np.hanning(L)
    S = np.abs(np.fft.rfft((seg - seg.mean()) * w))
    f = np.fft.rfftfreq(L, 1.0 / fs)
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return None
    return float(f[m][np.argmax(S[m])])


def mapas_ref_narrow(art, refs, fs, low, high, win_sec=8.0, n_harm=3,
                     hr_band=(0.625, 3.5), n_side=1, spread_hz=0.05):
    """MAPAS (LEDs -> ART) with a NARROW cardiac basis locked on the measured HR.

    Same two-stage projection as `mapas_ref`, but instead of a wide sin/cos
    dictionary spanning the whole HR band [0.625, 3.5] (which overlaps the RR band
    at high RR and can absorb respiration), the basis is built ONLY around the HR
    fundamental + harmonics estimated per window:

        f_hr = median FFT peak of the reference LEDs in hr_band
        basis freqs = { k*f_hr + j*spread_hz : k=1..n_harm, j=-n_side..n_side }

    Because the fundamental sits at the HR (>= ~1 Hz for a normal HR), the basis
    does not reach down into the respiration band, so high-RR breathing survives.
    The small +/- spread absorbs mild HR drift within the window. (When HR ~ RR
    exactly, the fundamental unavoidably coincides with respiration — that is the
    fundamental collision limit, not a tuning issue.)
    """
    from ..preprocessing.filters import butter_bandpass_filtfilt
    art = np.asarray(art, np.float64)
    refs = [np.asarray(r, np.float64) for r in refs]
    n = art.size
    win = max(16, int(round(win_sec * fs)))
    if n <= win:
        win = n
    hop = max(1, win // 2)
    out = np.zeros(n)
    wsum = np.zeros(n)
    hann = np.hanning(win) if win > 1 else np.ones(win)
    nyq = fs / 2.0

    a = 0
    while a < n:
        b = min(a + win, n)
        L = b - a
        hh = hann[:L]
        if L < 16 or not refs:
            out[a:b] += (art[a:b] - art[a:b].mean()) * hh
            wsum[a:b] += hh
            if b >= n:
                break
            a += hop
            continue
        # HR estimate = median LED FFT peak in the HR band
        peaks = [_peak_freq_in_band(r[a:b], fs, hr_band[0], hr_band[1]) for r in refs]
        peaks = [p for p in peaks if p]
        if not peaks:
            out[a:b] += (art[a:b] - art[a:b].mean()) * hh
            wsum[a:b] += hh
            if b >= n:
                break
            a += hop
            continue
        f_hr = float(np.median(peaks))
        # narrow basis: harmonics of f_hr, each with a small +/- spread
        freqs = []
        for k in range(1, n_harm + 1):
            for j in range(-n_side, n_side + 1):
                fk = k * f_hr + j * spread_hz
                if 0 < fk < nyq:
                    freqs.append(fk)
        freqs = np.unique(np.asarray(freqs))
        if freqs.size == 0:
            out[a:b] += (art[a:b] - art[a:b].mean()) * hh
            wsum[a:b] += hh
            if b >= n:
                break
            a += hop
            continue
        t = np.arange(L) / fs
        base = np.outer(t, freqs)
        w_h = np.hstack([np.cos(2 * np.pi * base), np.sin(2 * np.pi * base)])
        cols = []
        for r in refs:
            seg = r[a:b] - r[a:b].mean()
            c, *_ = np.linalg.lstsq(w_h, seg, rcond=None)
            cols.append(w_h @ c)
        w_a = np.vstack(cols).T
        seg_art = art[a:b] - art[a:b].mean()
        c2, *_ = np.linalg.lstsq(w_a, seg_art, rcond=None)
        resid = seg_art - w_a @ c2
        out[a:b] += resid * hh
        wsum[a:b] += hh
        if b >= n:
            break
        a += hop

    wsum[wsum == 0] = 1.0
    bw = out / wsum
    return np.asarray(butter_bandpass_filtfilt(bw, fs, low, high), np.float64)


METHODS = {
    "MA": ma_cascade,
    "Butter": butterworth,
    "Wavelet": wavelet,
    "MAPAS": mapas,
    "Legacy": legacy_detrend,   # detrend-peaks extraction (pair with fundamental ridge)
    # "MAPASref" / "MAPASnar" are multi-channel (need reference LEDs) so they are
    # dispatched separately by the harness, not through this single-channel table.
}

# methods that require reference LED channels (target, refs, fs, low, high)
REF_METHODS = {
    "MAPASref": mapas_ref,
    "MAPASnar": mapas_ref_narrow,
}
