"""
Spectrogram -> per-segment adaptive band-pass, for the REMbo reference.

Faithful port of the "Breath by breath RR - Poly + REMbo.html" spectrogram path
(DISPLAY-ONLY alternative RR; the used RR stays the MA-cascade zero-crossing one):

  computeSpectrogram   @1669  -> compute_spectrogram
  computeSegmentBandpass @1958 -> compute_segment_bandpass

Pipeline (mirrors runAnalysis @4573-4600):
  1. STFT of the MA-cascade-filtered nasal pressure.
  2. Per `high_seg_sec` segment: take the strongest spectral cell (dominant
     respiration frequency), read its -5 dB bandwidth, and Butterworth band-pass
     the ORIGINAL nasal signal for that block; stitch the blocks.
  3. Breath detection (find_breath_crossings + detect_breaths) then runs on the
     stitched signal (done by the caller, same pipeline as the used RR).
"""
import numpy as np

from ..preprocessing.filters import butter_bandpass_filtfilt


def compute_spectrogram(signal, fs, t0, fft_seconds, f_min_hz, f_max_hz,
                        noise_intervals=None):
    """Sliding Hann-window STFT -> {times, freqs, power_db[bin][frame]} (computeSpectrogram @1669).

    Frames whose window overlaps a noise interval are blanked (NaN) so an
    artifact never enters the FFT. Returns None if the window exceeds the signal.
    """
    x = np.asarray(signal, np.float64)
    n_sig = x.size
    win_len = max(2, int(round(fft_seconds * fs)))
    if win_len > n_sig:
        return None

    # Per-sample noise flag -> prefix sum, so each frame can test overlap in O(1).
    flag = np.zeros(n_sig, dtype=np.int64)
    for (s, e) in (noise_intervals or []):
        a = int(np.floor((s - t0) * fs))
        b = int(np.ceil((e - t0) * fs))
        a = max(0, a); b = min(n_sig, b)
        if b > a:
            flag[a:b] = 1
    noise_prefix = np.concatenate(([0], np.cumsum(flag)))

    nfft = 1
    while nfft < win_len:
        nfft <<= 1
    hop = max(1, int(round(win_len / 20)))          # ~95% overlap (~20 frames/window)
    n_frames = (n_sig - win_len) // hop + 1
    if n_frames <= 0:
        return None

    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(win_len) / (win_len - 1)))

    bin_hz = fs / nfft
    min_bin = max(0, int(np.floor(f_min_hz / bin_hz)))
    max_bin = min(nfft // 2, int(np.ceil(f_max_hz / bin_hz)))
    n_bins = max_bin - min_bin + 1
    if n_bins <= 0:
        return None
    freqs = (min_bin + np.arange(n_bins)) * bin_hz

    times = np.empty(n_frames)
    power_db = np.full((n_bins, n_frames), np.nan)
    frame = np.zeros(nfft)
    for fi in range(n_frames):
        f_start = fi * hop
        times[fi] = t0 + (f_start + win_len / 2.0) / fs
        if noise_prefix[f_start + win_len] - noise_prefix[f_start] > 0:
            continue                                 # overlaps noise -> blank (NaN)
        frame[:win_len] = x[f_start:f_start + win_len] * hann
        frame[win_len:] = 0.0
        spec = np.fft.rfft(frame)
        pwr = (spec.real ** 2 + spec.imag ** 2)[min_bin:max_bin + 1]
        power_db[:, fi] = 10.0 * np.log10(pwr + 1e-30)
    return {"times": times, "freqs": freqs, "power_db": power_db}


def compute_segment_bandpass(spec, signal, fs, t0, high_seg_sec, f_min_hz, f_max_hz):
    """Per-segment adaptive Butterworth band-pass from the spectrogram (computeSegmentBandpass @1958).

    Returns {segments: [{t_start,t_end,center_freq,low_cut,high_cut}], filtered}.
    """
    x = np.asarray(signal, np.float64)
    n = x.size
    out = np.zeros(n)
    segments = []
    if spec is None or spec["times"].size == 0 or n == 0:
        return {"segments": segments, "filtered": out}

    freqs = spec["freqs"]
    power_db = spec["power_db"]
    times = spec["times"]
    n_bins = freqs.size
    nyq = fs / 2.0
    W = max(1.0, float(high_seg_sec))
    t_end_all = t0 + n / fs
    prev_low = prev_high = prev_center = None

    seg_start = t0
    while seg_start < t_end_all - 1e-9:
        seg_end = min(seg_start + W, t_end_all)

        # Strongest (frame, bin) among frames whose centre lands in this segment.
        peak_p = -np.inf
        frame_max = bin_max = -1
        for fi in np.nonzero((times >= seg_start) & (times < seg_end))[0]:
            col = power_db[:, fi]
            if not np.isfinite(col).any():
                continue
            b = int(np.nanargmax(col))
            if col[b] > peak_p:
                peak_p = col[b]; frame_max = fi; bin_max = b

        if frame_max >= 0 and bin_max >= 0:
            center_freq = freqs[bin_max]
            col = power_db[:, frame_max]
            thr = peak_p - 5.0                       # -5 dB relative to the peak
            # Low cutoff: walk down from the peak to the first sub-threshold bin, interpolate.
            low_cut = freqs[0]
            for b in range(bin_max, 0, -1):
                if not (col[b] > thr):
                    p1, p2 = col[b], col[b + 1]
                    f1, f2 = freqs[b], freqs[b + 1]
                    low_cut = (f1 + (thr - p1) * (f2 - f1) / (p2 - p1)) \
                        if (np.isfinite(p1) and p2 != p1) else f2
                    break
            # High cutoff: walk up from the peak to the first sub-threshold bin, interpolate.
            high_cut = freqs[n_bins - 1]
            for b in range(bin_max, n_bins - 1):
                if not (col[b] > thr):
                    p1, p2 = col[b - 1], col[b]
                    f1, f2 = freqs[b - 1], freqs[b]
                    high_cut = (f1 + (thr - p1) * (f2 - f1) / (p2 - p1)) \
                        if (np.isfinite(p2) and p2 != p1) else f1
                    break
        elif prev_low is not None:
            low_cut, high_cut, center_freq = prev_low, prev_high, prev_center  # carry forward
        else:
            low_cut, high_cut, center_freq = f_min_hz, f_max_hz, np.sqrt(f_min_hz * f_max_hz)

        # Sanitise so the Butterworth design stays stable.
        low_cut = max(0.01, min(low_cut, nyq * 0.98))
        high_cut = max(low_cut + 0.02, min(high_cut, nyq * 0.99))
        prev_low, prev_high, prev_center = low_cut, high_cut, center_freq

        # Filter this block with up to 20 s of context padding on each side.
        i_start = max(0, int(round((seg_start - t0) * fs)))
        i_end = min(n, int(round((seg_end - t0) * fs)))
        if i_end <= i_start:
            seg_start += W
            continue
        pad = min(int(round(20 * fs)), i_start, n - i_end)
        a, b_idx = i_start - pad, i_end + pad
        filt = butter_bandpass_filtfilt(x[a:b_idx], fs, low_cut, high_cut)
        out[i_start:i_end] = filt[i_start - a:i_end - a]

        segments.append({"t_start": seg_start, "t_end": seg_end,
                         "center_freq": center_freq, "low_cut": low_cut, "high_cut": high_cut})
        seg_start += W

    return {"segments": segments, "filtered": out}
