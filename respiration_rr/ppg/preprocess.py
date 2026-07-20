"""
Watch front-end preprocessing: resample to 256 Hz -> FFT super-resolution to
1024 Hz -> edge trim, plus accelerometer movement / noise regions.

Mirrors the front of runAnalysis in the PPG analyzer (parse -> upsampleFFT ->
_trimUpsampledEdges -> movement detection).
"""

from dataclasses import dataclass
import numpy as np

from ..settings import PPG
from ..preprocessing.resample import resample_linear, upsample_fft
from ..preprocessing.movement import merge_intervals
from .dsp import moving_average


@dataclass
class WatchSignals:
    time: np.ndarray                 # 1024 Hz timeline (s), head/tail trimmed
    fs: float                        # target_fs (1024)
    channels: dict                   # name -> upsampled+trimmed signal
    move_energy: np.ndarray
    move_threshold: float
    move_regions: list               # list[(start, end)] seconds


# rt_flow column -> analyzer channel name
_CH_MAP = {"ppg": "Green", "red": "Red", "infra_red": "IR", "artifact": "Artifact",
           "ecg": "ECG"}


def prepare_watch(watch, cfg=PPG):
    """Turn a read_watch_csv() dict into 1024 Hz channels ready for analysis.

    Steps: resample each channel from the file rate to 256 Hz, FFT-upsample x4 to
    1024 Hz, trim `trim_head_sec` / `trim_tail_sec`, and compute accelerometer
    movement regions (jerk energy, threshold, +-margin expansion, gap merge).
    """
    src_fs = watch["fs"]
    # 1) resample present channels to FS_ORIG (256)
    at256 = {}
    for col, name in _CH_MAP.items():
        if col in watch:
            at256[name] = resample_linear(watch[col], src_fs, cfg.fs_orig)
    acc256 = {}
    for col in ("acc_x", "acc_y", "acc_z"):
        if col in watch:
            acc256[col] = resample_linear(watch[col], src_fs, cfg.fs_orig)

    # 2) FFT upsample x4 -> 1024 Hz
    up = {name: upsample_fft(sig, cfg.upsample_factor) for name, sig in at256.items()}

    fs = cfg.target_fs
    n = min((v.size for v in up.values()), default=0)

    # 2b) movement energy — HTML-faithful: jerk on the ORIGINAL 256 Hz accel
    #     (x fs_orig), then FFT-upsample the ENERGY, then smooth at target_fs.
    move_energy_full = None
    if {"acc_x", "acc_y", "acc_z"} <= set(acc256):
        move_energy_full = _watch_movement_energy(acc256, cfg)
        n = min(n, move_energy_full.size) if n else move_energy_full.size

    for k in up:
        up[k] = up[k][:n]

    # 3) trim edge transients
    h = int(round(cfg.trim_head_sec * fs))
    tl = int(round(cfg.trim_tail_sec * fs))
    lo, hi = h, max(h, n - tl)
    time = np.arange(lo, hi) / fs
    channels = {k: v[lo:hi] for k, v in up.items()}

    # 4) movement / noise regions
    move_energy = np.zeros(hi - lo)
    move_threshold = cfg.move_thresh_gs
    move_regions = []
    if move_energy_full is not None:
        move_energy = move_energy_full[lo:hi]
        move_regions = build_move_regions(time, move_energy, move_threshold, cfg)

    return WatchSignals(time=time, fs=fs, channels=channels,
                        move_energy=move_energy, move_threshold=move_threshold,
                        move_regions=move_regions)


def _watch_movement_energy(acc256, cfg):
    """Jerk energy on 256 Hz accel (x fs_orig) -> FFT-upsample -> smooth at target_fs
    (matches runAnalysis @7555-7568)."""
    ax = acc256["acc_x"]; ay = acc256["acc_y"]; az = acc256["acc_z"]
    n = ax.size
    e = np.zeros(n)
    if n > 1:
        e[1:] = np.sqrt(np.diff(ax) ** 2 + np.diff(ay) ** 2 + np.diff(az) ** 2) * cfg.fs_orig
        e[0] = e[1]
    e_up = upsample_fft(e, cfg.upsample_factor)
    return moving_average(e_up, int(round(cfg.target_fs * cfg.move_smooth_sec)))


def build_move_regions(time, energy, threshold, cfg):
    """Above-threshold movement regions (runAnalysis @7570-7651):
    threshold-cross -> drop blips < move_min_noise_sec -> expand +-margin & merge
    -> fill clean gaps < move_min_clean_gap_sec."""
    above = energy > threshold
    regions = []
    i, n = 0, above.size
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            regions.append([time[i], time[min(j, n - 1)]])
            i = j
        else:
            i += 1
    # drop brief blips
    if cfg.move_min_noise_sec > 0:
        regions = [r for r in regions if (r[1] - r[0]) >= cfg.move_min_noise_sec]
    if not regions:
        return []
    # expand by margin + merge
    m = cfg.move_margin_sec
    t0, t1 = time[0], time[-1]
    regions = [(max(t0, s - m), min(t1, e + m)) for s, e in regions]
    regions = [list(r) for r in merge_intervals(regions)]
    # fill short clean gaps
    out = [regions[0]]
    for s, e in regions[1:]:
        if s - out[-1][1] < cfg.move_min_clean_gap_sec:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]
