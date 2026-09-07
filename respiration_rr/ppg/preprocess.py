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
    sync_channel: str = None         # preferred display-name channel for MSD/SS sync
                                     # (e.g. "Green" for watch-13); None -> caller default
    sync_fiducial: str = None        # preferred sync fiducial "SS"|"MSD" (e.g. "SS" for
                                     # watch-13); None -> caller default ("MSD")


# file channel key -> analyzer channel name. "yellow" is the fourth wrist LED that
# only the watch-13 file carries (read_wrist13_csv); it is absent from other watches
# so it simply doesn't appear in `channels` for them.
_CH_MAP = {"ppg": "Green", "red": "Red", "infra_red": "IR", "artifact": "Artifact",
           "yellow": "Yellow", "ecg": "ECG"}


def prepare_watch(watch, cfg=PPG):
    """Turn a read_watch_csv() dict into 1024 Hz channels ready for analysis.

    Steps: resample each channel from the file rate to a base rate, FFT-upsample to
    target_fs (1024 Hz), trim `trim_head_sec` / `trim_tail_sec`, and compute
    accelerometer movement regions (jerk energy, threshold, +-margin expansion,
    gap merge).

    The base rate is cfg.fs_orig (256, upsample x4) for the 256 Hz watches, but a
    reader may set watch["native_fs"] to resample to a different base and pick the
    matching integer upsample factor — the watch-13 file (512 Hz, primary watch)
    uses native_fs = 512 so it FFT-upsamples x2 straight to 1024 instead of being
    downsampled through 256 Hz first.
    """
    src_fs = watch["fs"]
    base_fs = float(watch.get("native_fs", cfg.fs_orig))
    factor = int(round(cfg.target_fs / base_fs))
    # 1) resample present channels to the base rate
    at_base = {}
    for col, name in _CH_MAP.items():
        if col in watch:
            at_base[name] = resample_linear(watch[col], src_fs, base_fs)
    acc_base = {}
    for col in ("acc_x", "acc_y", "acc_z"):
        if col in watch:
            acc_base[col] = resample_linear(watch[col], src_fs, base_fs)

    # 2) FFT upsample -> target_fs (1024 Hz)
    up = {name: upsample_fft(sig, factor) for name, sig in at_base.items()}

    fs = cfg.target_fs
    n = min((v.size for v in up.values()), default=0)

    # 2b) movement energy — HTML-faithful: jerk on the base-rate accel (x base_fs),
    #     then FFT-upsample the ENERGY, then smooth at target_fs.
    move_energy_full = None
    if {"acc_x", "acc_y", "acc_z"} <= set(acc_base):
        move_energy_full = _watch_movement_energy(acc_base, cfg, base_fs, factor)
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

    # Preferred sync hints (canonical col -> display name; fiducial passthrough).
    sync_col = watch.get("sync_channel_col")
    sync_channel = _CH_MAP.get(sync_col) if sync_col else None
    sync_fiducial = watch.get("sync_fiducial")

    return WatchSignals(time=time, fs=fs, channels=channels,
                        move_energy=move_energy, move_threshold=move_threshold,
                        move_regions=move_regions, sync_channel=sync_channel,
                        sync_fiducial=sync_fiducial)


def _watch_movement_energy(acc_base, cfg, base_fs, factor):
    """Jerk energy on the base-rate accel (x base_fs) -> FFT-upsample by `factor`
    -> smooth at target_fs (matches runAnalysis @7555-7568)."""
    ax = acc_base["acc_x"]; ay = acc_base["acc_y"]; az = acc_base["acc_z"]
    n = ax.size
    e = np.zeros(n)
    if n > 1:
        e[1:] = np.sqrt(np.diff(ax) ** 2 + np.diff(ay) ** 2 + np.diff(az) ** 2) * base_fs
        e[0] = e[1]
    e_up = upsample_fft(e, factor)
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
