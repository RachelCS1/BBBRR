"""
CSV readers.

  read_watch_csv : CardiacSense watch rt_flow_*.csv  (ppg/ecg/artifact/red/ir/accel/spo2)
  read_poly_csv  : Polysomnograph CSV with block-average downsampling (parseCSV @1130)
"""

import numpy as np
import pandas as pd


def read_watch_csv(path: str, channels=None):
    """Read a watch rt_flow CSV.

    Returns dict with:
      time   : seconds from first sample (float64)
      fs     : estimated sampling rate (median 1/dt)
      <ch>   : one array per requested channel present in the file
      raw    : the pandas DataFrame (for access to extra columns)

    The watch tool sign-inverts PPG-type channels later in preprocessing, not here.
    """
    if channels is None:
        channels = ["ppg", "ecg", "artifact", "red", "infra_red",
                    "acc_x", "acc_y", "acc_z", "sp_o2", "respiration_rate"]

    df = pd.read_csv(path, low_memory=False)

    # Time base from the sampling_time timestamp column.
    ts = pd.to_datetime(df["sampling_time"], errors="coerce")
    t0 = ts.iloc[0]
    time = (ts - t0).dt.total_seconds().to_numpy(dtype=np.float64)

    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = float(1.0 / np.median(dt)) if dt.size else 256.0

    out = {"time": time, "fs": fs, "raw": df}
    for ch in channels:
        if ch in df.columns:
            out[ch] = pd.to_numeric(df[ch], errors="coerce").to_numpy(dtype=np.float64)
    return out


def read_poly_csv(path: str, value_cols, downsample_factor=1, time_col="time_s"):
    """Read a polysomnograph CSV, optionally block-average downsampling.

    Mirrors parseCSV(@1130): when downsample_factor > 1, consecutive rows are
    *averaged* in blocks of that size (anti-aliasing), matching the JS exactly.

    Parameters
    ----------
    value_cols : list[str]   columns to extract (besides time)
    Returns (time, *columns) as float64 arrays.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    cols = [time_col] + list(value_cols)
    df = pd.read_csv(path, usecols=cols)[cols]
    arr = df.to_numpy(dtype=np.float64)

    if downsample_factor and downsample_factor > 1:
        n = (arr.shape[0] // downsample_factor) * downsample_factor
        arr = arr[:n].reshape(-1, downsample_factor, arr.shape[1]).mean(axis=1)

    columns = [arr[:, i] for i in range(arr.shape[1])]
    return tuple(columns)  # (time, col0, col1, ...)
