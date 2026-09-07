"""
CSV readers.

  read_watch_csv   : CardiacSense watch rt_flow_*.csv (ppg/ecg/artifact/red/ir/accel/spo2)
  read_monitor_csv : monitor-mode watch CSV (optical + accelerometer, no timestamp, 64 Hz)
  read_wrist13_csv : watch-13 monitor CSV (labeled multi-LED wrist, real ts, 512 Hz, +Yellow)
  read_watch_auto  : dispatch to the right reader by sniffing the header
  read_poly_csv    : Polysomnograph CSV with block-average downsampling (parseCSV @1130)
"""

import re

import numpy as np
import pandas as pd

from ..settings import PPG


def _norm(name):
    """Normalise a column name for tolerant matching: lower-case, alphanumerics only.

    So "RED SIG", "red_sig", "Red Sig" all collapse to "redsig"; "XL-X" -> "xlx".
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


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


def read_monitor_csv(path: str, fs=None, cfg=PPG):
    """Read a monitor-mode watch CSV (optical channels + accelerometer, no ECG).

    Monitor files have a header like "PPG,Artifact,RED SIG,IR,XL-X,XL-Y,XL-Z" and
    NO timestamp column, so the sampling rate can't be measured from the file — it
    is declared (default cfg.monitor_row_fs = 64 Hz; pass `fs` to override). Column
    names are mapped to the canonical keys used downstream (including the three
    accelerometer axes), so prepare_watch computes movement/noise regions exactly
    as it does for real-time files.

    Returns the same dict contract as read_watch_csv:
      time, fs, raw, and one array per channel present (ppg/red/infra_red/artifact,
      acc_x/acc_y/acc_z).
    """
    fs = float(fs) if fs is not None else float(cfg.monitor_row_fs)
    df = pd.read_csv(path, low_memory=False)

    by_norm = {_norm(c): c for c in df.columns}
    out = {"fs": fs, "raw": df}
    for src, dst in cfg.monitor_channel_cols.items():
        col = by_norm.get(_norm(src))
        if col is not None:
            out[dst] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

    n = len(df)
    out["time"] = np.arange(n, dtype=np.float64) / fs   # synthetic timeline from fs
    return out


def read_wrist13_csv(path: str, fs=None, cfg=PPG):
    """Read a watch-13 monitor CSV — the primary experiment watch.

    Labeled multi-LED wrist file with a header like
      "TIMESTAMP,FP,ECG AC,PPG Green (Wrist),PPG Red (Wrist),PPG IR (Wrist),
       PPG Yellow (Wrist),Artifact (Wrist),PPG Red (Right),...,XL X,XL Y,XL Z".
    Only the WRIST optical channels (Green/Red/IR/Yellow), the wrist Artifact and
    the accelerometer are extracted (mapped to canonical keys via
    cfg.wrist13_channel_cols); the Right-hand LEDs, FP and ECG AC are ignored. This
    is the only reader that emits the new "yellow" channel key.

    Rate: the file's timestamps declare a NOMINAL 512 Hz, but they are SYNTHETIC
    (millisecond-quantized with the 1 ms-gap fraction = exactly 3/64 = round(index*
    1000/512)), so they only echo the assumed 512 and cannot verify the true ADC
    rate. Cross-checking the cardiac fundamental against the REMbo Pulse Wave (which
    the older 256/64 Hz watches match to <0.1%) shows watch-13 runs ~5% slow — TRUE
    rate ~488 Hz (declared in cfg.wrist13_row_fs; pass `fs` to override). 488 is not
    a clean divisor of target_fs, so this reader does NOT set native_fs: prepare_watch
    resamples from this true rate to 256 Hz and FFT-upsamples x4 to 1024, keeping the
    timeline on real seconds.

    Returns the same dict contract as the other watch readers:
      time, fs, raw, and one array per channel present
      (ppg/red/infra_red/yellow/artifact, acc_x/acc_y/acc_z).
    """
    fs = float(fs) if fs is not None else float(cfg.wrist13_row_fs)
    df = pd.read_csv(path, low_memory=False)

    by_norm = {_norm(c): c for c in df.columns}
    # Sync hints (read by prepare_watch/main): Green channel (wrist IR too weak) and
    # the SS fiducial on both sides.
    out = {"fs": fs, "raw": df,
           "sync_channel_col": cfg.wrist13_sync_channel_col,
           "sync_fiducial": cfg.wrist13_sync_fiducial}
    for src, dst in cfg.wrist13_channel_cols.items():
        col = by_norm.get(_norm(src))
        if col is not None:
            out[dst] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

    # Real timestamps (informational — prepare_watch rebuilds its own timeline from
    # fs). Fall back to a synthetic timeline if the column is missing/unparseable.
    tcol = by_norm.get(_norm(cfg.wrist13_time_col))
    if tcol is not None:
        ts = pd.to_datetime(df[tcol], errors="coerce")
        out["time"] = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    else:
        out["time"] = np.arange(len(df), dtype=np.float64) / fs
    return out


def read_watch_auto(path: str, cfg=PPG):
    """Read a watch CSV, auto-detecting the format by header.

    Dispatch order:
      1. rt_flow real-time file — carries the timestamp column cfg.csv_time_col
         ("sampling_time") -> read_watch_csv.
      2. watch-13 monitor file — carries the labeled wrist LED column
         cfg.wrist13_signature_col ("PPG Green (Wrist)") -> read_wrist13_csv.
      3. otherwise a legacy monitor file (no timestamp) -> read_monitor_csv.
    So the same pipeline analyses all three.
    """
    header = pd.read_csv(path, nrows=0).columns
    cols = {_norm(c) for c in header}
    if _norm(cfg.csv_time_col) in cols:
        return read_watch_csv(path)
    if _norm(cfg.wrist13_signature_col) in cols:
        return read_wrist13_csv(path, cfg=cfg)
    return read_monitor_csv(path, cfg=cfg)


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
