#!/usr/bin/env python
"""
dreamt_to_recording.py — convert a DREAMT (PhysioNet / Empatica E4 + PSG) CSV into a
recording folder the EXISTING pipeline reads unchanged. STANDALONE — imports nothing
from respiration_rr, mutates nothing; it only WRITES two files.

Why this exists
---------------
DREAMT has no "Artifact" channel and only one optical channel (BVP), so it can't
exercise the Artifact-driven routing/fusion layers. But the REFERENCE-split
per-category path (per_category_lab --route ref) routes on reference-RR x PPG-HR,
both of which DREAMT provides (FLOW/PTAF airflow + BVP). This converter maps:

    DREAMT BVP   -> watch "ppg" (-> "Green" channel)          [read_watch_csv]
    DREAMT ACC   -> watch acc_x/acc_y/acc_z (movement gate)
    DREAMT FLOW  -> EDF "Nasal P"   (reference airflow)        [EDFReader / analyze_reference]
    DREAMT BVP   -> EDF "Pulse Wave"(time-sync; same clock -> offset ~0)
    DREAMT ACC|  -> EDF "Activity"  (reference movement gate, optional)

Both outputs sit on ONE clock (DREAMT's shared TIMESTAMP), so the PPG<->REMbo MSD
sync locks at ~0 s automatically (Pulse Wave is literally the same BVP).

Output folder (ready for the existing tools):
    <out>/rt_flow_<id>.csv     watch: sampling_time, ppg, acc_x, acc_y, acc_z
    <out>/<id>.edf             reference: Nasal P, Pulse Wave, [Activity]

Then, with NO code changes:
    py main.py            --recording <out>              # reference + watch + sync + compare
    py main3.py           --recording <out>              # per-signal figure suite
    py per_category_lab.py --edf <out>/<id>.edf --watch <out>/rt_flow_<id>.csv \
                           --rec-id DREAMT/<id> --route ref --channels Green

Usage
-----
    # inspect columns + rate without converting (match spellings vs Metadata.txt):
    py dreamt_to_recording.py --csv S002_PSG_df.csv --probe

    # convert a 20-min window (fast first pass) of the 100 Hz PSG file.
    # NOTE: write OUTSIDE OneDrive (the repo is OneDrive-synced and blocks Python
    # file creation). Omit --out to use the default %LOCALAPPDATA%\BBBRR_DREAMT\<id>:
    py dreamt_to_recording.py --csv S002_PSG_df.csv --id S002 --start 12000 --duration 1200

Notes / pitfalls handled
------------------------
  * Only the needed columns are read (usecols) so the 100 Hz files (up to ~1.3 GB)
    don't blow up memory. Parsing the text still takes time; use --max-rows or a
    --start/--duration window for quick iteration.
  * Column names are matched tolerantly (case/spacing/punctuation-insensitive) and
    are overridable; --probe prints what was found.
  * BVP inside the 100 Hz file is sample-and-hold upsampled from 64 Hz; accel from
    32 Hz. That's fine for beat/movement detection but is NOT true high-rate data.
    For the cleanest BVP use the 64 Hz whole-night file (--csv ..._whole_df.csv),
    which has no FLOW -> watch-only (main3 --watch / main.py --watch).
  * No real date exists in DREAMT (TIMESTAMP = s since wristband start); a synthetic
    --base-date is used only to satisfy the datetime column. Irrelevant to results.
"""

import argparse
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------- column matching
def _norm(name):
    """lower-case, alphanumerics only — 'ACC_X' / 'acc x' / 'AccX' all collapse."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _match(header, wanted):
    """Return the actual header column whose normalised form == normalised `wanted`."""
    by_norm = {_norm(c): c for c in header}
    return by_norm.get(_norm(wanted))


# ---------------------------------------------------------------- minimal EDF writer
def _edf_num(x):
    """Format a float into an EDF numeric field (<=8 ascii chars, float()-parseable)."""
    for prec in (6, 5, 4, 3, 2, 1, 0):
        s = f"{float(x):.{prec}f}"
        if len(s) <= 8:
            return s
    return f"{float(x):.0f}"[:8]


def _fit(s, n):
    """ascii, right-padded/truncated to exactly n bytes (EDF fixed-width field)."""
    return str(s).encode("ascii", "replace")[:n].ljust(n, b" ")


def _to_digital(x, dmin=-32767, dmax=32767):
    """Scale a physical channel to int16 digital codes + return (phys_min, phys_max)."""
    x = np.asarray(x, np.float64)
    finite = x[np.isfinite(x)]
    pmin = float(finite.min()) if finite.size else 0.0
    pmax = float(finite.max()) if finite.size else 1.0
    if pmax <= pmin:
        pmax = pmin + 1.0
    x = np.nan_to_num(x, nan=pmin)
    scale = (pmax - pmin) / (dmax - dmin)
    dig = np.round((x - pmin) / scale + dmin)
    dig = np.clip(dig, dmin, dmax).astype("<i2")
    return dig, pmin, pmax


def write_edf(path, channels, fs, base_date="01.01.22", start_time="00.00.00"):
    """Write a minimal EDF (16-bit) that respiration_rr/io/edf_reader.py reads faithfully.

    channels : list of (label, signal_array). All share `fs`; each is trimmed to a whole
               number of 1-second records.
    """
    spr = int(round(fs))                                   # samples per 1 s record
    if spr < 1:
        raise ValueError(f"fs rounds to {spr} samples/record")
    n = min(len(sig) for _, sig in channels)
    nrec = n // spr
    if nrec < 1:
        raise ValueError("not enough samples for one EDF record")
    ns = len(channels)

    digs, pmins, pmaxs = [], [], []
    for _, sig in channels:
        d, pmn, pmx = _to_digital(np.asarray(sig)[: nrec * spr])
        digs.append(d.reshape(nrec, spr))
        pmins.append(pmn); pmaxs.append(pmx)

    header = bytearray()
    header += _fit("0", 8)                                 # version
    header += _fit("X DREAMT", 80)                         # patient id
    header += _fit("Startdate X DREAMT converted", 80)     # recording id
    header += _fit(base_date, 8)                           # startdate dd.mm.yy
    header += _fit(start_time, 8)                          # starttime hh.mm.ss
    header += _fit(str(256 * (ns + 1)), 8)                 # header bytes
    header += _fit("", 44)                                 # reserved
    header += _fit(str(nrec), 8)                           # number of data records
    header += _fit("1", 8)                                 # record duration (s)
    header += _fit(str(ns), 4)                             # number of signals

    labels = [lbl for lbl, _ in channels]
    header += b"".join(_fit(l, 16) for l in labels)        # labels
    header += b"".join(_fit("", 80) for _ in channels)     # transducer
    header += b"".join(_fit("", 8) for _ in channels)      # physical dimension
    header += b"".join(_fit(_edf_num(v), 8) for v in pmins)   # physical min
    header += b"".join(_fit(_edf_num(v), 8) for v in pmaxs)   # physical max
    header += b"".join(_fit("-32767", 8) for _ in channels)   # digital min
    header += b"".join(_fit("32767", 8) for _ in channels)    # digital max
    header += b"".join(_fit("", 80) for _ in channels)     # prefiltering
    header += b"".join(_fit(str(spr), 8) for _ in channels)   # samples per record
    header += b"".join(_fit("", 32) for _ in channels)     # reserved

    block = np.concatenate(digs, axis=1)                   # (nrec, ns*spr), record-major
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(block.astype("<i2").tobytes())
    return nrec, spr


# ---------------------------------------------------------------- conversion
def convert(args):
    header = pd.read_csv(args.csv, nrows=0).columns
    tcol = _match(header, args.time_col)
    bcol = _match(header, args.bvp_col)
    rcol = _match(header, args.flow_col)
    acc_cols = [_match(header, c) for c in args.acc_cols]
    acc_cols = [c for c in acc_cols if c]

    print(f"Columns in {os.path.basename(args.csv)}:")
    print(f"  time    '{args.time_col}'  -> {tcol}")
    print(f"  BVP     '{args.bvp_col}'   -> {bcol}")
    print(f"  airflow '{args.flow_col}'  -> {rcol}")
    print(f"  accel   {args.acc_cols} -> {acc_cols or 'none'}")
    if tcol is None or bcol is None:
        sys.exit("ERROR: need at least a TIMESTAMP and a BVP column (check spellings; "
                 "see --probe and Metadata.txt). Override with --time-col/--bvp-col.")

    usecols = [tcol, bcol] + ([rcol] if rcol else []) + acc_cols
    df = pd.read_csv(args.csv, usecols=usecols,
                     nrows=(args.max_rows if args.max_rows else None), low_memory=False)

    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(np.float64)
    keep = np.isfinite(t)
    df, t = df[keep].reset_index(drop=True), t[keep]

    # window in TIMESTAMP seconds
    if args.start is not None or args.duration is not None:
        lo = args.start if args.start is not None else t[0]
        hi = (lo + args.duration) if args.duration is not None else t[-1]
        m = (t >= lo) & (t <= hi)
        df, t = df[m].reset_index(drop=True), t[m]
        if t.size < 10:
            sys.exit(f"ERROR: window [{lo}, {hi}] s keeps only {t.size} rows.")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    fs = float(1.0 / np.median(dt)) if dt.size else 64.0
    dur = t[-1] - t[0]
    print(f"\n{len(df)} rows | {dur:.0f} s | measured fs = {fs:.2f} Hz "
          f"(TIMESTAMP {t[0]:.0f}..{t[-1]:.0f} s)")
    if args.probe:
        print("\n[probe] no files written. Re-run without --probe to convert.")
        return

    def col(c):
        return pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)

    bvp = col(bcol)
    bvp = pd.Series(bvp).interpolate(limit_direction="both").to_numpy()

    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"ERROR: could not create output folder {args.out!r}: {e}\n"
                 "       OneDrive-synced folders (the repo itself) block directory "
                 "creation by Python. Pass --out to a NON-OneDrive path, e.g.\n"
                 f"       --out %LOCALAPPDATA%\\BBBRR_DREAMT\\{args.id or 'S002'}")
    rid = args.id or os.path.splitext(os.path.basename(args.csv))[0]
    rid = re.sub(r"[^A-Za-z0-9_.-]", "_", rid)

    # ---- watch rt_flow CSV ----
    base = pd.Timestamp(args.base_date)
    sampling_time = base + pd.to_timedelta(t - t[0], unit="s")
    watch = {"sampling_time": sampling_time, "ppg": bvp}
    acc_mag = None
    if acc_cols:
        ax, ay, az = (col(c) for c in (acc_cols + acc_cols[:3])[:3])
        for k, v in zip(("acc_x", "acc_y", "acc_z"), (ax, ay, az)):
            watch[k] = pd.Series(v).interpolate(limit_direction="both").to_numpy()
        acc_mag = np.sqrt(watch["acc_x"] ** 2 + watch["acc_y"] ** 2 + watch["acc_z"] ** 2)
    watch_path = os.path.join(args.out, f"rt_flow_{rid}.csv")
    pd.DataFrame(watch).to_csv(watch_path, index=False)
    print(f"\nwrote watch  -> {watch_path}  ({len(df)} rows)")

    # ---- reference EDF ----
    if rcol is None:
        print("no airflow column -> EDF skipped (watch-only). Use main3.py --watch / "
              "main.py --watch for signal inspection; per_category_lab needs the EDF.")
        return
    flow = col(rcol)
    flow = pd.Series(flow).interpolate(limit_direction="both").to_numpy()
    chans = [("Nasal P", flow), ("Pulse Wave", bvp)]
    if acc_mag is not None:
        chans.append(("Activity", acc_mag))
    edf_path = os.path.join(args.out, f"{rid}.edf")
    nrec, spr = write_edf(edf_path, chans, fs)
    print(f"wrote EDF    -> {edf_path}  (channels {[c for c, _ in chans]}, "
          f"{nrec} recs x {spr} samp @ {int(round(fs))} Hz)")

    print(f"\nRecording ready: {args.out}")
    print("Run, with NO code changes:")
    print(f"  py main.py --recording \"{args.out}\" --no-show")
    print(f"  py main3.py --recording \"{args.out}\"")
    print(f"  py per_category_lab.py --edf \"{edf_path}\" --watch \"{watch_path}\" "
          f"--rec-id DREAMT/{rid} --route ref --channels Green --no-show")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert a DREAMT CSV into a pipeline recording (EDF + watch CSV)")
    ap.add_argument("--csv", required=True, help="DREAMT Sxxx_PSG_df.csv (100 Hz) or Sxxx_whole_df.csv (64 Hz)")
    ap.add_argument("--out", default=None, help="output recording folder (default: Data/DREAMT/<id>)")
    ap.add_argument("--id", default=None, help="recording id (default: from the CSV filename)")
    ap.add_argument("--time-col", default="TIMESTAMP")
    ap.add_argument("--bvp-col", default="BVP")
    ap.add_argument("--flow-col", default="FLOW", help="reference airflow column (e.g. FLOW or PTAF)")
    ap.add_argument("--acc-cols", nargs="+", default=["ACC_X", "ACC_Y", "ACC_Z"])
    ap.add_argument("--start", type=float, default=None, help="window start (TIMESTAMP seconds)")
    ap.add_argument("--duration", type=float, default=None, help="window length (seconds)")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows READ (fast dev; reads the file head)")
    ap.add_argument("--base-date", default="2022-01-01", help="synthetic date for sampling_time (cosmetic)")
    ap.add_argument("--probe", action="store_true", help="print detected columns + fs, write nothing")
    args = ap.parse_args(argv)

    if args.out is None:
        # Default OUTSIDE OneDrive: the repo is OneDrive-synced and blocks Python
        # directory/file creation (os.makedirs -> WinError 2). LOCALAPPDATA always works.
        rid = args.id or os.path.splitext(os.path.basename(args.csv))[0]
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        args.out = os.path.join(base, "BBBRR_DREAMT", re.sub(r"[^A-Za-z0-9_.-]", "_", rid))
    convert(args)


if __name__ == "__main__":
    main()
