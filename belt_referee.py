#!/usr/bin/env python
"""
belt_referee.py - how much of the airflow "irregularity" is REAL?

For every recording, detect breaths on BOTH the airflow and the effort belt, then on a
5 s grid compute local breath-to-breath |dRR| for each. A grid point is:
  * airflow-irregular   : |dRR|_air  >= thr
  * belt-confirmed      : |dRR|_air >= thr AND |dRR|_effort >= thr   (survives the referee)
  * the rest (air irregular, effort regular) = detection artifact, removed.
Movement (DREAMT ACC_X/Y/Z, REMbo Activity) is flagged so we can see how much of what
survives coincides with motion (candidate movement artifact).

Prints per-recording lines and a cohort aggregate. STANDALONE, read-only.

Usage:  py belt_referee.py                 (all DREAMT + REMbo, thr 5 & 8)
        py belt_referee.py --limit 8
"""
import argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from respiration_rr.reference.airflow import analyze_reference
from respiration_rr.compare.compare import reference_rr_series
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

DREAMT_DIR = r"C:\Users\RachelMizrahi\CardiacSense\shares - BBB Respiration Rate\Data\Exp3\DreamT\PSG_100hz"
REMBO_DIR = r"C:\Users\RachelMizrahi\CardiacSense\shares - BBB Respiration Rate\Data\Exp3\Rembo nights - 2022_2023"
THRS = (5.0, 8.0)
GRID = 5.0
WIN = 15.0


def _perbreath(t, sig, fs):
    bt, br = reference_rr_series(analyze_reference(t, sig, fs))
    br = np.asarray(br, float); bt = np.asarray(bt, float)
    m = np.isfinite(bt) & np.isfinite(br)
    return bt[m], br[m]


def _local_dRR(bt, br, grid):
    if bt.size < 3:
        return np.zeros(grid.size)
    d = np.abs(np.diff(br)); td = bt[1:]
    lo = np.searchsorted(td, grid - WIN, "left"); hi = np.searchsorted(td, grid + WIN, "right")
    out = np.zeros(grid.size)
    for i in range(grid.size):
        if hi[i] > lo[i]:
            out[i] = np.median(d[lo[i]:hi[i]])
    return out


def _local_move(tm, mv, grid):
    """RMS movement in +-WIN of each grid point, then flag bursts (> median + 3*MAD)."""
    if tm is None or mv.size < 3:
        return np.zeros(grid.size, bool)
    lo = np.searchsorted(tm, grid - WIN, "left"); hi = np.searchsorted(tm, grid + WIN, "right")
    e = np.zeros(grid.size)
    for i in range(grid.size):
        if hi[i] > lo[i]:
            e[i] = np.std(mv[lo[i]:hi[i]])
    med = np.median(e); mad = np.median(np.abs(e - med)) or 1e-9
    return e > med + 3 * 1.4826 * mad


def analyze(air, eff, mov, t0, t1):
    grid = np.arange(t0, t1, GRID)
    la = _local_dRR(*air, grid); le = _local_dRR(*eff, grid)
    mv_hi = _local_move(mov[0], mov[1], grid) if mov else np.zeros(grid.size, bool)
    r = {"dur": grid.size * GRID}
    for thr in THRS:
        a = la >= thr; c = a & (le >= thr)
        r[thr] = {"air": a.sum() * GRID, "conf": c.sum() * GRID,
                  "conf_move": (c & mv_hi).sum() * GRID, "conf_still": (c & ~mv_hi).sum() * GRID}
    return r


def do_rembo(path):
    from respiration_rr.io.edf_reader import EDFReader
    r = EDFReader(path)
    ta, sa, fa = r.read_channel("Nasal P"); te, se, fe = r.read_channel("Chest")
    try:
        tm, mv, _ = r.read_channel("Activity")
    except Exception:
        tm = mv = None
    air = _perbreath(ta, sa, fa); eff = _perbreath(te, se, fe)
    t0, t1 = ta[0], ta[-1]
    return analyze(air, eff, (tm, mv) if tm is not None else None, t0, t1)


def do_dreamt(path):
    import pandas as pd
    from dreamt_night_report import _num, _block_avg, DECIM_HZ
    df = pd.read_csv(path, usecols=["TIMESTAMP", "FLOW", "THORAX", "ACC_X", "ACC_Y", "ACC_Z"])
    t = np.asarray(_num(df["TIMESTAMP"]), float)
    fs = 1.0 / np.median(np.diff(t)); dec = max(1, int(round(fs / DECIM_HZ))); fs_d = fs / dec
    td = t[::dec]
    fa = _block_avg(_num(df["FLOW"]), dec); fe = _block_avg(_num(df["THORAX"]), dec)
    n = min(len(td), len(fa), len(fe)); td, fa, fe = td[:n], fa[:n], fe[:n]
    acc = np.sqrt(_num(df["ACC_X"]) ** 2 + _num(df["ACC_Y"]) ** 2 + _num(df["ACC_Z"]) ** 2)
    air = _perbreath(td, fa, fs_d); eff = _perbreath(td, fe, fs_d)
    return analyze(air, eff, (t, np.asarray(acc, float)), td[0], td[-1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    jobs = [("DREAMT", p, do_dreamt) for p in sorted(glob.glob(os.path.join(DREAMT_DIR, "S*_PSG_df.csv")))]
    jobs += [("REMbo", p, do_rembo) for p in sorted(set(glob.glob(os.path.join(REMBO_DIR, "*.edf")) + glob.glob(os.path.join(REMBO_DIR, "*.EDF"))))]
    if a.limit:
        jobs = jobs[:a.limit // 2] + jobs[-(a.limit - a.limit // 2):]
    agg = {src: {"dur": 0, **{thr: {"air": 0, "conf": 0, "conf_move": 0, "conf_still": 0} for thr in THRS}} for src in ("DREAMT", "REMbo")}
    for i, (src, p, fn) in enumerate(jobs):
        sid = os.path.splitext(os.path.basename(p))[0].replace("_PSG_df", "")
        try:
            r = fn(p)
        except Exception as e:
            print(f"[{sid}] ERROR {e}", flush=True); continue
        agg[src]["dur"] += r["dur"]
        for thr in THRS:
            for k in ("air", "conf", "conf_move", "conf_still"):
                agg[src][thr][k] += r[thr][k]
        a5 = r[5.0]
        print(f"[{i+1}/{len(jobs)}] {src} {sid}: air-irreg {a5['air']/60:.0f}min -> belt-confirmed {a5['conf']/60:.0f}min "
              f"({100*a5['conf']/max(a5['air'],1):.0f}% survive)", flush=True)
    print("\n================ COHORT ================")
    tot = {"dur": 0, **{thr: {"air": 0, "conf": 0, "conf_move": 0, "conf_still": 0} for thr in THRS}}
    for src in ("DREAMT", "REMbo"):
        A = agg[src]; tot["dur"] += A["dur"]
        for thr in THRS:
            for k in A[thr]:
                tot[thr][k] += A[thr][k]
        print(f"\n{src} (total {A['dur']/3600:.0f}h):")
        for thr in THRS:
            T = A[thr]; d = A["dur"] or 1
            print(f"  |dRR|>={thr:.0f}: airflow-irregular {100*T['air']/d:.1f}%  ->  belt-confirmed {100*T['conf']/d:.1f}%  "
                  f"(survive {100*T['conf']/max(T['air'],1):.0f}%); of confirmed, {100*T['conf_move']/max(T['conf'],1):.0f}% during movement")
    print(f"\nALL ({tot['dur']/3600:.0f}h):")
    for thr in THRS:
        T = tot[thr]; d = tot["dur"] or 1
        print(f"  |dRR|>={thr:.0f}: airflow-irregular {100*T['air']/d:.1f}%  ->  belt-confirmed {100*T['conf']/d:.1f}%  "
              f"(survive {100*T['conf']/max(T['air'],1):.0f}%); still (no movement) {100*T['conf_still']/d:.1f}%")


if __name__ == "__main__":
    main()
