#!/usr/bin/env python
"""
main_sync.py — inspect the PPG time-synchronisation with ONE figure:
the fiducial-interval (IBI) overlay, HTML-style.

The sync fiducial follows the file: legacy watches use finger-IR MSD; watch-13 uses
Green SS (both set via the reader's sync_channel_col / sync_fiducial hints). After the
offset is applied, the watch and REMbo interval-over-time curves (the shared HRV
pattern) should coincide beat-for-beat over the whole recording.
That overlay is the definitive "did the sync work?" check — the raw pulse
WAVEFORMS differ by sensor/site and are not expected to overlay, only the timing.

Usage
-----
    py main_sync.py                              # default data root, first recording
    py main_sync.py --recording Exp1/001         # one recording
    py main_sync.py --edf a.edf --watch b.csv    # explicit files
    py main_sync.py --minutes 12                  # cap analysed length (speed)
    py main_sync.py --show                        # open the window (else save a PNG)
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from respiration_rr.io.csv_reader import read_watch_auto
from respiration_rr import sync as S


DEFAULT_DATA_ROOT = (r"C:\Users\RachelMizrahi\CardiacSense"
                     r"\shares - BBB Respiration Rate\Data")
DEFAULT_OUT = r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\sync_inspect"

W = "#22c55e"   # watch  (green)
R = "#2563eb"   # REMbo  (blue)


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def _find_inputs(folder):
    edfs = glob.glob(os.path.join(folder, "*.edf"))
    csvs = (glob.glob(os.path.join(folder, "rt_flow*.csv"))
            or glob.glob(os.path.join(folder, "*.csv")))
    return (edfs[0] if edfs else None), (csvs[0] if csvs else None)


def discover(data_root):
    out = []
    for exp in sorted(glob.glob(os.path.join(data_root, "Exp*"))):
        rec_root = os.path.join(exp, "recordings data")
        if not os.path.isdir(rec_root):
            continue
        for folder in sorted(glob.glob(os.path.join(rec_root, "*"))):
            edf, csv = _find_inputs(folder)
            if edf and csv:
                out.append((f"{os.path.basename(exp)}/{os.path.basename(folder)}", edf, csv))
    return out


def _cap(sig, fs, minutes):
    if minutes and minutes > 0:
        return np.asarray(sig)[:int(round(minutes * 60 * fs))]
    return np.asarray(sig)


# ----------------------------------------------------------------------
# The one figure — fiducial interval (IBI) overlay
# ----------------------------------------------------------------------
def fig_interval(plt, res, rid, fiducial="MSD"):
    """Watch vs REMbo `fiducial`-interval-over-time, overlaid after the shift.

    `fiducial` ("MSD" | "SS") is only the label — the data in res.watch/res.rembo is
    already the fiducial the sync ran on. Full recording on top, a 40 s zoom below.
    Y is clamped to the physiological range so a stray outlier cannot stretch the scale.
    """
    ws, rs = res.watch, res.rembo
    off = res.offset_sec

    def ibi(t):
        t = np.asarray(t)
        return (t[:-1] + t[1:]) / 2, np.diff(t) * 1000.0

    wt, wi = ibi(ws.msd_t)
    rt, ri = ibi(rs.msd_t)
    lo = max(rt[0], wt[0] + off) if (rt.size and wt.size) else 0.0
    hi = min(rt[-1], wt[-1] + off) if (rt.size and wt.size) else 1.0
    allv = np.concatenate([ri, wi]) if (ri.size or wi.size) else np.array([600.0, 900.0])
    ylo = max(0.0, np.percentile(allv, 1) - 100)
    yhi = np.percentile(allv, 99) + 100

    flag = "  ⚠ LOW CONFIDENCE" if res.low_confidence else ""
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.4))
    fig.suptitle(f"{fiducial} interval overlay — {rid}   |   offset {off:+.3f} s  ·  "
                 f"matched {res.matched} ({res.matched_frac*100:.0f}%)  ·  "
                 f"median {res.median_resid_ms:.1f} ms  ·  {res.prominence:.1f}σ{flag}",
                 fontweight="bold", color=("#b91c1c" if res.low_confidence else "#111827"))
    for a in ax:
        a.plot(rt, ri, "-o", color=R, ms=3, lw=1.1, label=f"REMbo IBI ({ri.size})")
        a.plot(wt + off, wi, "-o", color=W, ms=3.4, lw=1.1, alpha=0.85,
               label=f"watch IBI (shift {off:+.2f}s)")
        a.set_ylabel(f"{fiducial} interval (ms)"); a.set_ylim(ylo, yhi)
        a.legend(loc="upper right", fontsize=8); a.grid(True, alpha=0.15)
    if hi > lo:
        ax[0].set_xlim(lo, hi)
    ax[0].set_title("full recording — the HRV pattern should coincide beat-for-beat", loc="left", fontsize=9)
    c = 0.5 * (lo + hi)
    ax[1].set_xlim(c - 20, c + 20)
    ax[1].set_title("zoom (40 s)", loc="left", fontsize=9)
    ax[1].set_xlabel("REMbo-clock time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Fiducial interval-overlay inspector for the PPG sync")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--recording", default=None, help="recording id, e.g. Exp1/001")
    ap.add_argument("--edf", default=None)
    ap.add_argument("--watch", default=None)
    ap.add_argument("--minutes", type=float, default=0.0, help="cap analysed length (0 = full)")
    ap.add_argument("--no-polarity", action="store_true", help="do not try REMbo inversion")
    ap.add_argument("--show", action="store_true", help="open the window (else save a PNG)")
    args = ap.parse_args(argv)

    edf_path, csv_path, rid = args.edf, args.watch, "custom"
    if not (edf_path and csv_path):
        recs = discover(args.data_root)
        if args.recording:
            recs = [r for r in recs if r[0] == args.recording]
        if not recs:
            ap.error(f"No recording found (root={args.data_root}, filter={args.recording})")
        rid, edf_path, csv_path = recs[0]

    print(f"Recording : {rid}")
    watch = read_watch_auto(csv_path)
    # Default sync channel is finger IR / MSD; the watch-13 reader overrides to Green
    # ("ppg") because its wrist IR is too weak, and to the SS fiducial on both sides.
    sync_col = watch.get("sync_channel_col", S.SYNC.msd_channel_col)
    sync_fid = watch.get("sync_fiducial", S.SYNC.msd_fiducial)
    wir, wfs = watch.get(sync_col), watch.get("fs")
    if wir is None:
        ap.error(f"watch CSV has no '{sync_col}' channel for sync")
    print(f"Sync channel: {sync_col} | fiducial: {sync_fid}")
    rpw, rfs, rname = S.read_rembo_pulse_wave(edf_path)
    wir = _cap(wir, wfs, args.minutes)
    rpw = _cap(rpw, rfs, args.minutes)

    print("Computing offset ...")
    res = S.compute_offset(wir, wfs, rpw, rfs, try_polarity=not args.no_polarity,
                           fiducial=sync_fid)
    print(f"  offset {res.offset_sec:+.3f} s | REMbo polarity "
          f"{'inverted' if res.polarity < 0 else 'as-is'} | matched {res.matched} "
          f"({res.matched_frac*100:.0f}%) | median {res.median_resid_ms:.1f} ms | "
          f"{res.prominence:.1f}σ | {'LOW — ' + res.reason if res.low_confidence else 'OK'}")

    import matplotlib
    matplotlib.use("TkAgg" if args.show else "Agg")
    import matplotlib.pyplot as plt

    fig = fig_interval(plt, res, rid, fiducial=sync_fid)
    if args.show:
        print("\nOpening figure — close the window to exit.")
        plt.show()
    else:
        os.makedirs(args.out, exist_ok=True)
        png = os.path.join(args.out, rid.replace("/", "_") + "_interval.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {png}")


if __name__ == "__main__":
    main()
