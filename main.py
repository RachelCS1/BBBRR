#!/usr/bin/env python
"""
Respiration-rate toolchain — main entry point.

Runs, on a single recording, the Python port of all three HTML tools:
  1. Reference analyzer  (REMbo EDF / Polysomnograph CSV)  -> breath-by-breath RR
  2. Watch PPG analyzer   (rt_flow CSV: Green/Red/IR/Artifact) -> 4 respiration params
  3. Cross-device comparison (watch RR candidates vs REMbo reference, ranked by MAE)

and opens matplotlib windows for each, mirroring what the HTML tools display.

Usage
-----
    py Code/main.py                          # defaults to Data/Exp1/recordings data/001
    py Code/main.py --recording "<dir>"      # a folder containing one .edf + one rt_flow*.csv
    py Code/main.py --edf a.edf --watch b.csv
    py Code/main.py --no-show                # compute + print summary, don't open windows

All algorithm parameters live in respiration_rr/settings.py — edit there to tune.
"""

import argparse
import glob
import os
import sys

# make the package importable when run as `py Code/main.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:                       # make Unicode prints safe on cp1252 Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from respiration_rr.settings import REFERENCE, PPG, COMPARE
from respiration_rr.io.edf_reader import EDFReader
from respiration_rr.io.csv_reader import read_watch_auto
from respiration_rr.reference.airflow import analyze_reference
from respiration_rr.ppg.preprocess import prepare_watch
from respiration_rr.ppg.respiration import analyze_ppg
from respiration_rr.compare.compare import compare_watch_vs_reference
from respiration_rr import viz


DEFAULT_RECORDING = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data", "Exp1", "recordings data", "001")


def _find_inputs(recording_dir):
    edfs = glob.glob(os.path.join(recording_dir, "*.edf"))
    csvs = glob.glob(os.path.join(recording_dir, "rt_flow*.csv"))
    if not csvs:
        csvs = glob.glob(os.path.join(recording_dir, "*.csv"))
    return (edfs[0] if edfs else None), (csvs[0] if csvs else None)


def run_reference(edf_path):
    """Reference (REMbo EDF) analysis -> ReferenceResult."""
    print(f"\n[Reference]  {os.path.basename(edf_path)}")
    reader = EDFReader(edf_path)
    t, sig, fs = reader.read_channel(REFERENCE.edf_airflow_channel)
    act_ch = reader.pick_activity_channel(REFERENCE.edf_activity_keywords)
    activity = activity_fs = None
    if act_ch:
        _, activity, activity_fs = reader.read_channel(act_ch)
    ref = analyze_reference(t, sig, fs, activity=activity, activity_fs=activity_fs)
    rates = np.array([b.rate for b in ref.breaths])
    ies = np.array([b.ie_ratio for b in ref.breaths if np.isfinite(b.ie_ratio)])
    print(f"  airflow '{REFERENCE.edf_airflow_channel}' @ {fs:.0f} Hz, {t[-1]:.0f}s | "
          f"activity gate '{act_ch}' thr={ref.move_threshold:.2f}")
    print(f"  breaths: {len(ref.breaths)} | RR mean {rates.mean():.1f} "
          f"(min {rates.min():.1f}, max {rates.max():.1f}) bpm | "
          f"I:E mean {ies.mean():.2f} | noise regions {len(ref.merged_noise)}")
    return ref


def run_ppg(csv_path):
    """Watch PPG analysis -> dict of PPGChannelResult."""
    print(f"\n[Watch PPG]  {os.path.basename(csv_path)}")
    watch = read_watch_auto(csv_path)          # real-time (rt_flow) or monitor CSV
    sig = prepare_watch(watch)
    results = analyze_ppg(sig.channels, sig.time, sig.fs, move_regions=sig.move_regions)
    print(f"  {watch['fs']:.0f} Hz -> {PPG.target_fs:.0f} Hz, {sig.time[-1]-sig.time[0]:.0f}s | "
          f"channels {list(results)} | movement regions {len(sig.move_regions)}")
    print("  mean RR (bpm) by parameter:")
    print("    %-9s %6s %6s %6s %6s %6s" % ("channel", "RSA", "RIIV", "AUC", "LP", "Ridge"))
    for ch, res in results.items():
        row = [ch]
        for p in ("RSA", "RIIV", "AUC", "LP"):
            pr = res.params.get(p)
            v = np.nanmean(pr.rr_bpm) if pr and pr.rr_bpm.size else np.nan
            row.append(f"{v:.1f}")
        ridge = np.nanmean(res.ridge_rr) if res.ridge_rr is not None and np.isfinite(res.ridge_rr).any() else np.nan
        row.append(f"{ridge:.1f}")
        print("    %-9s %6s %6s %6s %6s %6s" % tuple(row))
    return results, sig


def run_comparison(ppg_results, ref):
    print("\n[Comparison]  watch RR candidates vs REMbo reference")
    cmp = compare_watch_vs_reference(ppg_results, ref, offset="auto")
    print(f"  best alignment offset = {cmp.offset_sec:+.0f} s | "
          f"{len(cmp.ranked)} scorable candidates")
    print(f"  top {COMPARE.top_n} by MAE (bpm):")
    for cand, mae, n in cmp.ranked[:COMPARE.top_n]:
        print(f"    {cand.label:<14} MAE {mae:5.2f}  (n={n})")
    return cmp


def main(argv=None):
    ap = argparse.ArgumentParser(description="CardiacSense respiration-rate toolchain (Python port)")
    ap.add_argument("--recording", default=None, help="folder with one .edf + one rt_flow*.csv")
    ap.add_argument("--edf", default=None, help="reference EDF path (overrides --recording)")
    ap.add_argument("--watch", default=None, help="watch rt_flow CSV path (overrides --recording)")
    ap.add_argument("--no-show", action="store_true", help="compute + print only; no figure windows")
    args = ap.parse_args(argv)

    edf_path, csv_path = args.edf, args.watch
    if edf_path is None or csv_path is None:
        rec = args.recording or DEFAULT_RECORDING
        e, c = _find_inputs(rec)
        edf_path = edf_path or e
        csv_path = csv_path or c
        print(f"Recording: {rec}")

    if not edf_path and not csv_path:
        ap.error("No inputs found. Pass --recording DIR or --edf/--watch paths.")

    ref = run_reference(edf_path) if edf_path else None
    ppg_results = sig = None
    if csv_path:
        ppg_results, sig = run_ppg(csv_path)

    cmp = None
    if ref is not None and ppg_results:
        cmp = run_comparison(ppg_results, ref)

    if not args.no_show:
        if ref is not None:
            viz.plot_reference(ref)
        if ppg_results:
            for ch, res in ppg_results.items():
                viz.plot_ppg_channel(res)
                viz.plot_ppg_spectrograms(res)   # 4 per-parameter spectrograms
        if cmp is not None:
            viz.plot_comparison(cmp, top_n=COMPARE.top_n)
        print("\nOpening figures — close the windows to exit.")
        viz.show_all()


if __name__ == "__main__":
    main()
