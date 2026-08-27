#!/usr/bin/env python
"""
spectro_avg_lab.py — design the spectrogram -> average-RR method on the REMbo
(reference) airflow signal, where the truth is known.

The reference airflow is a clean, direct respiration signal, so it is the right
place to calibrate HOW we read an average RR from a spectrogram before applying
the same recipe to the (noisier) BW-extraction methods.

For each FFT window length it:
  * computes the STFT ridge (per-window dominant frequency in the RR band),
  * smooths it into an average (same windowed-median convention as everywhere),
  * scores MAE (bpm) vs the breath-counted average RR (the truth),
  * reports two reference-free confidence measures — spectral prominence
    (peak / band-median) and ridge stability (IQR of the per-window RR).

Usage
-----
    py spectro_avg_lab.py                       # first recording
    py spectro_avg_lab.py --rec-id Exp2/004
    py spectro_avg_lab.py --windows 12 16 24 32
    py spectro_avg_lab.py --save DIR --no-show
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import bw_bp_sweep as S      # reuse discover / _resolve_inputs


def spectro_average(sig, fs, win_sec, cfg, avg_fn, fundamental=False, ratio_th=0.85):
    """Return (ridge_t, ridge_rr, avg_t, avg_rr, med_prominence).

    fundamental=True uses ridge_rr_fundamental (harmonic-aware peak selection)
    instead of the bare-argmax ridge_rr.
    """
    from respiration_rr.ppg.spectrogram import compute_spectrogram, ridge_rr, ridge_rr_fundamental
    spect = compute_spectrogram(np.asarray(sig, np.float64) - np.mean(sig),
                                fs, win_sec, cfg.rr_band_high_hz + 0.2)
    if spect is None:
        return None
    if fundamental:
        st, rr = ridge_rr_fundamental(spect, cfg.rr_band_low_hz, cfg.rr_band_high_hz, ratio_th)
    else:
        st, rr = ridge_rr(spect, cfg.rr_band_low_hz, cfg.rr_band_high_hz)
    # spectral prominence per column: peak / band-median (robust "how it stands out")
    freqs = spect["freqs"]
    band = (freqs >= cfg.rr_band_low_hz) & (freqs <= cfg.rr_band_high_hz)
    P = spect["power_lin"][band, :]
    peak = P.max(axis=0)
    med = np.median(P, axis=0)
    prom = np.divide(peak, med, out=np.full_like(peak, np.nan), where=med > 0)
    at, ar = avg_fn(st, rr)                          # windowed-median smoothing
    return st, rr, at, ar, float(np.nanmedian(prom))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrate spectrogram->average RR on the REMbo signal")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--edf", default=None)
    ap.add_argument("--watch", default=None)
    ap.add_argument("--windows", type=float, nargs="+", default=[12, 16, 24, 32])
    ap.add_argument("--out", default=r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\spectro_avg")
    ap.add_argument("--save", default=None, metavar="DIR")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    # only the EDF (reference) is needed; allow it without a watch csv
    if args.edf:
        edf_path, rid = args.edf, "custom"
    else:
        rid, edf_path, _csv = S._resolve_inputs(args)
    print(f"Recording: {rid}\n  EDF   : {edf_path}")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    from respiration_rr.settings import PPG
    from main import run_reference
    from respiration_rr.rr_average import reference_average_rr, reference_rr_points, average_series, mae_over_overlap

    ref = run_reference(edf_path)
    sig, fs = np.asarray(ref.filtered, np.float64), float(ref.fs)
    truth_t, truth_r = reference_average_rr(ref)               # breath-counted average (truth)
    brk_t, brk_r = reference_rr_points(ref)                    # per-breath points
    avg_fn = lambda t, r: average_series(t, r, PPG)

    print(f"\n[Spectro->avg on REMbo]  fs={fs:.0f}Hz  band={PPG.rr_band_low_hz}-{PPG.rr_band_high_hz}Hz  "
          f"breaths={brk_t.size}")
    print(f"  {'win(s)':>6} {'MAE(bpm)':>9} {'prominence':>11} {'ridgeIQR':>9}")

    rows, best = {}, (None, np.inf)
    for W in args.windows:
        out = spectro_average(sig, fs, W, PPG, avg_fn)
        if out is None:
            print(f"  {W:6.0f}   (signal too short)")
            continue
        st, rr, at, ar, prom = out
        mae, n = mae_over_overlap(at, ar, truth_t, truth_r)
        iqr = float(np.nanpercentile(rr, 75) - np.nanpercentile(rr, 25)) if np.isfinite(rr).any() else np.nan
        rows[W] = dict(st=st, rr=rr, at=at, ar=ar, prom=prom, mae=mae, iqr=iqr)
        flag = ""
        if np.isfinite(mae) and mae < best[1]:
            best = (W, mae); flag = "  <= best"
        print(f"  {W:6.0f} {mae:9.2f} {prom:11.1f} {iqr:9.2f}{flag}")

    if best[0] is None:
        sys.exit("No window scored.")
    print(f"\nBEST window: {best[0]:.0f}s  (MAE {best[1]:.2f} bpm vs breath-counted average)")

    # ---- figure: spectrogram + ridge + truth for the best window ----
    from respiration_rr.ppg.spectrogram import compute_spectrogram
    W = best[0]
    spect = compute_spectrogram(sig - sig.mean(), fs, W, PPG.rr_band_high_hz + 0.2)
    r = rows[W]
    fig, ax = plt.subplots(figsize=(15, 6))
    ex = [spect["times"][0], spect["times"][-1], spect["freqs"][0] * 60, spect["freqs"][-1] * 60]
    ax.imshow(spect["power_db"], aspect="auto", origin="lower", extent=ex, cmap="magma")
    ax.plot(r["st"], r["rr"], ".", color="#22d3ee", ms=3, alpha=0.5, label="ridge (per-window)")
    if brk_t.size:
        ax.plot(brk_t, brk_r, "o", color="#a3e635", ms=4, alpha=0.7, label="breath-counted RR")
    ax.plot(truth_t, truth_r, "-", color="#f8fafc", lw=2.4, label="truth (breath-counted avg)")
    ax.plot(r["at"], r["ar"], "-", color="#ef4444", lw=2.0, label="spectrogram avg")
    ax.set_ylim(0, (PPG.rr_band_high_hz + 0.1) * 60)
    ax.set_xlabel("time (s)"); ax.set_ylabel("RR (bpm)")
    ax.set_title(f"REMbo spectrogram -> average RR   |   {rid}   |   window {W:.0f}s   "
                 f"MAE {best[1]:.2f} bpm  ·  prominence {r['prom']:.1f}",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_dir = args.save or (args.out if args.no_show else None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f"{rid.replace('/', '_')}_spectro_avg.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"  saved -> {png}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
