#!/usr/bin/env python
"""
deck_figs.py — generate the stage-by-stage figures of the Butter average-RR
pipeline on ONE real recording, for the explainer deck. Saves PNGs.

Figures:
  1_stages.png    raw Artifact -> Butterworth 0.1-1.0 -> envelope-detrend
                  (envelopes + midline shown), on a ~80 s window
  2_spectro.png   full-recording spectrogram + argmax ridge (<=0.9 Hz)
                  + reference RR, with the 0.9-1.0 Hz "junk" band shaded
  3_column.png    one FFT column (power vs RR-bpm) showing the breathing peak
                  and the 54-60 bpm junk region that ceiling 0.9 removes
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import bw_bp_sweep as S

OUT = r"C:\Users\RACHEL~1\AppData\Local\Temp\claude\C--Users-RachelMizrahi-OneDrive---CardiacSense-Documents-GitHub-BBBRR\3bbdb805-5d0d-4f52-b50c-def0402d0def\scratchpad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-id", default=None)
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import dataclasses
    from respiration_rr.settings import PPG as _PPG
    cfg = dataclasses.replace(_PPG, rr_band_high_hz=0.9)
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr, average_series
    from respiration_rr.ppg.bw_methods import butterworth, legacy_detrend
    from respiration_rr.ppg.legacy_bw import fft_bandpass, _envelope
    from respiration_rr.ppg.dsp import moving_average
    from respiration_rr.ppg.spectrogram import compute_spectrogram, ridge_rr
    from scipy.signal import find_peaks

    # pick a recording
    if args.rec_id:
        recs = [r for r in S.discover(args.data_root) if r[0] == args.rec_id] or [S._resolve_inputs(args)]
    else:
        recs = S.discover(args.data_root)
    rid, edf, csv = recs[0]
    print(f"Using recording: {rid}")

    ppg_results, sig = run_ppg(csv)
    ref = run_reference(edf)
    offset = run_sync(edf, ppg_results, sig)
    truth_at, truth_ar = reference_average_rr(ref)
    x = np.asarray(sig.channels[args.channel], np.float64)
    fs = float(sig.fs)
    low, high = cfg.bw_band_low_hz, cfg.bw_band_high_hz

    # ---- stages ----
    bp = butterworth(x, fs, low, high)                       # stage 1
    det = legacy_detrend(bp, fs, low, high, baseline_sec=5.0)  # stage 2
    # recompute the envelopes/midline exactly as detrend_peaks does, for display
    filt = fft_bandpass(bp, fs, low, high)
    mov = moving_average(filt, max(3, int(round(5.0 * fs))))
    grid = np.arange(filt.size)
    up, _ = find_peaks(filt); up = up[filt[up] >= mov[up]]
    above = _envelope(up, filt[up], grid)
    dn, _ = find_peaks(-filt); dn = dn[filt[dn] <= mov[dn]]
    below = _envelope(dn, filt[dn], grid)
    mid = (above + below) / 2.0

    t = np.arange(x.size) / fs
    # choose an 80 s window ~40% into the record
    w0 = t[int(0.40 * x.size)]; w1 = w0 + 80.0
    m = (t >= w0) & (t < w1)

    # ---------- FIG 1: stages ----------
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    axes[0].plot(t[m], x[m], color="#64748b", lw=0.9)
    axes[0].set_title("1 · Raw Artifact channel", loc="left", fontweight="bold", fontsize=11)
    axes[0].set_ylabel("a.u.")
    axes[1].plot(t[m], filt[m], color="#2563eb", lw=1.1, label="band-passed 0.1–1.0 Hz")
    axes[1].plot(t[m], above[m], color="#16a34a", lw=0.8, alpha=0.8, label="upper envelope")
    axes[1].plot(t[m], below[m], color="#f59e0b", lw=0.8, alpha=0.8, label="lower envelope")
    axes[1].plot(t[m], mid[m], color="#dc2626", lw=1.0, ls="--", label="midline (subtracted)")
    axes[1].set_title("2a · Butterworth band-pass + envelope midline", loc="left", fontweight="bold", fontsize=11)
    axes[1].set_ylabel("a.u."); axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    axes[2].plot(t[m], det[m], color="#0f766e", lw=1.2)
    axes[2].axhline(0, color="#94a3b8", lw=0.6)
    axes[2].set_title("2b · After detrend = band-passed − midline  (centred, ready for the spectrogram)",
                      loc="left", fontweight="bold", fontsize=11)
    axes[2].set_ylabel("a.u."); axes[2].set_xlabel("time (s)")
    for a in axes:
        a.grid(True, alpha=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "1_stages.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- FIG 2: spectrogram + ridge ----------
    spect = compute_spectrogram(det - det.mean(), fs, 32.0, 1.1)
    st, rr = ridge_rr(spect, cfg.rr_band_low_hz, cfg.rr_band_high_hz)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ex = [spect["times"][0], spect["times"][-1], spect["freqs"][0] * 60, spect["freqs"][-1] * 60]
    ax.imshow(spect["power_db"], aspect="auto", origin="lower", extent=ex, cmap="magma")
    ax.axhspan(54, 60, color="#38bdf8", alpha=0.18, lw=0)
    ax.text(spect["times"][int(len(spect["times"]) * 0.5)], 57, "0.9–1.0 Hz junk band (cut by ceiling 0.9)",
            color="#e0f2fe", fontsize=8, ha="center", va="center")
    ax.plot(np.asarray(st), rr, ".", color="#22d3ee", ms=3, alpha=0.6, label="argmax ridge (≤0.9 Hz) → our RR")
    if truth_at.size:
        ax.plot(truth_at - offset, truth_ar, "-", color="#f8fafc", lw=2.2, label="reference RR")
    ax.set_ylim(0, 66); ax.set_xlabel("time (s)"); ax.set_ylabel("RR (bpm)")
    ax.set_title(f"3 · Spectrogram of the detrended signal + ridge   ({rid}, window 32 s)",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "2_spectro.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---------- FIG 3: one FFT column ----------
    freqs = spect["freqs"]; P = spect["power_lin"]
    ci = int(P.shape[1] * 0.5)
    col = P[:, ci]
    bpmf = freqs * 60
    show = bpmf <= 66
    lo_bpm = cfg.rr_band_low_hz * 60.0        # 6  bpm
    hi_bpm = cfg.rr_band_high_hz * 60.0       # 54 bpm
    norm = col[show] / np.nanmax(col[show])   # DC bin is usually the tallest -> shows the trap
    band = (freqs >= cfg.rr_band_low_hz) & (freqs <= cfg.rr_band_high_hz)
    pk = int(np.where(band)[0][np.nanargmax(col[band])])   # peak WITHIN the search band
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(bpmf[show], norm, color="#2563eb", lw=1.5, zorder=3)
    ax.axvspan(0, lo_bpm, color="#94a3b8", alpha=0.22, lw=0,
               label=f"excluded < {lo_bpm:.0f} bpm  (DC / residual drift)")
    ax.axvspan(hi_bpm, 60, color="#f87171", alpha=0.20, lw=0,
               label=f"excluded > {hi_bpm:.0f} bpm  (impossible)")
    ax.plot(bpmf[pk], norm[pk], "o", color="#16a34a", ms=9, zorder=4,
            label=f"selected peak in-band (~{bpmf[pk]:.0f} bpm)")
    ax.axvline(lo_bpm, color="#475569", lw=1.0, ls="--")
    ax.axvline(hi_bpm, color="#dc2626", lw=1.0, ls="--")
    ax.set_xlabel("RR (bpm)"); ax.set_ylabel("normalised power"); ax.set_ylim(0, 1.08)
    ax.set_title("4 · One spectrogram column — the search band [6–54 bpm] excludes the DC spike",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "3_column.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    print("saved 1_stages.png, 2_spectro.png, 3_column.png to", args.out)


if __name__ == "__main__":
    main()
