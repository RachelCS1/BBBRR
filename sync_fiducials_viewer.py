#!/usr/bin/env python
"""
Interactive manual-sync viewer — watch SS/MSD fiducials vs the REMbo Pulse Wave.

Opens THREE windows, all on the REMbo clock:

  1. watch raw   : the sync channel (Green for watch-13) baseline-corrected pulse
                   with its SS (^) and MSD (D) fiducials, SHIFTED by the offset.
  2. REMbo raw   : the REMbo finger Pulse Wave with its own SS / MSD fiducials
                   (the fixed reference clock).
  3. intervals   : the four inter-fiducial-interval (IBI) curves OVERLAID on one
                   axis — watch-SS, watch-MSD, REMbo-SS, REMbo-MSD interval-to-
                   previous — with a check-box per curve so you can hide some and
                   compare just two. The watch curves shift with the offset.

A single 'offset' slider (in window 3) drags the watch over the REMbo in real
time — in the raw-watch window and the interval window at once. The initial
offset is the automatic MSD lock (offset_from_msd); 'auto' resets to it.

The fiducial detection is EXACTLY the pipeline's: analyze_ppg_channel on the
1024 Hz channel (LPF-derivative beats -> SS refine -> MSD), so what you see is
how the real algorithm finds SS/MSD on the green signal under the sync.

Usage
-----
    py sync_fiducials_viewer.py --edf a.edf --watch b.csv
    py sync_fiducials_viewer.py --rec-id Exp2/002
    py sync_fiducials_viewer.py --edf ... --watch ... --offset 23.1 --channel Green
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

from respiration_rr.settings import PPG, SYNC
from respiration_rr.io.csv_reader import read_watch_auto
from respiration_rr.ppg.preprocess import prepare_watch
from respiration_rr.ppg.respiration import analyze_ppg_channel
from respiration_rr.preprocessing.resample import resample_linear, upsample_fft
from respiration_rr.sync import read_rembo_pulse_wave, offset_from_msd
import bw_bp_sweep as S          # discover / _find_inputs / DEFAULT_DATA_ROOT


CHANNEL_COLOR = {"Green": "#22c55e", "Red": "#ef4444", "IR": "#a855f7",
                 "Yellow": "#eab308", "Artifact": "#38bdf8"}
REMBO_COLOR = "#2563eb"


def _rembo_fiducials(pw, fs, invert, cfg=PPG):
    """REMbo Pulse Wave -> the same 1024 Hz beat pipeline -> (t, bc, ss_idx, msd_idx).

    Mirrors sync.msd_series (resample to fs_orig -> FFT-upsample x4 -> trim), but
    keeps the full fiducial result so SS as well as MSD can be drawn.
    """
    x = np.asarray(pw, np.float64)
    at256 = resample_linear(x, fs, cfg.fs_orig)
    up = upsample_fft(at256, cfg.upsample_factor)
    F = cfg.target_fs
    h = int(round(cfg.trim_head_sec * F))
    tl = int(round(cfg.trim_tail_sec * F))
    lo, hi = h, max(h + 1, up.size - tl)
    sig = up[lo:hi]
    t = np.arange(lo, hi) / F
    r = analyze_ppg_channel(sig, t, F, channel="IR", invert=invert, compute_ridge=False)
    idx = lambda a: np.asarray(a, int)[(np.asarray(a, int) >= 0) & (np.asarray(a, int) < t.size)]
    return t, np.asarray(r.bc, np.float64), idx(r.ss_idx), idx(r.msd_idx)


def _decimate(t, y, target_fs=120.0):
    """Thin a dense 1024 Hz trace for responsive redraw (markers stay exact).

    120 Hz keeps ~85 samples per pulse (plenty of waveform detail) while cutting
    the moving line to ~1/8 the points, so slider drags stay snappy.
    """
    if t.size < 2:
        return t, y
    fs = 1.0 / np.median(np.diff(t))
    step = max(1, int(round(fs / target_fs)))
    return t[::step], y[::step]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactive manual-sync viewer (watch SS/MSD vs REMbo)")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None, help="recording id, e.g. Exp2/002 (else first)")
    ap.add_argument("--edf", default=None, help="explicit reference EDF path (with --watch)")
    ap.add_argument("--watch", default=None, help="explicit watch CSV path (with --edf)")
    ap.add_argument("--channel", default=None,
                    help="watch channel to show (default: the recording's sync channel, "
                         "Green for watch-13; else IR)")
    ap.add_argument("--offset", type=float, default=None,
                    help="initial offset (s) to ADD to the watch clock (default: auto MSD lock)")
    ap.add_argument("--window-sec", type=float, default=90.0,
                    help="initial x-zoom span (s); pan/zoom with the toolbar afterwards")
    args = ap.parse_args(argv)

    # ---- resolve inputs (direct paths win; else discover by rec-id) ----
    if args.edf and args.watch:
        edf_path, csv_path = args.edf, args.watch
    elif args.edf or args.watch:
        sys.exit("--edf and --watch must be given together.")
    else:
        recs = S.discover(args.data_root)
        if args.rec_id:
            recs = [r for r in recs if r[0] == args.rec_id]
        if not recs:
            sys.exit("No recording found (pass --edf/--watch or a valid --rec-id).")
        _, edf_path, csv_path = recs[0]

    print(f"Watch : {csv_path}")
    print(f"EDF   : {edf_path}")

    watch = read_watch_auto(csv_path)
    sig = prepare_watch(watch)
    chan = args.channel or getattr(sig, "sync_channel", None) or "IR"
    if chan not in sig.channels:
        sys.exit(f"Channel '{chan}' not in {list(sig.channels)}")
    print(f"Sync channel: {chan}")

    # ---- watch fiducials on the chosen channel (real pipeline) ----
    gres = analyze_ppg_channel(sig.channels[chan], sig.time, sig.fs, channel=chan,
                               move_regions=sig.move_regions)
    w_bc = np.asarray(gres.bc, np.float64)
    w_t = np.asarray(sig.time, np.float64)
    w_ss = np.asarray(gres.ss_idx, int)
    w_msd = np.asarray(gres.msd_idx, int)

    # ---- REMbo Pulse Wave + auto offset (for the initial slider value + polarity) ----
    # Auto-lock on the recording's sync fiducial (SS for watch-13, else MSD), so the
    # initial offset matches what the real pipeline uses.
    sync_fid = getattr(sig, "sync_fiducial", None) or "MSD"
    pw, pwfs, pwname = read_rembo_pulse_wave(edf_path)
    w_fid_auto = w_t[w_ss] if sync_fid == "SS" else w_t[w_msd]
    auto = offset_from_msd(w_fid_auto, pw, pwfs, fiducial=sync_fid)
    invert = auto.polarity < 0
    r_t, r_bc, r_ss, r_msd = _rembo_fiducials(pw, pwfs, invert)
    init_off = args.offset if args.offset is not None else auto.offset_sec
    print(f"REMbo '{pwname}' @ {pwfs:.0f} Hz | polarity {'inverted' if invert else 'as-is'}"
          f" | auto-lock fiducial: {sync_fid}")
    print(f"auto offset = {auto.offset_sec:+.3f} s | matched {auto.matched} "
          f"({auto.matched_frac*100:.0f}%) | {auto.prominence:.1f}sigma"
          f"{'  LOW: ' + auto.reason if auto.low_confidence else ''}")
    print(f"initial offset = {init_off:+.3f} s   (drag the slider to refine)")

    # ---- figures ----
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, CheckButtons

    wc = CHANNEL_COLOR.get(chan, "#22c55e")
    w_td, w_bcd = _decimate(w_t, w_bc)               # thinned traces for fast redraw
    r_td, r_bcd = _decimate(r_t, r_bc)
    w_ss_t, w_msd_t = w_t[w_ss], w_t[w_msd]
    r_ss_t, r_msd_t = r_t[r_ss], r_t[r_msd]
    span = SYNC.max_offset_sec
    x0 = max(r_t[0], (w_msd_t[0] + init_off) if w_msd_t.size else r_t[0])
    xspan = args.window_sec

    # ============= Window 1: watch raw signal + SS/MSD =============
    figW, axW = plt.subplots(figsize=(13, 5.2), num=f"{chan} (watch) — raw + SS/MSD")
    figW.subplots_adjust(bottom=0.14)
    (w_line,) = axW.plot(w_td + init_off, w_bcd, color=wc, lw=0.7, alpha=0.9, label=chan)
    (w_ss_m,) = axW.plot(w_ss_t + init_off, w_bc[w_ss], "^", color="#16a34a", ms=7,
                         ls="none", label=f"SS ({w_ss.size})")
    (w_msd_m,) = axW.plot(w_msd_t + init_off, w_bc[w_msd], "D", color="#a855f7", ms=6,
                          ls="none", label=f"MSD ({w_msd.size})")
    axW.set_title(f"{chan} (watch)", fontsize=10); axW.set_xlabel("time — REMbo clock (s)")
    axW.set_ylabel("a.u."); axW.legend(loc="upper right", fontsize=8); axW.grid(True, alpha=.25)
    axW.set_xlim(x0, x0 + xspan)

    # ============= Window 2: REMbo raw signal + SS/MSD (static reference clock) =============
    figR, axR = plt.subplots(figsize=(13, 5.2), num="REMbo — raw + SS/MSD")
    figR.subplots_adjust(bottom=0.14)
    axR.plot(r_td, r_bcd, color=REMBO_COLOR, lw=0.7, alpha=0.9, label="REMbo")
    axR.plot(r_ss_t, r_bc[r_ss], "^", color="#0ea5e9", ms=7, ls="none", label=f"SS ({r_ss.size})")
    axR.plot(r_msd_t, r_bc[r_msd], "D", color="#ef4444", ms=6, ls="none", label=f"MSD ({r_msd.size})")
    axR.set_title("REMbo Pulse Wave", fontsize=10); axR.set_xlabel("time — REMbo clock (s)")
    axR.set_ylabel("a.u."); axR.legend(loc="upper right", fontsize=8); axR.grid(True, alpha=.25)
    axR.set_xlim(x0, x0 + xspan)

    # ============= Window 3: interval (IBI) curves overlaid + on/off toggles =============
    def _ibi(f):
        f = np.asarray(f, float)
        if f.size < 2:
            return np.zeros(0), np.zeros(0)
        return f[1:], np.diff(f)                       # (time, interval to previous fiducial, s)
    wss_x, wss_y = _ibi(w_ss_t); wmsd_x, wmsd_y = _ibi(w_msd_t)
    rss_x, rss_y = _ibi(r_ss_t); rmsd_x, rmsd_y = _ibi(r_msd_t)

    figI, axI = plt.subplots(figsize=(14, 6.5), num="Interval (IBI) alignment")
    figI.subplots_adjust(bottom=0.20, left=0.22)
    (l_wss,)  = axI.plot(wss_x + init_off,  wss_y,  "-^", color="#16a34a", ms=3, lw=1.0,
                         label="watch SS interval")
    (l_wmsd,) = axI.plot(wmsd_x + init_off, wmsd_y, "-D", color="#a855f7", ms=3, lw=1.0,
                         label="watch MSD interval")
    (l_rss,)  = axI.plot(rss_x,  rss_y,  "-^", color="#0ea5e9", ms=3, lw=1.0,
                         label="REMbo SS interval")
    (l_rmsd,) = axI.plot(rmsd_x, rmsd_y, "-D", color="#ef4444", ms=3, lw=1.0,
                         label="REMbo MSD interval")
    axI.set_ylabel("interval to previous fiducial (s)"); axI.set_xlabel("time — REMbo clock (s)")
    axI.legend(loc="upper right", fontsize=8); axI.grid(True, alpha=.25)
    axI.set_xlim(x0, x0 + xspan)

    ibi_labels = ["watch SS interval", "watch MSD interval", "REMbo SS interval", "REMbo MSD interval"]
    ibi_lines  = [l_wss, l_wmsd, l_rss, l_rmsd]

    # checkboxes: untick a curve to hide it and compare just two "ideal" curves
    axck = figI.add_axes([0.01, 0.30, 0.19, 0.42]); axck.set_title("show curves", fontsize=8)
    checks = CheckButtons(axck, ibi_labels, [True, True, True, True])
    def _toggle(label):
        i = ibi_labels.index(label)
        ibi_lines[i].set_visible(not ibi_lines[i].get_visible())
        figI.canvas.draw_idle()
    checks.on_clicked(_toggle)

    # shared offset -> shifts the WATCH artists in window 1 AND the two watch interval
    # curves in window 3 (REMbo is the fixed reference clock, never shifted).
    watch_shift = [(w_line, w_td), (w_ss_m, w_ss_t), (w_msd_m, w_msd_t),
                   (l_wss, wss_x), (l_wmsd, wmsd_x)]

    ax_sl = figI.add_axes([0.24, 0.06, 0.52, 0.03])
    slider = Slider(ax_sl, "offset (s)", init_off - span, init_off + span,
                    valinit=init_off, valfmt="%+.3f")
    figI.suptitle(f"Interval alignment — offset = {init_off:+.3f} s", fontsize=11)

    def update(_v=None):
        off = slider.val
        for art, bx in watch_shift:
            art.set_xdata(bx + off)
        figI.suptitle(f"Interval alignment — offset = {off:+.3f} s", fontsize=11)
        figW.canvas.draw_idle(); figI.canvas.draw_idle()
    slider.on_changed(update)

    ax_btn = figI.add_axes([0.80, 0.055, 0.1, 0.04])
    btn = Button(ax_btn, "auto")
    btn.on_clicked(lambda _e: slider.set_val(auto.offset_sec))

    print("\nThree windows: (1) the watch raw signal and (2) the REMbo raw signal, each with "
          "SS/MSD markers; (3) the four interval (IBI) curves overlaid. Drag 'offset' to slide "
          "the watch over the REMbo (windows 1 and 3); untick curves to compare just two. "
          "'auto' resets to the MSD lock. Close a window to exit.")
    figI._keep = (slider, checks, btn)                # keep widgets alive
    plt.show()


if __name__ == "__main__":
    main()
