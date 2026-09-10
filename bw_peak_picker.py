#!/usr/bin/env python
"""
bw_peak_picker.py — INTERACTIVE breath-peak picker on the raw-channel BW signal.

Purpose: on ONE channel's band-passed (BW) trace, check whether the breaths you
expect are actually PRESENT in the signal. The tool seeds the peaks with the
pipeline's own BW detection, then lets you CLICK to add the peaks it missed (and
remove wrong ones). The bottom panel shows the RR you'd get from the current peak
set against the reference — so you can see, live, whether hand-completing the peaks
makes the RR match the reference (i.e. the data really is in the signal).

Nothing here touches the pipeline. It band-passes the raw channel exactly like the
BW param (PPG.bw_band_low/high_hz) and reuses the pipeline's breath-start detector
only to SEED the picks.

Controls
--------
    LEFT click  on the trace  : add a breath peak (snaps to the nearest local max)
    RIGHT click near a marker : remove that peak
    r                         : reset to the auto-detected seed
    u                         : undo the last add/remove
    d                         : toggle a detrend (high-pass) on the displayed trace

Usage
-----
    py bw_peak_picker.py                        # Exp2/002, Artifact, global BW band
    py bw_peak_picker.py --rec-id Exp2/004
    py bw_peak_picker.py --channel IR --band 0.5 0.9   # focus on the high-rate band
    py bw_peak_picker.py --hp 0.4               # start with detrend on
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

import bw_bp_sweep as S
from respiration_rr.settings import PPG

SNAP_SEC = 0.4          # click snaps to the tallest sample within +/- this
REMOVE_SEC = 0.4        # right-click removes the nearest peak within +/- this


def _highpass(y, fs, cutoff, order=2):
    y = np.asarray(y, float)
    if y.size < 9 or cutoff <= 0:
        return y - (y.mean() if y.size else 0.0)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, y)
    except Exception:
        return y - y.mean()


def _seed_peaks(tr, fs, t, mr):
    """Seed with the pipeline's own BW breath-start detection (indices into tr)."""
    from respiration_rr.ppg.respiration import _breath_starts_raw
    bw_prom = getattr(PPG, "bw_prominence", None) or PPG.breath_start_prominence
    bs = _breath_starts_raw(tr, fs, bw_prom, t=t, cfg=PPG, move_regions=mr)
    return np.asarray(bs, int)


def _rr_from_peaks(pk_t):
    """RR (bpm) at each mid-interval of sorted peak TIMES (watch clock)."""
    pk = np.sort(np.asarray(pk_t, float))
    if pk.size < 2:
        return np.zeros(0), np.zeros(0)
    d = np.diff(pk)
    good = d > 1e-6
    mid = (pk[:-1] + pk[1:]) / 2.0
    return mid[good], 60.0 / d[good]


def _handpick_dir(create=False):
    """Where hand-picks live. Tries the script dir, then CWD, then temp (OneDrive can block
    new-dir creation). With create=False just returns the first existing (or the primary)."""
    import os, tempfile
    cands = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "bw_handpicks"),
             os.path.join(os.getcwd(), "bw_handpicks"),
             os.path.join(tempfile.gettempdir(), "bw_handpicks")]
    for d in cands:
        if os.path.isdir(d):
            return d
        if create:
            try:
                os.makedirs(d, exist_ok=True)
                return d
            except OSError:
                continue
    return cands[0]


def _handpick_path(stem, create=False):
    import os
    return os.path.join(_handpick_dir(create), stem + ".json")


class Picker:
    def __init__(self, t, tr_raw, fs, offset, mr, ref_t, ref_r, seed_idx, title, hp0, meta=None):
        self.t = np.asarray(t, float)
        self.tr_raw = np.asarray(tr_raw, float)
        self.fs = fs
        self.offset = offset
        self.mr = mr
        self.ref_t = np.asarray(ref_t, float)
        self.ref_r = np.asarray(ref_r, float)
        self.seed = np.sort(self.t[seed_idx]) if seed_idx.size else np.zeros(0)
        self.peaks = self.seed.copy()          # breath times, watch clock
        self.history = []                      # for undo
        self.hp = hp0                          # detrend cutoff (Hz) or None
        self.title = title
        self.meta = meta or {}
        # resume: if hand-picks were saved before for this rec/channel, start from them
        if self.meta.get("stem") and self.meta.get("load", True):
            import os, json
            p = _handpick_path(self.meta["stem"])
            if os.path.exists(p):
                try:
                    saved = np.asarray(json.load(open(p))["peaks_watch"], float)
                    if saved.size:
                        self.peaks = np.sort(saved)
                        print(f"loaded {saved.size} hand-picks from {p}")
                except Exception as e:
                    print(f"  (could not load existing hand-picks: {e})")

        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, (self.axT, self.axR) = plt.subplots(
            2, 1, figsize=(15, 7), sharex=False,
            gridspec_kw={"height_ratios": [3, 2]})
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    # ---- displayed trace (optionally detrended) ----
    def _trace(self):
        return _highpass(self.tr_raw, self.fs, self.hp) if self.hp else self.tr_raw

    def _snap(self, x_rembo):
        """Nearest local-max sample to a click (REMbo clock), within +/- SNAP_SEC.
        Returns a WATCH-clock time (peaks are stored on the watch clock)."""
        xw = x_rembo - self.offset                          # REMbo -> watch
        tr = self._trace()
        m = (self.t >= xw - SNAP_SEC) & (self.t <= xw + SNAP_SEC)
        if not m.any():
            return None
        idx = np.where(m)[0]
        return float(self.t[idx[np.argmax(tr[idx])]])

    def _on_click(self, ev):
        # Ignore clicks while the toolbar is in zoom/pan mode (else every zoom/pan adds a peak)
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        if ev.inaxes is not self.axT or ev.xdata is None:
            return
        if ev.button == 1:                                  # add
            pt = self._snap(ev.xdata)                        # watch-clock time
            if pt is None or (self.peaks.size and np.min(np.abs(self.peaks - pt)) < 1e-3):
                return
            self.history.append(self.peaks.copy())
            self.peaks = np.sort(np.append(self.peaks, pt))
        elif ev.button == 3 and self.peaks.size:            # remove nearest (compare on REMbo clock)
            disp = self.peaks + self.offset
            j = int(np.argmin(np.abs(disp - ev.xdata)))
            if abs(disp[j] - ev.xdata) <= REMOVE_SEC:
                self.history.append(self.peaks.copy())
                self.peaks = np.delete(self.peaks, j)
        else:
            return
        self._draw()

    def _on_key(self, ev):
        if ev.key == "r":
            self.history.append(self.peaks.copy())
            self.peaks = self.seed.copy()
        elif ev.key == "u" and self.history:
            self.peaks = self.history.pop()
        elif ev.key == "d":
            self.hp = None if self.hp else (self.hp_last if getattr(self, "hp_last", None) else 0.4)
            if self.hp:
                self.hp_last = self.hp
        elif ev.key == "s":
            self._save()
            return
        else:
            return
        self._draw()

    def _save(self):
        """Write the current hand-picked peaks (ground truth) to bw_handpicks/<stem>.json."""
        import json
        stem = self.meta.get("stem")
        if not stem:
            print("  (no stem — cannot save)")
            return
        path = _handpick_path(stem, create=True)
        data = {
            "rec_id": self.meta.get("rec_id"),
            "channel": self.meta.get("channel"),
            "band": self.meta.get("band"),
            "fs": self.fs,
            "offset": self.offset,
            "clock": "watch (add offset -> REMbo/reference clock)",
            "peaks_watch": [float(x) for x in np.sort(self.peaks)],
        }
        try:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            print(f"saved {self.peaks.size} hand-picks -> {path}")
        except OSError as e:
            print(f"  SAVE FAILED ({e}) — could not write {path}")

    def _draw(self):
        tr = self._trace()
        # preserve the current zoom/pan across redraws (only the FIRST draw sets full range)
        first = not getattr(self, "_drawn", False)
        if not first:
            txlim, tylim = self.axT.get_xlim(), self.axT.get_ylim()
            rxlim, rylim = self.axR.get_xlim(), self.axR.get_ylim()
        # ---- top: trace + peaks (displayed on the REMbo clock = watch + offset) ----
        self.axT.clear()
        self.axT.plot(self.t + self.offset, tr, "-", color="#1d4ed8", lw=0.8, alpha=0.85)
        if self.peaks.size:
            yv = np.interp(self.peaks, self.t, tr)
            xp = self.peaks + self.offset                   # peaks on REMbo clock for display
            seedset = set(np.round(self.seed, 3).tolist())
            is_seed = np.array([round(p, 3) in seedset for p in self.peaks])
            if is_seed.any():
                self.axT.plot(xp[is_seed], yv[is_seed], "v", ms=7, color="#1d4ed8",
                              label="auto (seed)")
            if (~is_seed).any():
                self.axT.plot(xp[~is_seed], yv[~is_seed], "o", ms=9, mfc="none",
                              mec="#dc2626", mew=2, label="added by hand")
        hp_s = f"  |  detrend HP={self.hp} Hz" if self.hp else "  |  no detrend"
        self.axT.set_title(f"{self.title}   breaths={self.peaks.size}{hp_s}\n"
                           "LEFT=add  RIGHT=remove  r=reset  u=undo  d=detrend  s=save", fontsize=10)
        self.axT.set_ylabel("BW (band-passed)")
        self.axT.legend(fontsize=8, loc="upper right")
        self.axT.grid(True, alpha=0.2)

        # ---- bottom: RR vs reference ----
        self.axR.clear()
        self.axR.plot(self.ref_t, self.ref_r, "-", color="#0f172a", lw=1.6, label="reference RR")
        rr_t, rr = _rr_from_peaks(self.peaks)
        mae_s = "n/a"
        if rr_t.size:
            rr_t_r = rr_t + self.offset                     # watch -> REMbo clock
            self.axR.plot(rr_t_r, rr, ".-", color="#dc2626", ms=5, lw=0.9, label="RR from picks")
            inside = (rr_t_r >= self.ref_t[0]) & (rr_t_r <= self.ref_t[-1])
            if inside.any():
                err = np.abs(rr[inside] - np.interp(rr_t_r[inside], self.ref_t, self.ref_r))
                mae_s = f"{err.mean():.2f} bpm  (n={int(inside.sum())})"
        self.axR.set_title(f"RR from picks vs reference   MAE={mae_s}", fontsize=10)
        self.axR.set_xlabel("time (s, REMbo clock)")
        self.axR.set_ylabel("RR (bpm)")
        self.axR.legend(fontsize=8, loc="upper right")
        self.axR.grid(True, alpha=0.2)
        # restore the view: full range on first draw, else keep the user's zoom/pan
        if first:
            self.axT.set_xlim(self.ref_t[0], self.ref_t[-1])
            self.axR.set_xlim(self.ref_t[0], self.ref_t[-1])
            self._drawn = True
        else:
            self.axT.set_xlim(txlim); self.axT.set_ylim(tylim)
            self.axR.set_xlim(rxlim); self.axR.set_ylim(rylim)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Interactive BW breath-peak picker (one channel)")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default="Exp2/002", help="recording id (default Exp2/002)")
    ap.add_argument("--channel", default="Artifact", help="channel (default Artifact)")
    ap.add_argument("--band", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
                    help="BW band-pass Hz (default = PPG.bw_band_low/high_hz)")
    ap.add_argument("--hp", type=float, default=None,
                    help="start with a detrend high-pass at this cutoff (Hz)")
    ap.add_argument("--no-seed", action="store_true",
                    help="start EMPTY (no auto-seed, no reload) — mark only your segment from scratch")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("TkAgg")

    from respiration_rr.ppg.dsp import bandpass_filter
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import reference_rr_series

    recs = [r for r in S.discover(args.data_root) if r[0] == args.rec_id]
    if not recs:
        sys.exit(f"recording {args.rec_id} not found")
    rid, edf, csv = recs[0]

    ref = run_reference(edf)
    results, sig = run_ppg(csv)
    offset = run_sync(edf, results, sig)
    ref_t, ref_r = reference_rr_series(ref)

    if args.channel not in results:
        sys.exit(f"channel {args.channel} not in {list(results)}")
    raw = np.asarray(sig.channels[args.channel], float)
    t = np.asarray(sig.time, float)
    fs = sig.fs
    mr = results[args.channel].move_regions

    lo = args.band[0] if args.band else PPG.bw_band_low_hz
    hi = args.band[1] if args.band else PPG.bw_band_high_hz
    tr = bandpass_filter(raw, fs, lo, hi, PPG.bw_filter_order)["filtered"]

    seed = np.zeros(0, int) if args.no_seed else _seed_peaks(tr, fs, t, mr)
    title = f"{rid} · {args.channel} · BW {lo}-{hi} Hz   (offset {offset:+.2f}s)"
    print(f"{title} | seed peaks: {seed.size} | reference breaths span "
          f"{ref_t[0]:.1f}-{ref_t[-1]:.1f}s")
    print("Close the window when done.")

    meta = {"rec_id": rid, "channel": args.channel, "band": [lo, hi],
            "stem": f"{rid.replace('/', '_')}_{args.channel}", "load": not args.no_seed}
    picker = Picker(t, tr, fs, offset, mr, ref_t, ref_r, seed, title, args.hp, meta)
    import matplotlib.pyplot as plt
    plt.show()
    return picker            # keep a strong ref alive (matplotlib callbacks are weak-ref'd)


if __name__ == "__main__":
    main()
