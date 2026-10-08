#!/usr/bin/env python
"""fit_sim.py — strap-placement simulator (single resting recording -> verdict).

Feed ANY resting recording; for each still, settled window it classifies strap
placement as GOOD / TOO TIGHT / TOO LOOSE, then gives one overall verdict.

STATUS (lab, NOT validated): built as scaffolding. On the two Exp3/Test loosening
sweeps the SNR-based "alive" test FAILED (cardiac SNR runs backwards — higher when
tighter; resp SNR flat). The only tightness-sensitive feature is artifact RMS, which
is not absolute-transferable (~6x across subjects); DC-IR is the transferable axis.
Thresholds below are PROVISIONAL and will be calibrated once 3-level LABELED
fixed-tightness recordings (too tight / good / too loose, >=2-3 subjects) exist.
See memory: strap-tightness-detector.

Usage:  python fit_sim.py [recording.csv ...]   (default: the two Exp3 Test recs)
"""
import sys, os, numpy as np
REPO = r"C:\Users\RachelMizrahi\OneDrive - CardiacSense\Documents\GitHub\BBBRR"
sys.path.insert(0, REPO)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from scipy.signal import butter, sosfiltfilt, welch
from main import run_ppg

USER = r"C:\Users\RachelMizrahi"
DEFAULT = [
    os.path.join(USER, r"CardiacSense\shares - BBB Respiration Rate\Data\Exp3\Test\001\pilot01_13.csv"),
    os.path.join(USER, r"CardiacSense\shares - BBB Respiration Rate\Data\Exp3\Test\002\pilot02_13.csv"),
]

WIN, HOP = 8.0, 2.0
SIG = (0.10, 3.0)          # artifact signal band (RMS)
MOT = (5.0, 20.0)          # motion / contact-noise band
RESP = (0.12, 0.60)        # respiration peak search
CARD = (0.70, 2.50)        # cardiac peak search

# --- provisional thresholds (to calibrate from labeled data) ---
SNR_GOOD = 6.0             # artifact SNR above this = alive  (NOTE: SNR shown unreliable)
DC_TIGHT = 55000.0         # DC-IR above this = compressed/tight
MOT_PCT = 60              # window valid if motion below this percentile
DC_SLOPE_MAX = 1500.0      # |dDC/dt| per sec below this = settled


def bp(x, fs, lo, hi):
    ny = fs / 2.0
    sos = butter(2, [lo/ny, min(hi, ny*.99)/ny], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def band_snr(seg, fs, lo, hi):
    f, p = welch(seg, fs=fs, nperseg=min(len(seg), int(fs*4)))
    b = (f >= lo) & (f <= hi)
    if b.sum() < 2:
        return 0.0
    return float(p[b].max() / (np.median(p[b]) + 1e-12))


def features(csv):
    _, sig = run_ppg(csv); fs = float(sig.fs)
    art = np.asarray(sig.channels["Artifact"], float)
    ir = np.asarray(sig.channels["IR"], float)
    a_sig = bp(art, fs, *SIG); a_mot = bp(art, fs, *MOT)
    w, h = int(WIN*fs), int(HOP*fs)
    rows = []
    prev_dc = None
    for s in range(0, len(art) - w + 1, h):
        t = (s + w/2) / fs
        seg_raw = art[s:s+w]
        dc = float(np.mean(ir[s:s+w]))
        slope = 0.0 if prev_dc is None else abs(dc - prev_dc) / HOP
        prev_dc = dc
        rows.append(dict(
            t=t, dc=dc, dc_slope=slope,
            motion=float(np.sqrt(np.mean(a_mot[s:s+w]**2))),
            art_rms=float(np.sqrt(np.mean(a_sig[s:s+w]**2))),
            resp_snr=band_snr(seg_raw, fs, *RESP),
            card_snr=band_snr(seg_raw, fs, *CARD),
        ))
    return rows


def classify(r):
    snr = max(r["resp_snr"], r["card_snr"])
    if snr >= SNR_GOOD:
        return "GOOD"
    return "TIGHT" if r["dc"] >= DC_TIGHT else "LOOSE"


def run(csv):
    name = os.path.basename(os.path.dirname(csv)) + "/" + os.path.basename(csv)
    rows = features(csv)
    mot_thr = np.percentile([r["motion"] for r in rows], MOT_PCT)
    print("\n" + "=" * 78)
    print("%s   (%d windows, motion_thr=%.0f)" % (name, len(rows), mot_thr))
    print("  %5s %8s %8s %7s %7s %8s  %-6s %s" %
          ("t(s)", "DC-IR", "dDC/dt", "respSN", "cardSN", "artRMS", "class", "valid"))
    valid_classes = []
    for r in rows:
        valid = (r["motion"] <= mot_thr) and (r["dc_slope"] <= DC_SLOPE_MAX)
        cls = classify(r) if valid else "-"
        if valid:
            valid_classes.append(cls)
        print("  %5.0f %8.0f %8.0f %7.1f %7.1f %8.0f  %-6s %s" %
              (r["t"], r["dc"], r["dc_slope"], r["resp_snr"], r["card_snr"],
               r["art_rms"], cls, "ok" if valid else "skip(motion/slope)"))
    if valid_classes:
        from collections import Counter
        c = Counter(valid_classes)
        verdict = c.most_common(1)[0][0]
        frac = c[verdict] / len(valid_classes)
        print("  --> VERDICT: %s  (%.0f%% of %d valid windows)  breakdown=%s" %
              (verdict, 100*frac, len(valid_classes), dict(c)))


if __name__ == "__main__":
    recs = sys.argv[1:] or DEFAULT
    for csv in recs:
        try: run(csv)
        except Exception:
            import traceback; traceback.print_exc()
