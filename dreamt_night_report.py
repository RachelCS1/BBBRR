#!/usr/bin/env python
"""
dreamt_night_report.py — characterize DREAMT PSG nights from the REFERENCE side
(polysomnograph airflow + scoring), to understand what the nights look like and
design an experiment. STANDALONE: reads the pipeline's reference functions, edits
nothing, writes HTML + JSON OUTSIDE OneDrive (the repo blocks Python file writes).

Per subject it derives, over the whole night:
  * Respiration rate from AIRFLOW (FLOW) and, overlaid, from EFFORT (THORAX) — the
    key apnea signature: airflow stops while effort continues = obstructive; both
    stop = central.
  * Event bands: Obstructive_Apnea / Central_Apnea / Hypopnea / Multiple_Events
    (contiguous runs, not rows).
  * SpO2 with desaturation markers (>=3% dips), snore energy (SNORE envelope),
    and the sleep-stage hypnogram.
  * Summary stats: night/sleep length, time per stage, AHI, event counts, RR
    median/IQR, RR per stage, min/mean SpO2, ODI, snore burden.
Then a cross-subject summary.html (table + aggregate distributions + patterns).

The reference RR uses the project's OWN validated breath detector (analyze_reference
+ reference_average_rr), so it matches how the pipeline reads REMbo airflow.

Usage
-----
    # prototype on two subjects:
    py dreamt_night_report.py --subjects S002 S004
    # everyone + the cross-subject summary:
    py dreamt_night_report.py --summary
    # custom data root / output:
    py dreamt_night_report.py --psg-dir "<...>/DreamT/PSG_100hz" --out "<dir>"

Outputs land in %LOCALAPPDATA%/BBBRR_DREAMT_report by default (one <Sxxx>.html +
<Sxxx>.json each, plus summary.html with --summary).
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from respiration_rr.reference.airflow import analyze_reference
from respiration_rr.rr_average import reference_average_rr
from respiration_rr.compare.compare import reference_rr_series

DEFAULT_PSG_DIR = (r"C:\Users\RachelMizrahi\CardiacSense\shares - BBB Respiration Rate"
                   r"\Data\Exp3\DreamT\PSG_100hz")

# When set (via --inline-plotly), the whole Plotly library is embedded in each page so
# it opens with a double-click, offline, no sibling file or CDN needed.
_PLOTLY_JS = None


def _apply_inline(html):
    if _PLOTLY_JS:
        return html.replace('<script src="plotly.min.js"></script>',
                            "<script>" + _PLOTLY_JS + "</script>")
    return html

USECOLS = ["TIMESTAMP", "FLOW", "THORAX", "ABDOMEN", "SNORE", "SAO2", "HR",
           "Sleep_Stage", "Obstructive_Apnea", "Central_Apnea", "Hypopnea",
           "Multiple_Events"]
EVENT_COLS = ["Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events"]
EVENT_LABEL = {"Obstructive_Apnea": "Obstructive apnea", "Central_Apnea": "Central apnea",
               "Hypopnea": "Hypopnea", "Multiple_Events": "Multiple events"}
EVENT_COLOR = {"Obstructive_Apnea": "#dc2626", "Central_Apnea": "#7c3aed",
               "Hypopnea": "#f59e0b", "Multiple_Events": "#0891b2"}
STAGE_LEVEL = {"W": 4, "R": 3, "N1": 2, "N2": 1, "N3": 0}   # hypnogram y-levels
RR_BINS = [("slow <12", 0, 12), ("normal 12-20", 12, 20),
           ("elevated 20-30", 20, 30), ("fast 30+", 30, 99)]
DECIM_HZ = 25.0          # airflow/effort decimation for breath detection
COARSE_HZ = 1.0          # SpO2 / snore / HR trace rate


# ----------------------------------------------------------------- helpers
def _num(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(np.float64)


def _block_avg(x, factor):
    factor = int(factor)
    if factor <= 1:
        return np.asarray(x, np.float64)
    n = (len(x) // factor) * factor
    return np.asarray(x[:n], np.float64).reshape(-1, factor).mean(axis=1)


def _runs_seconds(flag, tt):
    """Contiguous runs where flag>0 -> list of (start_s, end_s) on the tt clock."""
    b = np.nan_to_num(_num(flag), nan=0.0) > 0
    if not b.any():
        return []
    edges = np.diff(b.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if b[0]:
        starts = [0] + starts
    if b[-1]:
        ends = ends + [len(b) - 1]
    return [(float(tt[s]), float(tt[min(e, len(tt) - 1)])) for s, e in zip(starts, ends)]


def _stage_epochs(stage_series, tt, fs, epoch=30.0):
    """Sample Sleep_Stage every `epoch` s (PSG scoring is per-epoch). -> (t0_list, stage_list)."""
    s = stage_series.to_numpy()
    step = int(round(epoch * fs))
    idx = np.arange(0, len(s), step)
    return [float(tt[i]) for i in idx], [str(s[i]) for i in idx]


def _rolling_max(x, win):
    """Causal-ish rolling max via pandas (baseline for desat detection)."""
    return pd.Series(x).rolling(int(win), min_periods=1, center=True).max().to_numpy()


def _desaturations(spo2_pct, t_coarse, drop=3.0, min_gap_s=10.0):
    """Desat = SpO2 dips >= `drop`% below a ~100 s rolling baseline. Returns marker times + ODI."""
    if spo2_pct.size < 5:
        return [], 0.0
    base = _rolling_max(spo2_pct, win=100)        # 1 Hz coarse grid -> ~100 s
    below = (base - spo2_pct) >= drop
    runs = _runs_seconds(below.astype(float), t_coarse)
    # one marker per run (its minimum)
    marks = []
    for a, b in runs:
        m = (t_coarse >= a) & (t_coarse <= b)
        if m.any():
            marks.append(float(t_coarse[m][np.argmin(spo2_pct[m])]))
    hrs = (t_coarse[-1] - t_coarse[0]) / 3600.0 if t_coarse.size else 0.0
    odi = (len(marks) / hrs) if hrs > 0 else 0.0
    return marks, odi


def _stage_at(times, ep_t, ep_s):
    """Nearest preceding stage label for each time."""
    ep_t = np.asarray(ep_t, float)
    out = []
    for t in np.asarray(times, float):
        i = np.searchsorted(ep_t, t, side="right") - 1
        out.append(ep_s[i] if 0 <= i < len(ep_s) else "?")
    return out


# ---------------------------------------------------------------- RR dynamics
def _segment_levels(t, y, tol=2.0, min_len_s=30.0):
    """Greedy piecewise-constant segmentation of a smoothed RR curve into 'regimes'
    (plateaus). A regime extends while the value stays within +-tol bpm of its running
    mean; shorter-than-min_len runs are merged into the previous regime. Returns a list
    of (t_start, t_end, level_bpm)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if y.size < 3:
        return []
    segs = []
    s = 0
    acc = [y[0]]
    for i in range(1, y.size):
        if abs(y[i] - np.mean(acc)) <= tol:
            acc.append(y[i])
        else:
            segs.append([t[s], t[i - 1], float(np.mean(acc))])
            s = i
            acc = [y[i]]
    segs.append([t[s], t[-1], float(np.mean(acc))])
    # merge too-short regimes into the neighbour closest in level
    out = []
    for seg in segs:
        if (seg[1] - seg[0]) < min_len_s and out:
            out[-1][1] = seg[1]                  # absorb into previous
        else:
            out.append(seg)
    return [tuple(s) for s in out]


def _dominant_cycle(t, y, lo_s=20.0, hi_s=120.0):
    """Dominant oscillation period of a smoothed RR curve (autocorrelation of the
    detrended signal) within [lo_s, hi_s]. Returns (period_s, strength 0..1).
    Catches periodic breathing / Cheyne-Stokes (rhythmic rise-fall of RR)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if y.size < 20:
        return float("nan"), 0.0
    fs = 1.0 / np.median(np.diff(t))              # grid rate (~0.25 Hz at 4 s step)
    yd = y - moving_mean(y, int(round(fs * hi_s)))  # detrend (remove slow drift)
    yd = yd - yd.mean()
    if np.allclose(yd, 0):
        return float("nan"), 0.0
    ac = np.correlate(yd, yd, "full")[yd.size - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(fs * lo_s), min(int(fs * hi_s), ac.size - 1)
    if hi <= lo + 1:
        return float("nan"), 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    return float(k / fs), float(ac[k])


def moving_mean(y, w):
    w = max(1, int(w))
    if w <= 1 or y.size < 3:
        return np.zeros_like(y)
    k = np.ones(w) / w
    return np.convolve(y, k, "same")


def _merge_spans(spans):
    spans = sorted(spans)
    out = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def event_locking(bt, br, evt_spans, win=30.0):
    """Are the big RR changes time-locked to scored apnea/hypopnea events?
    Compares the fraction of abnormal RR jumps (>10 bpm breath-to-breath) that fall
    within +-win s of an event, against the baseline fraction of time that is near an
    event. enrichment = how many times more likely a jump is near an event than chance
    (>1 => event-driven; ~1 => random). Also splits jumps into drops vs rises."""
    bt = np.asarray(bt, float); br = np.asarray(br, float)
    m = np.isfinite(bt) & np.isfinite(br)
    bt, br = bt[m], br[m]
    nan = {"near_time_frac": float("nan"), "jump_near_frac": float("nan"),
           "enrichment": float("nan"), "n_jumps": 0, "drop_near_frac": float("nan"),
           "rise_near_frac": float("nan")}
    if bt.size < 20 or not evt_spans:
        return nan
    exp = _merge_spans([(s - win, e + win) for s, e in evt_spans])
    starts = np.array([s for s, e in exp]); ends = np.array([e for s, e in exp])

    def near(ts):
        ts = np.asarray(ts, float)
        if ts.size == 0:
            return np.zeros(0, bool)
        i = np.searchsorted(starts, ts, "right") - 1
        ok = i >= 0
        out = np.zeros(ts.shape, bool)
        out[ok] = ts[ok] <= ends[np.clip(i[ok], 0, ends.size - 1)]
        return out

    d = np.diff(br)
    jt = bt[1:][np.abs(d) > 10]                 # times of abnormal jumps
    drops = bt[1:][d < -10]; rises = bt[1:][d > 10]
    near_all = float(near(bt).mean())
    return {"near_time_frac": round(near_all, 3),
            "jump_near_frac": round(float(near(jt).mean()), 3) if jt.size else float("nan"),
            "enrichment": round(float(near(jt).mean()) / near_all, 2) if (jt.size and near_all > 0) else float("nan"),
            "n_jumps": int(jt.size),
            "drop_near_frac": round(float(near(drops).mean()), 3) if drops.size else float("nan"),
            "rise_near_frac": round(float(near(rises).mean()), 3) if rises.size else float("nan")}


def _runs_bool(b):
    runs = []
    i, n = 0, b.size
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            runs.append((i, j - 1)); i = j
        else:
            i += 1
    return runs


def smoothed_excursions(at, ar, evt_spans, win=30.0, dev_thr=4.0, min_dur_s=10.0, abrupt_s=15.0):
    """On the SMOOTHED RR curve: find sustained departures from a 5-min baseline (RR
    going faster = above, slower = below). For each, is the onset ABRUPT (reaches its
    extreme within abrupt_s) or GRADUAL, and is it time-locked to a scored event
    (+-win s)? Answers: do rate changes happen in steps or suddenly, randomly or near
    apneas/hypopneas."""
    at = np.asarray(at, float); ar = np.asarray(ar, float)
    m = np.isfinite(at) & np.isfinite(ar)
    at, ar = at[m], ar[m]
    nan = {"n_exc": 0, "frac_fast": float("nan"), "frac_slow": float("nan"),
           "frac_abrupt": float("nan"), "frac_near": float("nan"),
           "near_fast": float("nan"), "near_slow": float("nan"), "onset_med_s": float("nan")}
    if ar.size < 20:
        return nan
    step = float(np.median(np.diff(at))) or 4.0
    w = max(3, int(round(300.0 / step)))                 # ~5 min baseline
    base = pd.Series(ar).rolling(w, min_periods=max(3, w // 4), center=True).median().to_numpy()
    dev = ar - base
    runs = [r for r in _runs_bool(np.abs(dev) > dev_thr) if (at[r[1]] - at[r[0]]) >= min_dur_s]
    if not runs:
        return nan
    exp = _merge_spans([(s - win, e + win) for s, e in evt_spans]) if evt_spans else []
    starts = np.array([s for s, e in exp]); ends = np.array([e for s, e in exp])

    def near1(t):
        if starts.size == 0:
            return False
        i = int(np.searchsorted(starts, t, "right")) - 1
        return bool(i >= 0 and t <= ends[min(i, ends.size - 1)])

    fast = slow = abrupt = near = nf = ns = 0
    onsets = []
    for i0, i1 in runs:
        seg = dev[i0:i1 + 1]; tseg = at[i0:i1 + 1]
        pk = int(np.argmax(np.abs(seg)))
        is_fast = seg[pk] > 0                            # above baseline = faster RR
        onset = float(tseg[pk] - tseg[0]); onsets.append(onset)
        isnear = near1(float(tseg[pk]))
        fast += is_fast; slow += (not is_fast); abrupt += (onset <= abrupt_s); near += isnear
        if is_fast:
            nf += isnear
        else:
            ns += isnear
    n = len(runs)
    return {"n_exc": n, "frac_fast": round(fast / n, 2), "frac_slow": round(slow / n, 2),
            "frac_abrupt": round(abrupt / n, 2), "frac_near": round(near / n, 2),
            "near_fast": round(nf / max(fast, 1), 2), "near_slow": round(ns / max(slow, 1), 2),
            "onset_med_s": round(float(np.median(onsets)), 1)}


def rr_dynamics(bt, br, at, ar):
    """Characterise how RR moves over the night: breath-to-breath jumps, step/plateau
    structure, and periodicity. bt/br = per-breath (s, bpm); at/ar = smoothed curve."""
    bt = np.asarray(bt, float); br = np.asarray(br, float)
    m = np.isfinite(bt) & np.isfinite(br)
    bt, br = bt[m], br[m]
    nan = {"bb_change_med": float("nan"), "jump_rate_hr": float("nan"),
           "n_regimes": 0, "regime_min_med": float("nan"), "step_med": float("nan"),
           "big_steps": 0, "big_steps_hr": float("nan"),
           "cycle_sec": float("nan"), "periodicity": 0.0, "pattern": "—"}
    if bt.size < 20:
        return nan
    dur_h = (bt[-1] - bt[0]) / 3600.0 or 1e-9
    dd = np.abs(np.diff(br))
    bb_change_med = float(np.median(dd))
    jump_rate_hr = float((dd > 10).sum() / dur_h)        # "abnormal jumps" (>10 bpm breath-to-breath) per hour

    at = np.asarray(at, float); ar = np.asarray(ar, float)
    segs = _segment_levels(at, ar, tol=2.0, min_len_s=30.0)
    n_regimes = len(segs)
    reg_lens = [(s[1] - s[0]) / 60.0 for s in segs]
    regime_min_med = float(np.median(reg_lens)) if reg_lens else float("nan")
    steps = [abs(segs[i + 1][2] - segs[i][2]) for i in range(len(segs) - 1)]
    step_med = float(np.median(steps)) if steps else 0.0
    big_steps = int(sum(s >= 5 for s in steps))          # discrete level shifts >= 5 bpm ("stairs")
    big_steps_hr = big_steps / dur_h
    cycle_sec, periodicity = _dominant_cycle(at, ar)

    # classification (Hebrew labels; see glossary)
    if periodicity >= 0.4 and np.isfinite(cycle_sec):
        pattern = "מחזורי"                 # periodic (rhythmic rise-fall, e.g. periodic breathing)
    elif big_steps_hr >= 3 and step_med >= 4:
        pattern = "מדרגות"                 # step-like (plateaus with discrete jumps)
    elif jump_rate_hr >= 12:
        pattern = "לא-יציב"                # irregular (many abnormal jumps)
    else:
        pattern = "יציב/הדרגתי"            # stable / gradual drift
    return {"bb_change_med": round(bb_change_med, 2), "jump_rate_hr": round(jump_rate_hr, 1),
            "n_regimes": n_regimes, "regime_min_med": round(regime_min_med, 2),
            "step_med": round(step_med, 2), "big_steps": big_steps,
            "big_steps_hr": round(big_steps_hr, 2), "cycle_sec": round(cycle_sec, 1) if np.isfinite(cycle_sec) else None,
            "periodicity": round(periodicity, 2), "pattern": pattern}


# ----------------------------------------------------------------- per subject
def process_subject(path, sid):
    print(f"[{sid}] reading {os.path.basename(path)} ...", flush=True)
    df = pd.read_csv(path, usecols=USECOLS)
    t = _num(df["TIMESTAMP"])
    keep = np.isfinite(t)
    df, t = df[keep].reset_index(drop=True), t[keep]
    tt = t - t[0]
    fs = float(1.0 / np.median(np.diff(t)))
    dur_h = (tt[-1] - tt[0]) / 3600.0
    print(f"[{sid}] {len(df):,} rows | {dur_h:.2f} h | fs~{fs:.1f} Hz | deriving RR ...", flush=True)

    # ---- RR from airflow (FLOW) and effort (THORAX), decimated ----
    dec = max(1, int(round(fs / DECIM_HZ)))
    fs_d = fs / dec
    t_d = _block_avg(tt, dec)

    def rr_curves(col):
        sig = _block_avg(_num(df[col]), dec)
        n = min(len(t_d), len(sig))
        ref = analyze_reference(t_d[:n], sig[:n], fs_d)
        bt, br = reference_rr_series(ref)                 # per-breath (center, rate)
        at, ar = reference_average_rr(ref)                # smoothed curve (same method)
        return ref, bt, br, at, ar

    ref_air, bt, br, at_air, ar_air = rr_curves("FLOW")
    _, _, _, at_eff, ar_eff = rr_curves("THORAX")

    # ---- RAW airflow + breath separators (for visual verification) ----
    wf = np.asarray(ref_air.raw, float)                # the RAW airflow signal (as requested)
    wf = wf - np.nanmean(wf)                            # center only (keep raw morphology)
    wstd = float(np.nanstd(wf)) or 1.0
    wf = wf / wstd                                      # scale to O(1) for a sane axis (shape unchanged)
    wt = np.asarray(ref_air.time, float)               # s from night start (same clock as everything)
    dispf = max(1, int(round(fs_d / 8.0)))             # ~8 Hz for display (breaths well resolved)
    wt_d, wf_d = wt[::dispf], wf[::dispf]
    onset = np.array([b.start for b in ref_air.breaths if np.isfinite(b.start)], float)  # breath boundaries
    onset_y = np.zeros(0)

    # ---- RR dynamics: jumps, step/plateau structure, periodicity ----
    dyn = rr_dynamics(bt, br, at_air, ar_air)

    # ---- coarse traces (SpO2 / snore / HR @ ~1 Hz) ----
    cf = max(1, int(round(fs / COARSE_HZ)))
    t_c = _block_avg(tt, cf)
    spo2 = _block_avg(_num(df["SAO2"]), cf) * 100.0       # fraction -> %
    spo2[spo2 < 50] = np.nan                              # sensor dropout (0 / invalid) -> ignore
    hr = _block_avg(_num(df["HR"]), cf)
    snore_raw = _num(df["SNORE"])
    n_sn = (len(snore_raw) // cf) * cf
    snore_e = np.sqrt((snore_raw[:n_sn].reshape(-1, cf) ** 2).mean(axis=1))   # 1 s RMS envelope
    snore_e = snore_e / (np.median(snore_e) + 1e-12)      # relative to baseline
    m = min(t_c.size, spo2.size, hr.size, snore_e.size)
    t_c, spo2, hr, snore_e = t_c[:m], spo2[:m], hr[:m], snore_e[:m]

    desat_t, odi = _desaturations(spo2, t_c)
    snore_pct = 100.0 * float((snore_e > 3.0).mean())     # % time snore energy > 3x baseline

    # ---- stages ----
    ep_t, ep_s = _stage_epochs(df["Sleep_Stage"], tt, fs)
    stage_codes = [STAGE_LEVEL.get(s, np.nan) for s in ep_s]
    stage_time = {}
    for s in ep_s:
        stage_time[s] = stage_time.get(s, 0.0) + 30.0
    tst_h = sum(v for k, v in stage_time.items() if k not in ("W", "P", "nan")) / 3600.0

    # ---- events ----
    events = {c: _runs_seconds(df[c], tt) for c in EVENT_COLS}
    n_ev = {c: len(events[c]) for c in EVENT_COLS}
    _evt_spans = [sp for c in EVENT_COLS for sp in events[c]]
    evt_lock = event_locking(bt, br, _evt_spans, win=30.0)
    exc = smoothed_excursions(at_air, ar_air, _evt_spans, win=30.0)
    ahi = (n_ev["Obstructive_Apnea"] + n_ev["Central_Apnea"] + n_ev["Hypopnea"]) / tst_h if tst_h > 0 else 0.0

    # ---- RR stats ----
    br_f = br[np.isfinite(br)] if br.size else np.zeros(0)
    rr_med = float(np.median(br_f)) if br_f.size else float("nan")
    rr_iqr = [float(np.percentile(br_f, 25)), float(np.percentile(br_f, 75))] if br_f.size else [float("nan")] * 2
    rr_bin_pct = {name: 100.0 * float(((br_f >= lo) & (br_f < hi)).mean()) if br_f.size else 0.0
                  for name, lo, hi in RR_BINS}
    rr_by_stage = {}
    if bt.size:
        bst = _stage_at(bt, ep_t, ep_s)
        for st in set(bst):
            vals = br[np.array(bst) == st]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                rr_by_stage[st] = float(np.median(vals))

    def dj(x):   # VALUE arrays -> plain lists, 3 decimals (lighter HTML)
        return [None if not np.isfinite(v) else round(float(v), 3) for v in x]

    def dh(x):   # TIME arrays (hours) -> 6 decimals; 3 would quantize to 3.6 s and flatten the waveform
        return [None if not np.isfinite(v) else round(float(v), 6) for v in x]

    data = {
        "sid": sid, "dur_h": round(dur_h, 2), "tst_h": round(tst_h, 2), "fs": round(fs, 1),
        "wave": {"t": dh(wt_d / 3600.0), "y": dj(wf_d),
                 "onset_t": dh(onset / 3600.0), "onset_y": dj(onset_y),
                 "n_breaths": int(onset.size)},
        "rr": {"air_t": dh(at_air / 3600.0), "air": dj(ar_air),
               "eff_t": dh(at_eff / 3600.0), "eff": dj(ar_eff),
               "breath_t": dh(bt / 3600.0), "breath": dj(br)},
        "spo2": {"t": dh(t_c / 3600.0), "v": dj(spo2), "desat_t": dh(np.array(desat_t) / 3600.0)},
        "snore": {"t": dh(t_c / 3600.0), "v": dj(snore_e)},
        "hr": {"t": dh(t_c / 3600.0), "v": dj(hr)},
        "hypno": {"t": [round(x / 3600.0, 4) for x in ep_t], "stage": ep_s,
                  "level": [None if (v is None or not np.isfinite(v)) else int(v) for v in stage_codes]},
        "events": {c: [[round(a / 3600.0, 4), round(b / 3600.0, 4)] for a, b in events[c]] for c in EVENT_COLS},
        "stats": {"ahi": round(ahi, 1), "n_events": n_ev, "rr_median": round(rr_med, 1),
                  "rr_std": round(float(np.std(br_f)), 1) if br_f.size else float("nan"),
                  "rr_iqr": [round(v, 1) for v in rr_iqr], "rr_bin_pct": {k: round(v, 1) for k, v in rr_bin_pct.items()},
                  "rr_by_stage": {k: round(v, 1) for k, v in rr_by_stage.items()},
                  "spo2_min": round(float(np.nanmin(spo2)), 1), "spo2_mean": round(float(np.nanmean(spo2)), 1),
                  "odi": round(odi, 1), "snore_pct": round(snore_pct, 1),
                  "stage_min": {k: round(v / 60.0, 1) for k, v in stage_time.items()},
                  "dyn": dyn, "evt": evt_lock, "exc": exc},
        "rr_series": {"bt": dh(bt / 3600.0), "br": dj(br),
                      "at": dh(at_air / 3600.0), "ar": dj(ar_air)},
        "event_spans": {c: [[round(s / 3600.0, 5), round(e / 3600.0, 5)] for s, e in events[c]] for c in EVENT_COLS},
    }
    print(f"[{sid}] AHI={ahi:.1f} | O/C/H={n_ev['Obstructive_Apnea']}/{n_ev['Central_Apnea']}/"
          f"{n_ev['Hypopnea']} | RR med {rr_med:.1f} [{rr_iqr[0]:.0f}-{rr_iqr[1]:.0f}] | "
          f"SpO2 min {data['stats']['spo2_min']} ODI {odi:.1f} | snore {snore_pct:.0f}%", flush=True)
    return data


# ----------------------------------------------------------------- HTML (per subject)
def _page_html(data):
    sid = data["sid"]
    s = data["stats"]
    cards = [
        ("משך לילה", f"{data['dur_h']} ש'"), ("זמן שינה", f"{data['tst_h']} ש'"),
        ("AHI", f"{s['ahi']}"), ("אפנאה חסימתית", f"{s['n_events']['Obstructive_Apnea']}"),
        ("אפנאה מרכזית", f"{s['n_events']['Central_Apnea']}"), ("היפופנאה", f"{s['n_events']['Hypopnea']}"),
        ("RR חציון", f"{s['rr_median']} [{s['rr_iqr'][0]}-{s['rr_iqr'][1]}]"),
        ("SpO2 מינ'", f"{s['spo2_min']}%"), ("ODI", f"{s['odi']}/ש'"), ("נחירה", f"{s['snore_pct']}%"),
    ]
    card_html = "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
                        for k, v in cards)
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DREAMT {sid} — סקירת לילה</title>
<script src="plotly.min.js"></script>
<script>if(typeof Plotly==="undefined"){{var s=document.createElement("script");
  s.src="https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js";document.head.appendChild(s);}}</script>
<style>
  :root {{ --bg:#ffffff; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#f8fafc; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
    --bg:#0b1220; --fg:#e5e7eb; --muted:#94a3b8; --line:#1e293b; --card:#111a2e; }} }}
  body {{ background:var(--bg); color:var(--fg); font-family:system-ui,Segoe UI,Arial,sans-serif;
         margin:0; padding:16px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); font-size:13px; margin-bottom:12px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px; margin-bottom:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 10px; }}
  .card .k {{ color:var(--muted); font-size:12px; }} .card .v {{ font-size:17px; font-weight:600; margin-top:2px; }}
  #plot {{ width:100%; height:880px; }}
  .note {{ color:var(--muted); font-size:12px; margin-top:8px; line-height:1.5; }}
</style></head>
<body>
  <h1>DREAMT · נבדק {sid}</h1>
  <div class="sub">סקירת לילה מהרפרנס (PSG airflow) · RR מאוויר מול מאמץ · אירועים · סטורציה · נחירות · שלבי שינה</div>
  <div class="cards">{card_html}</div>
  <div id="plot"></div>
  <div class="note">פאנל עליון = אות זרימת האוויר <b>הגולמי</b> עם <b>פסים אדומים</b> שמפרידים בין הנשימות שהגלאי מצא — זַמְני (זום) כדי לאמת שהחלוקה לנשימות נכונה.
  RR כחול = זרימת אוויר (FLOW), כתום מקווקו = מאמץ חזה (THORAX); פער ביניהם (אוויר נעצר בעוד המאמץ נמשך) = חתימה חסימתית.
  פסים אנכיים = אירועים (מתיוג DREAMT). נקודות על SpO2 = ירידות סטורציה (≥3%). גלול/זום עם סרגל הזמן למטה.</div>
<script>
const D = {json.dumps(data)};
const EV = {{"Obstructive_Apnea":"#dc2626","Central_Apnea":"#7c3aed","Hypopnea":"#f59e0b","Multiple_Events":"#0891b2"}};
const EVN = {{"Obstructive_Apnea":"אפנאה חסימתית","Central_Apnea":"אפנאה מרכזית","Hypopnea":"היפופנאה","Multiple_Events":"אירועים מרובים"}};
function col(a,i){{return a.map(r=>r[i]);}}
// breath separators: a single trace of vertical line segments at each breath boundary
const sepX=[], sepY=[]; for(const t of D.wave.onset_t){{ sepX.push(t,t,null); sepY.push(-100,100,null); }}
// desat marker y = the SpO2 value nearest each desaturation time (computed up-front, no fragile index)
const dy = []; for(const dt of D.spo2.desat_t){{ let bi=0,bd=1e9; for(let i=0;i<D.spo2.t.length;i++){{let d=Math.abs(D.spo2.t[i]-dt); if(d<bd){{bd=d;bi=i;}}}} dy.push(D.spo2.v[bi]); }}
const traces = [
  {{x:D.wave.t, y:D.wave.y, name:"זרימת אוויר (גולמי)", mode:"lines", line:{{color:"#0f766e",width:0.9}}, type:"scattergl", yaxis:"y5", xaxis:"x"}},
  {{x:sepX, y:sepY, name:"הפרדת נשימות ("+D.wave.n_breaths+")", mode:"lines",
     line:{{color:"#dc2626",width:0.7}}, type:"scattergl", yaxis:"y5", xaxis:"x", hoverinfo:"skip"}},
  {{x:D.rr.breath_t, y:D.rr.breath, name:"BBBRR (פר-נשימה)", mode:"markers",
     marker:{{color:"#93c5fd",size:3,opacity:0.55}}, type:"scattergl", yaxis:"y", xaxis:"x"}},
  {{x:D.rr.air_t, y:D.rr.air, name:"RR אוויר (ממוצע)", mode:"lines", line:{{color:"#2563eb",width:1.6}}, yaxis:"y", xaxis:"x"}},
  {{x:D.rr.eff_t, y:D.rr.eff, name:"RR מאמץ", mode:"lines", line:{{color:"#f59e0b",width:1.3,dash:"dot"}}, yaxis:"y", xaxis:"x"}},
  {{x:D.spo2.t, y:D.spo2.v, name:"SpO2 %", mode:"lines", line:{{color:"#0ea5e9",width:1.2}}, yaxis:"y2", xaxis:"x"}},
  {{x:D.spo2.desat_t, y:dy, name:"דה-סטורציה", mode:"markers",
     marker:{{color:"#dc2626",size:6,symbol:"triangle-down"}}, yaxis:"y2", xaxis:"x", hoverinfo:"x"}},
  {{x:D.snore.t, y:D.snore.v, name:"נחירה (יחסי)", mode:"lines", line:{{color:"#16a34a",width:1}}, yaxis:"y3", xaxis:"x"}},
  {{x:D.hypno.t, y:D.hypno.level, name:"שלב שינה", mode:"lines", line:{{color:"#64748b",width:1.4,shape:"hv"}}, yaxis:"y4", xaxis:"x"}},
];
// event shapes (on the RR row = y domain of y axis)
const shapes = [];
for(const c of Object.keys(D.events)){{ for(const [a,b] of D.events[c]){{
  shapes.push({{type:"rect", xref:"x", yref:"paper", x0:a, x1:b, y0:0.63, y1:1.0,
    fillcolor:EV[c], opacity:0.22, line:{{width:0}}, layer:"below"}});
}}}}
// legend proxies for events
for(const c of Object.keys(D.events)){{ if(D.events[c].length) traces.push(
  {{x:[null], y:[null], name:EVN[c]+" ("+D.events[c].length+")", mode:"markers",
    marker:{{color:EV[c],size:10,symbol:"square"}}, yaxis:"y", xaxis:"x"}}); }}
const layout = {{
  paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
  font:{{color:getComputedStyle(document.body).color}},
  margin:{{l:55,r:20,t:10,b:40}}, hovermode:"x unified", shapes:shapes,
  legend:{{orientation:"h", y:1.06}},
  grid:{{rows:5, columns:1, pattern:"independent", roworder:"top to bottom"}},
  xaxis:{{domain:[0,1], anchor:"y4", rangeslider:{{visible:true, thickness:0.05}}, title:"זמן (שעות)"}},
  yaxis5:{{domain:[0.83,1.0], title:"אוויר", zeroline:false, autorange:true}},
  yaxis:{{domain:[0.63,0.80], title:"RR (bpm)", rangemode:"tozero"}},
  yaxis2:{{domain:[0.43,0.60], title:"SpO2 %"}},
  yaxis3:{{domain:[0.24,0.40], title:"נחירה"}},
  yaxis4:{{domain:[0.0,0.18], title:"שלב", tickvals:[0,1,2,3,4], ticktext:["N3","N2","N1","REM","Wake"]}},
}};
function draw(){{
  var gd=document.getElementById("plot");
  Plotly.newPlot(gd, traces, layout, {{responsive:true, displaylogo:false}}).then(function(){{
    // auto-fit the raw-airflow panel's y to whatever time window is shown (pan/zoom/slider)
    gd.on("plotly_relayout", function(ev){{
      if(!("xaxis.range[0]" in ev || "xaxis.range" in ev || "xaxis.autorange" in ev)) return;
      var r=gd.layout.xaxis.range; if(!r) return;
      var lo=r[0], hi=r[1], mn=Infinity, mx=-Infinity;
      for(var i=0;i<D.wave.t.length;i++){{ var t=D.wave.t[i];
        if(t>=lo && t<=hi){{ var v=D.wave.y[i]; if(v!=null){{ if(v<mn)mn=v; if(v>mx)mx=v; }} }} }}
      if(mn<mx){{ var pad=0.12*(mx-mn)||1; Plotly.relayout(gd,{{"yaxis5.range":[mn-pad,mx+pad]}}); }}
    }});
  }});
}}
(function w(){{ if(typeof Plotly!=="undefined"){{draw();}} else {{setTimeout(w,100);}} }})();
</script>
</body></html>"""


def write_subject(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{data['sid']}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)
    html = os.path.join(out_dir, f"{data['sid']}.html")
    with open(html, "w", encoding="utf-8") as f:
        f.write(_apply_inline(_page_html(data)))
    return html


# ----------------------------------------------------------------- summary page
def write_summary(out_dir):
    js = sorted(glob.glob(os.path.join(out_dir, "S*.json")))
    rows = [json.load(open(p, encoding="utf-8")) for p in js]
    if not rows:
        print("no per-subject JSON found; run subjects first."); return None
    table = []
    for d in rows:
        s = d["stats"]
        table.append({"sid": d["sid"], "dur_h": d["dur_h"], "tst_h": d["tst_h"], "ahi": s["ahi"],
                      "O": s["n_events"]["Obstructive_Apnea"], "C": s["n_events"]["Central_Apnea"],
                      "H": s["n_events"]["Hypopnea"], "rr_med": s["rr_median"], "rr_iqr": s["rr_iqr"],
                      "spo2_min": s["spo2_min"], "odi": s["odi"], "snore": s["snore_pct"],
                      "rr_bin": s["rr_bin_pct"], "rr_stage": s["rr_by_stage"]})
    payload = json.dumps(table)
    trh = "".join(
        f"<tr><td>{r['sid']}</td><td>{r['dur_h']}</td><td>{r['tst_h']}</td><td>{r['ahi']}</td>"
        f"<td>{r['O']}</td><td>{r['C']}</td><td>{r['H']}</td><td>{r['rr_med']} "
        f"[{r['rr_iqr'][0]}-{r['rr_iqr'][1]}]</td><td>{r['spo2_min']}</td><td>{r['odi']}</td>"
        f"<td>{r['snore']}</td></tr>" for r in table)
    html = f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DREAMT — סיכום כל הנבדקים</title>
<script src="plotly.min.js"></script>
<script>if(typeof Plotly==="undefined"){{var s=document.createElement("script");
  s.src="https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js";document.head.appendChild(s);}}</script>
<style>
 :root {{ --bg:#fff; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#f8fafc; }}
 @media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0b1220;--fg:#e5e7eb;--muted:#94a3b8;--line:#1e293b;--card:#111a2e;}}}}
 body{{background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Arial;margin:0;padding:16px;}}
 h1{{font-size:20px;margin:0 0 12px;}} h2{{font-size:15px;margin:22px 0 8px;}}
 table{{border-collapse:collapse;width:100%;font-size:13px;}} th,td{{border:1px solid var(--line);padding:5px 8px;text-align:center;}}
 th{{background:var(--card);position:sticky;top:0;}} .plot{{width:100%;height:340px;margin-top:6px;}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}} @media(max-width:800px){{.grid{{grid-template-columns:1fr;}}}}
</style></head><body>
<h1>DREAMT · סיכום {len(table)} נבדקים (רפרנס PSG)</h1>
<table><thead><tr><th>נבדק</th><th>לילה(ש')</th><th>שינה(ש')</th><th>AHI</th><th>חסימתי</th><th>מרכזי</th>
<th>היפופ'</th><th>RR חציון[IQR]</th><th>SpO2 מינ'</th><th>ODI</th><th>נחירה%</th></tr></thead><tbody>{trh}</tbody></table>
<h2>התפלגויות מצרפיות</h2>
<div class="grid">
 <div id="ahi" class="plot"></div><div id="rr" class="plot"></div>
 <div id="stage" class="plot"></div><div id="spo2" class="plot"></div>
</div>
<script>
const T = {payload};
const dark = matchMedia("(prefers-color-scheme:dark)").matches;
const base = {{paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{{color:dark?"#e5e7eb":"#0f172a"}},margin:{{l:45,r:12,t:28,b:36}}}};
function draw(){{
Plotly.newPlot("ahi",[{{x:T.map(r=>r.ahi),type:"histogram",marker:{{color:"#2563eb"}}}}],
  Object.assign({{title:"AHI (אירועים/שעת שינה)"}},base),{{displaylogo:false}});
Plotly.newPlot("rr",[{{x:T.map(r=>r.rr_med),type:"histogram",marker:{{color:"#16a34a"}}}}],
  Object.assign({{title:"RR חציון לנבדק (bpm)"}},base),{{displaylogo:false}});
// RR time-in-bin stacked
const bins=["slow <12","normal 12-20","elevated 20-30","fast 30+"];
const cols=["#60a5fa","#34d399","#fbbf24","#f87171"];
Plotly.newPlot("stage", bins.map((b,i)=>({{x:T.map(r=>r.sid),y:T.map(r=>r.rr_bin[b]||0),name:b,type:"bar",marker:{{color:cols[i]}}}})),
  Object.assign({{title:"% זמן בכל טווח RR",barmode:"stack"}},base),{{displaylogo:false}});
Plotly.newPlot("spo2",[{{x:T.map(r=>r.spo2_min),y:T.map(r=>r.ahi),text:T.map(r=>r.sid),mode:"markers",
  type:"scatter",marker:{{color:"#dc2626",size:8}}}}],
  Object.assign({{title:"AHI מול SpO2 מינימלי",xaxis:{{title:"SpO2 מינ' %"}},yaxis:{{title:"AHI"}}}},base),{{displaylogo:false}});
}}
(function w(){{ if(typeof Plotly!=="undefined"){{draw();}} else {{setTimeout(w,100);}} }})();
</script>
</body></html>"""
    p = os.path.join(out_dir, "summary.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_apply_inline(html))
    print(f"wrote summary -> {p}  ({len(table)} subjects)")
    return p


# ----------------------------------------------------------------- plotly vendoring
def _ensure_plotly(out_dir):
    """Download plotly.min.js next to the pages (self-contained, offline-capable)."""
    p = os.path.join(out_dir, "plotly.min.js")
    if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
        return p
    import urllib.request
    url = "https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js"
    try:
        urllib.request.urlretrieve(url, p)
        print(f"vendored plotly.min.js ({os.path.getsize(p)//1024} KB)")
    except Exception as e:
        print(f"[warn] could not download plotly.min.js ({e}); pages will fall back to the CDN (needs internet).")
    return p


def rebuild_html(out_dir):
    """Regenerate every <Sxxx>.html from its cached JSON (no CSV re-read)."""
    n = 0
    for p in sorted(glob.glob(os.path.join(out_dir, "S*.json"))):
        data = json.load(open(p, encoding="utf-8"))
        write_subject(data, out_dir)
        n += 1
    print(f"rebuilt {n} subject page(s) from JSON")
    return n


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="DREAMT PSG night characterization report")
    ap.add_argument("--psg-dir", default=DEFAULT_PSG_DIR)
    ap.add_argument("--subjects", nargs="+", default=None, help="e.g. S002 S004 (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N subjects")
    ap.add_argument("--out", default=None, help="output dir (default %LOCALAPPDATA%/BBBRR_DREAMT_report)")
    ap.add_argument("--summary", action="store_true", help="also build summary.html across all JSONs")
    ap.add_argument("--summary-only", action="store_true", help="only rebuild summary.html from existing JSONs")
    ap.add_argument("--rebuild-html", action="store_true", help="regenerate pages from cached JSON (no CSV re-read)")
    ap.add_argument("--inline-plotly", action="store_true",
                    help="embed Plotly inside each HTML -> one self-contained file (opens on double-click, offline)")
    ap.add_argument("--no-pages", action="store_true",
                    help="stats only: save a light <Sxxx>.json (metrics, no waveform/HTML) -> fast cross-subject batch")
    ap.add_argument("--skip-existing", action="store_true",
                    help="with --no-pages: skip subjects whose JSON already has the current metrics (no CSV re-read)")
    args = ap.parse_args(argv)

    out = args.out or os.path.join(os.environ.get("LOCALAPPDATA") or ".", "BBBRR_DREAMT_report")
    os.makedirs(out, exist_ok=True)
    plotly_path = _ensure_plotly(out)
    if args.inline_plotly:
        global _PLOTLY_JS
        try:
            _PLOTLY_JS = open(plotly_path, encoding="utf-8").read()
            print(f"inlining Plotly ({len(_PLOTLY_JS)//1024} KB) into each page")
        except Exception as e:
            print(f"[warn] inline-plotly requested but could not read {plotly_path}: {e}")

    if args.rebuild_html:
        rebuild_html(out)
        if args.summary:
            write_summary(out)
        return
    if args.summary_only:
        p = write_summary(out)
        if p:
            print(f"\nOpen: {p}")
        return

    files = {re.match(r"(S\d+)", os.path.basename(f)).group(1): f
             for f in sorted(glob.glob(os.path.join(args.psg_dir, "S*_PSG_df.csv")))}
    sids = args.subjects or list(files)
    if args.limit:
        sids = sids[:args.limit]

    done = []
    for sid in sids:
        if sid not in files:
            print(f"[{sid}] not found in {args.psg_dir}"); continue
        if args.no_pages and args.skip_existing:
            jp = os.path.join(out, f"{sid}.json")
            try:
                if "exc" in json.load(open(jp, encoding="utf-8")).get("stats", {}):
                    print(f"[{sid}] skip (cached)", flush=True); done.append(jp); continue
            except Exception:
                pass
        try:
            print(f"[{sid}] processing ...", flush=True)
            data = process_subject(files[sid], sid)
            if args.no_pages:                       # light JSON: metrics + RR series (for summary + dynamics)
                p = os.path.join(out, f"{sid}.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({k: data[k] for k in ("sid", "dur_h", "tst_h", "stats", "rr_series", "event_spans")}, f)
                done.append(p)
            else:
                done.append(write_subject(data, out))
        except Exception as e:
            import traceback
            print(f"[{sid}] ERROR: {e}"); traceback.print_exc()

    print(f"\nwrote {len(done)} subject page(s) -> {out}")
    for h in done:
        print("  " + h)
    if args.summary and done:
        write_summary(out)


if __name__ == "__main__":
    main()
