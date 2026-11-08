#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Noise / red-zone detection for the BBBRR pipeline (Exp3 / watch-13, where the built-in
movement gating is not active). A red zone is a time span we DO NOT analyse — not for
categorization, not for BBBRR, not for the average. We never interpolate a value across a red
zone (it is a hole, not a gap to bridge). Errors are later computed on clean spans only, and we
report what % of time fell on noise.

A segment is red if EITHER side (REMbo reference OR watch) is noisy there: watch noise -> we can't
see respiration; reference noise -> no trustworthy reference to score against. All zones live on the
REMbo clock (watch zones are shifted by the sync offset).

Sources (each tagged so the viewer can show them apart):
  ref_existing  : whatever the existing reference gating already flagged (ReferenceResult.merged_noise)
  ref_jump      : REMbo breath-to-breath RR jump > JUMP_BPM (implausible -> a mis-detected breath)
  ref_apnea     : REMbo nasal-airflow respiration-envelope COLLAPSE (breath-hold): band-passed
                  envelope stays below REF_K_COLLAPSE * rolling-median baseline for >= REF_APNEA_MIN_SEC.
  watch_move    : whatever the existing watch movement gating flagged (WatchSignals.move_regions)
  watch_apnea   : watch Artifact respiration-envelope COLLAPSE (apnea / lost contact): the pre-AGC
                  band-passed envelope stays below K_COLLAPSE * global-median for >= APNEA_MIN_SEC.
"""
import glob, json, os, re
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import median_filter

# ---- tunable detector parameters (defaults agreed on real data) --------------------------------
JUMP_BPM = 25.0            # ref: consecutive-breath RR jump above this -> noisy
REF_K_COLLAPSE = 0.35      # ref apnea: airflow envelope below this * rolling baseline = breath-hold
REF_APNEA_MIN_SEC = 8.0    # ref apnea: airflow collapse must last at least this long
REF_BASE_SEC = 60.0        # ref apnea: rolling-median baseline window (tracks slow cannula gain drift)
K_COLLAPSE = 0.30          # watch apnea: envelope below this * global median = collapsed
APNEA_MIN_SEC = 10.0       # watch apnea: collapse must last at least this long
ENV_SMOOTH_SEC = 2.0       # watch apnea: envelope smoothing (de-ripple only; long windows smear a
                           # real collapse below the min-duration and hide it)
APNEA_JOIN_GAP = 2.0       # watch apnea: bridge sub-threshold runs split by < this (channel jitter)
BW_LOW_HZ, BW_HIGH_HZ = 0.10, 0.90


# ---- interval helpers --------------------------------------------------------------------------
def _as_intervals(regions):
    """Coerce an existing-gate region list into [(start, end), ...] floats."""
    out = []
    for r in (regions or []):
        if isinstance(r, dict):
            s = r.get("start", r.get("begin", r.get("t0")))
            e = r.get("end", r.get("stop", r.get("t1")))
        else:
            s, e = r[0], r[1]
        if s is not None and e is not None and np.isfinite(s) and np.isfinite(e) and e > s:
            out.append((float(s), float(e)))
    return out


def merge_intervals(intervals, join_gap=0.0):
    """Sort + merge overlapping/adjacent intervals (optionally joining gaps <= join_gap)."""
    iv = sorted((s, e) for s, e in intervals if e > s)
    if not iv:
        return []
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1] + join_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _mask_to_intervals(mask, t, min_sec):
    """Contiguous True runs in `mask` (aligned to times `t`) lasting >= min_sec -> [(s, e), ...]."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size]
    out = []
    for s, e in zip(starts, ends):
        if t[e - 1] - t[s] >= min_sec:
            out.append((float(t[s]), float(t[e - 1])))
    return out


def total_seconds(intervals):
    return float(sum(e - s for s, e in intervals))


# ---- detectors ---------------------------------------------------------------------------------
def reference_jump_zones(ref, jump_bpm=JUMP_BPM):
    """REMbo breaths whose RR jumps by more than jump_bpm from the previous breath -> noisy span."""
    br = list(getattr(ref, "breaths", []))
    out = []
    for a, b in zip(br[:-1], br[1:]):
        if np.isfinite(a.rate) and np.isfinite(b.rate) and abs(b.rate - a.rate) > jump_bpm:
            out.append((float(a.start), float(b.end)))      # cover both breaths involved
    return merge_intervals(out)


OPTICAL = ("Green", "Red", "IR", "Yellow")


def _resp_envelope(x, fs, smooth_sec=ENV_SMOOTH_SEC):
    """Pre-AGC band-passed respiration envelope (smoothed) + its global median."""
    ny = fs / 2.0
    xbw = sosfiltfilt(butter(2, [BW_LOW_HZ / ny, BW_HIGH_HZ / ny], btype="band", output="sos"),
                      np.asarray(x, float))
    env = np.abs(hilbert(xbw))
    w = max(1, int(round(smooth_sec * fs)))
    env_s = np.convolve(env, np.ones(w) / w, mode="same")
    return env_s, float(np.median(env_s))


def apnea_zones(sig, channel="Artifact", k=K_COLLAPSE, min_sec=APNEA_MIN_SEC,
                smooth_sec=ENV_SMOOTH_SEC, corroborate=True, return_envelope=False):
    """Watch respiration-envelope collapse (apnea / lost contact) on the PRE-AGC band-passed signal.
    AGC would renormalise a collapsed span back up and hide it, so we work on the raw envelope.

    MULTI-CHANNEL: a real cessation means NO channel sees breathing. We flag a span only where the
    Artifact envelope collapses (< k*median) AND every available optical channel ALSO collapses.
    This rejects the common false positive where the Artifact is simply weak at fast rates while the
    optical channels (and the reference) still show clear breathing."""
    t = np.asarray(sig.time, float); fs = float(sig.fs)
    env_a, med_a = _resp_envelope(sig.channels[channel], fs, smooth_sec)
    mask = env_a < (k * med_a) if med_a > 0 else np.zeros(env_a.size, bool)
    if corroborate:
        for ch in OPTICAL:
            if ch in sig.channels and ch != channel:
                env_c, med_c = _resp_envelope(sig.channels[ch], fs, smooth_sec)
                if med_c > 0:
                    mask &= env_c < (k * med_c)          # require this channel collapsed too
    raw = merge_intervals(_mask_to_intervals(mask, t, 0.0), join_gap=APNEA_JOIN_GAP)  # bridge jitter gaps
    zones = [(s, e) for s, e in raw if e - s >= min_sec]
    if return_envelope:
        return zones, t, env_a, med_a
    return zones


def reference_apnea_zones(ref, k=REF_K_COLLAPSE, min_sec=REF_APNEA_MIN_SEC,
                          base_sec=REF_BASE_SEC, smooth_sec=ENV_SMOOTH_SEC, join_gap=APNEA_JOIN_GAP):
    """REMbo nasal-airflow respiration-envelope COLLAPSE (breath-hold / apnea). During a hold the
    airflow flattens: the band-passed envelope drops far below its rolling baseline. A rolling-median
    baseline (not a single global median) tracks slow gain drift in the nasal cannula.

    Validated on the hand-labelled Exp3 apneas (airflow clock == annotation clock, no sync offset):
    env < 0.35 * baseline sustained >= 8 s catches 17/18 holds with ~10 s total false positive across
    the 13 annotated recordings (apnea env/baseline median 0.21 vs 1.03 when breathing)."""
    x = np.asarray(getattr(ref, "filtered", getattr(ref, "raw", [])), float)
    t = np.asarray(getattr(ref, "time", []), float)
    if x.size < 10 or t.size != x.size:
        return []
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return []
    fs = 1.0 / float(dt)
    env, _ = _resp_envelope(x, fs, smooth_sec)
    base = median_filter(env, size=max(1, int(round(base_sec * fs))), mode="nearest")
    mask = env < (k * base)
    raw = merge_intervals(_mask_to_intervals(mask, t, 0.0), join_gap=join_gap)
    return [(s, e) for s, e in raw if e - s >= min_sec]


def reference_noise_zones(ref, jump_bpm=JUMP_BPM):
    """All REMbo-side red zones, tagged. Returns {tag: [(s, e), ...]} on the REMbo clock."""
    return {
        "ref_existing": merge_intervals(_as_intervals(getattr(ref, "merged_noise", []))),
        "ref_jump": reference_jump_zones(ref, jump_bpm),
        "ref_apnea": reference_apnea_zones(ref),
    }


def watch_noise_zones(sig, **apnea_kw):
    """All watch-side red zones, tagged. Returns {tag: [(s, e), ...]} on the WATCH clock."""
    return {
        "watch_move": merge_intervals(_as_intervals(getattr(sig, "move_regions", []))),
        "watch_apnea": apnea_zones(sig, **apnea_kw),
    }


def red_zones(ref, sig, offset, jump_bpm=JUMP_BPM, **apnea_kw):
    """Unify both sides onto the REMbo clock (watch zones shifted by +offset).
    Returns (by_source, merged, stats):
      by_source : {tag: [(s, e), ...]} on REMbo clock (for display)
      merged    : [(s, e), ...] union of ALL sources (what the pipeline excludes)
      stats     : {'noise_sec', 'span_sec', 'noise_pct', 'per_source_sec': {...}}"""
    by_source = {}
    by_source.update(reference_noise_zones(ref, jump_bpm))
    for tag, zs in watch_noise_zones(sig, **apnea_kw).items():
        by_source[tag] = [(s + offset, e + offset) for s, e in zs]      # watch clock -> REMbo clock
    all_iv = [iv for zs in by_source.values() for iv in zs]
    merged = merge_intervals(all_iv)
    t = np.asarray(ref.time, float)
    span = float(t[-1] - t[0]) if t.size else 0.0
    noise = total_seconds(merged)
    stats = {"noise_sec": noise, "span_sec": span,
             "noise_pct": (100.0 * noise / span) if span > 0 else float("nan"),
             "per_source_sec": {tag: total_seconds(zs) for tag, zs in by_source.items()}}
    return by_source, merged, stats


def manual_zones(rid, ann_dir):
    """Load hand annotations for `rid` from <ann_dir>/annotations_<rid with / -> _>.json.
    Returns (merged, by_type): merged = union [(s, e)] on the REMbo clock; by_type = {type: [(s,e)...]}.
    Missing file -> ([], {}). Noise AND apnea regions both count as red (excluded)."""
    base = "annotations_%s" % rid.replace("/", "_")
    # exact file, or a browser re-download "base (N).json" -> take the most recently modified
    cands = [c for c in glob.glob(os.path.join(ann_dir, base + "*.json"))
             if re.fullmatch(re.escape(base) + r"(?: \(\d+\))?\.json", os.path.basename(c))]
    if not cands:
        return [], {}
    fn = max(cands, key=os.path.getmtime)
    with open(fn, encoding="utf-8") as fh:
        d = json.load(fh)
    by_type = {}
    for r in d.get("regions", []):
        s, e = float(r["t0"]), float(r["t1"])
        if e > s:
            by_type.setdefault(r.get("type", "noise"), []).append((s, e))
    merged = merge_intervals([iv for v in by_type.values() for iv in v])
    return merged, {k: merge_intervals(v) for k, v in by_type.items()}


def in_red(times, merged):
    """Boolean mask: which of `times` (REMbo clock) fall inside any merged red zone."""
    times = np.asarray(times, float)
    bad = np.zeros(times.shape, bool)
    for s, e in merged:
        bad |= (times >= s) & (times <= e)
    return bad
