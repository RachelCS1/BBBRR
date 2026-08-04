#!/usr/bin/env python
"""
Synthetic breath-by-breath RR test-data generator.

Produces MATCHED pairs of files that breathe on the *same* timeline, so the
ground-truth breath-by-breath RR is identical in the watch signal and the
reference — exactly what you need to validate the RR algorithm before real
recordings exist.

For each breathing scenario it writes a folder containing:

  monitor_watch.csv   Monitor-mode watch CSV (read by read_monitor_csv / read_watch_auto)
                      header: PPG,Artifact,RED SIG,IR,XL-X,XL-Y,XL-Z  (no timestamp, 64 Hz)
                      Realistic cardiac pulse train (systolic peak + dicrotic notch)
                      modulated by respiration through ALL FOUR parameters at once:
                        - RSA  : heart-rate varies with the breath
                        - RIIV : pulse amplitude varies with the breath
                        - AUC  : pulse area varies with the breath (follows amplitude)
                        - BW   : baseline wander at the respiration frequency
                      Accelerometer is near-flat -> no movement/noise regions (clean).

  reference.edf       REMbo-style EDF (read by EDFReader): a "Nasal P" nasal-pressure
                      airflow channel breathing the same pattern + an "Activity" channel.

  ground_truth_rr.csv Per-breath truth: breath_start_s, rr_bpm (=60/dt) + the target RR(t).

Scenarios (all within 6-40 bpm, 5 min):
  stepped   normal->slow->fast->very-slow->fast segments
  constant  steady 15 bpm
  ramp      6 -> 40 bpm linear sweep

Each scenario is written CLEAN plus five moderate NOISE variants, all sharing the
same breathing timeline (so ground_truth_rr is identical across variants):
  white     elevated Gaussian sensor noise
  mains     50 Hz power-line + harmonics (aliases to ~14 Hz at 64 Hz Fs)
  spikes    sharp impulse glitches, random sign, on the optical channels
  motion    3 artifact windows: baseline excursion + burst on PPG, a CORRELATED
            accelerometer oscillation, and an EDF Activity spike (both gates fire)
  combined  white + mains + spikes + motion together, moderate

Noise is added ON TOP of the clean signal; the ground truth is never touched, so
you can measure how much RR the algorithm still recovers under each condition.

Run:  py Data/Synth/generate_synth.py
"""

import os
import struct
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Global recording parameters
# ----------------------------------------------------------------------------
T          = 300.0        # seconds (5 min)
FS_WATCH   = 64.0         # monitor CSV declared rate (settings.monitor_row_fs)
FS_REF     = 25.0         # EDF airflow rate (samples per 1-s record)
FS_ACT     = 1.0          # EDF activity rate
FS_FINE    = 512.0        # fine grid for phase integration
HR0        = 72.0         # baseline heart rate (bpm)
RSA_BPM    = 4.0          # heart-rate swing with respiration (RSA depth, bpm)
RIIV_DEPTH = 0.08         # pulse-amplitude modulation depth (fraction)
BW_FRAC    = 0.18         # baseline-wander amplitude as fraction of pulse AC
                          # (kept modest: a large baseline leaks into the per-beat
                          #  amplitude series and corrupts the RIIV estimator)
SEED       = 12345

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Breathing-rate profiles  f_resp(t) in Hz  (RR_bpm / 60), all within 6-40 bpm
# ----------------------------------------------------------------------------
def rr_profile(name, t):
    """Return instantaneous respiration rate in bpm at times t for a scenario."""
    if name == "constant":
        return np.full_like(t, 15.0)

    if name == "ramp":
        return 6.0 + (40.0 - 6.0) * (t / T)

    if name == "stepped":
        segs = [(0, 60, 15.0), (60, 120, 6.0), (120, 180, 30.0),
                (180, 240, 10.0), (240, T, 40.0)]
        rr = np.empty_like(t)
        for a, b, v in segs:
            rr[(t >= a) & (t < b)] = v
        rr[t >= T] = segs[-1][2]
        return rr

    raise ValueError(name)


# ----------------------------------------------------------------------------
# Respiration phase + ground-truth breath starts
# ----------------------------------------------------------------------------
def respiration_phase(name):
    """Integrate f_resp to a continuous phase phi(t) on the fine grid.

    Returns (t_fine, phi) with phi(0)=0; a breath *starts* each time phi crosses
    a multiple of 2*pi.  RR is encoded exactly, so 60/dt between crossings is the
    true breath-by-breath RR.
    """
    t = np.arange(0.0, T, 1.0 / FS_FINE)
    f = rr_profile(name, t) / 60.0                      # Hz
    phi = 2.0 * np.pi * np.concatenate(([0.0], np.cumsum((f[:-1] + f[1:]) * 0.5) / FS_FINE))
    return t, phi


def breath_starts(t, phi):
    """Times where phi crosses 2*pi*k upward -> breath-start instants."""
    starts = []
    k = 1
    target = 2.0 * np.pi * k
    for i in range(1, phi.size):
        while phi[i] >= target:
            # linear interpolation to the exact crossing
            frac = (target - phi[i - 1]) / (phi[i] - phi[i - 1])
            starts.append(t[i - 1] + frac / FS_FINE)
            k += 1
            target = 2.0 * np.pi * k
    return np.asarray(starts)


def phi_at(t_query, t_fine, phi):
    return np.interp(t_query, t_fine, phi)


# ----------------------------------------------------------------------------
# Waveform building blocks
# ----------------------------------------------------------------------------
def ppg_pulse(u):
    """One cardiac cycle, u in [0,1): systolic peak + smaller dicrotic bump.

    Peak-normalised to ~1; near zero at the cycle edges (diastole).
    """
    systolic = np.exp(-((u - 0.18) / 0.075) ** 2)
    dicrotic = 0.35 * np.exp(-((u - 0.42) / 0.10) ** 2)
    return (systolic + dicrotic) / 1.30


def cardiac_phase(t_fine, phi_resp):
    """Heart phase theta(t): HR = HR0 + RSA*sin(phi_resp) (RSA modulation)."""
    hr = HR0 + RSA_BPM * np.sin(phi_resp)               # bpm
    f = hr / 60.0                                       # Hz
    theta = 2.0 * np.pi * np.concatenate(
        ([0.0], np.cumsum((f[:-1] + f[1:]) * 0.5) / FS_FINE))
    return theta


def optical_channel(t_out, t_fine, phi_resp, theta, dc, ac, bw_amp, noise_std, rng):
    """Build one optical channel at output times t_out.

    Raw sensor polarity: pulses deflect DOWN at systole (reflectance PPG), so the
    analyzer's sign-inversion (settings.invert_ppg=True) turns them upright.
      raw = DC  - A(t)*pulse(u)  + baseline_wander(t) + noise
      A(t) modulated by respiration (RIIV/AUC), baseline_wander at f_resp (BW/LP).
    """
    phr = phi_at(t_out, t_fine, phi_resp)
    th = np.interp(t_out, t_fine, theta)
    u = np.mod(th, 2.0 * np.pi) / (2.0 * np.pi)
    amp = ac * (1.0 + RIIV_DEPTH * np.sin(phr))         # RIIV + AUC modulation
    baseline = bw_amp * np.sin(phr)                     # BW / LP modulation
    sig = dc - amp * ppg_pulse(u) + baseline
    if noise_std > 0:
        sig = sig + rng.normal(0.0, noise_std, sig.size)
    return sig


def airflow_reference(t_out, t_fine, phi_resp):
    """Nasal-pressure airflow: sinusoid + mild 2nd harmonic (slight I:E asymmetry).

    Breath period is preserved (extra harmonic is too small to add zero crossings),
    so the reference breath timing matches the watch signal exactly.
    """
    phr = phi_at(t_out, t_fine, phi_resp)
    return np.sin(phr) + 0.15 * np.sin(2.0 * phr)


# ----------------------------------------------------------------------------
# Minimal EDF writer (matches respiration_rr/io/edf_reader.py byte-for-byte)
# ----------------------------------------------------------------------------
def _edf_field(text, width):
    s = str(text)[:width]
    return s.ljust(width).encode("ascii", "replace")


def write_edf(path, signals, rec_duration=1.0):
    """Write a minimal but valid EDF.

    signals: list of dicts {label, data (float array), fs (== samples per record),
             phys_min, phys_max}.  All channels must span the same total duration.
    """
    n_sig = len(signals)
    samples_per_rec = [int(round(s["fs"] * rec_duration)) for s in signals]
    num_records = min(len(s["data"]) // spr for s, spr in zip(signals, samples_per_rec))

    DIG_MIN, DIG_MAX = -32768, 32767

    # ---- main header (256 bytes) ----
    hdr = b""
    hdr += _edf_field("0", 8)                                   # version
    hdr += _edf_field("X X X Synthetic", 80)                   # patient id
    hdr += _edf_field("Startdate 01-JAN-2026 Synthetic RR", 80)  # recording id
    hdr += _edf_field("01.01.26", 8)                            # start date dd.mm.yy
    hdr += _edf_field("00.00.00", 8)                            # start time hh.mm.ss
    hdr += _edf_field(256 + 256 * n_sig, 8)                     # header bytes
    hdr += _edf_field("", 44)                                   # reserved
    hdr += _edf_field(num_records, 8)                           # number of records
    hdr += _edf_field(("%g" % rec_duration), 8)                # record duration
    hdr += _edf_field(n_sig, 4)                                 # number of signals

    # ---- per-signal headers ----
    labels      = b"".join(_edf_field(s["label"], 16) for s in signals)
    transducer  = b"".join(_edf_field("", 80) for _ in signals)
    phys_dim    = b"".join(_edf_field(s.get("dim", "a.u."), 8) for s in signals)
    phys_min    = b"".join(_edf_field(("%g" % s["phys_min"]), 8) for s in signals)
    phys_max    = b"".join(_edf_field(("%g" % s["phys_max"]), 8) for s in signals)
    dig_min     = b"".join(_edf_field(DIG_MIN, 8) for _ in signals)
    dig_max     = b"".join(_edf_field(DIG_MAX, 8) for _ in signals)
    prefilter   = b"".join(_edf_field("", 80) for _ in signals)
    spr_field   = b"".join(_edf_field(spr, 8) for spr in samples_per_rec)
    reserved    = b"".join(_edf_field("", 32) for _ in signals)

    sig_hdr = (labels + transducer + phys_dim + phys_min + phys_max +
               dig_min + dig_max + prefilter + spr_field + reserved)

    # ---- digitise each channel ----
    digital = []
    for s, spr in zip(signals, samples_per_rec):
        d = np.asarray(s["data"][: num_records * spr], np.float64)
        pmn, pmx = s["phys_min"], s["phys_max"]
        q = (d - pmn) / (pmx - pmn) * (DIG_MAX - DIG_MIN) + DIG_MIN
        q = np.clip(np.round(q), DIG_MIN, DIG_MAX).astype("<i2")
        digital.append(q.reshape(num_records, spr))

    # ---- interleave records ----
    body = bytearray()
    for r in range(num_records):
        for ch in range(n_sig):
            body += digital[ch][r].tobytes()

    with open(path, "wb") as f:
        f.write(hdr + sig_hdr + bytes(body))


# ----------------------------------------------------------------------------
# Channel specs: CSV column -> (DC level, AC pulse amplitude, base sensor noise)
# ----------------------------------------------------------------------------
SPECS = {
    "PPG":     (20000, 2500, 30),
    "Artifact": (10000, 800, 60),
    "RED SIG": (30000, 2000, 35),
    "IR":      (45000, 4000, 45),
}

_SEED_OFFSET = {"stepped": 1, "constant": 2, "ramp": 3}

# variant name -> extra seed offset (keeps every (pattern, variant) reproducible)
VARIANTS = ("clean", "white", "mains", "spikes", "motion", "combined")
_NOISE_SEED = {"clean": 0, "white": 10, "mains": 20, "spikes": 30, "motion": 40, "combined": 50}


# ----------------------------------------------------------------------------
# Clean signal (no noise) — the noise variants are all built on top of this,
# so the ground-truth breathing timeline is identical across every variant.
# ----------------------------------------------------------------------------
def build_clean(name, rng):
    t_fine, phi = respiration_phase(name)
    theta = cardiac_phase(t_fine, phi)
    starts = breath_starts(t_fine, phi)

    tw = np.arange(0.0, T, 1.0 / FS_WATCH)
    chans = {col: optical_channel(tw, t_fine, phi, theta, dc, ac, ac * BW_FRAC, bn, rng)
             for col, (dc, ac, bn) in SPECS.items()}

    ax = rng.normal(0, 0.05, tw.size)               # near-flat accel = subject at rest
    ay = rng.normal(0, 0.05, tw.size)
    az = 1000.0 + rng.normal(0, 0.05, tw.size)

    tr = np.arange(0.0, T, 1.0 / FS_REF)
    airflow = airflow_reference(tr, t_fine, phi)
    ta = np.arange(0.0, T, 1.0 / FS_ACT)
    activity = np.zeros(ta.size)

    return dict(name=name, tw=tw, chans=chans, ax=ax, ay=ay, az=az,
                airflow=airflow, ta=ta, activity=activity, starts=starts)


# ----------------------------------------------------------------------------
# Noise injectors (moderate level). Each mutates copies in place.
# ----------------------------------------------------------------------------
def _white(chans, rng, frac):
    """Elevated Gaussian sensor noise, scaled per channel AC."""
    for col, (dc, ac, bn) in SPECS.items():
        chans[col] += rng.normal(0, frac * ac, chans[col].size)


def _mains(chans, tw, comps):
    """Power-line interference: 50 Hz + harmonics. NB the watch samples at 64 Hz,
    so 50 Hz aliases to ~14 Hz; it still sits above the 8 Hz PPG LP, so this mainly
    exercises the band-pass. Comps = [(freq_hz, amp_frac_of_AC), ...]."""
    for col, (dc, ac, bn) in SPECS.items():
        s = np.zeros_like(tw)
        for f, frac in comps:
            s += frac * ac * np.sin(2.0 * np.pi * f * tw)
        chans[col] += s


def _spikes(chans, tw, rng, rate_hz, amp_frac_range):
    """Sharp impulse glitches (electrical taps), shared sign across optical channels."""
    n = int(rate_hz * T)
    pos = rng.integers(0, tw.size - 2, n)
    for p in pos:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        amp = rng.uniform(*amp_frac_range)
        w = int(rng.integers(1, 3))                 # 1-2 samples wide
        for col, (dc, ac, bn) in SPECS.items():
            chans[col][p:p + w] += sign * amp * ac


def _pick_windows(rng, n, dur_range):
    """n non-overlapping windows spread across the record (avoids the first/last 10 s)."""
    seg = (T - 20.0) / n
    out = []
    for k in range(n):
        dur = rng.uniform(*dur_range)
        lo = 10.0 + k * seg
        hi = max(lo, 10.0 + (k + 1) * seg - dur)
        st = rng.uniform(lo, hi)
        out.append((st, st + dur))
    return out


def _motion(chans, tw, ax, ay, az, activity, ta, rng, n_windows):
    """Motion-artifact windows: big baseline excursion + broadband burst on the
    optical channels, a CORRELATED accelerometer oscillation (so the watch
    movement-gate fires), and an Activity spike on the EDF side (so the REMbo
    gate fires on the same windows). Returns the window list."""
    windows = _pick_windows(rng, n_windows, (5.0, 15.0))
    for (s, e) in windows:
        m = (tw >= s) & (tw < e)
        nsel = int(m.sum())
        if nsel < 2:
            continue
        loc = (tw[m] - s) / (e - s)
        bump = 0.5 - 0.5 * np.cos(2.0 * np.pi * loc)          # smooth 0->1->0 envelope
        for col, (dc, ac, bn) in SPECS.items():
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            chans[col][m] += sgn * 5.0 * ac * bump             # baseline excursion
            chans[col][m] += rng.normal(0, 0.3 * ac, nsel)     # broadband burst
        f = rng.uniform(2.0, 4.0)
        osc = 250.0 * np.sin(2.0 * np.pi * f * tw[m] + rng.uniform(0, 2 * np.pi)) * bump
        ax[m] += osc
        ay[m] += osc * rng.uniform(0.5, 1.0)
        az[m] += osc * rng.uniform(0.5, 1.0)
        am = (ta >= s) & (ta < e)
        activity[am] = 50.0                                    # REMbo activity spike
    return windows


def apply_noise(base, variant, rng):
    """Return noisy copies (chans, ax, ay, az, activity) for a variant."""
    chans = {k: v.copy() for k, v in base["chans"].items()}
    ax, ay, az = base["ax"].copy(), base["ay"].copy(), base["az"].copy()
    activity = base["activity"].copy()
    tw, ta = base["tw"], base["ta"]

    if variant == "clean":
        pass
    elif variant == "white":
        _white(chans, rng, 0.08)
    elif variant == "mains":
        _mains(chans, tw, [(50, 0.10), (100, 0.05), (150, 0.03)])
    elif variant == "spikes":
        _spikes(chans, tw, rng, 0.25, (3.0, 8.0))
    elif variant == "motion":
        _motion(chans, tw, ax, ay, az, activity, ta, rng, 3)
    elif variant == "combined":                                # everything, moderate
        _white(chans, rng, 0.05)
        _mains(chans, tw, [(50, 0.06), (100, 0.03)])
        _spikes(chans, tw, rng, 0.15, (3.0, 6.0))
        _motion(chans, tw, ax, ay, az, activity, ta, rng, 2)
    else:
        raise ValueError(variant)
    return chans, ax, ay, az, activity


# ----------------------------------------------------------------------------
# Write one variant (matched watch CSV + REMbo EDF + ground truth)
# ----------------------------------------------------------------------------
def write_variant(out_dir, base, variant, chans, ax, ay, az, activity):
    name = base["name"]
    prefix = f"{name}_" if variant == "clean" else f"{name}_{variant}_"
    tw, starts = base["tw"], base["starts"]

    watch_df = pd.DataFrame({
        "PPG": chans["PPG"], "Artifact": chans["Artifact"],
        "RED SIG": chans["RED SIG"], "IR": chans["IR"],
        "XL-X": ax, "XL-Y": ay, "XL-Z": az,
    })
    watch_df.to_csv(os.path.join(out_dir, prefix + "monitor_watch.csv"),
                    index=False, float_format="%.3f")

    write_edf(os.path.join(out_dir, prefix + "reference.edf"), [
        {"label": "Nasal P",  "data": base["airflow"], "fs": FS_REF, "phys_min": -1.5, "phys_max": 1.5, "dim": "cmH2O"},
        {"label": "Activity", "data": activity,        "fs": FS_ACT, "phys_min": 0.0,  "phys_max": 100.0, "dim": "a.u."},
    ])

    rr_true = 60.0 / np.diff(starts)
    pd.DataFrame({"breath_start_s": starts[1:], "rr_bpm": rr_true}).to_csv(
        os.path.join(out_dir, prefix + "ground_truth_rr.csv"), index=False, float_format="%.4f")
    pd.DataFrame({"time_s": tw, "target_rr_bpm": rr_profile(name, tw)}).to_csv(
        os.path.join(out_dir, prefix + "target_rr.csv"), index=False, float_format="%.4f")
    return prefix


def build_scenario(name, out_dir):
    base = build_clean(name, np.random.default_rng(SEED + _SEED_OFFSET[name]))
    starts = base["starts"]
    rr_true = 60.0 / np.diff(starts)
    print(f"[{name:8s}] {len(starts)} breaths | "
          f"RR {rr_true.min():.1f}-{rr_true.max():.1f} bpm (mean {rr_true.mean():.1f})")
    for variant in VARIANTS:
        rng = np.random.default_rng(SEED + _SEED_OFFSET[name] + _NOISE_SEED[variant])
        chans, ax, ay, az, activity = apply_noise(base, variant, rng)
        prefix = write_variant(out_dir, base, variant, chans, ax, ay, az, activity)
        print(f"    -> {prefix}*")
    return out_dir


def _writable(d):
    """True if we can create a new file in directory d (catches OneDrive
    online-only placeholder folders, which reject new-file creation)."""
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write_probe")
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def main():
    import sys
    import tempfile

    repo_root = os.path.dirname(os.path.dirname(HERE))          # .../BBBRR
    if len(sys.argv) > 1:
        candidates = [sys.argv[1]]
    else:
        # NB: do NOT default into HERE (Data/Synth) — the editor created it as a
        # OneDrive online-only placeholder that refuses new files. Use a fresh
        # folder in the repo root, then fall back to the system temp dir.
        candidates = [os.path.join(repo_root, "synth_output"),
                      os.path.join(tempfile.gettempdir(), "bbbrr_synth")]

    out_dir = next((d for d in candidates if _writable(d)), None)
    if out_dir is None:
        raise SystemExit(
            "No writable output directory found (tried: %s).\n"
            "Pass one explicitly, e.g.:  py Data/Synth/generate_synth.py C:/Temp/synth"
            % ", ".join(candidates))

    print(f"Generating synthetic RR data ({T:.0f}s, {FS_WATCH:.0f} Hz watch) -> {out_dir}")
    print(f"variants per scenario: {', '.join(VARIANTS)}\n")
    for name in ("stepped", "constant", "ramp"):
        build_scenario(name, out_dir)
    n = 3 * len(VARIANTS)
    print(f"\nDone -> {out_dir}  ({n} matched pairs)")
    print("Each pair: <name>[_<noise>]_monitor_watch.csv + _reference.edf + _ground_truth_rr.csv")
    print(f"\nRun the pipeline, e.g. (clean vs a noisy variant):\n"
          f'  py main.py --edf "{os.path.join(out_dir, "stepped_reference.edf")}" '
          f'--watch "{os.path.join(out_dir, "stepped_monitor_watch.csv")}"\n'
          f'  py main.py --edf "{os.path.join(out_dir, "stepped_motion_reference.edf")}" '
          f'--watch "{os.path.join(out_dir, "stepped_motion_monitor_watch.csv")}"')


if __name__ == "__main__":
    main()
