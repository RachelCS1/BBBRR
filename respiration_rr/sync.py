"""
IR-PPG MSD time-synchronisation — watch <-> REMbo.

Finds the time offset between the two device clocks by matching the MSD
(max-systolic-derivative) fiducials of the SHARED finger pulse: the watch
`infra_red` channel vs the REMbo `Pulse Wave` channel. Both go through the SAME
existing beat pipeline (analyze_ppg_channel -> compute_systolic_analysis), so the
matched points are identical fiducials on both devices. Matching is nearest-
neighbour on the discrete MSD times — no resampling onto a common grid.

    offset_sec = time to ADD to the WATCH clock to reach the REMbo clock
    rembo_time ~= watch_time + offset_sec

The pulse is respiration-independent, so this sync does NOT depend on any
respiration/RR method we later benchmark.
"""

from dataclasses import dataclass, field
import numpy as np

from .settings import PPG, SYNC, REFERENCE
from .preprocessing.resample import resample_linear, upsample_fft
from .ppg.respiration import analyze_ppg_channel
from .io.edf_reader import EDFReader


def _params(params=None):
    """Merge caller overrides over the settings.SYNC defaults (single source)."""
    base = dict(
        max_offset_sec=SYNC.max_offset_sec, coarse_step_sec=SYNC.coarse_step_sec,
        fine_step_sec=SYNC.fine_step_sec, match_tol_sec=SYNC.match_tol_sec,
        min_overlap_sec=SYNC.min_overlap_sec, min_matched=SYNC.min_matched,
        window_sec=SYNC.window_sec, min_prominence=SYNC.min_prominence,
    )
    base.update(params or {})
    return base


@dataclass
class MsdSeries:
    """Per-signal diagnostics: raw pulse, band-passed pulse, and MSD fiducials."""
    label: str
    raw: np.ndarray
    raw_fs: float
    t: np.ndarray            # 1024 Hz trimmed timeline (s)
    filtered: np.ndarray     # band-passed cardiac pulse
    msd_t: np.ndarray        # MSD fiducial times (s)
    invert: bool


@dataclass
class SyncResult:
    offset_sec: float
    matched: int
    matched_frac: float
    median_resid_ms: float
    prominence: float
    overlap_sec: float
    polarity: int            # REMbo polarity used: +1 = as-is, -1 = inverted
    low_confidence: bool
    reason: str = ""
    # ---- diagnostics (for the inspector) ----
    watch: MsdSeries = None
    rembo: MsdSeries = None
    coarse_tau: np.ndarray = None
    coarse_n: np.ndarray = None
    coarse_med: np.ndarray = None
    fine_tau: np.ndarray = None
    fine_n: np.ndarray = None
    residuals_ms: np.ndarray = None
    window: tuple = None      # (start, end) of the clean window (REMbo clock)
    params: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Stage 1 — pre-processing -> MSD fiducials  (same pipeline for both signals)
# ----------------------------------------------------------------------
def msd_series(raw, raw_fs, label, invert, cfg=PPG):
    """Raw pulse channel -> the existing 1024 Hz beat pipeline -> MSD times.

    resample to fs_orig -> FFT-upsample x4 -> trim head/tail -> analyze_ppg_channel
    (invert -> band-pass -> LPF-derivative beats -> SS refine -> MSD). Uses the same
    clock convention as prepare_watch (t = index/fs, head-trim offset included).
    """
    x = np.asarray(raw, np.float64)
    at256 = resample_linear(x, raw_fs, cfg.fs_orig)
    up = upsample_fft(at256, cfg.upsample_factor)
    fs = cfg.target_fs
    h = int(round(cfg.trim_head_sec * fs))
    tl = int(round(cfg.trim_tail_sec * fs))
    lo, hi = h, max(h + 1, up.size - tl)
    sig = up[lo:hi]
    t = np.arange(lo, hi) / fs
    res = analyze_ppg_channel(sig, t, fs, channel="IR", invert=invert,
                              compute_ridge=False, move_regions=None)
    msd = np.asarray(res.msd_idx, dtype=int)
    msd = msd[(msd >= 0) & (msd < t.size)]
    return MsdSeries(label=label, raw=x, raw_fs=float(raw_fs), t=t,
                     filtered=np.asarray(res.filtered, np.float64),
                     msd_t=t[msd], invert=bool(invert))


# ----------------------------------------------------------------------
# Stages 2-3 — nearest-neighbour MSD-time matching
# ----------------------------------------------------------------------
def _matches(watch_t, rembo_t, tau, tol):
    """Matched (nearest REMbo time, signed residual) for watch MSD shifted by tau.

    residual = nearest_rembo - (watch + tau).  Only pairs within +-tol are kept.
    """
    if watch_t.size == 0 or rembo_t.size < 2:
        return np.array([]), np.array([])
    s = watch_t + tau
    m = (s >= rembo_t[0]) & (s <= rembo_t[-1])
    s = s[m]
    if s.size == 0:
        return np.array([]), np.array([])
    idx = np.clip(np.searchsorted(rembo_t, s), 1, rembo_t.size - 1)
    left, right = rembo_t[idx - 1], rembo_t[idx]
    nearest = np.where(s - left <= right - s, left, right)
    resid = nearest - s
    within = np.abs(resid) <= tol
    return nearest[within], resid[within]


def _score(watch_t, rembo_t, tau, tol):
    """(n_matched, median_abs_residual) at offset tau."""
    _, resid = _matches(watch_t, rembo_t, tau, tol)
    if resid.size == 0:
        return 0, np.nan
    return int(resid.size), float(np.median(np.abs(resid)))


def _argbest(n, med):
    """Index with the most matches; ties broken by smallest median distance."""
    best = n.max()
    cand = np.where(n == best)[0]
    m = np.where(np.isfinite(med[cand]), med[cand], np.inf)
    return int(cand[np.argmin(m)])


def _clean_window_correction(watch_t, rembo_t, tau, tol, window_sec):
    """Median residual inside the densest window_sec-long window (drift-robust lock).

    Returns (correction_sec, (win_start, win_end) | None).
    """
    rn, resid = _matches(watch_t, rembo_t, tau, tol)
    if rn.size == 0:
        return 0.0, None
    if rn[-1] - rn[0] <= window_sec:
        return float(np.median(resid)), (float(rn[0]), float(rn[-1]))
    best_cnt, best = -1, None
    for w0 in np.arange(rn[0], rn[-1] - window_sec, window_sec / 2.0):
        sel = (rn >= w0) & (rn < w0 + window_sec)
        c = int(sel.sum())
        if c > best_cnt:
            best_cnt, best = c, (w0, w0 + window_sec, sel)
    w0, w1, sel = best
    return float(np.median(resid[sel])), (float(w0), float(w1))


def match_offset(watch_t, rembo_t, p):
    """Coarse scan -> fine refine -> clean-window residual lock. Returns a dict."""
    watch_t = np.asarray(watch_t, np.float64)
    rembo_t = np.asarray(rembo_t, np.float64)
    tol, mo = p["match_tol_sec"], p["max_offset_sec"]

    # --- coarse ---
    ctau = np.arange(-mo, mo + p["coarse_step_sec"], p["coarse_step_sec"])
    cn = np.zeros(ctau.size)
    cmed = np.full(ctau.size, np.nan)
    for i, tau in enumerate(ctau):
        cn[i], cmed[i] = _score(watch_t, rembo_t, tau, tol)
    if cn.max() == 0:
        return dict(offset=0.0, matched=0, median_resid_ms=np.nan, prominence=0.0,
                    ctau=ctau, cn=cn, cmed=cmed, ftau=np.array([]), fn=np.array([]),
                    residuals_ms=np.array([]), window=None)
    b = _argbest(cn, cmed)
    tau0 = float(ctau[b])
    prom = float((cn[b] - np.median(cn)) / (np.std(cn) + 1e-9))

    # --- fine ---
    ftau = np.arange(tau0 - p["coarse_step_sec"], tau0 + p["coarse_step_sec"] + p["fine_step_sec"],
                     p["fine_step_sec"])
    fn = np.zeros(ftau.size)
    fmed = np.full(ftau.size, np.nan)
    for i, tau in enumerate(ftau):
        fn[i], fmed[i] = _score(watch_t, rembo_t, tau, tol)
    tau_fine = float(ftau[_argbest(fn, fmed)])

    # --- clean-window residual lock ---
    corr, window = _clean_window_correction(watch_t, rembo_t, tau_fine, tol, p["window_sec"])
    offset = tau_fine + corr

    rn, resid = _matches(watch_t, rembo_t, offset, tol)
    med_ms = float(np.median(np.abs(resid)) * 1000) if resid.size else np.nan
    return dict(offset=float(offset), matched=int(resid.size), median_resid_ms=med_ms,
                prominence=prom, ctau=ctau, cn=cn, cmed=cmed, ftau=ftau, fn=fn,
                residuals_ms=resid * 1000, window=window,
                matched_rembo_t=rn)


# ----------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------
def read_rembo_pulse_wave(edf_path, cfg=REFERENCE):
    """(signal, fs, channel_name) for the REMbo finger-PPG channel (robust to naming)."""
    r = EDFReader(edf_path)
    labels = r.channel_labels()
    name = cfg.edf_ppg_channel if cfg.edf_ppg_channel in labels else None
    if name is None:
        for lbl in labels:
            if any(k in lbl.lower() for k in ("pulse", "ppg", "wave")):
                name = lbl
                break
    if name is None:
        raise ValueError(f"No Pulse Wave / PPG channel in EDF. Available: {labels}")
    _, sig, fs = r.read_channel(name)
    return sig, fs, name


def _match_best(watch_msd_t, rembo_pw, rembo_fs, p, try_polarity, cfg):
    """Try REMbo polarities, keep the one matching more beats. Returns (r, rembo_series, inv)."""
    best = None
    for inv in ([False, True] if try_polarity else [False]):
        rs = msd_series(rembo_pw, rembo_fs, "REMbo Pulse Wave", invert=inv, cfg=cfg)
        r = match_offset(watch_msd_t, rs.msd_t, p)
        if best is None or r["matched"] > best[0]["matched"]:
            best = (r, rs, inv)
    return best


def _finalize(watch_msd_t, r, rs, inv, p, watch_series):
    """Build a SyncResult (offset + confidence + diagnostics) from a match dict."""
    s = np.asarray(watch_msd_t) + r["offset"]
    in_overlap = int(((s >= rs.msd_t[0]) & (s <= rs.msd_t[-1])).sum()) if rs.msd_t.size else 0
    matched_frac = (r["matched"] / in_overlap) if in_overlap else 0.0
    rn = r.get("matched_rembo_t", np.array([]))
    overlap_sec = float(rn[-1] - rn[0]) if rn.size >= 2 else 0.0

    reasons = []
    if r["prominence"] < p["min_prominence"]:
        reasons.append(f"prominence {r['prominence']:.1f}<{p['min_prominence']}sigma")
    if overlap_sec < p["min_overlap_sec"]:
        reasons.append(f"overlap {overlap_sec:.0f}<{p['min_overlap_sec']:.0f}s")
    if r["matched"] < p["min_matched"]:
        reasons.append(f"matched {r['matched']}<{p['min_matched']}")

    return SyncResult(
        offset_sec=r["offset"], matched=r["matched"], matched_frac=matched_frac,
        median_resid_ms=r["median_resid_ms"], prominence=r["prominence"],
        overlap_sec=overlap_sec, polarity=(-1 if inv else 1), low_confidence=bool(reasons),
        reason="; ".join(reasons), watch=watch_series, rembo=rs,
        coarse_tau=r["ctau"], coarse_n=r["cn"], coarse_med=r["cmed"],
        fine_tau=r["ftau"], fine_n=r["fn"], residuals_ms=r["residuals_ms"],
        window=r["window"], params=p)


def compute_offset(watch_ir, watch_fs, rembo_pw, rembo_fs,
                   params=None, try_polarity=True, cfg=PPG):
    """Full sync from RAW signals -> SyncResult (with watch diagnostics for the inspector).

    The watch IR uses the standard PPG inversion (cfg.invert_ppg); the REMbo Pulse
    Wave polarity is unknown, so both are tried and the better match is kept.
    """
    p = _params(params)
    ws = msd_series(watch_ir, watch_fs, "watch IR", invert=cfg.invert_ppg, cfg=cfg)
    r, rs, inv = _match_best(ws.msd_t, rembo_pw, rembo_fs, p, try_polarity, cfg)
    return _finalize(ws.msd_t, r, rs, inv, p, ws)


def offset_from_msd(watch_msd_t, rembo_pw, rembo_fs,
                    params=None, try_polarity=True, cfg=PPG):
    """Light entry for the live pipeline: watch MSD times are ALREADY computed
    (by analyze_ppg), so only the REMbo Pulse Wave is processed here."""
    p = _params(params)
    watch_msd_t = np.asarray(watch_msd_t, np.float64)
    r, rs, inv = _match_best(watch_msd_t, rembo_pw, rembo_fs, p, try_polarity, cfg)
    return _finalize(watch_msd_t, r, rs, inv, p, None)
