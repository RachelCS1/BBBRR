"""
Cross-device comparison — Tool 3.

Ranks every watch PPG respiration-rate series (parameter x channel) against the
REMbo zero-crossing reference RR by mean absolute error (MAE, bpm) over their
temporal overlap, after a time-offset shift. Mirrors the "Best 3 vs ref" view of
the Combined RR Analyzer (renderSimCompare): shift each candidate onto the device
clock, score by MAE, draw the reference plus the closest N.
"""

from dataclasses import dataclass, field
import numpy as np

from ..settings import COMPARE


@dataclass
class Candidate:
    label: str          # e.g. "IR·RSA"
    channel: str
    param: str
    t: np.ndarray       # candidate RR timestamps (watch clock)
    rr: np.ndarray      # candidate RR values (bpm)


@dataclass
class CompareResult:
    offset_sec: float
    ref_time: np.ndarray
    ref_rr: np.ndarray
    ranked: list                    # list of (Candidate, mae, n_overlap)
    candidates: list = field(default_factory=list)


def mae_by_candidate(cmp):
    """{(channel, param): (mae, n_overlap)} for every scored candidate.

    Convenience lookup so plots can annotate each parameter with its own MAE."""
    return {(c.channel, c.param): (mae, n) for (c, mae, n) in cmp.ranked}


def reference_rr_series(ref_result):
    """(center_time, rate) for accepted reference breaths, sorted by time."""
    b = [(br.center, br.rate) for br in ref_result.breaths
         if np.isfinite(br.center) and np.isfinite(br.rate)]
    b.sort()
    if not b:
        return np.zeros(0), np.zeros(0)
    t = np.array([x[0] for x in b])
    r = np.array([x[1] for x in b])
    return t, r


def collect_watch_candidates(ppg_results, params=("RSA", "RSA_spline", "RSA_ssp", "RIIV", "RIIV_spline", "RIIV_ssp", "AUC", "AUC_spline", "AUC_ssp", "LP", "BWlegacy", "BWbank")):
    """Flatten analyze_ppg() output into a list of Candidate series."""
    cands = []
    for channel, res in ppg_results.items():
        for p in params:
            pr = res.params.get(p)
            if pr is None or pr.rr_time.size == 0:
                continue
            cands.append(Candidate(label=f"{channel}/{p}", channel=channel, param=p,
                                   t=np.asarray(pr.rr_time), rr=np.asarray(pr.rr_bpm)))
        # spectral ridge as an extra candidate
        if res.ridge_rr is not None and np.isfinite(res.ridge_rr).any():
            m = np.isfinite(res.ridge_rr)
            cands.append(Candidate(label=f"{channel}/Ridge", channel=channel, param="Ridge",
                                   t=np.asarray(res.ridge_time)[m], rr=np.asarray(res.ridge_rr)[m]))
    return cands


def score_candidate(cand, ref_time, ref_rr, offset, min_overlap_sec=None):
    """MAE (bpm) of a candidate vs reference over their overlap.

    Candidate timestamps are shifted by +offset onto the device clock, then each
    is compared to the reference RR linearly interpolated at that time. Returns
    (mae, n_overlap) or (nan, 0) if the overlap is shorter than min_overlap_sec.
    """
    if min_overlap_sec is None:
        min_overlap_sec = COMPARE.min_overlap_sec
    if cand.t.size == 0 or ref_time.size < 2:
        return float("nan"), 0
    ts = cand.t + offset
    inside = (ts >= ref_time[0]) & (ts <= ref_time[-1])
    if inside.sum() < 2:
        return float("nan"), 0
    span = ts[inside].max() - ts[inside].min()
    if span < min_overlap_sec:
        return float("nan"), int(inside.sum())
    ref_at = np.interp(ts[inside], ref_time, ref_rr)
    mae = float(np.nanmean(np.abs(cand.rr[inside] - ref_at)))
    return mae, int(inside.sum())


def rank_candidates(cands, ref_time, ref_rr, offset, min_overlap_sec=None):
    """Score and sort candidates by ascending MAE (best first)."""
    scored = []
    for c in cands:
        mae, n = score_candidate(c, ref_time, ref_rr, offset, min_overlap_sec)
        if np.isfinite(mae):
            scored.append((c, mae, n))
    scored.sort(key=lambda x: x[1])
    return scored


def compare_watch_vs_reference(ppg_results, ref_result, offset=0.0,
                               top_n=None, params=("RSA", "RSA_spline", "RSA_ssp", "RIIV", "RIIV_spline", "RIIV_ssp", "AUC", "AUC_spline", "AUC_ssp", "LP", "BWlegacy", "BWbank")):
    """End-to-end comparison. `offset` (seconds) shifts every watch candidate onto
    the REMbo clock; supply the IR-PPG MSD sync offset (see respiration_rr.sync)."""
    if top_n is None:
        top_n = COMPARE.top_n
    ref_t, ref_r = reference_rr_series(ref_result)
    cands = collect_watch_candidates(ppg_results, params)
    ranked = rank_candidates(cands, ref_t, ref_r, offset)
    return CompareResult(offset_sec=float(offset), ref_time=ref_t, ref_rr=ref_r,
                         ranked=ranked, candidates=cands)
