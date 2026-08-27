#!/usr/bin/env python
"""
ref_split_viewer.py — SEE the reference-driven category split per recording.

Read-only viewer. It shows, for each recording, how the recording splits into
RR x HR categories when the routing RR source is the REFERENCE median-average
(the same "route-by ref" idea as `bw_category_map.py --route-by ref`). Purpose:
eyeball the split and DECIDE the segment length before building the per-category
BBBRR analysis.

It reuses the EXACT routing pieces from bw_category_map (RR_EDGES / HR_EDGES /
_mode_bin / hr_series_from_beats / _labels) and rr_average.reference_average_rr,
so nothing here diverges from the scoring tool — and none of those tools are
modified.

Per recording -> one figure:
  (1) reference average RR over time, background shaded by RR bin, raw per-breath
      RR points overlaid, segment boundaries for the FIRST --segment-sec drawn;
  (2) HR (from PPG beats) over time, background shaded by HR bin;
  (3) one category strip per --segment-sec value (stacked), each block coloured by
      the segment's (RR bin, HR bin) category, so you can compare segment lengths.

A compact text summary per length is printed: segment count, per-cell counts, and
mean "purity" (fraction of a segment's reference RR samples that sit in the
majority RR bin — shorter segments are purer but hold fewer samples).

Usage
-----
    py ref_split_viewer.py --rec-id Exp2/002
    py ref_split_viewer.py --rec-id Exp2/002 --segment-sec 15 30 60
    py ref_split_viewer.py --all
    py ref_split_viewer.py --all --save DIR --no-show
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

import bw_bp_sweep as S            # discover / _find_inputs / DEFAULT_DATA_ROOT
import bw_category_map as CM       # RR_EDGES / HR_EDGES / _mode_bin / hr_series_from_beats / _labels

# Colours: RR bins (<12 / 12-30 / >30) and HR bins (<60 / 60-85 / >85).
RR_BAND_COLORS = ["#dbeafe", "#dcfce7", "#fee2e2"]     # light blue / green / red (backgrounds)
RR_LINE_COLORS = ["#2563eb", "#16a34a", "#dc2626"]     # saturated for the category strip
HR_BAND_COLORS = ["#ede9fe", "#fef9c3", "#ffedd5"]     # light purple / yellow / orange


def _segment_recording(truth_at, truth_ar, hr_t, hr_v, seg_sec, segment_min,
                       rr_edges, hr_edges, fill=True):
    """Mirror bw_category_map's segment loop; return a list of segment dicts.

    Each dict: {a0, a1, rb, hb, purity, n, filled}. A segment with too few
    reference RR samples (or no HR) cannot be scored on its own; when `fill` is on
    the missing bin is CARRIED FORWARD from the previous resolved segment
    (breathing drifts gradually, so the regime is assumed to continue) and the
    segment is marked filled=True. rb/hb stay None only when nothing precedes it to
    carry (recording start), leaving it skipped."""
    n_rr = len(rr_edges) + 1
    t0 = max(truth_at[0], hr_t[0])
    t1 = min(truth_at[-1], hr_t[-1])
    out = []
    last_rb = last_hb = None
    seg = t0
    while seg + seg_sec <= t1 + 1e-9:
        a0, a1 = seg, seg + seg_sec
        seg += seg_sec                                   # non-overlapping
        rr_ref = truth_ar[(truth_at >= a0) & (truth_at < a1) & np.isfinite(truth_ar)]
        rr_ref = rr_ref[(rr_ref >= 4) & (rr_ref <= 55)]
        hseg = hr_v[(hr_t >= a0) & (hr_t < a1)]
        rb_cur = CM._mode_bin(rr_ref, n_rr, rr_edges) if rr_ref.size >= segment_min else None
        hb_cur = int(np.digitize(np.median(hseg), hr_edges)) if hseg.size >= 1 else None
        purity = (float(np.mean(np.digitize(rr_ref, rr_edges) == rb_cur))
                  if rb_cur is not None else np.nan)
        # resolve: keep this segment's own bin, else carry the previous one forward
        rb = rb_cur if rb_cur is not None else (last_rb if fill else None)
        hb = hb_cur if hb_cur is not None else (last_hb if fill else None)
        filled = (rb is not None and hb is not None) and (rb_cur is None or hb_cur is None)
        if rb is not None:
            last_rb = rb
        if hb is not None:
            last_hb = hb
        out.append(dict(a0=a0, a1=a1, rb=rb, hb=hb, purity=purity, n=rr_ref.size,
                        filled=filled))
    return out


def _segment_by_crossing(truth_at, truth_ar, hr_t, hr_v, min_subseg_sec,
                         rr_edges, hr_edges, fill=True):
    """Split at CATEGORY CROSSINGS instead of a fixed grid, so a sustained regime
    change is never swallowed by a majority vote.

    Works on the reference median-average grid (≈4 s spacing). Each grid point gets
    its own (RR bin, HR bin); a NEW category is ACCEPTED only once it persists for
    >= min_subseg_sec (causal hysteresis: brief excursions shorter than that are
    absorbed into the current regime — "the transition need not be instant"). Once
    accepted, the boundary is placed at the TRUE crossing point, so no part of a
    regime is mislabelled. Missing reference points are carried forward when
    `fill` is on. Returns the same segment-dict shape as _segment_recording
    (purity is NaN — each run is single-category by construction; filled=True if the
    run spans a reference gap)."""
    n_rr = len(rr_edges) + 1
    t0 = max(truth_at[0], hr_t[0])
    t1 = min(truth_at[-1], hr_t[-1])
    m = (truth_at >= t0) & (truth_at <= t1)
    g = truth_at[m]
    rr = truth_ar[m]
    if g.size == 0:
        return []
    dt = float(np.median(np.diff(g))) if g.size > 1 else 4.0
    min_pts = max(1, int(round(min_subseg_sec / dt)))
    hr_i = np.interp(g, hr_t, hr_v)                  # HR at each grid point

    # per-point RR bin only (with carry-forward over gaps). The HR bin is assigned
    # per RUN below from the median HR — segmenting on the (RR,HR) tuple lets HR-bin
    # flicker / HR spikes fragment solid RR runs and swallow real RR transitions.
    rb_all = np.full(g.size, -1, int)
    last = None
    for k in range(g.size):
        v = rr[k]
        if np.isfinite(v) and 4.0 <= v <= 55.0:
            rb_all[k] = int(np.digitize(v, rr_edges)); last = rb_all[k]
        elif fill and last is not None:
            rb_all[k] = last
    gap = np.array([not (np.isfinite(rr[k]) and 4.0 <= rr[k] <= 55.0)
                    for k in range(g.size)])
    cat = [None if rb_all[k] < 0 else int(rb_all[k]) for k in range(g.size)]

    # causal min-duration hysteresis on the RR bin -> accepted RR bin per point
    accepted = [None] * g.size
    cur = None
    k = 0
    while k < g.size:
        c = cat[k]
        if c is None:
            accepted[k] = cur; k += 1; continue
        if cur is None or c == cur:
            cur = c; accepted[k] = cur; k += 1; continue
        j = k                                            # length of this contiguous new-bin run
        while j < g.size and cat[j] == c:
            j += 1
        if (j - k) >= min_pts:                           # sustained -> accept from the true crossing
            cur = c
        for x in range(k, j):                            # else absorb into the current regime
            accepted[x] = cur
        k = j

    # build runs -> segment dicts; HR bin from the median HR over the run (robust to
    # HR-bin flicker / spikes); boundaries at grid midpoints
    out = []
    p = 0
    while p < g.size:
        if accepted[p] is None:
            p += 1; continue
        q = p
        while q + 1 < g.size and accepted[q + 1] == accepted[p]:
            q += 1
        a0 = t0 if p == 0 else (g[p - 1] + g[p]) / 2.0
        a1 = t1 if q == g.size - 1 else (g[q] + g[q + 1]) / 2.0
        rb = accepted[p]
        hb = int(np.digitize(np.median(hr_i[p:q + 1]), hr_edges))
        out.append(dict(a0=a0, a1=a1, rb=rb, hb=hb, purity=np.nan,
                        n=q - p + 1, filled=bool(gap[p:q + 1].any())))
        p = q + 1
    return out


def _summarise(rid, seg_sec, segs, rr_labels, hr_labels, n_rr, n_hr):
    resolved = [s for s in segs if s["rb"] is not None and s["hb"] is not None]
    real = [s for s in resolved if not s["filled"]]
    filled = [s for s in resolved if s["filled"]]
    grid = np.zeros((n_rr, n_hr), int)
    for s in resolved:
        grid[s["rb"], s["hb"]] += 1
    purity = np.nanmean([s["purity"] for s in real]) if real else np.nan
    print(f"  seg={seg_sec:g}s | segments={len(resolved)} "
          f"(real={len(real)}, filled={len(filled)}, skipped={len(segs) - len(resolved)}) | "
          f"mean purity (real)={purity*100:.0f}%")
    print("     " + "".join(f"{h:>10}" for h in hr_labels))
    for i in range(n_rr):
        print(f"    {rr_labels[i]:8}" + "".join(f"{grid[i, j]:>10}" for j in range(n_hr)))
    return grid


def _plot(rid, truth_at, truth_ar, raw_t, raw_r, hr_t, hr_v, strips, boundary_segs,
          rr_edges, hr_edges, rr_labels, hr_labels, plt):
    """strips = list of (label, segs); boundary_segs = the strip whose boundaries are
    drawn as vlines on the RR panel."""
    n_rr, n_hr = len(rr_edges) + 1, len(hr_edges) + 1
    n_strip = len(strips)
    fig = plt.figure(figsize=(13, 6.2 + 0.5 * n_strip))
    gs = fig.add_gridspec(2 + n_strip, 1, height_ratios=[3, 2] + [0.6] * n_strip,
                          hspace=0.35)
    tmin = min(truth_at[0], hr_t[0])
    tmax = max(truth_at[-1], hr_t[-1])

    # ---- (1) reference RR ----
    ax = fig.add_subplot(gs[0])
    rr_lo = [0.0] + list(rr_edges)
    rr_hi = list(rr_edges) + [max(55.0, np.nanmax(truth_ar) + 5)]
    for i in range(n_rr):
        ax.axhspan(rr_lo[i], rr_hi[i], color=RR_BAND_COLORS[i], zorder=0)
    for e in rr_edges:
        ax.axhline(e, color="#64748b", lw=0.8, ls="--", zorder=1)
    ax.plot(raw_t, raw_r, ".", ms=3, color="#94a3b8", alpha=0.6, label="reference per-breath RR")
    ax.plot(truth_at, truth_ar, "-", color="#0f172a", lw=1.8, label="reference median-avg RR")
    for s in boundary_segs:                               # boundaries of the chosen strip
        ax.axvline(s["a0"], color="#334155", lw=0.5, alpha=0.35, zorder=1)
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0, rr_hi[-1])
    ax.set_ylabel("RR (bpm)")
    ax.set_title(f"{rid} — reference category split   (bands = RR bins {rr_labels})",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- (2) HR ----
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    hr_lo = [0.0] + list(hr_edges)
    hr_hi = list(hr_edges) + [max(120.0, np.nanmax(hr_v) + 10)]
    for j in range(n_hr):
        ax2.axhspan(hr_lo[j], hr_hi[j], color=HR_BAND_COLORS[j], zorder=0)
    for e in hr_edges:
        ax2.axhline(e, color="#64748b", lw=0.8, ls="--", zorder=1)
    ax2.plot(hr_t, hr_v, "-", color="#7c3aed", lw=1.3)
    ax2.set_xlim(tmin, tmax)
    ax2.set_ylim(hr_lo[0], hr_hi[-1])
    ax2.set_ylabel("HR (bpm)")

    # ---- (3) category strips ----
    for k, (label, segs) in enumerate(strips):
        axs = fig.add_subplot(gs[2 + k], sharex=ax)
        for s in segs:
            if s["rb"] is None or s["hb"] is None:
                axs.axvspan(s["a0"], s["a1"], color="#e5e7eb", zorder=0)   # skipped = grey
            else:
                # carried-forward segments get a hatch so they read as inferred, not measured
                hatch = "///" if s["filled"] else None
                axs.axvspan(s["a0"], s["a1"], color=RR_LINE_COLORS[s["rb"]],
                            alpha=0.35 + 0.22 * s["hb"], zorder=0,
                            hatch=hatch, edgecolor="#0f172a" if s["filled"] else None,
                            lw=0.0)
                axs.text((s["a0"] + s["a1"]) / 2, 0.5,
                         f"{s['rb']}{s['hb']}", ha="center", va="center", fontsize=6.5,
                         color="#0f172a")
        axs.set_yticks([])
        axs.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        axs.set_xlim(tmin, tmax)
    axs.set_xlabel("time (s, REMbo clock)   |   strip label = RR-bin HR-bin ; alpha = HR bin ; "
                   "hatch = filled from previous ; grey = skipped")
    fig.tight_layout()
    return fig


def _process(rid, edf, csv, seg_secs, segment_min, min_subseg_sec, rr_edges, hr_edges,
             fill, run_reference, run_ppg, run_sync, reference_average_rr,
             reference_rr_points, plt):
    n_rr, n_hr = len(rr_edges) + 1, len(hr_edges) + 1
    rr_labels = CM._labels(rr_edges, "RR")
    hr_labels = CM._labels(hr_edges, "HR")

    ref = run_reference(edf)
    ppg_results, sig = run_ppg(csv)
    offset = run_sync(edf, ppg_results, sig)
    truth_at, truth_ar = reference_average_rr(ref)
    raw_t, raw_r = reference_rr_points(ref)
    hr_t, hr_v = CM.hr_series_from_beats(ppg_results, sig, offset)
    if truth_at.size < 2 or hr_t.size < 2:
        print(f"  [skip] {rid}: no reference / HR")
        return None

    print(f"\n=== {rid} ===")
    strips = []
    for ss in seg_secs:                                   # fixed-length majority strips
        segs = _segment_recording(truth_at, truth_ar, hr_t, hr_v, ss, segment_min,
                                  rr_edges, hr_edges, fill=fill)
        _summarise(rid, ss, segs, rr_labels, hr_labels, n_rr, n_hr)
        strips.append((f"{ss:g}s", segs))
    cross = _segment_by_crossing(truth_at, truth_ar, hr_t, hr_v, min_subseg_sec,
                                 rr_edges, hr_edges, fill=fill)  # split-at-crossing strip
    lens = [s["a1"] - s["a0"] for s in cross]
    print(f"  split@{min_subseg_sec:g}s | runs={len(cross)} | "
          f"median run={np.median(lens):.0f}s (min {min(lens):.0f}, max {max(lens):.0f})"
          if cross else f"  split@{min_subseg_sec:g}s | runs=0")
    strips.append((f"split@{min_subseg_sec:g}s", cross))

    return _plot(rid, truth_at, truth_ar, raw_t, raw_r, hr_t, hr_v, strips, cross,
                 rr_edges, hr_edges, rr_labels, hr_labels, plt)


def main(argv=None):
    ap = argparse.ArgumentParser(description="See the reference-driven RRxHR category split per recording")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--rec-id", default=None, help="recording id, e.g. Exp2/002")
    ap.add_argument("--recordings", nargs="+", default=None, help="explicit rec ids")
    ap.add_argument("--all", action="store_true", help="loop over every recording")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--segment-sec", type=float, nargs="+", default=[30.0],
                    help="one or more segment lengths (s) to draw/compare")
    ap.add_argument("--segment-min", type=int, default=3,
                    help="min reference RR samples in a segment to score it (matches bw_category_map)")
    ap.add_argument("--no-fill", action="store_true",
                    help="do NOT carry a missing-reference segment's category forward from the "
                         "previous window (default: fill it, marked with a hatch)")
    ap.add_argument("--min-subseg-sec", type=float, default=8.0,
                    help="split-at-crossing strip: a new category must persist this long to be "
                         "accepted (causal hysteresis); shorter excursions stay with the current "
                         "regime. Larger = closer to fixed-window majority; smaller = more reactive.")
    ap.add_argument("--rr-edges", type=float, nargs="+", default=CM.RR_EDGES)
    ap.add_argument("--hr-edges", type=float, nargs="+", default=CM.HR_EDGES)
    ap.add_argument("--save", default=None, metavar="DIR")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    recs = S.discover(args.data_root)
    if args.recordings:
        recs = [r for r in recs if r[0] in args.recordings]
    elif args.rec_id:
        recs = [r for r in recs if r[0] == args.rec_id]
    elif not args.all:
        recs = recs[:1]                                  # default: first recording only
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        sys.exit("No recordings.")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr, reference_rr_points

    out_dir = args.save
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for rid, edf, csv in recs:
        try:
            fig = _process(rid, edf, csv, args.segment_sec, args.segment_min,
                           args.min_subseg_sec, list(args.rr_edges), list(args.hr_edges),
                           not args.no_fill, run_reference, run_ppg, run_sync,
                           reference_average_rr, reference_rr_points, plt)
        except Exception as e:
            print(f"  [ERR] {rid}: {e}")
            continue
        if fig is None:
            continue
        if out_dir:
            png = os.path.join(out_dir, f"ref_split_{rid.replace('/', '_')}.png")
            fig.savefig(png, dpi=120, bbox_inches="tight")
            print(f"  saved -> {png}")
        if args.no_show:
            plt.close(fig)

    if not args.no_show:
        print("\nOpening figures — close the windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
