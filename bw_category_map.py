#!/usr/bin/env python
"""
bw_category_map.py — category (RR x HR) evaluation of BW-extraction methods.

The end goal is category-driven analysis: split the RR x HR plane into cells and
let each cell choose its own extraction method. So we score each method PER CELL
(not by one global MAE), because a global average hides the edge cases that matter.

Grid (3x3, tunable):
    RR bins (bpm):  <12 | 12-24 | 24-50
    HR bins (bpm):  40-60 | 60-85 | >85     (HR from IR-PPG beats)
The collision corner is low-HR x high-RR (heart fundamental meets the RR band).

Per (method, cell) we accumulate, over all recordings, the fraction of
spectrogram-average RR samples within +/- tol bpm of the reference average, plus
the sample count. Then per method we report:
    * worst-case cell (min % over cells with enough samples),
    * how many cells passed (>= pass_thr) and which cells FAILED,
and a best-method-per-cell map.

Usage
-----
    py bw_category_map.py                          # all recordings, Artifact
    py bw_category_map.py --limit 4                # first 4 recordings (quick)
    py bw_category_map.py --recordings Exp1/001 Exp2/004
    py bw_category_map.py --methods MA Butter MAPAS MAPASref
    py bw_category_map.py --save DIR --no-show
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import dataclasses

import numpy as np

import bw_bp_sweep as S

RR_EDGES = [12.0, 30.0]           # -> bins: <12, 12-30, >30
HR_EDGES = [60.0, 85.0]           # -> bins: <60, 60-85, >=85
RR_LABELS = []                    # filled from edges in main()
HR_LABELS = []
REF_LEDS = ("Green", "Red", "IR")


def _mode_bin(vals, n_bins, edges):
    """Majority RR bin of `vals` given digitize `edges` (n_bins categories)."""
    bins = np.digitize(np.asarray(vals, np.float64), edges)
    return int(np.argmax(np.bincount(bins, minlength=n_bins)))


def _labels(edges, prefix):
    """Human labels for digitize bins defined by `edges` (len(edges)+1 bins)."""
    out = [f"{prefix}<{edges[0]:g}"]
    for i in range(len(edges) - 1):
        out.append(f"{prefix}{edges[i]:g}-{edges[i+1]:g}")
    out.append(f"{prefix}{edges[-1]:g}+")
    return out


def hr_series_from_beats(ppg_results, sig, offset):
    """Per-beat HR (bpm) on the REMbo clock, from the cleanest available LED."""
    for ch in ("IR", "Green", "Red"):
        res = ppg_results.get(ch)
        if res is not None and len(getattr(res, "ss_idx", []) or []) > 3:
            bt = np.asarray(sig.time)[np.asarray(res.ss_idx, dtype=int)] + offset
            dt = np.diff(bt)
            ok = dt > 0
            hr = 60.0 / dt[ok]
            ht = ((bt[:-1] + bt[1:]) / 2)[ok]
            good = (hr >= 35) & (hr <= 180)
            return ht[good], hr[good]
    return np.zeros(0), np.zeros(0)


def accumulate(rec_id, edf, csv, channel, methods, tol, specB_window, metric, raw, fundamental,
               detrend, detrend_cross, detrend_baseline, segment_sec, segment_min,
               tot, inb, run_ref, run_ppg, run_sync, cfg, deps, legacy_hamming=False,
               mapas_hrlock=False, butter_order=None, route_by="signal"):
    (Candidate, score_candidate, reference_average_rr, average_series,
     spectro_average, METHODS, REF_METHODS) = deps
    ref = run_ref(edf)
    ppg_results, sig = run_ppg(csv)
    offset = run_sync(edf, ppg_results, sig)
    truth_at, truth_ar = reference_average_rr(ref)
    if truth_at.size < 2 or (route_by == "signal" and channel not in sig.channels):
        print(f"  [skip] {rec_id}: no reference / channel")
        return
    hr_t, hr_v = hr_series_from_beats(ppg_results, sig, offset)
    hr_t_sig, hr_v_sig = hr_series_from_beats(ppg_results, sig, 0.0)   # signal clock (for HR-lock)
    if hr_t.size < 2:
        print(f"  [skip] {rec_id}: no HR from beats")
        return
    x = (np.asarray(sig.channels[channel], np.float64)
         if channel in sig.channels else None)     # unused in route_by="ref"
    fs = float(sig.fs)
    low, high = cfg.bw_band_low_hz, cfg.bw_band_high_hz
    avg_fn = lambda tt, rr: average_series(tt, rr, cfg)

    for name in methods:
        if route_by == "ref":
            # ---- route-by-REFERENCE: the reference median-average IS the routing
            # source (no signal extraction). The per-cell totals then show the
            # category SPLIT as the reference sees it; HR bins still come from the
            # PPG beats. Co-location is trivially ~100% here (our series == truth),
            # so read the SPLIT (segment counts per cell), not the % score.
            b_at, b_ar = np.asarray(truth_at), np.asarray(truth_ar)
            series_offset = 0.0                       # reference already on REMbo clock
        else:
            if name in REF_METHODS:
                refs = [np.asarray(sig.channels[c], np.float64)
                        for c in REF_LEDS if c in sig.channels and c != channel]
                if not refs:
                    continue
                bw = REF_METHODS[name](x, refs, fs, low, high)
            else:
                fn = METHODS.get(name)
                if fn is None:
                    continue
                if name == "MA":
                    bw = fn(x, fs, low, high, cfg.bw_filter_order)
                elif name == "Butter" and butter_order is not None:
                    bw = fn(x, fs, low, high, order=butter_order)   # Butterworth order sweep
                elif name == "MAPAS" and mapas_hrlock:
                    # lock the cardiac fundamental to the measured beat HR (signal clock)
                    from respiration_rr.ppg.bw_methods import mapas_hrlock as _mapas_hrlock
                    bw = _mapas_hrlock(x, fs, low, high, hr_t_sig, hr_v_sig)
                elif name == "Legacy":
                    # Legacy IS the detrend extraction — wire its own baseline + the
                    # optional pre-FFT Hamming window (this run's tuning knobs).
                    bw = fn(x, fs, low, high, baseline_sec=detrend_baseline,
                            use_hamming=legacy_hamming)
                else:
                    bw = fn(x, fs, low, high)
            use_fund = fundamental or (name == "Legacy")     # Legacy always uses fundamental ridge
            if detrend == "adaptive" and name != "Legacy":
                # fuse two ridges: plain (good at low RR) + detrended (good at high RR).
                # regime is read from the PLAIN estimate, which never over-reads, so a
                # plain value >= cross reliably means we are in the high-RR regime.
                from respiration_rr.ppg.bw_methods import legacy_detrend
                o_p = spectro_average(bw, fs, specB_window, cfg, avg_fn, fundamental=use_fund)
                o_d = spectro_average(legacy_detrend(bw, fs, low, high, baseline_sec=detrend_baseline),
                                      fs, specB_window, cfg, avg_fn, fundamental=use_fund)
                if o_p is None or o_d is None:
                    continue
                st, sr_p, _, _, prom = o_p
                _, sr_d, _, _, _ = o_d
                sr_f = np.where(np.isfinite(sr_p) & (sr_p >= detrend_cross), sr_d, sr_p)
                nanp = ~np.isfinite(sr_p)
                sr_f[nanp] = np.asarray(sr_d)[nanp]
                at_f, ar_f = avg_fn(st, sr_f)
                out = (st, sr_f, at_f, ar_f, prom)
            else:
                if detrend == "on" and name != "Legacy":     # envelope-detrend post-step (Legacy already has it)
                    from respiration_rr.ppg.bw_methods import legacy_detrend
                    bw = legacy_detrend(bw, fs, low, high, baseline_sec=detrend_baseline)
                out = spectro_average(bw, fs, specB_window, cfg, avg_fn, fundamental=use_fund)
            if out is None:
                continue
            _st, _sr, _at, _ar, _prom = out
            b_at, b_ar = (_st, _sr) if raw else (_at, _ar)     # raw ridge vs windowed-median
            b_at, b_ar = np.asarray(b_at), np.asarray(b_ar)
            series_offset = offset
        bt = b_at + series_offset
        n_rr = len(RR_EDGES) + 1

        if segment_sec and segment_sec > 0:
            # ---- SEGMENT-level: one vote per segment by MAJORITY category ----
            t0 = max(truth_at[0], hr_t[0])
            t1 = min(truth_at[-1], hr_t[-1])
            seg = t0
            while seg + segment_sec <= t1 + 1e-9:
                a0, a1 = seg, seg + segment_sec
                seg += segment_sec                            # non-overlapping
                rr_ref = truth_ar[(truth_at >= a0) & (truth_at < a1) & np.isfinite(truth_ar)]
                rr_ref = rr_ref[(rr_ref >= 4) & (rr_ref <= 55)]
                rr_our = b_ar[(bt >= a0) & (bt < a1) & np.isfinite(b_ar)]
                hseg = hr_v[(hr_t >= a0) & (hr_t < a1)]
                if rr_ref.size < segment_min or rr_our.size < segment_min or hseg.size < 1:
                    continue
                ref_bin = _mode_bin(rr_ref, n_rr, RR_EDGES)   # true category (majority)
                our_bin = _mode_bin(rr_our, n_rr, RR_EDGES)   # our category (majority)
                hb = int(np.digitize(np.median(hseg), HR_EDGES))
                tot[name][ref_bin, hb] += 1
                if our_bin == ref_bin:
                    inb[name][ref_bin, hb] += 1
            continue

        # ---- per-sample scoring ----
        m = (np.isfinite(b_ar) & (bt >= truth_at[0]) & (bt <= truth_at[-1])
             & (bt >= hr_t[0]) & (bt <= hr_t[-1]))
        if not m.any():
            continue
        ref_rr = np.interp(bt[m], truth_at, truth_ar)
        hr_at = np.interp(bt[m], hr_t, hr_v)
        rr_bin = np.digitize(ref_rr, RR_EDGES)      # reference RR bin (true category)
        hr_bin = np.digitize(hr_at, HR_EDGES)
        our_rr_bin = np.digitize(b_ar[m], RR_EDGES)  # OUR estimated RR bin
        err = np.abs(b_ar[m] - ref_rr)
        keep = (ref_rr >= 4) & (ref_rr <= 55)
        for i in np.where(keep)[0]:
            rb, hb = int(rr_bin[i]), int(hr_bin[i])
            tot[name][rb, hb] += 1
            # "coloc" = our RR routed to the same RR category as the reference;
            # "band"  = our RR within +/- tol bpm of the reference.
            success = (our_rr_bin[i] == rb) if metric == "coloc" else (err[i] <= tol)
            if success:
                inb[name][rb, hb] += 1
    print(f"  [ok]   {rec_id}")


def main(argv=None):
    global RR_EDGES, HR_EDGES, RR_LABELS, HR_LABELS
    ap = argparse.ArgumentParser(description="RR x HR category map of BW-extraction methods")
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--recordings", nargs="+", default=None, help="explicit rec ids")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N recordings")
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--methods", nargs="+",
                    default=["MA", "Butter", "MAPAS", "MAPASref", "MAPASnar", "Legacy"])
    ap.add_argument("--route-by", choices=("signal", "ref"), default="signal",
                    help="routing RR source: signal = BW-extraction methods (default, unchanged); "
                         "ref = the reference median-average itself (shows the reference category "
                         "SPLIT; HR bins still from PPG beats). ref ignores --methods/--channel.")
    ap.add_argument("--metric", choices=("coloc", "band"), default="coloc",
                    help="coloc = routed to same RR category; band = within +/- tol bpm")
    ap.add_argument("--tol", type=float, default=4.0)
    ap.add_argument("--pass-thr", type=float, default=70.0, help="cell passes if in-band %% >= this")
    ap.add_argument("--min-samples", type=int, default=1, help="min samples to trust a cell")
    ap.add_argument("--specB-window", type=float, default=16.0)
    ap.add_argument("--no-avg", action="store_true",
                    help="use the raw spectrogram ridge (skip the windowed-median smoothing)")
    ap.add_argument("--fundamental", action="store_true",
                    help="use fundamental-selecting ridge (harmonic-aware) instead of bare argmax")
    ap.add_argument("--rr-band-high", type=float, default=1.0,
                    help="upper RR search band (Hz) for the spectral ridge (1.0 = 60 bpm)")
    ap.add_argument("--avg-window", type=float, default=None, help="median-average window (s); needs no --no-avg")
    ap.add_argument("--avg-stride", type=float, default=None, help="median-average stride (s)")
    ap.add_argument("--avg-method", choices=("median", "mean"), default=None)
    ap.add_argument("--segment-sec", type=float, default=0.0,
                    help="segment length (s) for majority-vote category scoring; 0 = per-sample")
    ap.add_argument("--segment-min", type=int, default=3,
                    help="min RR samples (each side) in a segment to score it")
    ap.add_argument("--detrend", choices=("off", "on", "adaptive"), default="off",
                    help="envelope-detrend post-step (except Legacy): off | on (always) | "
                         "adaptive (fuse plain+detrend ridges by RR regime)")
    ap.add_argument("--detrend-cross", type=float, default=12.0,
                    help="adaptive detrend: plain RR (bpm) above which the detrended ridge is trusted "
                         "(set at the RR<12 bin edge: low RR keeps plain, everything else gets detrend)")
    ap.add_argument("--detrend-baseline", type=float, default=5.0,
                    help="detrend envelope-gate baseline window (s); legacy 5; longer helps slow breaths")
    ap.add_argument("--legacy-hamming", action="store_true",
                    help="Legacy only: apply the original FFTfilter's pre-FFT Hamming window")
    ap.add_argument("--mapas-hrlock", action="store_true",
                    help="MAPAS only: lock the cardiac fundamental to the measured beat HR "
                         "instead of the ART FFT peak")
    ap.add_argument("--butter-order", type=int, default=None,
                    help="Butter only: Butterworth order (2 = hand-rolled port; 3+ = scipy)")
    ap.add_argument("--rr-edges", type=float, nargs="+", default=RR_EDGES, help="RR bin edges (bpm)")
    ap.add_argument("--hr-edges", type=float, nargs="+", default=HR_EDGES, help="HR bin edges (bpm)")
    ap.add_argument("--out", default=r"C:\Users\RachelMizrahi\AppData\Local\Temp\bbbrr_bench\category_map")
    ap.add_argument("--save", default=None, metavar="DIR")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    RR_EDGES = list(args.rr_edges)
    HR_EDGES = list(args.hr_edges)
    RR_LABELS = _labels(RR_EDGES, "RR")
    HR_LABELS = _labels(HR_EDGES, "HR")
    n_rr, n_hr = len(RR_EDGES) + 1, len(HR_EDGES) + 1

    recs = S.discover(args.data_root)
    if args.recordings:
        recs = [r for r in recs if r[0] in args.recordings]
    if args.limit:
        recs = recs[:args.limit]
    if not recs:
        sys.exit("No recordings.")

    import matplotlib
    matplotlib.use("Agg" if args.no_show else "TkAgg")
    import matplotlib.pyplot as plt

    from respiration_rr.settings import PPG
    _avg = {}
    if args.avg_window is not None:
        _avg["rr_avg_window_sec"] = args.avg_window
    if args.avg_stride is not None:
        _avg["rr_avg_step_sec"] = args.avg_stride
    if args.avg_method is not None:
        _avg["rr_avg_method"] = args.avg_method
    cfg_run = dataclasses.replace(PPG, rr_band_high_hz=args.rr_band_high, **_avg)
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.compare.compare import Candidate, score_candidate
    from respiration_rr.rr_average import reference_average_rr, average_series
    from respiration_rr.ppg.bw_methods import METHODS, REF_METHODS
    from spectro_avg_lab import spectro_average
    deps = (Candidate, score_candidate, reference_average_rr, average_series,
            spectro_average, METHODS, REF_METHODS)

    methods = ["Reference"] if args.route_by == "ref" else args.methods
    tot = {m: np.zeros((n_rr, n_hr)) for m in methods}
    inb = {m: np.zeros((n_rr, n_hr)) for m in methods}

    metric_label = ("routed-correctly %" if args.metric == "coloc"
                    else f"within +/-{args.tol:.0f} bpm %")
    print(f"Category map | channel={args.channel} | {len(recs)} recordings | "
          f"metric={args.metric} ({metric_label}) | pass>={args.pass_thr:.0f}% | min_n={args.min_samples}")
    for rid, edf, csv in recs:
        try:
            accumulate(rid, edf, csv, args.channel, methods, args.tol, args.specB_window, args.metric,
                       args.no_avg, args.fundamental, args.detrend, args.detrend_cross,
                       args.detrend_baseline, args.segment_sec, args.segment_min,
                       tot, inb, run_reference, run_ppg, run_sync, cfg_run, deps,
                       legacy_hamming=args.legacy_hamming, mapas_hrlock=args.mapas_hrlock,
                       butter_order=args.butter_order, route_by=args.route_by)
        except Exception as e:
            print(f"  [ERR]  {rid}: {e}")

    # ---- per-method report ----
    pct = {}
    print("\n================  PER-METHOD (rows=RR, cols=HR)  ================")
    for m in methods:
        with np.errstate(invalid="ignore", divide="ignore"):
            p = 100.0 * inb[m] / tot[m]
        p[tot[m] < args.min_samples] = np.nan
        pct[m] = p
        valid = np.isfinite(p)
        worst = np.nanmin(p) if valid.any() else np.nan
        passed = int(np.sum(valid & (p >= args.pass_thr)))
        failed = [(RR_LABELS[i], HR_LABELS[j], p[i, j])
                  for i in range(n_rr) for j in range(n_hr)
                  if valid[i, j] and p[i, j] < args.pass_thr]
        na = int(np.sum(~valid))
        print(f"\n--- {m} ---   worst-case={worst:.0f}%   passed={passed}/{int(valid.sum())} cells"
              f"   (n/a={na})")
        print(f"    {'':9}" + "".join(f"{h:>10}" for h in HR_LABELS))
        for i in range(n_rr):
            cells = []
            for j in range(n_hr):
                cells.append(f"{int(inb[m][i,j])}/{int(tot[m][i,j])}")   # correct / total segments
            print(f"    {RR_LABELS[i]:9}" + "".join(f"{c:>10}" for c in cells))
        if failed:
            print("    FAILED: " + "; ".join(f"{r}x{h}={v:.0f}%" for r, h, v in failed))

    # ---- best method per cell ----
    print("\n================  BEST METHOD PER CELL  ================")
    best_map = np.empty((n_rr, n_hr), dtype=object)
    print(f"    {'':9}" + "".join(f"{h:>14}" for h in HR_LABELS))
    for i in range(3):
        row = []
        for j in range(3):
            cand = [(m, pct[m][i, j]) for m in methods if np.isfinite(pct[m][i, j])]
            if cand:
                bm, bv = max(cand, key=lambda z: z[1])
                best_map[i, j] = bm
                row.append(f"{bm}({bv:.0f}%)")
            else:
                best_map[i, j] = None
                row.append("-- (n/a)")
        print(f"    {RR_LABELS[i]:9}" + "".join(f"{c:>14}" for c in row))

    # ---- figure: per-method heatmaps ----
    ncol = len(methods)
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.8))
    if ncol == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        p = pct[m]
        ax.imshow(np.ma.masked_invalid(p), origin="lower", cmap="RdYlGn",
                  vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(n_hr)); ax.set_xticklabels(HR_LABELS, fontsize=7, rotation=30)
        ax.set_yticks(range(n_rr)); ax.set_yticklabels(RR_LABELS, fontsize=7)
        worst = np.nanmin(p) if np.isfinite(p).any() else np.nan
        ax.set_title(f"{m}\nworst {worst:.0f}%", fontsize=9)
        for i in range(n_rr):
            for j in range(n_hr):
                txt = f"{int(inb[m][i,j])}/{int(tot[m][i,j])}"          # correct / total segments
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.suptitle(f"BW methods — {metric_label} per RR x HR cell "
                 f"({args.channel}, metric={args.metric}, {len(recs)} recs)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_dir = args.save or (args.out if args.no_show else None)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        png = os.path.join(out_dir, f"category_map_{args.channel}.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print(f"\n  saved -> {png}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
