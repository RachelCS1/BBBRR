"""
Matplotlib figures mirroring the HTML tools' graphs.

  plot_reference     : airflow signal + breaths, RR, RRV, I:E   (Tool 2)
  plot_ppg_channel   : one channel's 4 respiration params + ridge (Tool 1)
  plot_ppg_overview  : mean RR per param x channel bar summary
  plot_comparison    : reference RR + best-N watch candidates      (Tool 3)
"""

import numpy as np
import matplotlib.pyplot as plt

_PARAM_COLORS = {"RSA": "#22c55e", "RIIV": "#38bdf8", "AUC": "#f472b6",
                 "LP": "#f97316", "BWlegacy": "#e11d48", "BWbank": "#14b8a6",
                 "Ridge": "#a855f7",
                 # cubic-spline variants (darker shade of each base param)
                 "RSA_spline": "#15803d", "RIIV_spline": "#0369a1",
                 "AUC_spline": "#be185d",
                 # smoothing-spline variants (distinct third shade)
                 "RSA_ssp": "#65a30d", "RIIV_ssp": "#0891b2",
                 "AUC_ssp": "#db2777"}


def _shade_noise(ax, regions, color="#ef4444", alpha=0.10):
    for (s, e) in regions or []:
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


def _in_noise_mask(times, regions):
    """Boolean mask: True where a time falls inside any (start, end) noise region."""
    times = np.asarray(times, np.float64)
    mask = np.zeros(times.size, dtype=bool)
    for (s, e) in regions or []:
        mask |= (times >= s) & (times <= e)
    return mask


def plot_reference(ref, title="Reference (REMbo / Poly) — airflow RR"):
    """Four-panel reference figure: signal+breaths, RR, RRV, I:E."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    t = ref.time

    # 1) filtered airflow + breath-start crossings
    ax = axes[0]
    ax.plot(t, ref.filtered, color="#2563eb", lw=0.6, label="Filtered airflow")
    for c in ref.crossings:
        ax.axvline(c, color="#f59e0b", lw=0.4, alpha=0.5)
    _shade_noise(ax, ref.merged_noise)
    ax.set_ylabel("Airflow (a.u.)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Nasal-pressure airflow + {len(ref.breaths)} detected breaths "
                 f"(orange = breath boundaries)", fontsize=9)

    # 2) breath-by-breath RR + optional device reference
    ax = axes[1]
    bx = [b.center for b in ref.breaths]
    br = [b.rate for b in ref.breaths]
    ax.plot(bx, br, "o-", color="#2563eb", ms=3, lw=0.8, label="Computed RR")
    if getattr(ref, "seg_breaths", None):
        sx = [b.center for b in ref.seg_breaths]
        sr = [b.rate for b in ref.seg_breaths]
        ax.plot(sx, sr, "^--", color="#16a34a", ms=3, lw=0.8, alpha=0.85,
                label="Computed RR (spectrogram segment-bandpass — display only)")
    if ref.ref_time is not None and len(ref.ref_time):
        ax.plot(ref.ref_time, ref.ref_rate, color="#ef4444", lw=0.8, alpha=0.7, label="Device RR")
    _shade_noise(ax, ref.merged_noise)
    ax.set_ylabel("RR (bpm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Respiration rate (breath-by-breath)", fontsize=9)

    # 3) RRV
    ax = axes[2]
    if ref.rrv_time.size:
        ax.plot(ref.rrv_time, ref.rrv_value, color="#3b82f6", lw=1.0)
    else:
        ax.text(0.5, 0.5, "Not enough breaths for RRV window", transform=ax.transAxes,
                ha="center", va="center", color="#888")
    ax.set_ylabel("RRV (bpm²)")
    ax.set_title("Respiration-rate variability (moving variance)", fontsize=9)

    # 4) I:E ratio
    ax = axes[3]
    iex = [b.center for b in ref.breaths if np.isfinite(b.ie_ratio)]
    iey = [b.ie_ratio for b in ref.breaths if np.isfinite(b.ie_ratio)]
    ax.plot(iex, iey, "o-", color="#8b5cf6", ms=3, lw=0.8)
    ax.axhline(0.5, color="#888", ls="--", lw=0.6, label="rest ≈ 0.5 (1:2)")
    ax.set_ylabel("I:E (inhale/exhale)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Inhale / exhale ratio", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ppg_channel(res, title=None, mae_by_param=None):
    """One PPG channel: the respiration-parameter envelopes with breath-starts.

    mae_by_param : optional {param_name: (mae, n)} to annotate each panel with its
    own MAE vs the reference (from compare.mae_by_candidate, filtered to this channel).
    """
    params = [p for p in ("RSA", "RSA_spline", "RSA_ssp", "RIIV", "RIIV_spline", "RIIV_ssp",
                          "AUC", "AUC_spline", "AUC_ssp", "LP", "BWlegacy", "BWbank")
              if p in res.params]
    n = len(params)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title or f"Watch PPG — {res.channel} channel — respiration parameters",
                 fontsize=13, fontweight="bold")

    for ax, pname in zip(axes, params):
        pr = res.params[pname]
        col = _PARAM_COLORS.get(pname, "#333")
        # envelope + breath-starts
        if pr.env_x.size:
            ax.plot(pr.env_x, pr.env_y, color=col, lw=0.8, label=f"{pname} envelope")
        if pr.bs_times.size:
            ymark = np.interp(pr.bs_times, pr.env_x, pr.env_y) if pr.env_x.size else np.zeros_like(pr.bs_times)
            in_noise = _in_noise_mask(pr.bs_times, res.move_regions)
            ax.plot(pr.bs_times[~in_noise], ymark[~in_noise], "v", color="#fbbf24", ms=6,
                    label="Breath start")
            if in_noise.any():                       # detected inside noise -> excluded from RR
                ax.plot(pr.bs_times[in_noise], ymark[in_noise], "v", mfc="none",
                        mec="#9ca3af", ms=6, label="Breath start (in noise, excluded)")
        _shade_noise(ax, res.move_regions)
        ax.set_ylabel(pname)
        mean_rr = np.nanmean(pr.rr_bpm) if pr.rr_bpm.size else float("nan")
        extra = ""
        if mae_by_param and pname in mae_by_param:
            mm, nn = mae_by_param[pname]
            extra = f" | MAE {mm:.2f} bpm (n={nn})"
        ax.set_title(f"{pname}: {pr.rr_bpm.size} breaths, mean RR = {mean_rr:.1f} bpm{extra}",
                     fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_spline_comparison(res, title=None, window=None, mae_by_param=None):
    """Per parameter (RSA / RIIV / AUC): overlay the final respiration-rate (RR)
    curve from each interpolation method's detected peaks:
      - linear+BP  (original: linear interp + band-pass)
      - spline     (interpolating cubic spline, no band-pass)
      - smoothing  (penalized smoothing spline, no band-pass)

    One panel per parameter that has the base "X" plus at least one spline
    variant. Each curve is RR (bpm) over time from that method's breath-start
    peaks — the graph that relies on the peaks each method found.
    mae_by_param : optional {param_name: (mae, n)} to annotate each line's MAE.
    """
    bases = [b for b in ("RSA", "RIIV", "AUC")
             if b in res.params and any(f"{b}_{v}" in res.params
                                        for v in ("spline", "ssp"))]
    if not bases:
        return None
    n = len(bases)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title or f"Watch PPG — {res.channel} — respiration rate (RR): "
                 f"linear+BP vs cubic spline vs smoothing spline (from each method's peaks)",
                 fontsize=13, fontweight="bold")

    def _mae(pname):
        if mae_by_param and pname in mae_by_param:
            mm, nn = mae_by_param[pname]
            return f", MAE {mm:.2f}"
        return ""

    def _line(ax, pr, name, style, label):
        if pr is not None and pr.rr_bpm.size:
            ax.plot(pr.rr_time, pr.rr_bpm, style, color=_PARAM_COLORS.get(name, "#333"),
                    ms=3, lw=0.9,
                    label=f"{label} ({pr.rr_bpm.size} br, {np.nanmean(pr.rr_bpm):.1f} bpm{_mae(name)})")

    for ax, base in zip(axes, bases):
        _line(ax, res.params.get(base), base, "o-", "linear+BP")
        _line(ax, res.params.get(f"{base}_spline"), f"{base}_spline", "s--", "spline")
        _line(ax, res.params.get(f"{base}_ssp"), f"{base}_ssp", "d:", "smoothing")
        _shade_noise(ax, res.move_regions)      # RR holes should line up with these
        ax.set_ylabel(f"{base}\nRR (bpm)")
        ax.set_title(f"{base} — RR from peaks: linear+BP (○ solid) vs spline (□ dashed) "
                     f"vs smoothing (◇ dotted)", fontsize=9, loc="left")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (s)")
    if window:
        axes[-1].set_xlim(*window)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ppg_spectrograms(res, title=None, max_bpm=54):
    """One PPG channel: the STFT spectrogram of EACH of the four RR parameters
    (RSA/RIIV/AUC/LP), with the extracted respiration ridge overlaid.

    Mirrors the HTML's RR-envelope spectrogram, but per parameter — four panels,
    frequency (bpm) on the y-axis, power (dB) as colour, time on the x-axis."""
    params = [p for p in ("RSA", "RSA_spline", "RSA_ssp", "RIIV", "RIIV_spline", "RIIV_ssp",
                          "AUC", "AUC_spline", "AUC_ssp", "LP")
              if p in res.params and res.params[p].spect is not None]
    if not params:
        return None
    n = len(params)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title or f"Watch PPG — {res.channel} — per-parameter spectrograms",
                 fontsize=13, fontweight="bold")
    for ax, pname in zip(axes, params):
        pr = res.params[pname]
        sp = pr.spect
        bpm = sp["freqs"] * 60.0
        m = bpm <= max_bpm
        pcm = ax.pcolormesh(sp["times"], bpm[m], sp["power_db"][m, :],
                            shading="auto", cmap="turbo")
        if pr.ridge_rr is not None and np.isfinite(pr.ridge_rr).any():
            ax.plot(pr.ridge_time, pr.ridge_rr, color="#ffffff", lw=1.1, alpha=0.85,
                    label="ridge")
            ax.legend(loc="upper right", fontsize=7)
            mean_ridge = np.nanmean(pr.ridge_rr)
        else:
            mean_ridge = float("nan")
        _shade_noise(ax, res.move_regions, alpha=0.18)
        ax.set_ylabel(f"{pname}\n(bpm)")
        ax.set_ylim(0, max_bpm)
        ax.set_title(f"{pname} spectrogram — ridge mean RR = {mean_ridge:.1f} bpm",
                     fontsize=9, loc="left")
        fig.colorbar(pcm, ax=ax, pad=0.01, label="dB")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ppg_overview(ppg_results):
    """Grouped bar chart: mean RR per parameter for each channel."""
    channels = list(ppg_results)
    params = ["RSA", "RIIV", "AUC", "LP"]
    for sp in ("RSA_spline", "RIIV_spline", "AUC_spline",
               "RSA_ssp", "RIIV_ssp", "AUC_ssp"):                     # only when produced
        if any(sp in ppg_results[ch].params for ch in channels):
            params.append(sp)
    if any("BWlegacy" in ppg_results[ch].params for ch in channels):   # only when produced
        params.append("BWlegacy")
    if any("BWbank" in ppg_results[ch].params for ch in channels):     # only when produced
        params.append("BWbank")
    params.append("Ridge")
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15 if len(params) <= 5 else 0.13
    xbase = np.arange(len(channels))
    for i, p in enumerate(params):
        vals = []
        for ch in channels:
            res = ppg_results[ch]
            if p == "Ridge":
                v = np.nanmean(res.ridge_rr) if res.ridge_rr is not None and np.isfinite(res.ridge_rr).any() else np.nan
            else:
                pr = res.params.get(p)
                v = np.nanmean(pr.rr_bpm) if pr and pr.rr_bpm.size else np.nan
            vals.append(v)
        ax.bar(xbase + i * width, vals, width, label=p, color=_PARAM_COLORS.get(p))
    ax.set_xticks(xbase + width * (len(params) - 1) / 2)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Mean RR (bpm)")
    ax.set_title("Watch PPG — mean respiration rate by parameter × channel", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_comparison(cmp, top_n=3, title="Cross-device comparison — watch vs REMbo reference"):
    """Reference RR + the best-N watch candidates (ranked by MAE)."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(cmp.ref_time, cmp.ref_rr, "o-", color="#ef4444", ms=3, lw=1.2,
            label="REMbo reference RR", zorder=5)
    colors = ["#22c55e", "#38bdf8", "#f472b6", "#f97316", "#a855f7"]
    for i, (cand, mae, n) in enumerate(cmp.ranked[:top_n]):
        ax.plot(cand.t + cmp.offset_sec, cand.rr, "o-", ms=2.5, lw=0.8,
                color=colors[i % len(colors)],
                label=f"{cand.label}  (MAE {mae:.2f} bpm, n={n})")
    ax.set_xlabel("Time (s, device clock)")
    ax.set_ylabel("RR (bpm)")
    ax.set_title(f"{title}\noffset = {cmp.offset_sec:+.0f} s", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_preprocess_stages(channel_name, stages, window=None, base_color="#22c55e"):
    """Stacked panels, one per preprocessing stage, for a single channel.

    window : optional (t0, t1) seconds to zoom all panels to a detail view.
    """
    n = len(stages)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.7 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Watch preprocessing — {channel_name} channel"
                 + (f"  (zoom {window[0]:.0f}–{window[1]:.0f}s)" if window else ""),
                 fontsize=13, fontweight="bold", color=base_color)
    for ax, st in zip(axes, stages):
        for tr in st["traces"]:
            ax.plot(tr["t"], tr["y"], color=tr["color"], lw=tr["lw"], alpha=tr["alpha"],
                    label=tr["label"])
        for mk in st.get("markers", []):
            ax.scatter(mk["t"], mk["y"], s=mk.get("size", 20), c=mk["color"],
                       marker=mk.get("marker", "o"), edgecolors="none",
                       label=mk["label"], zorder=5)
        ax.set_ylabel(st["ylabel"], fontsize=8)
        ax.set_title(st["title"], fontsize=9, loc="left")
        if len(st["traces"]) + len(st.get("markers", [])) > 1:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.15)
    axes[-1].set_xlabel("Time (s)")
    if window:
        axes[-1].set_xlim(*window)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_movement_preprocess(mv, window=None):
    """Accelerometer axes + jerk energy + movement threshold."""
    if mv is None:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    fig.suptitle("Watch preprocessing — accelerometer / movement", fontsize=13, fontweight="bold")
    ax = axes[0]
    ax.plot(mv["t"], mv["ax"], lw=0.5, label="acc_x", color="#ef4444")
    ax.plot(mv["t"], mv["ay"], lw=0.5, label="acc_y", color="#22c55e")
    ax.plot(mv["t"], mv["az"], lw=0.5, label="acc_z", color="#2563eb")
    ax.set_ylabel("accel (ADC)", fontsize=8)
    ax.set_title("0 · Accelerometer axes (upsampled, trimmed)", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.15)
    ax = axes[1]
    ax.plot(mv["t"], mv["energy"], lw=0.7, color="#f59e0b", label="jerk energy")
    ax.axhline(mv["threshold"], color="#ef4444", ls="--", lw=1.0,
               label=f"threshold {mv['threshold']:.0f} g/s")
    ax.set_ylabel("g/s", fontsize=8)
    ax.set_title("1 · Jerk energy √(dx²+dy²+dz²)·fs + movement threshold", fontsize=9, loc="left")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.15)
    axes[-1].set_xlabel("Time (s)")
    if window:
        axes[-1].set_xlim(*window)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_mae_overview(cmp, title="All watch candidates — MAE vs reference"):
    """Horizontal bar of EVERY scored candidate's MAE, sorted best-first.

    One figure to see all final results at a glance (not just the top N). Bars are
    coloured by parameter; the value and overlap count are printed on each bar."""
    if cmp is None or not cmp.ranked:
        return None
    labels = [c.label for (c, mae, n) in cmp.ranked]
    maes = np.array([mae for (c, mae, n) in cmp.ranked])
    ns = [n for (c, mae, n) in cmp.ranked]
    cols = [_PARAM_COLORS.get(c.param, "#888") for (c, mae, n) in cmp.ranked]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.34 * len(labels))))
    ax.barh(y, maes, color=cols)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()                                    # best (lowest MAE) on top
    for yi, (m, nn) in enumerate(zip(maes, ns)):
        ax.text(m, yi, f" {m:.2f} (n={nn})", va="center", fontsize=7)
    ax.margins(x=0.12)
    ax.set_xlabel("MAE (bpm)")
    ax.set_title(f"{title}\noffset = {cmp.offset_sec:+.0f} s  ·  {len(labels)} candidates",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def show_all():
    plt.show()
