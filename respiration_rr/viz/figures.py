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
                 "LP": "#f97316", "Ridge": "#a855f7"}


def _shade_noise(ax, regions, color="#ef4444", alpha=0.10):
    for (s, e) in regions or []:
        ax.axvspan(s, e, color=color, alpha=alpha, lw=0)


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


def plot_ppg_channel(res, title=None):
    """One PPG channel: the four respiration params, each with its RR trend."""
    params = [p for p in ("RSA", "RIIV", "AUC", "LP") if p in res.params]
    n = len(params) + 1
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
            ax.plot(pr.bs_times, ymark, "v", color="#fbbf24", ms=6, label="Breath start")
        ax.set_ylabel(pname)
        mean_rr = np.nanmean(pr.rr_bpm) if pr.rr_bpm.size else float("nan")
        ax.set_title(f"{pname}: {pr.rr_bpm.size} breaths, mean RR = {mean_rr:.1f} bpm",
                     fontsize=9)
        ax.legend(loc="upper right", fontsize=8)

    # spectral ridge panel
    ax = axes[-1]
    if res.ridge_rr is not None and np.isfinite(res.ridge_rr).any():
        ax.plot(res.ridge_time, res.ridge_rr, color=_PARAM_COLORS["Ridge"], lw=1.0)
        ax.set_title(f"Spectral ridge: mean RR = {np.nanmean(res.ridge_rr):.1f} bpm", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Ridge unavailable (series too short)", transform=ax.transAxes,
                ha="center", va="center", color="#888")
    ax.set_ylabel("Ridge RR")
    ax.set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_ppg_overview(ppg_results):
    """Grouped bar chart: mean RR per parameter for each channel."""
    channels = list(ppg_results)
    params = ["RSA", "RIIV", "AUC", "LP", "Ridge"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.15
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
    ax.set_xticks(xbase + width * 2)
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


def show_all():
    plt.show()
