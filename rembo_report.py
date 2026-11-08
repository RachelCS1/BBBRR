#!/usr/bin/env python
"""
rembo_report.py - single-night HTML review of a REMbo (TEMEC PSG) .edf recording.

Shows, over one shared time axis:
  * raw nasal-pressure AIRFLOW with the breath boundaries the detector found
  * RR from those breaths (per-breath BBBRR dots + windowed-median curve), airflow
    vs chest-effort
  * SpO2 (with desaturation markers)
  * pulse rate (HR) from the PPG Pulse Wave, with irregular-beat markers (a PPG-based
    IRREGULAR-PULSE screen - not an ECG arrhythmia diagnosis; this file has no ECG)
  * snore envelope and body position

Breath detection + RR reuse the project's OWN validated reference pipeline
(analyze_reference + reference_average_rr), so this matches how the pipeline reads
REMbo airflow. STANDALONE - imports read-only, writes nothing in the repo.

Usage
-----
    py rembo_report.py --edf "C:\\path\\REMbo_Rec.edf"
    py rembo_report.py --edf <path> --out <dir> --no-inline-plotly
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from respiration_rr.io.edf_reader import EDFReader
from respiration_rr.reference.airflow import analyze_reference
from respiration_rr.compare.compare import reference_rr_series
from respiration_rr.rr_average import reference_average_rr
from dreamt_night_report import rr_dynamics, smoothed_excursions   # shared RR-dynamics analysis

DISP_HZ = 8.0            # display rate for the dense waveforms
RR_BINS = {"slow <12": (0, 12), "normal 12-20": (12, 20), "elevated 20-30": (20, 30), "fast 30+": (30, 1e9)}


def dj(x):              # value arrays -> rounded lists
    return [None if not np.isfinite(v) else round(float(v), 3) for v in np.asarray(x, float)]


def dh(x):              # time arrays (hours) -> 6 decimals (ms precision; 3 would flatten the wave)
    return [None if not np.isfinite(v) else round(float(v), 6) for v in np.asarray(x, float)]


def _dec_to(t, sig, fs, hz=DISP_HZ):
    step = max(1, int(round(fs / hz)))
    return t[::step], sig[::step]


def _norm(sig):
    sig = np.asarray(sig, float)
    sig = sig - np.nanmean(sig)
    sd = np.nanstd(sig) or 1.0
    return sig / sd


def _rr_curves(t, sig, fs):
    ref = analyze_reference(t, sig, fs)
    bt, br = reference_rr_series(ref)
    at, ar = reference_average_rr(ref)
    onset = np.array([b.start for b in ref.breaths if np.isfinite(b.start)], float)
    return bt, br, at, ar, onset


def _pulse(t, pw, fs):
    """PPG beats -> instantaneous HR + irregular-beat times (|dIBI|>120 ms).
    Validated against the device PR channel (MAE ~3 bpm)."""
    b, a = butter(2, [0.6 / (fs / 2), 3.5 / (fs / 2)], "band")
    pwf = filtfilt(b, a, pw)
    pk, _ = find_peaks(pwf, distance=int(fs * 0.5), prominence=0.4 * np.std(pwf))
    tp = t[pk]
    ibi = np.diff(tp)
    tbeat = tp[1:]
    ok = (ibi > 0.4) & (ibi < 1.5)                       # 40-150 bpm plausible
    tbeat, ibi = tbeat[ok], ibi[ok]
    hr = 60.0 / ibi
    dibi = np.abs(np.diff(ibi))
    irr = tbeat[1:][dibi > 0.12]                         # beat-to-beat jump > 120 ms
    return tbeat, hr, irr, float(np.mean(dibi > 0.12) * 100.0)


def _snore_env(t, sig, fs):
    env = np.abs(np.asarray(sig, float))
    w = max(1, int(fs * 0.5))
    env = np.convolve(env, np.ones(w) / w, "same")
    return _dec_to(t, env, fs, 4.0)


def _desats(t, spo2, drop=3.0, win_s=120.0):
    """Nadir times of >=drop% dips below a rolling-max baseline (ODI-style)."""
    import pandas as pd
    s = pd.Series(spo2)
    base = s.rolling(int(win_s), min_periods=5, center=True).max().to_numpy()
    dip = (base - spo2) >= drop
    out = []
    i, n = 0, len(spo2)
    while i < n:
        if dip[i]:
            j = i
            while j < n and dip[j]:
                j += 1
            k = i + int(np.argmin(spo2[i:j]))
            out.append(t[k]); i = j
        else:
            i += 1
    return np.array(out)


def build(edf_path):
    r = EDFReader(edf_path)
    labels = set(r.channel_labels())

    def ch(label):
        return r.read_channel(label)

    # --- airflow + breaths + RR ---
    ta, air, fsa = ch("Nasal P")
    bt, br, at_a, ar_a, onset = _rr_curves(ta, air, fsa)
    wt, wf = _dec_to(ta, air, fsa)                        # display waveform
    wf = _norm(wf)

    # effort RR (chest belt)
    at_e = ar_e = np.zeros(0)
    if "Chest" in labels:
        te, chest, fse = ch("Chest")
        _, _, at_e, ar_e, _ = _rr_curves(te, chest, fse)

    # --- SpO2 ---
    tsp, spo2, fssp = ch("SpO2")
    spo2 = np.where((spo2 >= 40) & (spo2 <= 100), spo2, np.nan)
    desat_t = _desats(tsp, spo2)
    tsp_d, spo2_d = _dec_to(tsp, spo2, fssp, 1.0)

    # --- pulse / HR + irregularity (PPG) ---
    hr_t = hr_v = irr_t = np.zeros(0); irr_pct = float("nan")
    if "Pulse Wave" in labels:
        tp, pw, fsp = ch("Pulse Wave")
        hr_t, hr_v, irr_t, irr_pct = _pulse(tp, pw, fsp)

    # --- snore ---
    sn_t = sn_v = np.zeros(0)
    if "NP Snore" in labels:
        tsn, snore, fssn = ch("NP Snore")
        sn_t, sn_v = _snore_env(tsn, snore, fssn)

    # --- body position ---
    bd_t = bd_v = np.zeros(0)
    if "Body" in labels:
        tb, body, fsb = ch("Body")
        bd_t, bd_v = _dec_to(tb, body, fsb, 1.0)

    dur_h = float(ta[-1] / 3600.0)
    brf = br[np.isfinite(br)]
    dyn = rr_dynamics(bt, br, at_a, ar_a)
    exc = smoothed_excursions(at_a, ar_a, [], win=30.0)        # no scored events in REMbo
    bins = ({k: round(float(np.mean((brf >= lo) & (brf < hi)) * 100.0), 1) for k, (lo, hi) in RR_BINS.items()}
            if brf.size else {})
    stats = {
        "dur_h": round(dur_h, 2), "n_breaths": int(onset.size),
        "rr_median": round(float(np.median(brf)), 1) if brf.size else None,
        "rr_med": round(float(np.median(brf)), 1) if brf.size else None,     # alias (HTML cards)
        "rr_std": round(float(np.std(brf)), 1) if brf.size else None,
        "rr_iqr": [round(float(np.percentile(brf, 25)), 1), round(float(np.percentile(brf, 75)), 1)] if brf.size else [None, None],
        "rr_bin_pct": bins, "dyn": dyn, "exc": exc,
        "spo2_mean": round(float(np.nanmean(spo2)), 1), "spo2_min": round(float(np.nanmin(spo2)), 1),
        "n_desat": int(desat_t.size), "odi": round(desat_t.size / dur_h, 1),
        "hr_med": round(float(np.median(hr_v)), 0) if hr_v.size else None,
        "irr_pct": round(irr_pct, 1) if np.isfinite(irr_pct) else None,
        "n_irr": int(irr_t.size),
    }
    data = {
        "name": os.path.splitext(os.path.basename(edf_path))[0], "stats": stats,
        "wave": {"t": dh(wt / 3600.0), "y": dj(wf), "onset_t": dh(onset / 3600.0), "n_breaths": int(onset.size)},
        "rr": {"bt": dh(bt / 3600.0), "br": dj(br), "air_t": dh(at_a / 3600.0), "air": dj(ar_a),
               "eff_t": dh(at_e / 3600.0), "eff": dj(ar_e)},
        "rr_series": {"bt": dh(bt / 3600.0), "br": dj(br), "at": dh(at_a / 3600.0), "ar": dj(ar_a)},
        "spo2": {"t": dh(tsp_d / 3600.0), "v": dj(spo2_d), "desat_t": dh(desat_t / 3600.0)},
        "hr": {"t": dh(hr_t / 3600.0), "v": dj(hr_v), "irr_t": dh(irr_t / 3600.0)},
        "snore": {"t": dh(sn_t / 3600.0), "v": dj(sn_v)},
        "body": {"t": dh(bd_t / 3600.0), "v": dj(bd_v)},
    }
    return data


def _page_html(data):
    s = data["stats"]
    cards = [
        ("משך הקלטה", f"{s['dur_h']} ש'"), ("נשימות", f"{s['n_breaths']}"),
        ("RR חציון", f"{s['rr_med']} [{s['rr_iqr'][0]}-{s['rr_iqr'][1]}]"),
        ("SpO2 ממוצע", f"{s['spo2_mean']}%"), ("SpO2 מינ'", f"{s['spo2_min']}%"),
        ("ODI", f"{s['odi']}/ש'"), ("דופק חציון", f"{s['hr_med']} bpm"),
        ("פעימות לא-סדירות", f"{s['n_irr']} ({s['irr_pct']}%)"),
    ]
    card_html = "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cards)
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REMbo {data['name']} — סקירת לילה</title>
<script src="plotly.min.js"></script>
<script>if(typeof Plotly==="undefined"){{var sc=document.createElement("script");
  sc.src="https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js";document.head.appendChild(sc);}}</script>
<style>
  :root {{ --bg:#fff; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#f8fafc; }}
  @media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0b1220;--fg:#e5e7eb;--muted:#94a3b8;--line:#1e293b;--card:#111a2e;}}}}
  body {{ background:var(--bg); color:var(--fg); font-family:system-ui,Segoe UI,Arial,sans-serif; margin:0; padding:16px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); font-size:13px; margin-bottom:12px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px; margin-bottom:14px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px 10px; }}
  .card .k {{ color:var(--muted); font-size:12px; }} .card .v {{ font-size:17px; font-weight:600; margin-top:2px; }}
  #plot {{ width:100%; height:950px; }}
  .note {{ color:var(--muted); font-size:12px; margin-top:8px; line-height:1.5; }}
</style></head>
<body>
  <h1>REMbo · {data['name']}</h1>
  <div class="sub">אות גולמי מהקלטת REMbo · חלוקה לנשימות · RR מהנשימות · דופק ואי-סדירות (PPG) · SpO2 · נחירה · תנוחה</div>
  <div class="cards">{card_html}</div>
  <div id="plot"></div>
  <div class="note">פאנל עליון = זרימת אוויר (Nasal P) <b>גולמית</b> עם <b>פסים אדומים</b> שמפרידים בין הנשימות — זַמְני כדי לאמת את החלוקה.
  RR כחול = אוויר, כתום מקווקו = מאמץ חזה, נקודות תכלת = BBBRR פר-נשימה. משולשים אדומים על SpO2 = ירידות סטורציה.
  פאנל דופק: קו = HR מה-PPG, נקודות כתומות = <b>פעימות לא-סדירות</b> (קפיצת מרווח >120ms) — <b>סקר אי-סדירות מבוסס-PPG, לא אבחון אריתמיה (אין ECG בקובץ)</b>.</div>
<script>
const D = {json.dumps(data)};
const sepX=[], sepY=[]; for(const t of D.wave.onset_t){{ sepX.push(t,t,null); sepY.push(-100,100,null); }}
// desat/irregular marker y = nearest value on their panel's line
function nearY(ts, tarr, varr){{ const out=[]; for(const x of ts){{ let bi=0,bd=1e9; for(let i=0;i<tarr.length;i++){{let d=Math.abs(tarr[i]-x); if(d<bd){{bd=d;bi=i;}}}} out.push(varr[bi]); }} return out; }}
const dy = nearY(D.spo2.desat_t, D.spo2.t, D.spo2.v);
const iy = nearY(D.hr.irr_t, D.hr.t, D.hr.v);
const traces = [
  {{x:D.wave.t, y:D.wave.y, name:"זרימת אוויר (גולמי)", mode:"lines", line:{{color:"#0f766e",width:0.9}}, type:"scattergl", yaxis:"y6", xaxis:"x"}},
  {{x:sepX, y:sepY, name:"הפרדת נשימות ("+D.wave.n_breaths+")", mode:"lines", line:{{color:"#dc2626",width:0.7}}, type:"scattergl", yaxis:"y6", xaxis:"x", hoverinfo:"skip"}},
  {{x:D.rr.bt, y:D.rr.br, name:"BBBRR (פר-נשימה)", mode:"markers", marker:{{color:"#93c5fd",size:3,opacity:0.55}}, type:"scattergl", yaxis:"y", xaxis:"x"}},
  {{x:D.rr.air_t, y:D.rr.air, name:"RR אוויר", mode:"lines", line:{{color:"#2563eb",width:1.6}}, yaxis:"y", xaxis:"x"}},
  {{x:D.rr.eff_t, y:D.rr.eff, name:"RR מאמץ", mode:"lines", line:{{color:"#f59e0b",width:1.3,dash:"dot"}}, yaxis:"y", xaxis:"x"}},
  {{x:D.spo2.t, y:D.spo2.v, name:"SpO2 %", mode:"lines", line:{{color:"#0ea5e9",width:1.2}}, yaxis:"y2", xaxis:"x"}},
  {{x:D.spo2.desat_t, y:dy, name:"דה-סטורציה", mode:"markers", marker:{{color:"#dc2626",size:6,symbol:"triangle-down"}}, yaxis:"y2", xaxis:"x", hoverinfo:"x"}},
  {{x:D.hr.t, y:D.hr.v, name:"דופק (PPG)", mode:"lines", line:{{color:"#b91c1c",width:1}}, type:"scattergl", yaxis:"y3", xaxis:"x"}},
  {{x:D.hr.irr_t, y:iy, name:"פעימה לא-סדירה", mode:"markers", marker:{{color:"#f97316",size:5,symbol:"circle"}}, type:"scattergl", yaxis:"y3", xaxis:"x", hoverinfo:"x"}},
  {{x:D.snore.t, y:D.snore.v, name:"נחירה", mode:"lines", line:{{color:"#16a34a",width:1}}, type:"scattergl", yaxis:"y4", xaxis:"x"}},
  {{x:D.body.t, y:D.body.v, name:"תנוחת גוף", mode:"lines", line:{{color:"#64748b",width:1.2,shape:"hv"}}, type:"scattergl", yaxis:"y5", xaxis:"x"}},
];
const layout = {{
  paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)", font:{{color:getComputedStyle(document.body).color}},
  margin:{{l:55,r:20,t:10,b:40}}, hovermode:"x unified", legend:{{orientation:"h", y:1.05}},
  grid:{{rows:6, columns:1, pattern:"independent", roworder:"top to bottom"}},
  xaxis:{{domain:[0,1], anchor:"y5", rangeslider:{{visible:true, thickness:0.04}}, title:"זמן (שעות)"}},
  yaxis6:{{domain:[0.855,1.0], title:"אוויר", zeroline:false, autorange:true}},
  yaxis:{{domain:[0.69,0.84], title:"RR (bpm)", rangemode:"tozero"}},
  yaxis2:{{domain:[0.53,0.67], title:"SpO2 %"}},
  yaxis3:{{domain:[0.37,0.51], title:"דופק bpm", rangemode:"tozero"}},
  yaxis4:{{domain:[0.19,0.33], title:"נחירה"}},
  yaxis5:{{domain:[0.0,0.14], title:"תנוחה"}},
}};
function draw(){{
  var gd=document.getElementById("plot");
  Plotly.newPlot(gd, traces, layout, {{responsive:true, displaylogo:false}}).then(function(){{
    gd.on("plotly_relayout", function(ev){{
      if(!("xaxis.range[0]" in ev || "xaxis.range" in ev || "xaxis.autorange" in ev)) return;
      var r=gd.layout.xaxis.range; if(!r) return;
      var lo=r[0], hi=r[1], mn=Infinity, mx=-Infinity;
      for(var i=0;i<D.wave.t.length;i++){{ var t=D.wave.t[i];
        if(t>=lo && t<=hi){{ var v=D.wave.y[i]; if(v!=null){{ if(v<mn)mn=v; if(v>mx)mx=v; }} }} }}
      if(mn<mx){{ var pad=0.12*(mx-mn)||1; Plotly.relayout(gd,{{"yaxis6.range":[mn-pad,mx+pad]}}); }}
    }});
  }});
}}
(function w(){{ if(typeof Plotly!=="undefined"){{draw();}} else {{setTimeout(w,100);}} }})();
</script>
</body></html>"""


def _inline_plotly(html, out_dir):
    pj = os.path.join(out_dir, "plotly.min.js")
    if not os.path.exists(pj):                 # reuse the DREAMT copy if present
        alt = os.path.join(os.environ.get("LOCALAPPDATA", ""), "BBBRR_DREAMT_report", "plotly.min.js")
        if os.path.exists(alt):
            pj = alt
    if os.path.exists(pj):
        js = open(pj, encoding="utf-8").read()
        return html.replace('<script src="plotly.min.js"></script>', f"<script>{js}</script>")
    return html


def _light(data):
    return {"sid": data["name"], "source": "REMbo", "dur_h": data["stats"]["dur_h"],
            "stats": data["stats"], "rr_series": data["rr_series"]}


def _summary_line(s):
    return (f"breaths={s['n_breaths']} RR={s['rr_median']} SpO2 {s['spo2_mean']}/{s['spo2_min']} "
            f"HR={s['hr_med']} irr={s['n_irr']}({s['irr_pct']}%) desats={s['n_desat']}")


def main(argv=None):
    import glob
    ap = argparse.ArgumentParser(description="REMbo .edf report: single-night HTML (--edf) or batch metrics (--folder)")
    ap.add_argument("--edf", default=None, help="one .edf -> full interactive HTML")
    ap.add_argument("--folder", default=None, help="a folder of .edf -> light metrics JSON per file (for cross-source conclusions)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-inline-plotly", action="store_true")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(os.environ.get("LOCALAPPDATA") or ".", "BBBRR_REMbo_report")
    os.makedirs(out, exist_ok=True)

    if args.folder:
        edfs = sorted(set(glob.glob(os.path.join(args.folder, "*.edf")) + glob.glob(os.path.join(args.folder, "*.EDF"))))
        print(f"{len(edfs)} EDF files in {args.folder}", flush=True)
        for i, e in enumerate(edfs):
            sid = os.path.splitext(os.path.basename(e))[0]
            jp = os.path.join(out, f"{sid}.json")
            if args.skip_existing and os.path.exists(jp):
                try:
                    if "exc" in json.load(open(jp, encoding="utf-8")).get("stats", {}):
                        print(f"[{i+1}/{len(edfs)}] {sid} skip (cached)", flush=True); continue
                except Exception:
                    pass
            try:
                print(f"[{i+1}/{len(edfs)}] {sid} ...", flush=True)
                data = build(e)
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(_light(data), f)
                print(f"    {_summary_line(data['stats'])}", flush=True)
            except Exception as ex:
                import traceback
                print(f"[{sid}] ERROR: {ex}"); traceback.print_exc()
        print(f"\ndone -> {out}")
        return

    if not args.edf:
        ap.error("pass --edf <file> or --folder <dir>")
    print(f"reading {args.edf} ...", flush=True)
    data = build(args.edf)
    print(f"  {_summary_line(data['stats'])}")
    html = _page_html(data)
    if not args.no_inline_plotly:
        html = _inline_plotly(html, out)
    with open(os.path.join(out, f"{data['name']}.json"), "w", encoding="utf-8") as f:
        json.dump(_light(data), f)
    p = os.path.join(out, f"{data['name']}.html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {p}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
