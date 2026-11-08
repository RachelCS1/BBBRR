#!/usr/bin/env python
"""
air_effort_compare.py - show, on the RAW data, a window where the AIRFLOW breath
detection and the EFFORT-belt breath detection DISAGREE (one over/under-segments),
which the single-channel RR then reads as "irregularity".

Two stacked panels over one time axis: raw airflow + its breath marks, raw effort
belt + its breath marks. When the airflow has ~2x the marks of the effort (or vice
versa), the per-breath RR is a detection artifact, not real irregular breathing.

Usage
-----
  REMbo:  py air_effort_compare.py --edf "<file.edf>" --t0 4.54 --t1 4.58
  DREAMT: py air_effort_compare.py --csv "<Sxxx_PSG_df.csv>" --t0 3.19 --t1 3.22
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from respiration_rr.reference.airflow import analyze_reference
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

OUT = os.path.join(os.environ.get("LOCALAPPDATA") or ".", "BBBRR_REMbo_report")


def _norm(x):
    x = np.asarray(x, float); x = x - np.nanmean(x)
    return x / (np.nanstd(x) or 1.0)


def _onsets(t, sig, fs):
    ref = analyze_reference(t, sig, fs)
    return np.array([b.start for b in ref.breaths if np.isfinite(b.start)], float)


def load_rembo(path):
    from respiration_rr.io.edf_reader import EDFReader
    r = EDFReader(path)
    ta, air, fa = r.read_channel("Nasal P")
    te, eff, fe = r.read_channel("Chest")
    return ("Nasal P (אוויר)", ta, air, fa), ("Chest (מאמץ)", te, eff, fe)


def load_dreamt(path):
    import pandas as pd
    from dreamt_night_report import _num, _block_avg, DECIM_HZ
    df = pd.read_csv(path, usecols=["TIMESTAMP", "FLOW", "THORAX"])
    t = np.asarray(_num(df["TIMESTAMP"]), float)
    fs = 1.0 / np.median(np.diff(t)); dec = max(1, int(round(fs / DECIM_HZ))); fs_d = fs / dec
    td = t[::dec]
    air = _block_avg(_num(df["FLOW"]), dec); eff = _block_avg(_num(df["THORAX"]), dec)
    n = min(len(td), len(air), len(eff))
    return ("FLOW (אוויר)", td[:n], air[:n], fs_d), ("THORAX (מאמץ)", td[:n], eff[:n], fs_d)


def panel(label, t, sig, fs, t0, t1):
    on = _onsets(t, sig, fs)                       # detect on full signal (needs context)
    m = (t >= t0 * 3600) & (t <= t1 * 3600)
    th = t[m] / 3600.0
    y = _norm(sig[m])
    onw = on[(on >= t0 * 3600) & (on <= t1 * 3600)] / 3600.0
    n_win = int(onw.size)
    return dict(label=label, t=[round(float(v), 6) for v in th], y=[round(float(v), 3) for v in y],
                onset=[round(float(v), 6) for v in onw], n=n_win)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf"); ap.add_argument("--csv")
    ap.add_argument("--t0", type=float, required=True); ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.edf:
        (la, ta, sa, fa), (le, te, se, fe) = load_rembo(a.edf); name = os.path.splitext(os.path.basename(a.edf))[0]
    else:
        (la, ta, sa, fa), (le, te, se, fe) = load_dreamt(a.csv); name = os.path.basename(a.csv).split("_")[0]
    P = [panel(la, ta, sa, fa, a.t0, a.t1), panel(le, te, se, fe, a.t0, a.t1)]
    print(f"{name} [{a.t0}-{a.t1}h]: {P[0]['label']} {P[0]['n']} breaths vs {P[1]['label']} {P[1]['n']} breaths "
          f"(ratio {P[0]['n']/max(P[1]['n'],1):.2f})")
    pj = os.path.join(os.environ.get("LOCALAPPDATA", ""), "BBBRR_DREAMT_report", "plotly.min.js")
    pljs = open(pj, encoding="utf-8").read() if os.path.exists(pj) else ""
    D = json.dumps({"name": name, "t0": a.t0, "t1": a.t1, "panels": P})
    html = f"""<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>אוויר מול מאמץ · {name}</title>
<script>{pljs}</script>
<style>:root{{--bg:#fff;--fg:#0f172a;--muted:#64748b}}@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#0b1220;--fg:#e5e7eb;--muted:#94a3b8}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Arial;margin:0;padding:14px}}h1{{font-size:18px;margin:0 0 2px}}.sub{{color:var(--muted);font-size:13px;margin-bottom:8px}}#p{{width:100%;height:520px}}</style></head>
<body><h1>אוויר מול מאמץ · {name}</h1>
<div class="sub">אותו חלון זמן. פסים אדומים = נשימות שהגלאי מצא בכל ערוץ. אם לאוויר הרבה יותר פסים מהמאמץ — ה"אי-סדירות" היא שגיאת-זיהוי, לא נשימה אמיתית.</div>
<div id="p"></div>
<script>const D={D};
const tr=[]; const yax=["y","y2"]; const col=["#0f766e","#7c3aed"];
D.panels.forEach((P,i)=>{{
 tr.push({{x:P.t,y:P.y,name:P.label+" ("+P.n+" נשימות)",mode:"lines",line:{{color:col[i],width:1}},type:"scattergl",yaxis:yax[i],xaxis:"x"}});
 const sx=[],sy=[]; for(const o of P.onset){{sx.push(o,o,null);sy.push(-3,3,null);}}
 tr.push({{x:sx,y:sy,mode:"lines",line:{{color:"#dc2626",width:0.8}},type:"scattergl",yaxis:yax[i],xaxis:"x",showlegend:false,hoverinfo:"skip"}});
}});
const lay={{paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{{color:getComputedStyle(document.body).color}},
 margin:{{l:40,r:14,t:6,b:34}},legend:{{orientation:"h",y:1.08}},hovermode:"x unified",
 grid:{{rows:2,columns:1,pattern:"independent",roworder:"top to bottom"}},
 xaxis:{{domain:[0,1],anchor:"y2",title:"זמן (שעות)",range:[D.t0,D.t1]}},
 yaxis:{{domain:[0.54,1],title:D.panels[0].label,zeroline:false}},
 yaxis2:{{domain:[0,0.46],title:D.panels[1].label,zeroline:false}}}};
(function w(){{if(typeof Plotly!=="undefined"){{Plotly.newPlot("p",tr,lay,{{displaylogo:false,responsive:true}});}}else setTimeout(w,100);}})();
</script></body></html>"""
    out = a.out or os.path.join(OUT, f"cmp_{name}_{a.t0}-{a.t1}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
