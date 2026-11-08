#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Interactive review of the noise / red-zone detection (noise_zones.py) for one recording.
Top: raw REMbo airflow with breath boundaries + red zones. Below: raw watch channels (Artifact
with its respiration envelope + collapse threshold) + the SAME red zones, all on the REMbo clock.
Shared pan (drag) / zoom (wheel); double-click resets. Header shows % of time on noise.

    py noise_viewer.py --rec-id Exp3/009
    py noise_viewer.py --rec-id Exp3/001 --k 0.3 --apnea-sec 10 --jump-bpm 20
"""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import bw_bp_sweep as S
import noise_zones as NZ
from main import run_reference, run_ppg, run_sync

SRC_COLORS = {"ref_existing": "#f59e0b", "ref_jump": "#ef4444",
              "watch_move": "#a855f7", "watch_apnea": "#22d3ee"}
SRC_LABEL = {"ref_existing": "ref: existing gate", "ref_jump": "ref: RR jump >20",
             "watch_move": "watch: movement", "watch_apnea": "watch: apnea/collapse"}


def decimate(t, y, t0, t1, nbins):
    t = np.asarray(t, float); y = np.asarray(y, float)
    if t1 <= t0 or t.size == 0:
        return [], [], []
    idx = np.clip(((t - t0) / (t1 - t0) * nbins).astype(int), 0, nbins - 1)
    lo = np.full(nbins, np.inf); hi = np.full(nbins, -np.inf)
    np.minimum.at(lo, idx, y); np.maximum.at(hi, idx, y)
    tc = t0 + (np.arange(nbins) + 0.5) * (t1 - t0) / nbins
    empty = ~np.isfinite(lo)
    loL = [None if e else round(float(v), 3) for v, e in zip(lo, empty)]
    hiL = [None if e else round(float(v), 3) for v, e in zip(hi, empty)]
    return [round(float(x), 2) for x in tc], loL, hiL


def main(argv=None):
    ap = argparse.ArgumentParser(description="Noise / red-zone review viewer")
    ap.add_argument("--rec-id", default=None)
    ap.add_argument("--edf", default=None); ap.add_argument("--watch", default=None)
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nbins", type=int, default=5000, help="display decimation resolution")
    ap.add_argument("--k", type=float, default=NZ.K_COLLAPSE)
    ap.add_argument("--apnea-sec", type=float, default=NZ.APNEA_MIN_SEC)
    ap.add_argument("--smooth-sec", type=float, default=NZ.ENV_SMOOTH_SEC)
    ap.add_argument("--jump-bpm", type=float, default=NZ.JUMP_BPM)
    args = ap.parse_args(argv)

    if args.edf and args.watch:
        edf, csv = args.edf, args.watch; rid = args.rec_id or "custom"
    else:
        recs = S.discover(args.data_root)
        seen = set(); recs = [r for r in recs if not (r[0] in seen or seen.add(r[0]))]
        rid = args.rec_id or recs[0][0]
        hit = [r for r in recs if r[0] == rid]
        if not hit:
            sys.exit(f"recording {rid} not found")
        _, edf, csv = hit[0]

    ref = run_reference(edf); ppg, sig = run_ppg(csv); off = run_sync(edf, ppg, sig)

    apnea_kw = dict(k=args.k, min_sec=args.apnea_sec, smooth_sec=args.smooth_sec)
    by_source, merged, stats = NZ.red_zones(ref, sig, off, jump_bpm=args.jump_bpm, **apnea_kw)
    _, env_t, env_s, env_med = NZ.apnea_zones(sig, return_envelope=True, **apnea_kw)

    t = np.asarray(ref.time, float); t0 = float(t[0]); t1 = float(t[-1])
    N = args.nbins
    tracks = []
    # REMbo airflow
    tc, lo, hi = decimate(ref.time, ref.raw, t0, t1, N)
    tracks.append({"name": "REMbo airflow", "tc": tc, "lo": lo, "hi": hi, "thr": None, "breaths": True})
    # watch Artifact + envelope threshold line (env plotted on its own track)
    wt = np.asarray(sig.time, float) + off
    for ch in ["Artifact", "Green", "Red", "IR", "Yellow"]:
        if ch in sig.channels:
            tc, lo, hi = decimate(wt, sig.channels[ch], t0, t1, N)
            tracks.append({"name": f"watch {ch}", "tc": tc, "lo": lo, "hi": hi, "thr": None, "breaths": False})
    # Artifact respiration envelope (pre-AGC) with collapse threshold k*median
    tc, lo, hi = decimate(env_t + off, env_s, t0, t1, N)
    tracks.append({"name": "Artifact resp-envelope", "tc": tc, "lo": lo, "hi": hi,
                   "thr": round(float(args.k * env_med), 3), "breaths": False})

    breaths = [{"s": round(float(b.start), 2), "e": round(float(b.end), 2),
                "r": round(float(b.rate), 1)} for b in ref.breaths]
    zones = [{"s": round(float(s), 2), "e": round(float(e), 2), "src": tag}
             for tag, zs in by_source.items() for (s, e) in zs]

    payload = {"rid": rid, "t0": t0, "t1": t1, "tracks": tracks, "breaths": breaths,
               "zones": zones, "stats": stats, "offset": round(float(off), 2),
               "colors": SRC_COLORS, "labels": SRC_LABEL,
               "params": {"k": args.k, "apnea_sec": args.apnea_sec, "jump_bpm": args.jump_bpm,
                          "smooth_sec": args.smooth_sec}}

    out = args.out or os.path.join(tempfile.gettempdir(), f"noise_viewer_{rid.replace('/', '_')}.html")
    out = os.path.abspath(out)
    open(out, "w", encoding="utf-8").write(_html(payload))
    print(f"\n  recording {rid} | offset {off:+.2f}s | noise {stats['noise_pct']:.1f}% of {stats['span_sec']:.0f}s")
    for tag, sec in stats["per_source_sec"].items():
        print(f"     {SRC_LABEL.get(tag, tag):24} {sec:6.1f}s")
    print(f"\n  viewer -> {out}")


def _html(payload):
    data = json.dumps(payload).replace("</", "<\\/")
    return r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Noise viewer</title>
<style>
  :root{--bg:#0b1220;--fg:#e6eef8;--mut:#9fb0c8;--grid:#1e2a3f;--card:#111a2e}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:10px 14px;border-bottom:1px solid var(--grid);position:sticky;top:0;background:var(--bg);z-index:5}
  h1{font-size:15px;margin:0 0 4px} .sub{color:var(--mut);font-size:12px}
  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px}
  .lg{display:flex;align-items:center;gap:5px;font-size:12px}
  .sw{width:14px;height:10px;border-radius:2px;display:inline-block;opacity:.55}
  .wrap{padding:6px 14px}
  .trk{margin:6px 0}
  .trk .lab{color:var(--mut);font-size:11px;margin:0 0 2px}
  canvas{width:100%;height:110px;display:block;background:var(--card);border:1px solid var(--grid);border-radius:6px;cursor:ew-resize}
  b{color:var(--fg)}
</style></head><body>
<header>
  <h1 id="ttl"></h1>
  <div class="sub" id="sub"></div>
  <div class="legend" id="legend"></div>
</header>
<div class="wrap" id="wrap"></div>
<script>
const D=""" + data + r""";
const DPR=Math.max(1,window.devicePixelRatio||1);
let dom=[D.t0,D.t1];                    // shared x domain (s)
document.getElementById("ttl").textContent="Noise review — "+D.rid;
document.getElementById("sub").innerHTML="offset <b>"+D.offset+"s</b> &middot; red zones = <b>"
  +D.stats.noise_pct.toFixed(1)+"%</b> of "+D.stats.span_sec.toFixed(0)+"s &middot; k="+D.params.k
  +", apnea&ge;"+D.params.apnea_sec+"s, jump>"+D.params.jump_bpm+" &middot; drag=pan, wheel=zoom, dbl-click=reset";
const leg=document.getElementById("legend");
for(const tag in D.colors){const sec=(D.stats.per_source_sec[tag]||0);
  const d=document.createElement("div");d.className="lg";
  d.innerHTML="<span class='sw' style='background:"+D.colors[tag]+"'></span>"+D.labels[tag]+" ("+sec.toFixed(0)+"s)";
  leg.appendChild(d);}

const wrap=document.getElementById("wrap");
const cans=[];
D.tracks.forEach((tk,i)=>{
  const box=document.createElement("div");box.className="trk";
  const lab=document.createElement("div");lab.className="lab";lab.textContent=tk.name;
  const c=document.createElement("canvas");
  box.appendChild(lab);box.appendChild(c);wrap.appendChild(box);
  cans.push({c,tk}); attach(c);
});

function xpx(t,w){return (t-dom[0])/(dom[1]-dom[0])*w;}
function draw(){
  for(const {c,tk} of cans){
    const w=c.clientWidth,h=c.clientHeight;
    if(c.width!==w*DPR){c.width=w*DPR;c.height=h*DPR;}
    const x=c.getContext("2d");x.setTransform(DPR,0,0,DPR,0,0);x.clearRect(0,0,w,h);
    // zones
    for(const z of D.zones){
      if(z.e<dom[0]||z.s>dom[1])continue;
      const a=Math.max(0,xpx(z.s,w)),b=Math.min(w,xpx(z.e,w));
      x.fillStyle=D.colors[z.src]||"#888";x.globalAlpha=.30;x.fillRect(a,0,Math.max(1,b-a),h);x.globalAlpha=1;
    }
    // y range from visible data
    let ymin=Infinity,ymax=-Infinity;
    for(let k=0;k<tk.tc.length;k++){const tt=tk.tc[k];if(tt<dom[0]||tt>dom[1])continue;
      const a=tk.lo[k],b=tk.hi[k];if(a==null)continue;if(a<ymin)ymin=a;if(b>ymax)ymax=b;}
    if(tk.thr!=null){ymin=Math.min(ymin,tk.thr);ymax=Math.max(ymax,tk.thr);}
    if(!isFinite(ymin)){ymin=-1;ymax=1;}
    if(ymin===ymax){ymin-=1;ymax+=1;}
    const pad=(ymax-ymin)*0.08;ymin-=pad;ymax+=pad;
    const ypx=v=>h-(v-ymin)/(ymax-ymin)*h;
    // zero / mid grid
    x.strokeStyle="#1e2a3f";x.lineWidth=1;x.beginPath();x.moveTo(0,ypx((ymin+ymax)/2));x.lineTo(w,ypx((ymin+ymax)/2));x.stroke();
    // breaths (airflow track only)
    if(tk.breaths){x.strokeStyle="rgba(120,200,255,.35)";x.lineWidth=1;x.beginPath();
      for(const br of D.breaths){if(br.s<dom[0]||br.s>dom[1])continue;const px=xpx(br.s,w);x.moveTo(px,0);x.lineTo(px,h);}x.stroke();}
    // signal min/max area
    x.strokeStyle="#cfe3ff";x.lineWidth=1;x.beginPath();let moved=false;
    for(let k=0;k<tk.tc.length;k++){const tt=tk.tc[k];if(tt<dom[0]||tt>dom[1])continue;
      const a=tk.lo[k];if(a==null){moved=false;continue;}const b=tk.hi[k];const px=xpx(tt,w);
      x.moveTo(px,ypx(a));x.lineTo(px,ypx(b));}
    x.stroke();
    // threshold line (envelope)
    if(tk.thr!=null){x.strokeStyle="#22d3ee";x.setLineDash([5,4]);x.lineWidth=1;x.beginPath();
      x.moveTo(0,ypx(tk.thr));x.lineTo(w,ypx(tk.thr));x.stroke();x.setLineDash([]);}
  }
}
function attach(c){
  let drag=null;
  c.addEventListener("wheel",e=>{e.preventDefault();const w=c.clientWidth;const r=c.getBoundingClientRect();
    const tx=dom[0]+(e.clientX-r.left)/w*(dom[1]-dom[0]);const f=e.deltaY<0?0.82:1.22;
    let a=tx-(tx-dom[0])*f,b=tx+(dom[1]-tx)*f;a=Math.max(D.t0,a);b=Math.min(D.t1,b);
    if(b-a>0.5){dom=[a,b];draw();}},{passive:false});
  c.addEventListener("mousedown",e=>{drag={x:e.clientX,d0:dom[0],d1:dom[1]};});
  window.addEventListener("mousemove",e=>{if(!drag)return;const w=c.clientWidth;
    const dt=(e.clientX-drag.x)/w*(drag.d1-drag.d0);let a=drag.d0-dt,b=drag.d1-dt;
    if(a<D.t0){b+=D.t0-a;a=D.t0;}if(b>D.t1){a-=b-D.t1;b=D.t1;}dom=[Math.max(D.t0,a),Math.min(D.t1,b)];draw();});
  window.addEventListener("mouseup",()=>drag=null);
  c.addEventListener("dblclick",()=>{dom=[D.t0,D.t1];draw();});
}
window.addEventListener("resize",draw);
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
