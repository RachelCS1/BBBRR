#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manual noise / apnea annotation tool. Shows, on ONE synchronized time axis (REMbo clock; the
watch is shifted by the sync offset), the raw REMbo airflow with its breath segmentation and every
raw watch channel. You drag to mark time regions, tagging each as 'noise' or 'apnea', on whichever
signal it belongs to. Regions are saved to the browser (localStorage) and exported as JSON so the
pipeline can later red-zone them and we can design a dedicated detector from the hand labels.

    py noise_annotator.py --rec-id Exp3/021
    py noise_annotator.py --rec-id Exp3/015 --out labels_015.html

Export writes annotations_<rec>.json (download), shape:
    {"rid":..., "clock":"rembo", "offset":<s>, "regions":[{"t0":,"t1":,"track":,"type":}]}
"""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import bw_bp_sweep as S
from noise_viewer import decimate
from main import run_reference, run_ppg, run_sync


def main(argv=None):
    ap = argparse.ArgumentParser(description="Manual noise/apnea annotator")
    ap.add_argument("--rec-id", default=None)
    ap.add_argument("--edf", default=None); ap.add_argument("--watch", default=None)
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nbins", type=int, default=6000)
    args = ap.parse_args(argv)

    if args.edf and args.watch:
        edf, csv = args.edf, args.watch; rid = args.rec_id or "custom"
    else:
        recs = S.discover(args.data_root); seen = set()
        recs = [r for r in recs if not (r[0] in seen or seen.add(r[0]))]
        rid = args.rec_id or recs[0][0]
        hit = [r for r in recs if r[0] == rid]
        if not hit:
            sys.exit(f"recording {rid} not found")
        _, edf, csv = hit[0]

    ref = run_reference(edf); ppg, sig = run_ppg(csv); off = run_sync(edf, ppg, sig)
    t = np.asarray(ref.time, float); t0 = float(t[0]); t1 = float(t[-1]); N = args.nbins

    tracks = []
    tc, lo, hi = decimate(ref.time, ref.raw, t0, t1, N)
    tracks.append({"name": "REMbo airflow", "tc": tc, "lo": lo, "hi": hi, "breaths": True})
    wt = np.asarray(sig.time, float) + off                       # watch -> REMbo clock (synchronized)
    for ch in ["Artifact", "Green", "Red", "IR", "Yellow"]:
        if ch in sig.channels:
            tc, lo, hi = decimate(wt, sig.channels[ch], t0, t1, N)
            tracks.append({"name": ch, "tc": tc, "lo": lo, "hi": hi, "breaths": False})
    breaths = [{"s": round(float(b.start), 2), "e": round(float(b.end), 2)} for b in ref.breaths]

    payload = {"rid": rid, "t0": t0, "t1": t1, "offset": round(float(off), 2),
               "tracks": tracks, "breaths": breaths}
    out = args.out or os.path.join(tempfile.gettempdir(), f"annotator_{rid.replace('/', '_')}.html")
    out = os.path.abspath(out)
    open(out, "w", encoding="utf-8").write(_html(payload))
    print(f"\n  recording {rid} | offset {off:+.2f}s | {len(breaths)} breaths | span {t1-t0:.0f}s")
    print(f"  annotator -> {out}")
    print(f"  (mark regions, then 'Download JSON' -> annotations_{rid.replace('/', '_')}.json)")


def _html(payload):
    data = json.dumps(payload).replace("</", "<\\/")
    return r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Noise annotator</title>
<style>
  :root{--bg:#0b1220;--fg:#e6eef8;--mut:#9fb0c8;--grid:#1e2a3f;--card:#111a2e;--noise:#f59e0b;--apnea:#22d3ee}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:9px 14px;border-bottom:1px solid var(--grid);position:sticky;top:0;background:var(--bg);z-index:5}
  h1{font-size:15px;margin:0 0 4px}.sub{color:var(--mut);font-size:12px}
  .tools{display:flex;gap:8px;align-items:center;margin-top:7px;flex-wrap:wrap}
  button{background:var(--card);color:var(--fg);border:1px solid var(--grid);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px}
  button.on{border-color:#5b9cff;background:#17233c}
  button.noise.on{border-color:var(--noise)} button.apnea.on{border-color:var(--apnea)}
  .sp{flex:1}
  .layout{display:flex;gap:10px;padding:6px 14px}
  .tracks{flex:1;min-width:0}.side{width:280px;flex:none}
  .trk{margin:5px 0}.trk .lab{color:var(--mut);font-size:11px;margin:0 0 2px}
  canvas{width:100%;height:96px;display:block;background:var(--card);border:1px solid var(--grid);border-radius:6px}
  .side h3{font-size:12px;margin:4px 0;color:var(--mut)}
  .reg{display:flex;gap:6px;align-items:center;font-size:11px;padding:3px 5px;border:1px solid var(--grid);border-radius:5px;margin:3px 0;cursor:pointer}
  .reg .dot{width:9px;height:9px;border-radius:2px;flex:none}
  .reg .x{margin-left:auto;color:#ff8080;cursor:pointer;padding:0 4px}
  textarea{width:100%;height:120px;background:#0a1120;color:var(--mut);border:1px solid var(--grid);border-radius:6px;font:11px monospace;margin-top:6px}
  .hint{color:var(--mut);font-size:11px}
</style></head><body>
<header>
  <h1 id="ttl"></h1><div class="sub" id="sub"></div>
  <div class="tools">
    <button id="mPan" class="on" onclick="setMode('pan')">Pan / Zoom</button>
    <button id="mNoise" class="noise" onclick="setMode('noise')">Mark Noise</button>
    <button id="mApnea" class="apnea" onclick="setMode('apnea')">Mark Apnea</button>
    <span class="sp"></span>
    <button onclick="resetView()">Reset view</button>
    <button onclick="clearAll()">Clear all</button>
    <button onclick="download()">⬇ Download JSON</button>
  </div>
</header>
<div class="layout">
  <div class="tracks" id="tracks"></div>
  <div class="side">
    <h3 id="cnt"></h3>
    <div id="list"></div>
    <h3>export</h3>
    <div class="hint">autosaved in this browser. Download to a file the pipeline can read.</div>
    <textarea id="json" readonly onclick="this.select()"></textarea>
  </div>
</div>
<script>
const D=""" + data + r""";
const DPR=Math.max(1,window.devicePixelRatio||1);
const KEY="annot_"+D.rid;
let dom=[D.t0,D.t1], mode="pan";
let regions=[]; try{regions=JSON.parse(localStorage.getItem(KEY))||[]}catch(e){}
document.getElementById("ttl").textContent="Noise annotator — "+D.rid;
document.getElementById("sub").innerHTML="synchronized (REMbo clock) · offset <b>"+D.offset
  +"s</b> · drag in a Mark mode to tag a region on that signal · wheel=zoom, Pan mode drag=pan";

const tracksEl=document.getElementById("tracks"); const cans=[];
D.tracks.forEach(tk=>{
  const box=document.createElement("div");box.className="trk";
  const lab=document.createElement("div");lab.className="lab";lab.textContent=tk.name;
  const c=document.createElement("canvas");
  box.appendChild(lab);box.appendChild(c);tracksEl.appendChild(box);
  const o={c,tk}; cans.push(o); attach(o);
});
function setMode(m){mode=m;for(const id of ["pan","noise","apnea"])document.getElementById("m"+id[0].toUpperCase()+id.slice(1)).classList.toggle("on",id===m);}
function resetView(){dom=[D.t0,D.t1];draw();}
function clearAll(){if(confirm("Delete all "+regions.length+" regions?")){regions=[];save();}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(regions))}catch(e){};renderList();draw();}
function xpx(t,w){return (t-dom[0])/(dom[1]-dom[0])*w;}
function pxt(px,w){return dom[0]+px/w*(dom[1]-dom[0]);}

let drag=null;        // {o, mode, x0, t0} for region or pan
function attach(o){
  const c=o.c;
  c.addEventListener("wheel",e=>{e.preventDefault();const w=c.clientWidth,r=c.getBoundingClientRect();
    const tx=pxt(e.clientX-r.left,w),f=e.deltaY<0?0.82:1.22;
    let a=tx-(tx-dom[0])*f,b=tx+(dom[1]-tx)*f;a=Math.max(D.t0,a);b=Math.min(D.t1,b);
    if(b-a>0.5){dom=[a,b];draw();}},{passive:false});
  c.addEventListener("mousedown",e=>{const w=c.clientWidth,r=c.getBoundingClientRect();
    if(mode==="pan"){drag={pan:true,x:e.clientX,d0:dom[0],d1:dom[1]};}
    else{drag={o,mode,t0:pxt(e.clientX-r.left,w),t1:pxt(e.clientX-r.left,w)};}});
  window.addEventListener("mousemove",e=>{if(!drag)return;const w=c.clientWidth,r=c.getBoundingClientRect();
    if(drag.pan){const dt=(e.clientX-drag.x)/w*(drag.d1-drag.d0);let a=drag.d0-dt,b=drag.d1-dt;
      if(a<D.t0){b+=D.t0-a;a=D.t0;}if(b>D.t1){a-=b-D.t1;b=D.t1;}dom=[Math.max(D.t0,a),Math.min(D.t1,b)];draw();}
    else if(drag.o===o){drag.t1=pxt(e.clientX-r.left,w);draw();}});
  window.addEventListener("mouseup",()=>{if(drag&&!drag.pan){let a=Math.min(drag.t0,drag.t1),b=Math.max(drag.t0,drag.t1);
    if(b-a>=0.3){regions.push({t0:+a.toFixed(2),t1:+b.toFixed(2),track:drag.o.tk.name,type:drag.mode});save();}else draw();}
    drag=null;});
  c.addEventListener("dblclick",()=>resetView());
}
function draw(){
  for(const {c,tk} of cans){
    const w=c.clientWidth,h=c.clientHeight;
    if(c.width!==w*DPR){c.width=w*DPR;c.height=h*DPR;}
    const x=c.getContext("2d");x.setTransform(DPR,0,0,DPR,0,0);x.clearRect(0,0,w,h);
    // existing regions: strong on own track, faint elsewhere (shared time axis)
    for(const g of regions){if(g.t1<dom[0]||g.t0>dom[1])continue;
      const a=Math.max(0,xpx(g.t0,w)),b=Math.min(w,xpx(g.t1,w));
      x.fillStyle=g.type==="apnea"?"#22d3ee":"#f59e0b";
      x.globalAlpha=(g.track===tk.name)?0.33:0.08;x.fillRect(a,0,Math.max(1,b-a),h);x.globalAlpha=1;
      if(g.track===tk.name){x.strokeStyle=g.type==="apnea"?"#22d3ee":"#f59e0b";x.globalAlpha=.8;x.strokeRect(a,0.5,Math.max(1,b-a),h-1);x.globalAlpha=1;}}
    // in-progress region
    if(drag&&!drag.pan&&drag.o.tk.name===tk.name){const a=xpx(Math.min(drag.t0,drag.t1),w),b=xpx(Math.max(drag.t0,drag.t1),w);
      x.fillStyle=drag.mode==="apnea"?"#22d3ee":"#f59e0b";x.globalAlpha=.25;x.fillRect(a,0,Math.max(1,b-a),h);x.globalAlpha=1;}
    // y range
    let ymin=Infinity,ymax=-Infinity;
    for(let k=0;k<tk.tc.length;k++){const tt=tk.tc[k];if(tt<dom[0]||tt>dom[1])continue;const lo=tk.lo[k];if(lo==null)continue;if(lo<ymin)ymin=lo;if(tk.hi[k]>ymax)ymax=tk.hi[k];}
    if(!isFinite(ymin)){ymin=-1;ymax=1;}if(ymin===ymax){ymin-=1;ymax+=1;}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
    const ypx=v=>h-(v-ymin)/(ymax-ymin)*h;
    if(tk.breaths){x.strokeStyle="rgba(120,200,255,.3)";x.lineWidth=1;x.beginPath();
      for(const br of D.breaths){if(br.s<dom[0]||br.s>dom[1])continue;const px=xpx(br.s,w);x.moveTo(px,0);x.lineTo(px,h);}x.stroke();}
    x.strokeStyle="#cfe3ff";x.lineWidth=1;x.beginPath();
    for(let k=0;k<tk.tc.length;k++){const tt=tk.tc[k];if(tt<dom[0]||tt>dom[1])continue;const lo=tk.lo[k];if(lo==null)continue;const px=xpx(tt,w);x.moveTo(px,ypx(lo));x.lineTo(px,ypx(tk.hi[k]));}
    x.stroke();
  }
}
function renderList(){
  document.getElementById("cnt").textContent=regions.length+" region(s)";
  const L=document.getElementById("list");L.innerHTML="";
  regions.map((g,i)=>[g,i]).sort((p,q)=>p[0].t0-q[0].t0).forEach(([g,i])=>{
    const d=document.createElement("div");d.className="reg";
    d.innerHTML="<span class='dot' style='background:"+(g.type==="apnea"?"#22d3ee":"#f59e0b")+"'></span>"
      +"<b>"+g.type+"</b> "+g.track+" <span class='hint'>"+g.t0.toFixed(1)+"–"+g.t1.toFixed(1)+"s</span>"
      +"<span class='x' data-i='"+i+"'>✕</span>";
    d.onclick=ev=>{if(ev.target.classList.contains("x")){regions.splice(+ev.target.dataset.i,1);save();}
      else{const m=(g.t0+g.t1)/2,span=Math.max(10,(g.t1-g.t0)*3);dom=[Math.max(D.t0,m-span),Math.min(D.t1,m+span)];draw();}};
    L.appendChild(d);
  });
  document.getElementById("json").value=JSON.stringify({rid:D.rid,clock:"rembo",offset:D.offset,regions},null,1);
}
function download(){const blob=new Blob([JSON.stringify({rid:D.rid,clock:"rembo",offset:D.offset,regions},null,1)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="annotations_"+D.rid.replace(/\//g,"_")+".json";a.click();}
window.addEventListener("resize",draw);
renderList();draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
