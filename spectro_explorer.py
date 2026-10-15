#!/usr/bin/env python
"""
spectro_explorer.py — per-recording interactive HTML to inspect EVERY signal's
spectrogram with the reference RR and the strongest-frequency (ridge) drawn on it.

Pick, from one dropdown, any of ~25 spectrograms and view it large:
  1. RAW   — the unfiltered channel               (Green/Red/IR/Yellow/Artifact)
  2. BW    — the channel band-passed 0.1-1.0 Hz    (same 5 channels)
  3. PARAM — per-beat RSA / AMP / AUC envelope, SPLINE reconstruction from
             the MAIN pipeline (cubic_spline_rr_series @ 10 Hz), before per-category
             analysis                              (5 channels x 3 params = 15)

On every spectrogram: the reference RR (faint per-breath points + averaged 20 s
line, both shifted to the watch clock) and the argmax ridge in 0.10-0.80 Hz.
STFT window is switchable in the page (default 24 s; also 32 / 48 s), hop 5 s.

    py spectro_explorer.py --rec-id Exp3/003
    py spectro_explorer.py --rec-id Exp3/003 --windows 24,32,48
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import bw_bp_sweep as S
from param_category_sweep import stft_power, ridge_rr, envelope_on_grid, GRID_FS

OUT_DEFAULT = r"C:\Users\RACHEL~1\AppData\Local\Temp\claude\C--Users-RachelMizrahi-OneDrive---CardiacSense-Documents-GitHub-BBBRR\3bbdb805-5d0d-4f52-b50c-def0402d0def\scratchpad"
CHANNELS = ("Green", "Red", "IR", "Yellow", "Artifact")
PARAMS = ("RSA", "AMP", "AUC")            # AMP = per-beat amplitude (max-min); code key was "RIIV"
PARAM_LABEL = {"RSA": "RSA", "AMP": "AMP", "AUC": "AUC"}
RECON = "spl"                               # spline — the main uniform analysis
HOP_SEC = 5.0
F_HI_DISP = 1.15                            # display up to ~69 bpm


def _key(x):
    return str(int(x)) if float(x).is_integer() else ("%g" % x)


def spec_payload(sigarr, fs, windows, hops):
    """Nested {winkey: {hopkey: {times, fbpm, heat, ridge}}} for one signal."""
    out = {}
    for w in windows:
        wk = _key(w); out[wk] = {}
        for h in hops:
            hk = _key(h)
            t, fr, P = stft_power(sigarr, fs, float(w), float(h))
            if t.size == 0:
                out[wk][hk] = None
                continue
            ridge = ridge_rr(fr, P)                       # argmax in 0.10-0.80 Hz (bpm)
            keep = fr <= F_HI_DISP / 1.0
            fb = (fr[keep] * 60.0)
            Pd = P[keep, :]                               # linear power; global dB scaling done in JS
            ref = np.percentile(Pd, 99.0)                 # ROBUST scale (not global max): a single
            Pn = Pd / ref if ref > 0 else Pd              #   spike column no longer crushes the quiet
            cols = [np.round(Pn[:, c], 6).tolist() for c in range(Pn.shape[1])]   # columns to 0 under rounding
            # NOTE: this is a global constant rescale -> every within-column excess/argmax ridge is
            # unchanged; color is re-normalized in buildOff(). It only un-crushes quiet columns so the
            # JS excess ridge matches the full-precision pipeline (no spurious NaN gaps in the orange line).
            out[wk][hk] = {"times": np.round(t, 1).tolist(),
                           "fbpm": np.round(fb, 2).tolist(),
                           "heat": cols,
                           "ridge": [None if not np.isfinite(v) else round(float(v), 2) for v in ridge]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-id", default=None); ap.add_argument("--recording", default=None)
    ap.add_argument("--edf", default=None); ap.add_argument("--watch", default=None)
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--windows", default="16,24,32,48", help="STFT windows (s), comma-list; first = default")
    ap.add_argument("--hops", default="5,2.5,10,15", help="STFT hops (s), comma-list; first = default (2.5=small/dense ... 15=big/sparse)")
    ap.add_argument("--bw-low", type=float, default=0.10, help="BW band-pass low cutoff (Hz)")
    ap.add_argument("--bw-high", type=float, default=0.80, help="BW band-pass high cutoff (Hz); 0.8=48bpm, 1.0=60bpm")
    args = ap.parse_args()
    windows = [float(x) for x in args.windows.split(",") if x.strip()]
    hops = [float(x) for x in args.hops.split(",") if x.strip()]

    from respiration_rr.settings import PPG as ppg
    from scipy.signal import butter, sosfiltfilt
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr, reference_rr_points

    rid, edf, csv = ([r for r in S.discover(args.data_root) if r[0] == args.rec_id] or [S._resolve_inputs(args)])[0] \
        if args.rec_id else S._resolve_inputs(args)
    print("Recording:", rid, "| windows", windows)
    ppg_res, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg_res, sig)
    fs = float(sig.fs); n = int(np.asarray(sig.channels["Artifact"]).size)
    ny = fs / 2.0
    sos = butter(2, [args.bw_low / ny, args.bw_high / ny], btype="band", output="sos")
    bwlab = "%.2g-%.2g" % (args.bw_low, args.bw_high)
    g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)

    # reference on the WATCH clock (subtract sync offset)
    tav, rav = reference_average_rr(ref)
    pt, pr = reference_rr_points(ref)
    reference = {
        "avgT": np.round(tav - offset, 1).tolist(),
        "avgR": [None if not np.isfinite(v) else round(float(v), 2) for v in rav],
        "ptT": np.round(pt - offset, 1).tolist(),
        "ptR": [round(float(v), 2) for v in pr],
    }

    # heart rate over time from the GREEN LED beats, on the WATCH clock (same x-axis as the spectrogram)
    hr = {"t": [], "v": []}
    try:
        res = ppg_res.get("Green")
        if res is not None and len(getattr(res, "ss_idx", []) or []) > 3:
            bt = np.asarray(sig.time)[np.asarray(res.ss_idx, int)]     # watch clock (no offset)
            dt = np.diff(bt); ok = dt > 0
            hrv = 60.0 / dt[ok]; ht = ((bt[:-1] + bt[1:]) / 2)[ok]
            good = (hrv >= 35) & (hrv <= 180)
            hr = {"t": np.round(ht[good], 2).tolist(), "v": np.round(hrv[good], 1).tolist()}
        else:
            print("  [warn] HR: Green LED has too few beats")
    except Exception as e:
        print("  [warn] HR series skipped (%s)" % e)

    signals = []
    for c in CHANNELS:
        x = np.asarray(sig.channels[c], np.float64)
        signals.append({"key": "raw:" + c, "group": "RAW", "label": "%s — raw" % c,
                        "byWin": spec_payload(x, fs, windows, hops)})
    for c in CHANNELS:
        x = sosfiltfilt(sos, np.asarray(sig.channels[c], np.float64))
        signals.append({"key": "bw:" + c, "group": "BW", "label": "%s — BW (%s)" % (c, bwlab),
                        "byWin": spec_payload(x, fs, windows, hops)})
    for c in CHANNELS:
        res = ppg_res.get(c)
        for p in PARAMS:
            prm = res.params.get(p) if res and getattr(res, "params", None) else None
            bx = np.asarray(getattr(prm, "series_x", []), np.float64) if prm else np.zeros(0)
            by = np.asarray(getattr(prm, "series_y", []), np.float64) if prm else np.zeros(0)
            if bx.size < 4:
                signals.append({"key": "%s:%s" % (p, c), "group": "PARAM",
                                "label": "%s — %s (spline)" % (c, PARAM_LABEL[p]), "byWin": None})
                continue
            env = envelope_on_grid(bx, by, RECON, g_t, ppg)
            signals.append({"key": "%s:%s" % (p, c), "group": "PARAM",
                            "label": "%s — %s (spline)" % (c, PARAM_LABEL[p]),
                            "byWin": spec_payload(env, GRID_FS, windows, hops)})
        print("  channel %-9s done" % c)

    data = {"rid": rid,
            "windows": [int(w) if float(w).is_integer() else w for w in windows],
            "hops": [int(h) if float(h).is_integer() else h for h in hops],
            "topBpm": round(args.bw_high * 60.0, 1),      # excess-ridge search cap = BW band-pass top
            "reference": reference, "hr": hr, "signals": signals}
    html = (TEMPLATE.replace("__RID__", rid.replace("/", "_"))
                    .replace("__DATA__", json.dumps(data)))
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "spectro_%s.html" % rid.replace("/", "_"))
    open(outp, "w", encoding="utf-8").write(html)
    print("  saved ->", outp, "(%.0f KB)" % (len(html) / 1024))


TEMPLATE = r"""<title>Spectrogram explorer — __RID__</title>
<style>
  :root{--bg:#0c111c;--panel:#121a29;--line:#243349;--ink:#e6eef8;--mut:#8ba0ba;
    --cyan:#22d3ee;--amber:#f59e0b;--white:#ffffff;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,Segoe UI,sans-serif}
  .wrap{max-width:1200px;margin:0 auto;padding:16px 20px 60px}
  h1{font-size:19px;margin:0 0 2px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:12px}
  .ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:10px 0}
  select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:8px 12px;font-size:14px;font-family:inherit;min-width:280px}
  .wins{display:flex;gap:6px}
  .wins button{background:var(--panel);color:var(--mut);border:1px solid var(--line);border-radius:7px;
    padding:7px 14px;font-size:13px;cursor:pointer;font-family:"IBM Plex Mono",monospace}
  .wins button.on{background:var(--cyan);color:#05202a;border-color:var(--cyan);font-weight:700}
  .nav{display:flex;gap:6px}
  .nav button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:7px;
    padding:7px 12px;font-size:13px;cursor:pointer}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;margin-top:10px}
  .panel h2{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--mut);
    margin:0 0 8px;font-family:"IBM Plex Mono",monospace}
  canvas{width:100%;display:block;border-radius:6px}
  .leg{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:var(--mut);margin-top:8px}
  .leg i{display:inline-block;width:12px;height:3px;vertical-align:3px;margin-inline-end:5px;border-radius:2px}
  .empty{color:var(--mut);font-family:"IBM Plex Mono",monospace;padding:40px;text-align:center}
</style>
<div class="wrap">
  <h1>Spectrogram explorer · <span id="rid"></span></h1>
  <div class="sub">reference RR (points + 20s line) &amp; argmax ridge 0.10-0.80 Hz drawn on each · hop 5 s · params = spline from main pipeline</div>
  <div class="ctl">
    <select id="sel"></select>
    <div class="nav"><button id="prev">◀ prev</button><button id="next">next ▶</button></div>
    <div class="wins" id="wins"></div>
    <div class="wins" id="hops"></div>
    <div class="wins"><button id="robustBtn">robust norm</button></div>
    <div class="wins"><button id="exBtn">excess ridge</button></div>
    <div id="exCtl" class="wins" hidden>
      <button id="harmBtn">harmonic guard</button>
      <button id="persBtn">persistence</button>
      <label style="font-family:var(--mono);font-size:11px;color:var(--mut);display:flex;align-items:center;gap:6px">fast&#8209;pref &alpha; <input id="alpha" type="range" min="0.2" max="1" step="0.05" value="0.6" style="width:84px"><span id="aval">0.60</span></label>
    </div>
  </div>
  <div class="panel">
    <h2 id="title"></h2>
    <canvas id="cv"></canvas>
    <div id="msg" class="empty" style="display:none">no data for this signal</div>
    <div class="leg">
      <span><i style="background:#f59e0b"></i>ridge (strongest freq)</span>
      <span><i style="background:#ffffff"></i>reference avg (20s)</span>
      <span><i style="background:#9fb4d0"></i>reference per-breath</span>
      <span>heat = power, dB (global, 60 dB range)</span>
      <span>hover the spectrogram to read its spectrum below</span>
    </div>
  </div>
  <div class="panel">
    <h2>heart rate (bpm) over time</h2>
    <canvas id="hrcv"></canvas>
    <div class="leg"><span><i style="background:#f472b6"></i>HR per-beat (bpm), same time axis as the spectrogram</span></div>
  </div>
  <div class="panel">
    <h2 id="spectitle">spectrum @ cursor</h2>
    <canvas id="spec"></canvas>
  </div>
</div>
<script>
const D=__DATA__;
document.getElementById('rid').textContent=D.rid;
let curWin=String(D.windows[0]), curHop=String(D.hops[0]), curIdx=0, curCol=0, ROBUST=false, EXRIDGE=false, ALPHA=0.6, HARMG=false, PERSIST=false;
const TOPB = D.topBpm || 60;      // excess-ridge search cap (= BW band-pass top, bpm)

// ---- dropdown grouped by RAW / BW / PARAM ----
const sel=document.getElementById('sel');
["RAW","BW","PARAM"].forEach(g=>{
  const og=document.createElement('optgroup'); og.label=g;
  D.signals.forEach((s,i)=>{ if(s.group===g){const o=document.createElement('option');o.value=i;o.textContent=s.label;og.appendChild(o);} });
  sel.appendChild(og);
});
sel.addEventListener('change',()=>{curIdx=+sel.value;draw();});
document.getElementById('prev').onclick=()=>{curIdx=(curIdx-1+D.signals.length)%D.signals.length;sel.value=curIdx;draw();};
document.getElementById('next').onclick=()=>{curIdx=(curIdx+1)%D.signals.length;sel.value=curIdx;draw();};

// ---- window buttons ----
const wins=document.getElementById('wins');
D.windows.forEach(w=>{const b=document.createElement('button');b.textContent='win '+w+'s';b.dataset.w=w;
  b.onclick=()=>{curWin=String(w);[...wins.children].forEach(c=>c.classList.toggle('on',c.dataset.w==w));draw();};
  if(String(w)===curWin)b.classList.add('on');wins.appendChild(b);});

// ---- hop buttons (step between windows; small = dense, big = sparse) ----
const hopsEl=document.getElementById('hops');
D.hops.forEach(h=>{const b=document.createElement('button');b.textContent='hop '+h+'s';b.dataset.h=h;
  b.onclick=()=>{curHop=String(h);[...hopsEl.children].forEach(c=>c.classList.toggle('on',c.dataset.h==h));draw();};
  if(String(h)===curHop)b.classList.add('on');hopsEl.appendChild(b);});

// robust-normalization toggle (percentile white-point vs global max)
const robustBtn=document.getElementById('robustBtn');
robustBtn.onclick=()=>{ROBUST=!ROBUST;robustBtn.classList.toggle('on',ROBUST);draw();};

// excess-ridge toggle: background-subtract each column, argmax the EXCESS (prominence),
// and prefer a peak in the 30-60 bpm band when a below-30 peak won but the fast bump is significant.
const exBtn=document.getElementById('exBtn'), exCtl=document.getElementById('exCtl'),
      harmBtn=document.getElementById('harmBtn'), persBtn=document.getElementById('persBtn'),
      alphaIn=document.getElementById('alpha'), aval=document.getElementById('aval');
exBtn.onclick=()=>{EXRIDGE=!EXRIDGE;exBtn.classList.toggle('on',EXRIDGE);exCtl.hidden=!EXRIDGE;render();};
harmBtn.onclick=()=>{HARMG=!HARMG;harmBtn.classList.toggle('on',HARMG);if(EXRIDGE)render();};
persBtn.onclick=()=>{PERSIST=!PERSIST;persBtn.classList.toggle('on',PERSIST);if(EXRIDGE)render();};
alphaIn.oninput=()=>{ALPHA=+alphaIn.value;aval.textContent=ALPHA.toFixed(2);if(EXRIDGE)render();};

function excessRidge(wd, alpha, harmGuard, persist){
  const heat=wd.heat, fb=wd.fbpm, n=heat.length, nb=fb.length;
  const df = nb>1 ? (fb[1]-fb[0]) : 2.5;
  const K = Math.max(2, Math.round(9/df));          // background median half-window ~ +-9 bpm
  const base=new Array(n), fast=new Array(n), promoted=new Array(n).fill(false);
  for(let c=0;c<n;c++){
    const col=heat[c]; const E=new Float64Array(nb); let colMax=0;
    for(let k=0;k<nb;k++){
      const a=Math.max(0,k-K), b=Math.min(nb-1,k+K), win=[];
      for(let j=a;j<=b;j++) win.push(col[j]);
      win.sort((x,y)=>x-y);
      const bg=win[(win.length-1)>>1];              // local median = background skirt
      const e=col[k]-bg; E[k]= e>0?e:0;
      if(fb[k]>=6 && fb[k]<=TOPB && col[k]>colMax) colMax=col[k];
    }
    let fAll=null,eAll=0, fFast=null,eFast=0;
    for(let k=0;k<nb;k++){
      if(fb[k]<6||fb[k]>TOPB) continue;
      if(E[k]>eAll){eAll=E[k]; fAll=fb[k];}
      if(fb[k]>=30 && E[k]>eFast){eFast=E[k]; fFast=fb[k];}
    }
    base[c]=fAll; fast[c]=fFast;
    let promote = (fAll!=null && fAll<30 && fFast!=null && eAll>0 && eFast>=alpha*eAll && eFast>=0.05*colMax);
    if(promote && harmGuard){                        // reject if fFast looks like the 2nd harmonic of a slower fundamental
      const half=fFast/2; let eHalf=0;
      for(let k=0;k<nb;k++){ if(Math.abs(fb[k]-half)<=df && E[k]>eHalf) eHalf=E[k]; }
      if(eHalf>=0.5*eFast) promote=false;
    }
    promoted[c]=promote;
  }
  if(persist){                                       // keep a fast promotion only if it holds >=2 consecutive columns
    const keep=promoted.slice();
    for(let c=0;c<n;c++) if(promoted[c] && !((c>0&&promoted[c-1])||(c<n-1&&promoted[c+1]))) keep[c]=false;
    for(let c=0;c<n;c++) promoted[c]=keep[c];
  }
  const out=new Array(n);
  for(let c=0;c<n;c++) out[c]= promoted[c] ? fast[c] : base[c];
  return out;
}

// ---- magma colormap (same stops as route_fast) ----
const CM=[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function col255(v){v=Math.max(0,Math.min(1,v));const x=v*(CM.length-1),i=Math.floor(x),f=x-i,a=CM[i],b=CM[Math.min(i+1,CM.length-1)];
  return [Math.round(a[0]+(b[0]-a[0])*f),Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f)];}

const cv=document.getElementById('cv'),cx=cv.getContext('2d');
const sc=document.getElementById('spec'),sx=sc.getContext('2d');
const hc=document.getElementById('hrcv'),hx=hc.getContext('2d');
const M={l:52,r:14,t:10,b:34}, YMAX=69;         // bpm axis top
const BC=['rgba(96,165,250,.12)','rgba(52,211,153,.12)','rgba(245,158,11,.12)','rgba(244,114,182,.12)'];
let G={};                                        // cached render state for cheap cursor moves

// build the heatmap once (per signal/window) into an offscreen canvas: dB, 60 dB range.
// ROBUST: white-point = 99th percentile of the dB values (ignores bright transients/artifacts)
// so the darker regions expand into the colormap; default: white-point = global max.
function buildOff(wd){
  const heat=wd.heat,fb=wd.fbpm,nCol=heat.length,nBin=fb.length;
  let dbMax;
  if(ROBUST){const db=[];for(let c=0;c<nCol;c++)for(let b=0;b<nBin;b++)db.push(20*Math.log10(heat[c][b]+1e-9));
    db.sort((a,b)=>a-b);dbMax=db[Math.floor(0.99*(db.length-1))];}
  else{dbMax=-1e9;for(let c=0;c<nCol;c++)for(let b=0;b<nBin;b++){const v=20*Math.log10(heat[c][b]+1e-9);if(v>dbMax)dbMax=v;}}
  const dbMin=dbMax-60;
  const off=document.createElement('canvas');off.width=nCol;off.height=nBin;
  const o=off.getContext('2d'),img=o.createImageData(nCol,nBin);
  for(let c=0;c<nCol;c++)for(let b=0;b<nBin;b++){
    const t=(20*Math.log10(heat[c][b]+1e-9)-dbMin)/(dbMax-dbMin);const g=col255(t);
    const y=nBin-1-b,k=(y*nCol+c)*4;img.data[k]=g[0];img.data[k+1]=g[1];img.data[k+2]=g[2];img.data[k+3]=255;}
  o.putImageData(img,0,0);
  return {off,nCol,nBin,fbmax:fb[nBin-1]};
}

function draw(){
  const s=D.signals[curIdx];
  document.getElementById('title').textContent=s.label+'   ·   window '+curWin+'s · hop '+curHop+'s';
  sel.value=curIdx;
  const wd=(s.byWin && s.byWin[curWin]) ? s.byWin[curWin][curHop] : null;
  const msg=document.getElementById('msg');
  if(!wd){cv.style.display='none';sc.style.display='none';msg.style.display='block';return;}
  cv.style.display='block';sc.style.display='block';msg.style.display='none';
  const ob=buildOff(wd);
  G={wd:wd,off:ob.off,fbmax:ob.fbmax,t:wd.times,fb:wd.fbpm,heat:wd.heat,ridge:wd.ridge};
  curCol=Math.max(0,Math.min(wd.times.length-1,curCol));
  render();
}

function render(){
  const dpr=window.devicePixelRatio||1;
  const W=cv.clientWidth,H=440;
  cv.width=W*dpr;cv.height=H*dpr;cx.setTransform(dpr,0,0,dpr,0,0);cx.clearRect(0,0,W,H);
  const t=G.t,fb=G.fb,R=D.reference;
  const ridge = EXRIDGE ? excessRidge(G.wd, ALPHA, HARMG, PERSIST) : G.ridge;
  G.activeRidge = ridge;
  const x0=M.l,x1=W-M.r,y0=H-M.b,y1=M.t;
  const t0=t[0],t1=t[t.length-1];
  const TX=v=>x0+(x1-x0)*(v-t0)/(t1-t0||1);
  const BY=v=>y0+(y1-y0)*(v/YMAX);
  const cw=(x1-x0)/t.length;
  G.x0=x0;G.x1=x1;G.cw=cw;G.dpr=dpr;G.t0=t0;G.t1=t1;
  // heatmap (scaled from offscreen)
  cx.fillStyle='#05070c';cx.fillRect(x0,y1,x1-x0,y0-y1);
  cx.imageSmoothingEnabled=true;
  const yTop=BY(G.fbmax);
  cx.drawImage(G.off,x0,yTop,x1-x0,y0-yTop);
  // category boundary lines + y ticks
  cx.fillStyle='rgba(230,238,248,.75)';cx.font='11px monospace';cx.lineWidth=1;
  [15,22,30,45,60].forEach(b=>{cx.strokeStyle='rgba(230,238,248,.28)';cx.beginPath();cx.setLineDash([4,4]);cx.moveTo(x0,BY(b));cx.lineTo(x1,BY(b));cx.stroke();cx.setLineDash([]);
    cx.fillText(b,6,BY(b)+3);});
  cx.fillText('bpm',6,y1+4);
  const span=t1-t0,step=span>500?120:(span>250?60:30);
  for(let tt=Math.ceil(t0/step)*step;tt<=t1;tt+=step){cx.fillText(tt+'s',TX(tt)-8,y0+18);}
  // reference per-breath points + averaged line
  cx.fillStyle='rgba(159,180,208,.8)';
  for(let i=0;i<R.ptT.length;i++){const tp=R.ptT[i];if(tp<t0||tp>t1)continue;const v=R.ptR[i];if(v>YMAX)continue;cx.beginPath();cx.arc(TX(tp),BY(v),1.6,0,7);cx.fill();}
  cx.strokeStyle='#ffffff';cx.lineWidth=2;cx.beginPath();let pen=false;
  for(let i=0;i<R.avgT.length;i++){const tp=R.avgT[i],v=R.avgR[i];if(tp<t0||tp>t1||v==null){pen=false;continue;}const X=TX(tp),Y=BY(v);if(!pen){cx.moveTo(X,Y);pen=true;}else cx.lineTo(X,Y);}
  cx.stroke();
  // old argmax ridge (faint dashed) for comparison, when excess mode is on
  if(EXRIDGE){cx.strokeStyle='rgba(148,163,184,.65)';cx.lineWidth=1.3;cx.setLineDash([5,4]);cx.beginPath();pen=false;
    for(let i=0;i<G.ridge.length;i++){const v=G.ridge[i];if(v==null){pen=false;continue;}const X=TX(t[i]),Y=BY(v);if(!pen){cx.moveTo(X,Y);pen=true;}else cx.lineTo(X,Y);}
    cx.stroke();cx.setLineDash([]);}
  // active ridge
  cx.strokeStyle='#f59e0b';cx.lineWidth=2;cx.beginPath();pen=false;
  for(let i=0;i<ridge.length;i++){const v=ridge[i];if(v==null){pen=false;continue;}const X=TX(t[i]),Y=BY(v);if(!pen){cx.moveTo(X,Y);pen=true;}else cx.lineTo(X,Y);}
  cx.stroke();
  // cursor
  const cxpos=x0+cw*(curCol+0.5);
  cx.strokeStyle='rgba(255,255,255,.85)';cx.lineWidth=1.2;cx.beginPath();cx.moveTo(cxpos,y1);cx.lineTo(cxpos,y0);cx.stroke();
  drawSpec();
  drawHR();
}

// heart-rate strip: same time axis (G.t0..G.t1) as the spectrogram above
function drawHR(){
  const dpr=window.devicePixelRatio||1, W=hc.clientWidth, H=210;
  hc.width=W*dpr;hc.height=H*dpr;hx.setTransform(dpr,0,0,dpr,0,0);hx.clearRect(0,0,W,H);
  const x0=M.l,x1=W-M.r,y0=H-24,y1=10, t0=G.t0,t1=G.t1;
  const hr=D.hr||{t:[],v:[]};
  hx.fillStyle='#05070c';hx.fillRect(x0,y1,x1-x0,y0-y1);
  if(!hr.t.length){hx.fillStyle='#8ba0ba';hx.font='12px monospace';hx.fillText('no HR data',x0+10,(y0+y1)/2);return;}
  // zoomed y-range: robust 3rd..97th percentile (ignore stray beats), tight padding, clamp 30..180
  const sorted=hr.v.slice().sort((a,b)=>a-b), q=p=>sorted[Math.min(sorted.length-1,Math.max(0,Math.round(p*(sorted.length-1))))];
  let lo=q(0.03), hiv=q(0.97), pad=Math.max(3,(hiv-lo)*0.12);
  lo=Math.max(30,Math.floor((lo-pad)/2)*2); hiv=Math.min(180,Math.ceil((hiv+pad)/2)*2); if(hiv-lo<8)hiv=lo+8;
  const TX=v=>x0+(x1-x0)*(v-t0)/((t1-t0)||1), HY=v=>y0+(y1-y0)*(v-lo)/((hiv-lo)||1);
  // y ticks
  hx.fillStyle='rgba(230,238,248,.7)';hx.font='11px monospace';hx.strokeStyle='rgba(230,238,248,.15)';hx.lineWidth=1;
  for(let g=0;g<=3;g++){const v=lo+(hiv-lo)*g/3;hx.beginPath();hx.moveTo(x0,HY(v));hx.lineTo(x1,HY(v));hx.stroke();hx.fillText(Math.round(v),6,HY(v)+3);}
  hx.fillText('bpm',6,y1+2);
  // x ticks (same as spectrogram)
  const span=t1-t0,step=span>500?120:(span>250?60:30);
  for(let tt=Math.ceil(t0/step)*step;tt<=t1;tt+=step)hx.fillText(tt+'s',TX(tt)-8,y0+16);
  // HR line (clamp to the panel so a stray fast/slow beat doesn't shoot off the zoomed range)
  const clampY=y=>Math.max(y1,Math.min(y0,y));
  hx.strokeStyle='#f472b6';hx.lineWidth=1.6;hx.beginPath();let pen=false;
  for(let i=0;i<hr.t.length;i++){const tp=hr.t[i];if(tp<t0||tp>t1){pen=false;continue;}const X=TX(tp),Y=clampY(HY(hr.v[i]));if(!pen){hx.moveTo(X,Y);pen=true;}else hx.lineTo(X,Y);}
  hx.stroke();
  // cursor aligned with the spectrogram
  if(G.t&&G.t.length){const cxpos=x0+((x1-x0)*(G.t[curCol]-t0)/((t1-t0)||1));
    hx.strokeStyle='rgba(255,255,255,.7)';hx.lineWidth=1;hx.beginPath();hx.moveTo(cxpos,y1);hx.lineTo(cxpos,y0);hx.stroke();}
}

// spectrum of the current column
function drawSpec(){
  const dpr=window.devicePixelRatio||1,W=sc.clientWidth,H=210;
  sc.width=W*dpr;sc.height=H*dpr;sx.setTransform(dpr,0,0,dpr,0,0);sx.clearRect(0,0,W,H);
  const fb=G.fb,colv=G.heat[curCol],rv=(G.activeRidge||G.ridge)[curCol],tt=G.t[curCol];
  document.getElementById('spectitle').textContent='spectrum @ t = '+tt+' s'+(rv!=null?('   ·   ridge '+rv+' bpm'):'');
  const x0=52,x1=W-14,y0=H-24,y1=10;
  const X=b=>x0+(x1-x0)*(b/YMAX),Y=v=>y0+(y1-y0)*v;
  let mx=0;for(let k=0;k<colv.length;k++){if(fb[k]<=YMAX&&colv[k]>mx)mx=colv[k];}if(mx<=0)mx=1;
  // category bands
  const bn=[0,15,22,30,YMAX];
  for(let i=0;i<bn.length-1;i++){sx.fillStyle=BC[i%4];sx.fillRect(X(bn[i]),y1,X(bn[i+1])-X(bn[i]),y0-y1);}
  // axis + ticks
  sx.strokeStyle='rgba(255,255,255,.2)';sx.beginPath();sx.moveTo(x0,y0);sx.lineTo(x1,y0);sx.stroke();
  sx.fillStyle='rgba(230,238,248,.75)';sx.font='11px monospace';
  [0,15,22,30,45,60].forEach(b=>{sx.fillText(b,X(b)-4,y0+15);});
  sx.fillText('bpm',x1-24,y0+15);
  // ridge marker
  if(rv!=null){sx.strokeStyle='#f59e0b';sx.lineWidth=1.5;sx.setLineDash([3,3]);sx.beginPath();sx.moveTo(X(rv),y1);sx.lineTo(X(rv),y0);sx.stroke();sx.setLineDash([]);}
  // spectrum line
  sx.strokeStyle='#22d3ee';sx.lineWidth=1.8;sx.beginPath();let started=false;
  for(let k=0;k<colv.length;k++){if(fb[k]>YMAX)break;const px=X(fb[k]),py=Y(colv[k]/mx);if(!started){sx.moveTo(px,py);started=true;}else sx.lineTo(px,py);}
  sx.stroke();
}

// hover/click the spectrogram -> pick the time column
function pick(e){const r=cv.getBoundingClientRect();const mxp=(e.clientX-r.left);const c=Math.round((mxp-G.x0)/G.cw-0.5);
  const nc=(c<0?0:(c>=G.t.length?G.t.length-1:c));if(nc!==curCol){curCol=nc;render();}}
cv.addEventListener('mousemove',e=>{if(e.buttons||true)pick(e);});
cv.addEventListener('click',pick);
window.addEventListener('resize',()=>{if(G.off)render();});
window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')document.getElementById('prev').click();
  if(e.key==='ArrowRight')document.getElementById('next').click();});
draw();
</script>
"""

if __name__ == "__main__":
    main()
