#!/usr/bin/env python
"""
signal_explorer.py — self-contained interactive HTML to deep-dive ONE recording's
average-RR pipeline. Embeds THREE spectrograms (raw Artifact / Butterworth /
Butter+detrend); a selector in the page switches between them live, so you can see
what each stage does to the spectrum and which peak the ridge picks — all vs the
reference. Data is pre-computed in Python; interaction runs in the browser.

    py signal_explorer.py --rec-id Exp2/002
    py signal_explorer.py --rec-id Exp2/pilot01 --specB-window 32 --detrend-baseline 5

Panels: (1) result vs reference over time, (2) spectrogram + ridge + match strip,
(3) selected-column spectrum (selected peak vs global-max vs reference),
(4) signal stages around the cursor. Scrub with mouse / slider / arrow keys.
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bw_bp_sweep as S

RR_EDGES = [12.0, 30.0]
OUT_DEFAULT = r"C:\Users\RACHEL~1\AppData\Local\Temp\claude\C--Users-RachelMizrahi-OneDrive---CardiacSense-Documents-GitHub-BBBRR\3bbdb805-5d0d-4f52-b50c-def0402d0def\scratchpad"

TEMPLATE = r"""<title>Signal Explorer — __RID__</title>
<style>
  :root{--bg:#0c111c;--panel:#121a29;--line:#243349;--ink:#e6eef8;--mut:#8ba0ba;
    --cyan:#22d3ee;--green:#34d399;--red:#f87171;--grey:#94a3b8;--blue:#60a5fa;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,Segoe UI,sans-serif}
  .wrap{max-width:1200px;margin:0 auto;padding:18px 20px 60px}
  h1{font-size:20px;margin:0 0 2px;font-weight:700}
  .sub{color:var(--mut);font-size:13px;margin-bottom:12px}
  .readout{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 8px;font-family:"IBM Plex Mono",ui-monospace,monospace}
  .readout span{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:5px 11px;font-size:12.5px}
  .readout b{color:var(--cyan)} .readout .r{color:var(--green)} .readout .e{color:var(--red)}
  .vsel{display:flex;gap:8px;margin:6px 0 14px;flex-wrap:wrap}
  .vbtn{background:var(--panel);border:1px solid var(--line);color:var(--mut);border-radius:8px;
    padding:7px 14px;font-size:13px;cursor:pointer;font-family:inherit;transition:.12s}
  .vbtn:hover{border-color:var(--cyan)}
  .vbtn.on{background:var(--cyan);color:#04222a;border-color:var(--cyan);font-weight:600}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 12px 8px;margin:12px 0}
  .panel h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 8px;font-weight:600;font-family:"IBM Plex Mono",monospace}
  canvas{width:100%;display:block;border-radius:6px;cursor:crosshair}
  .ctl{display:flex;align-items:center;gap:12px;margin:6px 2px 0}
  input[type=range]{flex:1;accent-color:var(--cyan)}
  .leg{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--mut);margin-top:6px}
  .leg i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-inline-end:5px;vertical-align:-1px}
  .hint{color:var(--mut);font-size:12px;margin-top:4px}
</style>
<div class="wrap">
  <h1>Signal Explorer · <span id="rid"></span></h1>
  <div class="sub">average-RR pipeline · channel Artifact · ridge band [6–54 bpm] · switch the spectrogram source below and scrub time</div>

  <div class="vsel">
    <button class="vbtn" data-v="raw">Raw Artifact</button>
    <button class="vbtn" data-v="butter">Butter</button>
    <button class="vbtn on" data-v="detrend">Butter + detrend</button>
    <button class="vbtn" data-v="fence">Butter + fenced detrend</button>
    <button class="vbtn" data-v="mapas">MAPAS + detrend</button>
    <button class="vbtn" data-v="legacy">Legacy (detrend-peaks)</button>
  </div>

  <div class="readout">
    <span>t = <b id="rt">–</b> s</span>
    <span>our RR = <b id="ro">–</b> bpm</span>
    <span>reference = <span class="r" id="rr">–</span> bpm</span>
    <span>|error| = <span class="e" id="re">–</span> bpm</span>
    <span>prominence = <b id="rp">–</b></span>
    <span>category = <b id="rc">–</b></span>
  </div>

  <div class="panel">
    <h2>Result vs reference · RR over time — our ridge (cyan) vs reference (white), over the RR-category bands</h2>
    <canvas id="result" height="200"></canvas>
    <div class="leg">
      <span><i style="background:var(--cyan)"></i>our RR (ridge)</span>
      <span><i style="background:#f59e0b"></i>Butter+detrend (previous, for compare)</span>
      <span><i style="background:#fff"></i>reference RR</span>
      <span><i style="background:rgba(96,165,250,.5)"></i>RR&lt;12</span>
      <span><i style="background:rgba(52,211,153,.5)"></i>RR 12–30</span>
      <span><i style="background:rgba(245,158,11,.5)"></i>RR&gt;30</span>
    </div>
  </div>

  <div class="panel">
    <h2 id="heatTitle">Spectrogram · our ridge (cyan) vs reference (white) · category-match strip below</h2>
    <canvas id="heat" height="300"></canvas>
    <canvas id="strip" height="16"></canvas>
    <div class="ctl"><input type="range" id="slider" min="0" value="0" step="1"><span id="pos" style="font-family:monospace;color:var(--mut);font-size:12px"></span></div>
    <div class="leg">
      <span><i style="background:var(--cyan)"></i>our ridge</span>
      <span><i style="background:#f59e0b"></i>Butter+detrend (previous)</span>
      <span><i style="background:#fff"></i>reference RR</span>
      <span><i style="background:var(--green)"></i>match</span><span><i style="background:var(--red)"></i>mismatch</span>
      <span><i style="background:var(--grey)"></i>no reference</span>
    </div>
    <div class="hint">Mouse over the heatmap, drag the slider, or use ← → keys.</div>
  </div>

  <div class="panel">
    <h2>Selected column spectrum · <span id="specT">–</span> s — which peak was chosen</h2>
    <canvas id="spec" height="240"></canvas>
    <div class="leg">
      <span><i style="background:var(--green)"></i>selected in-band peak (our RR)</span>
      <span><i style="background:var(--grey)"></i>global max (DC/junk trap)</span>
      <span><i style="background:#fff"></i>reference RR</span>
      <span>shaded = excluded bands (&lt;6 &amp; &gt;54 bpm)</span>
    </div>
  </div>

  <div class="panel">
    <h2>Signal stages around the cursor (±40 s) · raw → band-pass → detrend</h2>
    <canvas id="stage" height="330"></canvas>
  </div>
</div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
$('rid').textContent = DATA.rid;
const SP = DATA.spec, ST = DATA.stage, EDG = DATA.rrEdges, LO = DATA.loBpm, HI = DATA.hiBpm;
const nCol = SP.times.length, nBin = SP.freqsBpm.length;
const VLABEL = {raw:'Raw Artifact', butter:'Butter (band-pass)', detrend:'Butter + detrend',
  fence:'Butter + fenced detrend', mapas:'MAPAS + detrend', legacy:'Legacy (detrend-peaks)'};
let variant = 'detrend', V = SP.variants[variant];
let cur = Math.floor(nCol/2);
$('slider').max = nCol-1; $('slider').value = cur;
const YMAXbpm = 66;

const STOPS=[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function cmap(v){v=Math.max(0,Math.min(1,v));const x=v*(STOPS.length-1);const i=Math.floor(x),f=x-i;
  const a=STOPS[i],b=STOPS[Math.min(i+1,STOPS.length-1)];return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}
function getCss(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}

let off, dbMin, dbMax;
function buildHeat(){
  off=document.createElement('canvas'); off.width=nCol; off.height=nBin;
  const octx=off.getContext('2d'); const img=octx.createImageData(nCol,nBin);
  dbMax=-1e9; const DB=[];
  for(let c=0;c<nCol;c++){const col=V.lin[c],d=new Float32Array(nBin);
    for(let b=0;b<nBin;b++){const v=20*Math.log10(col[b]+1e-9);d[b]=v;if(v>dbMax)dbMax=v;} DB.push(d);}
  dbMin=dbMax-60;
  for(let c=0;c<nCol;c++)for(let b=0;b<nBin;b++){let t=(DB[c][b]-dbMin)/(dbMax-dbMin);const rgb=cmap(t);
    const y=nBin-1-b,k=(y*nCol+c)*4;img.data[k]=rgb[0];img.data[k+1]=rgb[1];img.data[k+2]=rgb[2];img.data[k+3]=255;}
  octx.putImageData(img,0,0);
}
function binToY(bpm,h){return h-(bpm/YMAXbpm)*h;}

function drawHeat(){
  const cv=$('heat'),ctx=cv.getContext('2d');const w=cv.width=cv.clientWidth*devicePixelRatio;const h=cv.height=300*devicePixelRatio;
  ctx.imageSmoothingEnabled=true; const topBpm=SP.freqsBpm[nBin-1];
  ctx.drawImage(off,0,binToY(topBpm,h),w,h-binToY(topBpm,h));
  ctx.fillStyle='#0c111c';ctx.fillRect(0,0,w,binToY(topBpm,h));
  const gl=(bpm,col,dash)=>{ctx.strokeStyle=col;ctx.setLineDash(dash||[]);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,binToY(bpm,h));ctx.lineTo(w,binToY(bpm,h));ctx.stroke();ctx.setLineDash([]);};
  EDG.forEach(e=>gl(e,'rgba(255,255,255,.25)',[4,4])); gl(LO,'rgba(148,163,184,.7)',[3,3]); gl(HI,'rgba(248,113,113,.7)',[3,3]);
  const xOf=c=>c/(nCol-1)*w;
  const line=(arr,col,wd)=>{ctx.strokeStyle=col;ctx.lineWidth=wd;ctx.beginPath();let s=false;
    for(let c=0;c<nCol;c++){const v=arr[c];if(v==null||isNaN(v)){s=false;continue;}const x=xOf(c),y=binToY(v,h);if(!s){ctx.moveTo(x,y);s=true;}else ctx.lineTo(x,y);}ctx.stroke();};
  line(SP.refRR,'rgba(255,255,255,.9)',2*devicePixelRatio);
  if(variant!=='detrend')line(SP.variants['detrend'].selRR,'rgba(245,158,11,.7)',1.2*devicePixelRatio);  // ghost: previous state
  ctx.fillStyle=getCss('--cyan');for(let c=0;c<nCol;c++){const v=V.selRR[c];if(v==null||isNaN(v))continue;ctx.beginPath();ctx.arc(xOf(c),binToY(v,h),1.6*devicePixelRatio,0,7);ctx.fill();}
  ctx.strokeStyle=getCss('--cyan');ctx.lineWidth=1.5*devicePixelRatio;ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(xOf(cur),0);ctx.lineTo(xOf(cur),h);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='rgba(230,238,248,.75)';ctx.font=(11*devicePixelRatio)+'px monospace';[10,20,30,40,50,60].forEach(b=>ctx.fillText(b,3,binToY(b,h)-2));
}
function drawStrip(){
  const cv=$('strip'),ctx=cv.getContext('2d');const w=cv.width=cv.clientWidth*devicePixelRatio;const h=cv.height=16*devicePixelRatio;
  const bw=w/nCol;for(let c=0;c<nCol;c++){const m=V.match[c];ctx.fillStyle=m===1?getCss('--green'):(m===0?getCss('--red'):'#33415580');ctx.fillRect(c*bw,0,Math.ceil(bw)+1,h);}
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();ctx.moveTo(cur/(nCol-1)*w,0);ctx.lineTo(cur/(nCol-1)*w,h);ctx.stroke();
}
function drawResult(){
  const cv=$('result'),ctx=cv.getContext('2d');const w=cv.width=cv.clientWidth*devicePixelRatio;const h=cv.height=200*devicePixelRatio;
  ctx.clearRect(0,0,w,h);const x0=30*devicePixelRatio,x1=w-6*devicePixelRatio,y0=h-4*devicePixelRatio,y1=6*devicePixelRatio;
  const X=c=>x0+c/(nCol-1)*(x1-x0),Y=bpm=>y0-(bpm/YMAXbpm)*(y0-y1);
  [[0,EDG[0],'rgba(96,165,250,.10)'],[EDG[0],EDG[1],'rgba(52,211,153,.10)'],[EDG[1],YMAXbpm,'rgba(245,158,11,.10)']].forEach(b=>{ctx.fillStyle=b[2];ctx.fillRect(x0,Y(b[1]),x1-x0,Y(b[0])-Y(b[1]));});
  EDG.forEach(e=>{ctx.strokeStyle='rgba(255,255,255,.18)';ctx.setLineDash([4,4]);ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x0,Y(e));ctx.lineTo(x1,Y(e));ctx.stroke();ctx.setLineDash([]);});
  const plot=(arr,c,wd)=>{ctx.strokeStyle=c;ctx.lineWidth=wd;ctx.beginPath();let s=false;for(let i=0;i<nCol;i++){const v=arr[i];if(v==null||isNaN(v)){s=false;continue;}const x=X(i),y=Y(v);if(!s){ctx.moveTo(x,y);s=true;}else ctx.lineTo(x,y);}ctx.stroke();};
  plot(SP.refRR,'rgba(255,255,255,.9)',2*devicePixelRatio);
  if(variant!=='detrend')plot(SP.variants['detrend'].selRR,'rgba(245,158,11,.7)',1.2*devicePixelRatio);  // ghost: previous state
  plot(V.selRR,getCss('--cyan'),1.5*devicePixelRatio);
  ctx.strokeStyle=getCss('--cyan');ctx.setLineDash([5,4]);ctx.lineWidth=1.4*devicePixelRatio;ctx.beginPath();ctx.moveTo(X(cur),y1);ctx.lineTo(X(cur),y0);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='rgba(230,238,248,.7)';ctx.font=(10*devicePixelRatio)+'px monospace';[12,30,50].forEach(b=>ctx.fillText(b,3,Y(b)-2));
}
function drawSpec(){
  const cv=$('spec'),ctx=cv.getContext('2d');const w=cv.width=cv.clientWidth*devicePixelRatio;const h=cv.height=240*devicePixelRatio;
  ctx.clearRect(0,0,w,h);const col=V.lin[cur];
  let mx=0;for(let b=0;b<nBin;b++){const f=SP.freqsBpm[b];if(f>=LO&&f<=HI&&col[b]>mx)mx=col[b];}  // scale to IN-BAND peak
  if(mx<=0){for(let b=0;b<nBin;b++)mx=Math.max(mx,col[b]);} if(mx<=0)mx=1;
  const pad=34*devicePixelRatio,x0=pad,x1=w-8*devicePixelRatio,y0=h-22*devicePixelRatio,y1=8*devicePixelRatio;
  const X=bpm=>x0+(bpm/YMAXbpm)*(x1-x0),Y=v=>y0+(v)*(y1-y0);
  ctx.fillStyle='rgba(148,163,184,.16)';ctx.fillRect(X(0),y1,X(LO)-X(0),y0-y1);
  ctx.fillStyle='rgba(248,113,113,.15)';ctx.fillRect(X(HI),y1,X(60)-X(HI),y0-y1);
  ctx.strokeStyle='rgba(255,255,255,.2)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x1,y0);ctx.stroke();
  ctx.fillStyle='rgba(230,238,248,.7)';ctx.font=(11*devicePixelRatio)+'px monospace';[0,10,20,30,40,50,60].forEach(b=>ctx.fillText(b,X(b)-4,y0+15*devicePixelRatio));
  ctx.strokeStyle=getCss('--blue');ctx.lineWidth=1.6*devicePixelRatio;ctx.beginPath();
  for(let b=0;b<nBin;b++){const x=X(SP.freqsBpm[b]),y=Y(col[b]/mx);if(b===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
  const gi=V.gmaxIdx[cur],si=V.selIdx[cur];
  const dot=(bpm,val,c,r)=>{ctx.fillStyle=c;ctx.beginPath();ctx.arc(X(bpm),Y(val),r*devicePixelRatio,0,7);ctx.fill();};
  if(gi>=0)dot(SP.freqsBpm[gi],Math.min(col[gi]/mx,1.06),getCss('--grey'),5);
  if(si>=0)dot(SP.freqsBpm[si],Math.min(col[si]/mx,1.06),getCss('--green'),6.5);
  ctx.fillStyle='rgba(139,160,186,.7)';ctx.font=(10*devicePixelRatio)+'px monospace';
  ctx.fillText('scaled to in-band peak · DC/low clipped',x0+4*devicePixelRatio,y1+11*devicePixelRatio);
  const rf=SP.refRR[cur];if(rf!=null&&!isNaN(rf)){ctx.strokeStyle='rgba(255,255,255,.85)';ctx.lineWidth=1.4*devicePixelRatio;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(X(rf),y1);ctx.lineTo(X(rf),y0);ctx.stroke();ctx.setLineDash([]);}
  const vl=(bpm,c)=>{ctx.strokeStyle=c;ctx.lineWidth=1;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(X(bpm),y1);ctx.lineTo(X(bpm),y0);ctx.stroke();ctx.setLineDash([]);};
  vl(LO,'rgba(148,163,184,.6)');vl(HI,'rgba(248,113,113,.6)');
}
function drawStage(){
  const cv=$('stage'),ctx=cv.getContext('2d');const w=cv.width=cv.clientWidth*devicePixelRatio;const h=cv.height=330*devicePixelRatio;
  ctx.clearRect(0,0,w,h);const tC=SP.times[cur],W=40,t0=tC-W,t1=tC+W;
  const names=['raw','butter','detrend'],cols=['#64748b',getCss('--blue'),getCss('--green')];
  const i0=Math.max(0,Math.floor((t0-ST.t0)*ST.fs)),i1=Math.min(ST.t.length-1,Math.ceil((t1-ST.t0)*ST.fs));const ph=h/3;
  names.forEach((nm,p)=>{const arr=ST[nm];let mn=1e18,mx=-1e18;for(let i=i0;i<=i1;i++){const v=arr[i];if(v<mn)mn=v;if(v>mx)mx=v;}if(mx<=mn)mx=mn+1;
    const yTop=p*ph+8*devicePixelRatio,yBot=(p+1)*ph-8*devicePixelRatio;const X=t=>((t-t0)/(t1-t0))*w,Y=v=>yBot-((v-mn)/(mx-mn))*(yBot-yTop);
    ctx.strokeStyle=cols[p];ctx.lineWidth=1.3*devicePixelRatio;ctx.beginPath();
    for(let i=i0;i<=i1;i++){const t=ST.t0+i/ST.fs,x=X(t),y=Y(arr[i]);if(i===i0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}ctx.stroke();
    ctx.fillStyle=getCss('--mut');ctx.font=(11*devicePixelRatio)+'px monospace';ctx.fillText(nm,6,yTop+12*devicePixelRatio);
    ctx.strokeStyle='rgba(148,163,184,.15)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,yBot);ctx.lineTo(w,yBot);ctx.stroke();});
  ctx.strokeStyle=getCss('--cyan');ctx.lineWidth=1.5*devicePixelRatio;ctx.setLineDash([5,4]);const xc=((tC-t0)/(t1-t0))*w;ctx.beginPath();ctx.moveTo(xc,0);ctx.lineTo(xc,h);ctx.stroke();ctx.setLineDash([]);
}
function fmt(x,d){return (x==null||isNaN(x))?'–':Number(x).toFixed(d===undefined?1:d);}
function binName(rr){if(rr==null||isNaN(rr))return '–';if(rr<EDG[0])return 'RR<'+EDG[0];if(rr<EDG[1])return 'RR'+EDG[0]+'–'+EDG[1];return 'RR>'+EDG[1];}
function update(){
  cur=Math.max(0,Math.min(nCol-1,cur));$('slider').value=cur;
  const o=V.selRR[cur],r=SP.refRR[cur];
  $('rt').textContent=fmt(SP.times[cur],0);$('ro').textContent=fmt(o);$('rr').textContent=fmt(r);
  $('re').textContent=(o==null||r==null||isNaN(o)||isNaN(r))?'–':fmt(Math.abs(o-r));
  $('rp').textContent=fmt(V.prom,1);
  $('rc').textContent=binName(o)+((r!=null&&!isNaN(r))?(binName(o)===binName(r)?' ✓':' ✗ (ref '+binName(r)+')'):'');
  $('specT').textContent=fmt(SP.times[cur],0);$('pos').textContent=(cur+1)+' / '+nCol;
  $('heatTitle').textContent='Spectrogram · '+VLABEL[variant]+' · prominence '+fmt(V.prom,1)+' · ridge (cyan) vs reference (white)';
  drawResult();drawHeat();drawStrip();drawSpec();drawStage();
}
function setVariant(v){variant=v;V=SP.variants[v];buildHeat();
  document.querySelectorAll('.vbtn').forEach(b=>b.classList.toggle('on',b.dataset.v===v));update();}
document.querySelectorAll('.vbtn').forEach(b=>b.addEventListener('click',()=>setVariant(b.dataset.v)));
function colFromX(cv,ev){const r=cv.getBoundingClientRect();return Math.round((ev.clientX-r.left)/r.width*(nCol-1));}
['heat','strip','result'].forEach(id=>{const cv=$(id);
  cv.addEventListener('mousemove',e=>{if(e.buttons||id==='heat'){cur=colFromX(cv,e);update();}});
  cv.addEventListener('click',e=>{cur=colFromX(cv,e);update();});});
$('slider').addEventListener('input',e=>{cur=+e.target.value;update();});
window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){cur--;update();}if(e.key==='ArrowRight'){cur++;update();}});
window.addEventListener('resize',update);
buildHeat();update();
</script>
"""


def _variant(sigy, fs, cfg, tat, tar, offset, win):
    from respiration_rr.ppg.spectrogram import compute_spectrogram
    sp = compute_spectrogram(np.asarray(sigy, np.float64) - np.mean(sigy), fs, win, 1.1)
    freqs = sp["freqs"]; bpmf = freqs * 60.0; P = sp["power_lin"]; times = sp["times"]
    band = (freqs >= cfg.rr_band_low_hz) & (freqs <= cfg.rr_band_high_hz)
    bidx = np.where(band)[0]
    selIdx = np.full(P.shape[1], -1, int); gmaxIdx = np.full(P.shape[1], -1, int)
    for c in range(P.shape[1]):
        col = P[:, c]; gmaxIdx[c] = int(np.argmax(col))
        selIdx[c] = int(bidx[np.argmax(col[bidx])]) if bidx.size else -1
    selRR = np.where(selIdx >= 0, bpmf[selIdx], np.nan)
    Pb = P[bidx, :]; pk = Pb.max(0); med = np.median(Pb, 0)
    prom = float(np.nanmedian(np.divide(pk, med, out=np.full_like(pk, np.nan), where=med > 0)))
    refRR = np.interp(times + offset, tat, tar, left=np.nan, right=np.nan)
    ob = np.digitize(selRR, RR_EDGES); rb = np.digitize(refRR, RR_EDGES)
    match = np.where(np.isfinite(refRR) & np.isfinite(selRR), (ob == rb).astype(int), -1)
    def clean(a): return [None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 3) for v in a]
    return {"times": times, "bpmf": bpmf,
            "lin": [np.round(P[:, c], 5).tolist() for c in range(P.shape[1])],
            "selIdx": selIdx.tolist(), "gmaxIdx": gmaxIdx.tolist(),
            "selRR": clean(selRR), "refRR": clean(refRR), "match": match.tolist(), "prom": round(prom, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-id", default=None); ap.add_argument("--recording", default=None)
    ap.add_argument("--edf", default=None); ap.add_argument("--watch", default=None)
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--channel", default="Artifact")
    ap.add_argument("--detrend-baseline", type=float, default=5.0)
    ap.add_argument("--env-gate", type=float, default=0.5,
                    help="envelope-distance fence: keep peaks with prominence >= this * P2P reference")
    ap.add_argument("--env-win", type=float, default=20.0,
                    help="fence reference window (s): >0 = LOCAL adaptive threshold, 0 = global median")
    ap.add_argument("--specB-window", type=float, default=32.0)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    import dataclasses
    from respiration_rr.settings import PPG as _PPG
    cfg = dataclasses.replace(_PPG, rr_band_high_hz=0.9)
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr
    from respiration_rr.ppg.bw_methods import butterworth, legacy_detrend, mapas

    if args.rec_id:
        recs = [r for r in S.discover(args.data_root) if r[0] == args.rec_id] or [S._resolve_inputs(args)]
    else:
        recs = [S._resolve_inputs(args)]
    rid, edf, csv = recs[0]
    print("Recording:", rid)

    ppg, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg, sig)
    tat, tar = reference_average_rr(ref)
    x = np.asarray(sig.channels[args.channel], np.float64); fs = float(sig.fs)
    low, high = cfg.bw_band_low_hz, cfg.bw_band_high_hz
    bp = butterworth(x, fs, low, high)
    det = legacy_detrend(bp, fs, low, high, baseline_sec=args.detrend_baseline)

    sig_variants = {                                    # method menu (each best-tuned)
        "raw": x,
        "butter": bp,
        "detrend": det,                                             # Butter + detrend (baseline 5)
        "fence": legacy_detrend(bp, fs, low, high, baseline_sec=args.detrend_baseline,
                                env_gate_frac=args.env_gate, env_gate_win=args.env_win),  # + local envelope fence
        "mapas": legacy_detrend(mapas(x, fs, low, high), fs, low, high, baseline_sec=15.0),
        "legacy": legacy_detrend(x, fs, low, high, baseline_sec=5.0),
    }
    var_out = {}
    for k, s in sig_variants.items():
        v = _variant(s, fs, cfg, tat, tar, offset, args.specB_window)
        var_out[k] = {kk: v[kk] for kk in ("lin", "selIdx", "gmaxIdx", "selRR", "refRR", "match", "prom")}
    # shared axes / reference from any variant (identical across the three)
    v0 = _variant(det, fs, cfg, tat, tar, offset, args.specB_window)  # for axes
    spec = {"times": np.round(v0["times"], 1).tolist(), "freqsBpm": np.round(v0["bpmf"], 3).tolist(),
            "refRR": var_out["detrend"]["refRR"], "variants": var_out}
    for vo in var_out.values():
        vo.pop("refRR", None)  # refRR is shared

    disp_fs = min(fs, 64.0); td = np.arange(0, x.size / fs, 1.0 / disp_fs); ts = np.arange(x.size) / fs
    def rs(a): return np.round(np.interp(td, ts, a), 3).tolist()
    data = {"rid": rid, "loBpm": cfg.rr_band_low_hz * 60, "hiBpm": cfg.rr_band_high_hz * 60, "rrEdges": RR_EDGES,
            "spec": spec,
            "stage": {"t0": 0.0, "fs": disp_fs, "t": td.round(2).tolist(),
                      "raw": rs(x), "butter": rs(bp), "detrend": rs(det)}}
    print("prominence:", {k: var_out[k]["prom"] for k in sig_variants})
    html = TEMPLATE.replace("__RID__", rid.replace("/", "_")).replace("__DATA__", json.dumps(data))
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "explorer_%s.html" % rid.replace("/", "_"))
    open(outp, "w", encoding="utf-8").write(html)
    print("saved ->", outp, "(%.0f KB)" % (len(html) / 1024))


if __name__ == "__main__":
    main()
