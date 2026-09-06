#!/usr/bin/env python
"""
route_fast_explorer.py — interactive HTML comparing, per recording:
  * Artifact RAW      (no band-pass) argmax ridge
  * Artifact BW       (BP 0.1-1.0)   argmax ridge
  * route_fast        (BW if >=22 bpm else median IR RSA/RIIV/AUC spline)
against the reference. Shows the raw Artifact spectrogram, RR over time (all three
+ reference + params-median), and THREE category-match strips so you see which
segments each method gets right.

Config = the winner: exact Hann window 48 s, hop 5 s, argmax ridge 0.10-0.80 Hz.

    py route_fast_explorer.py --rec-id Exp2/002
    py route_fast_explorer.py --rec-id Exp2/pilot01
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import bw_bp_sweep as S
from param_category_sweep import (stft_power, ridge_rr, envelope_on_grid,
                                  RR_EDGES, SEG_SEC, HOP_SEC, WIN_SEC, GRID_FS)
from param_fusion_sweep import seg_medians

OUT_DEFAULT = r"C:\Users\RACHEL~1\AppData\Local\Temp\claude\C--Users-RachelMizrahi-OneDrive---CardiacSense-Documents-GitHub-BBBRR\3bbdb805-5d0d-4f52-b50c-def0402d0def\scratchpad"
PARAM_CH, RECON, PARAMS = "IR", "spl", ("RSA", "RIIV", "AUC")
F_HI_DISP = 1.05
TAU = 22.0

TEMPLATE = r"""<title>Artifact raw vs BW vs route_fast — __RID__</title>
<style>
  :root{--bg:#0c111c;--panel:#121a29;--line:#243349;--ink:#e6eef8;--mut:#8ba0ba;
    --cyan:#22d3ee;--green:#34d399;--red:#f87171;--grey:#94a3b8;--amber:#f59e0b;--violet:#a78bfa;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,Segoe UI,sans-serif}
  .wrap{max-width:1200px;margin:0 auto;padding:18px 20px 60px}
  h1{font-size:20px;margin:0 0 2px;font-weight:700}
  .sub{color:var(--mut);font-size:13px;margin-bottom:10px}
  .scores{display:flex;gap:10px;margin:8px 0 12px;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace}
  .scores .s{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:7px 13px;font-size:13px}
  .scores .s b{font-size:16px}
  .scores .rw b{color:var(--violet)} .scores .bw b{color:var(--cyan)} .scores .rt b{color:var(--amber)}
  .readout{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0;font-family:"IBM Plex Mono",monospace}
  .readout span{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:5px 11px;font-size:12.5px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 12px 8px;margin:12px 0}
  .panel h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:0 0 8px;font-weight:600;font-family:"IBM Plex Mono",monospace}
  canvas{width:100%;display:block;border-radius:6px;cursor:crosshair}
  .striplab{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--mut);margin:7px 0 2px}
  .ctl{display:flex;align-items:center;gap:12px;margin:6px 2px 0}
  input[type=range]{flex:1;accent-color:var(--cyan)}
  .leg{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--mut);margin-top:6px}
  .leg i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-inline-end:5px;vertical-align:-1px}
</style>
<div class="wrap">
  <h1>Artifact raw vs BW vs route_fast · <span id="rid"></span></h1>
  <div class="sub">RAW (no band-pass) · BW (BP 0.1-1.0) · route_fast (BW if &ge;22 bpm else median IR RSA/RIIV/AUC spline) · window 48 s · categories &lt;15 / 15-22 / 22-30 / &gt;30</div>
  <div class="scores">
    <div class="s rw">Artifact raw: <b id="accraw">–</b>%</div>
    <div class="s bw">Artifact BW: <b id="accbw">–</b>%</div>
    <div class="s rt">route_fast: <b id="accrt">–</b>%</div>
  </div>

  <div class="readout">
    <span>t = <b id="rt">–</b> s</span>
    <span>reference = <b id="rr" style="color:var(--green)">–</b></span>
    <span>raw = <b id="rraw" style="color:var(--violet)">–</b></span>
    <span>BW = <b id="rbw" style="color:var(--cyan)">–</b></span>
    <span>params-med = <b id="rpar" style="color:var(--grey)">–</b></span>
    <span>route → <b id="rroute" style="color:var(--amber)">–</b></span>
    <span>source = <b id="rsrc">–</b></span>
  </div>

  <div class="panel">
    <h2>RR over time — reference (white) · raw (violet) · BW (cyan) · route_fast (amber) · params-med (grey)</h2>
    <canvas id="result" height="230"></canvas>
  </div>

  <div class="panel">
    <h2>Artifact RAW spectrogram · raw ridge (violet) · BW ridge (cyan) · route_fast (amber) · reference (white)</h2>
    <canvas id="heat" height="300"></canvas>
    <div class="striplab">category match — Artifact raw</div><canvas id="stripRaw" height="14"></canvas>
    <div class="striplab">category match — Artifact BW</div><canvas id="stripBw" height="14"></canvas>
    <div class="striplab">category match — route_fast</div><canvas id="stripRt" height="14"></canvas>
    <div class="ctl"><input type="range" id="slider" min="0" value="0" step="1"><span id="pos" style="font-family:monospace;color:var(--mut);font-size:12px"></span></div>
    <div class="leg"><span><i style="background:var(--green)"></i>correct category</span><span><i style="background:var(--red)"></i>wrong</span><span><i style="background:#33415580"></i>no reference</span></div>
  </div>

  <div class="panel">
    <h2>RAW column spectrum · <span id="specT">–</span> s</h2>
    <canvas id="spec" height="220"></canvas>
  </div>
</div>
<script>
const D = __DATA__, $ = id => document.getElementById(id);
$('rid').textContent = D.rid; $('accraw').textContent = D.accRaw; $('accbw').textContent = D.accBw; $('accrt').textContent = D.accRoute;
const T = D.times, nCol = T.length, F = D.freqsBpm, nBin = F.length, EDG = D.rrEdges, LO = D.loBpm, HI = D.hiBpm;
const YMAX = 66; let cur = Math.floor(nCol/2);
$('slider').max = nCol-1; $('slider').value = cur;
const STOPS=[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[250,193,39],[252,255,164]];
function cmap(v){v=Math.max(0,Math.min(1,v));const x=v*(STOPS.length-1),i=Math.floor(x),f=x-i,a=STOPS[i],b=STOPS[Math.min(i+1,7)];return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
let off,dbMin,dbMax;
function buildHeat(){off=document.createElement('canvas');off.width=nCol;off.height=nBin;const o=off.getContext('2d'),img=o.createImageData(nCol,nBin);dbMax=-1e9;const DB=[];
  for(let c=0;c<nCol;c++){const col=D.heat[c],d=new Float32Array(nBin);for(let b=0;b<nBin;b++){const v=20*Math.log10(col[b]+1e-9);d[b]=v;if(v>dbMax)dbMax=v;}DB.push(d);}dbMin=dbMax-60;
  for(let c=0;c<nCol;c++)for(let b=0;b<nBin;b++){let t=(DB[c][b]-dbMin)/(dbMax-dbMin);const g=cmap(t),y=nBin-1-b,k=(y*nCol+c)*4;img.data[k]=g[0];img.data[k+1]=g[1];img.data[k+2]=g[2];img.data[k+3]=255;}o.putImageData(img,0,0);}
function bY(bpm,h){return h-(bpm/YMAX)*h;}
function drawHeat(){const cv=$('heat'),x=cv.getContext('2d'),w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=300*devicePixelRatio;
  x.imageSmoothingEnabled=true;const top=F[nBin-1];x.drawImage(off,0,bY(top,h),w,h-bY(top,h));x.fillStyle='#0c111c';x.fillRect(0,0,w,bY(top,h));
  const gl=(b,c,d)=>{x.strokeStyle=c;x.setLineDash(d||[]);x.lineWidth=1;x.beginPath();x.moveTo(0,bY(b,h));x.lineTo(w,bY(b,h));x.stroke();x.setLineDash([]);};
  EDG.forEach(e=>gl(e,'rgba(255,255,255,.22)',[4,4]));
  const xo=c=>c/(nCol-1)*w;
  const line=(a,c,wd)=>{x.strokeStyle=c;x.lineWidth=wd;x.beginPath();let s=false;for(let i=0;i<nCol;i++){const v=a[i];if(v==null||isNaN(v)){s=false;continue;}const px=xo(i),py=bY(v,h);if(!s){x.moveTo(px,py);s=true;}else x.lineTo(px,py);}x.stroke();};
  line(D.refCol,'rgba(255,255,255,.9)',2*devicePixelRatio);
  line(D.routeCol,css('--amber'),2*devicePixelRatio);
  x.fillStyle=css('--violet');for(let i=0;i<nCol;i++){const v=D.rawRidge[i];if(v==null||isNaN(v))continue;x.beginPath();x.arc(xo(i),bY(v,h),2.3*devicePixelRatio,0,7);x.fill();}
  x.fillStyle=css('--cyan');for(let i=0;i<nCol;i++){const v=D.bwRidge[i];if(v==null||isNaN(v))continue;x.beginPath();x.arc(xo(i),bY(v,h),1.5*devicePixelRatio,0,7);x.fill();}
  x.strokeStyle=css('--cyan');x.setLineDash([5,4]);x.lineWidth=1.4*devicePixelRatio;x.beginPath();x.moveTo(xo(cur),0);x.lineTo(xo(cur),h);x.stroke();x.setLineDash([]);
  x.fillStyle='rgba(230,238,248,.7)';x.font=(11*devicePixelRatio)+'px monospace';[15,22,30,45,60].forEach(b=>x.fillText(b,3,bY(b,h)-2));}
function drawStrip(id,key){const cv=$(id),x=cv.getContext('2d'),w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=14*devicePixelRatio;
  for(const s of D.segs){const x0=s.c0/(nCol-1)*w,x1=(s.c1+1)/(nCol-1)*w,m=s[key];x.fillStyle=m===1?css('--green'):(m===0?css('--red'):'#33415580');x.fillRect(x0,0,x1-x0+1,h);}
  x.strokeStyle='#fff';x.lineWidth=1.3*devicePixelRatio;x.beginPath();x.moveTo(cur/(nCol-1)*w,0);x.lineTo(cur/(nCol-1)*w,h);x.stroke();}
function drawResult(){const cv=$('result'),x=cv.getContext('2d'),w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=230*devicePixelRatio;
  x.clearRect(0,0,w,h);const x0=30*devicePixelRatio,x1=w-6*devicePixelRatio,y0=h-4*devicePixelRatio,y1=6*devicePixelRatio;
  const X=c=>x0+c/(nCol-1)*(x1-x0),Y=b=>y0-(b/YMAX)*(y0-y1);
  const BC=['rgba(96,165,250,.10)','rgba(52,211,153,.10)','rgba(245,158,11,.10)','rgba(244,114,182,.10)'],bn=[0].concat(EDG,[YMAX]);
  for(let i=0;i<bn.length-1;i++){x.fillStyle=BC[i%4];x.fillRect(x0,Y(bn[i+1]),x1-x0,Y(bn[i])-Y(bn[i+1]));}
  const plot=(a,c,wd,dash)=>{x.strokeStyle=c;x.lineWidth=wd;x.setLineDash(dash||[]);x.beginPath();let s=false;for(let i=0;i<nCol;i++){const v=a[i];if(v==null||isNaN(v)){s=false;continue;}const px=X(i),py=Y(v);if(!s){x.moveTo(px,py);s=true;}else x.lineTo(px,py);}x.stroke();x.setLineDash([]);};
  plot(D.parCol,css('--grey'),1.1*devicePixelRatio,[3,3]);
  plot(D.rawRidge,css('--violet'),1.2*devicePixelRatio);
  plot(D.bwRidge,css('--cyan'),1.2*devicePixelRatio);
  plot(D.refCol,'rgba(255,255,255,.9)',2*devicePixelRatio);
  plot(D.routeCol,css('--amber'),2.2*devicePixelRatio);
  x.strokeStyle=css('--cyan');x.setLineDash([5,4]);x.lineWidth=1.3*devicePixelRatio;x.beginPath();x.moveTo(X(cur),y1);x.lineTo(X(cur),y0);x.stroke();x.setLineDash([]);
  x.fillStyle='rgba(230,238,248,.7)';x.font=(10*devicePixelRatio)+'px monospace';[15,22,30,45].forEach(b=>x.fillText(b,3,Y(b)-2));}
function drawSpec(){const cv=$('spec'),x=cv.getContext('2d'),w=cv.width=cv.clientWidth*devicePixelRatio,h=cv.height=220*devicePixelRatio;
  x.clearRect(0,0,w,h);const col=D.heat[cur];let mx=0;for(let b=0;b<nBin;b++){const f=F[b];if(f>=LO&&f<=HI&&col[b]>mx)mx=col[b];}if(mx<=0)mx=Math.max(...col)||1;
  const x0=34*devicePixelRatio,x1=w-8*devicePixelRatio,y0=h-22*devicePixelRatio,y1=8*devicePixelRatio,X=b=>x0+(b/YMAX)*(x1-x0),Y=v=>y0+v*(y1-y0);
  x.fillStyle='rgba(148,163,184,.14)';x.fillRect(X(0),y1,X(LO)-X(0),y0-y1);x.fillStyle='rgba(248,113,113,.13)';x.fillRect(X(HI),y1,X(60)-X(HI),y0-y1);
  x.strokeStyle='rgba(255,255,255,.2)';x.beginPath();x.moveTo(x0,y0);x.lineTo(x1,y0);x.stroke();
  x.fillStyle='rgba(230,238,248,.7)';x.font=(11*devicePixelRatio)+'px monospace';[0,15,22,30,45,60].forEach(b=>x.fillText(b,X(b)-4,y0+15*devicePixelRatio));
  x.strokeStyle=css('--violet');x.lineWidth=1.6*devicePixelRatio;x.beginPath();for(let b=0;b<nBin;b++){const px=X(F[b]),py=Y(col[b]/mx);if(b===0)x.moveTo(px,py);else x.lineTo(px,py);}x.stroke();
  const rf=D.refCol[cur];if(rf!=null&&!isNaN(rf)){x.strokeStyle='#fff';x.setLineDash([4,4]);x.lineWidth=1.3*devicePixelRatio;x.beginPath();x.moveTo(X(rf),y1);x.lineTo(X(rf),y0);x.stroke();x.setLineDash([]);}}
function fmt(v){return (v==null||isNaN(v))?'–':(+v).toFixed(1);}
function segAt(c){for(const s of D.segs)if(c>=s.c0&&c<=s.c1)return s;return null;}
function update(){cur=Math.max(0,Math.min(nCol-1,cur));$('slider').value=cur;const s=segAt(cur);
  $('rt').textContent=fmt(T[cur]);$('rr').textContent=fmt(D.refCol[cur]);$('rraw').textContent=fmt(D.rawRidge[cur]);$('rbw').textContent=fmt(D.bwRidge[cur]);
  $('rpar').textContent=s?fmt(s.par):'–';$('rroute').textContent=s?fmt(s.route):'–';$('rsrc').textContent=s?s.src:'–';
  $('specT').textContent=fmt(T[cur]);$('pos').textContent=(cur+1)+' / '+nCol;
  drawResult();drawHeat();drawStrip('stripRaw','rawOk');drawStrip('stripBw','bwOk');drawStrip('stripRt','rtOk');drawSpec();}
['heat','result'].forEach(id=>{const cv=$(id);cv.addEventListener('mousemove',e=>{if(e.buttons||id==='heat'){const r=cv.getBoundingClientRect();cur=Math.round((e.clientX-r.left)/r.width*(nCol-1));update();}});cv.addEventListener('click',e=>{const r=cv.getBoundingClientRect();cur=Math.round((e.clientX-r.left)/r.width*(nCol-1));update();});});
$('slider').addEventListener('input',e=>{cur=+e.target.value;update();});
window.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){cur--;update();}if(e.key==='ArrowRight'){cur++;update();}});
window.addEventListener('resize',update);buildHeat();update();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-id", default=None); ap.add_argument("--recording", default=None)
    ap.add_argument("--edf", default=None); ap.add_argument("--watch", default=None)
    ap.add_argument("--data-root", default=S.DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    from respiration_rr.settings import PPG as ppg
    from scipy.signal import butter, sosfiltfilt
    from main import run_reference, run_ppg, run_sync
    from respiration_rr.rr_average import reference_average_rr

    rid, edf, csv = ([r for r in S.discover(args.data_root) if r[0] == args.rec_id] or [S._resolve_inputs(args)])[0] \
        if args.rec_id else S._resolve_inputs(args)
    print("Recording:", rid)
    ppg_res, sig = run_ppg(csv); ref = run_reference(edf); offset = run_sync(edf, ppg_res, sig)
    tat, tar = reference_average_rr(ref)
    fs = float(sig.fs); n = int(np.asarray(sig.channels["Artifact"]).size)
    art = np.asarray(sig.channels["Artifact"], np.float64)

    # RAW spectrogram (heatmap) + raw ridge
    times, freqs, Praw = stft_power(art, fs, WIN_SEC, HOP_SEC)
    fkeep = np.where(freqs <= F_HI_DISP)[0]
    Pd = Praw[fkeep, :]; fbpm = (freqs[fkeep] * 60.0)
    rawRidge = ridge_rr(freqs, Praw)
    # BW ridge
    ny = fs / 2.0
    sos = butter(2, [0.10 / ny, 1.0 / ny], btype="band", output="sos")
    _, fbw, Pbw = stft_power(sosfiltfilt(sos, art), fs, WIN_SEC, HOP_SEC)
    bwRidge = ridge_rr(fbw, Pbw)
    refCol = np.interp(times + offset, tat, tar, left=np.nan, right=np.nan)

    segs = np.arange(times[0], times[-1], SEG_SEC)
    raw_seg = seg_medians(times, rawRidge, segs)
    bw_seg = seg_medians(times, bwRidge, segs)
    g_t = np.arange(0.0, n / fs, 1.0 / GRID_FS)
    res = ppg_res.get(PARAM_CH)
    par_rows = []
    for p in PARAMS:
        pr = res.params.get(p) if res and getattr(res, "params", None) else None
        bx = np.asarray(getattr(pr, "series_x", []), np.float64) if pr else np.zeros(0)
        by = np.asarray(getattr(pr, "series_y", []), np.float64) if pr else np.zeros(0)
        if bx.size < 4:
            par_rows.append(np.full(segs.size, np.nan)); continue
        env = envelope_on_grid(bx, by, RECON, g_t, ppg)
        tt, ff, PP = stft_power(env, GRID_FS, WIN_SEC, HOP_SEC)
        par_rows.append(seg_medians(tt, ridge_rr(ff, PP), segs))
    par_rows = np.array(par_rows)

    def bin_(v):
        return int(np.digitize(v, RR_EDGES)) if np.isfinite(v) else -1

    seg_objs = []
    parCol = np.full(times.size, np.nan); routeCol = np.full(times.size, np.nan)
    nseg = ok_raw = ok_bw = ok_rt = 0
    for i, s in enumerate(segs):
        c0 = int(np.searchsorted(times, s)); c1 = int(np.searchsorted(times, s + SEG_SEC)) - 1
        c1 = max(c0, min(c1, times.size - 1))
        pv = par_rows[:, i]; pv = pv[np.isfinite(pv)]
        pmed = float(np.median(pv)) if pv.size else np.nan
        bw = bw_seg[i]; raw = raw_seg[i]
        use_bw = np.isfinite(bw) and bw >= TAU
        route = bw if use_bw else (pmed if np.isfinite(pmed) else bw)
        refv = np.interp(s + SEG_SEC / 2 + offset, tat, tar, left=np.nan, right=np.nan)
        rc = bin_(refv)
        def ok(v):
            b = bin_(v)
            return 1 if (rc >= 0 and rc == b) else (0 if rc >= 0 else -1)
        rawOk, bwOk, rtOk = ok(raw), ok(bw), ok(route)
        if rc >= 0:
            nseg += 1; ok_raw += (rawOk == 1); ok_bw += (bwOk == 1); ok_rt += (rtOk == 1)
        parCol[c0:c1 + 1] = pmed; routeCol[c0:c1 + 1] = route
        seg_objs.append({"c0": c0, "c1": c1,
                         "par": None if not np.isfinite(pmed) else round(pmed, 1),
                         "route": None if not np.isfinite(route) else round(route, 1),
                         "src": "BW" if use_bw else "par", "rawOk": rawOk, "bwOk": bwOk, "rtOk": rtOk})

    if os.environ.get("RF_DEBUG"):
        print("\n  seg  t0     ref   raw    bw    par  route  | rawOk bwOk rtOk  src")
        for i, s in enumerate(segs):
            o = seg_objs[i]
            refv = np.interp(s + SEG_SEC / 2 + offset, tat, tar, left=np.nan, right=np.nan)
            print("  %3d %6.1f %5.1f %5.1f %5.1f %5s %5s  | %5d %4d %4d  %s"
                  % (i, s, refv, raw_seg[i], bw_seg[i],
                     ("%.1f" % o["par"]) if o["par"] is not None else "--",
                     ("%.1f" % o["route"]) if o["route"] is not None else "--",
                     o["rawOk"], o["bwOk"], o["rtOk"], o["src"]))

    def clean(a):
        return [None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 3) for v in a]
    data = {"rid": rid, "times": np.round(times, 1).tolist(), "freqsBpm": np.round(fbpm, 2).tolist(),
            "heat": [np.round(Pd[:, c], 5).tolist() for c in range(Pd.shape[1])],
            "rawRidge": clean(rawRidge), "bwRidge": clean(bwRidge), "refCol": clean(refCol),
            "routeCol": clean(routeCol), "parCol": clean(parCol), "segs": seg_objs, "rrEdges": RR_EDGES,
            "loBpm": 0.10 * 60, "hiBpm": 0.80 * 60,
            "accRaw": round(100 * ok_raw / max(1, nseg)), "accBw": round(100 * ok_bw / max(1, nseg)),
            "accRoute": round(100 * ok_rt / max(1, nseg))}
    print("  raw=%d%%  BW=%d%%  route_fast=%d%%" % (data["accRaw"], data["accBw"], data["accRoute"]))
    html = TEMPLATE.replace("__RID__", rid.replace("/", "_")).replace("__DATA__", json.dumps(data))
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, "routefast_%s.html" % rid.replace("/", "_"))
    open(outp, "w", encoding="utf-8").write(html)
    print("  saved ->", outp, "(%.0f KB)" % (len(html) / 1024))


if __name__ == "__main__":
    main()
