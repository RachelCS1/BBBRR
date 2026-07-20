# Code map & usage guide

How every module and function fits together, and how to call them. Pair this
with `README.md` (quick start) and `respiration_rr/settings.py` (all knobs).

---

## 1. The big picture — three pipelines

```
                      ┌─────────────────────────────────────────────┐
   REMbo EDF  ───────▶│ REFERENCE PIPELINE  (respiration_rr.reference)│──▶ breaths, RR, I:E, RRV
   (Nasal P,          │   airflow → bandpass → zero-crossing breaths  │        │
    Activity)         └─────────────────────────────────────────────┘        │
                                                                              ▼
                      ┌─────────────────────────────────────────────┐   ┌──────────────┐
   Watch CSV  ───────▶│ PPG PIPELINE  (respiration_rr.ppg)           │──▶│ COMPARISON   │
   (ppg/red/ir/       │   resample→FFT×4→1024Hz → beats → systolic →  │   │ (…compare)   │
    artifact/acc)     │   4 params (RSA/RIIV/AUC/LP) + spectral ridge │──▶│ rank by MAE  │
                      └─────────────────────────────────────────────┘   └──────────────┘
                                          │                                     │
                                          ▼                                     ▼
                                   respiration_rr.viz  (matplotlib figures) ◀────┘
```

`main.py` wires all three together. Everything reads its parameters from
`settings.py` (`REFERENCE`, `PPG`, `COMPARE`).

---

## 2. Module-by-module data flow

### Reference (Tool 2) — `respiration_rr/reference/airflow.py`
```
EDFReader.read_channel("Nasal P")           → time, airflow, fs
EDFReader.read_channel("Activity")          → activity (movement gate)
        │
analyze_reference(time, airflow, fs, activity=…, activity_fs=…)
        │   internally:
        │     bandpass_ma_cascade            (HP 0.05 / LP 1.0 Hz)
        │     compute_activity_energy + auto_activity_threshold (Otsu ×2)
        │     compute_noise_regions          (movement + bridge gaps)
        │     find_breath_crossings          (hysteresis zero-crossing)
        │     detect_breaths                 (RR = 60/Δt, gate 4–40 bpm)
        │     compute_ie_ratios              (inhale/exhale per breath)
        │     compute_rrv                    (moving variance)
        │     compute_breath_metrics         (agreement vs device, if a ref RR given)
        ▼
ReferenceResult  → .breaths[Breath], .crossings, .rrv_time/value, .metrics, …
```

### Watch PPG (Tool 1) — `respiration_rr/ppg/`
```
read_watch_csv(csv)                          → {time, fs, ppg, red, infra_red, artifact, acc_*}
prepare_watch(watch)                         → WatchSignals(.time @1024Hz, .channels, .move_regions)
        │   internally: resample_linear → upsample_fft(×4) → trim edges → movement
        ▼
analyze_ppg(channels, time, fs)              → { "IR": PPGChannelResult, "Green": …, … }
        │   per channel: analyze_ppg_channel
        │     bandpass_filter (0.5–4 Hz)  → detect_beats → refine_ss_by_derivative
        │     remove_dc_beat_aligned      → compute_systolic_analysis (SS/SE/MSD/AUC/maxH)
        │     build 4 params → bp_rr_series → breath-starts → RR
        │     compute_spectrogram + ridge_rr
        ▼
PPGChannelResult → .params{"RSA","RIIV","AUC","LP"}→RRParam, .ridge_rr, .ss_idx, …
```

### Comparison (Tool 3) — `respiration_rr/compare/compare.py`
```
compare_watch_vs_reference(ppg_results, ref_result, offset="auto")
        │   collect_watch_candidates  (every param×channel + ridge → Candidate)
        │   auto_align_offset         (scan offsets, minimise best MAE)
        │   rank_candidates           (MAE vs reference RR over overlap)
        ▼
CompareResult → .ranked[(Candidate, mae, n)], .offset_sec, .ref_time/ref_rr
```

---

## 3. Function reference

### `respiration_rr.io`
| Function | Signature | Returns |
|---|---|---|
| `EDFReader` | `EDFReader(path)` | object; `.meta`, `.channel_labels()`, `.read_channel(label)`, `.pick_activity_channel(keywords)` |
| `EDFReader.read_channel` | `(label)` | `(time, signal, fs)` — digital→physical scaled |
| `read_edf_channel` | `(path, label)` | `(time, signal, fs)` (one-shot) |
| `read_watch_csv` | `(path, channels=None)` | dict: `time`, `fs`, `raw`, one array per channel present |
| `read_poly_csv` | `(path, value_cols, downsample_factor=1, time_col="time_s")` | `(time, col0, col1, …)`; block-averages when downsampling |

### `respiration_rr.preprocessing`
| Function | Signature | Returns |
|---|---|---|
| `resample_linear` | `(signal, src_fs, dst_fs)` | resampled array |
| `upsample_fft` | `(signal, factor)` | FFT-super-resolution upsampled array |
| `moving_average` | `(signal, window_size)` | centered MA (edge-shrinking; reference variant) |
| `bandpass_ma_cascade` | `(signal, fs, f_low, f_high)` | `(filtered, baseline, high_passed)` |
| `butter_bandpass_filtfilt` | `(signal, fs, f_low, f_high)` | zero-phase Butterworth band-pass |
| `compute_movement_energy` | `(ax, ay, az, fs_acc, smooth_sec)` | jerk-energy envelope |
| `compute_activity_energy` | `(activity, fs_act, smooth_sec)` | smoothed activity index |
| `compute_noise_regions` | `(move_time, move_energy, threshold, min_clean_sec)` | `{move_regions, bridge_regions, merged_noise}` |
| `otsu_threshold` / `auto_activity_threshold` | `(values)` / `(energy)` | scalar threshold (Otsu, 2-pass) |

### `respiration_rr.reference`
| Function | Signature | Returns |
|---|---|---|
| `analyze_reference` | `(airflow_time, airflow_signal, fs, move_time=, move_energy=, accel=, activity=, activity_fs=, ref_time=, ref_rate=, ref_fs=25, user_noise=, cfg=REFERENCE)` | `ReferenceResult` |
| `find_breath_crossings` | `(time, signal, fs, rms_window_sec=, hyst_frac=)` | `(crossing_times, local_rms)` |
| `detect_breaths` | `(crossings, min_duration, max_duration, merged_noise)` | `list[Breath]` |
| `compute_ie_ratios` | `(breaths, time, filtered, fs)` | mutates each `Breath` (`.ie_ratio`, `.inhale_dur`, `.exhale_dur`) |
| `compute_rrv` | `(breaths, window_breaths=, gap_thresh_sec=)` | `(x, y)` |
| `compute_breath_metrics` | `(breaths, ref_time, ref_rate, ref_fs, time_shift)` | `{metrics, pairs, avg_pairs}` |

`Breath` fields: `start, end, center, duration, rate, inhale_dur, exhale_dur, ie_ratio`.
`ReferenceResult` fields: `time, raw, filtered, fs, crossings, breaths, merged_noise, rrv_time, rrv_value, metrics, pairs, avg_pairs, …`.

### `respiration_rr.ppg`
| Function | Signature | Returns |
|---|---|---|
| `prepare_watch` | `(watch, cfg=PPG)` | `WatchSignals(time, fs, channels, move_energy, move_threshold, move_regions)` |
| `analyze_ppg` | `(channels, t, fs, cfg=PPG, which=("Green","Red","IR","Artifact"))` | `{name: PPGChannelResult}` |
| `analyze_ppg_channel` | `(signal, t, fs, channel="Green", cfg=PPG, min_msd_ms=, invert=, compute_ridge=True)` | `PPGChannelResult` |
| `bp_rr_series` | `(x, y, f_hp, f_lp, order, resample_fs=None)` | `(grid_x, filtered_y)` |
| `detect_beats` | `(signal, fs, min_pr, max_pr)` | list of trough indices |
| `remove_dc_beat_aligned` | `(signal, peaks)` | baseline-corrected array |
| `refine_ss_by_derivative` | `(beats, bp_signal, fs)` | refined indices |
| `compute_systolic_analysis` | `(bc, t, beat_indices, fs, min_msd_ms=40, msd_min_pct_d1=30)` | `SystolicResult` |
| `compute_spectrogram` | `(signal, fs, fft_seconds, max_freq_store)` | dict `{times, freqs, power_db, power_lin}` |
| `ridge_rr` | `(spect, f_low, f_high)` | `(times, rr_bpm)` |

`PPGChannelResult`: `.channel, .fs, .filtered, .bc, .ss_idx, .msd_idx, .params{name:RRParam}, .ridge_time, .ridge_rr, .mean_rr(param)`.
`RRParam`: `.name, .series_x, .series_y, .env_x, .env_y, .bs_times, .rr_time, .rr_bpm`.
`SystolicResult`: `.ss_idx, .se_idx, .msd_idx, .auc_x/auc_y, .maxht_x/maxht_y, .msd_ss_ms`.

### `respiration_rr.compare`
| Function | Signature | Returns |
|---|---|---|
| `compare_watch_vs_reference` | `(ppg_results, ref_result, offset="auto", top_n=None, params=(…))` | `CompareResult` |
| `collect_watch_candidates` | `(ppg_results, params=(…))` | `list[Candidate]` |
| `score_candidate` | `(cand, ref_time, ref_rr, offset, min_overlap_sec=None)` | `(mae, n_overlap)` |
| `auto_align_offset` | `(cands, ref_time, ref_rr, search_range=None, step=1.0)` | `(best_offset, best_mae)` |

`CompareResult`: `.offset_sec, .ref_time, .ref_rr, .ranked[(Candidate, mae, n)], .candidates`.

### `respiration_rr.viz`
`plot_reference(ref)`, `plot_ppg_channel(res)`, `plot_ppg_overview(ppg_results)`,
`plot_comparison(cmp, top_n=3)` — each returns a matplotlib `Figure`. `show_all()`
opens the windows.

---

## 4. Recipes

**Run everything (same as `main.py`):**
```python
import sys; sys.path.insert(0, "Code")
from respiration_rr.io.edf_reader import EDFReader
from respiration_rr.io.csv_reader import read_watch_csv
from respiration_rr.reference.airflow import analyze_reference
from respiration_rr.ppg.preprocess import prepare_watch
from respiration_rr.ppg.respiration import analyze_ppg
from respiration_rr.compare.compare import compare_watch_vs_reference
from respiration_rr import viz

r = EDFReader("Data/Exp1/recordings data/001/1.edf")
t, sig, fs = r.read_channel("Nasal P")
_, act, act_fs = r.read_channel(r.pick_activity_channel())
ref = analyze_reference(t, sig, fs, activity=act, activity_fs=act_fs)

watch = read_watch_csv("Data/Exp1/recordings data/001/rt_flow_1112_1782377641000.csv")
ws = prepare_watch(watch)
ppg = analyze_ppg(ws.channels, ws.time, ws.fs)

cmp = compare_watch_vs_reference(ppg, ref, offset="auto")
viz.plot_comparison(cmp); viz.show_all()
```

**Get one channel's RR from one parameter:**
```python
ir = ppg["IR"]
rsa = ir.params["RSA"]
print(rsa.rr_time, rsa.rr_bpm)      # per-breath RR series
print(ir.mean_rr("RSA"))            # mean RR for that param
```

**Tune a parameter (no code edits):**
```python
from respiration_rr.settings import PPG, REFERENCE
PPG.rr_band_high_hz = 0.6           # narrow the respiration band
REFERENCE.hyst_frac = 0.25          # stricter breath-crossing hysteresis
# …then re-run analyze_ppg / analyze_reference
```

**Run a single channel by hand (e.g. just the Artifact belt):**
```python
from respiration_rr.ppg.respiration import analyze_ppg_channel
res = analyze_ppg_channel(ws.channels["Artifact"], ws.time, ws.fs, channel="Artifact")
```

**Score your own RR series against the reference:**
```python
from respiration_rr.compare.compare import reference_rr_series, score_candidate, Candidate
import numpy as np
rt, rr = reference_rr_series(ref)
cand = Candidate("mine", "X", "X", t=np.array([...]), rr=np.array([...]))
mae, n = score_candidate(cand, rt, rr, offset=0)
```

**Enable the full agreement stats (needs a device reference RR channel — poly CSV):**
```python
from respiration_rr.io.csv_reader import read_poly_csv
tair, air = read_poly_csv("airflow.csv", "Nasal Pressure", downsample_factor=8)
tref, rref = read_poly_csv("ref.csv", "Resp Rate")
ref = analyze_reference(tair, air, 25.0, ref_time=tref, ref_rate=rref, ref_fs=25.0)
print(ref.metrics)     # MAE, RMSE, %≤3bpm, Bland-Altman pairs
```

---

## 5. Where each HTML function lives now

| HTML function | Python location |
|---|---|
| `parseEDFChannel` | `io/edf_reader.py :: EDFReader.read_channel` |
| `bandpassFilter` (reference MA-cascade) | `preprocessing/filters.py :: bandpass_ma_cascade` |
| `findBreathCrossings` / `detectBreaths` | `reference/airflow.py` |
| `computeIeRatios` / `computeBreathMetrics` | `reference/airflow.py` |
| `_otsuThreshold` / `autoActivityThreshold` | `preprocessing/movement.py` |
| `upsampleFFT` / `resampleLinear` | `preprocessing/resample.py` |
| `detectBeats` / `removeDCBeatAligned` / `refineSSByDerivative` | `ppg/beats.py` |
| `computeSystolicAnalysis` | `ppg/systolic.py` |
| `bpRRSeries` / `filterPeaksByProminence` + 4-param logic | `ppg/respiration.py` |
| `computeSpectrogram` (+ ridge) | `ppg/spectrogram.py` |
| Combined `renderSimCompare` / candidate scoring | `compare/compare.py` |

Docstrings in each file cite the original HTML line numbers.
```
