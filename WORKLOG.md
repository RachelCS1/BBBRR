# Work Log — BBB Respiration Rate

> At the end of each work day a new entry is added here: a **date** followed by a
> **short summary** of what was done / changed that day.
> To add an entry: at the end of the day just say **"day summary"** and a new
> dated entry is added at the top (below these instructions).
> (This is written on request — it does not run automatically at day's end.)

---

## 2026-07-23

### Short summary
- Added a **per-parameter spectrogram**: `analyze_ppg_channel` now computes an STFT + respiration ridge for **each** of the four RR parameters (RSA/RIIV/AUC/LP), not just RSA — i.e. **4 spectrograms per channel**. New `RRParam` fields `spect` / `ridge_time` / `ridge_rr` hold them. LP (full-rate trace) is resampled to the 10 Hz RR grid first so all four are comparable.
- **Fixed the empty-ridge-band bug** flagged on 07-21: the ridge search now spans the respiration band `rr_band_low_hz..rr_band_high_hz` (0.1–0.7 Hz) instead of the degenerate `rr_spec_*` pair `[0.1, 0.1]`. The ridge now returns real values (e.g. Green ~8.5, Red/IR ~11 bpm) and even ranks as the **best comparison candidate** (Red/IR Ridge MAE 0.61 vs the REMbo reference).
- `viz`: added **`plot_ppg_spectrograms(res)`** — four heatmap panels (bpm × time, power in dB) with the ridge overlaid; and **removed the spectral-ridge panel** from `plot_ppg_channel` (now shows only the 4 parameter-envelope panels).
- `main.py`: opens the per-parameter spectrogram figure for each channel, and **no longer opens** the mean-RR overview bar chart (`plot_ppg_overview`).
- Built (as an analysis aid, outside the repo) an interactive HTML comparing, per channel × parameter, the **raw / band-passed / Hamming-windowed** series each with its FFT — illustrating that band-passing stabilises the respiration peak while the raw/Hamming spectra can lock onto wrong peaks.

### Notes
- Changes were applied to the `main` working tree by direct file edits (OneDrive was transiently blocking new-file creation under `.git`, so `git commit`/`merge` could not run at the time). Content verified identical to the worktree; commit performed separately.

## 2026-07-21

### Short summary
- Completed the **migration of the three HTML tools to Python** (`respiration_rr` package): EDF/CSV readers, preprocessing (resample → FFT upsample to 1024 Hz → filtering → movement), the reference tool (airflow zero-crossing RR + I:E + RRV + agreement), the PPG tool (beat detection, systolic SS/SE/MSD, the four respiration parameters RSA/RIIV/AUC/LP + spectral ridge), cross-device comparison (MAE ranking), matplotlib figures, and `main.py`.
- Created: central `settings.py` (all parameters + the HTML on-screen labels as comments), `README.md`, `GUIDE.md`, `Code_Map.html` (visual code map), and `main2.py` (step-by-step preprocessing inspector).
- Ported the **HTML's actual beat-detection method** (LPF-derivative override) and made it toggleable (`use_lpf_derivative_beats`).
- Fixed a **movement-energy scaling bug** (jerk computed on the 256 Hz accel, then the energy is upsampled — matching the HTML).
- Wired all beat-detection thresholds into `settings.py`; fixed a units bug in `msd_min_pct_d1`; changed ECG wavelet levels 4 → 6.
- Gave **BW** its own frequency band (`bw_band_low_hz` / `bw_band_high_hz` / `bw_filter_order`).
- Added to `Sweep_Report.html`: a **raw** trace (right axis), per-config show/hide checkboxes, and a clickable legend; ran a **4-config sweep** (BP 1–10 / 0.5–8 × dLP 6 / 10) over all 14 recordings.
- Added to `Code_Map.html`: a **"Preprocessing steps (main2)"** tab + two beat-detection method diagrams (trough vs LPF-derivative).
- Added `se_idx` to `PPGChannelResult`.
- **Python-vs-HTML gap diagnosis:** on Green/AUC the envelopes are identical (0.66% diff) and the code is identical — the gap is **100% the BS Prominence parameter** (HTML ran ≈1, Python 10 → 30 vs 16 peaks).

### Value contradictions found — docs vs code (documented only, NOT fixed)
> Context: the code was retuned during the day; the docs still describe the original defaults.

**Code_Map.html**

| Value | Docs say | Code says |
|---|---|---|
| PPG LP cutoff | 4 Hz | **8** (`ppg_lp_hz`) |
| Filter order | 4 | **2** (`ppg_filter_order`, `rr_filter_order`) |
| MSD min % of d1 | 30% | **50** (`msd_min_pct_d1`) |
| BS prominence | 10% | **0.001** (`breath_start_prominence`) |
| Movement margin | 5 s | **3 s** (`move_margin_sec`) |
| Ridge FFT window | 16 s | **32 s** (`rr_spec_window_sec`) |
| Ridge band | 0.1–0.5 Hz | **0.1–0.1** (`rr_spec_high_hz` — empty band) |

**GUIDE.md**
- Line 60: `bandpass_filter (0.5–4 Hz)` → code is **0.5–8 Hz**.
- Line 127: `compute_systolic_analysis(..., msd_min_pct_d1=30)` → live value is **50**.

**README.md**
- Run commands are written `py Code/main.py` → files are now in `BBBRR\` (should be `py main.py`).
- Validation numbers (35 breaths, 13.4 bpm, I:E 0.83, IR/RIIV MAE 0.89, IR/AUC 1.45) — stale after the retuning.

**Contradictions inside the code itself**
- `breath_start_prominence = 0.001` but its comment says "UI is 10 %" — 10% is 0.10, not 0.001.
- `rr_spec_high_hz = 0.1` equals `rr_spec_low_hz = 0.1` → empty ridge band → the ridge returns `nan`.

**Reference values that drifted from the docs**
- `min_rate/max_rate`: 4–40 → **3–50 bpm** · `spec_top_db`: 5 → **100** · `spec_high_seg_sec`: 60 → **32**.

### Open / next
- BS Prominence percent/fraction trap — sync HTML and Python before comparing.
- Consider a robust prominence threshold (percentile-based) instead of the global `(max−min)×frac`.
- Later: move to the frequency domain (window + FFT) for more robust RR detection.
