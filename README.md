# Respiration-Rate Toolchain — Python port

Python port of the three HTML respiration-rate tools in `../HTML's/`:

| HTML tool | Ported to | What it does |
|---|---|---|
| `Cardiacsense Respiration Rate Analyzer.html` | `respiration_rr/ppg/` | RR from watch PPG (Green/Red/IR + Artifact) via the **four respiration parameters** |
| `Breath by breath RR - Poly + REMbo.html` | `respiration_rr/reference/` | Breath-by-breath RR from REMbo/polysomnograph airflow (the reference) |
| `Combined RR Analyzer.html` | `respiration_rr/compare/` | Cross-device comparison, ranking watch RR vs the reference by MAE |

## Run it

```bash
py Code/main.py                       # defaults to Data/Exp1/recordings data/001
py Code/main.py --recording "Data/Exp1/recordings data/003"
py Code/main.py --edf path/to/1.edf --watch path/to/rt_flow.csv
py Code/main.py --no-show             # print summary only, no figure windows
```

Requires: `numpy`, `pandas`, `scipy`, `matplotlib`, `PyWavelets` (`py -m pip install numpy pandas scipy matplotlib PyWavelets`).

`main.py` loads a recording, runs the reference analysis (EDF), the watch PPG
analysis (rt_flow CSV), and the comparison, prints a summary, and opens
matplotlib windows mirroring the HTML graphs.

## The four respiration parameters (watch PPG)

For each PPG channel, RR is derived from four independent per-beat modulation
series (`respiration_rr/ppg/respiration.py`):

1. **RSA**  — RR-interval (ms) between systolic starts (respiratory sinus arrhythmia)
2. **RIIV** — per-beat max-height amplitude
3. **AUC**  — per-beat area under the baseline-corrected curve
4. **LP**   — the raw channel band-passed directly at the respiration band

Params 1–3 → `bp_rr_series` (resample to 10 Hz → band-pass 0.1–0.7 Hz → DC-restore),
then breath-starts = prominence-filtered envelope peaks; param 4 detects peaks on
the full-rate band-passed trace. In all cases **RR = 60 / Δt** between breath-starts.
A fifth estimator — the **spectral ridge** — is the dominant STFT frequency per frame.

## Tuning parameters

**All tunable parameters live in `respiration_rr/settings.py`** — the single
place mirroring the HTML settings panels (`REFERENCE`, `PPG`, `COMPARE`). Edit a
value there; every module reads from it.

## Package layout

```
respiration_rr/
  settings.py         # all parameters (edit here)
  io/                 # edf_reader, csv_reader
  preprocessing/      # resample (FFT x4 -> 1024 Hz), filters, movement/noise
  reference/          # airflow zero-crossing RR + I:E + RRV + agreement (Tool 2)
  ppg/                # dsp, beats, systolic, respiration (4 params), spectrogram (Tool 1)
  compare/            # cross-device ranking by MAE (Tool 3)
  viz/                # matplotlib figures
```

## Validation (Data/Exp1/001)

- Reference EDF: 35 breaths, mean RR 13.4 bpm (dips to ~8 bpm during the
  slow-breathing minute), I:E ≈ 0.83.
- Watch PPG: four params + ridge computed on all four channels.
- Comparison: best candidate **IR/RIIV, MAE 0.89 bpm** vs the reference
  (offset +9 s), IR/AUC 1.45, IR/Ridge 1.51.

## Scope notes

- Faithful to the HTML algorithms: preprocessing chains, filter orders,
  thresholds, and the four-parameter + ridge respiration logic are ported
  verbatim (source line numbers cited in docstrings).
- The watch tool's ~90-graph engine also computes SpO₂, HRV, and hemodynamics
  (PAT/SV/CO/APG). Those are **out of scope** for this RR port (agreed as
  optional stretch); the full beat/systolic/derivative core they need *is*
  ported, so they can be added on top of `ppg/systolic.py` later.
- `post_corr_filter` (Pearson beat gate) is ported but OFF by default, matching
  the shipped HTML where its calls were commented out.

## To verify further

- Run other recordings (`002`–`014`) and confirm RR stays physiologically
  plausible and the comparison MAE stays low.
- If a poly 3-CSV dataset appears, `reference.analyze_reference` accepts
  `ref_time`/`ref_rate` to enable the full agreement stats + Bland-Altman.
```
