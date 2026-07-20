"""
Central parameter store for the whole respiration-rate toolchain.

This is the Python equivalent of the "Settings" panels in the three HTML tools.
Every tunable value that used to live in an HTML <input> is collected here so you
can change it in one place, without touching any algorithm code.

Naming: the Python field names are descriptive (they're referenced throughout the
code, so they can't be renamed to match the HTML without breaking imports).
Instead each field's comment gives the label AS IT APPEARS ON SCREEN in the HTML
settings panel, in quotes, followed by the internal id in parentheses, e.g.
    ppg_hp_hz = 0.5    # "HP Cutoff (Hz)"  (paramHP)
Values that are hard-coded in the JS (no on-screen control) are marked
"not in UI".  Where the HTML shows a percentage but the Python stores a fraction,
the comment says so.

Usage
-----
    from respiration_rr.settings import REFERENCE, PPG, COMPARE
    fs = PPG.target_fs
    REFERENCE.hp_cutoff_hz = 0.03      # override before running
"""

from dataclasses import dataclass, field


# ======================================================================
# TOOL 2 — Reference analyzer (REMbo EDF + Polysomnograph CSV)
#   Source: "Breath by breath RR - Poly + REMbo.html"
# ======================================================================
@dataclass
class ReferenceSettings:
    # ---- Bandpass (MA-cascade, not Butterworth) ----
    hp_cutoff_hz: float = 0.05      # "HP Cutoff (Hz)"  (paramHP)
    lp_cutoff_hz: float = 1.0       # "LP Cutoff (Hz)"  (paramLP)

    # ---- Breath rate gate ----
    min_rate_bpm: float = 4.0       # "Min Rate (bpm)"  (paramMinB) -> max breath duration
    max_rate_bpm: float = 40.0      # "Max Rate (bpm)"  (paramMaxB) -> min breath duration

    # ---- Movement / activity gating ----
    move_thresh_gs: float = 0.15    # "Move Thresh (g/s)"  (paramMoveThresh, CSV/Poly mode)
    activity_mode: str = "auto"     # "Activity Threshold"  (paramActivityMode: auto | manual)
    activity_thresh_manual: float = 3.0   # "Manual value (0-100)"  (paramActivityThresh)
    move_smooth_sec: float = 2.0    # not in UI: 2-s activity/accel smoothing

    # ---- Noise regions ----
    min_clean_sec: float = 20.0     # "Min Clean (s)"  (paramMinClean)

    # ---- Breath-crossing hysteresis (findBreathCrossings) ----
    rms_window_sec: float = 10.0    # not in UI: local-RMS window
    hyst_frac: float = 0.20         # not in UI: HYST_FRAC = 20% of local RMS

    # ---- Reference agreement (computeBreathMetrics) ----
    time_shift_sec: float = -846.0  # "Time Shift (s)"  (paramTimeShift)
    stale_run_sec: float = 30.0     # not in UI: reference frozen >= 30 s = stale
    ref_gate_min_bpm: float = 3.0   # not in UI: device rate < 3 skipped
    ref_gate_max_bpm: float = 40.0  # not in UI: device rate > 40 skipped
    within_bpm: float = 3.0         # not in UI: "% within N bpm" (|err| <= 3)
    avg_window_sec: float = 30.0    # not in UI: 30-s sliding-window averaging (MAE30 / BA)
    avg_step_sec: float = 1.0       # not in UI: 1-s window step

    # ---- RRV ----
    rrv_window_breaths: int = 100   # "RRV window (breaths)"  (paramRRVWindow)
    rrv_gap_thresh_sec: float = 15.0  # not in UI: break the RRV line across gaps

    # ---- Spectrogram (opt-in) ----
    fft_len_sec: float = 100.0      # "FFT length (s)"  (paramFFTLen)
    spec_min_hz: float = 0.08       # "Spec min freq (Hz)"  (paramSpecMin)
    spec_max_hz: float = 1.0        # "Spec max freq (Hz)"  (paramSpecMax)
    spec_top_db: float = 5.0        # "top dB in spectogram"  (paramTopDB)
    spec_high_seg_sec: float = 60.0 # "highest value segment (s)"  (paramHighSeg)

    # ---- Source formats ----
    csv_airflow_cols: tuple = ("time_s", "Nasal Pressure")   # "200 Hz Airflow CSV" columns
    csv_airflow_downsample: int = 8         # not in UI: 200 Hz -> 25 Hz
    csv_airflow_fs: float = 25.0            # not in UI
    csv_ref_cols: tuple = ("time_s", "Resp Rate")            # "25 Hz Reference CSV" columns
    csv_ref_fs: float = 25.0                # not in UI
    csv_accel_cols: tuple = ("time_s", "X Axis", "Y Axis", "Z Axis")  # "20 Hz Accel CSV" columns
    csv_accel_fs: float = 20.0              # not in UI
    edf_airflow_channel: str = "Nasal P"    # "Airflow Channel"  (edfAirflowChannel)
    edf_ppg_channel: str = "Pulse Wave"     # "PPG (Pulse Wave) Channel"  (edfPPGChannel)
    edf_activity_keywords: tuple = ("activ", "actigraph")  # not in UI: activity-channel auto-pick


# ======================================================================
# TOOL 1 — Watch PPG + Artifact analyzer
#   Source: "Cardiacsense Respiration Rate Analyzer.html"
# ======================================================================
@dataclass
class PPGSettings:
    # ---- Front-end sampling ----
    fs_orig: float = 256.0          # not in UI: FS_ORIG (native watch rate)
    upsample_factor: int = 4        # not in UI: UPSAMPLE_FACTOR
    target_fs: float = 1024.0       # not in UI: FS = FS_ORIG * UPSAMPLE_FACTOR
    fft_chunk: int = 8192           # not in UI: CHUNK in upsampleFFT
    fft_overlap: int = 512          # not in UI: OVERLAP in upsampleFFT
    trim_head_sec: float = 3.0      # not in UI: _TRIM_LEAD_SEC
    trim_tail_sec: float = 15.0     # not in UI: _TRIM_TAIL_SEC

    # ---- PPG bandpass (MA-cascade) ----
    ppg_hp_hz: float = 0.5          # "HP Cutoff (Hz)"  (paramHP)
    ppg_lp_hz: float = 8.0         # "LP Cutoff (Hz)"  (paramLP)
    ppg_filter_order: int = 2       # "Filter Order"  (paramFilterOrder)
    invert_ppg: bool = True         # not in UI: sign-invert PPG/Red/IR/Artifact

    # ---- ECG chain (optional; not RR-critical) ----
    ecg_lp_hz: float = 40.0         # not in UI: ECG pre-bandpass 0.5-40 Hz
    ecg_notch_hz: float = 50.0      # not in UI: notch f0
    ecg_notch_q: float = 30.0       # not in UI: notch Q
    ecg_wavelet: str = "db4"        # not in UI: ECG wavelet
    ecg_wavelet_levels: int = 6     # not in UI: ecgWaveletFilter levels

    # ---- Movement (accelerometer jerk) ----
    move_thresh_gs: float = 1400.0  # "Move Thresh (g/s)"  (paramMoveThresh)
    move_smooth_sec: float = 0.5    # not in UI: energy smoothing (round(FS*0.5))
    move_margin_sec: float = 3.0    # "Move Margin (s)"  (paramMoveMargin)
    move_min_clean_gap_sec: float = 10.0  # "Min Clean Gap (s)"  (paramMinCleanGap)
    move_min_noise_sec: float = 0.2       # "Min Noise Time (s)"  (paramMinNoiseTime)

    # ---- Beat detection (detectBeats) ----
    hr_min_bpm: float = 30.0        # "Min HR (bpm)"  (paramMinHR)
    hr_max_bpm: float = 200.0       # "Max HR (bpm)"  (paramMaxHR)
    dicrotic_rr_frac: float = 0.65    # not in UI: neighbour interval < 65% of median
    dicrotic_amp_frac: float = 0.75   # not in UI: valley amplitude < 75% of typical
    upslope_frac: float = 0.40        # not in UI: >= 40% of median 80 ms gain
    upslope_window_sec: float = 0.08  # not in UI: 80 ms gain window

    # ---- LPF-derivative beat override (graph 2b/15b) ----
    use_lpf_derivative_beats: bool = True   # not in UI: HTML always overrides detectBeats
    beat_snap_window_sec: float = 0.2       # not in UI: +-0.2 s snap to trough (sw2b/sw15b)

    # ---- Systolic / MSD ----
    msd_min_ms: float = 40.0        # "Min MSD Green/Red/IR/Artifact (ms)"  (paramMinMsd*, all 40)
    msd_min_pct_d1: float = 50.0    # "MSD min % of d1 peak"  (paramMsdMinPctD1, PERCENT)
    deriv_lpf_hz: float = 6.0       # "Deriv LPF (Hz)"  (paramDerivLPF)
    d34_smooth_ms: float = 12.0     # not in UI: +-12 ms 3rd/4th-derivative smoothing

    # ---- Beat quality (Pearson correlation reject; OFF by default in the HTML) ----
    corr_green: float = 0.60        # "CORR Thresh"  (paramCorrThresh)
    corr_red_ir: float = 0.50       # "COR_R_IR"  (paramCorrRIR)
    var_gate_height_ratio: float = 0.442  # "R/IR Hgt Var Thresh"  (paramRIRVarThresh)

    # ---- Pan-Tompkins QRS (ECG; optional) ----
    pt_bp_low_hz: float = 5.0       # not in UI: Pan-Tompkins bandpass low
    pt_bp_high_hz: float = 15.0     # not in UI: Pan-Tompkins bandpass high
    pt_mwi_ms: float = 150.0        # not in UI: moving-window integrator width
    pt_refractory_ms: float = 200.0 # not in UI: refractory period
    pt_fiducial_search_ms: tuple = (-150.0, 30.0)  # not in UI: snap-to-raw-R window

    # ---- Hemodynamics (optional) ----
    pat_cap_ms: float = 500.0       # not in UI: PAT cap
    co_window_sec: float = 60.0     # not in UI: CO = sum of SV over trailing 60 s

    # ---- Respiration: the FOUR parameters + spectral ridge (bpRRSeries) ----
    rr_params: tuple = ("RSA", "RIIV", "AUC", "LP")   # (labels; not a UI control)
    rr_resample_fs: float = 10.0    # not in UI: FS_RESAMP in bpRRSeries
    rr_band_low_hz: float = 0.1     # "HP Cutoff RR (Hz)"  (paramHP_RR)
    rr_band_high_hz: float = 0.7    # "LP Cutoff RR (Hz)"  (paramLP_RR)
    rr_filter_order: int = 2        # "Filter Order"  (paramFilterOrder, shared)
    breath_start_prominence: float = 0.10  # "BS Prominence (%)"  (paramBSProm; UI is 10 %, stored as 0.10)
    rr_spec_window_sec: float = 32.0       # "FFT Length (s)"  (paramFftLength)
    rr_spec_low_hz: float = 0.1            # not in UI: ridge search-band low
    rr_spec_high_hz: float = 0.1           # "Spect Min Freq (Hz)"  (paramSpectMinFreq)

    # ---- BW (baseline-wander) parameter — its OWN band on the raw signal ----
    # RSA/RIIV/AUC use rr_band_* above (per-beat series). BW band-passes the raw
    # channel directly, so it gets independent cutoffs — tune these separately.
    bw_band_low_hz: float = 0.05    # BW high-pass (raw signal)
    bw_band_high_hz: float = 1.0   # BW low-pass (raw signal)
    bw_filter_order: int = 4        # BW filter order

    # ---- SpO2 (optional stretch; ratio-of-ratios) ----
    spo2_R_clamp: tuple = (0.3, 2.0)             # not in UI: clamp R
    spo2_cal: tuple = (-45.060, 30.354, 94.845)  # not in UI: paramA, paramB, paramC (a*R^2+b*R+c)

    # ---- On-screen HTML controls not (yet) ported, listed for reference ----
    # "MA Window (samples)"       (paramMAWindow, 32)      — DC/LP MA window for PI
    # "Min Perf. Index"          (paramMinPI, 0.05)        — minimum perfusion-index gate
    # "R/IR Hgt Var Length"      (paramRIRVarLen, 6)       — beats in height-ratio var window
    # "Art Detrend MA"           (paramArtDetrendMA, 64)   — artifact-channel detrend MA window
    # "HR Tracking Threshold (%)"(paramHRTrackThresh, 7)   — HR-tracking gate in noise segments
    # "HR Window (s)"            (paramHRWindow, 10)       — HR-tracking filter window
    # "HR BW (Hz)"               (paramHRBW, 0.1)          — HR-tracking filter bandwidth
    # "Spect dB Range"           (paramSpectRange, 10)     — spectrogram display range
    # "Spect Threshold"          (paramSpectThresh, 0.8)   — spectrogram display threshold
    # "Spect Max Freq (Hz)"      (paramSpectMaxFreq, 4)    — spectrogram display max freq

    # ---- Input format hints (rt_flow watch CSV) ----
    csv_time_col: str = "sampling_time"    # not in UI: watch CSV timestamp column
    csv_channel_cols: dict = field(default_factory=lambda: {
        "ppg": "ppg", "ecg": "ecg", "artifact": "artifact",
        "red": "red", "infra_red": "infra_red",
        "acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "acc_z",
        "sp_o2": "sp_o2",
    })


# ======================================================================
# TOOL 3 — Cross-device comparison (Combined shell)
#   Source: "Combined RR Analyzer.html"
# ======================================================================
@dataclass
class CompareSettings:
    offset_range_sec: float = 300.0     # not in UI: +-300 s alignment slider range
    min_overlap_sec: float = 20.0       # not in UI: require >= 20 s overlap to score
    top_n: int = 3                      # not in UI: "Best 3 vs ref"
    # scoring metric is MAE (bpm) of a watch RR series vs the REMbo zero-crossing RR


# Singletons — import these.
REFERENCE = ReferenceSettings()
PPG = PPGSettings()
COMPARE = CompareSettings()
