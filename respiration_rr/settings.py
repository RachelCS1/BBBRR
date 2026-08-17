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
    min_rate_bpm: float = 3.0       # "Min Rate (bpm)"  (paramMinB) -> max breath duration
    max_rate_bpm: float = 50.0      # "Max Rate (bpm)"  (paramMaxB) -> min breath duration

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
    time_shift_sec: float = 0.0     # manual fixed reference-device shift; 0 = off
                                    # (set only if a reference device has a known constant offset)
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
    spec_top_db: float = 100.0        # "top dB in spectogram"  (paramTopDB)
    spec_high_seg_sec: float = 32.0 # "highest value segment (s)"  (paramHighSeg)

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
    # Cubic-spline RR variant (comparison alternative to linear-interp + BP):
    # when True, RSA/RIIV/AUC each get a second parameter ("*_spline") whose
    # envelope is a cubic spline through the per-beat series with NO band-pass,
    # so the spline method can be compared head-to-head with the existing
    # linear-interp+band-pass envelope. The original params are untouched.
    rr_spline_enabled: bool = True
    # Breath-start prominence for the spline variants ONLY. None -> reuse
    # breath_start_prominence (kept in lock-step with the linear+BP params).
    # Set a value to tune the spline peak detector independently — e.g. raise
    # it to reject the extra spurious peaks the un-band-passed spline can add.
    rr_spline_prominence: float = 0.08
    # Spline breath-start detection: False = peaks (maxima, default, unchanged),
    # True = valleys (minima). Applies to the spline variants ONLY; the linear+BP
    # params always use peaks. Valley detection = same detector on the inverted
    # envelope, so RR marks the opposite respiration phase.
    rr_spline_use_valleys: bool = True
    # Smoothing-spline RR variant (THIRD approach): a penalized spline that
    # denoises the beat-to-beat jitter WITHOUT a band-pass — less information
    # loss than linear+BP, cleaner than the interpolating spline. Adds "*_ssp"
    # params (RSA_ssp/RIIV_ssp/AUC_ssp). Reuses rr_spline_prominence and
    # rr_spline_use_valleys for peak detection.
    rr_ssp_enabled: bool = True
    # Smoothing strength, expressed as an effective -3 dB CUTOFF in Hz (intuitive
    # and scale-invariant: the signal is standardised before fitting). Keep
    # respiration (<=~0.8 Hz ≈ 50 bpm) and smooth faster beat-to-beat jitter.
    # Lower cutoff = smoother. Derived to lam = 1/(2*pi*fc)^4 for the cubic
    # smoothing spline. Set rr_ssp_lam to override with a raw lambda instead.
    rr_ssp_cutoff_hz: float = 0.45
    rr_ssp_lam: float = None        # raw smoothing lambda; None -> derive from rr_ssp_cutoff_hz
    # Breath-start prominence for the smoothing-spline variants ONLY.
    # None -> reuse rr_spline_prominence (which itself falls back to
    # breath_start_prominence). Set a value to tune the ssp peak detector
    # independently of the interpolating-spline variants.
    rr_ssp_prominence: float = 0.01
    # Combined global+local prominence for breath-start detection on the RR
    # envelope (ALL envelope variants: linear, spline, ssp). A peak must clear
    # BOTH: the global floor (its own prominence x global range) AND a local
    # check (rr_local_prom_frac x local range), where local range is max-min in
    # a +-rr_local_prom_win_sec window around the peak. The local term only
    # bites when rr_local_prom_frac exceeds the global fraction, rejecting peaks
    # that pass the absolute floor but don't stand out locally. Set either to
    # None/0 for global-only prominence (original behaviour).
    rr_local_prom_win_sec: float = 4.0   # local window (s); None/0 -> global only
    rr_local_prom_frac: float = 0.2      # local-relative fraction f_l; None/0 -> global only
    # ---- Noise-region gating on the FINAL RR (RSA/RIIV/AUC[/spline/ssp] + LP) ----
    # In a movement/noise region the beats are dropped, so the per-beat series has
    # a hole that the envelope (linear+BP / cubic spline / smoothing spline) then
    # silently interpolates across. A breath-start straddling that gap emits a
    # spurious (usually very low) RR that is purely an artefact of the gap fill.
    # These gates drop any breath-to-breath interval whose span overlaps a
    # movement region (the same treatment BWlegacy/BWbank already apply), and
    # optionally clamp the surviving RR to a plausible band. IMPORTANT: bs_times
    # are NOT gated — the detected breath-starts still plot (so you can SEE that a
    # peak was found in the noise), only the emitted RR series gets the hole.
    # Default: ONLY the noise-spanning gate is on, so the sole behaviour change vs
    # the original is that RR intervals overlapping a noise region are dropped.
    # rr_valid_bpm is an opt-in extra clamp (set a (lo, hi) tuple to enable). To
    # reproduce the pre-gate output byte-for-byte set rr_reject_noise_spanning=False
    # (rr_valid_bpm is already None).
    rr_reject_noise_spanning: bool = True   # drop RR intervals overlapping a move region
    rr_valid_bpm: tuple = None              # optional (lo, hi) clamp on final RR; None -> off
    # The breath-start prominence floor is prom_frac * amplitude range. A high-
    # amplitude noise burst inflates that range (max-min), pushing the floor so
    # high that real breaths elsewhere fall below it and almost nothing is
    # detected. When True, the range is measured from noise-free samples only
    # (via move_regions), so a noise spike no longer starves peak detection.
    # No effect when there are no move_regions -> output unchanged.
    rr_prominence_ignore_noise: bool = True
    # Robust prominence range: scale the floor by a robust spread (IQR) instead of
    # max-min. max-min is inflated by ANY single extreme sample — a startup/edge
    # transient or a spline overshoot that sits OUTSIDE the noise window and so
    # survives rr_prominence_ignore_noise. IQR ignores the outer quartiles (≈25%
    # breakdown), so it estimates the typical breath amplitude regardless of where
    # the outliers are or what fraction they are — no per-recording tuning. The
    # k factor maps IQR to peak-to-peak (√2 for a clean sinusoid, so clean
    # recordings stay ≈ unchanged); raise k if spurious peaks appear, lower it if
    # real breaths are still missed. Falls back to max-min if IQR is degenerate.
    rr_prominence_robust: bool = True
    rr_prominence_iqr_k: float = 1.4
    rr_resample_fs: float = 10.0    # not in UI: FS_RESAMP in bpRRSeries
    rr_band_low_hz: float = 0.1     # "HP Cutoff RR (Hz)"  (paramHP_RR)
    rr_band_high_hz: float = 1.0    # "LP Cutoff RR (Hz)"  (paramLP_RR)
    rr_filter_order: int = 2        # "Filter Order"  (paramFilterOrder, shared)
    breath_start_prominence: float = 0.001  # "BS Prominence (%)"  (paramBSProm; UI is 10 %, stored as 0.001)
    rr_spec_window_sec: float = 32.0       # "FFT Length (s)"  (paramFftLength)
    rr_spec_low_hz: float = 0.1            # not in UI: ridge search-band low
    rr_spec_high_hz: float = 0.1           # "Spect Min Freq (Hz)"  (paramSpectMinFreq)

    # ---- BW (baseline-wander) parameter — its OWN band on the raw signal ----
    # RSA/RIIV/AUC use rr_band_* above (per-beat series). BW band-passes the raw
    # channel directly, so it gets independent cutoffs — tune these separately.
    bw_band_low_hz: float = 0.1    # BW high-pass (raw signal)
    bw_band_high_hz: float = 1.0   # BW low-pass (raw signal)
    bw_filter_order: int = 2        # BW filter order

    # ---- BWlegacy — legacy Breath_by_Breath zero-cross valley detector ----
    # Runs on the SAME BW band-passed trace, but finds breaths with the ported
    # legacy method (per-segment rarer-polarity peaks + zero-cross dedup) instead
    # of prominence peaks. Exposed as an extra per-channel parameter so the two
    # BW detectors can be compared side by side.
    bwlegacy_enabled: bool = True  # add the BWlegacy parameter to every channel
    # BWlegacy now runs the FULL legacy chain: build the signal the old way
    # (FFT band-pass 0.1-0.5 Hz + peak-envelope detrend + spline gap-fill) and
    # then detect peaks the old way. p2p_th is the legacy amplitude gate; it is
    # sensor-scale dependent (tuned for the 64 Hz artifact ADC) so tune it for
    # the watch channels, or set 0 to disable the amplitude gate.
    legacy_bw_p2p_th: float = 20.0  # legacy peak-to-peak "too flat = noise" threshold
    legacy_bw_detrend_band: tuple = (0.1, 0.7)  # FFT band-pass (Hz) before the legacy
                                                # detrend (old upper cutoff 0.5 -> 0.7)
    # Legacy average-RR gate (BWlegacy only): exclude times with no valid
    # spectrogram average, and reject a breath whose rate disagrees with the
    # local average by more than legacy_bw_avg_ratio (the old BBB_RR behaviour).
    legacy_bw_avg_gate: bool = False         # apply the average-RR gate to BWlegacy
    legacy_bw_avg_ratio: float = 0.30       # reject breath if |BBB - avg|/avg exceeds this
    legacy_bw_valid_bpm: tuple = (6.0, 40.0)  # legacy valid breath-rate range (bpm)
    legacy_bw_avg_win_sec: float = 8.0        # STFT window for the legacy average RR
                                              # (short -> broad coverage; the shared
                                              #  rr_spec_window_sec is too long to gate)

    # ---- BWbank — MATLAB filter-bank + stitching BBB (BBB_SignalCreation.m) ----
    # A SECOND breath-by-breath method, independent of BWlegacy: a bank of narrow
    # band-passes (one per rate range), per-instant pick of the best-fitting band
    # (energy + wide-band match), mode-smoothed level choice, zero-crossing
    # STITCHING at level transitions, then a trend state-machine breath detector.
    # OFF by default so existing outputs are byte-identical until enabled.
    bwbank_enabled: bool = True               # add the BWbank parameter to every channel
    bwbank_work_fs: float = 16.0              # working rate: channel is decimated to this
    bwbank_rate_bands_bpm: tuple = ((4, 10), (8, 14), (12, 18),
                                    (16, 22), (20, 30), (28, 40))  # rateBank (bpm)
    bwbank_wide_band_hz: tuple = (0.04, 0.7)  # wide band-pass (reference signal)
    bwbank_align_win_sec: float = 16.0        # moving-average window for DC removal (aligned)
    bwbank_score_win_sec: float = 30.0        # per-band scoring window
    bwbank_score_overlap_sec: float = 1.0     # scoring step
    bwbank_min_valid_sec: float = 4.0         # min non-noise seconds in a window to score it
    bwbank_level_smooth_n: int = 5            # mode-smoothing frames (LevelSelection N)
    bwbank_bp_order: int = 3                  # Butterworth order (per narrow/wide band)
    bwbank_peak_down_tol_sec: float = 0.5     # downward tolerance (peak detector)
    bwbank_peak_up_th_frac: float = 0.2       # rise threshold as a fraction of the period
    bwbank_valid_bpm: tuple = (6.0, 40.0)     # valid breath-rate range (bpm)

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

    # ---- Monitor-mode CSV (optical channels + accelerometer, NO timestamps) ----
    # Monitor files have a header like "PPG,Artifact,RED SIG,IR,XL-X,XL-Y,XL-Z" and
    # no sampling_time column, so the rate can't be measured from the file — it is
    # declared here (64 Hz). No ECG; the three XL axes map to the accelerometer, so
    # movement/noise gating runs exactly as for real-time files.
    monitor_row_fs: float = 64.0           # not in UI: declared monitor sampling rate
    monitor_channel_cols: dict = field(default_factory=lambda: {
        "PPG": "ppg", "Artifact": "artifact", "RED SIG": "red", "IR": "infra_red",
        "XL-X": "acc_x", "XL-Y": "acc_y", "XL-Z": "acc_z",
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


# ======================================================================
# IR-PPG MSD time-synchronisation (watch <-> REMbo)
#   Source: respiration_rr/sync.py
# ======================================================================
@dataclass
class SyncSettings:
    max_offset_sec: float = 100.0   # search range +- (devices start near-simultaneously)
    coarse_step_sec: float = 0.05   # coarse offset scan step
    fine_step_sec: float = 0.005    # fine refine step (~5 ms)
    match_tol_sec: float = 0.10     # a watch MSD matches a REMbo MSD if within this
    min_overlap_sec: float = 60.0   # require >= this matched span to trust the offset
    min_matched: int = 20           # require >= this many matched beats
    window_sec: float = 120.0       # clean-window length for the residual lock
    min_prominence: float = 3.0     # flag LOW if the match-count peak is < this many sigma


# Singletons — import these.
REFERENCE = ReferenceSettings()
PPG = PPGSettings()
COMPARE = CompareSettings()
SYNC = SyncSettings()
