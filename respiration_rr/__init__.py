"""Respiration-rate toolchain — Python port of the CardiacSense RR HTML tools.

Modules
-------
io             : EDF (REMbo/poly) + CSV (watch rt_flow, poly) readers
preprocessing  : resampling, filters, movement/noise detection
reference      : Tool 2 — airflow zero-crossing RR + agreement (ground truth)
ppg            : Tool 1 — watch PPG beats/systolic/derivatives + 4 respiration params
compare        : Tool 3 — cross-device comparison
viz            : matplotlib figures
settings       : central tunable parameter store
"""
