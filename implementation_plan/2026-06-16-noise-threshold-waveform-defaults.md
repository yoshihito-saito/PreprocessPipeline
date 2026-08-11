# Noise Threshold Waveform Defaults

Date: 2026-06-16

## Goal and Motivation

Remove waveform-shape template metrics from the default hard noise-labeling thresholds while keeping those metrics computed and visible for manual review.

## Current Problem

Several units with clean CCGs, adequate firing rates, presence ratios, SNR, and amplitudes are being labeled as noise solely because waveform-shape metrics cross fixed thresholds. In particular, `slope`, `peak_trough_ratio`, `half_width`, and `peak_to_valley` are sensitive to positive-going or triphasic templates because the underlying template metrics assume a negative trough followed by a positive peak.

## Why Now

The current default settings make waveform-shape metrics hard reject criteria. This can incorrectly exclude review-worthy units when waveform polarity or shape violates that assumption.

## Affected Files

- `src/postprocess/metafile.py`
- `src/preprocess/gui/config_model.py`
- `src/preprocess/gui/web_app.py`
- `Run_preprocessSession.py`
- `Run_preprocessSession.ipynb`
- `README.md`
- `tests/postprocess/test_metric_enrichment.py`

## Public Parameters and API Changes

No parameters are removed. The default `noise_thresholds` no longer includes these keys:

- `peak_to_valley_gt`
- `peak_trough_ratio_lt`
- `halfwidth_gt`
- `slope_lt`

Users can still opt into these thresholds by adding the keys explicitly in code or notebooks.

## Expected Behavior

Default noise labeling uses timing, presence, SNR, amplitude, and firing-rate thresholds. Waveform-shape metrics remain computed and exported to `cluster_info.tsv`, but they do not label units as noise unless the user explicitly enables those thresholds in code or notebooks. The GUI does not expose waveform-shape threshold controls.

## Verification

- Run focused postprocess unit tests.
- Inspect notebook JSON and GUI threshold defaults to confirm waveform-shape keys are absent from defaults.

## Non-goals

- Do not change SpikeInterface template metric computation.
- Do not implement polarity-aware waveform metrics in this change.
- Do not remove waveform metrics from Phy export or `cluster_info.tsv`.
