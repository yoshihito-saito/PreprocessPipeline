# Kilosort4 Customization Plan

## Goal

- Reduce over-merge.
- Reduce cross-cluster double detection.
- Target recording rate: `20000.0 Hz`.

## Working hypothesis

- Some units differ mainly by amplitude.
- KS4 weakens that difference through whitening, local PC features, and normalized template similarity.
- Result: raw waveforms can look different while clustering features look too similar.

## First change

- File: `sorter/Kilosort4/kilosort/clustering_qr.py`
- Add an amplitude-aware clustering feature.
- Start with a small flag-gated change, e.g. `use_amplitude_feature`.
- Candidate feature: `log(norm(tF) + eps)`.

## If that is not enough

- `template_matching.py`: add an anti-merge guard for clusters with similar shape but different amplitude structure.
- `postprocessing.py`: add cross-cluster duplicate cleanup.

## Do not start with

- removing whitening
- changing snippet normalization in `spikedetect.py`
- large algorithm rewrites

## Validation

- Check the KS4 diagnostics notebook.
- Test on at least one known bad session.
- Compare suspicious clusters in Phy before and after.
