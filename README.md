# 🧩 Overview

A  preprocessing pipeline for spike sorting and neural data analysis.
This project provides a streamlined workflow for loading raw electrophysiological recordings, applying preprocessing steps (such as filtering, artifact removal, and spike detection), and preparing data for downstream analysis or sorting algorithms.

# ⚙️ Installation
### Environment
```
conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets
conda activate phy2
pip install git+https://github.com/cortex-lab/phy.git
pip install "spikeinterface[full]"
pip install kilosort[gui]
pip uninstall torch

Install pytorch (Check compatible version from here)
https://pytorch.org/get-started/locally/
e.g. for cuda 11.8
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

## Pipeline Steps (Preprocess + Postprocess)

### Preprocess (`run_preprocess_session`)

1. Make session context: resolve `basepath/basename/output_dir`, ensure `.xml/.rhd`, and load metadata.
2. Discover recordings: find input `amplifier.dat` files and build acquisition catalog.
3. Build merge points: compute cumulative sample boundaries and save `basename.MergePoints.events.mat`.
4. Prepare sidecar DATs: concatenate `analogin/digitalin/auxiliary/time` binaries when needed.
5. Export analog/digital events: generate `*.events.mat` from sidecar signals.
6. Load and concatenate amplifier DATs: read each subsession and combine into one recording.
7. Save raw DAT (optional): write `basename_raw.dat` when `save_raw=True`.
8. Attach probe and mark bad channels: apply `chanMap` and bad-channel masks.
9. Run CMR preprocess (optional): bandpass + referencing on good channels (shape-preserving).
10. Run TTL artifact removal (optional): detect TTL windows and clean waveform segments.
11. Run high-amplitude artifact removal (optional): detect large transients and clean segments.
12. Write final DAT output: save `basename.dat` (preprocessed if enabled).
13. Write LFP output (optional): generate and save `basename.lfp`.
14. Build `session.mat`: write Neurocode-compatible session metadata.
15. Run state scoring (optional): generate sleep-state outputs and figures.
16. Run spike sorter (optional): run configured sorter (e.g., Kilosort/Kilosort4).
17. Save manifest and return: persist params/manifest and return `PreprocessResult`.

### Postprocess (`run_postprocess_session` / `run_postprocess_from_preprocess`)

1. Resolve inputs: decide Kilosort folder, recording source (`recording` or `dat_path`), and bad channels.
2. Prepare output folders: create `<KilosortFolder>_spi` and `analyzer_cache` (binary analyzer mode).
3. Build postprocess recording: attach probe and optionally preprocess good channels only.
4. Pre-filter low-rate units: mark units with firing rate `< 0.01 Hz` as `noise` in `cluster_group.tsv`.
5. Load sorting from Phy: read non-noise clusters (`exclude_cluster_groups` applied).
6. Run core curation: remove duplicated spikes and redundant units.
7. Merge and split units: merge similar units, then auto-split outliers.
8. Compute final features/metrics: calculate quality features and save `quality_metrics.csv`.
9. Export Phy output: export into `<KilosortFolder>_spi` and fix `params.py` (`dat_path`, `hp_filtered`).
10. Relabel noise from metrics: update `cluster_group.tsv` based on quality thresholds.
11. Return result: output paths + unit/spike summary in `PostprocessResult`.

## Artifact Removal Options (Current)

- TTL artifact removal: `remove_artifact_TTL=True`
- TTL channel selection: `artifact_TTL_channel` (0-based `[0..15]` or 1-based `[1..16]`)
- TTL edge behavior:
  - default: rising edges (`digitalIn.timestampsOn`)
  - include falling edges: `artifact_TTL_include_offset=True` (`timestampsOn + timestampsOff`)
- TTL cleaning params: `artifact_TTL_ms_before`, `artifact_TTL_ms_after`, `artifact_TTL_mode`, `artifact_TTL_by_group`
- High-amplitude artifact removal: `remove_highamp_artifact=True`
- High-amplitude params: `highamp_*`, `highamp_ms_before`, `highamp_ms_after`, `highamp_mode`, `highamp_remove_by_group`
- Output files:
  - `basename.artifactTTL.events.mat`
  - `basename.artifactHigh.events.mat`


## Python Implementation of Neurocode `preprocessSession` - To do

- [x] `session` metafile (`basename.session.mat`)
- [x] `MergePoints` metafile (`basename.MergePoints.events.mat`)
- [x] Concatenate `.dat` files (`basename.dat`) across multiple sessions
- [x] Analog/Digital input processing (`analogin.dat`, `digitalin.dat`, `*.events.mat`) (needs double-check)
- [x] LFP extraction (exact sample-level parity)
- [x] Bad-channel handling (sorting target channels and output channel maps)
- [x] Artifact removal (`remove_artifact_TTL`, `remove_highamp_artifact`)
- [ ] Denoise (`removeNoise`)
- [x] State scoring
- [x] Spike sorting
- [ ] Acceleration extraction (`getAcceleration` / `computeIntanAccel`)
- [ ] Tracking/DLC (`getPos`, `path_to_dlc_bat_file`, `general_behavior_file`)
- [ ] Session summary (`runSummary` / `sessionSummary`)
- [ ] Concatenation option (`fillMissingDatFiles`)
- [ ] Regression tests for MATLAB vs Python output
- [ ] End-to-end comparison notebook updates and documentation cleanup
