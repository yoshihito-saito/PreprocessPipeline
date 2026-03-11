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

## Pipeline Workflow

### 1. Setup & Configuration ⚙️
- Select Data: Choose the folder containing raw recording files.
- Map Channels: Define probe geometry and exclude known bad channels.
- Set Parameters: Configure filtering, artifact removal rules, and spike sorting options.

### 2. Data Preparation 📂
- Merge Files: Discover and concatenate raw `.dat` files across subsessions.
- Extract Events: Export analog, digital, and TTL event timestamps.

### 3. Signal Processing (Cleaning) 🧹
- Filter: Apply bandpass filtering and Common Median Reference (CMR).
- Remove Artifacts: Detect and remove TTL stimulation artifacts and high-amplitude noise windows.

### 4. Output & Analysis 📊
- Save Clean Data: Export the cleaned continuous `.dat` and downsampled `LFP` files.
- State Scoring: Optionally run sleep/wake state scoring.
- Spike Sorting: Run Kilosort (or another sorter) to extract unit candidates.

### 5. Post-Processing 🛠️
- Refine Sorting: Clean sorting outputs by removing duplicate spikes, merging fragmented units, splitting outliers, and labeling noisy units.

### Postprocess Metrics and Noise Rules

- `quality_metrics`:
  - `firing_rate`
  - `isi_violation`
  - `presence_ratio`
  - `snr`
  - `amplitude_median`
- `template_metrics`:
  - `peak_to_valley`
  - `peak_trough_ratio`
  - `half_width`
  - `repolarization_slope`
  - `recovery_slope`
  - `slope = min(abs(repolarization_slope), abs(recovery_slope)) / 1000` (`uV/ms`)

noise thresholds:

- `isi_violations_ratio_gt = 5.0`
- `isi_violations_count_gt = 50.0`
- `presence_ratio_lt = 0.1`
- `snr_lt = 2.0`
- `amplitude_median_lt = 5.0`
- `amplitude_median_gt = 2000.0`
- `peak_to_valley_gt = 0.85`
- `peak_trough_ratio_lt = -0.5`
- `halfwidth_gt = 0.4`
- `slope_lt = 100.0`
- `firing_rate_lt = 0.01`

Notes:

- In this pipeline, `half_width` and `peak_to_valley` are converted to milliseconds before thresholding and export to Phy.
- `amplitude_median_lt` / `amplitude_median_gt` use `abs(amplitude_median)`, so waveform sign does not affect the threshold.
- `peak_to_valley_gt = 0.85` means clusters above `0.85 ms` are marked as noise.
- `peak_trough_ratio_lt = -0.5` is unitless and marks clusters with ratio `<= -0.5` as noise.
- `halfwidth_gt = 0.4` means clusters above `0.4 ms` are marked as noise.
- `slope_lt = 100.0` is interpreted in `uV/ms`.
- If you rerun `mark_noise_clusters_from_metrics(...)` manually in the notebook, keep the same threshold dict there or the Phy labels will reflect the manual values instead of `post_cfg.noise_thresholds`.

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
