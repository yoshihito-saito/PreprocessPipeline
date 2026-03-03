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

## Preprocess Pipeline Steps

1. Make session metafile context
2. Discover input recordings
3. Build merge points
4. Prepare sidecar dat files
5. Export analog/digital events
6. Load and concatenate amplifier dat
7. Save raw dat (optional)
8. Attach probe and mark bad channels
9. Run CMR preprocess (optional)
10. Run TTL artifact removal (optional)
11. Run high-amplitude artifact removal (optional)
12. Write final dat output
13. Write LFP output (optional)
14. Build session.mat
15. Run state scoring (optional)
16. Run spike sorter (optional)
17. Save manifest and return

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
