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
- [ ] Artifact removal (`cleanArtifacts`, pulse-based cleaning)
- [ ] Denoise (`removeNoise`)
- [x] State scoring
- [ ] Spike sorting
- [ ] Acceleration extraction (`getAcceleration` / `computeIntanAccel`)
- [ ] Tracking/DLC (`getPos`, `path_to_dlc_bat_file`, `general_behavior_file`)
- [ ] Session summary (`runSummary` / `sessionSummary`)
- [ ] Concatenation option (`fillMissingDatFiles`)
- [ ] Regression tests for MATLAB vs Python output
- [ ] End-to-end comparison notebook updates and documentation cleanup
