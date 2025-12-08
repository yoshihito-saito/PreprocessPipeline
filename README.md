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

