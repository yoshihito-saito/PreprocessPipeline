# 🧩 Overview

A  preprocessing pipeline for spike sorting and neural data analysis.
This project provides a streamlined workflow for loading raw electrophysiological recordings, applying preprocessing steps (such as filtering, artifact removal, and spike detection), and preparing data for downstream analysis or sorting algorithms.

# ⚙️ Installation
Use the OS-specific setup command below. These commands create or update the `preprocess` conda environment and install PyTorch automatically, so you do not need to run `conda env create -f environment.yml` first.

### Windows
```bat
setup_env_windows.bat
conda activate preprocess
```

To fully rebuild the environment from scratch instead of updating it:

```bat
setup_env_windows.bat --force-recreate
conda activate preprocess
```

### Linux/macOS
```bash
python scripts/setup_env.py
conda activate preprocess
```

To fully rebuild the environment from scratch instead of updating it:

```bash
python scripts/setup_env.py --force-recreate
conda activate preprocess
```

Under the hood, the setup helper uses a different strategy on Windows to avoid the `conda-forge` plus `pip torch` OpenMP conflict:

- Windows creates a minimal conda env from `environment.windows.yml`
- PyTorch is installed first with `scripts/install_torch.py`
- the project dependencies are then installed with `pip install -e .[dev,notebook]`
- Linux/macOS continues to use `conda` with `environment.yml`

PyTorch wheel selection is automatic:

- if `nvidia-smi` reports a supported NVIDIA CUDA level, the highest compatible `cu*` wheel is selected
- if no supported GPU is detected, the installer falls back to CPU wheels
- if you are replacing an older Windows environment, prefer `--force-recreate` so stale pip packages and DLLs do not survive the rebuild

- Phy GUI:
```bash
pip install git+https://github.com/cortex-lab/phy.git
```
- Phy2 plugins:
  1. Download the plugins from `https://github.com/petersenpeter/phy2-plugins`.
  2. Copy the `plugins` folder to your Phy config directory:
     - Linux/macOS: `~/.phy`
     - Windows: `%USERPROFILE%\\.phy`
  3. Copy `tempdir.py` from this repository's `plugins` directory into `*YourPhyDirectory*/phy/utils`.

  4. If you use KlustaKwik on Windows, install `Visual C++ Redistributable for Visual Studio 2013`:
     - x64: `https://www.microsoft.com/en-us/download/details.aspx?id=40784`
- Kilosort / Torch:
  - the recommended path in this repository is `setup_env_windows.bat` on Windows or `python scripts/setup_env.py` on Linux/macOS
  - on Windows the setup script keeps `conda`, but only for a minimal env; `torch`, `numpy`, `scipy`, and `spikeinterface` are then installed by `pip` to avoid mixed OpenMP runtimes
  - `scripts/install_torch.py` auto-selects a PyTorch wheel from the NVIDIA driver's reported CUDA compatibility and falls back to CPU when needed
  - manual wheel selection is still available if you need it:
    - https://pytorch.org/get-started/locally/
  - this repository uses the bundled Kilosort4 code under `sorter/Kilosort4`
  - no separate `pip install kilosort[gui]` is required for the preprocessing pipeline
  - in the notebook/config, point the sorter to the repository copy:
```python
sorter='kilosort4'
sorter_path=Path('sorter') / 'Kilosort4'
sorter_config_path=Path('sorter') / 'Kilosort4_config.yaml'
```
  - if you update Kilosort4, update the files in `sorter/Kilosort4` and keep `sorter/Kilosort4_config.yaml` in sync
- MATLAB integration:
  - MATLAB is not installed by `environment.yml`; it must be installed separately and available on the system path if you use MATLAB-dependent steps.

### Kilosort1 MATLAB/CUDA compilation

Compile the Kilosort1 copy bundled with this repository.

Repository paths:

- Kilosort1 root: `sorter/KiloSort1`
- CUDA source and `mexGPUall.m`: `sorter/KiloSort1/CUDA`

#### Windows

If you need to compile Kilosort1 on a newer Windows PC, use an older MSVC toolset that is known to work with the legacy CUDA/MATLAB build.

1. Install Visual Studio 2022 with the older MSVC toolset.
   - Open Visual Studio Installer.
   - Select `Modify`.
   - Go to `Individual components`.
   - Search for `MSVC`.
   - Enable `MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.36-17.6)`.
2. Open MATLAB.
3. In MATLAB, go to this repository's Kilosort1 CUDA folder:

```matlab
cd(fullfile('<PreprocessPipeline repo root>', 'sorter', 'KiloSort1', 'CUDA'))
```

4. Edit this repository's `sorter/KiloSort1/CUDA/mexGPUall.m` so that the `mexcuda` lines include `NVCC_FLAGS="-allow-unsupported-compiler"`.
5. Run the modified commands from `sorter/KiloSort1/CUDA`:

```matlab
mexcuda -largeArrayDims mexMPmuFEAT.cu NVCC_FLAGS="-allow-unsupported-compiler"
mexcuda -largeArrayDims mexMPregMU.cu NVCC_FLAGS="-allow-unsupported-compiler"
mexcuda -largeArrayDims mexWtW2.cu NVCC_FLAGS="-allow-unsupported-compiler"
```

The generated `.mex*` files should remain under this repository's `sorter/KiloSort1/CUDA` folder so pipeline runs use the same compiled binaries.
This is only needed for Kilosort1 MATLAB/CUDA compilation on Windows. It is not required for the base Python environment in `environment.yml`.

#### Linux

Visual Studio is not needed on Linux. Use a MATLAB-supported Linux C++ host compiler plus a CUDA toolkit version compatible with your MATLAB release.

1. Compile this repository's Kilosort1 CUDA files from MATLAB:

```matlab
cd(fullfile('<PreprocessPipeline repo root>', 'sorter', 'KiloSort1', 'CUDA'))
mexGPUall
```

The generated `.mexa64` files should remain under this repository's `sorter/KiloSort1/CUDA` folder.

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


## Artifact Removal

- TTL artifact removal: `remove_artifact_TTL=True`
- TTL channel selection: `artifact_TTL_channel` (0-based `[0..15]`)
- TTL edge behavior:
  - default: rising edges (`digitalIn.timestampsOn`)
  - include falling edges: `artifact_TTL_include_offset=True` (`timestampsOn + timestampsOff`)
- TTL cleaning params: `artifact_TTL_ms_before`, `artifact_TTL_ms_after`, `artifact_TTL_mode`, `artifact_TTL_by_group`
- High-amplitude artifact removal: `remove_highamp_artifact=True`
- High-amplitude params: `highamp_*`, `highamp_ms_before`, `highamp_ms_after`, `highamp_mode`, `highamp_remove_by_group`
- Config index inputs are 0-based: `artifact_TTL_channel`, `sw_channels`, `theta_channels`, `reject_channels`, `alt_sort`
- Output files:
  - `basename.artifactTTL.events.mat`
  - `basename.artifactHigh.events.mat`

### Autosplit

Autosplit first identifies outlier spike candidates from PCA features using Mahalanobis distance. A waveform rescue step is then applied only to those candidates.

- Main idea:
  - candidate spikes are rescued only when waveform shape is similar to the clean template
  - and their best-channel PTP amplitude stays within `median(clean_amp) +/- split_amp_mad_scale * MAD(clean_amp)`
- Main parameter:
  - `split_amp_mad_scale = 10.0`
  - smaller values are stricter and keep more splits

Related autosplit settings in the notebook include `split_contamination`, `split_threshold_mode`, `split_wf_threshold`, `split_wf_n_chans`, and `split_amp_mad_scale`.


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
  - Exclude units with an excessively high refractory-period violation ratio.
- `isi_violations_count_gt = 50.0`
  - Exclude units with too many absolute refractory-period violations.
  - When both `isi_violations_ratio_gt` and `isi_violations_count_gt` are set, the unit is marked as noise only if both thresholds are exceeded.
- `presence_ratio_lt = 0.1`
  - Exclude units with too little presence across the full recording.
- `snr_lt = 2.0`
  - Exclude units with low SNR and poorly separated waveforms.
- `amplitude_median_lt = 5.0`
  - Exclude units whose absolute median spike amplitude is too small.
- `amplitude_median_gt = 2000.0`
  - Exclude likely artifacts whose absolute median spike amplitude is too large.
- `peak_to_valley_gt = 0.85`
  - Exclude units with excessively long peak-to-valley duration.
- `peak_trough_ratio_lt = -0.5`
  - Exclude units whose peak/trough ratio is below threshold and suggests an implausible waveform shape.
- `halfwidth_gt = 0.4`
  - Exclude units with overly broad spike half-width.
- `slope_lt = 100.0`
  - Exclude units with repolarization/recovery-derived slope that is too shallow.
- `firing_rate_lt = 0.01`
  - Exclude units with firing rate that is too low.


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
- [ ] Open Ephys `analog_inputs` support (currently TTL/digital only; no analog event export path)
- [ ] Acceleration extraction (`getAcceleration` / `computeIntanAccel`)
- [ ] Tracking/DLC (`getPos`, `path_to_dlc_bat_file`, `general_behavior_file`)
- [ ] Session summary (`runSummary` / `sessionSummary`)
- [ ] Concatenation option (`fillMissingDatFiles`)
- [ ] Regression tests for MATLAB vs Python output
- [ ] End-to-end comparison notebook updates and documentation cleanup
