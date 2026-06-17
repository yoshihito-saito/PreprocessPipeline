# PreprocessPipeline

## Overview

PreprocessPipeline is a preprocessing and postprocessing pipeline for spike sorting and neural data analysis.

## Installation

Clone the repository and move into the repository directory first:

```bash
git clone https://github.com/yoshihito-saito/PreprocessPipeline.git
cd PreprocessPipeline
```

Then run the setup command for your OS and activate the environment.

### Windows

```powershell
.\setup_env_windows.bat
conda activate preprocess
```

Start the GUI:

```powershell
preprocess-gui
```

Rebuild the Windows environment from scratch:

```powershell
.\setup_env_windows.bat --force-recreate
conda activate preprocess
```

### Linux/macOS

```bash
python scripts/setup_env.py
conda activate preprocess
```

Start the GUI:

```bash
preprocess-gui
```

Rebuild the Linux/macOS environment from scratch:

```bash
python scripts/setup_env.py --force-recreate
conda activate preprocess
```

### Advanced: Linux Locked Environment

Most users should use the OS-specific setup commands above. The files in
`env_locks/` are for exactly recreating the known working Linux server
environment when dependency changes cause NumPy/SciPy/PyTorch compatibility
issues.

Available lock/audit files:

- `env_locks/environment-linux-64.lock.yml`: full conda export including pip packages.
- `env_locks/conda-linux-64.explicit.txt`: exact conda package URLs for linux-64.
- `env_locks/pip-freeze.txt`: raw `pip freeze` output for auditing.

Exact Linux recreate:

```bash
conda env create -f env_locks/environment-linux-64.lock.yml
conda activate preprocess
python -m pip install -e .
```

Known working core versions are Python `3.11.14`, NumPy `1.26.4`, SciPy
`1.16.3`, Numba `0.62.1`, llvmlite `0.45.1`, SpikeInterface `0.103.2`, and
PySide6 `6.9.2`. PyTorch installs are pinned to Torch `2.9.1` and
TorchVision `0.24.1`.

## PreprocessPipeline GUI

The GUI provides basepath selection, preprocess/postprocess settings, `chanMap.mat` preview, preflight checks, run buttons, and pipeline logs.

Recommended command after installing the package in the active environment:

```bash
conda activate preprocess
preprocess-gui
```

The command starts the standalone Qt desktop GUI. It requires a working display
server, for example a local desktop session, X forwarding, or a VS Code remote
display setup.

If the package entry point is not registered in the active environment, use the
repository launcher:

```bash
python launch_gui.py
```

Convenience wrappers call the same Qt launcher:

Windows:

```powershell
.\launch_gui.bat
```

Linux/macOS:

```bash
./launch_gui.sh
```

To use a specific GUI default parameter file:

```bash
python launch_gui.py --config config/preprocess_gui_default_config.json
```

For private machine-specific defaults, create
`config/preprocess_gui_default_config.local.json`. The GUI loads this untracked
file automatically when `PREPROCESS_GUI_DEFAULT_CONFIG` is not set, while the
tracked public default config stays free of personal paths. In the desktop GUI,
use `Load config` and `Save config` to browse arbitrary GUI config JSON files.
When `local_root` is blank, the GUI automatically uses
`<PreprocessPipeline root>/preprocess_tmp` as the local working directory and
creates it if needed.

Advanced direct module launch:

```bash
python -m src.preprocess.gui.app
```

### Phy GUI

Install this only if you want to inspect or curate sorting results in the Phy GUI.

```bash
pip install git+https://github.com/cortex-lab/phy.git
```

### Phy2 Plugins

Install this only if you use the Phy plugin workflow.

1. Download the plugins from `https://github.com/petersenpeter/phy2-plugins`.
2. Copy the `plugins` folder to your Phy config directory.
   Linux/macOS: `~/.phy`
   Windows: `%USERPROFILE%\\.phy`
3. Copy `tempdir.py` from this repository's `plugins` directory into `*YourPhyDirectory*/phy/utils`.
4. If you use KlustaKwik on Windows, install `Visual C++ Redistributable for Visual Studio 2013`.
   x64: `https://www.microsoft.com/en-us/download/details.aspx?id=40784`

### MATLAB

Install MATLAB separately only if you use MATLAB-dependent steps.

## Kilosort1 MATLAB/CUDA Compilation

Only needed if you want to compile the bundled Kilosort1 copy in `sorter/KiloSort1/CUDA`.

### Windows

1. Install Visual Studio 2022 with `MSVC v143 - VS 2022 C++ x64/x86 build tools (v14.36-17.6)`.
2. In MATLAB, go to `sorter/KiloSort1/CUDA`.
3. Run:

```matlab
cd(fullfile('<PreprocessPipeline repo root>', 'sorter', 'KiloSort1', 'CUDA'))
mexGPUall
```

### Linux

1. In MATLAB, go to `sorter/KiloSort1/CUDA`.
2. Run:

```matlab
cd(fullfile('<PreprocessPipeline repo root>', 'sorter', 'KiloSort1', 'CUDA'))
mexGPUall
```

## Workflow

### Setup and Configuration

- Select Data: Choose the folder containing raw recording files.
- Map Channels: Define probe geometry and exclude known bad channels.
- Set Parameters: Configure filtering, artifact removal rules, and spike sorting options.

### Data Preparation

- Merge Files: Discover and concatenate raw `.dat` files across subsessions.
- Extract Events: Export analog, digital, and TTL event timestamps.

### Signal Processing

- Filter: Apply bandpass filtering and Common Median Reference (CMR).
- Remove Artifacts: Detect and remove TTL stimulation artifacts and high-amplitude noise windows.

### Output and Analysis

- Save Clean Data: Export the cleaned continuous `.dat` and downsampled `LFP` files.
- State Scoring: Optionally run sleep/wake state scoring.
- Spike Sorting: Run Kilosort (or another sorter) to extract unit candidates.

### Post-Processing

- Refine Sorting: Clean sorting outputs by removing duplicate spikes, merging fragmented units, splitting outliers, and labeling noisy units.

## Artifact Removal

- TTL artifact removal: `remove_artifact_TTL=True`
- TTL channel selection: `artifact_TTL_channel` (0-based `[0..15]`)
- TTL edge behavior:
  - default: rising edges only (`digitalIn.timestampsOn`; `artifact_TTL_include_offset=False`)
  - include falling edges: `artifact_TTL_include_offset=True` (`timestampsOn + timestampsOff`)
- TTL cleaning params: `artifact_TTL_ms_before`, `artifact_TTL_ms_after`, `artifact_TTL_mode`, `artifact_TTL_by_group`
- High-amplitude artifact removal: `remove_highamp_artifact=True`
- High-amplitude params: `highamp_*`, `highamp_ms_before`, `highamp_ms_after`, `highamp_mode`, `highamp_remove_by_group`
- Config index inputs are 0-based: `artifact_TTL_channel`, `sw_channels`, `theta_channels`, `reject_channels`, `alt_sort`
- Output files:
  - `basename.artifactTTL.events.mat`
  - `basename.artifactHigh.events.mat`

## Autosplit

Autosplit first identifies outlier spike candidates from PCA features using Mahalanobis distance. A waveform rescue step is then applied only to those candidates.

- Main idea:
  - candidate spikes are rescued only when waveform shape is similar to the clean template
  - and their best-channel PTP amplitude stays within `median(clean_amp) +/- split_amp_mad_scale * MAD(clean_amp)`
- Main parameter:
  - `split_amp_mad_scale = 10.0`
  - smaller values are stricter and keep more splits

Related autosplit settings in the notebook include `split_contamination`, `split_threshold_mode`, `split_wf_threshold`, `split_wf_n_chans`, and `split_amp_mad_scale`.

## Postprocess Metrics and Noise Rules

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

Noise thresholds:

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
- `firing_rate_lt = 0.01`
  - Exclude units with firing rate that is too low.

Waveform-shape thresholds are computed and exported for review, but are not
enabled as default hard noise criteria because they are sensitive to waveform
polarity and triphasic templates. The GUI does not expose these thresholds.
Add these keys explicitly to `noise_thresholds` in code or notebooks to opt in:

- `peak_to_valley_gt`
- `peak_trough_ratio_lt`
- `halfwidth_gt`
- `slope_lt`

## Python Implementation Status

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
