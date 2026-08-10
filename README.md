# PreprocessPipeline

## Overview

PreprocessPipeline is a preprocessing and postprocessing pipeline for spike sorting and neural data analysis.

## Installation

### Windows Installation

```bash
git clone https://github.com/yoshihito-saito/PreprocessPipeline.git
cd PreprocessPipeline
python scripts/setup_env.py
conda activate preprocess
```

To rebuild the environment from scratch:

```bash
python scripts/setup_env.py --force-recreate
conda activate preprocess
```

### Linux Installation

```bash
git clone https://github.com/yoshihito-saito/PreprocessPipeline.git
cd PreprocessPipeline
python scripts/setup_env.py
conda activate preprocess
```

To rebuild the environment from scratch:

```bash
python scripts/setup_env.py --force-recreate
conda activate preprocess
```

## Run GUI

The GUI provides basepath selection, preprocess/postprocess settings, `chanMap.mat` preview, preflight checks, run buttons, and pipeline logs.

Recommended command:

```bash
conda activate preprocess
preprocess-gui
```

The command starts the standalone Qt desktop GUI. It requires a working display
server, for example a local desktop session, X forwarding, or a VS Code remote
display setup.

## Persistent Local and Slurm Runs

Configure scientific parameters under **Ephys > Preprocess** and
**Ephys > Postprocess**, then use **Ephys > Run** to run the standard raw-data
workflow as persistent Stages:

```text
preprocess -> sorting -> postprocess
```

Each Run is stored under `<Local working dir>/.pipeline/<run-id>/`. The directory holds
the immutable analysis snapshot, execution resources, provenance, per-Attempt
specifications, logs, backend job IDs, results, failures, and a regenerable
`state.json`. Keep this directory; it is the recovery and audit record.

Local execution uses detached workers. Closing the GUI does not cancel the Run.
When the same session is selected after reopening the GUI, its active-Run claim
reconnects the compact **Ephys > Run** status automatically. Slurm also uses the
same Stage worker, submitted through one `sbatch` job per Attempt with `afterok`
dependencies. New Stage stdout/stderr and Kilosort progress are mirrored into
the main GUI Log while the persistent files remain the complete audit record.

Backend selection is strict:

- With no explicitly saved backend choice, an environment with the required
  Slurm client commands initially selects explicit **Slurm**. Other environments
  initially select **Auto**. A saved backend choice is preserved. A temporary
  scheduler outage therefore reports a Slurm error instead of silently changing
  this server default into a Local run.
- **Auto** uses Slurm only when the configured capability checks pass before any
  submission; otherwise it resolves to Local.
- **Local** never depends on Slurm commands.
- **Slurm** reports an error when Slurm is unavailable. It never silently starts
  the analysis locally.
- An ambiguous `sbatch` outcome is recorded as `lost`, not as an ordinary
  retryable failure, and is never retried locally.

For Slurm, the Local working directory, input data, sorter installation, and
output paths must be visible from the compute node. The **Run** page requests
independent **CPU cores** and host **Memory (GiB)** for every Stage. Preprocess
and postprocess request no GPU; Sorting requests exactly one untyped GPU and
shows `1 (auto)`. The GPU device ID is intentionally not selectable: Slurm
chooses an available device when the job starts and exposes it to the worker.
CPU values are exact requests and are not capped by the GUI host's CPU count.

GUI-created Runs always use the partition walltime default and omit the Slurm
`--time` directive. On this cluster's `regular` partition,
`DefaultTime=NONE`/`MaxTime=UNLIMITED`, so the effective limit is unlimited.
The execution model and CLI retain explicit walltime support for scripted or
backward-compatible use. Resources are released as soon as the Stage exits; a
hung Run can be cancelled with the single bottom **Force stop** button.

The persistent workspace is derived automatically from Local working dir, and
each Slurm script checks that it is readable and writable on its assigned
compute node before starting the worker. Paths under the system temporary
directory still produce a preflight warning.

Cancellation is explicit and preserves logs and partial outputs. The GUI's
**Force stop** cancels every active Local/Slurm Stage in the current Run. Stage
cancel, Attempt retry, manual reconcile, and resume remain controller/CLI
recovery operations rather than normal GUI controls. A new GUI Run reuses
compatible completed Stages and creates immutable Attempts from the first Stage
that must run; it never overwrites prior Attempt records or silently changes
scientific settings. A Stage is complete only after its expected outputs pass
validation.

The existing **Basepath** field may also point directly to a processed session,
for example `sorting_temp/RM018_day33_260611`. The GUI keeps using that directory
as the scientific output directory and recovers the original raw-data path from
`preprocess_run.yaml` or legacy preprocess metadata. With overwrite disabled,
compatible and deeply validated Stages are reused; execution restarts at the
first incomplete, invalid, or parameter-incompatible Stage. With overwrite
enabled, the requested Stage scope is recomputed. A partial processed folder
whose raw-data source cannot be recovered is rejected before a preprocess rerun.

The bottom **Move outputs to storage** action shows its destination before
moving. A single-day Run defaults to Basepath. A multi-day Run defaults to a
new sibling directory named from **Multi-day name**
(`<Basepath parent>/<Multi-day name>`), keeping combined output out of the first
raw day. **Browse save dir** selects an exact existing destination and updates
the displayed Basepath; the Local source resolved before that selection remains
the move source. The existing `.dat`, overwrite, and clean-Local options still
control which files are moved and whether the Local session is removed.

Only one persistent Run may mutate a processed session at a time, even when two
controllers use different Run workspaces. **Run all**, **Preprocess only**, and
**Postprocess only** are available only on the Run page and all use persistent
Local/Slurm execution. Phy and
CellExplorer remain interactive Local applications, do not request a Slurm job
or GPU, and are blocked while that session has an active persistent Run.

The visible successful-session record is intentionally compact:

```text
<session>/preprocess_run.yaml
<session>/preprocess.log
<session>/sorter_partition_manifest.json       # when partitioned sorting is used
<session>/Kilosort_<timestamp>/kilosort.log
<session>/Kilosort_<timestamp>/sorter_config_resolved.yaml
```

`preprocess_run.yaml` records the exact analysis parameters, execution
resources, input/code/environment provenance, Stage timings, scheduler job IDs,
terminal telemetry, validation, and warning/error summary. `preprocess.log` is
the consolidated human-readable Run log. Persistent Runs do not generate
`preprocessSession.log`; successful finalization removes Stage stdout/stderr,
SpikeInterface helper JSON, and an exactly duplicated `matlab_run.log`. The
exact submitted `job.sbatch` and submission intent remain in the hidden Run
record. Failed, cancelled, ambiguous, or lost Runs retain their full
diagnostics under `<Local working dir>/.pipeline/<run-id>/`.

Large `.dat` and `.lfp` outputs are published by same-filesystem atomic replace,
so a failed rewrite does not replace the previous valid file. Sorting reruns use
a new timestamped Kilosort directory. Postprocess overwrite versions every prior
`*_spi` directory before recomputation, so an interrupted replacement cannot
destroy a valid result; `.phy` and `phy.log` remain preserved with it.

On POSIX, every Kilosort Attempt uses its own immutable shim, startup file, and
MATLAB Processes job-storage directory. The generated startup runs only in the
parent MATLAB process, so concurrent or previously interrupted Attempts do not
share configuration or a MATLAB job queue.

An Attempt whose scheduler outcome is `lost` or ambiguous is not offered for
normal retry: investigate the scheduler first, because submitting a replacement
could duplicate a job that Slurm actually accepted. Both whole-Run cancellation
and cancellation from a selected Stage (including downstream Attempts) are
available.

The controller and worker are also available directly:

```bash
preprocess-run-controller --help
preprocess-stage-worker --help
```


## Phy2

Install this for curate sorting results in the Phy GUI.

```bash
pip install git+https://github.com/cortex-lab/phy.git
```

### Phy2 Plugins

Install this to use the Phy plugin workflow.

1. Download the plugins from `https://github.com/petersenpeter/phy2-plugins`.
2. Copy the `plugins` folder to your Phy config directory.
   Linux/macOS: `~/.phy`
   Windows: `%USERPROFILE%\\.phy`
3. Copy `tempdir.py` from this repository's `plugins` directory into `*YourPhyDirectory*/phy/utils`.
4. If you use KlustaKwik on Windows, install `Visual C++ Redistributable for Visual Studio 2013`.
   x64: `https://www.microsoft.com/en-us/download/details.aspx?id=40784`

### MATLAB

Install MATLAB separately.

## Kilosort1 MATLAB/CUDA Compilation

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

Waveform-shape thresholds are computed for review, but they are not used for
the default noise decision.

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
