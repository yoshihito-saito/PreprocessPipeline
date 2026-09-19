# PreprocessPipeline

PreprocessPipeline is a GUI application for preparing extracellular electrophysiology recordings, running spike sorting, and postprocessing sorted units.

It supports Intan and Open Ephys recordings, single-day and multi-day sessions, local or Slurm execution, Kilosort, Phy, and CellExplorer-compatible outputs.

## What the pipeline does

```text
raw recordings
    -> preprocess
    -> spike sorting
    -> postprocess
    -> Phy / CellExplorer
```

- **Preprocess:** discover recordings, concatenate selected sessions, filter and reference signals, remove artifacts, export events, generate `.dat` and `.lfp`, and optionally run state scoring.
- **Spike sorting:** run Kilosort on all channels or separately by probe or shank.
- **Postprocess:** remove duplicate spikes, merge or split units, calculate quality metrics, and label noise clusters.
- **Multi-day processing:** combine selected days and subepochs while preserving their order and per-recording channel metadata.
- **Persistent execution:** run locally or through Slurm and reconnect from the GUI after it is closed.

## Installation

### Requirements

- Windows or Linux
- Git
- uv, or Conda/Miniforge for the existing setup workflow
- A graphical display for the Qt GUI
- MATLAB and a compatible GPU/CUDA setup when required by the selected sorter
- Slurm client commands when submitting jobs to a Slurm cluster

### Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git,
then run these commands (an existing Python can launch the setup script):

```bash
git clone https://github.com/yoshihito-saito/PreprocessPipeline.git
cd PreprocessPipeline
python scripts/setup_uv.py
source .venv/bin/activate
preprocess-gui
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Setup creates a Python 3.11 `.venv` using uv, installs `requirements.uv.txt`,
checks dependency consistency, and imports the vendored Kilosort4 and Phy stack.
Rerun the same command to update an existing `.venv`; it never deletes it.
An active Conda environment is not the installation target.

The default is PyTorch 2.9.1 with CUDA 13.0. For CPU use
`python scripts/setup_uv.py --torch-backend cpu`; CUDA 12.6 and 12.8 are also
available via `cu126` and `cu128`. Backend selection uses the
[uv PyTorch interface](https://docs.astral.sh/uv/guides/integration/pytorch/).
GPU sorting still requires a compatible driver and GPU.

This workflow uses `uv pip`, with a focused dependency list instead of the full
Conda export. It does not supply a validated `uv.lock`; launch from the activated
environment rather than running `uv sync`, which uses a different dependency set.
GitHub access is required for Phy and nelpy. Git and system GUI libraries must be
available on the host; uv does not install Conda's operating-system libraries.

### Install with Conda

Clone the repository and create the environment:

```bash
git clone https://github.com/yoshihito-saito/PreprocessPipeline.git
cd PreprocessPipeline
python scripts/setup_env.py
conda activate preprocess
```

Kilosort4 runs from the customized source in `sorter/Kilosort4`; setup installs
its dependencies and checks that this source imports successfully. Keep the
repository checkout available when running the pipeline. Do not replace it with
a public Kilosort release to resolve installation errors.

If an older setup failed with `No matching distribution found for
kilosort==0.1.dev1513+g9f8e7052f`, obtain the corrected repository files and rerun
`python scripts/setup_env.py`. It updates an existing partially created environment;
deleting the environment is unnecessary. Linux retains the pinned CUDA 13.0
PyTorch build and explicitly includes its official wheel index.

The standard environment does not require the legacy Python package `klustakwik2`.
If an older environment file fails to build it, use the corrected file and rerun
setup. Phy plugins that invoke an external `KlustaKwik` executable require that
executable separately; installing `klustakwik2` does not supply that configuration.

Linux setup explicitly installs `nelpy @ git+https://github.com/nelpy/nelpy.git`,
matching the dependency declared by `neuro-analysis-py==0.0.2` for Phy log summaries.
Git is included in the environment; setup needs access to GitHub. This URL tracks
the upstream default branch, not a fixed commit. Do not add the exported
`nelpy==0.2.2` index pin: that version is not available from the package index.

To recreate the environment from scratch:

```bash
python scripts/setup_env.py --force-recreate
conda activate preprocess
```

## Start the GUI

Activate the environment created by your chosen installer (`source
.venv/bin/activate` for uv on Linux, or `conda activate preprocess` for Conda), then:

```bash
preprocess-gui
```

The GUI requires a local desktop, X forwarding, or another working remote display setup.

## Quick start

### Single-day session

1. Click **Browse basepath** at the top of the window.
2. Select the raw recording directory.
3. Click **Browse local** and select a **Local working dir** for temporary and processed output.
4. Under **Ephys > Preprocess**, review the channel map and configure preprocessing and Sorting.
5. Configure unit processing under **Ephys > Postprocess** when needed.
6. Open **Ephys > Run**, select **Local** or **Slurm**, and set Stage resources.
7. Click **Run all**, **Preprocess only**, or **Postprocess only**.

### Multi-day session

1. Click **Browse for multi-days** at the top of the window and select the day directories in processing order.
2. Select the subepochs to include from each day.
3. Enter a unique **Multi-day name**.
4. Review the common channel map and processing settings.
5. Start the Run from **Ephys > Run**.

Only selected subepochs contribute acquisition data and input provenance. Changes in unselected sibling recordings do not invalidate an otherwise compatible resume.

### Run button behavior

- **Run all:** preprocess, run Sorting when enabled, and then postprocess.
- **Preprocess only:** preprocess and also run Sorting when `run_sorter` is enabled.
- **Postprocess only:** postprocess an existing Kilosort/Phy result.

## Main preprocess parameters

Channel indices shown in the GUI and configuration are **0-based** unless stated otherwise. CellExplorer channel values saved to MATLAB files are 1-based.

| Group | Main parameters | Purpose |
|---|---|---|
| Signal processing | `do_preprocess`, `bandpass_min_hz`, `bandpass_max_hz` | Enable spike-band filtering and set its frequency range. |
| Reference | `reference`, `local_radius_um` | Select the reference method and local-reference radius. |
| Channels | `chanmap_mat_path`, `reject_channels`, `bad_channels`, `zero_bad` | Define probe geometry and exclude or zero unwanted channels. |
| TTL artifacts | `artifact_TTL_channel`, `artifact_TTL_ms_before`, `artifact_TTL_ms_after`, `artifact_TTL_mode` | Remove stimulation artifacts around selected TTL edges. |
| High-amplitude artifacts | `highamp_threshold_sigma`, `highamp_ms_before`, `highamp_ms_after`, `highamp_mode` | Detect and remove unusually large signal windows. |
| LFP | `make_lfp`, `lfp_fs` | Generate a downsampled `.lfp` output. |
| State scoring | `state_score`, `sw_channels`, `theta_channels` | Generate EMG, sleep-score LFP, sleep states, episodes, and diagnostic figures. |
| Events | `analog_inputs`, `digital_inputs` | Export available analog and digital event data. |
| Sorting | `sorter`, `sorter_config_path`, `sorter_partition_mode` | Select the sorter, its configuration, and all/probe/shank partitioning. |
| Existing output | `overwrite` | Reuse compatible outputs when disabled; explicitly rebuild outputs when enabled. |

Artifact windows are specified in milliseconds, frequencies in hertz, and local-reference radii in micrometers. TTL channels, state-scoring channels, rejected channels, and alternate sorter partitions use 0-based indices.

## Main postprocess parameters

| Group | Main parameters | Purpose |
|---|---|---|
| Input | `sorting_phy_folder`, `exclude_cluster_groups` | Select a Kilosort/Phy folder and omit groups such as `noise` or `mua`. |
| Optional filtering | `apply_preprocess`, `bandpass_min_hz`, `bandpass_max_hz`, `reference` | Apply filtering/reference while constructing the postprocess recording. |
| Duplicate removal | `duplicate_censored_period_ms`, `duplicate_threshold`, `remove_strategy` | Detect and remove duplicate spikes. |
| Merge | `merge_min_spikes`, `merge_corr_diff_thresh`, `merge_template_diff_thresh` | Merge likely fragments of the same unit. |
| Autosplit | `split_contamination`, `split_wf_threshold`, `split_amp_mad_scale` | Split feature outliers while retaining waveform-compatible spikes. |
| Metrics | `metric_names`, `template_metric_names` | Choose unit quality and waveform metrics. |
| Noise labeling | `noise_thresholds`, `noise_label_only` | Label clusters using firing rate, ISI, presence ratio, SNR, and amplitude rules. |
| Existing output | `overwrite` | Reuse a complete output when disabled or replace/version it when enabled. |

For external sorting, choose the matching recording in **Postprocess recording**.
The default follows the selected sorting's `params.py`, rather than substituting
a local file with the same basename. An explicit raw or unfiltered concatenated
recording is supported: enable `apply_preprocess` to filter/reference it lazily.
Its channel columns, sample order and time origin must match the sorting; 20 kHz
and 30 kHz are both supported without resampling. A second processed `.dat` is
not required. The selected chanMap is authoritative; otherwise the map beside
the selected recording is used. Automatic GUI probe assignments derive all XML
groups; explicitly saved assignments remain explicit.
An automatic multi-target invocation represents partitions of one recording
timeline. Run independent recordings separately; matching sample rates and
in-range spike samples alone cannot establish that two external binaries share
the same time origin.

Preprocessing keeps full binary columns and, with `zero_bad=True`, zeroes bad
columns. Full-width preprocessing requires a complete map. Analysis views may
select good channels while retaining original column IDs; bad channels do not
participate in common-median reference. XML skips, channels omitted from nonempty
XML spike-detection groups, map connectivity and manual exclusions are combined.
An entirely excluded probe is recorded as skipped; an entirely excluded session
finishes sorting/postprocessing without invoking the sorter or reusing old runs.
Direct sorter CLI runs (`python -m src.preprocess.sorter_runner`) also exit
successfully when the selected channels are all excluded, logging
`reason=no_active_channels` before MATLAB/sorter setup. This no-work result leaves
existing output and session manifests untouched and creates no sorter output.
Invalid channel metadata and other execution errors still fail normally.

Postprocess analysis still uses its original 1 ms before / 2 ms after window and
microvolt metrics. Phy display templates use a centered window of the same total
length in native binary units. Spike samples and analysis/PCA data are unchanged.
When optional in-memory filtering is enabled, display templates retain that
filtering/reference, while Phy's raw view reads the original binary and applies
its own display filter; these signal-processing differences are intentional.
`amplitudes.npy` retains SI's signed spike amplitudes in analyzer units (normally
microvolts), as do scientific amplitude metrics. These are physical amplitudes,
not Kilosort's dimensionless fitting coefficients; Phylib's
`get_amplitudes_true()` coefficient conversion is not applicable to SI exports.
`copy_binary=True` instead exports the selected processed recording as a compact
binary with matching compact channel indices. Original IDs remain available in
`channel_map_si.npy`. With `recording=`, provide `num_channels` for the original
binary when supplying a selected view and a full chanMap; `apply_preprocess`
applies to the supplied recording too, so leave it false for prepared inputs.
`make_post_recording(result, pre_config)` reuses an already preprocessed final
dat and does not filter it a second time.
Explicit analyzer caches now record the source sorting and recording identity,
including channel layout, gain, and the preprocessing graph. `skip_curation=True`
rejects mismatched or older unverified caches before modifying source labels;
rebuild those with `skip_curation=False, overwrite=True` once. Binary identity
uses the resolved path, size and modification time without rereading the entire
recording; source spike/label arrays are hashed.

The bundled Kilosort2.5 exporter writes input-binary channel/whitening metadata;
custom KS2.5 installations must use this compatible `rezToPhy.m` before the
pipeline can redirect native exports. Drift-corrected native templates describe
the reference position and can differ from individual uncorrected raw snippets.
KS4 receives separate probe/shank groups. Anatomical CSV columns and rows follow
XML electrode-group/channel order, including bad channels; loading a CSV does
not automatically rewrite it.
KS2.5's `skip_kilosort_preprocessing=True` path uses a compact good-channel input
so MATLAB's `Nchan` stride agrees with its binary; the original full binary stays
unchanged. The option still requires an explicit `scaleproc` as required by SI.

Autosplit first identifies feature outliers and then applies waveform and amplitude gates. Noise thresholds ending in `_lt` reject values below the threshold; thresholds ending in `_gt` reject values above it. When both ISI ratio and count thresholds are configured, both conditions must be met to label the unit as noise.

Settings can be saved and restored with **Save config** and **Load config**. **Load config** opens this repository's `config/` directory by default.

## Local and Slurm execution

Runs can execute locally or be submitted to Slurm from **Ephys > Run**. For Slurm, set CPU and memory separately for each Stage; Sorting requests one GPU when required. Input data, output directories, MATLAB, and sorter installations must be visible from the compute node.

Closing the GUI does not cancel a persistent Run. **Force stop** requests cancellation of all active Stages in the current Run.

Run metadata, scheduler job IDs, logs, results, and failures are stored under:

```text
<local-working-directory>/.pipeline/<run-id>/
```

Keep this directory while a Run may need to be inspected or resumed.

New pipeline processes default to `PYTORCH_ALLOC_CONF=expandable_segments:True`
at package startup, before importing Torch dependencies, to reduce CUDA memory
fragmentation. This applies to GUI, CLI, local workers and Slurm workers.
If either this variable or the legacy `PYTORCH_CUDA_ALLOC_CONF` is already set,
its value is preserved exactly; the default is not added to existing settings.
For example, set `PYTORCH_ALLOC_CONF=expandable_segments:False` before launching
the GUI or worker to opt out. Restart the GUI after updating; running workers
are unaffected. In notebooks, import `src` before Torch or set the variable
before starting the kernel.

Each worker records both variables under `allocator_environment` in its
`started.json`. Sorting also prints `[Torch allocator environment]` to
`stdout.log` and includes both variables in `gpu-selection.jsonl` events.
These records confirm the worker environment, not driver support or the
allocator state in a process that already imported Torch. Expandable segments
cannot guarantee that a dataset fits in VRAM or prevent contention with other
GPU processes.

## Resume an existing session

Click **Browse local session to resume** and select the processed session directory, for example:

```text
sorting_temp/RM018_day33_260611
sorting_temp/multiday_Day14_to_Day217
```

Select the session directory itself, not `.pipeline/` or a `Kilosort_*` directory. The GUI restores the raw source paths, selected subepochs, scientific settings, execution backend, and Stage resources. Browsing does not submit a job or change `overwrite`.

With `overwrite=False`, compatible completed outputs are validated and reused. Missing outputs are generated where safe. An incompatible or untrusted existing output causes the Run to stop before replacing it.

## Main outputs

Depending on the enabled options, a processed session contains:

```text
<session>/
|-- <basename>.dat
|-- <basename>.lfp
|-- <basename>.session.mat
|-- <basename>.MergePoints.events.mat
|-- <basename>.artifactTTL.events.mat
|-- <basename>.artifactHigh.events.mat
|-- <basename>.SleepScoreLFP.LFP.mat
|-- <basename>.SleepState.states.mat
|-- preprocess_run.yaml
|-- preprocess.log
`-- Kilosort_<timestamp>/
```

Very large `SleepScoreLFP` outputs use MATLAB v7.3/HDF5 when they exceed the MATLAB v5 format limit. Ordinary MAT outputs remain in v5 format.

State-scoring channel selection reads each candidate from the memory-mapped LFP
and converts only its downsampled samples to float64. Candidate evaluation uses
at most four concurrent workers (or the smaller requested `n_jobs`); other
preprocessing stages retain their requested worker counts. All candidate channels,
scoring parameters, and output formats are preserved. Memory still grows with
recording duration because each active channel's spectral calculation spans the
full recording.

Use **Copy outputs to storage** to copy selected outputs from the Local working directory to their final destination. Local outputs are retained by default. Enable **Delete local after verified copy** only when the complete local session directory should be removed after the staged copy is content-verified and published. After verification, the destination folder and its complete tree are made readable and writable by all users, with directory traversal enabled. The GUI displays which files will be copied, retained, or deleted before starting.

## Optional tools

### Standalone multi-day channel MAD check

The auxiliary `src.stability` module can compare channel noise across the subepochs
listed in a staged `multi_day_selected_subepochs.csv`. It is not called by the GUI or
preprocess pipeline.

```python
from src.stability import analyze_multi_day_mad

result = analyze_multi_day_mad(
    "/path/to/multi_day_selected_subepochs.csv",
    bandpass_min_hz=500.0,
    bandpass_max_hz=8000.0,
    window_duration_s=1.0,
    num_windows=20,
)

print(result.csv_path)     # multi_day_mad.csv
print(result.figure_path)  # multi_day_mad.png
```

To overlay every channel as a color-coded time series, with pre-filter and
post-bandpass values in separate panels, use:

```python
from src.stability import plot_multi_day_mad_by_channel

figure, _ = plot_multi_day_mad_by_channel(result.metrics, cmap="viridis")
figure.savefig("multi_day_mad_by_channel.png", dpi=200)
```

To rank adjacent-subepoch channel reversal or half-swap candidates from the saved
MAD profiles alone:

```python
from src.stability import analyze_multi_day_mapping_check

mapping = analyze_multi_day_mapping_check(result.csv_path)
print(mapping.csv_path)     # multi_day_mapping_check.csv
print(mapping.figure_path)  # multi_day_mapping_check.png
```

The mapping check evaluates a bounded set of plausible transformations separately
on each aligned 64-channel Intan headstage block. It reports identity similarity, conservative
pre/post consensus permutation gain, corrected similarity, and a priority rank.
These values shortlist transitions for manual review; they do not automatically
declare or correct a channel swap.

The CSV contains one row per selected subepoch and electrophysiology channel. The
only noise metric is the centered MAD converted to a Gaussian-equivalent scale in
µV, reported as `pre_filter_mad_uv` and `post_filter_mad_uv`. `post_filter` is
bandpass-filtered but not common-referenced. The table also records dtype and
per-channel gain/offset provenance. Intan input defaults to `int16` with
`0.195 µV/count`; pass `dtype`, `gain_to_uV`, or `offset_to_uV` when the acquisition
uses different values. Selected subepochs with inconsistent physical channel maps
are rejected. Existing outputs are not replaced unless `overwrite=True`.

### Phy

Linux setup installs Phy from the same Git commit recorded in the reference
phy2 environment. For a separate Phy installation in the environment used to
launch the GUI:

```bash
pip install "phy @ git+https://github.com/cortex-lab/phy.git@1ddcd015e0382c3fc0ba20cd99dd5b8771bb8702"
```

The GUI can launch Phy after Sorting. Optional plugins are available from [phy2-plugins](https://github.com/petersenpeter/phy2-plugins).

### MATLAB and Kilosort1

Install MATLAB separately. Kilosort1 may require compiling its CUDA functions from MATLAB:

```matlab
cd(fullfile('<PreprocessPipeline repo root>', 'sorter', 'KiloSort1', 'CUDA'))
mexGPUall
```

On Windows, install a compatible Visual Studio C++ build toolchain before compiling CUDA code.

## Current limitations

The Python pipeline does not yet implement:

- Open Ephys analog-event export
- acceleration extraction
- session-summary generation
