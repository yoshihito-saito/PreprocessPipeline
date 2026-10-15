# CLI execution guide

Run commands from the repository root in the activated environment described in
[README](../README.md#installation). Commands below use the same Python modules
on Linux and Windows. See [Parameter reference](parameters.md) for JSON keys,
defaults, units, and channel numbering.

## Choose an execution route

| Route | Configuration | Behavior |
| --- | --- | --- |
| `python -m src.preprocess.gui.run_pipeline` | GUI pipeline JSON | Direct foreground preprocessing/sorting/postprocessing. Useful inside a terminal or your own batch job. |
| `preprocess-run-controller` | An existing persistent Run directory | Submit, reconcile, cancel, or retry separate Local/Slurm Stages. |
| `python -m src.preprocess.sorter_runner` | Sorter YAML/JSON plus command options | Sort an existing binary without running session preprocessing or postprocessing. |

The installed controller command is also available as
`python -m src.execution.controller`. `preprocess-stage-worker` is launched by
the controller and is not the normal way to start a session.

## Direct single-day run

The easiest starting point is a JSON file saved through the GUI's **Save config**.
Alternatively, save this example as `session.json`, replacing the data/XML/output
paths and checking the channel settings:

```json
{
  "basepath": "/data/session01",
  "local_root": "/scratch/preprocess",
  "xml_path": "/data/session01/session01.xml",
  "preprocess": {
    "run_sorter": true,
    "sorter": "Kilosort",
    "sorter_path": "sorter/KiloSort1",
    "sorter_config_path": "sorter/Kilosort1_config.yaml",
    "reject_channels": [],
    "digital_inputs": true,
    "analog_inputs": false,
    "make_lfp": true,
    "state_score": false,
    "preprocess_worker_count": 4,
    "sorter_worker_count": 4,
    "overwrite": false
  },
  "postprocess": {
    "worker_count": 4,
    "overwrite": false
  }
}
```

This example disables optional state scoring and uses four requested workers.
Omitted fields retain GUI JSON defaults. Supply an acquisition-matching XML
with channel groups; preprocessing prepares the channel map from it. For Windows,
paths can be written as `"D:/data/session01"` and `"D:/scratch/preprocess"`.

```bash
python -m src.preprocess.gui.run_pipeline --config session.json --mode all
```

The direct command accepts just two required options:

| Option | Meaning |
| --- | --- |
| `--config` | Path to a GUI pipeline JSON file. |
| `--mode` | `all`, `preprocess`, `postprocess`, or `noise_label`. |

| Mode | Direct execution |
| --- | --- |
| `all` | Preprocess, sort if enabled, then attempt postprocessing. If sorting is disabled, an existing compatible sorting is needed for the postprocess step. |
| `preprocess` | Preprocess and also sort if `preprocess.run_sorter=true`. |
| `postprocess` | Process existing sorting and its matching recording. |
| `noise_label` | Update noise labels from existing quality metrics without full curation. |

To produce preprocessing outputs without sorting, set `preprocess.run_sorter`
to `false` and run:

```bash
python -m src.preprocess.gui.run_pipeline --config session.json --mode preprocess
```

Success exits with code 0 and prints a final JSON line prefixed by
`__PREPROCESS_GUI_RESULT__`. A processing exception exits with code 1 and prints
`__PREPROCESS_GUI_ERROR__` plus a traceback. Output includes
`<local_root>/<basename>/preprocess.log`; compatible completed files are reused
when overwrite is false. The command does not perform the final storage copy.

## Direct multi-day run

Add these top-level keys to the single-day JSON:

```json
{
  "multi_day_enabled": true,
  "multi_day_session_paths": ["/data/Day01", "/data/Day02"],
  "multi_day_selected_subepoch_paths": [],
  "multi_day_name": "animal_days01_02"
}
```

The snippet is a configuration fragment, not a separate complete input file.
Keep the `local_root`, XML, preprocessing, and postprocessing settings in the
same file. Use at least two day directories. Empty selected-subepoch paths
include all discovered subepochs; otherwise provide full source subepoch paths.
Review session/subepoch order and channel identity before concatenating.

```bash
python -m src.preprocess.gui.run_pipeline --config multiday.json --mode all
```

Mixed Intan/Open Ephys/WILD inputs still need compatible recording rates and
ephys channel layouts. ADC channels are not extra ephys channels to add to the
channel map. The multi-day staging output includes a manifest and selected
subepochs CSV so that the merged sample order can be inspected.

## Postprocess an external sorting

Save, for example, `postprocess.json`:

```json
{
  "basepath": "/data/session01",
  "local_root": "/scratch/preprocess",
  "xml_path": "/data/session01/session01.xml",
  "chanmap_path": "/scratch/preprocess/session01/chanMap.mat",
  "preprocess": {"reject_channels": [3]},
  "postprocess": {
    "sorting_phy_folder": "/scratch/preprocess/session01/Kilosort_external",
    "dat_path": "/scratch/preprocess/session01/session01.dat",
    "apply_preprocess": false,
    "exclude_cluster_groups": ["noise"],
    "worker_count": 4,
    "overwrite": false
  }
}
```

```bash
python -m src.preprocess.gui.run_pipeline --config postprocess.json --mode postprocess
```

Use the exact recording that underlies the sorting: channel columns, sampling
rate, sample order, and time origin must agree. Matching `params.py` metadata
also supplies dtype/byte offset. Enable `apply_preprocess` only for a raw input
that needs the configured filtering/reference. Full binary channel IDs remain
consistent through preprocessing, sorting, Phy, and CellExplorer metadata.

To revise the noise thresholds after metrics have been generated, edit the
`postprocess.noise_thresholds` object and run:

```bash
python -m src.preprocess.gui.run_pipeline --config postprocess.json --mode noise_label
```

This operation changes cluster-group labels. It requires the existing metrics
for the resolved postprocess output; it does not compute missing metrics.

## Persistent Local and Slurm Runs

Stage workers use Matplotlib's `Agg` backend to save diagnostic figures without
a display server. This overrides an inherited interactive backend such as
`QtAgg`, so event export and other saved plots can run on Slurm compute nodes.

Persistent execution separates preprocessing, sorting, and postprocessing into
Stages with saved analysis/configuration snapshots and per-attempt logs. The GUI
can create these Runs. To create one without opening the GUI, first add an
`execution` object to the pipeline JSON, for example:

```json
{
  "execution": {
    "requested_backend": "local",
    "workspace": "/scratch/preprocess",
    "preprocess": {"cpus": 4, "memory_mb": 32768, "gpu_count": 0},
    "sorting": {"cpus": 4, "memory_mb": 65536, "gpu_count": 1},
    "postprocess": {"cpus": 4, "memory_mb": 32768, "gpu_count": 0}
  }
}
```

Choose resource values for the actual job; the example is not a required
allocation. For Slurm, use `"requested_backend": "slurm"` and the site's
partition/account/GPU settings. All worker nodes must see the same paths and
environment. Auto backend selection can fall back to local; explicit Slurm
reports unavailable capabilities instead.

Save this small creation script as `create_run.py` in the repository root:

```python
from pathlib import Path

from src.execution.controller import create_run, resolve_backend
from src.execution.models import RequestedBackend
from src.preprocess.gui.config_model import PipelineGuiSettings

settings = PipelineGuiSettings.load(Path("session.json"))
resolved, capabilities = resolve_backend(
    RequestedBackend(settings.execution.requested_backend.lower()),
    require_sacct=settings.execution.require_sacct,
)
execution = settings.execution.to_execution_config(resolved_backend=resolved)
run_dir = create_run(
    settings=settings,
    execution=execution,
    mode="all",
    capabilities=capabilities,
)
print(run_dir)
```

```bash
python create_run.py
preprocess-run-controller --run-dir /scratch/preprocess/.pipeline/REPLACE_WITH_RUN_ID
```

Use the Run directory printed by the script. Run creation saves intent and claims
the session; the controller command submits it. No `--config` creation flag exists
on the controller. Creation supports `all`, `preprocess`, or `postprocess`.
In persistent `all` mode, disabling sorting also omits the postprocess Stage;
create a `postprocess` Run explicitly to work on an existing sorting.

### Monitor, cancel, and retry

```bash
preprocess-run-controller --run-dir /scratch/preprocess/.pipeline/REPLACE_WITH_RUN_ID --reconcile
preprocess-run-controller --run-dir /scratch/preprocess/.pipeline/REPLACE_WITH_RUN_ID --cancel
preprocess-run-controller --run-dir /scratch/preprocess/.pipeline/REPLACE_WITH_RUN_ID --cancel-stage sorting
preprocess-run-controller --run-dir /scratch/preprocess/.pipeline/REPLACE_WITH_RUN_ID --retry sorting --cpus 8 --memory-mb 65536
```

These are alternative actions, not commands to run in sequence. Reconcile updates
`state.json`; read it and `controller.log` for status, because controller output
is redirected to that log. Stage attempts contain `stdout.log` and `stderr.log`.
`analysis_config.json` and `execution_config.json` record the saved settings.

Retry is allowed for failed, cancelled, or blocked enabled Stages. It preserves
the saved analysis and can update resource allocation. For changes to channels,
filters, sorter algorithms, or other analysis settings, create a new Run. Closing
the GUI does not cancel a persistent Run; cancellation is an explicit request.

| Controller option | Meaning |
| --- | --- |
| `--run-dir` | Required existing Run directory. |
| No action flag | Submit the Run. |
| `--reconcile` | Refresh state from attempts/backend status. |
| `--cancel` | Request cancellation of the Run. |
| `--cancel-stage` | Request cancellation of `preprocess`, `sorting`, or `postprocess`. |
| `--retry` | Retry an enabled `preprocess`, `sorting`, or `postprocess` Stage. |
| `--cpus`, `--memory-mb` | Retry allocation overrides. |
| `--walltime-minutes` | Retry time-limit override in minutes. |
| `--unlimited-walltime` | Remove the explicit retry time limit; scheduler policy still applies. Mutually exclusive with `--walltime-minutes`. |
| `--gpu-gres-type`, `--gpu-constraint` | Retry GPU placement overrides. |
| `--partition`, `--account`, `--qos`, `--reservation` | Retry Slurm scheduling overrides. |

Only one action flag may be selected. Resource overrides are applied by `--retry`.

## Sort an existing binary

The standalone sorter command uses a sorter config file, not the pipeline JSON:

```bash
python -m src.preprocess.sorter_runner --sorter kilosort4 --config sorter/Kilosort4_config.yaml --dat-path /scratch/preprocess/session01/session01.dat --xml-path /scratch/preprocess/session01/session01.xml --chanmap /scratch/preprocess/session01/chanMap.mat --kilosort4-path sorter/Kilosort4 --output-folder /scratch/preprocess/session01/Kilosort4_manual
```

For Kilosort1, use `--sorter kilosort`, the Kilosort1 config, and
`--kilosort1-path sorter/KiloSort1`. Keep that repository directory's capitalization
on Linux. Kilosort2.5 uses `--sorter kilosort2.5` and
`--kilosort2-5-path sorter/Kilosort2.5`. MATLAB-based versions also need MATLAB.

MATLAB launch scripts, process-pool job files, and temporary logs use a private
directory created for each worker under the system temporary directory (including
`TMPDIR` when configured). They do not share a fixed `/tmp/preprocess_pipeline`
directory across users. The MATLAB log is copied into the sorter output directory
as `matlab_run.log`.

To choose the runtime location explicitly, set `PREPROCESS_RUNTIME_DIR` to a
directory you own that is writable on the compute node before submitting or
retrying the Run. For example:

```bash
export PREPROCESS_RUNTIME_DIR=/scratch/myuser/preprocess-runtime
mkdir -p -m 700 "$PREPROCESS_RUNTIME_DIR"
SBATCH_EXPORT=ALL preprocess-run-controller --run-dir /scratch/preprocess/.pipeline/REPLACE_WITH_RUN_ID --retry sorting
```

`SBATCH_EXPORT=ALL` passes the environment override to Slurm workers. Retrying
sorting reuses completed preprocessing and schedules a new postprocessing attempt.

| Option | Default | Meaning |
| --- | --- | --- |
| `--sorter` | `kilosort` | Sorter name: `kilosort`, `kilosort2.5`, or `kilosort4`. |
| `--config` | Sorter-specific file | Sorter YAML/YML/JSON algorithm configuration. |
| `--dat-path` | Required | Input binary path. This command does not perform session preprocessing. |
| `--xml-path` | Required at execution | Session XML. Although argparse marks it optional, the implementation requires it. |
| `--chanmap` | Unspecified | Probe geometry/connected-channel map. Supply the session map to retain channel exclusions and geometry. |
| `--sampling-frequency` | From XML | Recording rate, Hz. |
| `--num-channels` | From XML | Full binary channel count, including retained bad columns. |
| `--active-channels` | Derived | Comma-separated original 0-based channel IDs to sort. |
| `--exclude-channels` | Unspecified | Comma-separated original 0-based channel IDs to exclude. |
| `--dtype` | `int16` | Binary sample dtype. |
| `--gain-to-uV` | `0.195` | Conversion gain to µV per stored sample unit. |
| `--offset-to-uV` | `0.0` | Conversion offset in µV. |
| `--output-folder` | Required | Sorter output directory. |
| `--remove-existing-folder` | Off | Delete an existing sorter output folder before running. |
| `--kilosort1-path` | `sorter/KiloSort1` | Kilosort1 installation, matching the checked-in directory's capitalization. |
| `--kilosort2-5-path` | `sorter/Kilosort2.5` | Kilosort2.5 installation. |
| `--kilosort4-path` | `sorter/Kilosort4` | Kilosort4 repository containing the Python package. |
| `--matlab-path` | Discovery | MATLAB executable or bin directory. |
| `--matlab-max-workers` | Automatic | MATLAB process-pool worker cap. |
| `--docker-image` | Unspecified | Optional container image forwarded to the sorter runner. |
| `--sorter-verbose` | Off | Enable verbose sorter logs. |
| `--keep-temp-wh-dat` | Off | Keep the MATLAB sorter's `temp_wh.dat` instead of cleaning it up. |

Subset sorting keeps exported Phy channel metadata in the full binary's original
channel numbering. An empty set of eligible active channels is reported as
`no_active_channels` and sorting is skipped.

## Physical channel maps and existing recordings

A selected `chanMap.mat` supplies the physical geometry for preprocessing,
sorting, and postprocessing. Its channel IDs, x/y/z coordinates, shank and probe
IDs, face IDs, and custom metadata are preserved. GUI and XML exclusions only
disable entries in `connected`; loading a map does not rebuild its geometry from
the probe-template controls.

Without a selected map, preprocessing first uses the raw session's map. Explicit
probe-template assignments with an empty map path request generation; the GUI's
Generate chanMap action also requests generation. Editing geometry clears the
selected map, while editing bad channels retains it. Preview and local-CMR
preflight use the same map that the worker will use.

Persistent Runs save an immutable copy at `snapshots/chanMap.input.mat` and
include its checksum in preprocessing fingerprints. New preprocessing results
record the geometry used to produce the recording. Sorting and postprocessing
refuse an older result without that evidence or a map changed after preprocessing.

Runs made before this fix may have replaced a custom map with default geometry.
If a processed map differs from its selected source, changing only `chanMap.mat`
cannot repair local CMR or artifact detection that already used the wrong
geometry. Use a new local output root and rerun from raw data with the original
map selected. Alternatively, explicit overwrite archives the old preprocessing
products before producing replacements; existing sorter folders are retained.

The pipeline rejects incompatible cached maps when overwrite is disabled. It
does not silently repair a map and reuse the associated processed recording.

## Behavior and CellExplorer

The ephys CLI and persistent Stage list do not export tracking behavior or launch
MATLAB CellExplorer processing. Use the GUI's Behavior calibration/export actions
and CellExplorer launch action for those steps. Camera acquisition settings in
the pipeline JSON can still control the exported synchronization inputs.

Behavior export produces `<basename>.animal.behavior.mat`. Postprocessing produces
Phy-compatible sorting/metrics; CellExplorer processing is a separate action.


## TeraSort modes, KS4 settings and waveform length

Select `preprocess.sorter="terasort"`, point `preprocess.sorter_path` to the
updated TeraSort integration checkout with its `.venv` (or its Python executable),
and use `preprocess.sorter_config_path="sorter/TeraSort_config.yaml"`.
If the selected checkout uses a separate existing Python environment, set
`PREPROCESS_TERASORT_PYTHON` to that environment's Python executable before
starting PP. PP places the selected checkout's `src` first on the TeraSort child
process's import path, verifies the imported package origin, and pins a digest
of that package plus the Python path in the job identity. When a Python executable
is selected directly, PP pins the package actually importable in that environment.
Changing source files or Python path requires a new output rather than a bounded
resume; older jobs without a source identity cannot resume under this contract.
The runtime must contain the configurable modes/KS4 settings changes on
[TeraSort PR #1](https://github.com/ayalab1/TeraSort/pull/1)'s integration branch,
in addition to its preprocessed/source-column and flat Phy input/output contract.
PP does not vendor TeraSort or install its runtime. The GUI uses the existing
sorter/path/config controls; selecting the mode happens in the sorter YAML.

```yaml
mode: bounded
nt: 41
ks4_backend: deep_tiled
ks4_settings: {}  # Explicit overrides; empty preserves native KS4 defaults.
```

- `mode: bounded` calls `session-sort`: fixed-budget KS4 seed learning followed
  by the selected CUDA streaming profile, restartable shards and flat Phy export.
  This remains the PP default for large/long recordings.
- `mode: kilosort4` calls TeraSort's KS4-compatible `sort` route. Its feature and
  clustering memory can grow with recording length. It does not resume bounded
  checkpoints; persistent retries create fresh output directories.
- Original KS1 (`ks1-sort`) is a separate experimental route requiring the pinned
  original-KS1 CUDA/calibration sources, normally with MATLAB calibration. It
  rejects this integration's preprocessed/source-column input contract.

`ks4_backend` selects `auto`, `standard`, `deep_tiled` or `cublas` for full KS4
sorting or for bounded initial KS4 learning. It does not change the bounded dense
CUDA profile. The dedicated cuBLAS backend requires `nt=41` or `nt=61` and
`n_templates=6`, plus a platform-native bridge. Linux build/runtime remains unverified.

Use native KS4 parameter names inside `ks4_settings`, for example:

```yaml
ks4_settings:
  Th_single_ch: 6
  Th_universal: 9
  Th_learned: 8
  n_templates: 6
  nearest_chans: 10
  whitening_range: 32
  batch_size: 60000
  nblocks: 0
```

In bounded mode these affect initial KS4 learning only. Seed learning requires
`nblocks=0`; full `kilosort4` mode accepts its native drift setting and defaults
to `nblocks=1` when omitted. These values are not copied automatically from
`Kilosort4_config.yaml`. The default TeraSort YAML leaves scientific overrides
empty to preserve existing seed behavior. Bounded thresholds such as `floor_snr`
and `score_floor` retain their own meaning rather than being translated from KS4
thresholds. Unknown native keys/types/ranges are checked in the TeraSort runtime.

Common `nt=41` is 2.05 ms at 20 kHz. It is odd; bounded requires at least 21
for its existing raw-anchor holdout. TeraSort's standalone default remains 61. Sample rate, gain, full binary width,
source-column geometry and selected input interval come from PP input metadata.
Do not duplicate `nt`, `fs`, `n_chan_bin`, `scale`, `shift`, source/probe paths,
`tmin` or `tmax` in `ks4_settings`. PP owns high-pass filtering and reference:
`highpass_cutoff` and `do_CAR` are not sorter-config overrides here. Preprocessed
input bypasses high-pass filtering and CAR in both modes while retaining KS4
whitening/learning. No resampling or reinterpretation of the binary occurs.

The existing flat bounded controls remain supported: `half_width`, `shift_radius`,
`overlap_window_samples`, `halo_ms`, `core_seconds`, `shard_seconds`,
`read_buffer_mb`, `prefetch_depth` and `vram_limit_gb`. Additional explicit controls
are `floor_snr`, `score_floor`, `residual_passes`, `max_candidates`,
`freeze_templates`, `novelty` (`off`/`shadow`/`enroll`) and `preset`
(`default`/`fast`). These controls are rejected in `kilosort4` mode. Fast changes
residual passes, novelty admission and cache policy, so it is a scientific preset,
not another sorting mode. Omitted settings retain the selected profile, including
60-second shards, a 64 MiB read buffer and prefetch depth 4. The VRAM setting
budgets the CuPy pool, not all GPU allocations.

Both modes retain original binary-column IDs and publish flat Phy output per PP
partition. The current adapter requires planar XY geometry and rejects nonzero
or unknown active z coordinates. WILD-NP retains independent physical-probe
partitions. Postprocessing discovers every completed `Terasort_*` partition from
the saved manifest. If a TeraSort Phy export contains overlapping XY positions
across probe/shank groups, PP separates only their display X coordinates and
saves the original positions in `channel_positions_local.npy`; these display
offsets are not physical probe spacing. Input/config identities and
source/probe/settings JSON are stored in `.terasort/`; the output keeps its
source/resolved YAML snapshot and `terasort_run.json` pointer. Full KS4 writes
`terasort_settings.json` with native
effective settings and the actual selected backend. Bounded seed reports retain
the same effective KS4 record; native `run.json` records dense settings. Nonfinite
native limits are represented as strings only in the viewable effective-settings
metadata; runtime settings are unchanged.

Bounded retries retain checkpoints and seed banks. Explicit KS4 seed overrides
and backend changes are pinned in the seed plan/report and reject incompatible
resume. Changing `nt` or scientific settings requires a fresh output/seed bank.
The default empty KS4 overrides and `deep_tiled` seed backend retain existing
legacy seed-plan and PP job identities. No sorting, GUI, CUDA, resume or quality
validation is implied by the source implementation.
