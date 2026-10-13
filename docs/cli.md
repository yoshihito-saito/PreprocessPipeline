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

## Behavior and CellExplorer

The ephys CLI and persistent Stage list do not export tracking behavior or launch
MATLAB CellExplorer processing. Use the GUI's Behavior calibration/export actions
and CellExplorer launch action for those steps. Camera acquisition settings in
the pipeline JSON can still control the exported synchronization inputs.

Behavior export produces `<basename>.animal.behavior.mat`. Postprocessing produces
Phy-compatible sorting/metrics; CellExplorer processing is a separate action.


## Bounded TeraSort backend and waveform length

Select `preprocess.sorter="terasort"`, point `preprocess.sorter_path` to the
updated TeraSort integration checkout with its `.venv` (or its Python executable),
and use `preprocess.sorter_config_path="sorter/TeraSort_config.yaml"`.
The runtime must include [TeraSort PR #1](https://github.com/ayalab1/TeraSort/pull/1)
(or a later version containing it) for the preprocessed/source-column manifest,
configurable bounded nt and flat Phy export contract.
The GUI offers the same sorter/path/config controls. Per-session configuration
copies and sorter configuration snapshots use the existing pipeline workflow.

The default PP configuration contains `nt: 41`: 2.05 ms at 20 kHz. TeraSort's
standalone default remains 61. `nt` is an odd integer at least 21, required by
the existing 17-sample raw-anchor score plus its two-sample shift holdouts.
Sample rate, count scale and source-column geometry come from the input metadata;
changing `nt` does not resample or reinterpret the binary.

Seed learning, local model alignment, CPU/CUDA matching, novelty/PCA observations,
waveform caches and Phy templates use the same `nt`. The default strict template
overlap window follows `nt`; scoring half width, shift radius, thresholds,
distance/radius settings and other selected-profile values are preserved.
Optional config fields include `half_width`, `shift_radius`,
`overlap_window_samples`, `halo_ms`, `core_seconds`, `shard_seconds`,
`read_buffer_mb`, `prefetch_depth` and `vram_limit_gb`.

The backend retains full binary width and original channel IDs, explicitly marks
preprocessed input, and publishes flat Phy output per pipeline partition.
The current bounded geometry is planar XY; nonzero or unknown active z coordinates
are rejected rather than projected silently.
WILD-NP uses its existing independent probe partitions. Native checkpoints and
seed banks remain in `.terasort/` alongside the Phy export. Persistent retries
reuse the same native output; standalone retries use `--terasort-resume`.
A changed `nt` requires a new output/seed bank: incompatible cached model lengths
and recorded input/configuration identities are rejected. Existing 61-sample
runs retain their default configuration identity.

Only the bounded `session-sort` route is connected here. KS4-compatible sorting,
bounded KS1 and the Windows-specific cuBLAS bridge retain their existing behavior.
