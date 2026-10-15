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

## Use TeraSort for the sorting stage

Select `terasort` in the GUI, or change these keys in the existing pipeline
JSON's `preprocess` object:

```json
{
  "run_sorter": true,
  "sorter": "terasort",
  "sorter_path": "D:/TeraSort-preprocess-pipeline-integration",
  "sorter_config_path": "sorter/TeraSort_config.yaml"
}
```

The TeraSort checkout must have its own installed `.venv` and include the
PreprocessPipeline integration changes. Alternatively, set `sorter_path` to
the Python executable in an environment with that TeraSort version installed.
The adapter launches this environment as a child process and inherits the
worker's GPU visibility. It does not install TeraSort or its dependencies.

The pipeline still produces its usual final `.dat`, then passes it and the
effective `chanMap.mat` to TeraSort's standard `session-sort` route. It marks
this input as preprocessed so TeraSort skips a second reference/filter pass;
KS4 preview seed learning retains whitening, and INT16 matching retains its
headroom quantization. Postprocessing consumes the exported Phy folder through
the existing path. Existing Kilosort selection and defaults are unchanged.

`chanMap0ind` identifies columns of the actual input `.dat`, after any upstream
remapping. Geometry rows are selected by these IDs in the active-channel order.
Rejected/disconnected channels are omitted from sorting without rewriting or
renumbering the binary; Phy's `channel_map.npy` retains the original column IDs
and `params.py` records the full binary width. Distinct `(probe_id, shank_id)`
pairs stay separate even when probes reuse local shank numbers.

This route requires little-endian INT16, zero voltage offset, a positive
microvolt-per-count gain, and matching sample rate/geometry. The adapter treats
the final binary as one contiguous sample axis; it does not import WILD
`valid_samples.dat` masks or reconstruct acquisition gaps. Existing preprocessing
and disk-space preflight remain in effect, including the conservative two-times
source-size estimate; this integration does not establish end-to-end bounded
disk use for very large recordings.

The default `sorter/TeraSort_config.yaml` is `{}` and uses TeraSort's selected
session profile. To opt into 41-sample KS4 seeds with the native cuBLAS backend,
set `nt: 41` and `seed_backend: cublas` in that YAML. `nt` is the bounded
sorter's waveform length too; changing it requires a new output and seed bank.
The seed backend applies only to initial KS4 seed learning, not the bounded
matching stage. It requires the updated TeraSort checkout and its platform-native
bridge; an explicit `cublas` request fails if the bridge is unavailable. The
Kilosort4 YAML configuration is separate and is not inherited by TeraSort.
Optional runtime keys are `core_seconds`, `shard_seconds`, `read_buffer_mb`,
`prefetch_depth`, and `vram_limit_gb`. Omitted keys retain TeraSort's defaults.
The selected options are pinned in `.terasort/<output>.job.json` for retry.
The GUI makes an editable per-session copy of the selected YAML once; edit that
copy for an existing GUI session rather than expecting later template changes to
replace it.

Each result is a flat `Terasort_<timestamp>` Phy folder. Native shards, the pinned
manifest/settings, and preview seeds live beside it under `.terasort/`; retain
that directory for retry. Persistent Runs use the same output/checkpoints on
sorting retries. If different probe/shank groups have overlapping coordinates,
the integration separates them horizontally in Phy's `channel_positions.npy`.
`channel_positions_local.npy` retains the original selected-channel geometry;
`channel_positions_display.json` records the display offsets. These offsets
do not describe physical spacing between implanted probes. Sorting geometry,
binary channel IDs, and curation remain unchanged. After repairing an existing
export, save curation and close Phy, then reopen with `--clear-cache` to refresh
geometry-dependent caches. The standalone equivalent for an already preprocessed binary is:

```bash
python -m src.preprocess.sorter_runner --sorter terasort --config sorter/TeraSort_config.yaml --dat-path /scratch/preprocess/session01/session01.dat --xml-path /scratch/preprocess/session01/session01.xml --chanmap /scratch/preprocess/session01/chanMap.mat --terasort-path /opt/TeraSort --input-is-preprocessed --gain-to-uV 0.195 --output-folder /scratch/preprocess/session01/Terasort_manual
```

Use the actual recording gain, not the example value. For a raw binary, omit
`--input-is-preprocessed`. To resume the same standalone output, repeat the
command with `--terasort-resume`; changed inputs/settings require a new output.
A 600-second WILDX Neuropixels excerpt completed preprocessing, GPU sorting,
checkpoint resume, and Phy export through this route. Full-session GUI runs,
scientific sorting quality, and interactive Phy/postprocessing remain unverified.

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

## WILDX Neuropixels input and local excerpts

WILDX NP is detected from `neuropixels_probe*_metadata.json` receipts with
`format="WILDX NP applied probe receipt"` and `format_version=1`. The initial
route accepts one complete merged segment of NP1 `PRB_1_4_0480_1` probes per
session. Incomplete segments, multiple segments with unknown gap timing,
other probe models/storage layouts, and mixed acquisition formats are rejected.

Select the day folder as basepath when it contains
`NP_SESSION_*/segment_*`. A capture folder with a direct segment child, or the
segment itself, is also supported. The selected basepath and its output basename
are retained. Each discovered segment must contain `neuropixels_ap.dat`,
`neuropixels_probes.txt`, and one applied-probe JSON
per physical port. `neuropixels_lfp.dat` is required when `make_lfp=true`.
No input XML or IMRO is needed for this route: saved stream metadata determine
the AP width, little-endian INT16 storage, sample rate, and physical-port order;
the receipts determine hardware-channel/site selection and validity.

The worker creates XML, `chanMap.mat`, channel coordinates, and
`wildx_np_metadata.json` locally. The map retains every saved binary column,
including disconnected/standby channels with no site. Unknown coordinates stay
NaN; these channels are excluded from reference estimation and sorting. The NP1
geometry uses physical site IDs and micrometre coordinates, independent of
hardware channel number and bank. Filtering/reference estimation runs separately
for each physical probe, preserving its relative coordinates and the configured
reference radius/operator. Geometry and nominal gain conversion follow
the [Open Ephys NP1 implementation](https://github.com/open-ephys-plugins/neuropixels-pxi/blob/8c2d85a2ace63fd873b1f74fb7516a660b30562d/Source/Probes/Geometry.cpp#L141)
and its [gain table and voltage conversion](https://github.com/open-ephys-plugins/neuropixels-pxi/blob/8c2d85a2ace63fd873b1f74fb7516a660b30562d/Source/Probes/Neuropixels1.cpp#L471).

AP values retain the ADC count scale through the existing preprocessing path.
The nominal NP1 scale is `1.2e6 / 1024 / applied_gain` microvolts/count, using
uniform applied gain indices. For AP gain 500 it is 2.34375 microvolts/count.
Per-channel calibration has not been applied. The output scale is carried to
sorting and postprocessing rather than using the Intan default.

With `make_lfp=true`, the saved LF samples are exported at their recorded rate;
AP is not refiltered to synthesize LF. LF retains its own nominal count scale
and first-saved-row time origin, recorded in `<basename>.lfp_metadata.json` and
the session metadata. AP/LF alignment and the ADC DC zero point remain
unverified. Set `state_score=false`, `analog_inputs=false`, and
`digital_inputs=false`: state scoring needs a verified common time origin, and
the mixed WILDX sensor stream needs a separate decoder.

To make a local prefix copy without touching the server recording:

```powershell
python -m src.preprocess.wildx_np --source 'R:/data/segment_0001' --destination 'D:/np_validation/input/day_10min/segment_0001' --seconds 600
```

This copies whole AP/LF frames from sample zero, preserves receipts, saves the
original stream metadata as `neuropixels_probes.source.txt`, and updates the
excerpt's merged row counts. `neuropixels_excerpt.json` records each copied
stream's source, interval, byte count, and SHA-256. Unpaired tails and sensor
streams are not included. The destination must not already exist.

Use the copy's parent session as `basepath`, a local output directory as
`local_root`, and the existing `--mode preprocess` command. Set
`preprocess.run_sorter=true`, `preprocess.sorter="terasort"`, its dedicated
checkout/environment as `preprocess.sorter_path`, and
`sorter/TeraSort_config.yaml` as the sorter config. Leave XML/chanMap paths
empty for automatic generation. Keep `overwrite=false` to reuse compatible
outputs, including a one-subepoch MergePoints file.

For the GUI, keep the selected day folder as Basepath; NP discovery descends
through `NP_SESSION_*/segment_*`. Leave XML/chanMap and manual probe assignments empty;
the NP preview uses applied receipts and marks inactive binary slots. Run with
**Preprocess only** and **Run sorter** enabled. A persistent GUI Run uses the
checkout containing the launched GUI, unless `PREPROCESS_PIPELINE_ROOT` is
explicitly set. Restart an already open GUI after updating its code.

A persistent Run cannot adopt older foreground outputs without its compatible
output contract. Use a fresh `local_root` and execution workspace for that Run;
keep `overwrite=false` to preserve the older outputs and Phy curation.
