# Sorter Stage Reliability, Probe-Scoped Sorting, and Multi-Day Sessions

## Goal and Motivation

Plan the next implementation pass for three related workflow improvements:

1. robust handoff from MATLAB/Kilosort completion back to Python;
2. probe- or group-scoped sorting;
3. multi-day recording support.

These are related because they all affect how sorting jobs are partitioned,
tracked, finalized, and handed to postprocess. The implementation should avoid
making `ss.run_sorter()` the only source of truth for sorter completion.

## Current Problem

The observed failure mode on the server is that Kilosort appears to finish and
write Phy-compatible outputs, but Python does not continue past
`ss.run_sorter()`. In one inspected run:

- `spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `params.py`, and
  other Kilosort output files existed under `Kilosort_.../sorter_output`;
- `temp_wh.dat` remained, showing sorter cleanup did not run;
- `sorter_output` was not flattened into the run folder;
- sorter config snapshots were not saved;
- MATLAB itself was no longer running;
- `run_kilosort.sh` was left as a defunct child of the long-lived GUI process.

This indicates that the sorter completed enough to produce outputs, but the
Python process did not regain clean control. This problem is localized to the
sorter handoff around `ss.run_sorter()`, not the rest of preprocessing or
postprocessing.

Probe-scoped sorting is also needed because some recordings contain multiple
probes or group sets that should be sorted independently. The current GUI and
pipeline generate a combined `chanMap.mat` and pass one recording to one sorter
run.

Multi-day recordings are not yet represented as first-class sessions. Current
logic assumes a single basepath/basename and one local output directory per
session.

## Why This Is Needed Now

The standalone GUI is becoming the canonical front end. Long server-side
Kilosort jobs must be recoverable without closing the GUI, manually flattening
outputs, or re-running completed sorting. Probe-scoped and multi-day support
should be designed before adding more GUI controls so the implementation does
not accumulate incompatible session assumptions.

## Git State at Planning Time

- Branch: `main`
- Existing uncommitted source change: `src/preprocess/gui/app.py`
  - This is the separate chanMap auto-load and probe-color preview change.
- This plan is implementation-only documentation; no source changes are part of
  this plan file.

## 2026-06-24 Continuation: Multi-Kilosort and Multi-Day Recording

### Current Problem

The earlier plan left probe/group sorting and multi-day recording at the design
level. The next implementation pass needs concrete backend and GUI surfaces for:

- choosing whether Kilosort runs on all channels, each probe, or each shank;
- preserving original full-binary channel identities for Phy and CellExplorer;
- opening one selected Kilosort folder in Phy when multiple partitioned folders
  exist;
- running CellExplorer over multiple Kilosort folders using the Neurocode-style
  multi-Kilosort merge path;
- selecting multiple session folders for a future multi-day recording run.

The current CellExplorer wrapper accepts a single `sortingFolder` and configures
`session.spikeSorting{1}` only. That is correct for one sorter result but cannot
represent partitioned Kilosort outputs without relying on ambiguous
`dir('Kilosort*')` discovery.

### Updated Public Parameters

Use GUI wording that describes the action directly:

```python
sorter_partition_mode: Literal["all", "probe", "shank"] = "all"
```

The sorter CLI should also expose explicit 0-based channel lists for targeted
or recovered sorter runs:

```text
--active-channels 0,2,3
--exclude-channels 1,7
```

These CLI options are lower-level than `sorter_partition_mode`: they do not
build probe/shank partitions by themselves, but they let tests and manual
recovery commands exercise the same explicit channel subset path used by the
partitioned pipeline.

Notebook/API-facing text should expose this as `Run sorter on` with options:

- `All channels`;
- `Each probe`;
- `Each shank`.

The implementation should keep artifact grouping separate from sorter
partitioning. Existing artifact fields such as `artifact_group_probe` and
`artifact_group_shank` remain artifact-specific metadata; sorter partitioning
should build an explicit manifest from `chanMap.mat`.

### Updated Partition Design

Use `chanMap.mat` as the source of truth for partitions:

- `chanMap0ind` or `chanMap - 1` gives original 0-based device channels;
- `probe_ids` gives probe partitions;
- `kcoords` gives shank IDs;
- shank partitions are keyed by `(probe_id, shank_id)`, not `shank_id` alone;
- `connected == false`, GUI rejected channels, and XML skipped channels are
  excluded from sorter inputs but still recorded in the manifest.

Each partitioned sorter folder is independent for Phy compatibility:

```text
Kilosort_spi_probe1/
Kilosort_spi_probe2/
Kilosort_spi_probe1_shank1/
```

Write a session-level `sorter_partition_manifest.json` that records each
partition folder, mode, probe/shank identity, full-binary channels, excluded
channels, and status. This manifest is the GUI and CellExplorer source of
truth; do not discover partition folders by broad globbing when the manifest is
available.

Sorter output patching must preserve full-binary channel identity:

- `params.py` `dat_path` points at the raw/preprocessed full `.dat`;
- `params.py` `n_channels_dat` remains the full `.dat` channel count;
- `channel_map.npy` maps partition-local indices back to original 0-based
  full-binary channel IDs;
- `cluster_info.tsv` `ch` and related Phy metadata should be full-binary
  channel IDs when present.

### CellExplorer Multi-Kilosort Design

Mirror Neurocode's multi-Kilosort behavior rather than depending on
CellExplorer to process multiple `session.spikeSorting{}` entries:

1. accept an explicit list of selected sorting folders;
2. configure `session.spikeSorting` with entries for provenance, while keeping
   `session.spikeSorting{1}` valid for CellExplorer functions that assume it;
3. call `loadSpikes('session', session, 'clusteringpath', folder, ...)` once per
   selected folder;
4. merge the resulting `spikes` structs by offsetting `UID` and the unit IDs in
   `spindices`;
5. save the merged `spikes` file;
6. call `ProcessCellMetrics` once on the merged `spikes`.

Phy remains a single-folder action. The GUI should let the user choose exactly
which Kilosort folder to open in Phy. CellExplorer can use one or more selected
folders; when more than one is selected, the wrapper takes the multi-Kilosort
merge path.

### Updated Multi-Day Design

The first GUI surface should be `Browse for multi-days`, selecting multiple
session folders. Create local and server multi-day basepaths. Treat each
selected session as a subepoch, and if a selected session already contains
subepochs, concatenate those subepochs in date order.

Use a shared analysis epoch when practical, but keep provenance in a manifest:

- selected session folder;
- session/day label and ordering key;
- subepoch folder;
- source `.dat`;
- output offset and sample span;
- generated local/server output paths.

`MergePoints` rows should remain at the practical input-chunk level so later
debugging can trace a time range back to a source session/subepoch. The common
epoch label can be analysis-facing metadata, while the manifest stores the
full session/day/subepoch ownership.

### 2026-06-24 Multi-Day First Implementation

Implement multi-day as a staging layer in front of the existing single-session
pipeline:

1. GUI `Browse for multi-days` selects multiple session folders.
2. A server-side multi-day basepath is created under the common parent of the
   selected sessions, using a stable name such as `multiday_<first>_to_<last>`.
3. The local multi-day basepath is the existing local root plus the same
   multi-day basename.
4. The server-side multi-day basepath contains symlinks to each source
   session/subepoch directory, ordered by the existing subsession sort key and
   prefixed with session order so names remain unique.
5. The first session's XML and RHD metadata are copied into the multi-day
   basepath under the multi-day basename. The first pass requires compatible
   channel count and sampling metadata across sessions.
6. A `multi_day_manifest.json` records selected sessions, source subepochs,
   staged subepoch folders, and ordering/provenance.
7. The regular `run_preprocess_session` then runs on the staged multi-day
   basepath. `MergePoints` rows naturally stay at the staged subepoch level.

This first pass avoids copying large binary files. It is intended for standard
Intan-style subepoch folders and Open Ephys datetime folders that can be
represented as direct children of the staged multi-day basepath.

### 2026-06-24 XML Selection Update

Remove implicit parent-directory XML discovery. XML resolution should be
explicit and predictable:

1. If a GUI/user-selected XML path is provided, copy that XML to the local
   output as `<basename>.xml` and use it.
2. Otherwise, if `basepath/<basename>.xml` exists, use it automatically.
3. Otherwise, leave the GUI XML field empty and raise an error when the user
   runs preprocessing.

### 2026-06-24 CellExplorer X11 Visibility Update

When running from Windows through SSH/X11 forwarding, the CellExplorer manual
curation GUI can appear only as a MATLAB taskbar icon even though earlier
ProcessCellMetrics figures are visible. CellExplorer creates its main figure as
hidden and then shows it by setting `WindowState` to `maximize`; that maximize
step is less reliable under forwarded X11 window managers than ordinary MATLAB
figures.

The local vendored CellExplorer launch should keep the existing behavior on
native desktop environments but, on Linux/X11, show the main CellExplorer figure
as a normal on-screen window with an explicit size and bring it to the front.
This is a GUI visibility workaround only and must not change CellExplorer
metrics generation or sorter folder selection.

The GUI wrapper should request waveform extraction from dat for the normal
CellExplorer curation workflow. This is slower than using Kilosort templates,
but it matches the expected manual-curation waveform source and avoids relying
on template array ordering when raw data are available.

Waveform extraction should happen once in the wrapper's explicit `loadSpikes`
step. After the merged `spikes` struct is passed into `ProcessCellMetrics`, the
metrics step should not call `loadSpikes` or `getWaveformsFromDat` again.
MonoSynaptic connection metrics should be recomputed for the explicitly selected
single- or multi-sorter folder set only when no compatible mono_res file exists.
Because CellExplorer loads an existing mono_res file before recomputing it, the
wrapper should keep a compatible mono_res file so it is loaded directly. To
avoid stale cell-pair indices from a previous sorter selection, the wrapper
should validate the existing mono_res against the current merged cell count and
remove it only when its connection indices or CCG dimensions are incompatible.

Before launching the CellExplorer GUI, the wrapper should also ensure GUI
classification fields that CellExplorer assumes are present. In particular,
`deepSuperficial_num` may be absent when deep-superficial ripple classification
is skipped, even though CellExplorer's `updateUI` indexes it directly. The
wrapper should preserve the metric as `Unknown` and write
`deepSuperficial_num = 1` for all cells when no explicit classification exists.

Do not add wrapper-side or vendored CellExplorer logic that rewrites or
implicitly remaps Kilosort/Phy cluster ids or template indices. Preserving the
original Phy identifiers is more important than supporting the unused template
waveform fallback path. If template waveform loading is needed later, it should
be designed explicitly and tested against Phy/Kilosort outputs.

### 2026-06-24 GUI Result Serialization Fix

Multi-day probe-partitioned sorting can complete the preprocess stage and save
the sorter partition manifest, then fail while the GUI helper prints the final
JSON result. The run payload may still contain `pathlib.Path` values from
downstream return objects, and `json.dumps()` cannot serialize those objects.

The GUI runner should sanitize its final result payload recursively before
printing it with the `__PREPROCESS_GUI_RESULT__` prefix. The preprocess
`save_params_and_manifest()` path should also serialize newly introduced
`Path`-typed fields such as explicit `xml_path`, multi-sorter output folders,
and the sorter partition manifest path. These fixes should not change sorting
outputs or postprocess behavior.

### 2026-06-24 Multi-Sorter Postprocess Target Resolution

Manual curation needs a single selected sorting folder for Phy, but multi-sorter
postprocess should process every partition generated by a multi-sorter run. If
the postprocess search root contains `sorter_partition_manifest.json`, target
resolution should prefer the manifest partition `output_folder` list. This
prevents the Manual Curation selected folder, for example `probe3`, from
causing `Run postprocess` to create only `probe3_spi`.

The Manual Curation folder is for Phy only and should not update the
Postprocess target. `Run all` and `Run postprocess` should use the multi-sorter
manifest when present, including cases where sorting is skipped because
existing Kilosort folders are being reused. Single-folder explicit postprocess
remains valid when no matching multi-sorter manifest is available.

CellExplore folder selection should be explicit. The GUI should show
CellExplore candidate folders in a table with checkboxes, seeded from the
multi-sorter manifest when available. All newly discovered candidates should be
unchecked by default; `Run CellExplore` should error until the user registers at
least one checked folder. One registered folder runs the single-sorter path,
while two or more registered folders run the wrapper's multi-sorter merge path.
The selected Phy folder is not an implicit CellExplore input.
Raw Kilosort folders and their corresponding `_spi` postprocess folders may
both be displayed, but they should share one selection group so only one of the
pair can be checked at a time.

For multi-day staging, an explicitly loaded XML should update the staged
`<multi_day_name>.xml` even if one already exists. If no explicit XML is
provided, an existing staged XML is used; otherwise the first selected session's
basename XML is copied as the first-pass default. Parent-directory XML files and
arbitrary one-off XML files inside a folder are not implicit candidates. When a
loaded or staged XML is available, it is treated as the authoritative XML; when
none is available, each selected session must have its own basename XML so the
staging step can validate channel count and sampling rate compatibility.

### 2026-06-24 Open Ephys Multi-Day Sample Counts

Align multi-day staging with the existing single-session Open Ephys path. The
single-session pipeline resolves Open Ephys frame size from `structure.oebin`,
not from the XML `nChannels`, because `continuous.dat` can include non-sorting
channels such as ADC channels. Multi-day staging should use the same rule when
it computes per-subepoch sample counts for the manifest:

1. If a discovered subepoch is an Open Ephys recording root, read
   `structure.oebin` with the existing stream resolver and use that stream's
   `num_channels` and `sample_rate` for binary frame-size checks.
2. Keep XML `nChannels` as the sorting/chanMap metadata, not as the Open Ephys
   raw binary frame size.
3. Match single-session behavior by rejecting Open Ephys subepochs whose stream
   channel count or sampling frequency differ within the staged multi-day run.

### 2026-06-24 Multi-Day GUI Usability Update

Improve GUI feedback for XML and multi-day session ordering:

1. Move `Load XML` into the top of the `Session and channel map` section so XML
   selection is visually tied to channel-map generation and preview.
2. Do not print missing-XML channel-map preview failures to the terminal from
   GUI refresh paths. If the XML field is empty, passive preview should stay
   quiet. If the user entered a non-existent XML path, the GUI should log a
   warning such as "set XML first" once per missing XML state instead of
   showing an `Error:` line on stdout. Run preflight is stricter: explicit
   loaded XML paths satisfy the XML check, while no resolved XML is an error
   before preprocessing starts.
3. Keep selected multi-day session folders as an internal ordered list and show
   only a concise summary in the top bar.
4. Add a session-order viewer dialog with a table containing order, session
   name, and full path. The table should support direct order-number editing and
   an `Update order` action that writes the edited order back into the run
   settings. Drag/drop row movement is intentionally disabled because the Qt
   item move behavior can remove row contents when combined with immediate
   redraws.
5. Keep the session-order dialog readable under the dark GUI theme, make the
   session-name column wide enough to scan, and allow direct editing of the
   order number. Editing an order number should immediately reorder and redraw
   the table so the displayed order always matches the pending order.
6. Skip Intan `info.rhd` preflight checks when the selected basepath is detected
   as Open Ephys, because Open Ephys sessions do not need Intan header metadata.
7. Force stop should terminate the whole preprocessing process tree, including
   SpikeInterface/joblib workers used by binary writing and artifact detection.
   The GUI should re-enable run and manual-curation controls immediately after
   the stop request while still escalating termination in the background.
8. Manual curation should expose the active sorting folder next to `Launch Phy`
   so multi-Kilosort runs can launch each sorter folder explicitly. The field
   should stay synchronized with the existing Postprocess target sorting folder.
   `Run CellExplore postprocess` can sit below and use the same folder field;
   leaving it blank keeps the existing multi-folder CellExplorer behavior.

## Affected Modules and Files

Expected source modules:

- `src/preprocess/sorter_runner.py`
- `src/preprocess/pipeline.py`
- `src/preprocess/metafile.py`
- `src/preprocess/gui/config_model.py`
- `src/preprocess/gui/run_pipeline.py`
- `src/preprocess/gui/app.py`
- `src/postprocess/pipeline.py`
- `src/postprocess/metafile.py`

Expected tests:

- `tests/preprocess/test_sorter_runner_matlab_path.py`
- `tests/preprocess/test_gui_config_model.py`
- `tests/preprocess/test_gui_preflight.py`
- `tests/postprocess/test_postprocess_target_resolution.py`
- new focused tests for sorter completion detection and finalize recovery
- new focused tests for probe/group sorting target expansion
- new focused tests for multi-day session metadata resolution

## Phase 1: MATLAB/Kilosort Handoff Recovery

### Design

Treat `ss.run_sorter()` as an unreliable external boundary. The pipeline should
not require `ss.run_sorter()` to return normally if the sorter output is already
complete and stable.

Add a dedicated sorter stage wrapper:

1. run `ss.run_sorter()` in a short-lived sorter subprocess;
2. have the parent process monitor the sorter output folder;
3. detect completed sorter outputs from files, not only process return code;
4. if outputs are complete and stable but the sorter subprocess is stuck, kill
   the sorter subprocess/process group;
5. run sorter finalization as a separate idempotent step;
6. continue to postprocess when finalization succeeds.

### Completion Criteria

For Kilosort/Phy output, completed output should require at least:

- `spike_times.npy`;
- `spike_clusters.npy`;
- `spike_templates.npy`;
- `templates.npy`;
- `channel_map.npy`;
- `channel_positions.npy`;
- `params.py`.

Additional validation:

- output files are readable;
- `spike_times.npy` and `spike_clusters.npy` have matching length;
- file size and modification time are stable for a configurable interval;
- no live MATLAB process remains for that run folder;
- remaining sorter shell wrapper processes are gone or defunct only.

### Finalization Step

Create an idempotent finalize function for sorter output. It should be safe to
run after normal sorter completion, after recovered completion, and manually
from the GUI.

Responsibilities:

- flatten `Kilosort_.../sorter_output` into `Kilosort_...` when needed;
- patch `params.py` to point to the correct raw/preprocessed `.dat`;
- patch `n_channels_dat` and `hp_filtered`;
- patch `channel_map.npy` when sorting used active-channel subsets;
- save source and resolved sorter config snapshots;
- persist MATLAB logs into the sorter folder;
- remove `temp_wh.dat` when cleanup is enabled.

### GUI Behavior

The GUI should show one of these sorter states:

- running;
- completed;
- finalize required;
- finalized;
- failed.

If completion is detected but finalization has not run, the GUI can either run
finalization automatically or expose a `Finalize sorter output` action. The
default should be automatic during `Run all`, because the user expects
postprocess to continue.

### Non-Goals

- Do not reimplement Kilosort.
- Do not depend on MATLAB log text alone for completion detection.
- Do not rely on zombie process cleanup as the main completion signal.

## Phase 2: Probe- or Group-Scoped Sorting

### Design

Add a sorter partitioning layer before `execute_sorting_job()`. The default
remains current behavior: one sorter run for the full active recording. New
options allow independent sorter runs by:

- probe;
- shank/group;
- explicit group sets.

The partitioning should use `chanMap.mat` metadata:

- `probe_ids` for probe-level partitions;
- `kcoords` for group/shank-level partitions;
- `chanMap0ind` for device channel mapping;
- `connected` plus GUI bad channels for exclusion.

### Public Parameters

Potential config fields:

```python
sorter_partition_mode: Literal["all", "probe", "group", "custom"] = "all"
sorter_partitions: list[dict] = []
```

For `custom`, each entry can contain:

```json
{
  "name": "probe1_front",
  "groups": [0, 1, 2, 3],
  "probe_ids": [1]
}
```

### Output Layout

For one full run:

```text
<local>/<basename>/Kilosort_YYYY-MM-DD_HHMMSS/
```

For partitioned runs:

```text
<local>/<basename>/Kilosort_YYYY-MM-DD_HHMMSS/
  partition_manifest.json
  probe_1/
  probe_2/
```

or, if simpler for compatibility:

```text
<local>/<basename>/Kilosort_probe1_YYYY-MM-DD_HHMMSS/
<local>/<basename>/Kilosort_probe2_YYYY-MM-DD_HHMMSS/
```

The second layout is easier for existing postprocess discovery because each
partition is an independent Kilosort result. A manifest can link the runs back
to a shared parent session.

### Expected Behavior

- `all`: current behavior.
- `probe`: create one sorter run per `probe_id`.
- `group`: create one sorter run per `kcoord` group.
- `custom`: create one sorter run per configured partition.

Each sorter run receives a recording view with only that partition's channels
and a matching partition-specific chanMap. Postprocess can run on each result
independently.

### Non-Goals

- Do not merge clusters across probes/groups automatically in the first pass.
- Do not change Kilosort parameters per partition unless explicitly configured.
- Do not make probe-scoped sorting the default.

## Phase 3: Multi-Day Recording Support

### Design

Represent a multi-day session as a collection of single-day/session entries
plus a shared output root and manifest. Avoid hiding multiple physical
recordings behind one ambiguous basepath.

Potential config model:

```python
multi_day_enabled: bool = False
session_entries: list[SessionEntry] = []
```

Each `SessionEntry` should include:

```python
basepath: Path
basename: str | None
local_output_dir: Path | None
day_label: str | None
```

### Output Layout

Prefer a manifest-based layout:

```text
<local>/<multi_day_name>/
  multi_day_manifest.json
  day01_<basename>/
  day02_<basename>/
  combined_postprocess/
```

Each day keeps its own generated files, chanMap, preprocessing outputs, sorter
outputs, logs, and postprocess outputs. The combined layer can be added later
after single-day subruns are reliable.

### Expected Behavior

Initial implementation should support:

- selecting multiple basepaths;
- running preprocess for each day;
- running sorter for each day, optionally with probe/group partitioning;
- running postprocess for each day;
- recording all outputs in a manifest.

Later implementation can add:

- combined reporting across days;
- shared unit curation helpers;
- optional cross-day alignment workflows.

### Non-Goals

- Do not assume continuous timestamps across days.
- Do not merge spike clusters across days in the first implementation.
- Do not require all days to share identical sorter output names.

## Implementation Order

1. Extract and test idempotent sorter finalization.
2. Add completed-output detection for Kilosort/Phy folders.
3. Run `ss.run_sorter()` in a dedicated sorter subprocess.
4. Add recovery path: detected completion -> kill stuck sorter subprocess ->
   finalize -> continue.
5. Add postprocess compatibility for recovered or unflattened sorter output.
6. Add partition model for probe/group sorting.
7. Add GUI controls for sorting mode only after backend behavior is tested.
8. Add multi-day session manifest model.
9. Add GUI multi-day selection only after backend manifest execution is tested.

## Verification Plan

Sorter handoff tests:

- simulate `ss.run_sorter()` normal return and verify finalization runs once;
- simulate complete output plus stuck sorter subprocess and verify recovery;
- simulate incomplete output and verify the run remains failed/stopped;
- verify no long-lived child process is attached to the GUI after force stop.

Sorter finalization tests:

- finalize a nested `sorter_output` layout;
- finalize an already flattened layout;
- verify `params.py` patching;
- verify `temp_wh.dat` cleanup;
- verify config snapshot writing.

Probe/group sorting tests:

- build partitions from a chanMap with multiple probes;
- build partitions from `kcoords`;
- verify partition-specific chanMaps contain only selected device channels;
- verify output folder names and manifest entries.

Multi-day tests:

- resolve multiple basepaths into session entries;
- run a dry-run manifest over multiple sessions;
- verify per-day output paths;
- verify failures are isolated to one day unless configured otherwise.

GUI checks:

- start GUI in offscreen mode;
- confirm new controls round-trip through `PipelineGuiSettings`;
- confirm default mode remains single-session, full-recording sorting.

## Known Risks

- Kilosort may write some final files before all internal work is fully done.
  Completion detection must require stable files and readable arrays, not just
  file existence.
- Windows and Linux process-group behavior differ. The sorter subprocess wrapper
  must have separate process-tree termination paths.
- Probe/group partitioning changes channel indices. `params.py`,
  `channel_map.npy`, and postprocess recording reconstruction must remain
  consistent with the original `.dat`.
- Multi-day support can expand scope quickly. The first version should focus on
  orchestrating multiple independent day runs, not cross-day unit matching.

## Open Questions

- Should recovered sorter finalization be fully automatic, or should the GUI
  show a confirmation before continuing to postprocess?
- For probe-scoped sorting, should the default output layout be independent
  Kilosort folders or nested partition folders under one parent run?
- For multi-day sessions, should the GUI allow mixed sorter settings per day,
  or enforce one shared config across all selected days?
