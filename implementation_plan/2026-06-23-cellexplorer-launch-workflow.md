# CellExplorer Launch Workflow

## Goal and motivation

Add a GUI workflow that launches the post-Phy CellExplorer processing path from
PreprocessPipeline. The intended path is:

1. launch Phy for manual curation;
2. launch CellExplorer;
3. open or update the session metadata with `gui_session`;
4. run the equivalent of neurocode `PreprocessSpikes`;
5. open CellExplorer on the generated `cell_metrics`.

This keeps the curation-to-cell-metrics workflow inside the preprocessing GUI
and avoids dependence on whichever CellExplorer version happens to be installed
on the user's MATLAB path.

## Current problem

The GUI currently has a `Run phy` button but no CellExplorer entrypoint. The
neurocode workflow depends on `gui_session` and `PreprocessSpikes`, while the
repository does not yet provide a controlled MATLAB launcher for those steps.

There are also three path risks:

- neurocode `PreprocessSpikes` defaults `prePhy` to `false`, which is correct
  after Phy/manual curation, but the meaning is easy to confuse;
- `PreprocessSpikes` discovers the first `Kilosort*` folder if
  `session.spikeSorting{1}.relativePath` is missing, which can pick the wrong
  folder when both raw Kilosort and SpikeInterface postprocess folders exist;
- ayalab1 CellExplorer's `getWaveformsFromDat` uses `basename.dat` when it
  exists and only falls back to MergePoints epoch files when that single dat is
  missing. The requested behavior is stricter: use the raw sub-epoch dat files
  under the source/basepath even if `basename.dat` exists in the local output.

## Why this is needed now

Manual curation is already exposed in the GUI. The next practical step is to
produce CellExplorer-compatible spikes and cell metrics from the curated Phy
output without manually editing MATLAB paths, session files, or sorting folder
fields.

## Relevant state

- Git status before planning showed an existing uncommitted change in
  `src/preprocess/gui/app.py`; implementation must preserve that work.
- The inspected ayalab1 CellExplorer checkout is at commit
  `b7e9314bcb3b909fcfaf03e7d614a32cf1cacbba`.
- When refreshing the vendored copy, import the latest
  `ayalab1/CellExplorer` working tree into `external/CellExplorer` rather than
  running `git pull` in place, because the vendored directory intentionally does
  not contain its upstream `.git` metadata.
- The inspected pyNeuroscope checkout is at commit
  `f853a177ee7b081ffee2107dac5750b2fc4855d3`.
- ayalab1 CellExplorer includes MergePoints fallback coverage for
  `amplifier.dat` and Open Ephys `continuous.dat`, but its current fallback is
  activated only when the resolved single dat file is absent.

## Affected modules/files

- `src/preprocess/gui/app.py`
- new or vendored MATLAB support under a repository-owned path such as
  `external/CellExplorer` or `third_party/CellExplorer`
- new MATLAB launcher/wrapper under repository-owned support scripts, if needed
- new anatomical map helper/editor modules if the pyNeuroscope logic is ported
  instead of imported directly
- `implementation_plan/README.md`
- `change_log/README.md`
- `change_log/2026-06-23-cellexplorer-launch-workflow.md`

## Public parameters or API changes

- Rename the manual curation button label from `Run phy` to `Launch Phy`.
- Add a `Run CellExplore postprocess` button below or next to `Launch Phy` in
  the `Manual Curation` section.
- Add anatomical map controls to the channel-map mini window:
  - `Load anatomical map`
  - `Edit anatomical map`
- The editor should support saving `anatomical_map.csv` to the local session
  output folder.
- No command-line API change is required for the first implementation, but the
  internal CellExplorer launcher should accept explicit basepath, sorting path,
  CellExplorer root, and raw waveform source policy.

## Expected behavior

### Phy and sorting folder selection

- `Launch Phy` keeps the existing Phy resolution behavior but changes only the
  user-facing label.
- `Run CellExplore postprocess` should resolve the curated Phy folder with the
  same target used by Phy launch.
- The default target should be the SpikeInterface/Phy postprocess folder whose
  name ends in `_spi`, for example `Kilosort_..._spi`.
- The launcher should explicitly set
  `session.spikeSorting{1}.relativePath` to the selected `_spi` folder before
  running CellExplorer processing so MATLAB does not guess from `dir('Kilosort*')`.

### `prePhy`

- Default CellExplorer processing should use `prePhy=false`.
- `prePhy=false` means the workflow assumes Phy/manual curation has already
  been performed and should process curated labels such as `good`.
- `prePhy=true` is a separate pre-curation/unsorted preview mode. It writes or
  uses unsorted-style outputs and should not be the default for the manual
  curation workflow.

### Raw dat selection

- The requested default is to use raw sub-epoch data under the source/basepath,
  not the merged local `basename.dat`.
- Because ayalab1 CellExplorer still prefers `basename.dat` when present, the
  implementation should add an explicit launcher policy. Acceptable approaches:
  - pass or synthesize a session/raw source configuration that forces
    MergePoints epoch-file extraction;
  - patch the vendored CellExplorer wrapper with an opt-in parameter such as
    `preferMergePointsDat`;
  - temporarily hide only the launcher-visible single dat path from the
    CellExplorer call without deleting or modifying the user's data.
- The implementation must not delete or overwrite `basename.dat`.
- The CellExplorer output should record the waveform source, ideally through
  `spikes.processinginfo.params.WaveformsSource`, so the user can confirm
  whether waveforms came from `MergePoints amplifier.dat files`,
  `MergePoints Open Ephys continuous.dat files`, or another explicit source.

### Vendored CellExplorer

- Vendor the ayalab1 CellExplorer version inside this repository at a pinned
  commit instead of relying on the user's global MATLAB path.
- The MATLAB launcher should add the vendored CellExplorer path before running
  `gui_session`, `loadSpikes`, `ProcessCellMetrics`, or `CellExplorer`.
- If neurocode `PreprocessSpikes.m` is not vendored, implement the equivalent
  directly in a small repository-owned MATLAB wrapper:
  - load or create `basename.session.mat`;
  - run `gui_session` and save the returned session after accepted edits;
  - load curated spikes from the selected Phy folder with label `good`;
  - run `ProcessCellMetrics` with `manualAdjustMonoSyn=false`;
  - pass already-loaded spikes to `ProcessCellMetrics` without requesting a
    second waveform extraction, because CellExplorer otherwise reloads spikes
    when `WaveformsSource` is not exactly `dat file`;
  - ensure required CellExplorer MEX helpers (`CCGHeart` and
    `FindInInterval`) are available, compiling the vendored C sources on Linux
    if platform-specific MEX binaries are missing;
  - save `basename.cell_metrics.cellinfo.mat` and updated session metadata;
  - launch CellExplorer from the saved basepath metrics file after verifying
    that non-empty `cell_metrics` were generated, matching the normal manual
    curation entrypoint.
  - exit MATLAB with status 0 after the CellExplorer GUI closes so the parent
    Qt GUI can re-enable Phy and CellExplorer controls.

### Anatomical map

- Add pyNeuroscope-style anatomical region editing to the channel-map mini
  window.
- Keep anatomical map internals CellExplorer-compatible and therefore
  1-based. The GUI preview and bad-channel controls use 0-based device channel
  indices, so the anatomical editor should display 0-based channel labels while
  saving/loading the 1-based CellExplorer channel identity internally.
- Render the anatomical editor from the actual `chanMap.mat` geometry
  (`xcoords`/`ycoords`) instead of a synthetic group-by-row grid. The saved CSV
  remains pyNeuroscope-compatible, but region assignment should happen on the
  same physical contact layout shown by the chanMap preview.
- Support mouse-wheel zoom in the anatomical editor so dense channel labels can
  be inspected without changing the saved channel/region mapping.
- Use the same CSV layout as pyNeuroscope:
  - no header;
  - rows are depth/order within each channel group;
  - columns are channel groups;
  - each cell contains the anatomical label for that group row.
- `Load anatomical map` should read an existing `anatomical_map.csv` and map
  labels back onto channels using the current channel groups.
- `Edit anatomical map` should open a small channel-region editor that allows
  assigning and clearing labels, then saving `anatomical_map.csv` locally.
- CellExplorer processing should read the local `anatomical_map.csv` before or
  during channel mapping so the generated session/cell metrics carry anatomical
  labels.

## Future multi-Kilosort plan

The first implementation should remain single-target: one curated `_spi` folder
feeds one CellExplorer run.

Future work should support multiple Kilosort outputs per session, for example
group-, region-, or shank-scoped sorting. That design should:

- store multiple explicit `session.spikeSorting{}` entries rather than relying
  on folder-name discovery;
- let the GUI choose or display the sorting target for each channel group;
- preserve unique unit IDs when combining outputs;
- associate each sorting group with channel groups and anatomical regions;
- define how repeated or overlapping channels are handled;
- make CellExplorer output provenance clear for each unit.

## Tests and checks

- Compile-check changed Python GUI modules.
- Add unit tests for anatomical map CSV round-tripping against channel groups.
- Add a MATLAB path smoke check that `which CellExplorer`, `which gui_session`,
  and `which ProcessCellMetrics` resolve to the vendored CellExplorer path.
- Add or document a MATLAB non-interactive smoke path for the post-gui portion
  where possible.
- Manually validate the interactive workflow:
  - `Launch Phy` opens the selected `_spi/params.py`;
  - `Run CellExplore postprocess` opens `gui_session`;
  - accepting `gui_session` saves session metadata;
  - CellExplorer processing uses `_spi`;
  - waveform source reports MergePoints epoch files when that policy is active.

## Non-goals

- Do not implement multi-Kilosort merging in the first pass.
- Do not change sorter execution behavior.
- Do not delete, rewrite, or move raw dat files.
- Do not require the user's globally installed CellExplorer to be modified.
