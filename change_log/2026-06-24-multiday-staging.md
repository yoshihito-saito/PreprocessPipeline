# Multi-Day Staging

## Date and Git State

- Date: 2026-06-24
- Branch: `feature/multi-kilosort-and-multi-day-recording`
- Commit: uncommitted
- Plan: [2026-06-17 sorter stage reliability, probe-scoped sorting, and multi-day sessions](../implementation_plan/2026-06-17-sorter-stage-and-multiday.md)

## What Changed

- Added `src/preprocess/multiday.py` for preparing a multi-day staging basepath.
- Added GUI settings for multi-day runs:
  - selected session folder list;
  - multi-day name;
  - `Browse for multi-days` button.
- Added GUI XML selection:
  - `Load XML` button;
  - automatic XML field population only from `basepath/<basename>.xml`;
  - no implicit parent-directory XML fallback;
  - run-time error when no XML is selected and no basename XML exists.
- Moved `Load XML` into the `Session and channel map` section and changed GUI
  channel-map preview/generation to avoid printing missing XML messages to the
  terminal. Empty XML fields stay quiet during passive preview; non-existent
  saved XML paths are cleared when settings are applied.
- Updated run preflight so explicit loaded XML paths satisfy the `Session XML`
  check, missing XML is an error before running, and `Intan info.rhd` checks are
  skipped for Open Ephys sessions.
- Strengthened GUI Force stop:
  - sends termination to the preprocessing process group and discovered child
    worker PIDs;
  - escalates to SIGKILL after a short delay so SpikeInterface/joblib workers
    used by binary writing or artifact detection do not keep running;
  - re-enables GUI controls immediately after the stop request;
  - allows closing the GUI after Force stop has been requested.
- Added a Manual Curation sorting-folder field beside `Launch Phy`, with a
  `Browse folder` button. It stays synchronized with the existing Postprocess
  target sorting folder, so each multi-Kilosort output can be opened in Phy by
  selecting that folder. `Run CellExplore postprocess` now sits below and uses
  the same selected folder; leaving it blank keeps the multi-folder
  CellExplorer behavior.
- Made GUI run result reporting JSON-safe. Multi-day probe-partitioned sorting
  can return `pathlib.Path` objects through downstream result payloads after
  the sorter manifest has already been saved; the GUI runner now recursively
  converts paths and other non-JSON primitives before printing the final result
  marker.
- Made `preprocessSession_params.json` and `preprocessSession_manifest.json`
  saving robust to new `Path` fields introduced by explicit XML selection and
  multi-sorter partitioning. Explicit `xml_path`, `sorter_output_dirs`, and
  `sorter_partition_manifest_path` are now saved as strings.
- Shortened the Manual Curation button to `Run CellExplore` and moved the
  multi-sorter folder hint to a wrapped line so it stays readable in the left
  settings panel.
- Decoupled the Manual Curation folder from Postprocess target resolution. The
  selected manual folder is used for one-at-a-time Phy launching, while
  `Run all` and `Run postprocess` use `sorter_partition_manifest.json` to
  process every multi-sorter partition when present.
- Updated CellExplorer folder resolution so the multi-sorter manifest is the
  primary source of truth. A selected manual Phy folder is only used as a
  fallback when no manifest is available.
- Decoupled the Manual Curation folder from the Postprocess target folder.
  Selecting one folder for Phy or CellExplore no longer changes which sorter
  folders `Run postprocess` processes.
- Updated multi-sorter postprocess target resolution so `Run all` and
  `Run postprocess` use every folder listed in `sorter_partition_manifest.json`
  when present. This also covers runs where sorting is skipped but an existing
  multi-sorter manifest is available.
- Changed CellExplore folder selection to an explicit checked-list workflow.
  The GUI now shows candidate folders in a table, leaves newly discovered
  candidates unchecked by default, registers only checked folders, and errors
  on `Run CellExplore` until at least one folder is registered. One registered
  folder runs as a single sorter; two or more registered folders use the
  multi-sorter merge wrapper.
- CellExplore folder candidates now show both raw Kilosort folders and matching
  `_spi` folders when both exist, but they are mutually exclusive in the table:
  checking one automatically unchecks the matching counterpart.
- Kept CellExplorer waveform extraction from dat enabled and
  `showWaveforms=true` for the manual-curation workflow. Multi-sorter loading
  still calls `loadSpikes` once per selected folder, so multiple waveform
  progress figures can appear during processing.
- Prevented duplicate CellExplorer waveform extraction by letting the explicit
  `loadSpikes` step extract raw-dat waveforms, then passing
  `getWaveformsFromDat=false` and `forceReloadSpikes=false` to
  `ProcessCellMetrics`.
- Re-enabled MonoSynaptic connection metrics in the CellExplorer wrapper for
  both pre-Phy and post-Phy paths. Before calling `ProcessCellMetrics`, the
  wrapper now validates the exact `.mono_res...mat` file for the current
  `saveAs` target. Compatible existing files are kept and loaded directly by
  CellExplorer; incompatible files are removed so stale cell-pair indices from a
  previous sorter selection are not reused.
- Added a GUI-field normalization step before launching CellExplorer. When
  deep-superficial ripple classification is skipped, the wrapper now saves
  `deepSuperficial`, `deepSuperficialDistance`, and `deepSuperficial_num`
  fallback values so CellExplorer's `updateUI` can open without missing-field
  errors.
- Adjusted the vendored CellExplorer main GUI launch for Linux/X11 sessions.
  On forwarded X11 displays, the main manual curation window is now shown as a
  normal on-screen figure and brought to the front instead of relying on
  `WindowState='maximize'`, which can leave the window only visible as a
  taskbar icon.
- Left the vendored CellExplorer Phy template-index behavior unchanged. The
  wrapper uses `getWaveformsFromDat=true` for manual curation, so the normal
  path extracts waveforms from raw data instead of relying on Kilosort template
  indexing.
- Replaced the long editable multi-day session path field with a concise
  summary plus a compact `Check order` table dialog button. The dialog shows
  order, session name, and path on a dark background, supports direct order
  number editing,
  redraws immediately after manual order edits, and applies the edited order
  with `Update order`. Drag/drop row movement is disabled to avoid Qt item-move
  row loss.
- Added run-pipeline staging before normal preprocessing. The staged server
  basepath becomes the effective `basepath`, while the local output remains
  `local_root/<multi_day_name>`.
- Added explicit run-time errors when multi-day preprocessing is enabled but
  fewer than two session folders or no multi-day basepath name were provided.
- The staging layer:
  - creates server and local multi-day basepaths;
  - copies an explicitly loaded XML to `<multi_day_name>.xml` when provided;
  - skips XML copying when the loaded XML is already the staged
    `<multi_day_name>.xml`, avoiding `SameFileError`;
  - otherwise uses an existing `<multi_day_name>.xml` in the staged server
    basepath, or copies the first session basename XML as the default;
  - validates channel count and sampling rate compatibility across sessions
    when no loaded or staged XML is already authoritative;
  - uses Open Ephys `structure.oebin` stream metadata, not XML `nChannels`, when
    computing `continuous.dat` sample counts;
  - rejects Open Ephys subepochs with mismatched stream channel counts or
    sampling frequencies, matching the single-session preprocessing path;
  - symlinks each source session/subepoch into the server multi-day basepath;
  - writes `multi_day_manifest.json` to both server and local basepaths.
- Existing `discover_subsessions`, concatenation, and `MergePoints` generation
  then operate on the staged basepath, so each staged subepoch remains
  traceable.

## Why

Multi-day recordings need a single preprocessing entrypoint while preserving
which session/day/subepoch each input chunk came from. A symlink-based staging
layer avoids copying large raw binaries and lets the existing single-session
pipeline do the actual concatenation and downstream processing.

## Verification

- `python -m py_compile src/preprocess/io.py src/preprocess/metafile.py src/preprocess/pipeline.py src/preprocess/multiday.py src/preprocess/gui/app.py src/preprocess/gui/preflight.py src/preprocess/gui/config_model.py src/preprocess/gui/run_pipeline.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_io_source_selection.py tests/preprocess/test_multiday.py tests/preprocess/test_gui_config_model.py tests/preprocess/test_gui_preflight.py tests/preprocess/test_sorter_partitions.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_gui_config_model.py tests/preprocess/test_multiday.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_multiday.py tests/preprocess/test_gui_preflight.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py tests/preprocess/test_gui_preflight.py`
- `python -m py_compile src/preprocess/gui/run_pipeline.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py`
- `python -m py_compile src/preprocess/io.py src/preprocess/gui/app.py src/preprocess/gui/run_pipeline.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_io_source_selection.py tests/preprocess/test_gui_config_model.py`
- `python -m py_compile src/postprocess/pipeline.py src/preprocess/gui/app.py src/preprocess/gui/run_pipeline.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/postprocess/test_postprocess_target_resolution.py tests/preprocess/test_gui_config_model.py`
- `python -m py_compile src/postprocess/pipeline.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/gui/run_pipeline.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/postprocess/test_postprocess_target_resolution.py tests/preprocess/test_gui_config_model.py`
- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/config_model.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py`
- `python -m py_compile src/preprocess/gui/app.py`
- `git diff --check -- external/CellExplorer/CellExplorer.m implementation_plan/2026-06-17-sorter-stage-and-multiday.md`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "issues = checkcode('external/CellExplorer/CellExplorer.m'); disp(numel(issues));"`
- `git diff --check -- external/matlab/run_cell_explorer_processing.m external/CellExplorer/calc_CellMetrics/loadSpikes.m implementation_plan/2026-06-17-sorter-stage-and-multiday.md change_log/2026-06-24-multiday-staging.md`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "issues = checkcode('external/matlab/run_cell_explorer_processing.m'); disp(numel(issues));"`

## Result

- Python compile passed.
- Focused Python tests passed: `48 passed`.
- Focused GUI/preflight tests passed after Force stop changes: `27 passed`.
- Focused multi-day/preflight tests passed after staged XML copy guard: `23 passed`.
- Focused GUI config/preflight tests passed after Manual Curation folder UI update: `17 passed`.
- GUI config tests passed after JSON-safe result reporting: `6 passed`.
- Focused IO/GUI config tests passed after preprocess manifest JSON-safe path
  handling: `23 passed`.
- Focused postprocess target/GUI config tests after multi-sorter manifest
  postprocess target resolution: `14 passed`.
- Focused postprocess target/GUI config tests passed after multi-sorter
  postprocess target resolution: `14 passed`.
- GUI config tests passed after explicit CellExplore folder registration:
  `8 passed`.
- GUI app compile passed after CellExplore raw/`_spi` mutual exclusion:
  passed.
- CellExplorer MATLAB static check completed and parsed the file; the vendored
  file still reports its existing checkcode warnings.
- CellExplorer wrapper MATLAB static check completed with `0` issues after
  restoring `getWaveformsFromDat=true`.

## Known Limitations and Next Steps

- This first pass uses symlinks rather than copying source binaries.
- Loaded or staged XML is treated as authoritative; when it differs from
  per-day XML metadata, the staged XML wins.
- For Open Ephys, XML `nChannels` remains sorting/chanMap metadata while the raw
  binary frame size comes from `structure.oebin`.
- It does not yet reconcile per-day anatomical metadata differences.
- GUI multi-day selection uses the staged basepath for preprocessing; richer
  per-session status and editable ordering are still pending.
- Open Ephys support is designed around symlinking datetime recording folders,
  but has not yet been exercised with an end-to-end Open Ephys fixture.
- The CellExplorer X11 visibility fix was checked statically; it still needs an
  interactive forwarded-X11 launch to confirm behavior on the user's Windows
  X server.
