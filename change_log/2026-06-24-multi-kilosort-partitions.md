# Multi-Kilosort Partitions

## Date and Git State

- Date: 2026-06-24
- Branch: `feature/multi-kilosort-and-multi-day-recording`
- Commit: uncommitted
- Plan: [2026-06-17 sorter stage reliability, probe-scoped sorting, and multi-day sessions](../implementation_plan/2026-06-17-sorter-stage-and-multiday.md)

## What Changed

- Added `sorter_partition_mode` with `all`, `probe`, and `shank` modes to
  preprocess config and GUI settings.
- Added the GUI `Run sorter on` selector with `All channels`, `Each probe`, and
  `Each shank` options.
- Added chanMap-based sorter partition construction:
  - probe partitions use `probe_ids`;
  - shank partitions use `(probe_id, kcoords)` pairs;
  - disconnected and rejected channels are excluded from partition inputs.
- Added `sorter_partition_manifest.json` with partition channel provenance,
  output folders, and status.
- Extended sorter execution so explicit partition channels can be sorted even
  when the input binary is already preprocessed.
- Added sorter CLI options for explicit 0-based channel lists:
  `--active-channels` and `--exclude-channels`.
- Patched partitioned Phy outputs back to full-binary channel identity through
  `channel_map.npy` and `cluster_info.tsv`.
- Updated GUI pipeline handoff so multiple sorter outputs use the sorting
  search root instead of silently selecting one folder.
- Extended the CellExplorer MATLAB wrapper to accept one or more sorting
  folders and merge multiple `loadSpikes` results in the Neurocode
  multi-Kilosort style before running `ProcessCellMetrics`.

## Why

Multi-probe and multi-shank recordings need independent sorter runs without
losing full-session channel identity. Phy should still open one selected folder,
while CellExplorer should be able to process all selected partition folders as a
single merged `spikes` struct.

## Verification

- `python -m py_compile src/preprocess/metafile.py src/preprocess/sorter_runner.py src/preprocess/pipeline.py src/preprocess/gui/config_model.py src/preprocess/gui/run_pipeline.py src/preprocess/gui/app.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_partitions.py tests/preprocess/test_gui_config_model.py`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "msgs = checkcode('external/matlab/run_cell_explorer_processing.m','-id'); if isempty(msgs), disp('checkcode: ok'); else, disp(struct2table(msgs)); error('checkcode reported messages'); end"`

## Result

- Python compile passed.
- Focused Python tests passed: `8 passed`.
- MATLAB `checkcode` passed.

## Known Limitations and Next Steps

- The GUI still uses the existing sorting-folder text field/browse workflow for
  choosing the Phy folder; a richer table with per-partition status and
  CellExplorer include checkboxes is still pending.
- Multi-day folder selection and manifest construction are planned but not yet
  implemented in this change.
- A broader existing sorter runner test file still has unrelated failures in
  pre-existing expectations around Kilosort helper behavior.
