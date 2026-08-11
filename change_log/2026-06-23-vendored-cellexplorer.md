# Vendored CellExplorer

## Date and state

- Date: 2026-06-23
- Git state: uncommitted
- Branch: `feature/integrating-cellexplore`
- Related plan:
  `implementation_plan/2026-06-23-cellexplorer-launch-workflow.md`

## What changed

- Created branch `feature/integrating-cellexplore`.
- Added ayalab1 CellExplorer under `external/CellExplorer`.
- Excluded the upstream `.git` directory from the vendored copy.
- Added `external/CellExplorer/VENDORED_SOURCE.md` documenting the upstream
  repository and pinned commit.
- Removed vendored `*.cell_metrics.cellinfo.mat` ground-truth metric files and
  ignored `+groundTruthData/*.cell_metrics.cellinfo.mat`.
- Added the GUI `Launch CellExplorer` action next to `Launch Phy`.
- Added anatomical map load/edit/save controls to the channel-map preview.
- Added `src/preprocess/gui/anatomical_map.py` for pyNeuroscope-style
  no-header `anatomical_map.csv` round-tripping.
- Reworked the anatomical map editor to follow the pyNeuroscope
  `brain_region_editor.py` interaction model: channel dot view,
  click/Ctrl-click selection, rectangle selection, assigned-region list,
  assign/clear actions, help, and CSV save-as.
- Added `external/matlab/run_cell_explorer_processing.m` to run
  `gui_session`, explicit `_spi` Phy loading, `ProcessCellMetrics`, and
  `CellExplorer` through the vendored CellExplorer path.
- Updated vendored `getWaveformsFromDat` to handle absolute MergePoints epoch
  folder paths, so local outputs can still read raw sub-epoch dat files from
  the source basepath.

## Why

PreprocessPipeline needs to call a controlled CellExplorer version for
`gui_session`, `ProcessCellMetrics`, `loadSpikes`, and related MATLAB workflow
steps instead of depending on the user's global MATLAB path.

## Verification

- Confirmed source commit:
  `git -C /tmp/ayalab1_CellExplorer rev-parse HEAD`
- Confirmed vendored root exists:
  `external/CellExplorer/CellExplorer.m`
- Confirmed the vendored copy does not include the upstream `.git` directory.
- Confirmed no vendored `*.cell_metrics.cellinfo.mat` files remain.
- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/anatomical_map.py tests/preprocess/test_anatomical_map_helpers.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_anatomical_map_helpers.py`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "msgs = checkcode('external/matlab/run_cell_explorer_processing.m','-id'); ..."`

## Known limitations and next steps

- The vendored tree still includes upstream `exampleData` and MATLAB toolboxes,
  making the directory about 113 MB.
- Interactive `gui_session` and CellExplorer launch were not exercised in this
  non-interactive verification pass.
- The first implementation remains single-sorter-target only; multi-Kilosort
  group/shank support is still future work.
