# CellExplorer Launch Workflow

## Date and state

- Date: 2026-06-23
- Git commit: uncommitted
- Plan: [2026-06-23 CellExplorer launch workflow](../implementation_plan/2026-06-23-cellexplorer-launch-workflow.md)

## What changed

- Added the CellExplorer launch workflow and anatomical map editor work under
  the existing CellExplorer integration plan.
- Clarified the anatomical map channel-number convention:
  - the chanMap preview and GUI bad-channel controls display 0-based device
    channel indices from `chanMap0ind`;
  - CellExplorer and anatomical map internals keep 1-based channel IDs;
  - the anatomical editor now displays 0-based channel labels while preserving
    1-based IDs for saving/loading.
- Changed the anatomical editor from a synthetic group-by-row display to the
  actual `chanMap.mat` spatial geometry using `xcoords` and `ycoords`.
- Added mouse-wheel zoom to the anatomical editor, centered on the current
  cursor position.
- Stopped the post-Phy CellExplorer wrapper from running `loadSpikes` twice by
  passing already-loaded spikes to `ProcessCellMetrics` with
  `getWaveformsFromDat=false`.
- Added CellExplorer MEX dependency handling for Linux. The wrapper now checks
  `CCGHeart` and `FindInInterval`, then compiles the vendored C sources with
  `mex -O` if the platform-specific MEX files are missing.
- Ignored generated Linux MEX binaries under the vendored CellExplorer tree.
- Changed the final manual curation launch to open CellExplorer from the saved
  basepath metrics file, after logging the generated cell count and expected
  metrics file path.
- Renamed the GUI button from `Launch CellExplorer` to
  `Run CellExplore postprocess`.
- Added `exit(0)` after successful MATLAB CellExplorer postprocess completion
  so the MATLAB process closes after the CellExplorer GUI is closed and the Qt
  buttons can be re-enabled.
- Refreshed the vendored ayalab1 CellExplorer copy from
  `b7e9314bcb3b909fcfaf03e7d614a32cf1cacbba` to
  `31ef26f9818b260a27273860f529ae40efa5b803`, which includes the upstream
  `fix-mergepoints-waveform-fallback` merge.

## Why

The chanMap preview and anatomical map editor previously appeared to disagree
because they displayed different channel-number conventions and because the
editor did not render the actual channel geometry. The CellExplorer side must
remain 1-based, but region assignment should happen on the same physical
contact layout shown by the chanMap preview.

The CellExplorer wrapper also called `loadSpikes` once explicitly and then
allowed `ProcessCellMetrics` to trigger a second waveform reload. That happened
because CellExplorer only treats `WaveformsSource='dat file'` as already
complete, while this workflow preserves MergePoints waveform provenance.

ACG metric calculation also depends on CellExplorer's compiled FMA helper
`CCGHeart`. The vendored CellExplorer includes Windows and macOS MEX binaries,
but Linux needs local `.mexa64` builds from `CCGHeart.c` and
`FindInInterval.c`.

The manual curation GUI should follow CellExplorer's usual single-session
entrypoint from `basepath`, not depend only on the in-memory `cell_metrics`
struct returned by `ProcessCellMetrics`.

MATLAB was previously left at the prompt after a successful CellExplorer run
because only the error path called `exit(1)`. The parent GUI therefore kept the
CellExplorer QProcess alive and disabled manual-curation buttons.

The vendored CellExplorer directory is not an in-place git checkout, so updating
it requires importing the latest upstream working tree while preserving local
PreprocessPipeline provenance and generated-MEX ignore rules.

## Verification

- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/anatomical_map.py`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_anatomical_map_helpers.py`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "msgs = checkcode('external/matlab/run_cell_explorer_processing.m','-id'); if isempty(msgs), disp('checkcode: ok'); else, disp(struct2table(msgs)); error('checkcode reported messages'); end"`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "mexDir = fullfile(pwd, 'external', 'CellExplorer', 'calc_CellMetrics', 'mex'); cd(mexDir); mex('-O', 'CCGHeart.c'); mex('-O', 'FindInInterval.c'); disp('CellExplorer MEX compile: ok')"`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "addpath(genpath(fullfile(pwd, 'external', 'CellExplorer'))); disp(which('CCGHeart')); disp(which('FindInInterval')); ..."`
- `git ls-remote https://github.com/ayalab1/CellExplorer.git HEAD refs/heads/main refs/heads/master`
- `git clone https://github.com/ayalab1/CellExplorer.git /tmp/ayalab1_CellExplorer_latest`
- `rsync -a --delete --exclude .git --exclude 'calc_CellMetrics/mex/*.mexa64' /tmp/ayalab1_CellExplorer_latest/ external/CellExplorer/`
- `/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "addpath(genpath(fullfile(pwd, 'external', 'CellExplorer'))); disp(which('CellExplorer')); disp(which('getWaveformsFromDat')); ..."`

## Result

- Python compile check passed.
- Anatomical map helper tests passed: `2 passed`.
- MATLAB `checkcode` passed.
- Linux MEX compilation completed with `gcc`, and MATLAB resolved both
  `CCGHeart.mexa64` and `FindInInterval.mexa64`.
- The CellExplorer wrapper now emits the cell count and metrics file path
  before opening `CellExplorer('basepath', basepath)`.
- Python compile check passed after the button rename and MATLAB `exit(0)`
  command update.
- Vendored CellExplorer now resolves to the refreshed local files, and MATLAB
  wrapper `checkcode` passed after the refresh.

## Known limitations and next steps

- The editor display now uses the physical chanMap geometry, while CSV
  save/load still uses the pyNeuroscope-compatible group/row order internally.
  That keeps CellExplorer compatibility but means CSV row order is not inferred
  visually from screen position.
- The first zoom implementation supports mouse-wheel zoom only. If dense probes
  need navigation after deep zoom, add explicit pan controls or a reset-view
  action.
