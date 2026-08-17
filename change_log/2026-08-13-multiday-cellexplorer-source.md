# Multi-day CellExplorer Source Resolution

## Date and status

- Date: 2026-08-13
- Status: uncommitted

## What changed

- CellExplorer launch now treats a local `multi_day_manifest.json` as the
  authoritative source for the staged waveform root.
- The launcher validates the manifest schema and session name, requires a
  usable `server_basepath`, and verifies that every manifest subepoch exists
  below that staged root before MATLAB starts.
- Manifest-backed resolution is independent of the GUI's `multi_day_enabled`
  state, so reopening an existing multi-day output still overrides a stale
  first-day `source_basepath` from prior run metadata.
- Single-day sessions without a multi-day manifest retain their existing raw
  source-basepath behavior.
- The resolved waveform source root is printed in the CellExplorer launch log.

## Why

Multi-day preprocessing stages epochs with names such as
`001_Day14_D14_postsleep_...`, but the GUI can retain the originally selected
`Day14` path after the worker switches to the staged multi-day basepath. The
MATLAB MergePoints normalizer therefore searched for the staged epoch below the
wrong source root and failed before waveform extraction.

## Verification

- `python -m py_compile src/preprocess/gui/app.py tests/preprocess/test_gui_config_model.py`
  - passed.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py -k 'cell_explorer_source'`
  - `6 passed, 33 deselected`.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py`
  - `39 passed`.
- The resolver was run against the existing
  `multiday_Day14_to_Day217` local output and returned its manifest-defined
  `/fs/.../multiday_Day14_to_Day217` staging root after validating all 51
  staged subepochs.
- `git diff --check`
  - passed.

## Known limitations

- The interactive MATLAB/CellExplorer GUI was not launched as part of the
  automated verification. The existing MATLAB wrapper and MergePoints
  normalization path are unchanged; only the source root supplied to them is
  corrected.

## Implementation plan

- [CellExplorer launch workflow](../implementation_plan/2026-06-23-cellexplorer-launch-workflow.md)
