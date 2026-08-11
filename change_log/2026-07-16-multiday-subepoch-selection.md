# Multi-Day Subepoch Selection

## Date and Git State

- Date: 2026-07-16
- Branch: `feature-multiday-subepoch`
- Commit: uncommitted
- Plan: [2026-07-16 multi-day subepoch selection](../implementation_plan/2026-07-16-multiday-subepoch-selection.md)

## What Changed

- Added `selected_subepoch_paths` to `prepare_multi_day_basepath()`.
  - Empty or omitted selection preserves the existing behavior and stages all
    discovered subepochs.
  - Non-empty selection filters by resolved source subepoch folder path.
  - Unknown selected subepoch paths fail before staging symlinks are created.
- Added `discover_multi_day_subepochs()` for shared GUI/backend discovery of
  session subepochs.
- Added cleanup of stale staged subepoch symlinks from the previous
  `multi_day_manifest.json` when overwrite is enabled, so excluded subepochs
  are not accidentally rediscovered in the staged multi-day basepath.
- Extended GUI settings with
  `multi_day_selected_subepoch_paths: list[str]`.
- Updated the GUI `Check order` dialog into a session-order plus subepoch
  selection table:
  - rows show `Use`, `Order`, `Session`, and `Subepoch`;
  - source paths remain available internally and as subepoch tooltips, but no
    longer consume visible table width;
  - unchecking a subepoch turns the table into an explicit selection;
  - editing order still reorders whole sessions;
  - applying the dialog saves both order and selected subepochs.
- Updated the GUI runner to pass selected subepoch paths into multi-day staging.
- Added `multi_day_selected_subepochs.csv` output to both the server and local
  multi-day basepaths. The CSV records final staged order, session/subepoch
  indices, source/staged paths, source binary path, source type, channel count,
  sampling frequency, and sample count.
- Added the selected subepochs CSV path to the GUI runner result payload and
  run log output.
- Updated the run preview and multi-day summary line to show whether all
  subepochs or an explicit count will be used.
- Added tests for selected subepoch filtering, stale staged-link cleanup,
  selected subepoch CSV output, unknown selected path rejection, and GUI
  settings roundtrip.

## Why

Multi-day runs previously staged every discovered subepoch under every selected
session. This made it difficult to exclude bad or irrelevant epochs without
changing the source directory layout. The new workflow keeps session/basepath
selection intact and makes subepoch inclusion explicit at the existing order
review step.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_multiday.py`
  - Result: `14 passed`
- `python -m py_compile src/preprocess/multiday.py src/preprocess/gui/config_model.py src/preprocess/gui/run_pipeline.py src/preprocess/gui/app.py tests/preprocess/test_multiday.py`
  - Result: passed
- `git diff --check -- src/preprocess/multiday.py src/preprocess/gui/config_model.py src/preprocess/gui/run_pipeline.py src/preprocess/gui/app.py tests/preprocess/test_multiday.py implementation_plan/2026-07-16-multiday-subepoch-selection.md implementation_plan/README.md`
  - Result: passed

## Known Limitations and Next Steps

- The bare `pytest` executable in this environment resolves to Python 2.7 and
  fails before collection on typed test files. Verification used the existing
  Python 3 `phy2` test environment instead.
- The GUI table itself is covered by syntax checks and backend/settings tests,
  but not by an automated Qt interaction test.
