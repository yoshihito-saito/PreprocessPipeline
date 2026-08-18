# Multi-Day Subepoch Selection

## Goal and Motivation

Allow multi-day preprocessing runs to include only selected subepochs from the
selected session/basepath folders. The GUI should keep the existing session
selection workflow, then expose subepoch checkboxes in the multi-day order
dialog so users can review session order and choose the recordings that will be
staged and sorted.

## Current Problem

The current multi-day workflow stores only `multi_day_session_paths` and
`prepare_multi_day_basepath()` automatically discovers and stages every
subepoch under each selected session folder. This is too coarse when a session
contains calibration, bad, exploratory, or otherwise excluded recording epochs.
The only available workaround is to make temporary directories or manually move
subepoch folders before running, which is error-prone and makes the staged
manifest less representative of the intended run.

When an explicit selection is saved, runtime discovery currently validates
every session folder before applying that selection. A session whose rows are
all unchecked can therefore fail the Run with `No subepochs found`, even though
none of its data will be staged.

## Why This Is Needed Now

The GUI is already the main front end for multi-day preprocessing, and the
existing `Check order` dialog is the natural checkpoint before staging. Adding
subepoch selection there keeps basepath/session identity intact while giving
the user explicit control over the staged recording list.

## Git State at Planning Time

- Date: 2026-07-16
- Branch: `feature-multiday-subepoch`
- Existing unrelated uncommitted change: `sorter/Kilosort4_config.yaml`

## Affected Modules and Files

- `src/preprocess/multiday.py`
- `src/preprocess/gui/app.py`
- `src/preprocess/gui/config_model.py`
- `src/preprocess/gui/run_pipeline.py`
- `tests/preprocess/test_multiday.py`
- `implementation_plan/README.md`
- `change_log/README.md`
- `change_log/2026-07-16-multiday-subepoch-selection.md`

## Public Parameters and API Changes

Add an optional selected-subepoch input to the staging API:

```python
prepare_multi_day_basepath(
    session_paths=[...],
    selected_subepoch_paths=[...],
)
```

When `selected_subepoch_paths` is omitted or empty, the existing behavior is
preserved: all discovered subepochs are staged. When it is provided, discovery
still runs per session, but only matching source subepoch folders are staged.
The GUI settings model should persist the selected source subepoch paths as:

```python
multi_day_selected_subepoch_paths: list[str]
```

The staging result should also expose a CSV summary path:

```python
selected_subepochs_csv_path: Path
```

The CSV should be written to both the server and local multi-day basepaths as
`multi_day_selected_subepochs.csv`.

## Expected Behavior

- `Browse for multi-days` still selects session folders, not subepoch folders.
- `Check order` shows session order plus discovered subepochs with a `Use`
  checkbox.
- The table should prioritize visible subepoch names. Source paths may remain
  available internally or as tooltips, but should not take visible table space
  from the `Subepoch` column.
- Editing session order updates the order while retaining selected subepochs.
- `Update order` saves both the ordered sessions and the checked subepochs.
- If no explicit subepoch selection exists, all discovered subepochs remain
  selected by default so old saved configs behave the same.
- Run-time staging uses only checked subepochs when a selection has been saved.
- With an explicit selection, sessions containing no checked subepochs are not
  discovered or validated; their original session-order indices remain intact
  for selected rows from later sessions.
- Staging writes a human-readable CSV containing the final staged order,
  session identity, source subepoch path, staged subepoch path, source binary,
  source type, channel count, sampling frequency, and sample count.
- Staging fails with a clear error if a non-empty subepoch selection produces
  no selected subepochs.

## Tests and Verification

- Add backend tests that verify selected subepochs are filtered and ordered
  through `prepare_multi_day_basepath()`.
- Add backend tests that verify the selected/staged subepoch CSV is written and
  matches the final staged order.
- Add run-pipeline or settings-level coverage that confirms selected subepoch
  paths are passed into staging.
- Run `pytest -q tests/preprocess/test_multiday.py`.
- Run syntax/format sanity checks where appropriate.

## Non-Goals

- Do not make subepoch folders replace `basepath` in the main GUI.
- Do not add nested manual editing of source paths outside the existing order
  dialog.
- Do not change single-day preprocessing discovery semantics.
