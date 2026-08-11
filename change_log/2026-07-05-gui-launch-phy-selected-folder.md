# GUI Launch Phy Selected Folder

## Date and state

- Date: 2026-07-05
- Git base: `19552a3`
- Status: uncommitted

## Plan

- [implementation_plan/2026-07-05-gui-launch-phy-selected-folder.md](../implementation_plan/2026-07-05-gui-launch-phy-selected-folder.md)

## What changed

- Updated `MainWindow._resolve_phy_params_path()` so the default Launch Phy
  resolution checks the selected folder before the matching `_spi` sibling.
- Added an explicit `prefer_postprocessed=True` mode and used it for
  CellExplorer resolution so existing CellExplorer postprocessed-folder
  preference stays unchanged.
- Added focused tests to `tests/preprocess/test_gui_config_model.py` for
  selecting raw folders, selecting `_spi` folders, fallback to `_spi` when the
  selected folder has no `params.py`, and the postprocessed preference mode.
- Updated implementation/change-log indexes.

## Why

Manual Curation lets the user choose a specific sorting folder. When both
`Kilosort_.../params.py` and `Kilosort_..._spi/params.py` existed, Launch Phy
ignored the explicit non-`_spi` selection and opened the `_spi` folder instead.

## Verification

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py
```

Result: `12 passed in 0.99s`.

```bash
python -m py_compile src/preprocess/gui/app.py
```

Result: passed.

An initial attempt with `python -m pytest tests/preprocess/test_gui_launch_phy.py
-q` failed because the default Python environment does not have `pytest`
installed.

## Known limitations and next steps

- This was verified with focused unit tests rather than an interactive GUI
  launch.
- Postprocess target resolution is unchanged.
