# GUI Launch Phy Selected Folder

## Goal and motivation

Make `Launch Phy` open the folder explicitly selected in the Manual Curation
folder field. If the user chooses a raw Kilosort folder without `_spi`, Phy must
launch that folder's `params.py` instead of silently switching to a sibling
`_spi` folder.

## Current problem

`MainWindow._resolve_phy_params_path()` currently checks
`postprocess_output_folder_for_sorting(sorting_folder)` before the selected
folder whenever the selected folder does not end in `_spi`. As a result,
selecting `Kilosort_.../` opens `Kilosort_..._spi/params.py` if that sibling
exists.

## Why this is needed now

Manual curation sometimes needs to inspect the raw sorter output before or
separately from the postprocessed SpikeInterface output. The GUI field already
allows selecting either folder, so launch behavior should match the user's
explicit selection.

## Relevant state

- Pre-edit `git status --short` showed existing local modifications in
  `change_log/README.md`, `sorter/Kilosort4_config.yaml`,
  `src/preprocess/gui/app.py`, and `src/preprocess/io.py`.
- The existing `src/preprocess/gui/app.py` changes appear unrelated to this
  Launch Phy bug and must be preserved.

## Affected modules/files

- `src/preprocess/gui/app.py`
- `tests/preprocess/test_gui_config_model.py`
- `implementation_plan/README.md`
- `change_log/README.md`
- `change_log/2026-07-05-gui-launch-phy-selected-folder.md`

## Public parameters or API changes

No public config, CLI, or Python API changes. This is a GUI behavior fix.

## Expected behavior

- Selecting a folder that contains `params.py` launches Phy in that exact
  folder, whether or not the folder name ends in `_spi`.
- Selecting a `_spi` folder continues to launch that `_spi` folder.
- For legacy selections that do not directly contain `params.py`, fallback
  candidates still include the run root, `sorter_output`, and matching `_spi`
  output so older workflows keep working.

## Tests and checks

- Add focused tests for `_resolve_phy_params_path()` covering raw-vs-`_spi`
  sibling selection and fallback behavior.
- Run the GUI config model test file containing the focused Launch Phy tests.
- Compile-check `src/preprocess/gui/app.py`.

## Non-goals

- Do not change postprocess target resolution.
- Do not change CellExplorer folder selection semantics.
- Do not alter how Phy itself is launched after `params.py` is resolved.
