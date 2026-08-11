# GUI Run Phy Button

## Goal and motivation

Add a manual curation control to the Ephys GUI so users can launch Phy directly
from the local working output after sorting/postprocessing.

## Current problem

Users currently need to leave the GUI and manually run `phy template-gui` from a
terminal. This is error-prone because the correct target is the local
Kilosort/Phy output folder, whose `params.py` may contain relative paths such as
`../<basename>.dat`.

## Why this is needed now

The pipeline is being run interactively from the standalone GUI, and manual
curation is the next step after waveform/template processing. Providing a GUI
button keeps the workflow in one place and reduces path mistakes.

## Relevant state

- Pre-edit status showed an existing unrelated local change in
  `src/preprocess/recording.py` from the LFP tolerance work.
- This implementation will be limited to `src/preprocess/gui/app.py` plus
  documentation index updates.

## Affected modules/files

- `src/preprocess/gui/app.py`
- `implementation_plan/README.md`
- `change_log/README.md`
- `change_log/2026-06-23-gui-run-phy-button.md`

## Public parameters or API changes

No config, CLI, or public Python API changes. The GUI will gain a new
`Manual Curation` group below `Ephys run` with a `Run phy` button.

## Behavior

- The button resolves the target sorting folder with the same GUI settings used
  by postprocess: explicit `sorting folder` first, otherwise the newest sorting
  folder under the local output/basepath roots.
- If a matching postprocess/SpikeInterface output folder named `<run>_spi`
  contains `params.py`, it is preferred for Phy launch.
- If no `_spi` `params.py` exists, the original target must contain
  `params.py`; legacy `<run>/sorter_output/params.py` is also accepted.
- The button launches `phy template-gui params.py` detached from the GUI.
- The working directory is the folder containing `params.py` so relative paths
  inside `params.py` remain valid.
- Launching is blocked while a pipeline job is running.

## Tests and checks

- Compile-check `src/preprocess/gui/app.py`.
- Manual runtime validation can be done by pressing `Run phy` in the GUI after
  a sorter output is available.

## Non-goals

- Do not manage the lifetime of the detached Phy process.
- Do not add curation save/load logic.
- Do not change postprocess target resolution semantics.
