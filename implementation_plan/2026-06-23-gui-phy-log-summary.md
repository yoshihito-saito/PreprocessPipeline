# GUI Phy Log Summary

## Goal and motivation

After launching Phy from the GUI, summarize the corresponding `phy.log` using
`neuro_py.raw.spike_sorting.phy_log_to_epocharray` and append the result to the
GUI log.

## Current problem

The new `Run phy` button starts Phy, but the GUI does not expose any summary of
prior or current manual curation activity recorded in `phy.log`. The requested
`neuro_py` helper can estimate curation epochs from Phy's timestamped log.

## Why this is needed now

Manual curation is now reachable from the GUI. Showing a log summary immediately
after launch makes it easier to confirm which Phy output folder is being used
and whether curation activity has been recorded there.

## Relevant state

- `Run phy` currently resolves `<run>_spi/params.py` first and launches Phy
  detached from the GUI.
- `neuro_py` is available from `/local/workdir/ys2375/GitHub/neuro_py`.
- Importing the helper needs writable matplotlib/numba cache directories in
  this environment.

## Affected modules/files

- `src/preprocess/gui/app.py`
- `change_log/2026-06-23-gui-phy-log-summary.md`

## Public parameters or API changes

No config or public API changes. The GUI log gains additional text after a
successful Phy launch.

## Expected behavior

- After `Run phy` starts successfully, the GUI schedules a short delayed read of
  `<params.py parent>/phy.log`.
- If `phy.log` exists, `phy_log_to_epocharray()` is called with the default
  merge gap and the GUI log shows the path, number of epochs, and total
  duration.
- If the log does not exist or the helper fails, the GUI shows a warning without
  failing the Phy launch.

## Tests and checks

- Compile-check `src/preprocess/gui/app.py`.
- Smoke-check the `neuro_py` helper against an existing `_spi/phy.log` with
  writable cache environment variables.

## Non-goals

- Do not monitor Phy continuously.
- Do not block or terminate Phy if log summarization fails.
- Do not parse `phy.log` manually when the `neuro_py` helper is available.
