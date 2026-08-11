# GUI Phy Live Curation Status

## Goal and motivation

Improve the GUI Phy launch workflow by showing readable curation time, live
cluster classification counts, and final counts when Phy exits.

## Current problem

The GUI currently launches Phy detached and summarizes `phy.log` once. Detached
launch prevents detecting when Phy closes, and the raw `nelpy` duration string
can be hard to read, for example `1:22:51:784 hours`.

## Why this is needed now

Manual curation is being run from the GUI. Users need to see whether
unclassified clusters remain while curating and need a final summary when Phy is
closed.

## Relevant state

- `Run phy` resolves `_spi/params.py` first.
- `neuro_py.raw.spike_sorting.phy_log_to_epocharray` can summarize `phy.log`
  when writable cache directories are set.
- `cluster_info.tsv` is the Phy-side table used by the referenced `neuro_py`
  progress helper.

## Affected modules/files

- `src/preprocess/gui/app.py`
- `change_log/2026-06-23-gui-phy-live-curation-status.md`

## Public parameters or API changes

No config or public API changes. GUI behavior changes:

- `Run phy` starts a managed Phy `QProcess` instead of a detached process.
- GUI log prints live curation counts while Phy is running.
- GUI log prints final counts and a formatted curation-time summary when Phy
  exits.

## Behavior

- Read `<params.py parent>/cluster_info.tsv` periodically.
- Count `good`, `mua`, `noise`, and unclassified clusters.
- Treat blank, `nan`, `none`, `unsorted`, and `unclassified` groups as
  unclassified.
- Only append live status when counts change, to avoid log spam.
- On Phy exit, append one final status summary and then summarize `phy.log`.
- Format durations as `1 h 22 min 51.784 s` rather than the raw `nelpy`
  duration text.

## Tests and checks

- Compile-check `src/preprocess/gui/app.py`.
- Smoke-check the cluster count helper against an existing `_spi` folder.

## Non-goals

- Do not continuously plot progress in the GUI.
- Do not force-save Phy state.
- Do not terminate Phy when the main pipeline force-stop button is pressed.
