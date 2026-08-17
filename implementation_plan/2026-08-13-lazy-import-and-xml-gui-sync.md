# Lazy startup imports and XML GUI synchronization

## Goal and motivation

Make local Windows GUI/controller startup reliable when SpikeInterface sorter
imports are slow, and make `Load XML` immediately refresh the chanMap preview
and probe-assignment controls.

## Current problem

- Importing the `src` package eagerly imports postprocess and preprocess
  pipelines, which eagerly imports `spikeinterface.sorters` and its complete
  sorter registry before a stage needs it. The observed local run was
  interrupted while compiling the Spyking Circus 2 sorter module.
- GUI `Load XML` only changes the XML path field. Preview refresh depends on a
  timer and silently ignores channel-map construction errors, while probe rows
  are never derived from the selected XML.

## Affected files

- `src/__init__.py`, `src/preprocess/__init__.py`, and
  `src/postprocess/__init__.py`: preserve public exports with lazy attribute
  loading for pipeline/stage functions.
- `src/preprocess/sorter_runner.py`: defer the sorter registry import until a
  sorter operation actually needs it.
- `src/preprocess/io.py` and `src/preprocess/gui/app.py`: derive a safe default
  assignment from XML groups/description and synchronously update the preview
  after XML selection.
- `tests/`: regression coverage for lazy imports and XML GUI synchronization.

## Expected behavior

- Importing `src.preprocess.gui.app` does not import the full
  `spikeinterface.sorters` registry.
- Running a sorter still imports and uses the same SpikeInterface sorter
  classes and parameters.
- Selecting a valid XML parses its anatomical groups, uses its description to
  choose the default geometry when available, updates the probe assignment
  rows, and renders the chanMap preview. Existing bad-channel controls remain
  unchanged unless XML marks channels as skipped.
- Invalid XML produces a visible GUI warning/log entry and does not leave stale
  preview state disguised as a successful refresh.

## Verification strategy

- Run focused GUI/config and channel-map tests.
- Test a clean Python import with `spikeinterface.sorters` absent from
  `sys.modules` until sorter functionality is requested.
- Run the relevant broader preprocess test subset and inspect the final diff.

## Non-goals

- Do not change sorting algorithms, sorter parameters, or XML channel-group
  semantics.
- Do not overwrite user-customized assignments when merely refreshing the
  preview; XML load itself uses XML-derived defaults because it is an explicit
  metadata selection action.
