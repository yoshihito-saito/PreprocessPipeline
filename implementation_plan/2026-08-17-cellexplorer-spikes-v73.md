# CellExplorer spikes v7.3 saving

## Goal

Ensure the CellExplorer wrapper saves large combined `spikes` structures instead
of leaving a header-only MAT file when the structure exceeds the MATLAB v7 size
limit.

## Affected file

- `external/matlab/run_cell_explorer_processing.m`

## Intended behavior

Save both pre-Phy and post-Phy spikes outputs with MATLAB `-v7.3`. Keep the
existing spike loading, multi-sorter merging, filenames, and CellExplorer
processing behavior unchanged.

## Verification

- Run MATLAB `checkcode` on the wrapper.
- Save and reload a small representative `spikes` structure with `-v7.3`.
- Inspect the final task-scoped diff.
