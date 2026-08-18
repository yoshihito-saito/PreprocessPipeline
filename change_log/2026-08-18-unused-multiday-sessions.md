# Ignore Unused Multi-Day Sessions

## Date And Commit

- Date: 2026-08-18
- Commit: uncommitted
- Plan: [2026-07-16 multi-day subepoch selection](../implementation_plan/2026-07-16-multiday-subepoch-selection.md)

## What Changed

- When an explicit `Use` selection exists, multi-day staging now discovers and validates only sessions containing at least one selected subepoch.
- Sessions whose rows are all unchecked may therefore contain no discoverable `amplifier.dat`, `continuous.dat`, or Open Ephys recording without failing the Run.
- Selected rows retain their original session-order indices in the manifest and CSV even when earlier sessions are skipped.
- Unknown selected paths and selected sessions with no discoverable data continue to fail clearly.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_multiday.py tests/preprocess/test_gui_config_model.py`
  - Result: 76 passed.
- `python -m py_compile src/preprocess/multiday.py tests/preprocess/test_multiday.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Known Limitations And Next Steps

- An empty selection still means the legacy default of using all discovered subepochs; it does not mean “use none.”
