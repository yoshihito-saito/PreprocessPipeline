# Copy outputs while retaining local data

- Date: 2026-09-01
- Status: permission extension uncommitted on `main`; initial copy behavior committed as `37d6ab7`
- Plan: [2026-09-01 copy outputs while retaining local data](../implementation_plan/2026-09-01-copy-outputs-retain-local.md)

## What changed

- Changed a successful storage transfer with local cleanup disabled to retain every local source item instead of deleting the selected copied items.
- Kept full local-session deletion available only through the explicit post-verification deletion option.
- Renamed GUI-facing storage action text from move to copy and changed the cleanup label to `Delete local after verified copy`.
- Made local deletion disabled by default.
- Applied collaborative permissions to the complete destination tree, including the destination root and pre-existing content, after copy verification.
- Updated the README and transfer regression tests for the copy semantics.

## Why

The prior unchecked cleanup state still deleted copied source files, which contradicted the safety expectation that disabling local cleanup preserves the local session.

## Verification

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_move_outputs.py
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/gui/app.py tests/preprocess/test_gui_move_outputs.py
```

Result: all 16 focused tests passed and compilation completed successfully.

## Known limitations

- Internal helper and result-field names retain their existing `move` terminology for compatibility.
- The copy runs synchronously in the GUI and does not yet expose progress or cancellation controls.
