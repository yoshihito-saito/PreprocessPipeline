# A5x12-16-Buz-Lin Probe Name

## Date And Commit

- Date: 2026-08-17
- Commit: uncommitted
- Implementation plan: not created, per user request

## What Changed

- Replaced the GUI probe label `middle_finger` with `A5x12-16-Buz-Lin-5mm-100-200-160-177`.
- Updated XML-description detection and chanMap dispatch to use the formal probe name.
- Kept the existing middle-finger coordinate geometry unchanged.
- Did not retain `middle_finger`, `middle finger`, or `middlefinger` as backward-compatible aliases, per user request.
- Added a tracked regression test for formal-name XML detection, GUI registration, and 64-channel chanMap generation.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/test_middle_finger_probe_name.py tests/preprocess/test_chanmap_geometry.py`
  - Result: 9 passed.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/test_buzsaki_5x12_geometry.py tests/preprocess/test_gui_config_model.py`
  - Result: 46 passed.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/io.py src/preprocess/gui/app.py tests/test_middle_finger_probe_name.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Known Limitations

- Saved configurations or XML descriptions that still use the old `middle_finger` label must be updated to the formal probe name.
