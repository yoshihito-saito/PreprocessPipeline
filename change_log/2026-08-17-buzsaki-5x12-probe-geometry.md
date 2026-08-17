# Buzsaki 5x12 Probe Geometry

## Date And Commit

- Date: 2026-08-17
- Commit: uncommitted
- Plan: [2026-08-17 Buzsaki 5x12 probe geometry](../implementation_plan/2026-08-17-buzsaki-5x12-probe-geometry.md)

## What Changed

- Added `Buzsaki 5x12` to the GUI probe geometry selector and chanMap builder.
- Added a fixed five-shank, 64-channel geometry with group sizes `[12, 12, 16, 12, 12]`.
- Kept the four side shanks on the existing 12-site `poly2` layout.
- Defined the center shank as four single-column sites at y = 800, 600, 400, and 200 um followed by the standard 12-site `poly2` tip cluster.
- Added XML-description and saved-config aliases for spaced, hyphenated, underscored, abbreviated, and multiplication-sign forms of the probe name.
- Normalized saved aliases before selecting the non-editable GUI geometry combo box.
- Added explicit validation so malformed group counts or sizes cannot silently produce a different physical design.
- Added tracked coordinate, chanMap integration, invalid-shape, XML detection, and GUI alias round-trip tests.

## Why

The existing `middle_finger` probe also has a specialized center shank, but its center is an entirely linear, deeper 16-site layout. The Buzsaki 5x12 instead retains the 12-site `poly2` tip cluster and adds four 200-um-spaced linear sites above it.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/test_buzsaki_5x12_geometry.py tests/preprocess/test_gui_config_model.py`
  - Result: 46 passed.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_chanmap_geometry.py`
  - Result: 8 passed.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/io.py src/preprocess/gui/app.py tests/test_buzsaki_5x12_geometry.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Known Limitations And Next Steps

- The center group is interpreted in top-to-bottom order: four linear sites followed by the 12 `poly2` sites. A hardware-specific channel wiring remap would require an electrical channel map with labeled sites.
- This geometry intentionally accepts only the five-group `[12, 12, 16, 12, 12]` hardware shape.
