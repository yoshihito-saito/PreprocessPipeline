# flex-G5 Probe Geometry

## Date And Commit

- Date: 2026-06-29
- Commit: uncommitted
- Plan: [2026-06-29 flex probe geometry](../implementation_plan/2026-06-29-flex-probe-geometry.md)

## What Changed

- Added `flex-G5` as a built-in chanMap geometry using the provided 64-channel wiring template.
- Added GUI support so `flex-G5` appears in the probe assignment geometry selector.
- Added support for XML files that represent one 64-channel flex-G5 block as two 32-channel anatomical groups.
- Documented and tested the intended 128-channel setup as two 64-channel `flex-G5` probe assignments, for example groups `0, 1` and groups `2, 3`.
- Set the flex-G5 top anchor contacts so channels 15 and 16 are 800 um above channels 49 and 46.
- Set the within-block 32-channel group spacing to 80 um, keeping the 800 um distance only for the top-anchor vertical separation.
- Increased the chanMap preview canvas size, layout priority, and full-geometry axis padding so wide and tall flex-G5 geometries remain inspectable, with labels allowed outside the data clip region.
- Regenerate `chanMap.mat` from the current GUI XML/probe assignments before Run all / Run preprocess so stale local chanMaps are not reused after changing XML or geometry settings.
- Synchronize the GUI bad-channels field with the generated chanMap when chanMap generation is confirmed or prepared for a run.
- Added unit tests for the standalone flex-G5 coordinate helper and chanMap generation.

## Why

The flex-G5 probe used for upcoming motion-correction tests needs an explicit geometry that can be generated from the preprocessing GUI instead of manually supplying an external chanMap.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_chanmap_geometry.py`
  - Result: 7 passed.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/io.py src/preprocess/gui/app.py`
  - Result: passed.

## Known Limitations And Next Steps

- The geometry assumes the provided 64-channel wiring template; larger groups are tiled as additional 64-channel blocks with 1000 um x offsets.
- Motion correction and data-level validation are not included in this change.
