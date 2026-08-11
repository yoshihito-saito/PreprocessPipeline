# Behavioral Data Process

## Date And Commit

- Date: 2026-06-19
- Commit: uncommitted

## Plan

- [Implementation plan](../implementation_plan/2026-06-19-behavioral-data-process.md)

## What Changed

- Added `src/preprocess/behavior.py` for neurocode-compatible DLC behavior export.
- Added DLC discovery with filtered-output priority: `*_filtered.h5`, `*_filtered.csv`, `*.h5`, then `*.csv`.
- Added DLC CSV/H5 loading, likelihood filtering, camera TTL synchronization, frame/TTL mismatch warnings, centimeter conversion, epoch-wise calibration support, optional tracker-clean mask support, short-gap interpolation, and `basename.animal.behavior.mat` saving.
- Added a GUI Behavior tab with DLC discovery, parsed DLC point-name selection, central behavior track preview, epoch frame tabs, two-click endpoint pixel measurement, batch calibration, polygon keep-range outlier cleanup, interpolation threshold, and local-output export.
- Reorganized the GUI into top-level `Ephys` and `Behavior` tabs.
- Added nested `Preprocess` and `Postprocess` tabs inside `Ephys`, with Ephys run buttons shown only inside that parent tab.
- Kept data-move controls in the persistent bottom bar because they still apply to local/basepath output transfer.
- Reworked the central Behavior viewer into `Calibration` and `Outlier cleanup` tabs, each with its own compact in-tab controls.
- Moved `known distance cm` and `Run calibration` into the `Calibration` viewer tab.
- Moved `interpolate gaps <= sec` and `Run outlier clean up` into the `Outlier cleanup` viewer tab.
- Changed the Behavior GUI to require explicit DLC tracking point selection, showing compact bodypart-style labels while keeping full parsed DLC point names internally.
- Fixed DLC point label parsing so bodypart names such as `left ear`, `body_1`, and `body_2` are preserved instead of collapsing to the last numeric token.
- Added automatic 2D track loading when opening the `Outlier cleanup` tab after calibration.
- Added epoch child tabs under `Outlier cleanup`, matching the Calibration epoch-tab structure.
- Restored neurocode-style DLC extra-point field naming (`<dlc_x_or_y_column>_point`) and enabled MATLAB long field-name saving to avoid SciPy's 31-character default limit.
- Changed Behavior GUI defaults to likelihood threshold `0.60`, fallback video FPS `40 Hz`, and interpolation gap threshold `1.0 s`.
- Changed `Run calibration` to update only the active Calibration epoch child tab; outlier cleanup/export still validate that all discovered epochs are calibrated.
- Removed the left-side `Spatial calibration` group so Behavior parameters live next to the viewer they control.
- Removed the visible left-side DLC discovery summary window so Behavior status and diagnostics are reported through the central log only.
- Added detailed TTL/frame mismatch diagnostics with epoch context, TTL/video counts, approximate mismatch duration in seconds, and the 1-based tail TTL or video frame indices truncated by the current alignment rule.
- Routed Behavior preview/export sync warnings into the GUI log and rendered warning entries in a red-toned text color.
- Stopped emitting expected behavior sync warnings through Python `warnings.warn`, so mismatch diagnostics no longer clutter the terminal when using `preprocess-gui`.
- Replaced outlier keep-polygon classification with an explicit closed-polygon ray-casting test, using the first clicked vertex as the start and closing from the last clicked vertex back to that start.
- Increased automatic x/y axis padding in the Outlier cleanup track plot so edge trajectories are easier to surround manually.
- Updated Behavior tracking-point changes so calibrated Outlier cleanup previews reload automatically with the newly selected DLC point, reusing existing calibration ratios and clearing stale point-specific outlier masks.
- Moved TTL/frame sync warning display to DLC discovery time only. Outlier cleanup preview rebuilds and tracking-point changes no longer repeat the same mismatch warning in the GUI log.
- Changed `Run outlier clean up` to apply the current polygon mask, run short-gap interpolation, and update the Outlier cleanup tabs with the post-cleanup track preview.
- Changed short-gap interpolation to run independently within each DLC epoch/subsession, preventing interpolation across concatenated epoch boundaries.
- Updated `Reset` in the Outlier cleanup viewer so that, after a post-cleanup preview is shown, it clears the applied mask and reloads the raw calibrated track for another cleanup pass.
- Treat polygon-excluded samples as outlier gaps: they are set to NaN before interpolation and can be filled by short-gap interpolation when the configured threshold allows it.
- Renamed the central cleanup viewer tab to `Outlier cleanup & interpolation`, changed the run button to `Apply outlier cleanup + interpolate`, and added GUI log details for rejected frame counts and interpolation fills.
- Fixed GUI log coloring so warning entries are red-toned while normal log entries are explicitly written in the default light text color.
- Added GUI config persistence for behavior settings.
- Exported behavior helpers from `src.preprocess`.
- Added runtime dependencies for Behavior preview/export: `imageio`, `imageio-ffmpeg`, and `tables`.
- Updated README implementation status for Tracking/DLC.

## Why

PreprocessPipeline had neurocode-compatible electrophysiology/session outputs but no equivalent for MATLAB `general_behavior_file`. The new behavior workflow lets users load DLC outputs from subepoch folders and write the final behavior MAT file to the local output directory instead of directly modifying source `basepath`.

## Verification

Focused behavior tests:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_behavior_neurocode_compat.py
```

Result: 8 passed.

Syntax check:

```bash
python -m py_compile src/preprocess/behavior.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/__init__.py
```

Result: passed.

Broader preprocess suite:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess
```

Result: 144 passed, 18 failed. The failures are in pre-existing artifact removal, pipeline artifact compatibility, and sorter runner tests outside the new behavior path.

## Known Limitations

- Calibration frame extraction uses OpenCV when available and falls back to `imageio`; `imageio-ffmpeg` is now declared for MP4 decoding.
- DLC H5 loading uses pandas HDF support, so `tables`/PyTables is now declared.
- Tracker jump cleaning currently keeps points inside a clicked polygon in x/y space and marks points outside that polygon as NaN.
- Legacy `.whl`, OptiTrack, and non-DLC behavior formats are not implemented in the Python behavior path.
- `implementation_plan/`, `change_log/`, and `tests/` are currently ignored by `.gitignore`; repository policy may need adjustment before committing these documentation/test artifacts.
