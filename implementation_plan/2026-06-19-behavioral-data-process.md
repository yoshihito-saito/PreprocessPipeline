# Behavioral Data Process

## Goal And Motivation

Add a neurocode-compatible behavioral data processing path for tracking data, with a GUI Behavior tab that lets users load DeepLabCut results and produce a standardized `basename.animal.behavior.mat` file.

The first target is parity with the practical DeepLabCut path in MATLAB neurocode:

- locate DLC CSV/H5 outputs in session folders;
- apply likelihood filtering;
- synchronize frames to camera TTL pulses from `digitalIn.events.mat` or `digitalin.dat`;
- concatenate across `MergePoints` subsessions;
- calibrate pixels to centimeters through a GUI measurement step;
- clean tracker jumps through a GUI outlier rejection step;
- interpolate short missing tracking gaps with explicit duration controls;
- save a CellExplorer/Buzcode-style `behavior` structure.

## Current Problem

PreprocessPipeline already has neurocode-compatible `session.mat`, `MergePoints.events.mat`, and analog/digital event outputs, but it does not yet implement the README TODO for Tracking/DLC:

```text
Tracking/DLC (`getPos`, `path_to_dlc_bat_file`, `general_behavior_file`)
```

MATLAB neurocode implements this through `preprocessSession(..., 'getPos', true)`, which optionally runs a DLC batch file and then calls `general_behavior_file('basepath', basepath)`. The relevant MATLAB files are:

- `/workdir/ys2375/GitHub/neurocode/preProcessing/preprocessSession.m`
- `/workdir/ys2375/GitHub/neurocode/behavior/general_behavior_file.m`
- `/workdir/ys2375/GitHub/neurocode/behavior/process_and_sync_dlc.m`
- `/workdir/ys2375/GitHub/neurocode/behavior/load_dlc_csv.m`

Unlike MATLAB neurocode, PreprocessPipeline should write the final behavior output to the local working output directory, not directly to source `basepath`. The local output can later be moved to `basepath` through the existing output move workflow.

## Branch And Worktree State

- Base branch at planning time: `main`
- Feature branch: `feature/behavioral-data-process`
- Worktree at branch creation: clean

## Affected Modules

Expected additions:

- `src/preprocess/behavior.py`: reusable behavior processing logic.
- `tests/preprocess/test_behavior_neurocode_compat.py`: focused tests for DLC CSV parsing, likelihood filtering, TTL/frame matching, and MAT struct output.

Expected integration points:

- `src/preprocess/metafile.py`: add behavior-related config fields.
- `src/preprocess/pipeline.py`: optionally run behavior processing after `MergePoints`, `digitalIn`, and `session.mat` are available.
- `src/preprocess/gui/config_model.py`: persist GUI behavior settings.
- `src/preprocess/gui/app.py`: add Behavior tab controls.
- `README.md`: update implementation status and user-facing notes.

## API And GUI Design

Add a dedicated Behavior tab rather than hiding behavior processing inside the main preprocessing controls.

Initial Behavior tab controls:

- enable behavior processing;
- DLC CSV/H5 discovery status by subsession;
- optional DLC batch/script path;
- primary DLC bodypart index, using MATLAB-compatible 1-based display and Python-normalized internal indexing;
- likelihood threshold, GUI default `0.60` for this workflow;
- frame/TTL sync tolerance, default `0.01`;
- maze/arena calibration distance, default empty and required for centimeter export;
- missing-gap interpolation threshold in seconds, GUI default `1.0`;
- optional overwrite;
- optional save/run button for behavior processing.

The tab should use existing pipeline outputs when available:

- `basename.MergePoints.events.mat`;
- per-subsession or output-level `digitalIn.events.mat`;
- `basename.session.mat`.

Behavior tab workflow:

1. Discover DLC outputs under basepath subsession folders. Prefer `*_filtered.h5`, then `*_filtered.csv`, then `*.h5`, then `*.csv`. Exclude non-DLC CSV/H5 files where the expected DLC columns are missing.
2. Sort subsession folders by recording order, using `MergePoints.foldernames` when available and falling back to the same discovery order used by preprocessing.
3. For each selected subsession, load the DLC result and associated video frame metadata. Display a representative recording frame in the GUI.
4. Let the user draw a calibration line over the frame and enter the physical distance, for example maze edge-to-edge `100 cm`. Compute `pixel_to_cm_ratio = pixel_distance / physical_distance_cm`.
5. Synchronize tracking frames to camera TTL pulses.
6. Apply likelihood filtering and pixel-to-cm conversion.
7. Open the tracker jump cleaning GUI and write the rejected frames as NaN.
8. Interpolate missing tracking segments only when the gap duration is below the configured threshold.
9. Save `basename.animal.behavior.mat` in neurocode-compatible format.

Revised interactive Behavior GUI workflow:

- The top-level Behavior tab label should be `Behavior`.
- When the Behavior tab is active, the central preview area should switch from
  `chanMap Preview` to `Behavior track preview`.
- DLC discovery should automatically populate the central preview with one tab
  per discovered DLC epoch. The tab label is the epoch/subsession folder name.
- Each epoch preview tab should load the first frame from the associated video
  and let the user define one calibration line directly over that frame by
  clicking the two endpoints.
- Calibration is applied in one batch from a `Run calibration` button in the
  preview area. If multiple DLC epochs are discovered, every epoch must have a
  drawn line before calibration succeeds.
- `known distance cm` remains the physical-distance input, but the pixel
  distances are measured from the preview tabs rather than manually entered in
  the left-side form.
- The DLC primary tracking point should be selectable by parsed DLC point/column
  name after discovery, instead of requiring users to know the MATLAB-style
  numeric coordinate index. The Behavior GUI should require the user to choose
  a DLC point from parsed point names and should display compact bodypart-like
  labels while retaining the full parsed name internally.
- A left-side `Run outlier clean up` button should compute a behavior preview,
  switch the central preview to a 2D track map, and let the user click polygon
  vertices around keep ranges. The polygon should be treated as closed even
  when the final clicked point is not exactly connected to the first point.
  Points outside the keep polygons are treated as outliers and replaced with
  NaN, following the practical intent of neurocode's manual tracker jump
  cleanup.
- Final export should use the calibrated epoch-wise ratios, apply the outlier
  mask, then interpolate short gaps before writing the local behavior MAT file.

Second-pass Behavior GUI layout refinement:

- The central Behavior viewer should expose separate `Calibration` and
  `Outlier cleanup` tabs rather than switching modes implicitly.
- The reset action should be a single `Reset` button that applies only to the
  active central viewer tab. In `Calibration`, it resets the current epoch tab's
  calibration line. In `Outlier cleanup`, it resets the current keep polygon(s).
- `Run calibration` and `Run outlier clean up` should both be placed below the
  central Behavior viewer, not in the left settings pane.
- The left-side `Spatial calibration` group should be removed. Its parameters
  should move below the central Behavior viewer: known distance for calibration
  and interpolation gap threshold for behavior export/cleanup.
- To reduce visual clutter, calibration controls should live inside the
  `Calibration` viewer tab, outlier/interpolation controls should live inside
  the `Outlier cleanup` viewer tab, long frame titles should be hidden, and
  intermediate calibration status text should be kept out of the viewer.
- `Run calibration` should apply to the active Calibration epoch child tab,
  allowing users to calibrate one epoch at a time. Export and outlier cleanup
  still require every discovered epoch to be calibrated.
- The fallback video FPS GUI default should be `40 Hz`.
- Saved MATLAB field names must stay within MATLAB's 31-character struct-field
  compatibility limit. Long DLC point labels should be preserved in
  `behavior.processinginfo`, while per-point coordinate fields should use short
  stable names.
- Outlier cleanup polygon classification should use the first clicked vertex as
  the polygon start, close the current polygon from the last clicked vertex back
  to that start, and keep only points inside that closed polygon. The track plot
  itself should include extra x/y axis padding so edge trajectories are easier
  to surround manually.
- Tracking point selection should be independent of spatial calibration. If the
  user calibrates first and then selects or changes the DLC tracking point while
  the Outlier cleanup viewer is active, the viewer should rebuild the calibrated
  centimeter-coordinate track using the existing epoch calibration ratios and
  should not require recalibration.
- TTL/frame mismatch warnings should be shown at DLC discovery time only. The
  user should not see the same sync warning every time the Outlier cleanup
  preview is rebuilt or a tracking point changes.
- `Run outlier clean up` should apply the current epoch-tab polygon masks,
  interpolate short gaps, and update the same Outlier cleanup tabs with the
  post-cleanup track so users can inspect the trajectory that will be exported.
  Short-gap interpolation should run independently per DLC epoch/subsession
  rather than across concatenated epoch boundaries.
- Polygon-excluded samples should be treated as outlier gaps: set them to NaN
  before interpolation, then let the configured short-gap interpolation fill
  them from neighboring kept positions when the gap is short enough and stays
  within the same DLC epoch/subsession. Longer outlier gaps should remain NaN.
- The cleanup tab should make the full operation explicit: outlier removal plus
  interpolation on the centimeter-converted behavior track.

## `behavior.trials` Decision

Do not preserve MATLAB's apparent overwrite bug.

In MATLAB `general_behavior_file.m`, `behavior.trials = trials;` is immediately followed by `behavior.trials = trialsID;`, which can erase valid trials when `trialsID` is empty. The Python implementation should instead use this rule:

- if explicit trial intervals are available, write them to `behavior.trials`;
- if trial IDs or state labels are available, write them to separate fields such as `behavior.trialID`, `behavior.trialIDname`, `behavior.states`, and `behavior.stateNames`;
- if only DLC subsession intervals are available, use `tracking.events.subSessions` as `behavior.trials`, matching the useful downstream behavior of neurocode without the overwrite.

This is a deliberate compatibility repair. The change log should state that the Python path does not reproduce the MATLAB overwrite because it loses valid metadata.

## Maze Size And Centimeter Conversion

Match the neurocode expectation that behavior coordinates are converted to centimeters by default.

The Behavior tab should require calibration before final export when `convert_xy_to_cm=true`:

- default export mode is centimeter output with `behavior.position.units = "cm"`;
- if the user has not provided a physical distance or calibration line, block export with a clear GUI validation error;
- support a GUI frame viewer where the user draws a line over a known maze/arena distance and enters that distance in centimeters;
- compute and store the pixel-to-centimeter ratio in the behavior metadata and non-sleep epochs, matching neurocode's `pix_to_cm_ratio` idea;
- optionally allow a developer/debug pixel-only mode, but not as the main GUI default.

## Tracker Jump Cleaning

`clean_tracker_jumps=true` can remain GUI-only and should be part of the normal Behavior tab workflow.

For the initial implementation:

- behavior export can be driven from the GUI, so manual tracker cleaning is acceptable;
- source logic should still separate the pure data transformation from the GUI mask editor;
- cleaned frames should be represented as NaN in all affected bodypart coordinate arrays;
- rejected frame masks should be saved in metadata so cleaning can be audited or reapplied.

## Algorithm Details

DeepLabCut file discovery and parsing:

- discover `*_filtered.h5`, `*_filtered.csv`, `*.h5`, then `*.csv` in that priority order;
- inspect CSV header rows until columns containing `x`, `y`, and `likelihood` are found;
- read H5 files using the DLC pandas table format when available;
- construct stable field names by joining DLC header rows, matching MATLAB's `load_dlc_csv` intent;
- coerce coordinate and likelihood columns to numeric;
- set `x` and `y` to NaN where likelihood is below threshold.

TTL synchronization:

- use the digital input channel with the most rising edges as the camera TTL channel;
- remove pulses where `diff(ttl) < (1 / video_fps) - (1 / video_fps) * pulses_delta_range`;
- TTL/frame mismatch diagnostics should report the epoch, frame/TTL counts,
  the 1-based tail indices truncated by the current alignment rule, and the
  approximate duration of the mismatch in seconds.
- apply MATLAB-compatible frame/TTL matching:
  - exact match: accept without warning;
  - absolute difference <= 2 frames: accept the aligned streams but emit a warning and record the mismatch in notes;
  - TTL has extra frames less than one second: truncate TTL to frame count and emit a warning;
  - video has extra frames less than one second: truncate tracking to TTL count and emit a warning;
  - larger mismatches: truncate to the closer stream, and keep a warning note;
  - very large mismatch can use interpolated timestamps only if no reliable TTL alignment is possible.

Velocity:

- compute speed from timestamped x/y positions with NaN-aware finite differences;
- if x/y are unavailable, fall back to linearized position if present;
- acceleration is `[0, diff(speed)]`, matching MATLAB.

Missing gap interpolation:

- identify NaN runs independently for each selected coordinate stream;
- convert run lengths to seconds using synchronized timestamps or video frame rate;
- interpolate only gaps shorter than or equal to the configured threshold;
- preserve longer gaps as NaN;
- record interpolation settings and counts in `behavior.processinginfo`.

MAT output:

- save `basename.animal.behavior.mat` containing a top-level `behavior` struct;
- write the MAT file under the local output directory;
- preserve MATLAB-style nested fields for `position`, `epochs`, and `processinginfo`;
- include source notes and the behavior settings used.

## Expected Behavior

Users can open the GUI Behavior tab, inspect detected DLC files, calibrate the recording space from a frame, clean tracking jumps, interpolate short gaps, and export a behavior MAT file without manually running MATLAB neurocode.

The first implementation should prioritize the GUI-backed DLC workflow, TTL synchronization, centimeter conversion, outlier cleaning, interpolation, and MAT output. Legacy `.whl`, OptiTrack, and manual position formats are non-goals for the first pass unless needed by existing test data.

## Verification

Planned tests:

- DLC CSV parser handles multi-row headers and numeric strings.
- DLC H5 parser handles standard DLC pandas output.
- Filtered DLC files are preferred over unfiltered files.
- Likelihood filtering replaces low-confidence coordinates with NaN.
- TTL/frame matcher reproduces MATLAB truncation behavior for equal, small positive, and small negative mismatches.
- Multiple subsessions concatenate with `MergePoints.timestamps` offsets.
- Calibration converts pixel coordinates to centimeters and stores the ratio.
- Tracker-cleaning masks set selected frames to NaN.
- Short missing gaps are interpolated while longer gaps remain NaN.
- Saved `animal.behavior.mat` exposes expected fields and shapes through `scipy.io.loadmat`.

Planned commands:

```bash
pytest -q tests/preprocess/test_behavior_neurocode_compat.py
pytest -q tests/preprocess
```

## Non-Goals

- Do not support headless centimeter export without calibration.
- Do not silently fall back to pixel output in the main GUI workflow.
- Do not implement every legacy input format from MATLAB `general_behavior_file.m` before the DLC path is stable.
