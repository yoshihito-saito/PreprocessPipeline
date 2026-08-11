# Standalone Qt GUI Parity

## Goal and Motivation

Provide a PySide6 standalone desktop GUI that exposes the same preprocessing
and postprocessing parameters and operational controls as the existing web GUI.
The desktop GUI should be the primary local interactive interface. Once the Qt
GUI has enough parity for local testing, remove the Web GUI from the public
launch surface so new GUI work targets the Qt implementation.

## Current Problem

The repository already contains two GUI entry points:

- `src/preprocess/gui/web_app.py`, which exposes the fuller parameter set and
  operational controls through a local web app.
- `src/preprocess/gui/app.py`, which provides a PySide6 desktop GUI but exposes
  only a subset of the web GUI settings and workflow actions.

This creates configuration drift. Some settings in `PipelineGuiSettings` are
not editable or round-tripped in the desktop GUI, including state-scoring
details, artifact enable flags, sorter enable state, postprocess noise-label
settings, and noise thresholds. Web-only operational actions such as default
config selection and moving local outputs back to the basepath are also missing
from the desktop GUI.

After the parity pass, keeping the Web GUI entry point active would make it
unclear which interface is canonical and would invite further duplicate GUI
work. The launcher scripts and README still describe the browser-based GUI,
even though the intended interface is now the standalone Qt GUI.

## Why This Is Needed Now

Future preprocessing features, including behavior preprocessing and oscillatory
event detection, will add more configuration surface area. Keeping the desktop
GUI behind the web GUI would increase the cost of each new feature. A parity
pass now lets both interfaces share the existing `config_model.py` and
`preflight.py` contract before adding new domains.

## Git State

Work branch: `feature/standalone-qt-gui`

The implementation starts from `main` with no local source changes.

## Affected Modules and Files

- `src/preprocess/gui/app.py`
  - Expand PySide6 controls to cover the web GUI parameter set.
  - Preserve `PipelineGuiSettings` round trips.
  - Add web-GUI-equivalent operational actions where practical.
- `src/preprocess/gui/config_model.py`
  - Keep as the shared source of truth; only adjust if Qt parity exposes a
    missing compatibility helper.
- `README.md`
  - Clarify the desktop GUI launch path and remove Web GUI launch guidance.
- `launch_gui.py`, `launch_gui.sh`, `launch_gui.bat`
  - Make repository launchers start the Qt GUI.
- `pyproject.toml`
  - Keep `preprocess-gui` as the GUI entry point and remove the Web GUI entry
    point.
- `src/preprocess/gui/web_app.py`
  - Remove the deprecated Web GUI module after Qt parity verification.
- `implementation_plan/README.md`
  - Add this plan to the index.
- `change_log/`
  - Record implementation and verification after code changes.

## Public Parameters and API Changes

No pipeline API changes are planned. The GUI should continue to map through
`PipelineGuiSettings.to_preprocess_config()` and
`PipelineGuiSettings.to_postprocess_config()`.

The `preprocess-gui` command remains the desktop entry point. The
`preprocess-webgui` command is removed.

## Design Details

The PySide6 desktop GUI will mirror the web GUI rather than introduce a new
workflow:

- Use the same defaults from `PipelineGuiSettings`.
- Use the same setting names and conversion behavior as web `collectSettings`.
- Use `run_preflight(settings, mode)` before runs and in preview.
- Generate or load `chanMap.mat` through existing `prepare_chanmap`.
- Keep run execution in a background `QThread` with stdout/stderr routed into
  the log panel.
- Add a non-web implementation of "move to basepath" using the existing web GUI
  helper logic as the behavioral reference.

`pyNeuroscope` is a design reference for PySide6 layout, file inspection, and
testable GUI/data separation. It is not planned as a runtime dependency.

Additional layout parity requirements:

- Use a black/dark visual theme aligned with the web GUI colors.
- Keep the same major spatial layout as the web GUI: settings tabs on the
  left, chanMap preview above the log in the center monitor column, setting
  preview on the right, and run controls in the bottom run bar.
- Keep preprocess sections in the same order as the web GUI, including placing
  preprocess workers in the Signal processing section and sorter workers in the
  Sorter and runtime section.
- Replace raw channel-map/probe JSON editing with explicit probe assignment
  rows for geometry, XML groups, and x offset. The GUI should still round-trip
  `probe_assignments` internally, but users should not have to edit JSON.
- Replace raw noise-threshold JSON editing with individual threshold fields
  matching the web GUI.
- Keep browseable config actions visible in the top bar:
  `Load config` and `Save config`. Startup still auto-loads the local/default
  config path.
- Prevent accidental parameter changes while scrolling by making numeric spin
  boxes and combo boxes ignore mouse-wheel events. Users should change these
  values through direct typing, arrows, or dropdown selection instead.
- Improve checkbox visibility within the black theme by styling checkbox
  indicators explicitly with high-contrast monochrome borders and checked
  state markers.
- Keep GUI interaction responsive by debouncing preflight/preview refreshes and
  avoiding repeated chanMap reloads when the resolved file has not changed.
- Use a neutral black/gray palette rather than blue-tinted dark colors, and use
  an explicit red-orange check mark for checked boxes.
- Match the PyNeuroscope dark-widget style more closely: remove bright group
  borders and white scroll-area backgrounds, use neutral gray controls, and
  show checked boxes as orange-filled checkboxes without an orange outline.
- Hide the direct `chanMap path` text field from the main GUI; users should use
  Load/Generate chanMap actions while the selected path is still stored
  internally for config round-trips.
- Public GUI defaults must not contain personal data paths, local output roots,
  or user-specific MATLAB executable paths. Default saving should strip those
  fields, while personal defaults can live in untracked user-specific files.
- `include TTL offset` should default to off.
- File/folder browse dialogs should remain readable under the dark theme by
  using Qt non-native dialogs with explicit dark styling for item views and
  headers.
- The GUI should auto-load an untracked local default config when present so
  users do not have to manually load private paths each session. Public defaults
  remain sanitized for GitHub.
- The chanMap preview should support interactive inspection: scroll-wheel zoom,
  drag-select a rectangle to zoom to that range, and double-click reset to the
  full view.
- Hide the explicit `Generate chanMap` button from the GUI. chanMap generation
  remains available internally when a run needs a missing chanMap.
- Combo-box popups must remain readable in the dark theme, including probe
  geometry selection.
- The preprocess overwrite option should be separated from the channel-map
  section and shown above it as its own option block.
- The postprocess target UI should expose only the sorting folder field with a
  short explanation. The search root stays internal for config compatibility.
- Postprocess overwrite should match the preprocess option layout, and
  `apply preprocess before metrics` should be renamed to
  `apply preprocess filter` with explanatory guidance.
- Postprocess-only preflight should give an actionable error when
  `basename.dat` is missing: skip spike sorting, run preprocess only to create
  `basename.dat`, then run postprocess again.
- The Qt top bar should use browseable `Load config` / `Save config` actions
  instead of default-only button labels, while startup still auto-loads the
  local/default config.
- The local output field should be labeled `Local working dir`, with its browse
  button placed to the left of that label.
- When `local_root` is blank in a config, automatically use and create
  `<PreprocessPipeline root>/preprocess_tmp` as the local working directory,
  while keeping the browse button available for manual override.
- Preflight warning and confirmation dialogs should use the same dark
  black/gray palette as the main GUI so warning text remains readable.
- GUI preflight should use the same Intan `.rhd` source discovery behavior as
  the preprocess runner. In particular, child-folder `info.rhd` files should be
  accepted because `ensure_rhd()` already uses them during actual preprocess.
- State scoring figure export should not use `matplotlib.pyplot` inside the Qt
  worker thread. Use non-GUI `matplotlib.figure.Figure` objects for saved
  figures so the standalone GUI does not trigger Matplotlib GUI-thread
  warnings during optional state scoring.
- Qt pipeline worker cleanup should follow the standard worker/thread signal
  pattern. Finished/failed handlers must run on the GUI thread and must not
  call `QThread.wait()` from inside the worker thread, because that can abort
  the GUI at shutdown.
- The local-output move-back action should write an explicit completion line
  to the GUI log only after all move operations and permission updates have
  finished successfully.
- Force stop should no longer exit the whole GUI process. In the current
  QThread-based execution model, avoid unsafe thread termination; keep the GUI
  open and clearly report that safe in-GUI force killing requires a future
  subprocess-based runner.
- chanMap preview should reflect the current GUI bad-channel and probe
  assignment fields immediately, without requiring the on-disk `chanMap.mat`
  file to change first. Loading an explicit chanMap file should also update the
  hidden stored path and the left-side fields when the file carries assignment
  metadata. Preview zoom should change the data view consistently, and contacts
  should be colored by XML group/shank while keeping bad-channel markers red.
- Sorter setup should be mostly automatic from the selected sorter. MATLAB path
  stays optional and is resolved from the environment at runtime. Sorter path
  and sorter config should default from repository sorter settings, the GUI
  should provide an `Open config` action for the selected sorter config, and
  each sorter run should save the source and resolved sorter config under the
  session-specific sorter output folder.
- MATLAB sorter diagnostics should be saved under the session-specific sorter
  output folder. On Windows, the MATLAB shim should also write wrapper
  start/finish diagnostics and captured stdout/stderr to the MATLAB log so a
  Kilosort non-zero exit does not leave an empty temp log.
- Repository-relative paths must resolve to the local `PreprocessPipeline`
  checkout, not the installed `site-packages` package directory. This matters
  for Kilosort because the sorter implementations and sorter config YAML files
  live in the local repository `sorter/` folder. Root discovery should prefer
  `PREPROCESS_PIPELINE_ROOT`, the current working directory, and the pip
  install source recorded in `direct_url.json` before falling back to the
  source file location. All Kilosort variants, including Kilosort2.5, should
  receive the resolved local sorter path from the pipeline config.
- GUI image assets used by the Qt stylesheet, including the checkbox SVG, must
  be included as package data so non-editable `pip install .` installs can load
  them from `site-packages`.
- Relative GUI paths should resolve from the discovered `PreprocessPipeline`
  repository root, not from the installed package directory. Root discovery
  should prefer the pip install source metadata and explicit environment
  override, should work from any launch directory, and should fail clearly
  rather than silently falling back to `site-packages`.
- Default GUI worker counts should show and use 128 when the platform can use
  128 workers. If the detected usable worker capacity is smaller, the GUI
  should show and use detected capacity minus 8. On Windows, the detected
  usable process-worker capacity must respect the multiprocessing handle limit
  rather than exposing 128.
- The same worker rule should apply to all GUI worker fields and package
  defaults, including preprocess, high-amplitude artifact detection,
  sorter/MATLAB workers, postprocess workers, and sorter-runner CLI defaults.
  Saved GUI configs with larger values should be capped to the current
  platform's effective worker default when loaded or collected by the GUI.
- Worker spin boxes should display the same capped value that the pipeline will
  use. Saved configs may still contain old `128` values, but after loading or
  before starting a run the visible GUI fields should be written back to the
  normalized effective count so the log and GUI do not disagree.
- Force stop should first show a concise confirmation dialog with an explicit
  Stop action. Stop behavior remains scoped to the current Qt runner and should
  avoid explanatory text in the normal confirmation path.
- Pipeline execution should run in a child process owned by the GUI. If a
  worker pool is already wedged, Force stop should terminate that child process
  tree after confirmation while keeping the GUI process alive. On Windows this
  should use `taskkill /T /F` for the child pipeline PID so `joblib`/`loky`
  worker processes do not keep running.
- The child-process runner should handle process start errors, structured
  child exceptions, high-volume stdout/stderr, and non-Windows process-group
  termination so Force stop is reliable across platforms while the GUI remains
  responsive.
- On Windows, final postprocess cleanup must not fail the whole run when
  `analyzer_cache` contains memory-mapped SpikeInterface files that remain
  locked after Phy export and noise labeling. The pipeline should still fail
  on cleanup errors that happen before required outputs are produced, but
  cache deletion after successful postprocess output generation should degrade
  to a warning and leave the cache folder in place for later manual cleanup.
- Because the Qt GUI runs the pipeline in a child process, any analyzer cache
  that remains locked inside the child should be reported back to the parent
  GUI. After the child process exits, the parent GUI should retry deleting
  those cache folders. This preserves successful run status while still
  cleaning temporary files when the lock is released by process exit.
- Postprocess-only runs should pass the recording metadata needed to read an
  existing `basename.dat`. When the GUI resolves `local_root/<basename>/<basename>.dat`
  without a fresh `PreprocessResult`, it should infer sampling frequency and
  channel count from the session XML under `basepath` so duplicate removal,
  merge/split, metrics, and Phy export can run from the saved binary.
- Postprocess-only preflight should treat missing session XML, missing
  chanMap, and mismatches between GUI bad-channel fields and chanMap connected
  channels as blocking errors for full postprocess. When the sorting folder is
  not explicitly selected, the GUI should search local working output first and
  then the basepath for Kilosort results; `basename.dat` should follow the same
  local-first, basepath-fallback behavior. Existing `_spi` outputs or analyzer
  caches should remain allowed but should produce a clear warning when they may
  be reused or cleaned.
- GUI-triggered chanMap generation must read the authoritative session XML from
  `basepath/<basename>.xml` and write `chanMap.mat` into the local working
  output directory. It must not require the XML to have already been copied
  into the local working directory before preprocess starts.
- The move-back action should offer a default-on `Clean local after move`
  option. After all selected outputs are moved successfully to the basepath,
  this option should delete the local session output folder so skipped local
  metadata copies and any intentionally un-moved local files do not remain in
  `preprocess_tmp`.

## Expected Behavior

The desktop GUI should let the user configure and run the same pipeline modes as
the web GUI:

- run all
- run preprocess only
- run postprocess only

The desktop GUI should expose the same core parameter groups:

- basepath/local output/default config
- preprocess core settings
- LFP and state scoring settings
- artifact removal settings
- chanMap/probe assignment settings
- sorter settings
- postprocess merge/split/duplicate settings
- noise-label thresholds

The desktop GUI should also support:

- loading and saving config JSON files
- generating/loading chanMap previews
- moving selected local output artifacts back to basepath
- clearing logs and previewing preflight checks
- scrolling through parameter panes without changing numeric or select values
  under the cursor
- loading an existing `chanMap.mat` when either basepath or local working
  directory is selected from Browse, then synchronizing bad channels, probe
  assignments when metadata is present, and the preview from that file while
  still leaving the GUI fields editable for manual corrections
- coloring the channel-map preview by probe, while keeping disconnected/bad
  channels visually distinct

Repository launchers should also start the same Qt GUI:

- `python launch_gui.py`
- `./launch_gui.sh`
- `launch_gui.bat`

Legacy Web GUI command-line options should be removed from `launch_gui.py` so
the launcher exposes only the standalone Qt GUI surface.

## Verification

Planned checks:

- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/config_model.py`
- `python -m py_compile launch_gui.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py`
- targeted pytest for GUI configuration/preflight if available
- instantiate `MainWindow` in offscreen Qt mode and round-trip default
  `PipelineGuiSettings`
- verify `preprocess-webgui` is no longer registered and no repository docs
  advertise `src.preprocess.gui.web_app`
- verify implementation documentation indices and change log links

## Non-Goals

- Do not migrate to PyQt6; continue using PySide6.
- Do not implement behavior preprocessing or oscillatory event detection in this
  change.
- Do not add pyNeuroscope as a package dependency.
- Do not redesign the GUI as a signal viewer; viewer improvements can be a
  later pass.
