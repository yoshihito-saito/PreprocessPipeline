# Standalone Qt GUI Parity

## Date and Commit

- Date: 2026-06-16
- Commit: 63827b1

## Linked Plan

- [Implementation plan](../implementation_plan/2026-06-16-standalone-qt-gui-parity.md)

## What Changed

- Expanded the PySide6 desktop GUI in `src/preprocess/gui/app.py` to cover the
  Web GUI parameter set more completely.
- Added desktop controls for state-scoring details, artifact enable flags,
  sorter enable/disable state, postprocess noise-label-only mode, and noise
  thresholds.
- Added desktop actions for default config save/load, noise-label-only runs,
  moving local output files back to basepath, and force-stop parity with the web
  GUI.
- Made the settings tabs scrollable so the expanded parameter set remains
  usable in smaller windows.
- Added a black/dark Qt stylesheet aligned with the web GUI.
- Reworked the desktop layout to match the web GUI: settings tabs on the left,
  chanMap preview above Log in the monitor column, Setting config on the right,
  and run controls in the bottom bar.
- Reordered preprocess and postprocess controls to match the web GUI section
  order, including moving preprocess workers into Signal processing and sorter
  workers into Sorter and runtime.
- Replaced direct channel-map/probe JSON editing with probe assignment rows for
  geometry, XML groups, and x offset.
- Replaced direct noise-threshold JSON editing with individual threshold fields
  matching the web GUI.
- Replaced default-only top-bar config actions with browseable `Load config`
  and `Save config` controls.
- Prevented accidental parameter changes during scrolling by making numeric
  spin boxes and combo boxes ignore mouse-wheel events.
- Improved checkbox visibility in the black theme with explicit monochrome
  indicator styling.
- Replaced the flickery gradient checkbox marker with a red-orange check-mark
  SVG and shifted the desktop GUI palette from blue-tinted dark colors to
  neutral black/gray tones.
- Refined the PyNeuroscope-style dark theme by removing bright group borders,
  fixing white scroll-area backgrounds, using gray widget surfaces, and
  changing checked boxes to orange-filled checkboxes without an orange outline.
- Adjusted checked boxes toward a redder, lower-contrast orange.
- Moved the probe assignment editor below the `probe assignments` label so
  geometry, XML groups, and x offset have the full form width.
- Hid the direct `chanMap path` row from the main settings UI while keeping the
  internal path field for Load/Generate chanMap and config round-trips.
- Changed `include TTL offset` to default off in GUI settings and the public
  default config.
- Sanitized public GUI defaults so `local_root` and `matlab_path` are blank,
  and made the Qt config-save helpers strip those user-specific paths.
- Added automatic local default discovery through
  `config/preprocess_gui_default_config.local.json`, with
  `PREPROCESS_GUI_DEFAULT_CONFIG` still taking precedence.
- Renamed the visible desktop top-bar actions from default-only labels to
  browseable `Load config` / `Save config` actions.
- Renamed the visible local output field to `Local working dir` and moved
  `Browse local` to the left of that label.
- Made blank `local_root` configs resolve automatically to
  `<PreprocessPipeline root>/preprocess_tmp`, creating that directory as needed,
  while keeping the `Browse local` button for manual override.
- Set the public default config to leave high-amplitude artifact removal
  unchecked by default.
- Updated the private ignored default config used on this machine so it keeps
  the local output root, MATLAB path, double-sided probe assignment, and
  `include TTL offset` off.
- Made browse dialogs use non-native Qt dialogs with explicit dark styling so
  directory/file lists remain readable under the dark theme.
- Added interactive chanMap preview inspection: scroll-wheel zoom, drag
  rectangle zoom, and double-click reset.
- Removed the visible `Generate chanMap` button from the settings UI while
  keeping internal chanMap generation for runs that need a missing chanMap.
- Styled combo-box popup lists so probe geometry selection remains readable
  under the dark theme.
- Moved `overwrite preprocess outputs` into a separate `Preprocess options`
  block above the channel-map section.
- Simplified `Postprocess target` to show only `sorting folder`, added guidance
  explaining the blank/default behavior, and kept `search root` internal for
  config compatibility.
- Renamed `Apply preprocess before metrics` to `apply preprocess filter` and
  added guidance about using it only for legacy raw `basename.dat`.
- Moved `overwrite postprocess outputs` into a separate `Postprocess options`
  block matching the preprocess options format.
- Updated postprocess-only preflight so a missing `basename.dat` tells the user
  to skip/disable spike sorting, run preprocess only to create `basename.dat`,
  and then run postprocess again.
- Fixed cramped/overlapping form rows by making labels and checkboxes
  transparent inside panels and applying consistent `QFormLayout` spacing.
- Improved GUI responsiveness by debouncing preview/preflight refreshes and
  avoiding repeated chanMap reloads when the file has not changed.
- Removed the deprecated Web GUI implementation and public launch path:
  - deleted `src/preprocess/gui/web_app.py`;
  - removed the `preprocess-webgui` package entry point;
  - switched `launch_gui.py`, `launch_gui.sh`, and `launch_gui.bat` to launch
    the Qt desktop GUI;
  - kept `launch_gui.py --config ...` for config selection and removed legacy
    browser-server options from the launcher.
- Updated README so `preprocess-gui` is the recommended GUI command, with
  `python launch_gui.py` and the wrapper scripts as fallback launchers.
- Styled `QMessageBox` dialogs with the same dark black/gray palette as the
  main GUI so preflight warning text remains readable.
- Added a short explanation to the preflight warning confirmation dialog that
  warnings are non-blocking and should be reviewed before continuing.
- Shortened the run warning confirmation dialog again by removing the
  preflight-specific title and explanatory sentence. The dialog now shows only
  `Warnings`, the warning lines, and `Continue?`.
- Added `find_rhd_source()` so GUI preflight uses the same Intan `.rhd`
  discovery behavior as actual preprocess without copying files. This accepts
  `info.rhd` in direct child folders instead of warning only because
  `basepath/info.rhd` is absent, and it also accepts an existing local
  `<basename>.rhd` copy.
- Changed state-scoring figure export to use non-GUI
  `matplotlib.figure.Figure` objects instead of `matplotlib.pyplot`, avoiding
  Matplotlib GUI-thread warnings when state scoring runs inside the Qt worker
  thread.
- Fixed Qt pipeline worker cleanup so finished/failed worker signals are
  queued back to the GUI thread, worker/thread objects are cleaned up with
  `deleteLater`, and the GUI no longer calls `QThread.wait()` from inside the
  worker thread. This addresses the `QThread::wait: Thread tried to wait on
  itself` / `QThread: Destroyed while thread is still running` crash at the end
  of a run.
- Added a close guard that prevents closing the GUI while a pipeline worker is
  still running, unless the user explicitly uses the existing force-stop path.
- Added an explicit `Move outputs to basepath complete!` line to the GUI log
  after the move-back operation has fully completed successfully.
- Changed `Force stop` so it no longer calls `os._exit()` and no longer closes
  the GUI process. In the current QThread runner it warns that safe in-GUI
  force killing requires a future subprocess-based runner, keeps the GUI open,
  and writes a log entry instead of terminating Python.
- Updated README desktop GUI notes.
- Updated chanMap preview behavior so the preview is generated from the current
  GUI bad-channel and probe-assignment fields when a basepath XML is available,
  instead of waiting for the on-disk `chanMap.mat` to change.
- Made `Load chanMap` import bad-channel state and saved probe assignments from
  the loaded file back into the left settings pane.
- Fixed chanMap scroll zoom by using a box-adjusted equal aspect view and
  clipping point labels/markers to the axes.
- Colored connected chanMap contacts by probe/group while preserving red
  markers and labels for bad channels.
- Kept MATLAB path optional in the GUI and preflight. Blank MATLAB path is now
  treated as runtime auto-detection through `PATH` or the platform default
  search behavior instead of a missing-path error.
- Added an `Open config` button for sorter configs. The button opens the
  config matching the currently selected sorter, resolving repository-relative
  config paths.
- Made GUI sorter path/config collection fall back to sorter-specific defaults
  when the visible fields are blank.
- Resolved GUI sorter path/config values relative to the repository root before
  building `PreprocessConfig`, so `preprocess-gui` does not depend on the
  current working directory.
- Saved sorter config snapshots into each session-specific sorter output
  folder: a source config copy and `sorter_config_resolved.yaml`.
- Replaced the personal MATLAB path example in the missing-MATLAB error with a
  generic PATH/matlab_path message.
- Changed project-root resolution for GUI, preflight, default local output,
  and sorter-runner defaults so repository-relative sorter paths prefer the
  local `PreprocessPipeline` checkout, especially the local `sorter/` folder,
  instead of an installed `site-packages` directory.
- Added pip `direct_url.json` source discovery to project-root resolution so a
  non-editable `pip install .` console command can still find the local clone's
  `sorter/` folder when launched outside the repository directory.
- Passed `config.sorter_path` through to the Kilosort2.5 runner as well as
  KiloSort1 and Kilosort4, so all Kilosort variants use the selected local
  sorter folder.
- Included the Qt GUI `assets/*.svg` files as package data so installed
  environments can load the checkbox SVG from `site-packages`.
- Changed repository-relative GUI paths and the blank-config default local
  working directory to resolve from the discovered `PreprocessPipeline`
  repository root. Root discovery now fails clearly if no checkout with
  `sorter/` can be found instead of silently using `site-packages`.
- Changed GUI worker defaults so Windows respects the process-worker handle
  limit before applying the 128-or-detected-minus-8 rule.
- Applied the same worker default and cap rule to all GUI worker fields and
  package defaults, including preprocess, high-amplitude artifact detection,
  sorter/MATLAB workers, postprocess workers, and sorter-runner defaults.
- Capped saved GUI config worker values when loading, applying, and collecting
  settings so Windows does not keep using a saved `128` worker value.
- Replaced the explanatory Force stop warning with a concise confirmation
  dialog that offers `Stop` and `Cancel`, and only starts stop handling when
  `Stop` is selected.
- Moved GUI pipeline execution to a child process launched by `QProcess`.
  Force stop now kills that child process tree after confirmation, using
  `taskkill /T /F` on Windows, while keeping the GUI process alive.
- Added process-runner hardening: `QProcess` start-error handling, structured
  JSON error summaries from the child runner, buffered GUI log flushing for
  high-volume stdout/stderr, and non-Windows process-group termination.
- Changed worker spin boxes so they are created with the current platform's
  effective worker count as both the default value and maximum. Loading or
  collecting GUI settings now also writes normalized worker values back into
  the visible preprocess, sorter, and postprocess worker fields, preventing
  stale saved values such as `128` from being displayed when the run will use a
  lower count.
- Changed MATLAB sorter logging so each MATLAB sorter run uses a per-sorter
  runtime log and persists it to the session-specific sorter output folder as
  `matlab_run.log`. Windows MATLAB shim logs now include wrapper start/exit
  diagnostics and captured MATLAB stdout/stderr in addition to MATLAB
  `-logfile` output, so non-zero Kilosort exits should no longer point only to
  an empty temp log.
- Changed final postprocess analyzer-cache cleanup so a Windows file-lock on
  SpikeInterface memory-mapped cache files no longer marks an otherwise
  completed run as failed. The cache is left in place with a warning when
  Windows still holds `analyzer_cache` files such as `waveforms.npy`.
- Reported any analyzer-cache folders left locked by the child pipeline process
  back to the Qt GUI, and made the parent GUI retry deleting those folders
  after the child process exits. This allows Windows to release child-process
  memory-mapped file handles before the cleanup retry.
- Fixed postprocess-only GUI runs so an existing local `basename.dat` receives
  `sampling_frequency` and `num_channels` inferred from the session XML under
  `basepath`, matching the metadata that all-mode runs get from
  `PreprocessResult`.
- Hardened postprocess-only preflight and target resolution: missing session
  XML, missing chanMap, and GUI/chanMap bad-channel mismatches are now blocking
  errors for full postprocess. Blank sorting-folder selection now searches the
  local working output first and then the basepath for Kilosort results, and
  `basename.dat` uses the same local-first/basepath-fallback behavior. Existing
  postprocess output or analyzer cache now produces a warning so users know
  outputs may be reused or cache cleanup may run.
- Fixed GUI-triggered chanMap generation so it reads the session XML from the
  source basepath and writes `chanMap.mat` into the local working output
  directory. This restores chanMap generation before preprocess has copied XML
  metadata into the local working directory.
- Added a default-on `Clean local after move` checkbox to the move-back
  controls. When enabled, the GUI deletes the local session output folder after
  all selected outputs have moved successfully to the basepath, so skipped
  metadata copies and un-moved local files do not remain under `preprocess_tmp`.

## Why

The desktop GUI previously exposed only a subset of the web GUI settings. This
caused configuration drift and made PySide6 unsuitable as the primary
standalone GUI before adding behavior preprocessing and oscillatory event
detection controls.

## Verification

- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/config_model.py`
  - Passed.
- `python -m py_compile launch_gui.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/gui/preflight.py`
  - Passed.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py`
  - Passed: 3 tests.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow ..."`
  - Passed: the PySide6 window instantiated offscreen and default settings
    collected probe assignments, worker count, and noise thresholds.
- `git diff --check`
  - Passed.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow button check ..."`
  - Passed: `Load config` and `Save config` are visible, while
    `Load default` and `Save default` are no longer visible.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow ..."`
  - Passed after checkbox, layout-spacing, and debounced-refresh updates.
- `rg -n "#0a1020|#0b1220|#111827|#1d4ed8|#3b82f6|#273244|#334155|#64748b|qradialgradient" src/preprocess/gui/app.py`
  - Passed: no former blue-tinted palette entries or gradient checkbox marker
    remain in `app.py`.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow label check ..."`
  - Passed: `chanMap path` is no longer present as a visible label and default
    probe assignments still collect correctly.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... QFileDialog option/default check ..."`
  - Passed: non-native file dialog option is enabled and collected defaults are
    `artifact_ttl_include_offset=False`, `matlab_path=''`, and `local_root=''`.
- `env -u PREPROCESS_GUI_DEFAULT_CONFIG QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... local default check ..."`
  - Passed: Qt GUI auto-loaded `config/preprocess_gui_default_config.local.json`
    with local root, MATLAB path, `double_sided` probe assignment, and
    `artifact_ttl_include_offset=False`.
- `rg -n "/local/workdir/ys2375|/workdir/ys2375|MATLAB/R2024b" config/preprocess_gui_default_config.json src/preprocess/gui README.md`
  - Passed: no personal paths remain in public default config, GUI source, or
    README.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... topbar labels check ..."`
  - Passed: `Load config` and `Save config` are visible, default-labeled
    buttons are not visible, and `Local working dir` replaces `Local root`.
- `env PREPROCESS_GUI_DEFAULT_CONFIG=config/preprocess_gui_default_config.json QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... local working dir fallback check ..."`
  - Passed: public config keeps `local_root=''`, Qt displays and collects the
    repo-local `preprocess_tmp` path, creates it, and keeps `Browse local`
    visible.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow button/chanMap preview check ..."`
  - Passed: `Generate chanMap` is no longer a visible button, `Load chanMap`
    remains visible, and the preview exposes reset/zoom handlers.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow group/button check ..."`
  - Passed: `Preprocess options` and `Session and channel map` are separate
    groups, `Load chanMap` remains visible, and `Generate chanMap` is not a
    visible button.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... postprocess label/options check ..."`
  - Passed: `Postprocess options` is present, `overwrite postprocess outputs`
    and `apply preprocess filter` are present, explanatory labels are present,
    and `search root` is no longer a visible label.
- `rg -n "web_app|preprocess-webgui|src\\.preprocess\\.gui\\.web_app|Web GUI|web GUI|webgui" README.md pyproject.toml launch_gui.py src/preprocess/gui tests/preprocess config`
  - Passed: no remaining Web GUI launch/module references in the active GUI
    source, package metadata, README, tests, or config.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... launch_gui.main([...]) with fake Qt main ..."`
  - Passed: `--config` resolves to
    `config/preprocess_gui_default_config.json`, legacy browser-server options
    are ignored with a warning, and `sys.argv` passed to the Qt GUI is
    sanitized.
- `python -m py_compile src/preprocess/gui/app.py`
  - Passed after the preflight warning dialog color/readability update.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow stylesheet check ..."`
  - Passed: the Qt stylesheet contains `QMessageBox`, `QMessageBox QLabel`,
    and `QMessageBox QPushButton` rules.
- `python -m py_compile src/preprocess/io.py src/preprocess/gui/preflight.py src/preprocess/gui/app.py`
  - Passed after sharing `.rhd` discovery with GUI preflight.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_io_source_selection.py`
  - Passed: 20 tests, including child-folder `info.rhd`, existing local
    `<basename>.rhd`, and source discovery checks.
- `python -m py_compile src/preprocess/state_scoring.py`
  - Passed after replacing state-scoring `pyplot` figure creation.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_state_scoring_neurocode_compat.py`
  - Passed: 9 tests.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... ThreadPoolExecutor ... _new_saved_figure ..."`
  - Passed: saved a figure from a worker thread with no captured Matplotlib
    GUI-thread warnings.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow dummy worker run ..."`
  - Passed: a dummy GUI worker completed, `_thread` returned to `None`, and
    `force_stop` was disabled without a QThread abort.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py`
  - Passed: 5 tests.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... _move_outputs_to_basepath ..."`
  - Passed: a successful move-back action appends
    `Move outputs to basepath complete!` as the final log line.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... _force_stop_process ..."`
  - Passed: Force stop appends a "GUI remains open" log message and does not
    call `os._exit()`.
- `python -m py_compile src/preprocess/gui/app.py`
  - Passed after chanMap preview synchronization and zoom updates.
- `env QT_QPA_PLATFORM=offscreen PREPROCESS_GUI_DEFAULT_CONFIG=/tmp/nonexistent_preprocess_gui_config.json PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow chanMap preview sync ..."`
  - Passed: current GUI bad channels/probe assignments update the preview
    summary; loaded chanMap data updates left-pane bad channels and assignments.
- `env QT_QPA_PLATFORM=offscreen PREPROCESS_GUI_DEFAULT_CONFIG=/tmp/nonexistent_preprocess_gui_config.json PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... ChanMapCanvas zoom ..."`
  - Passed: scroll zoom shrinks both x and y view limits while keeping equal
    aspect.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_io_source_selection.py`
  - Passed: 20 tests.
- `git diff --check`
  - Passed.
- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/gui/preflight.py src/preprocess/sorter_runner.py`
  - Passed after sorter config/opening and snapshot updates.
- `env QT_QPA_PLATFORM=offscreen PREPROCESS_GUI_DEFAULT_CONFIG=/tmp/nonexistent_preprocess_gui_config.json PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow sorter defaults ..."`
  - Passed: selected sorter fills the matching sorter path/config and blank
    fields collect back to sorter defaults.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... _save_sorter_config_snapshot ..."`
  - Passed: source config and resolved params YAML were written to a
    session-style sorter output folder.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py`
  - Passed: 5 tests.
- `python -m py_compile launch_gui.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/gui/preflight.py src/preprocess/io.py src/preprocess/state_scoring.py`
  - Passed after removing the final legacy Web GUI launcher options.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_io_source_selection.py tests/preprocess/test_state_scoring_neurocode_compat.py`
  - Passed: 29 tests.
- `rg -n "web_app|preprocess-webgui|webgui|Web GUI|web GUI|browser|--host|--port|--no-browser" README.md pyproject.toml launch_gui.py src tests config`
  - Passed: no remaining Web GUI or browser-server launcher references in the
    checked source, package metadata, README, tests, or config paths.
- `git diff --check`
  - Passed.
- `python -m py_compile src/preprocess/paths.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/gui/preflight.py src/preprocess/io.py src/preprocess/sorter_runner.py src/preprocess/pipeline.py`
  - Passed after local project-root path resolution updates.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... find_project_root / resolve_project_path ..."`
  - Passed: repository-relative `sorter/Kilosort1_config.yaml` resolved to
    `/local/workdir/ys2375/PreprocessPipeline/sorter/Kilosort1_config.yaml`
    and existed.
- `env QT_QPA_PLATFORM=offscreen PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow._current_sorter_config_path ..."`
  - Passed: the GUI resolved `Open config` for KiloSort1 to the local
    repository `sorter/Kilosort1_config.yaml`, not `site-packages/sorter`.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py`
  - Passed: 5 tests.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_runner_matlab_path.py -k kilosort25`
  - Passed: 3 tests, 37 deselected.
- `git diff --check`
  - Passed.
- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/worker_defaults.py src/preprocess/metafile.py src/postprocess/metafile.py`
  - Passed after worker-display normalization updates.
- `env QT_QPA_PLATFORM=offscreen PREPROCESS_GUI_DEFAULT_CONFIG=/tmp/nonexistent_preprocess_gui_config.json PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "... MainWindow worker normalization ..."`
  - Passed: with a simulated Windows 40-CPU environment, visible preprocess,
    sorter, and postprocess worker fields, their maximum values, and collected
    settings all normalize to `32` when a saved or typed value of `128` is
    present.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_metafile_defaults.py tests/postprocess/test_postprocess_metafile_defaults.py`
  - Passed: 11 tests.
- `python -m py_compile src/preprocess/sorter_runner.py`
  - Passed after MATLAB sorter log persistence updates.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_runner_matlab_path.py -k "matlab_shim or persist_matlab_log or passes_matlab_max_workers or cleans_up_runtime"`
  - Passed: 5 tests, 36 deselected.
- `python -m py_compile src/postprocess/pipeline.py`
  - Passed after Windows analyzer-cache cleanup handling.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/postprocess/test_postprocess_target_resolution.py -k "delete_final_analyzer_cache"`
  - Passed: Windows locked-cache cleanup is downgraded to a warning, while
    non-Windows permission errors still raise.
- `python -m py_compile src/preprocess/gui/app.py src/preprocess/gui/run_pipeline.py src/postprocess/pipeline.py`
  - Passed after parent-GUI analyzer-cache cleanup retry updates.
- `python -m py_compile src/preprocess/gui/config_model.py`
  - Passed after postprocess-only XML metadata inference.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py`
  - Passed: postprocess config infers `sampling_frequency` and `num_channels`
    from `basepath/<basename>.xml` when `basename.dat` exists.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py tests/preprocess/test_gui_preflight.py`
  - Passed: 11 tests covering postprocess-only XML/chanMap/bad-channel errors,
    basepath fallback for Kilosort results and `basename.dat`, and warnings for
    existing postprocess outputs/cache.
- `python -m py_compile src/preprocess/io.py`
  - Passed after chanMap source/local path fix.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_chanmap_geometry.py`
  - Passed: chanMap generation reads XML from basepath and writes
    `chanMap.mat` to the local output directory.
- `python -m py_compile src/preprocess/gui/app.py`
  - Passed after adding the move-after-clean option.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_move_outputs.py`
  - Passed: move-back cleanup removes the local session folder after successful
    move.
- `python -m py_compile src/preprocess/gui/app.py`
  - Passed after adding automatic existing-chanMap loading from browsed
    basepath/local roots and probe-level preview coloring.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_move_outputs.py tests/preprocess/test_gui_config_model.py`
  - Passed: 3 tests.
- `git diff --check`
  - Passed after worker-display normalization updates.

Additional attempted checks:

- `pytest -q tests/preprocess/test_metafile_defaults.py tests/postprocess/test_postprocess_metafile_defaults.py`
  - Not run in the base environment because the unqualified `pytest` command
    resolves to Python 2.7 and cannot parse the Python 3 test syntax.
- `python -m pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_metafile_defaults.py`
  - Not run in the base Python because `pytest` is not installed.
- `env PYTHONPATH=. /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_metafile_defaults.py`
  - `test_gui_preflight.py` passed, but `test_metafile_defaults.py` failed on an
    existing worker-count expectation unrelated to this GUI change
    (`highamp_n_jobs` expected 92, observed 88).

## Result

The PySide6 GUI is now the canonical GUI entry point. It maps substantially
closer to the former web GUI's settings, layout, visual style, and run controls
while still using the shared `PipelineGuiSettings` and preflight model.

Selecting a basepath or local working directory from Browse now auto-loads an
existing `chanMap.mat` when one is found for the selected session. The GUI
updates bad channels and probe assignments when the file carries assignment
metadata, displays the loaded file until the chanMap controls are edited, and
then regenerates the preview from the edited GUI fields. Channel-map preview
colors are assigned per probe, with disconnected channels still shown
separately.

## Known Limitations and Next Steps

- Windows `.exe` packaging is not implemented yet. For now, Windows users
  should use `preprocess-gui`, `python launch_gui.py`, or `launch_gui.bat`
  from an activated environment.
