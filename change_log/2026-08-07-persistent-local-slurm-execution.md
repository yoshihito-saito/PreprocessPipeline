# Persistent Local and Slurm Execution Framework

## Status

- Date: 2026-08-07
- Live Slurm verification: 2026-08-09
- Ephys GUI layout follow-up: 2026-08-09
- Narrow Run-page layout follow-up: 2026-08-10
- MATLAB completion-detection follow-up: 2026-08-10
- Operator-focused Run UI follow-up: 2026-08-10
- Common local-session resume and compact-layout follow-up: 2026-08-10
- Legacy provenance migration follow-up: 2026-08-11
- Commit: uncommitted working-tree implementation
- Plan: [persistent Local and Slurm execution framework](../implementation_plan/2026-08-07-persistent-local-slurm-execution.md)

## What Changed

- Fixed the old-to-new input-provenance migration used to establish a
  preprocess output contract. Expanded acquisition sidecars may now be added
  to a legacy snapshot only when all previously recorded raw inputs still
  match and each newly tracked file's modification and change times predate the
  immutable prior Run cutoff; later input changes and timestamp-preserving late
  copies still fail closed. New snapshots record `scan_started_at` before
  enumeration and bind `ctime_ns` as well as size and `mtime_ns`. Contract and
  adoption trust now re-enumerate live acquisition inputs and require the path
  set and metadata to match the current Run snapshot, detecting additions,
  removals, or modifications while a Run waits for its worker.
- Persisted prior Run lineage in `run.json` and the session claim so a later
  retry does not lose the evidence chain when the mutable claim is replaced.
- Restricted explicit multi-day acquisition provenance to selected subepochs;
  unselected sibling recordings no longer participate in resume identity.
  Existing broad-scope contracts migrate atomically only through matching
  prior-Run hash, session, stage fingerprint, and live selected-input checks.
- Fixed the live `run-20260811T175009Z-9bdca3bb` state-scoring failure caused by
  the undefined `save_files` name in EMG reuse. A valid existing EMG MAT is now
  validated and reused under `overwrite=False`.
- Made Load config start in the installed PreprocessPipeline `config/`
  directory instead of the GUI process working directory.
- Fixed `run-20260811T184917Z-f6ceb3fa`, where the approximately 5.38-GiB
  `SleepScoreLFP` struct exceeded MATLAB v5 serialization limits. Nested MAT
  payloads now remain v5 at ordinary sizes and automatically publish as
  MATLAB-compatible v7.3/HDF5 near the v5 limit. Shallow validation avoids
  loading the multi-GiB payload merely to confirm completion; real reuse still
  loads the full scientific data.

- Added versioned Run, immutable AnalysisConfig, ExecutionConfig, ResourceSpec,
  AttemptSpec, JobRef, and backend-status models.
- Added atomic persistent Run storage with append-only Attempt facts, backend
  observations, short locking, and regenerable `state.json`.
- Added one backend-independent Stage worker for preprocess, sorting, and
  postprocess. Completion now requires output validation in addition to process
  success.
- Extracted the existing sorter block into a reusable sorting Stage while
  preserving its scientific parameters and existing pipeline entry point.
- Added detached Local execution and Slurm execution with capability checks,
  exact resource scripts, upfront `afterok` dependencies, `sbatch --parsable`,
  `squeue`/`sacct` reconciliation, `scancel`, and no Local fallback after a
  Slurm submission attempt.
- Added Run- and Stage-scoped cancellation, lost-state recovery, immutable retries, upstream-result
  reuse, new downstream Attempts, persisted logs, scripts, job IDs, provenance,
  sorter-config checksum verification, and resource telemetry observations.
- Serialized controller mutations with a per-Run action lock, cancel active
  superseded/downstream jobs before retry, and keep ambiguous or unknown Slurm
  outcomes out of the normal retry path to prevent duplicate execution.
- Added an **Ephys > Run** page that owns Run all, Preprocess only, Postprocess
  only, backend selection, per-Stage resources, Run
  reopen/reconcile, Attempt history, cancellation, resource retry, and persisted
  log viewing. Persistent Runs do not block GUI close, and no duplicate run
  buttons remain outside the Run page.
- Made explicit Slurm the initial backend for a fresh or legacy GUI
  configuration without a backend choice when the required Slurm client
  commands are installed. Controller reachability does not weaken that server
  default, so a scheduler outage fails explicitly rather than risking an Auto
  Local run. Saved backend choices remain authoritative.
- Replaced the initial manual workspace acknowledgement with a workspace derived
  from Local working dir and a compute-node read/write guard in every Slurm job;
  obvious node-local temporary paths still warn before submission.
- Simplified GUI resources to CPU, RAM, optional walltime, and a read-only fixed
  GPU count. Scheduler-specific fields and MATLAB path are no longer shown;
  existing configuration models remain readable and MATLAB resolves from PATH.
- Added a partition-default walltime represented by `None`; Slurm omits
  `--time`, while custom positive limits and legacy configurations keep exact
  minute semantics. The current `regular` partition makes the default unlimited.
- Reflowed Run scope, Stage resources, and Run monitor for the 500-pixel
  settings pane. The three scope buttons no longer clip horizontally, each
  Stage uses a compact two-row resource layout, and Postprocess is reachable by
  ordinary vertical scrolling. Compact CPU, RAM, and walltime controls retain
  label buddies and Stage-specific accessible names.
- Simplified the normal Run UI around operator-facing concepts. Walltime is no
  longer shown and GUI Runs always use the scheduler default. Resources are
  labeled `CPU cores`, `Memory (GiB)`, and `GPUs`; Sorting shows `1 (auto)` with
  help explaining that Slurm selects the available GPU ID, while the other
  Stages request zero GPUs.
- Replaced the dense Attempt table and Reopen/Reconcile/Resume/Stage-action
  buttons with a three-row Preprocess/Sorting/Postprocess status view. Active
  Slurm Runs reconcile automatically, a restarted GUI reconnects through the
  session's active-Run claim, and advanced recovery remains available through
  the persistent controller CLI rather than the normal GUI.
- Made the existing bottom `Force stop` the sole visible pipeline cancellation
  action. It now cancels all jobs in a persistent Local/Slurm Run while keeping
  its prior process-tree behavior for legacy GUI-owned tasks.
- Made confirmed cancellation terminal in the compact status and disabled
  `Force stop` after confirmation even when Local backend observation is
  unavailable. Switching Basepath/Local working dir now rebinds monitoring—and
  synchronously revalidates before Force stop—to that session's claim, so status
  and cancellation cannot target a previously selected session.
- Made reconnected Local cancellation fail closed when persisted process-start
  identity cannot be verified. The originating backend may still cancel a child
  it directly owns; a reopened controller records cancellation failure instead
  of signalling a reused PID or falsely confirming that the worker stopped.
  Local TERM-to-KILL escalation now watches the complete verified process group,
  so an early-exiting worker leader cannot leave a TERM-ignoring MATLAB/shim
  descendant alive under a false cancellation confirmation.
- Streamed selected Attempt stdout/stderr and the active Kilosort log into the
  main GUI Log using byte offsets so refreshes neither wait for Stage completion
  nor repeat prior content. Local and Slurm workers now set
  `PYTHONUNBUFFERED=1`; the GUI Log is bounded while persistent files remain the
  complete record.
- Added dark item-view/header styling, reduced fixed chanMap preview minimums,
  made the center preview scrollable, reflowed the bottom action bar, and fit
  initial window dimensions to the available screen. The complete window now
  fits the 800x800 offscreen VNC-sized display used by the regression test.
- Renamed the final copy action to `Move outputs to storage`, added a visible
  destination and `Browse save dir`, and made the default destination Basepath
  for single-day sessions. Multi-day output now defaults to a newly created
  `<Basepath parent>/<Multi-day name>` directory rather than the first raw day.
  An explicitly browsed directory becomes both the exact destination and the
  displayed Basepath, while the original Local move source is retained safely.
- Reused the same runtime-only multi-day staging and chanMap preparation in both
  legacy and persistent preprocess entry points.
- Read sorter configuration once and preserved that byte sequence for both its
  checksum and immutable snapshot, so CRLF input remains checksum-identical and
  a concurrent source edit cannot make a newly created Run self-inconsistent.
- Isolated each POSIX Kilosort Attempt in its own immutable shim/startup
  directory and MATLAB Processes job storage, and invoked startup only in the
  parent MATLAB process. Shell-generated paths are quoted as data. This prevents
  concurrent Attempt configuration races, child-worker startup recursion, and
  stale global MATLAB job queues from delaying a new Attempt.
- Isolated POSIX MATLAB stdin/stdout/stderr from SpikeInterface's captured
  launcher pipes. A longer-lived MathWorks ServiceHost can no longer retain a
  pipe writer and leave the Sorting worker blocked in `pipe_read` after MATLAB
  and Kilosort have finished. Successful runs unlink the temporary stdio file;
  a non-zero MATLAB exit first appends launcher diagnostics to the existing
  MATLAB log.
- Documented backend policy, shared-storage requirements, persistence, retry,
  cancellation, and direct CLI entry points in `README.md`.
- Preserved the existing GUI workflow while allowing its Basepath field to
  recognize a processed local session, recover the original raw source from
  final or legacy metadata, and resume at the first invalid or incompatible
  Stage when overwrite is disabled.
- Routed postprocess-only execution through the persistent controller and added
  a cross-workspace per-session claim so two Runs, Phy, or CellExplorer cannot
  mutate the same session concurrently. Phy and CellExplorer remain Local
  interactive processes with GPU visibility disabled.
- Added Stage-specific parameter fingerprints and deep legacy validation for
  preprocess binaries/metadata, Phy sorting arrays, and postprocess exports.
  Adopted outputs are recorded explicitly; incomplete output is rerun without
  mixing old and new Stage artifacts.
- Published new `.dat` and `.lfp` files through validated, same-filesystem
  atomic replacement; sorting retries use timestamped directories; and
  postprocess overwrite versions every prior result instead of deleting a valid
  or manually curated output before its replacement succeeds.
- Added compact successful-session finalization with `preprocess_run.yaml` and
  `preprocess.log`, retained `kilosort.log`, suppressed the legacy
  `preprocessSession.log`, bounded live observation JSON, and removed only
  proven-redundant helper logs after conclusive success while retaining the
  exact Slurm submission script and intent. Failure,
  cancellation, and lost/ambiguous state keep detailed diagnostics.
- Tightened the final review fixes: successful cleanup now touches only the
  selected successful Attempts and retains earlier retry diagnostics;
  incompatible postprocess Attempts force a real rerun; legacy adoption needs
  positive parameter/input evidence; raw-input file metadata participates in
  persistent reuse; and completed Stage facts from a failed persistent Run can
  be adopted by its successor.
- Made Phy/CellExplorer exclusion atomic by holding the same session claim for
  the lifetime of the interactive process, and made claim safety inspect every
  Attempt with an unconfirmed backend job rather than only the Run headline.
  Postprocess-only finalization now merges prior preprocess/sorting Stage facts
  instead of replacing the canonical session history.
- Bound compatibility to concrete XML, chanMap, raw/sorting/dat input identities
  and reverify those recorded identities immediately before Stage execution.
  Persistent postprocess-only normalization preserves a partition manifest
  search root, so every listed sorting partition remains in scope.

## Real-data Local Run

The complete detached Local workflow was run against:

```text
/fs/cbsuruizfs1/storage/ys2375/Test data/test_rec_260509
```

The source recording remained read-only. Run state and outputs were isolated at:

```text
/local/workdir/ys2375/PreprocessPipeline/preprocess_tmp/persistent_e2e_20260807/
```

Final Run:

```text
runs/.pipeline/run-20260807T204609Z-35686b09
```

Result: `completed`. Preprocess Attempt 1 produced and validated the 128-channel,
20 kHz recording outputs. Sorting retry reused that completed Attempt without
rerunning preprocessing; Sorting Attempt 5 completed Kilosort1 and validated its
Phy artifacts. Postprocess Attempt 5 completed with 158 final units and 209,401
final spikes. All three selected Attempts have a validated `result.json` and a
recorded Local backend exit code of zero. Earlier failed/cancelled Attempts were
retained as immutable recovery evidence.

## Real-data Slurm Run

The same recording was rerun from a clean output directory with every Stage
submitted through Slurm. The source recording remained unchanged. Run state and
outputs are isolated at:

```text
/local/workdir/ys2375/PreprocessPipeline/preprocess_tmp/persistent_e2e_slurm_20260809/
```

Final Run:

```text
runs/.pipeline/run-20260809T195213Z-eecc3d26
```

The controller submitted Preprocess job 348, Sorting job 349 with
`--gres=gpu:RTXP6000:1 --dependency=afterok:348`, and Postprocess job 350 with
the persisted upstream dependencies. All three jobs completed on their first
Attempt with Slurm exit code `0:0`, validated `result.json` files, and terminal
`sacct` observations:

- Preprocess: 1:49 elapsed, 6,212,560 KiB MaxRSS;
- Sorting: 2:04 elapsed, 37,947,776 KiB MaxRSS;
- Postprocess: 0:48 elapsed, 3,091,220 KiB MaxRSS.

Sorting used scheduler-assigned GPU 0 while GPU 1 remained idle. Postprocess
produced 123 final units and 204,609 final spikes. Persisted logs contained no
traceback, exception, OOM, segmentation fault, fatal error, or validation error.
The only warning was Kilosort's existing recommendation to whiten 32 or fewer
channels when the configured partition whitened 64 channels.

## Live Completion-detection Defect and Recovery

GUI Run `run-20260810T104540Z-517c51ec` reproduced a POSIX descriptor-inheritance
failure after Kilosort wrote its required Phy artifacts. Sorting job 353 still
reported `RUNNING`, its Python worker was blocked in `pipe_read`, and MATLAB had
already exited. MathWorks ServiceHost PIDs 1767373 and 1768919 retained the
write end of the worker's captured pipe (`pipe:[3381778690]`), preventing EOF,
`result.json` publication, and release of dependent Postprocess job 354.

Because this was a disposable test Run, jobs 353 and 354 were cancelled with
`scancel`. Both jobs disappeared from `squeue`, and the Sorting worker plus the
two Run-associated ServiceHost processes exited. The scientific Kilosort output
was left in place for diagnosis; the cancelled Run was not promoted as a
successful persistent result.

## Verification

Commands run with `/workdir/ys2375/miniforge3/envs/phy2/bin/python`:

```text
python -m py_compile src/execution/*.py src/execution/backends/*.py \
  src/preprocess/runtime_prep.py src/preprocess/sorting_stage.py \
  src/preprocess/gui/config_model.py src/preprocess/gui/run_pipeline.py \
  src/preprocess/gui/app.py src/preprocess/gui/preflight.py
```

Result: passed.

```text
python -m pytest -q tests/execution
```

Result: 27 passed. This includes persistent
facts/state, exact 128-CPU resources, GPU policy, backend selection, Slurm
scripts/status/submission ambiguity, cancellation, retry dependencies, Local
process recovery, concurrent-controller idempotency, terminal unknown/lost
recovery, stronger output validation, immutable sorter snapshots, and isolated,
shell-safe POSIX MATLAB shims.

```text
python -m pytest -q \
  tests/execution \
  tests/preprocess/test_gui_config_model.py \
  tests/preprocess/test_gui_preflight.py \
  tests/preprocess/test_sorter_partitions.py \
  tests/preprocess/test_pipeline_mixed_sources.py \
  tests/preprocess/test_multiday.py \
  tests/postprocess/test_postprocess_target_resolution.py \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_uses_kilosort4_output_prefix \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_passes_matlab_max_workers_to_sorter \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_uses_basename_dat_for_sorter_and_marks_preprocessed_input \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_zeroes_and_excludes_bad_channels_for_sorter \
  tests/preprocess/test_sorter_runner_matlab_path.py::test_inject_matlab_shim_windows_creates_bat \
  tests/preprocess/test_sorter_runner_matlab_path.py::test_execute_sorting_job_passes_matlab_max_workers_to_shim
```

Result: 94 passed. This final focused run covers execution, GUI
config/preflight, multi-day staging, sorter partitions, mixed sources,
postprocess target resolution, the four existing sorter-equivalence cases, and
the parent-only/attempt-isolated MATLAB shim behavior.

Follow-up resume/output-retention verification:

```text
python -m pytest -q \
  tests/execution \
  tests/preprocess/test_gui_config_model.py \
  tests/preprocess/test_gui_preflight.py \
  tests/preprocess/test_sorter_partitions.py \
  tests/preprocess/test_pipeline_mixed_sources.py \
  tests/preprocess/test_multiday.py \
  tests/preprocess/test_recording_selected_transform.py \
  tests/preprocess/test_session_kilosort_folder_detection.py \
  tests/postprocess --ignore=tests/postprocess/test_ks4_diagnostics.py
```

Result: 138 passed. This includes processed-session recognition and source
recovery, validated Stage adoption, fingerprint mismatch reruns,
cross-workspace session exclusion, persistent postprocess-only Runs, successful
log compaction, atomic output replacement, and curated `_spi` preservation.

```text
python -m pytest -q tests/preprocess
```

Result: 217 passed and the same 17 pre-existing failures listed below.

```text
python -m pytest -q tests/preprocess/test_multiday.py
```

Result: 15 passed.

```text
python -m pytest -q \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_uses_kilosort4_output_prefix \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_passes_matlab_max_workers_to_sorter \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_uses_basename_dat_for_sorter_and_marks_preprocessed_input \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_zeroes_and_excludes_bad_channels_for_sorter
```

Result: 4 passed.

```text
python -m pytest -q tests/postprocess --ignore=tests/postprocess/test_ks4_diagnostics.py
```

Result: 29 passed.

```text
QT_QPA_PLATFORM=offscreen python -c "... MainWindow(); ..."
```

Result: passed at the time of the initial implementation; the later GUI layout
follow-up moved these controls under **Ephys > Run**.

```text
QT_QPA_PLATFORM=offscreen /local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/execution tests/preprocess/test_gui_config_model.py tests/preprocess/test_gui_preflight.py tests/preprocess/test_gui_move_outputs.py
```

Result after the selectable-storage follow-up: 82 passed. This constructs the
final **Ephys > Run** hierarchy, verifies that all three execution-scope buttons
are contained there, and covers the fresh or legacy-config server Slurm default
without replacing an explicitly saved Auto choice. It covers hidden GUI
walltime with omitted Slurm directives, fixed generic one-GPU requests,
compute-node workspace guards, resource validation, and controller behavior.
It also reproduces the original captured-pipe hang with a fake MATLAB
descendant, verifies prompt shim completion after the fix, and confirms that
non-zero launcher diagnostics remain available without a successful-run stdio
artifact. The GUI checks readable three-Stage status rows, incremental main-Log
progress without duplication, Kilosort log discovery, automatic active-Run
reconnection, persistent Force stop routing, Local/Slurm unbuffered output,
terminal cancellation display, session-switch rebinding, synchronous stop-target
validation, and fail-closed reconnected Local cancellation through the controller,
including escalation when the worker leader exits before a TERM-ignoring child,
plus single-day, multi-day sibling, and explicitly browsed storage destinations,
zero horizontal overflow in the Run page, Postprocess reachability by vertical
scrolling, and fitting the complete window within an 800x800 available screen.

```text
sbatch --test-only --gres=gpu:1 --cpus-per-task=1 --mem=1G --wrap=true
```

Result: accepted by the current `regular` partition without creating a queued
job, confirming both the generic whole-GPU request and an omitted walltime.

```text
git diff --check
```

Result: passed.

## Existing Suite Limitations

- `tests/preprocess`: 217 passed and 17 failed. The failures are in pre-existing,
  unchanged behavior/tests: artifact-removal backend expectations, legacy
  artifact configuration keyword names, a pre-existing save-raw expectation,
  high-amplitude worker-count semantics, and Kilosort4 helper/CLI expectations.
  The four existing tests that directly cover the extracted sorter path pass.
- `tests/postprocess` cannot collect `test_ks4_diagnostics.py` because
  `src.postprocess.ks4_diagnostics` is absent in the current baseline. The other
  29 postprocess tests pass.
- Live Slurm execution was verified on the single-node `regular` partition with
  `gpu:RTXP6000:1`. The cluster currently has `ConstrainDevices=no`, so Slurm
  logically reserves the GRES but does not hard-isolate GPU devices from
  processes launched outside Slurm; the live test was intentionally started
  only after both physical GPUs were idle.
- Ruff was not available in the selected environment (`No module named ruff`).
- The real-data workflow was exercised with the installed `phy2` Python
  environment and MATLAB R2024b. Portability to another Python environment still
  depends on installing the project's runtime dependencies there.
- The latest resume, atomic manual-lease, postprocess replacement,
  multi-partition normalization, and pre-execution artifact-identity guards have
  unit coverage but were not rerun as a new live Local/Slurm real-data workflow;
  the recorded live Slurm run predates this final hardening.
- The MATLAB descriptor-inheritance fix reproduces the live failure with a
  process-level regression test, but has not yet been followed by a new
  end-to-end real-MATLAB Slurm Run.
- The compact Run UI and live main-Log tailing have offscreen process-level
  coverage but have not yet been exercised through a new interactive VNC Slurm
  Run after this follow-up.

## Independent Review

The requested `~/.codex/agents/reviewer.toml` role, read-only policy, high
reasoning, and scientific/correctness priorities were applied. Its exact
`gpt-5.6` model name was unavailable for this ChatGPT account, so the audit was
rerun read-only with `gpt-5.6-sol` at high reasoning. After addressing its
findings on ambiguous submission, sacct state classification, retry/cancel job
coverage, controller locking, validation, reconciliation telemetry, workspace
confirmation, multi-day provenance, resume compatibility, output retention,
manual leases, and pre-execution input identity, the final 5.6-sol re-review of
the current tree reported no findings.

The 2026-08-10 completion-detection follow-up received a separate read-only
`gpt-5.6-sol` high-reasoning review. It reported no findings in descriptor
lifetime, shell quoting, cleanup, failure-log retention, concurrency isolation,
or regression-test validity. The reviewer identified only the documented lack
of a post-fix live Slurm rerun as a residual verification gap.

The operator-focused GUI follow-up received a final read-only `gpt-5.6-sol`
high-reasoning review. Its cancellation-status, session-rebinding, unverifiable
Local identity, and process-group descendant findings were fixed with controller
and real-process regressions. The final re-review reported no remaining findings;
the focused suite passed all 79 tests and `git diff --check` remained clean.

## 2026-08-10 Common Local-session Resume Follow-up

### What changed

- Added one **Browse local session to resume** action for both single-day and
  multi-day scientific output folders. Operators no longer need to browse the
  original file-server directories merely to reopen a local session.
- Added fail-closed recovery from `.pipeline-active-run.json`, including exact
  output-directory association, immutable AnalysisConfig hash validation, and
  validated ExecutionConfig restoration. A malformed active marker is reported
  instead of silently falling back to an older completion record.
- Added completed-session recovery from `preprocess_run.yaml` when no active
  marker exists. Both recovery paths restore scientific settings, multi-day
  source order and selected subepochs, backend/resources, and derived Stage
  worker counts while binding the chosen directory as the Local output.
- Claim-backed Runs reconnect and request reconciliation only. Browsing never
  submits or retries work and does not change overwrite or scientific settings.
- Reduced the base GUI font to 11 px and hints to 10 px, tightened control
  padding, and changed the initial settings pane from 520 px to 440 px. Resource
  rows were compacted enough to retain zero horizontal overflow at a 1200 px
  window width, and the resume button retains its full natural caption width.
- Documented the exact operator workflow in the README.

### Verification

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/preprocess/test_gui_config_model.py tests/execution/test_models_store.py tests/execution/test_session_resume.py
```

Result: 43 passed. Coverage includes active multi-day recovery, completed
single-day recovery, malformed-marker precedence, output mismatch rejection,
atomic GUI failure, Run reconnection without retry, compact fonts, full button
caption width, and zero Run-page horizontal overflow.

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution tests/preprocess/test_gui_config_model.py tests/preprocess/test_gui_preflight.py tests/preprocess/test_gui_move_outputs.py && /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/gui/config_model.py src/preprocess/gui/app.py && git diff --check
```

Result: 88 passed after restoring the shared `ResourceSpec` import identified
by the first broad run; Python compilation and `git diff --check` also passed.
The successful rerun covers the persistent execution suite plus GUI settings,
preflight, and move-output behavior.

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c 'from pathlib import Path; from src.preprocess.gui.config_model import load_local_session_resume; p=Path("/local/workdir/ys2375/PreprocessPipeline/sorting_temp/multiday_Day14_to_Day217"); r=load_local_session_resume(p); s=r.settings; print("source",r.metadata_source); print("run_dir",r.run_dir); print("name",s.multi_day_name); print("days",len(s.multi_day_session_paths)); print("subepochs",len(s.multi_day_selected_subepoch_paths)); print("backend",s.execution.requested_backend); print("workers",s.preprocess.preprocess_worker_count,s.preprocess.sorter_worker_count,s.postprocess.worker_count); print("output",s.local_output_dir)'
```

Result: recovered the claim-backed failed Run
`run-20260810T214402Z-3347d285`, name `multiday_Day14_to_Day217`, 60 source
sessions, 51 selected subepochs, Slurm backend, worker counts 128/128/128, and
the exact selected Local output directory. This diagnostic was read-only and
did not reconcile, submit, or retry the Run.

### Limitations

- Only sessions created by the persistent execution framework have the metadata
  needed for full recovery. A local folder without either supported marker is
  rejected and its current GUI state is left unchanged.
- The visual changes were verified offscreen; they have not yet been inspected
  interactively through VNC on every display scaling configuration.

## 2026-08-11 Terminal Backend Claim-release Follow-up

### What changed

- Fixed session exclusion after a conclusive backend failure. A submitted and
  started Attempt with a terminal observation and known unsuccessful outcome no
  longer remains active solely because an abrupt OOM kill prevented the worker
  from writing `failure.json`.
- Applied the same conclusive-terminal definition used by persistent
  observation storage: terminal plus a known success value, or an explicit
  cancelled/canceled state.
- Preserved fail-closed behavior for submitted/running work, terminal success
  without a validated Stage result, and lost/unknown outcomes.
- Kept Run records and `.pipeline-active-run.json` immutable during diagnosis.
  A normal subsequent `create_run` replaces the terminal claim atomically and
  records the old Run as `previous_run_dir`; no manual marker edit is needed.
- Documented the claim-release behavior in the README.

### Verification

The regression was first run before the source fix and reproduced the reported
`Another persistent Run still owns this session output` failure despite a
derived failed OOM state. After the fix:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution/test_session_resume.py
```

Result: 13 passed. The suite covers claim replacement after a terminal OOM
without `failure.json`, preservation of `previous_run_dir`, active claim
exclusion, manual leases, and lost/unknown blocking.

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution
```

Result: 48 passed.

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution tests/preprocess/test_gui_config_model.py && /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/execution/session.py && git diff --check
```

Result: 75 passed; Python compilation and `git diff --check` also passed.

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c 'from pathlib import Path; from src.execution.session import _claim_blocks_new_run; from src.execution.store import read_json; p=Path("sorting_temp/multiday_Day14_to_Day217/.pipeline-active-run.json"); c=read_json(p); print("run_id", c["run_id"]); print("blocks_new_run", _claim_blocks_new_run(c))'
```

Result for the unmodified real session marker: Run
`run-20260811T045641Z-0de3bbc5`, `blocks_new_run False`. Slurm job inspection
had already confirmed that jobs 360 and 361 were absent from the queue; the
only matching local process was the GUI itself.

### Limitations and next steps

- An already-running GUI process may have imported the pre-fix module and must
  be restarted once to load this code change.
- This change permits a new Run; it does not prevent OOM or automatically retry
  an immutable prior Run. The operator must select the intended Run scope and
  updated resources in the GUI.

## 2026-08-11 Immutable-output and transactional-resume follow-up

### Status

Date: 2026-08-11. The work is uncommitted.

Implementation plan: [persistent Local/Slurm execution plan](../implementation_plan/2026-08-07-persistent-local-slurm-execution.md#2026-08-11-follow-up-immutable-outputs-and-transactional-resume).

### What changed

- Removed the Persistent preprocess and postprocess worker overrides that
  silently changed the saved `overwrite=False` setting to `True`.
- Added producer-versioned output inventories and structural validation for all
  requested preprocess products, including DAT/LFP frame layout, MAT/XML/JSON,
  state-scoring figures, sidecar layout metadata, chanMap, and chanCoords.
- Added a pre-execution output contract bound to the preprocess fingerprint,
  input provenance, producer identity, and canonical session path. It survives
  OOM/cancellation, distinguishes validated legacy-resume outputs honestly from
  v2 outputs, and makes incompatible overwrite migrations rollback-safe.
- Made preprocess publication atomic for binaries, events, artifact MATs,
  MergePoints, session, state-scoring outputs, XML/RHD, channel metadata, and
  manifests. Compatible existing outputs are immutable under overwrite false;
  missing descendants are created; incompatible targets fail closed.
- Added per-output state-scoring recovery. A valid EMG, SleepScoreLFP,
  SleepState, episode MAT, or figure is reused independently, including repair
  of a missing SWTH figure without rewriting its valid SleepScore ancestor.
- Persisted sidecar channel-layout fingerprints and added rollback for the
  binary/fingerprint pair. The observed multi-day session uses identity or
  left-packed Intan ADC mappings, so its legacy `analogin.dat` remains eligible
  for validated reuse without a pre-existing fingerprint.
- Expanded acquisition provenance to analog, digital, auxiliary, supply, time,
  timestamp/sync, and Open Ephys TTL inputs.
- Split sorter progress manifests from the canonical completed manifest, made
  multi-day cleanup reject manifest path escapes, and publish links/XML/RHD plus
  server/local manifests and CSVs through coordinated same-filesystem
  transactions with cross-mount and cancellation rollback. Run-store and sorter
  provenance are retained after successful finalization.
- Made postprocess fail closed on partial existing `_spi` output when overwrite
  is false and publish a deeply validated complete attempt atomically. The
  intentionally retained input `cluster_group.tsv` update is now atomically
  published with an immediate SHA-validated backup/provenance.
- Made GUI Move stage and validate every selected transfer before publication,
  rewrite relocated metadata, retain required XML/RHD at custom destinations,
  reject ancestor paths, and remove Local sources only after destination
  validation. Added a common local session resume action for single- and
  multi-day output folders.
- Protected legacy noise labeling with the same per-session claim used by
  Persistent Runs. Preserved the established behavior that **Preprocess only**
  also submits Sorting when `run_sorter` is enabled.
- Stopped recursive permission changes over the complete session tree and
  stopped GUI cleanup from overriding the worker's analyzer-cache retention
  setting.

### Verification

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/execution/controller.py src/execution/session.py src/execution/worker.py src/postprocess/pipeline.py src/preprocess/events.py src/preprocess/gui/app.py src/preprocess/gui/config_model.py src/preprocess/io.py src/preprocess/mergepoints.py src/preprocess/multiday.py src/preprocess/pipeline.py src/preprocess/recording.py src/preprocess/runtime_prep.py src/preprocess/session.py src/preprocess/sorter_runner.py src/preprocess/sorting_stage.py src/preprocess/state_scoring.py
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/execution tests/preprocess/test_recording_selected_transform.py tests/preprocess/test_events_neurocode_compat.py tests/preprocess/test_io_source_selection.py tests/preprocess/test_anatomical_map_helpers.py tests/preprocess/test_session_mat_neurocode_compat.py tests/preprocess/test_pipeline_mixed_sources.py tests/preprocess/test_state_scoring_neurocode_compat.py tests/preprocess/test_sorter_partitions.py tests/preprocess/test_multiday.py tests/preprocess/test_gui_move_outputs.py tests/preprocess/test_gui_config_model.py tests/postprocess/test_postprocess_overwrite_skip.py tests/postprocess/test_postprocess_target_resolution.py
git diff --check
```

Result: 221 passed; Python compilation and `git diff --check` passed.

Legacy provenance migration follow-up verification:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/execution
python -m py_compile src/execution/session.py src/execution/controller.py
git diff --check
```

Result: 62 passed; Python compilation and `git diff --check` passed. A
read-only check of `run-20260811T172539Z-7c9bc52a` resolved the cancelled
`run-20260811T150535Z-eb1750a0` lineage, retained preprocess fingerprint
`b3a4e88b03688367350d3462eb990ab93271969b0119cecc766a123cda849ec8`,
and returned `trusted_prior_evidence=True`; it did not create or modify the
session contract.

Selected-subepoch/config/state-resume follow-up verification:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/execution tests/preprocess/test_state_scoring_neurocode_compat.py tests/preprocess/test_gui_config_model.py
python -m py_compile src/execution/controller.py src/execution/session.py src/preprocess/state_scoring.py src/preprocess/gui/app.py
git diff --check
```

Result: 109 passed; compilation and diff checks passed. A read-only check of
the real `run-20260811T175009Z-9bdca3bb` reduced provenance from 876 broad
Day-root entries to 366 selected-scope entries, confirmed that its contract
hash matches the prior snapshot and its preprocess fingerprint, and returned
`selected_inputs_compatible=True`. No real contract or session output was
modified by the diagnostic.

Large SleepScoreLFP follow-up verification:

```text
/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_io_source_selection.py tests/preprocess/test_state_scoring_neurocode_compat.py tests/execution
python -m py_compile src/preprocess/io.py src/preprocess/state_scoring.py src/execution/session.py
git diff --check
/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "x=load('/tmp/sleepscore-python-v73.mat'); assert(isequal(fieldnames(x.SleepScoreLFP),{'thLFP';'swLFP';'THchanID';'SWchanID';'sf';'t';'params'})); assert(isequal(x.SleepScoreLFP.thLFP,int16([1;2;3]))); assert(isequal(x.SleepScoreLFP.t,[0 .1 .2])); assert(strcmp(x.SleepScoreLFP.params.SWweights,'PSS')); assert(isequal(size(x.SleepScoreLFP.params.ignoretime),[0 0])); disp('PYTHON_V73_MATLAB_LOAD_OK');"
/local/workdir/ys2375/MATLAB/R2024b/bin/matlab -batch "x=load('/tmp/sleepscore-python-v73-compressed.mat'); assert(numel(x.SleepScoreLFP.t)==300000); assert(x.SleepScoreLFP.t(end)==(299999/1250)); assert(numel(x.SleepScoreLFP.thLFP)==300000); assert(x.SleepScoreLFP.thLFP(2)==int16(1)); disp('PYTHON_V73_COMPRESSED_MATLAB_LOAD_OK');"
```

Result: 99 passed; compilation and diff checks passed. MATLAB R2024b loaded the
Python-written v7.3 structure with the expected `thLFP`, `swLFP`, channel IDs,
sampling frequency, time vector, nested params, char string, and empty-array
shape (`PYTHON_V73_MATLAB_LOAD_OK`). MATLAB also loaded the chunked,
shuffle+DEFLATE compressed path (`PYTHON_V73_COMPRESSED_MATLAB_LOAD_OK`). The
real LFP size calculation produced
480,929,407 frames and a 5,771,152,884-byte (5.375-GiB) core payload. Atomic
failure cleanup left no canonical or partial `SleepScoreLFP` file from the
failed Run.

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/execution tests/preprocess tests/postprocess --ignore=tests/postprocess/test_ks4_diagnostics.py
```

Result: 358 passed and 17 failed. The 17 failures reproduce the existing,
unrelated expectations for threading instead of process-based artifact work,
eager raw loading, removed legacy artifact configuration fields, prior TTL
channel normalization, the old high-amplitude worker fallback, and historical
Kilosort-4 normalization/import/CLI behavior. The omitted diagnostic test module
cannot collect because `src.postprocess.ks4_diagnostics` is absent in the
baseline tree. None of these failures exercises the overwrite/resume changes.

### Known limitations

- Verification used synthetic and focused integration fixtures. The cancelled
  2.2-TB multi-day session has not yet been submitted again with this final
  source tree, so a live Slurm recovery remains an operator verification step.
- Multi-file publication uses validated same-directory staging and rollback,
  but a host or filesystem failure during the final rename/rollback window can
  still require inspection of retained partial/backup files.

The current cancelled real Run was also checked without writing any files:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c 'from pathlib import Path; from src.execution.store import RunStore; from src.execution.session import settings_for_store,session_output_dir,_preprocess_contract_identity,_trusted_preprocess_evidence; r=Path("/local/workdir/ys2375/PreprocessPipeline/sorting_temp/.pipeline/run-20260811T150535Z-eb1750a0"); s=RunStore(r); g=settings_for_store(s); d=session_output_dir(g); i=_preprocess_contract_identity(s,d); print("session",d); print("fingerprint",i["stage_fingerprint"]); print("trusted_prior_evidence",_trusted_preprocess_evidence(s,g,i)); print("existing_contract",(d/".preprocess-output-contract.json").exists())'
```

Result: the exact `multiday_Day14_to_Day217` output resolved correctly,
`trusted_prior_evidence=True`, and no contract currently exists. Therefore the
first new overwrite-disabled Run can create an explicitly
`validated-legacy-resume` contract from the matching prior scientific
fingerprint/input snapshot, validate the existing identity-layout sidecar as
eight channels, and create its layout JSON without rescanning artifactHigh.

### Independent review

A new read-only `gpt-5.6-sol` reviewer at high reasoning audited the complete
implementation and tests. Its initial review identified missing interrupted-run
producer evidence, multi-day config ordering, cross-filesystem staging,
transaction/interrupt rollback, Move content verification, post-only hash and
provenance merging, Open Ephys TTL provenance, postprocess publication, manual
claim lifetime, and sidecar-layout integrity findings. Those findings were
fixed and regressed. The final narrow re-review of the current tree reported
**no Blocker, High, Medium, or Low findings**; its final sidecar-layout check
passed 8 tests and `git diff --check`.
