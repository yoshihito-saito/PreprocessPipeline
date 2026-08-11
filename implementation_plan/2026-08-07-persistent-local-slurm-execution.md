# Persistent Local and Slurm Execution Framework

## Goal and Motivation

Implement a persistent, stage-based execution framework for the standard raw-data workflow:

```text
preprocess -> sorting -> postprocess
```

Runs must outlive the Qt GUI, use the same backend-independent stage worker under Local and Slurm, preserve existing scientific behavior, and keep analysis parameters separate from execution resources.

## Current Problem

The Qt GUI currently owns a `QProcess` that runs `src.preprocess.gui.run_pipeline`. That child executes preprocessing, sorting, and postprocessing in one process. Sorting is embedded in `run_preprocess_session`, pipeline stdout/stderr exists primarily in GUI memory, and the temporary GUI configuration is stored under the system temporary directory. Normal GUI close is blocked while the process is active, but there is no durable reconnect or recovery contract after GUI loss.

Worker counts are stored beside scientific GUI settings and normalized against the GUI host's CPU count. This cannot represent an exact 128-CPU Slurm request when the submission host exposes fewer CPUs. Slurm commands, job identifiers, scripts, dependencies, and accounting state are not currently represented.

## Affected Modules and Files

- `src/execution/`
  - Add versioned domain models, atomic Run storage, Stage workers, Local and Slurm backends, controller/reconciler, retry, and cancellation.
- `src/preprocess/pipeline.py`
  - Extract the existing sorter block into a reusable sorting-stage function without changing sorter parameters or numerical behavior.
- `src/preprocess/gui/config_model.py`
  - Add execution-only GUI settings while preserving older saved-config compatibility.
- `src/preprocess/gui/app.py`
  - Add Execution/resource controls and persistent Run monitoring; submit long-running work through detached execution rather than a GUI-owned pipeline `QProcess`.
- `src/preprocess/gui/preflight.py`
  - Add persistent workspace and backend capability checks where they can be established before submission.
- `pyproject.toml`
  - Add backend-independent controller/worker command entry points if useful for direct execution and Slurm scripts.
- `tests/execution/` and affected `tests/preprocess/`
  - Cover persistence, atomic facts, resource rendering, backend selection, Stage validation, retry, cancellation, and config compatibility.
- `README.md`
  - Document persistent Local/Slurm execution, shared storage, fallback, and recovery behavior.

## Domain and Persistence Design

Use dataclasses and versioned JSON to match current repository conventions.

- A Run owns an immutable analysis snapshot, initial execution configuration, provenance, and logical Stages.
- A Stage owns append-only Attempts.
- A retry always creates a new Attempt and never overwrites prior facts, logs, scripts, or results.
- `JobRef` is always stored as a list even though the initial implementation submits one job per Attempt.
- Attempt directories contain immutable `spec.json`, submission intent/facts, worker start/result/failure facts, logs, and an exact Slurm script when applicable.
- `state.json` is a derived materialized view and can be regenerated from Attempt facts plus backend observations.
- JSON writes use a same-directory temporary file, flush/fsync where practical, and `os.replace`.

Run layout:

```text
<workspace>/.pipeline/<run_id>/
    run.json
    analysis_config.json
    execution_config.json
    snapshots/
    stages/<stage>/attempt-NNN/
    state.json
```

The workspace must be writable and, for Slurm, visible from submission and compute nodes. Node-local temporary files are not authoritative Run state.

## Configuration Semantics

`analysis_config.json` snapshots current GUI scientific settings after removing execution-only CPU/resource fields. The sorter YAML is copied into `snapshots/` at Run creation, and the immutable analysis snapshot points to that copy.

Sorter snapshots are read once and copied byte-for-byte; the checksum is derived
from that same captured byte sequence. This preserves the checksum across
platform line endings (including CRLF configuration files) and prevents either
newline conversion or a concurrent source edit from creating a self-inconsistent
Run.

`ExecutionConfig` stores requested and resolved backend plus one independent `ResourceSpec` for each Stage. Exact requested CPU counts are positive integers and are not capped against `os.cpu_count()` on the GUI/login host.

Each generated Slurm job fails before Stage execution if its derived persistent
workspace is not readable and writable on the assigned compute node. Obvious
system-temporary paths remain visible as preflight warnings.

The Stage adapters apply the Attempt CPU count consistently to existing pipeline worker controls and thread-limiting environment variables. They do not change sorter YAML values, Kilosort `batch_size`, clustering parameters, filtering parameters, thresholds, seeds, or other scientific settings.

On POSIX, each Kilosort Attempt receives an immutable shim directory, invokes
its generated MATLAB startup explicitly in the parent MATLAB command, and uses
a unique MATLAB Processes `JobStorageLocation`. The shim is not added to the
inherited `MATLABPATH`, so child workers neither rerun the parent startup nor
join stale jobs from another Attempt. Generated shell literals are quoted as
data so valid paths containing shell metacharacters remain safe.

The POSIX shim must also isolate MATLAB's operating-system stdin/stdout/stderr
from SpikeInterface's captured launcher pipes. MathWorks ServiceHost processes
can outlive the MATLAB executable and inherit those descriptors; if they retain
the write end of a captured pipe, Python waits forever for EOF after Kilosort
has already produced its final outputs. Redirect MATLAB stdio to an
Attempt-specific temporary file, retain the existing MATLAB `-logfile` as the
authoritative runtime log, merge launcher stderr into that log only on failure,
and unlink the temporary stdio file when the shim exits. A detached descendant
that retains the temporary file descriptor must not delay shim or Stage-worker
completion.

Initial resource defaults:

- preprocess: GUI preprocess CPU value, 256 GiB host RAM, zero GPUs;
- sorting: GUI sorter CPU value, 512 GiB host RAM, and exactly one untyped GPU;
- postprocess: GUI postprocess CPU value, 256 GiB host RAM, zero GPUs.

## Stage Boundaries and Validation

The preprocess Stage calls the existing pipeline with sorting disabled and persists all data required by Sorting. The sorting Stage reads that result and reuses the extracted existing sorter implementation. The postprocess Stage reads the successful Sorting result and calls the existing postprocess pipeline.

Completion requires both process success and output validation. Required paths are checked before writing `result.json`; validation errors write `failure.json` and return non-zero. Stage workers write only inside their own Attempt directory and never submit downstream work.

## Backend and Lifecycle Design

The common backend interface is:

```text
submit(attempt_spec, dependency_job_refs=None) -> list[JobRef]
status(job_ref) -> BackendStatus
cancel(job_ref) -> None
```

Local uses a detached per-Run controller, not a persistent daemon. The controller starts one backend-independent Stage worker at a time and records each local process handle. Closing or crashing the GUI does not terminate the controller or workers. Persisted logs replace GUI-owned process output signals as the monitoring source.

Slurm generates and persists one sbatch script per Attempt, submits enabled stages upfront with `afterok` dependencies, and exits after successful submission. Scripts use exact Attempt resources, safely quoted arguments, and no GPU request for preprocess/postprocess. Sorting requests exactly one configured GPU. `squeue` supplies live state, `sacct` supplies terminal state/exit/telemetry when available, and `scancel` handles cancellation.

Auto resolves before any submission. Local fallback is allowed only when capability checks fail before a Slurm submission attempt begins. Ambiguous or failed Slurm submission never launches the same Attempt locally.

## Retry and Dependency Semantics

Retrying a failed Stage creates a new Attempt for that Stage and new downstream Attempts. A completed upstream StageResult is reused unless the user explicitly reruns it. Prior attempts become `superseded` for workflow selection but remain intact. Old dependent Slurm jobs are never reused after dependency failure.

## GUI Behavior

Add a **Run** page inside the existing Ephys tab, alongside Preprocess and
Postprocess. All execution actions live on this page: Run all, Preprocess only,
Postprocess only, requested backend, resolved/capability display, persistent
workspace, per-Stage resources, Run status, Stage/Attempt history, backend job
IDs and times, persisted stdout/stderr viewing, reconnect, cancellation, and
retry. Do not retain a second Ephys-wide run-button area or add a separate
top-level Execution tab.

For a fresh or legacy GUI configuration without an explicit backend choice,
capability detection selects explicit Slurm as the initial requested backend
when the required Slurm client commands (including `sacct` when required) are
installed. Controller reachability is deliberately not part of this server
classification: a transient scheduler outage must produce an explicit Slurm
error rather than changing the default to Auto and possibly running locally.
Without the Slurm clients, the initial selection remains Auto. A backend choice
loaded from a saved configuration is authoritative and is not replaced by
environment detection.

### Follow-up: simplified resources and unlimited walltime

Derive the persistent workspace from the existing Local working directory and
store Runs under its hidden `.pipeline` directory. Remove workspace,
acknowledgement, MATLAB path, accounting, QoS, reservation, partition, and GPU
type/constraint controls from the normal GUI. Keep these fields readable in the
versioned configuration models for backward compatibility, while newly
collected GUI settings use scheduler defaults, `sacct` recovery, automatic
MATLAB discovery, and an untyped one-GPU sorting request.

Expose only exact CPU count, host RAM, and walltime for each Stage; display the
fixed GPU count as read-only. Walltime is optional in the resource model.
`None` means partition default/no submitted `--time` directive and is the GUI
default. A positive custom hour value emits an exact Slurm time directive.
Legacy positive values retain their existing meaning. Do not encode unlimited
as zero. On the current `regular` partition, `DefaultTime=NONE` and
`MaxTime=UNLIMITED`, so an omitted directive is unlimited; other clusters remain
subject to their partition policy. Retry must preserve unlimited without
serializing the string `None` as a CLI integer.

### Follow-up: narrow Run-page layout

The Run page must remain usable at the existing 500-pixel settings-pane width.
Avoid controls whose minimum horizontal size forces a hidden horizontal canvas:
all three Run-scope buttons must be visible without horizontal scrolling. Make
each Stage resource group compact enough that Preprocess, Sorting, and
Postprocess are discoverable in the vertical flow, using a two-column CPU/RAM
row and a short walltime row. Keep resource values and persistence semantics
unchanged. Add an offscreen GUI regression that checks the Run scroll area has
no horizontal overflow at the normal window size and that the Postprocess
controls are reachable by vertical scrolling.

### Follow-up: operator-focused Run status

Keep walltime support in the execution model and CLI for compatibility, but do
not expose it in the normal GUI. GUI-created Runs always omit `--time` and use
the scheduler/partition default. Present resources using operational terms:
`CPU cores`, `Memory (GiB)`, and `GPUs`. Explain that memory is host RAM and that
Sorting requests one scheduler-assigned GPU. Do not expose a GPU device ID:
Slurm must choose an available device and communicate it through
`CUDA_VISIBLE_DEVICES`; Preprocess and Postprocess continue to request zero
GPUs.

Replace the Attempt-history table and its normal-operation recovery buttons
with a compact three-Stage progress view. Show one human-readable status and
the current Slurm/Local job identifier for Preprocess, Sorting, and Postprocess.
Reconciliation remains automatic while a backend job is active. Reopen,
manual reconcile, resume/submit, Stage cancel, and Stage retry remain available
through the persistent controller/CLI, but are not normal GUI controls. When a
GUI is restarted for a session with an active Run claim, reconnect to that Run
automatically. Re-evaluate that claim when Basepath/Local working dir changes so
status and Force stop cannot remain bound to a different session. A confirmed
cancellation is terminal in the compact status even if Local process accounting
is unavailable.

Use the existing bottom `Force stop` control as the only visible pipeline-stop
action. For persistent Runs it requests controller cancellation of all active
Stage jobs; for a legacy GUI-owned process it retains the existing process-tree
termination behavior. Do not silently cancel when the GUI closes.

Reconnected Local cancellation must verify persisted process identity before
signalling a process group. If the platform cannot provide the recorded process
start identity, report cancellation failure instead of risking a signal to a
reused PID or falsely confirming cancellation. A controller that still directly
owns its child may cancel that owned process. After signalling, track the whole
verified process group through TERM-to-KILL escalation; disappearance of the
worker leader alone is not cancellation confirmation while a MATLAB/shim
descendant remains.

Tail newly appended selected-Attempt stdout/stderr and active Kilosort logs into
the existing main Log pane without repeating already displayed bytes. Workers
must use unbuffered Python output under both Local and Slurm so progress becomes
visible during, rather than after, a Stage. Bound initial tail reads when
reconnecting to an existing Run and retain persistent files as the canonical
log source.

Apply dark foreground/background styling to item views and headers. Size the
initial main window to the available desktop geometry so it fits common VNC
screens; retain scrollable settings pages and avoid adding new minimum-size
constraints that exceed the screen.

### Follow-up: selectable output storage

Rename the final copy action to **Move outputs to storage** and show its resolved
destination in the bottom action area. For a single-day session, default to the
current Basepath. For multi-day processing, default to and create
`<Basepath parent>/<Multi-day name>` so combined output is not written into the
first day's raw directory. **Browse save dir** selects an exact destination and
updates the displayed Basepath. Preserve the pre-selection Local output path and
basename for that move so changing Basepath cannot redirect the move source.
Retain the existing overwrite, optional `.dat`, and local-cleanup semantics.

The GUI may close while persistent Runs are active. Closing never cancels a Run implicitly. Existing Phy and CellExplorer lifecycle behavior remains outside this framework.

## Follow-up: Minimal-GUI Resume, Output Safety, and Retention

Keep the current Qt tabs, Basepath/Local root fields, overwrite checkboxes,
Ephys-scoped Run controls and monitor, and Phy/CellExplorer buttons. Do not
introduce a replacement workflow screen or a new permanently visible planning
UI.

When Basepath names an existing processed local session, detect that layout and
use the selected directory itself as the session output directory. Recover the
raw acquisition source from the final Run record, persistent metadata, or the
legacy preprocess manifest. Sorting and postprocess may reuse a validated local
session without the raw source; a preprocess rerun must stop and request a raw
source when it cannot be recovered. Never reinterpret the already processed
`<basename>.dat` as raw acquisition input.

Before submission, derive a Stage reuse plan from atomic completion evidence,
Stage-specific analysis fingerprints, input provenance, and scientific output
validation. With overwrite disabled, reuse compatible completed Stages and
rerun from the first incomplete, failed, invalid, or incompatible Stage. A
Stage selected for rerun replaces its own incomplete generated artifacts even
when the user-level overwrite flag is false; it must not mix old and new
partial outputs. With overwrite enabled, force the requested Stage scope and
downstream Stages to rerun. Active or ambiguous prior jobs block adoption and
resubmission.

Legacy output directories without persistent completion facts may be adopted
only after equivalent deep validation. Persist the adoption decision so later
Runs do not infer it repeatedly. Parameter compatibility is Stage-specific so
that a postprocess-only parameter change does not invalidate preprocessing or
sorting.

Route `all`, `preprocess`, and `postprocess` GUI actions through the same
persistent Local/Slurm controller while preserving their existing buttons.
Explicit Slurm remains strict, Auto may fall back only before submission, and
Sorting requests exactly one scheduler-assigned GPU. The default persistent
workspace is the Local root rather than the visible session directory, keeping
`<local-root>/.pipeline/` separate from scientific outputs.

Serialize output mutation with a session-level action lock in addition to the
per-Run controller lock. A second Run, manual Phy launch, or CellExplorer launch
must not access a session while a persistent Stage can still mutate it. Lost or
ambiguous scheduler state never causes automatic lock removal.

Protect successful and manually curated outputs. Large generated binaries use
same-filesystem temporary paths and validation before atomic publication.
Rerunning Sorting creates a new timestamped Kilosort directory. Postprocess
overwrite must never recursively delete `.phy`, `phy.log`, or manual curation;
when generated results cannot be updated safely in place, preserve/version the
old directory rather than silently destroying it.

Phy and the current interactive CellExplorer wrapper remain Local. They are not
submitted as Slurm batch jobs and are disabled while a persistent Run for the
same session is active. Prefer the validated postprocessed `_spi` directory for
Phy, while retaining explicit manual folder selection.

During execution, retain all immutable controller/worker/backend facts needed
for recovery and diagnosis. After the complete Run has validated successfully
and every backend job has a conclusive successful terminal observation, write:

```text
<session>/preprocess_run.yaml
<session>/preprocess.log
<session>/sorter_partition_manifest.json
<session>/Kilosort_<timestamp>/kilosort.log
<session>/Kilosort_<timestamp>/sorter_config_resolved.yaml
```

The final record contains exact analysis parameters, input/code/environment
provenance, Stage resources and timings, scheduler job identifiers and terminal
telemetry, output validation, and detected warning/error counts. Consolidate
Stage stdout/stderr into the single pipeline log, suppress noninteractive
progress bars, retain `kilosort.log`, and remove only proven-redundant success
artifacts. Do not generate the copied-source `preprocessSession.log` for
persistent Runs. Do not retain duplicate `matlab_run.log` when it is identical
to `kilosort.log`. SpikeInterface helper JSON and launch scripts may be removed
only after successful sorting validation. Failure, cancellation, lost state, or
ambiguous submission retains detailed diagnostics and is never compacted
automatically. Legacy direct Python/MATLAB-compatible entry points retain their
existing output defaults unless explicitly configured otherwise.

GUI feedback uses the existing log pane, confirmation dialogs, and Run monitor.
When existing outputs are detected, emit a compact reuse/rerun summary and ask
for confirmation only when adoption, replacement, or curated-output protection
requires it.

## Verification Strategy

- Unit-test model round trips, immutable analysis hashing, atomic files, and derived state regeneration.
- Unit-test exact 128-CPU preservation and Slurm directive rendering.
- Unit-test zero-GPU preprocess/postprocess and exactly-one-GPU sorting scripts.
- Unit-test capability resolution and the no-fallback-after-submission rule.
- Unit-test `sbatch --parsable`, dependency construction, `squeue`/`sacct` normalization, cancellation, and retry attempt creation using mocked command execution.
- Unit-test Stage output validation and structured failure facts with lightweight fake stage functions.
- Exercise detached Local orchestration with small test workers rather than scientific dataset simplification.
- Exercise the complete detached Local workflow against the repository's test
  recording, preserving failed/cancelled Attempts and verifying retry reuse of a
  completed preprocess result.
- Exercise the complete real-data workflow through live Slurm jobs, verifying
  persisted scripts, `afterok` transitions, one-GPU Sorting allocation,
  scheduler accounting, logs, Stage validation, and final reconciliation.
- Run the existing preprocess and postprocess test suites to detect scientific or workflow regressions.
- Inspect the final diff and run an independent read-only reviewer audit using the repository-requested reviewer configuration.
- Regression-test raw Basepath behavior and processed-session Basepath
  detection without adding new persistent GUI controls.
- Test compatible Stage reuse, legacy adoption, incomplete-Stage replacement,
  Stage-specific fingerprint invalidation, active/ambiguous-job blocking, and
  overwrite-forced reruns.
- Test session-level exclusion across distinct Run IDs and manual curation
  actions.
- Test that postprocess overwrite cannot erase `.phy`, `phy.log`, or curated
  cluster files.
- Test persistent postprocess-only submission under both Local and Slurm.
- Test successful final log/record generation and compaction, plus diagnostic
  retention on failure/cancel/lost.
- Test timestamped Kilosort naming and conditional removal of exact duplicate
  MATLAB/SpikeInterface helper artifacts.
- Regression-test POSIX MATLAB completion with a fake MATLAB executable that
  exits after spawning a longer-lived descendant which inherits stdio. The shim
  and its captured caller must return promptly, proving that descriptor
  inheritance cannot hold the Sorting worker in `pipe_read`.

## Explicit Non-goals

- Slurm job arrays;
- per-Stage Local/Slurm hybrid selection;
- automatic memory or VRAM estimation;
- automatic scientific parameter modification after resource failure;
- Slurm execution of Phy or CellExplorer;
- replacement of the existing GUI layout or addition of a mandatory wizard;
- destructive replacement of manually curated Phy/CellExplorer outputs;
- a persistent daemon or external database;
- hashing large acquisition binaries.

## 2026-08-10 Follow-up: Reopen a Local Session and Compact the GUI

### Goal

Allow a user to select one existing local output session directory and recover
the saved single-day or multi-day GUI settings without selecting the original
file-server folders again. This must also reconnect the Run monitor when the
local session still points to a persistent Run. Reduce the GUI's overall text
size and initial left-pane width while keeping control text readable.

### Recovery semantics

- Add one source-type-neutral **Browse local session to resume** action. The
  selected directory is the scientific session output directory, not the
  hidden `.pipeline` directory and not a sorter output directory.
- Prefer `<session>/.pipeline-active-run.json`. Resolve its Run directory,
  validate `run.json`, the immutable `analysis_config.json` hash, the complete
  `execution_config.json`, and the exact `session_output_dir` association.
- If no active-Run pointer exists, accept a completed `preprocess_run.yaml`
  containing validated analysis and execution snapshots for that exact output
  directory.
- Restore the complete `PipelineGuiSettings`, including multi-day source
  sessions, selected subepochs, name/order, scientific parameters, backend,
  resources, and worker counts. Bind Local working directory to the selected
  session's parent so the selected folder remains the output target.
- A malformed or mismatched active pointer fails closed and does not silently
  fall back to older metadata. A directory without either supported marker is
  rejected without partially changing the current GUI.
- Reconnect and reconcile a claim-backed Run, but do not automatically submit,
  retry, cancel, enable overwrite, or change immutable scientific settings.
  Completed-session recovery restores settings but does not invent an active
  Run.

### GUI density

- Reduce the application-wide base font from 12 px to 11 px, with hints at
  10 px, and reduce control padding proportionally.
- Reduce the initial settings-pane width from 520 px to approximately 440 px
  and its minimum width from 340 px to 300 px.
- Keep button captions and explanatory text fully visible through natural
  size hints and word wrapping; do not use elision for operator-facing labels.

### Verification

- Unit-test active/failed Run recovery, completed YAML recovery, output-path
  mismatch rejection, malformed metadata, and single-day/multi-day settings.
- Add an offscreen GUI test for the dedicated browse action, atomic failure
  behavior, active Run reconnection, smaller base font, and narrower initial
  settings pane with the browse caption fully visible.
- Run the affected GUI/config tests, execution model/store tests, Python
  compilation, and `git diff --check`.

## 2026-08-11 Follow-up: Release Terminal Backend Claims

### Goal and observed failure

Allow a new GUI Run to replace a prior session claim after Slurm has
conclusively terminated every job, even when an abrupt backend failure such as
an OOM kill prevented the Stage worker from writing `failure.json`. The observed
Run `run-20260811T045641Z-0de3bbc5` is derived as failed from a terminal
`OUT_OF_MEMORY` observation, but claim exclusion still treats its persisted
`started.json` without a worker terminal fact as active.

### Intended semantics

- Treat a backend observation as conclusive only when it is terminal and has a
  known success value, or when it explicitly reports cancellation.
- Do not let a stale `started.json` keep ownership after a conclusive terminal
  backend observation. The derived failed/cancelled Run may be replaced by a
  new Run, preserving the prior Run directory in claim history.
- Continue blocking new Runs for submitted/running work, ambiguous submission,
  terminal success without a validated Stage result, and lost/unknown backend
  outcomes.
- Do not edit or delete the affected session's Run facts or active-Run marker;
  normal `create_run` claim replacement is the only mutation of that pointer.
- This change enables a new Run with updated resources. It does not
  automatically retry, resubmit, or change immutable analysis settings.

### Verification

- Reproduce the bug with submitted and started facts, no worker failure fact,
  and a conclusive terminal OOM observation.
- Verify that a subsequent Run is created and records the failed Run as
  `previous_run_dir`.
- Preserve regressions proving that active, ambiguous, and lost Runs still
  block session claim replacement.
- Run the focused session-resume tests, execution suite, Python compilation,
  and `git diff --check`.

## 2026-08-11 Follow-up: Immutable Outputs and Transactional Resume

### Goal and observed failure

Make the user-facing `overwrite=False` setting authoritative for persistent
Runs and allow an interrupted preprocess Stage to continue from validated
scientific outputs. A failed state-scoring attempt left valid sidecars,
artifact events, `.dat`, `.lfp`, `session.mat`, and EMG output, but the next
Persistent worker silently changed `config.overwrite` to `True` and began a
second high-amplitude artifact scan.

### Required behavior

- Never change the saved preprocess or postprocess overwrite flag inside a
  Stage worker. `overwrite=False` makes an existing canonical output immutable.
- Reuse an existing output only after validating its structure, dimensions,
  configuration/input provenance, and producer schema where applicable.
- Create missing outputs, but publish every binary, MAT, JSON, CSV, image, and
  manifest through a same-directory temporary path followed by validation and
  atomic replacement.
- When an existing output is invalid or incompatible and overwrite is false,
  fail before changing it and report the exact conflicting path. Explicit
  overwrite may rebuild or version the target.
- Persist an output/checkpoint inventory sufficient to distinguish completed
  preprocess subproducts after a worker OOM or cancellation. Completion and
  adoption validation must cover every output requested by the immutable
  analysis configuration, including LFP, sidecars/events, artifact events, and
  state-scoring MATs/figures.
- Treat state scoring as an output dependency graph: reuse valid EMG,
  SleepScoreLFP, SleepState, episode, and figure outputs individually; compute
  only missing descendants; never rewrite a valid ancestor under
  `overwrite=False`.
- Include analog, digital, auxiliary, time, supply, and Open Ephys TTL inputs in
  acquisition provenance. Add explicit producer/output schema identities so a
  scientific code-layout change can invalidate stale outputs without making
  unrelated repository edits invalidate reuse.
- Preserve the existing behavior that postprocess may update the input
  Kilosort `cluster_group.tsv`, but make that write atomic and recoverable.
- Preserve the current **Preprocess only** scope: Sorting is still submitted
  when `run_sorter` is enabled. Clarify the label/tooltip only if needed.
- Build sorter and multi-day manifests transactionally. A failed attempt must
  not replace the last completed canonical manifest, and a manifest may never
  authorize deletion outside its staging root.
- Keep success diagnostics in the Run store and restrict permission changes to
  newly created outputs rather than recursively changing the complete session.
- Serialize noise-label and other session-mutating GUI actions with the same
  session claim used by persistent Runs.

### Move behavior

Retain the existing selected-output and local-cleanup semantics, including the
ability to omit `.dat` and then remove the local source when cleanup is
explicitly enabled. Make the operation transactional:

1. resolve and display the move/retain/delete inventory, including byte sizes;
2. reject source/destination ancestor relationships and unsafe targets;
3. transfer into a destination-side temporary tree and validate it;
4. rewrite relocated absolute paths in `preprocess_run.yaml`, sorter manifests,
   and other canonical metadata;
5. atomically publish the destination where the filesystem permits;
6. delete the local source only after publication and validation succeed.

Required metadata such as XML/RHD must either already exist at a custom
destination or be transferred before cleanup. A moved session that is intended
to remain resumable must pass the same local-session recovery validation at its
new path.

### Verification

- Regress the observed failed-state-scoring case: the next Run keeps
  `overwrite=False`, reuses artifact events, `.dat`, `.lfp`, and EMG, and starts
  only missing state-scoring descendants.
- Assert hashes and mtimes of compatible existing outputs remain unchanged.
- Inject writer failures and cancellation into sidecar, MAT, JSON/CSV, figure,
  multi-day, and sorter-manifest publication; the prior canonical file must
  remain valid.
- Reject corrupt, shape-invalid, provenance-incompatible, and producer-version
  incompatible outputs before mutation.
- Verify complete requested-output validation and adoption for Intan, Open
  Ephys, mixed-source, single-day, and selected-subepoch multi-day inputs.
- Verify postprocess honors overwrite false while preserving the intentional
  atomic `cluster_group.tsv` update.
- Verify failed sorting leaves the previous canonical partition manifest
  untouched.
- Verify transactional Move succeeds across the supported destination layouts,
  rewrites recovery paths, and leaves the source intact under injected failure.
- Preserve the existing Preprocess-only-plus-optional-Sorting behavior.
- Run focused execution/preprocess/postprocess/GUI tests, the broad affected
  suites, compilation, `git diff --check`, and an independent read-only review
  using a new GPT-5.6-sol high reviewer.

### Implemented design notes

- Immutable reuse is producer-output based rather than Stage-existence based.
  Current Runs record schema-versioned inventories; legacy outputs retain a
  narrower compatibility path only where their format can still be validated.
- Existing compatible outputs remain at their canonical paths and keep their
  bytes/mtime. Missing outputs publish from same-directory temporary files.
  Existing incompatible outputs fail before mutation when overwrite is false.
- State scoring resumes as a dependency graph. It does not treat the four MAT
  files alone as completion because the four diagnostic figures are requested
  outputs too.
- Sorter progress is attempt-owned. Only a fully completed partition set may
  replace `sorter_partition_manifest.json`.
- Transactional Move is a validated copy/publish followed by source removal.
  If source cleanup fails after publication, the valid destination is retained
  and the error explicitly reports that duplicate/remnant Local data may remain.
- The postprocess input `cluster_group.tsv` mutation and Preprocess-only plus
  optional Sorting behavior remain explicit accepted exceptions requested by
  the operator.

### Final verification status

- Focused execution/preprocess/postprocess/GUI integration: 221 passed.
- Broad affected suite: 358 passed with the same 17 unrelated legacy failures
  documented in the change log.
- Python compilation and `git diff --check`: passed.
- Read-only `gpt-5.6-sol` high-reasoning final review: no Blocker, High,
  Medium, or Low findings remain.
- The real cancelled Run has matching trusted prior fingerprint/input evidence;
  its first new overwrite-disabled attempt can establish the validated-legacy
  contract and exact eight-channel identity layout without rewriting the
  existing analog sidecar or rescanning the existing artifactHigh events.

### Legacy provenance migration follow-up

The first live retry exposed a migration boundary that synthetic fixtures did
not cover: acquisition provenance gained 342 sidecar paths between the
cancelled Run and the retry. Preserve fail-closed reuse while allowing this
specific old-to-new schema transition:

- every path recorded by the legacy snapshot must still match its existence,
  type, size, and modification time exactly;
- a path newly collected by the expanded provenance scanner is accepted only
  when it is a file whose modification and change times both predate the
  immutable prior Run creation/scan-start cutoff;
- missing timestamps, removed legacy inputs, and sidecars added or modified
  after the legacy snapshot remain incompatible;
- record scan start before enumeration, including `ctime_ns`, so a later file
  copied with preserved `mtime` (`cp -p`/`rsync -a`) cannot pass migration;
- immediately before contract/adoption trust, re-enumerate the acquisition
  inputs and require the live path set and metadata to equal the immutable
  current Run snapshot, closing the Run-creation-to-worker TOCTOU window;
- persist the previous Run lineage in both the claim and `run.json`, retaining
  the mutable claim only as a backward-compatible lookup for already-created
  Runs.

Verification must reproduce the actual `run-20260811T172539Z-7c9bc52a`
contract decision without writing to the session, and include negative tests
for a newly modified sidecar.

### Selected-subepoch provenance and state-resume follow-up

- For multi-day runs with explicit subepoch selections, recursively inventory
  acquisition inputs only beneath those selected subepochs. Do not let files
  in an unselected sibling subepoch invalidate reusable output. Multi-day runs
  without subepoch selections continue to inventory their selected session
  roots, and single-day runs retain the existing source-root behavior.
- Permit an existing compatible contract to migrate from the former broad
  Day-root provenance hash to the selected-only hash only when the old hash is
  bound to a same-session, same-stage-fingerprint prior Run and selected inputs
  pass live metadata/cutoff validation.
- Reuse a valid `EMGFromLFP` MAT whenever overwrite is false. The EMG producer
  has no optional `save_files` mode; an undefined variable must not prevent the
  interrupted state-scoring Run from continuing.
- Open the GUI Load config dialog at `<PreprocessPipeline>/config`, resolved by
  the installed/editable project-root mechanism rather than the process CWD.

### Large multi-day SleepScoreLFP MAT follow-up

The real 123,117,928,192-byte LFP contains 480,929,407 frames. The requested
`SleepScoreLFP` payload contains two int16 channel vectors plus one double time
vector, approximately 5.38 GiB before metadata, and cannot be represented as a
MATLAB v5 structure.

- Keep MATLAB v5 for ordinary outputs and preserve all existing field names,
  dtypes, shapes, and state-scoring semantics.
- Estimate the complete nested payload before writing. When it approaches the
  v5 signed-size boundary, write MATLAB v7.3/HDF5 instead of first attempting a
  doomed v5 serialization.
- Match MATLAB's v7.3 struct, dimension-order, numeric, char, and empty-array
  representation; validate the top-level payload without loading multi-GiB
  arrays during atomic publication or completed-output checks.
- Retain full payload loading only when an interrupted Run must actually reuse
  `SleepScoreLFP` for downstream state computation.
- Verify Python round-trip, MATLAB R2024b load compatibility, atomic behavior,
  and the original real-session size calculation.
