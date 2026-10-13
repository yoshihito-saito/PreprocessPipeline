# Configurable bounded TeraSort waveform length

## Goal and scope
Use one odd `nt` for bounded seed learning, model alignment, CPU/CUDA matching, novelty/PCA observations, cached waveforms and Phy output. Preserve the default 61-sample behavior and existing checkpoint identities. Expose `nt: 41` in the PreprocessPipeline TeraSort config for 20 kHz input. Use updated PP main in the new worktree and retain existing opt-in TeraSort integration changes.

## Implementation steps
1. Derive waveform dimensions and centered offsets from model/config nt; preserve unrelated CUDA warp sizes, time budgets, thresholds and parameters.
2. Thread nt through seed learning and run configuration; reject incompatible seeds/restarts without rewriting old outputs.
3. Update matching, novelty, PCA/cache and export consumers to use the same waveform length.
4. Add the bounded backend/config adapter to updated PP main, preserving WILD-NP preprocessing and the existing Phy postprocess contract.
5. Review task diffs and perform at most one lightweight changed-file check; record actual validation and remaining runtime/quality checks.

## Decisions and validation boundary
- PP: `C:/Users/aalab/.codex/worktrees/terasort-nt-config/PreprocessPipeline`, based on main `83ebca0`; TeraSort: `D:/TeraSort-preprocess-pipeline-integration`.
- One agent; no commits/pushes, new tests, dependency changes, sorting, simulations or broad checks.
- nt defaults to 61 for TeraSort callers; PP config selects 41. No implicit sample-rate conversion, padding to 61, tolerance changes or alteration of other scientific settings.

## Completion
- Implemented the bounded-only backend in the fresh PP worktree and `nt: 41` in `sorter/TeraSort_config.yaml`. GUI/CLI selection, per-session config copies, original-column manifests, independent NP probe partitions, flat Phy output and persistent native retry use the existing pipeline contracts.
- TeraSort uses nt for seed KS4 learning/reconstruction, alignment, CPU/CUDA matching/subtraction, overlap guards, novelty/PCA and heldout observations, caches, optional template consumers and Phy export. Other thresholds, scoring/shift defaults, spatial parameters and sample-rate/gain metadata are preserved.
- nt must be an odd integer >=21 because the selected raw-anchor holdout needs 17 samples plus +/-2 shifts. The centered offset is nt//2; NT-dependent overlap intervals follow the complete waveform. Sampling rate is never overridden or resampled by this config.
- Standalone default 61 is retained. Nondefault nt is recorded in seed/run plans; cached/restored models must match. Default 61 configuration identities retain their prior representation. Changing nt requires a new output and seed bank. Existing outputs/curation were not run against or modified.
- Source review identified the bounded manifest's XY-only geometry. The wrapper explicitly rejects nonzero/unknown active z coordinates rather than silently discarding 3D geometry. Extending the bounded geometry model is outside this change. KS1, KS4-compatible sorting and the Windows cuBLAS bridge are unchanged.
- Reviewed the task-only TeraSort diff against before-edit copies, including CUDA source, and the PP diff/new adapter/config/docs. One AST-only changed-file check passed for 33 Python files. No imports, unit/full suites, new tests, CUDA compilation, GPU/CPU sorting, simulations, scientific equivalence/quality comparisons or GUI/Phy runs were performed.
- Remaining user-owned validation: nt41 GPU execution and exported PCA/template dimensions, interrupted nt41 retry, actual legacy61 replay, and sorting-quality comparison on representative data. Implementation is not scientific validation. No commits or pushes.

## TeraSort main PR handoff
- Published [draft TeraSort PR #1](https://github.com/ayalab1/TeraSort/pull/1), targeting main, from isolated branch `codex/preprocess-bounded-nt` at `e40d169`. The standalone main checkout and original integration branch were not committed, rebased or merged.
- The backend PR includes opt-in preprocessed/full-binary-column input and flat Phy export, which the PP wrapper requires, together with bounded nt propagation. PP retains its nt=41 config; TeraSort's standalone default remains 61.
- Publication source review found and corrected the saved nt-dependent peak-alignment comparison on resume, plus the overlap-option help. These final corrections were reviewed in source only; the earlier 33-file AST-only check was not rerun and no extra runtime/quality checks were performed.
- TeraSort runtime/quality and large-session validation remain open. The PP wrapper/config are still local worktree changes, and TeraSort source has not been vendored into `sorter/TeraSort` in PP.
## Review status
- On 2026-10-12, changed TeraSort PR #1 to Ready for review at the user's request. Removed only the stale draft sentence from its body; the unrun runtime/scientific validation items remain explicit. GitHub returned `isDraft: false`, `state: OPEN`.
- No source changes, new checks, commits, pushes or main merge were performed for this status transition.
## Publish the PP backend branch
Goal: publish the bounded TeraSort backend from updated PreprocessPipeline main, retaining the completed WILD-NP and worker changes already in main. Scope: the existing backend/GUI/config/CLI integration and its documentation; TeraSort engine changes remain in TeraSort PR #1.

1. Confirm the current base against local and remote main and review the backend input/output contract.
2. Create `codex/terasort-backend` from the main-based implementation worktree and document the required TeraSort PR #1 runtime.
3. Commit only the task-owned backend/config/docs/log files and publish the branch without changing the original main or the older dirty integration worktree.
4. Open the main-targeted PP review PR, preserving the recorded validation limits and noting the TeraSort dependency.

Confirmed HEAD, local main and GitHub main are all `83ebca0`; no main merge or conflict resolution is required. This base already contains WILD-NP integration, headless workers and the MATLAB runtime-directory fix. Reuse the earlier source review and AST-only result; no new verification is planned for publication.
## PP publication handoff
- The backend branch is based on the verified latest main `83ebca0`; WILD-NP partitioning, headless workers and the MATLAB runtime-directory fix are already included. No conflict resolution or duplication of the old integration tree was needed.
- Reviewed the backend input/output contract and the added dependency documentation. The runtime must contain TeraSort PR #1 (or a later version with it). TeraSort source/environment are configured separately and are not vendored by this change.
- Reused the earlier source/diff review and AST-only check. No additional checks, new tests, dependency installation, GPU/CPU sorting, GUI runs or scientific validation were performed during publication.
- Runtime/quality and interrupted-resume checks remain open. Ready for code review records the review stage; the implementation has not been demonstrated on a large session.
## Branch execution before integration
Goal: execute TeraSort from PP PR #39 before considering main integration. The PR is now Draft and its body records this gate. TeraSort PR #1 remains a separate backend dependency.

1. Reuse the existing WILDX NP 600-second preprocessed AP recording, its XML/chanMap and recorded gain, preserving all earlier outputs and the source data.
2. Create a fresh execution directory and run-local YAML with nt=41; retain the selected scientific profile and independent physical-probe partitions.
3. Call this branch's `run_sorting_stage` with the published TeraSort PR source and the existing isolated runtime, then collect process completion and required Phy output metadata.
4. Record the actual result, failures or fixes and remaining scope. Keep PR #39 unmerged; this execution is not a scientific-quality or full-session validation.

Input: 12,000,000 samples, 20,000 Hz, 768 original binary columns, nominal 2.34375 uV/count, previously excluded columns 191 and 575. Original filtering/reference are already applied. Output: `D:/PreprocessPipeline/diagnostics/terasort_nt41_20261012_pr39/`.
Runtime budget: the local GPU is an RTX 3080 Ti with 12 GiB. The run-local config sets `vram_limit_gb: 8` to leave allocation headroom; this caps the CuPy pool, not all GPU allocations. nt=41 and all other selected scientific settings are retained. No re-preprocessing, dataset expansion, benchmarks, simulations, ground-truth evaluation or automatic main merge are part of this run.
## Main refresh after execution cancellation
- The user cancelled execution. Interrupted the foreground process (exit code 1) and confirmed no Python processes carrying this run's unique output marker remained. Probe1 seed learning had completed at nt41; no completed sorting/Phy validation result was reported. Partial files stay in the fresh execution directory; no further run is requested.
- Updated origin/main is `c17fbc6`, including PR #37's supplied-channel-map preservation and manual-session ownership fixes.

1. Merge the updated main into `codex/terasort-backend` without rewriting branch history.
2. Resolve overlapping integration changes while preserving main's supplied physical channel maps, source-column IDs and ownership behavior, together with the bounded TeraSort route/config.
3. Inspect the scoped merge diff and publish the branch update. Keep PR #39 Draft and perform no further sorting, GPU execution or checks.
## Main refresh handoff
- Merged main `c17fbc6` without conflicts. Reviewed the overlapping sorting-stage/worker/GUI changes and the channel-layout caller contract. Main's verified physical-map selection, geometry/hash checks and manual-session ownership remain intact; the bounded TeraSort route, nt41 config and original-column manifest/export contract remain present.
- The user-cancelled run is not a successful execution result. No sorting, GPU execution, tests or lightweight checks were run after cancellation; the merge was reviewed in source/diff only. The partial new execution directory is retained and is not published as code or validation evidence of completion.
- PR #39 stays Draft and unmerged. Its execution gate remains for a future user-requested run; current work is limited to updating and publishing the integration branch.