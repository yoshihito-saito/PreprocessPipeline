# TeraSort sorting-stage integration

## Goal and scope
Use the existing PreprocessPipeline preprocessing and postprocessing stages with the standard TeraSort session sorter. Preserve existing Kilosort paths, TeraSort raw-input defaults, public calls, and old checkpoint identity. Keep source recordings unchanged.

## Implementation steps
1. Add explicit source-column mapping and an opt-in preprocessed-input contract to TeraSort manifests; carry both through reads, seed learning, resume identity, and Phy export.
2. Preserve the selected TeraSort algorithm while bypassing repeated reference/filter operations only for explicitly preprocessed input.
3. Add a PreprocessPipeline adapter that validates effective channel geometry, launches the configured TeraSort environment, and registers the exported Phy folder; connect sorter selection and persistent retry.
4. Document configuration, input/channel contracts, and validation boundaries; inspect both task diffs and perform at most one lightweight changed-file syntax check.

## Decisions and boundaries
- Standard session-sort is the selected integration route. The separate KS4-compatible sort and bounded KS1 routes are unchanged.
- Channel IDs name columns of the actual input binary, after any upstream remapping. Selected geometry follows the explicit channel map.
- New input contracts are opt-in; omitted fields retain legacy source records and signal processing.
- No dependency installation, new tests, sorter runs, benchmarks, broad validation, commits, or pushes are planned.

## Completion
- Implemented in the two dedicated worktrees: `D:/PreprocessPipeline-integrating-terasort` and `D:/TeraSort-preprocess-pipeline-integration`. Original checkouts and unrelated work were left unchanged; no commit or push was made.
- PreprocessPipeline now offers `terasort` in the GUI/CLI, builds a manifest from the effective channel map, launches the separate TeraSort environment, and passes a flat Phy folder to the existing postprocessing path. Geometry is matched by actual binary column ID; probe/shank pairs remain distinct.
- TeraSort accepts opt-in full binary width, selected source columns, and preprocessed input. Raw-input defaults and omitted-field source identity remain unchanged. Preprocessed preview learning requires pinned KS4 4.1.7 and keeps whitening while bypassing the extra CAR/high-pass pass; dense INT16 matching keeps its existing headroom quantization.
- Native checkpoints and pinned inputs/settings stay under the session's `.terasort/` directory. Persistent retries reuse that run; standalone retries require `--terasort-resume`. Completed Phy exports are preserved.
- Reviewed the task diffs in both repositories and the final corrections. Ran one AST-only syntax check: all 20 changed Python files parsed successfully (8 PreprocessPipeline, 12 TeraSort). No application imports, unit/regression tests, dependency installation, GPU sorting, benchmarks, notebooks, or dataset validation were run.
- Usage and input contracts are documented in `docs/cli.md` and TeraSort's `docs/session_sort.md` / `docs/kilosort_export.md`.
- Remaining validation: configure an environment containing the updated TeraSort checkout, then exercise a representative recording, interrupted retry, and downstream Phy/postprocessing. Existing-workflow compatibility has source review only, not an executed regression result. WILD validity-mask integration and end-to-end large-recording disk behavior remain outside this change.

## Completion of nt/seed-backend forwarding (2026-10-15)

- Added optional `nt` (odd >=21) and `seed_backend` values to the existing TeraSort YAML adapter. Explicit cuBLAS accepts nt=41/61; invalid combinations fail before input pinning. Omitted keys preserve the existing nt=61/deep_tiled session defaults.
- Reviewed the GUI's per-session YAML copy, sorter config load, subprocess command, pinned `.job.json`, TeraSort CLI, seed plan/report and bounded run nt. Existing Kilosort4 config is not inherited. The GUI copy and source template are documented separately.
- One focused tracked pytest passed for nt41/cublas command forwarding, job pinning, unchanged resume and changed-nt rejection. The TeraSort Windows GPU smoke passed for 41/61 taps; no PP GUI/full preprocessing/sorting run or Linux execution was performed. No commits or pushes were requested.
