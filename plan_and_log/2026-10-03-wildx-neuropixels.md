# WILDX Neuropixels input and ten-minute validation

## Goal and scope
Automatically recognize WILDX NP recordings from applied-probe receipts, derive original-column channel metadata from the applied site selection and the NP1 geometry, and run the existing preprocessing/sorting pipeline on a local copy of the first 600 seconds of one complete recording. Preserve existing Intan, Open Ephys, WILD, Kilosort, and TeraSort behavior and the saved binary width. Fix the single-subepoch MergePoints reuse shape mismatch.

## Implementation steps
1. Add a strict WILDX NP metadata reader and a model-specific NP1 site geometry; connect discovery and acquisition metadata to the existing pipeline.
2. Generate the effective XML/chanMap from applied settings, keep inactive binary slots, and route the recorded LF stream with explicit voltage and timing provenance.
3. Normalize single-epoch MergePoints arrays on reuse and document the new input contract.
4. Copy only the first 600 seconds of the confirmed complete segment to a dedicated local directory, then run the requested preprocessing/sorter route using that local input and inspect its final results.

## Decisions and validation boundary
- Work in the existing `feature/integrating-terasort` worktree; preserve its earlier TeraSort integration changes. One agent; no commits or pushes.
- Raw server data remain read-only. The excerpt retains applied receipts and source metadata, records its exact copied rows, and excludes unpaired tails beyond the excerpt.
- The applied receipt identifies the recording format; the probe model identifies geometry. Saved stream metadata determine file width, order, and rates.
- The NP1 nominal ADC count scale and gain-index table were confirmed in the Open Ephys source. Per-channel calibration, the ADC DC zero point, and AP/LF time alignment remain unverified; retain the recorded LF stream with its own time-origin metadata and block state scoring/sensor export.
- Validation is limited to the requested ten-minute real-data excerpt and the concrete single-epoch reuse issue. No full-session runs or broad suites.

## Completion
Implemented automatic NP discovery, applied-site geometry/channel maps, nominal count-to-voltage provenance, recorded LF export, and single-subepoch MergePoints reuse. The requested local 600-second preprocessing-to-TeraSort execution completed with exit code 0 after the documented metadata/environment corrections.

- Output inspection: AP 12,000,000 samples at 20,000 Hz and LF 750,000 samples at 1,250 Hz, both retaining 768 binary columns. Sorting retained 383 channels per probe (766 total), excluding original columns 191 and 575 from this recording's applied receipts. Phy column IDs and coordinates matched the generated map exactly.
- TeraSort exported all 10 shards and 436,140 spikes. The arrays contain 141 distinct assigned cluster IDs; its export summary reports 365 registered groups and 366 templates. All cluster labels remain unsorted. Its experimental quality flag is hard-coded false; this run establishes execution, not scientific sorting quality.
- The manifest retained AP gain 2.34375 microvolts/count, full source width 768, and `input_is_preprocessed=true`. LF metadata retained its separate gain 4.6875 microvolts/count and unverified AP alignment. Spike samples ranged from 85 to 11,999,952, within the copied interval.
- Reusing the existing single-subepoch metadata and already-written AP/LF succeeded. Source/diff inspection and the requested real-data execution/output review were performed. No new tests, package-wide suites, notebooks, full-session runs, GUI/interactive Phy, or waveform/ground-truth quality assessment were performed.
- First-release scope remains one complete merged NP1 segment. Gap-aware multi-segment handling, sensor decoding, per-channel calibration, ADC DC zero point, and AP/LF common-time validation remain outside this implementation.
- Reusable input, configuration, final result and output review: `D:/PreprocessPipeline/diagnostics/wildx_np_10min/`. The successful continuation log is `validation_run_resume1.log`; earlier failure logs remain available. Sorting output is `output/RM034_261002_day04_10min/Terasort_2026-10-03_114339`. No commits or pushes.

## Execution preparation
- Local excerpt copied successfully: AP 12,000,000 rows and LF 750,000 rows at 768 columns, under `D:/PreprocessPipeline/diagnostics/wildx_np_10min/input/RM034_261002_day04_10min/segment_0001`. SHA-256 values and original metadata are retained.
- Installed TeraSort only in its integration worktree `.venv`, using the pinned Torch 2.10.0 CUDA 12.8, KS4 4.1.7, and CuPy requirements. The preprocessing conda environment is unchanged.
- Initial NP support is one complete merged NP1 segment; gap-aware multi-segment support remains outside this first implementation. NP output gain/offset are optional trailing result fields for separate sorting/postprocessing Stages.
- Source/diff inspection completed before the requested foreground preprocessing-to-TeraSort run; the selected scientific filtering/reference and sorter profile are unchanged.
- The first real-data attempt reached CMR but failed because SI channel aggregation rejects equal relative coordinates on separate probes. Aggregate NP runs without locations on private clones, then restore real coordinates and original IDs. Apply the existing filter/reference independently per physical NP probe, preserving the configured radius and operator.
- Direct-interpreter retries failed at binary writing: the four-worker run raised a Windows MKL threadpool initialization error, and the single-worker run exited with the same native error code. The next run activates the existing preprocessing conda environment and restores the original four-worker configuration. No scientific settings or environment packages were changed.
- The activated-environment run completed AP and recorded LF writing at 768 columns (18,432,000,000 and 1,152,000,000 bytes). TeraSort selected 766 channels and completed its initial 351-template learning, then failed on a CuPy reduction compiled with mismatched CUDA 13.1 headers.
- Installed only `nvidia-cuda-runtime-cu12==12.9.79` in the isolated TeraSort environment, matching its NVRTC 12.9. The [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html#cupy-always-raises-nvrtc-error-compilation-6) documents these wheel-provided headers. Resume through the existing preprocessing/sorting stage APIs, reusing AP/LF and the `Terasort_2026-10-03_114339` checkpoint; the configured filtering, reference, and chosen sorter policy stay unchanged.

## Phy display correction
Goal: prevent Phylib from replacing overlapping coordinates on separate NP probes with a linear layout. Preserve the physical geometry, binary channel IDs, spikes, and curation.
Scope: opt-in display correction for the TeraSort integration; legacy Kilosort metadata behavior remains unchanged. No re-sorting, raw-data changes, or Phy cache/curation deletion.
1. Add opt-in horizontal separation to the existing Phy channel-metadata writer, retaining local coordinates and display-offset provenance.
2. Enable it for TeraSort exports and repair the existing ten-minute output using the same writer.
3. Review the changed source and perform one focused geometry readback; document the restart needed for the open Phy GUI.
Completion: the existing ten-minute Phy export was repaired with the same opt-in writer now used by future TeraSort integration outputs. The 766 input coordinates had 624 distinct pairs across two probes; each probe had 383 distinct pairs. Phylib's duplicate-position fallback therefore produced the observed linear layout.
- The repaired display has 766 distinct pairs, with horizontal offsets 0 and 148 micrometres. `channel_positions_local.npy` exactly matches the selected source chanMap rows, and `channel_positions_display.json` records the display convention. Within-group geometry and all y coordinates are preserved.
- Source/diff inspection and one focused geometry readback passed. No sorter, pipeline, suite, or additional check was run. Legacy metadata calls retain the default-disabled correction.
- Spike assignments, channel maps, curation and Phy caches were not rewritten or deleted. Save and close the currently open GUI, then reopen with `--clear-cache` so derived geometry caches refresh. Visual confirmation in the restarted GUI remains pending. No commits or pushes.

## GUI launch and NP preview correction
Goal: make persistent GUI Runs use the same checkout as the launched GUI, show receipt-derived NP geometry, and keep the user's selected day03 folder as Basepath and the output basename.
Scope: explicit project-root discovery for GUI modules, NP preview without an explicit chanMap, bounded discovery through `NP_SESSION_*/segment_*`, and a separate local output/workspace. Preserve the ten-minute outputs, Phy curation, raw server data, and the persistent output-contract guard.
1. Prefer an explicitly supplied root-search origin and supply each GUI module's file as that origin.
2. Build the NP preview from the existing applied-receipt reader without requiring or applying an unrelated XML.
3. Share NP directory discovery for the selected day folder, capture folder, or segment; keep existing ordering/exclusions and single-segment validation.
4. Save the selected day03 folder configuration with empty XML/chanMap paths and a fresh local root; inspect the diff and perform one focused checkout-resolution check.
Decision: the user requested retaining the currently selected day03 folder after the initial capture-folder guidance. Extend only the known WILDX capture/segment hierarchy; do not rewrite Basepath, add arbitrary recursive discovery, or combine segments without timing support.
Completion: explicit GUI root discovery and receipt-derived NP previews are implemented. Shared bounded NP discovery now recognizes `day/NP_SESSION_*/segment_*` while retaining the chosen Basepath and day-folder output basename. It feeds GUI preview/preflight/Check order and pipeline workers through the existing discovery APIs; existing ordering, ignore-folder handling, strict receipts, and single-complete-segment restrictions remain in force.
- Saved `D:/PreprocessPipeline/diagnostics/wildx_np_10min/gui_day03_config.json` with the user's selected `R:/shared/data/Neuropixels/NP1.0/RM034/RM034_261001_day03` Basepath. XML/chanMap/probe assignments are empty, digital/analog/state scoring are disabled, and `D:/PreprocessPipeline/preprocess_tmp/wildx_np_day03_gui` is a separate output/workspace with overwrite disabled. The basename remains `RM034_261001_day03`; this configuration selects the full day03 recording rather than the ten-minute excerpt.
- Scoped source/diff review completed, including the selected-day discovery extension. The one focused GUI/controller checkout-resolution import check passed in the existing activated preprocessing environment before that extension: both roots resolved `D:/PreprocessPipeline-integrating-terasort` despite the environment's older editable install. The later hierarchy edits were inspected in source only. No additional check, GUI interaction, worker submission, full-session processing, sorting, or package change was performed.
- The failed Run stopped at the output-contract guard before preprocessing. Existing ten-minute outputs, Phy curation/caches, server data, and contract validation were preserved. Restart the integration-worktree GUI, load the new configuration, and use Preprocess only with Run sorter enabled. No commits or pushes.
