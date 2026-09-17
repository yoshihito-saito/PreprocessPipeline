# Channel consistency and Phy export

Date: 2026-09-17. Uncommitted work on `bug/channel-consistency-phy-export`,
based on `05803e9`.

Plan: [channel consistency and Phy export](../implementation_plan/2026-09-17-channel-consistency-phy-export.md).

## Changes and rationale

- Standalone GUI postprocess follows the selected sorting's binary by default,
  supports an explicit raw/concatenated recording, and uses one consistent path,
  width, dtype, byte offset and sampling rate. Invalid metadata is reported before
  waveform extraction. Explicit maps take precedence; otherwise use the map
  beside the selected recording. Execution snapshots preserve these selections.
- XML skips, omission from nonempty spike-detection groups, map connectivity and
  manual rejects agree across preview, preprocessing, sorting and postprocess.
  Original binary columns are retained and existing zero-bad preprocessing is
  preserved. Good-channel analysis keeps original IDs; partial maps cannot
  silently shrink the full preprocessing binary. Invalid maps/IDs fail explicitly.
- All-bad probes produce a skipped partition with `no_active_channels`. Entirely
  bad sessions skip GPU selection and downstream analysis normally. Execution,
  resume and discovery recognize that result without selecting stale outputs.
  Existing manifests never fall back to globbing older runs when no targets remain.
- Postprocess optionally filters/references the selected good channels lazily.
  No additional processed binary is mandatory. The notebook helper no longer
  filters an already processed final binary a second time.
- Phy display templates are re-extracted with a centered window in native binary
  units, using the same selected spikes and boundary padding as Phylib. Scientific
  SI templates/PCA/metrics retain their 1 ms before / 2 ms after window and units;
  spike samples are not shifted. Compact copied exports and full external-binary
  exports each have matching maps, paths, dtype, rate and offsets, including after
  atomic publication to the final output directory.
- Native KS2.5 exports original input indices and `Wrot/scaleproc` whitening;
  incompatible custom exporters are detected before redirection/temporary-file
  cleanup. Its skip-preprocessing path uses a compact input matching `Nchan`.
  KS4 receives distinct probe/shank groups and exclusions in the actual sorter
  recording's index space. Native redirection also preserves dtype/byte order.
- Explicit analyzer-cache reuse is bound to recording graph/properties and
  source sorting/label identity. Old unverified or mismatched caches fail before
  source-label mutation. All CSV/TSV source property files are covered, including
  `cluster_info.csv` labels accepted by SI.
- Anatomical CSV follows XML electrode-group/channel order, retains bad-channel
  positions, and handles repeated shank IDs on different probes. Loading never
  rewrites the CSV. Labels are bound to the current session/XML; invalid or excess
  cells are reported. Loaded map exclusions survive geometry edits. Automatic
  GUI probe assignments derive all XML groups instead of a fixed eight groups.

README and both notebook clients were updated. Task regression tests that were
previously hidden by the repository's broad test ignore rule are included with
the change; no unrelated test assertions were weakened.

## Verification

All Python verification used the existing `phy2` environment (Python 3.11,
SpikeInterface 0.103.2). Synthetic data and temporary directories only; no live
sorting run, session binary or existing analysis output was modified.

Final focused regressions: **45 passed**, two expected dummy-probe warnings.
Real SI/Phylib round trips cover 20/30 kHz, unequal gains and offsets, interleaved
bad channels, nonmonotonic/partial maps, full/compact binary output, header offsets,
relative/absolute paths, publication renaming, centered boundary waveforms,
unchanged spike samples/source bytes/scientific features, lazy filtering, and
the wrong-same-named-binary flat-waveform reproduction. Sorter adapter tests use
real SI recordings with the expensive sorter call mocked.

```bash
QT_QPA_PLATFORM=offscreen /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/postprocess/test_channel_contract.py \
  tests/preprocess/test_channel_contract_sorting.py \
  tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_zeroes_and_excludes_bad_channels_for_sorter \
  tests/preprocess/test_pipeline_rhd_integration.py::test_xml_spike_group_omissions_join_manual_exclusions \
  tests/preprocess/test_pipeline_rhd_integration.py::test_partial_map_cannot_shrink_preprocessing_binary
```

Final GUI/cache follow-up: **24 passed** (the cache test overlaps the 45 above).
Includes real Qt canvas rendering of a zero-based-only map and consistent
nonpositive connectivity, plus CSV label changes invalidating cached analysis.

```bash
QT_QPA_PLATFORM=offscreen /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/postprocess/test_channel_contract.py::test_cache_reuse_rejects_another_input_before_changing_labels \
  tests/preprocess/test_gui_channel_consistency.py
```

Broad relevant suite: **294 passed, 17 failed**, two dummy-probe warnings.
The final follow-up above verifies the subsequent small CSV/cache and canvas
corrections. No new broad-suite regression was observed.

```bash
QT_QPA_PLATFORM=offscreen /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/postprocess/test_channel_contract.py \
  tests/preprocess/test_channel_contract_sorting.py \
  tests/postprocess/test_postprocess_analyzer_sparsity.py \
  tests/postprocess/test_phy_export_paths.py \
  tests/postprocess/test_postprocess_target_resolution.py \
  tests/postprocess/test_postprocess_overwrite_skip.py \
  tests/postprocess/test_attach_existing_sorting_result.py \
  tests/postprocess/test_metric_enrichment.py \
  tests/postprocess/test_postprocess_metafile_defaults.py \
  tests/postprocess/test_unit_split.py \
  tests/preprocess/test_sorter_partitions.py \
  tests/preprocess/test_kilosort_partition_channels.py \
  tests/preprocess/test_sorter_runner_matlab_path.py \
  tests/preprocess/test_recording_probe_attach.py \
  tests/preprocess/test_recording_selected_transform.py \
  tests/execution/test_input_identity.py \
  tests/execution/test_local_backend.py \
  tests/execution/test_gpu_selection.py \
  tests/preprocess/test_gui_channel_consistency.py \
  tests/preprocess/test_anatomical_map_helpers.py \
  tests/preprocess/test_gui_config_model.py \
  tests/preprocess/test_gui_preflight.py \
  tests/preprocess/test_gui_move_outputs.py \
  tests/preprocess/test_pipeline_artifact_helpers.py \
  tests/preprocess/test_pipeline_rhd_integration.py \
  tests/preprocess/test_pipeline_mixed_sources.py
```

All 17 failures were separately reproduced against an isolated `git archive HEAD`
of the unchanged base commit in `/tmp/preprocess-channel-baseline.VYgjCg`.
Ignored preexisting tests were copied into that archive. The intentional new
expectation forwarding `[1, 3]` to the sorter was restored to its original `None`
assertion for this baseline check.

```bash
# In the isolated baseline directory:
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/preprocess/test_pipeline_artifact_helpers.py \
  tests/preprocess/test_pipeline_rhd_integration.py \
  tests/preprocess/test_pipeline_mixed_sources.py
# 15 passed, 12 failed (preexisting artifact/default/old-API expectations).

QT_QPA_PLATFORM=offscreen /local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/preprocess/test_sorter_runner_matlab_path.py::test_normalize_kilosort4_params_coerces_use_amplitude_feature \
  tests/preprocess/test_sorter_runner_matlab_path.py::test_resolve_kilosort4_import_root_accepts_repo_root_and_package_dir \
  tests/preprocess/test_sorter_runner_matlab_path.py::test_run_sorter_cli_kilosort4_without_config_passes_none_config_path \
  tests/preprocess/test_gui_preflight.py::test_preflight_reports_local_cmr_radius_without_neighbors \
  tests/preprocess/test_gui_preflight.py::test_preflight_accepts_wider_local_cmr_radius_for_middle_finger
# 5 failed, matching the other broad-suite failures.
```

Independent read-only reviews verified the display-template numerical means,
MATLAB whitening algebra, KS2.5 compact-map redirection, GUI anatomy/exclusions,
and real filtered/unfiltered binary-backed cache reuse. Final AST parsing of all
29 changed Python files and notebook JSON validation passed. Final
`git diff --check` passed.

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B - <<'PY'
import ast
import json
from pathlib import Path
import subprocess
paths = subprocess.check_output(['git', 'diff', '--name-only', '--', '*.py'], text=True).splitlines()
for name in paths:
    ast.parse(Path(name).read_text(encoding='utf-8-sig'), filename=name)
notebook = json.loads(Path('Run_preprocessSession.ipynb').read_text())
assert notebook['nbformat'] == 4 and isinstance(notebook['cells'], list)
print(f'Parsed all {len(paths)} changed Python files and notebook JSON successfully.')
PY
git diff --check
```

## Follow-up: direct CLI all-bad skip (2026-09-17)

Uncommitted follow-up on the same branch. The user confirmed that different
recording inputs will not be mixed in this workflow, so no heterogeneous-input
policy was added. Existing staged work was preserved.

The direct CLI previously performed dependency setup before discovering that all
selected channels were excluded. A new regression reproduced the failure before
the source edit: even an all-bad KS1 input raised a missing-Kilosort-path error.

Channel selection now runs before dependency/config/log/binary setup.
`NoActiveChannels` identifies a valid empty selection, and only the CLI catches
it to print `reason=no_active_channels` and return with exit status 0. Nonempty
execution keeps its existing Path return contract. Invalid channel metadata,
nonpositive counts, and nonfinite/nonpositive sampling rates remain errors.
Skipped CLI runs leave existing outputs and session manifests untouched, including
when `--remove-existing-folder` is present. They create no output or skip manifest;
stage orchestration retains its existing persistent skip manifests.

Verification in phy2:

```bash
# Before source edits: reproduced failure (1 failed, stopped at first failure).
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/preprocess/test_channel_contract_sorting.py -k cli_all_bad --maxfail=1

# After source edits: 41 passed, 23 deselected.
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/preprocess/test_channel_contract_sorting.py -k cli

# Relevant regression suite: 127 passed, 3 unchanged baseline failures.
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/preprocess/test_channel_contract_sorting.py \
  tests/preprocess/test_sorter_partitions.py \
  tests/preprocess/test_kilosort_partition_channels.py \
  tests/preprocess/test_sorter_runner_matlab_path.py \
  tests/execution/test_input_identity.py \
  tests/execution/test_local_backend.py
```

The 41 cases cover map exclusions, explicit rejects, their union, excluded active
subsets, KS1/KS2.5/KS4, and 20/30 kHz. Six cases invoke the actual CLI in separate
Python processes and verify exit status 0 without installed sorter paths. Other
cases verify existing binary/output/manifest bytes remain unchanged, invalid
metadata fails, and nonempty dependency errors and unrelated sorter exceptions
are not suppressed. The three broad-suite failures are the same KS4 parameter,
import-root helper, and CLI default-path baseline failures listed above. An
independent read-only review found no blocking issue in this follow-up.

Skipped runs intentionally do not validate unused binary/configuration inputs.
MATLAB/GPU sorting remains unexecuted; the CLI subprocesses verify the no-work
path directly. No live data or running jobs were touched.

## Limits

- MATLAB/GPU sorting itself was not executed. Native adapter behavior is tested
  with mocked sorter execution and source-level numerical review. Interactive
  desktop Phy was not launched; actual Phylib loading and offscreen Qt tests ran.
- Existing unrelated failures remain as described above. Whole postprocess test
  collection also has a preexisting import of missing `src.postprocess.ks4_diagnostics`.
- Independent external recordings need separate calls. One multi-target call
  assumes a shared channel layout and time origin; compatible sizes/rates cannot
  prove that an arbitrarily selected raw binary represents the same timeline.
- Native drift-corrected KS2.5 templates describe the reference position; lazy
  postprocess filters/reference can differ from Phy's raw view. SI physical
  `amplitudes.npy` values are not KS fitting coefficients and must not be passed
  through Phylib's coefficient-based `get_amplitudes_true()` conversion.
- Cached binary identity uses path/size/mtime rather than hashing a potentially
  enormous recording. Old caches need a one-time verified rebuild.
- Session orchestration and direct CLI provide normal all-bad skips. Direct
  low-level `execute_sorting_job` signals an empty selection with
  `NoActiveChannels`, a `ValueError` subclass; successful calls still return Path.
