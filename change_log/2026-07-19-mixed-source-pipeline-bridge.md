# Mixed Intan/Open Ephys ADC Handling

## Date and Git State

- Date: 2026-07-19
- Branch: `feature-multiday-subepoch`
- Commit: uncommitted
- Plan: [2026-07-19 mixed Intan/Open Ephys sessions and Open Ephys ADC separation](../implementation_plan/2026-07-19-mixed-source-openephys-adc.md)

## What Changed

- Added Open Ephys stream classification from `structure.oebin` channel
  metadata, with explicit ephys/ADC channel indices and memory-usage streams
  ignored.
- Updated subepoch discovery and `AcquisitionCatalog` construction so one run
  can contain Intan and Open Ephys subepochs.
- Added per-subsession catalog metadata for source type, total source channels,
  ephys channels, ADC channels, ADC channel names/indices, and analog source
  paths.
- Open Ephys recordings are loaded through SpikeInterface and sliced to ephys
  channels before preprocessing, raw save, LFP, and sorting.
- Open Ephys recordings are re-renamed to contiguous `0..N-1` channel ids
  after ephys-only selection, so downstream bad-channel handling, probe
  attachment, sorter partitions, and final binary column order share the same
  ephys-only index space.
- Non-prefix Open Ephys ephys selections now write the local output
  `chanMap.mat` in final ephys-only dat column coordinates when a chanMap is
  provided. Probe attachment, bad-channel loading, artifact grouping, sorter
  partitioning, sorter launch, and postprocess recording reconstruction all use
  this canonical local chanMap, so source-coordinate chanMap rows are converted
  to final ephys-only dat columns before sorting and later waveform analysis.
- Manual `reject_channels` are remapped through the same source-to-final channel
  map when a source-coordinate Open Ephys chanMap is remapped. XML skipped
  channels remain final/session-channel coordinates.
- Merge points now use catalog sample counts, so Open Ephys `continuous.dat`
  files containing ADC channels do not get divided by the ephys-only channel
  count.
- Analog sidecar writing can extract selected ADC columns from interleaved Open
  Ephys `continuous.dat`, pad epochs with fewer ADC channels, and zero-fill
  epochs without analog data.
- Mixed Open Ephys TTL inputs are passed per subepoch, with `None` entries for
  non-OE epochs.
- Sorter launch now passes `sampling_frequency`, `num_channels`, `dtype`,
  `gain_to_uV`, and `offset_to_uV` from the final pipeline state instead of
  letting sorter setup infer channel count from XML.
- Kilosort 1/2.5 chanMap override now keeps `chanMap` synchronized with
  `chanMap0ind` and filters copied chanMaps to the final recording channel
  range without renumbering valid final channel ids.
- Multi-day staging CSV/manifest rows now record `source_total_channels`,
  `source_ephys_channels`, `source_adc_channels`, and `binary_n_channels`.

## Why

Real datasets can mix Intan and Open Ephys recordings across subepochs. Some
Open Ephys acquisition-board streams store ADC channels in the same
`continuous.dat` as ephys channels. Without source-aware classification, ADC
channels could be included in the ephys binary and sent to sorting.

## Verification

- `python -m py_compile src/preprocess/io.py src/preprocess/metafile.py src/preprocess/mergepoints.py src/preprocess/recording.py src/preprocess/pipeline.py src/preprocess/multiday.py tests/preprocess/test_openephys_stream_classification.py tests/preprocess/test_pipeline_mixed_sources.py tests/preprocess/test_recording_selected_transform.py tests/preprocess/test_multiday.py`
  - Result: passed
- `python -m py_compile src/preprocess/pipeline.py src/postprocess/pipeline.py src/preprocess/sorter_runner.py tests/preprocess/test_sorter_partitions.py tests/postprocess/test_postprocess_analyzer_sparsity.py`
  - Result: passed
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_openephys_stream_classification.py tests/preprocess/test_io_source_selection.py tests/preprocess/test_pipeline_mixed_sources.py tests/preprocess/test_recording_selected_transform.py tests/preprocess/test_multiday.py tests/preprocess/test_events_neurocode_compat.py tests/preprocess/test_intan_rhd_header.py`
  - Result: `61 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_pipeline_mixed_sources.py tests/preprocess/test_recording_selected_transform.py tests/preprocess/test_sorter_partitions.py tests/postprocess/test_postprocess_analyzer_sparsity.py`
  - Result: `32 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_partitions.py`
  - Result: `10 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_partitions.py tests/postprocess/test_postprocess_analyzer_sparsity.py`
  - Result: `17 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_partitions.py tests/preprocess/test_openephys_stream_classification.py tests/preprocess/test_pipeline_mixed_sources.py tests/preprocess/test_multiday.py tests/postprocess/test_postprocess_analyzer_sparsity.py`
  - Result: `41 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_preflight.py tests/preprocess/test_recording_probe_attach.py tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_zeroes_and_excludes_bad_channels_for_sorter`
  - Result: `17 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_runner_matlab_path.py::test_kilosort1_chanmap_override_copies_given_chanmap_and_restores tests/preprocess/test_sorter_runner_matlab_path.py::test_execute_sorting_job_kilosort4_filters_unknown_params_and_skips_matlab tests/preprocess/test_sorter_runner_matlab_path.py::test_execute_sorting_job_applies_preprocessing_to_sorter_input`
  - Result: `3 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_uses_kilosort4_output_prefix tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_passes_matlab_max_workers_to_sorter tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_uses_basename_dat_for_sorter_and_marks_preprocessed_input tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_zeroes_and_excludes_bad_channels_for_sorter`
  - Result: `4 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_pipeline_rhd_integration.py::test_run_preprocess_session_generates_digital_events_from_openephys_ttl`
  - Result: `1 passed`
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_pipeline_rhd_integration.py`
  - Result: `10 passed, 11 failed`.
  - Remaining failures are unrelated to mixed-source/OE ADC handling: stale
    `PreprocessConfig` artifact-removal keyword aliases, an existing save-raw
    eager-load expectation, and high-amplitude artifact job-count defaults.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_sorter_runner_matlab_path.py`
  - Result: `38 passed, 3 failed`.
  - Remaining failures are unrelated to channel-coordinate safety:
    Kilosort4 boolean parameter coercion, missing legacy
    `_resolve_kilosort4_import_root`, and CLI default Kilosort4 path
    expectation.
- Read-only `jq` checks on real Open Ephys metadata:
  - ayadataB4 Day22 postsleep acquisition board: `total=128`,
    `ephys=128`, `adc=0`.
  - cbsuruizfs RM018 day41 acquisition board: `total=200`,
    `ephys=192`, `adc=8`.
- `git diff --check`
  - Result: passed.

## Known Limitations and Next Steps

- Full `tests/preprocess/test_pipeline_rhd_integration.py` still has unrelated
  legacy failures around stale `PreprocessConfig` artifact-removal keyword
  aliases, an existing save-raw eager-load expectation, and high-amplitude
  artifact job-count defaults.
- Mixed Intan/Open Ephys analog event export requires ADC sample rates to be
  compatible with the merged timeline; incompatible layouts should be treated
  as early-fail cases rather than silently resampled.
- Open Ephys ADC extraction preserves raw 16-bit sample words in the generated
  `analogin.dat`; voltage-unit rescaling from `bit_volts` is not yet applied.
