# Kilosort4 partition channel exclusions

- Date: 2026-09-10; uncommitted.
- Plan: [partition channel exclusions](../implementation_plan/2026-09-10-kilosort-partition-channels.md).

## Change
Do not send original binary channel exclusions to Kilosort4 after selecting a channel subset. SpikeInterface exports a compact probe map; reapplying original indices caused `Channel '48' was not in probe['chanMap']` in the Day10–Day245 probe1 Run. Full-width preprocessed recordings retain their bad-channel list. Scientific parameters and partition membership are unchanged.

## Verification
Interpreter: `/workdir/ys2375/miniforge3/envs/phy2/bin/python`.

- `-m pytest -q tests/preprocess/test_kilosort_partition_channels.py` before fix: both raw/preprocessed regression cases failed on the original-index bad_channels list.
- `-m pytest -q tests/preprocess/test_kilosort_partition_channels.py tests/preprocess/test_sorter_partitions.py tests/preprocess/test_sorter_runner_matlab_path.py::test_execute_sorting_job_preprocessed_input_skips_preprocess_and_channel_slice`: 13 passed after fix.
- `-m pytest -q tests/preprocess/test_kilosort_partition_channels.py tests/preprocess/test_sorter_runner_matlab_path.py tests/preprocess/test_sorter_partitions.py`: 52 passed, 3 failed. The failures concern feature bool coercion, a missing import-root helper, and the CLI default package path. All three also fail with the HEAD sorter_runner source loaded in memory (via `git show HEAD:src/preprocess/sorter_runner.py`, UTF-8-sig decoding, and pytest `-k 'coerces_use_amplitude_feature or accepts_repo_root_and_package_dir or without_config_passes_none_config_path'`). They are pre-existing and outside this fix.
- `git diff --check`: passed.

No GPU sorting job launched. Retry sorting on Run `run-20260910T203235Z-7540aa29` to reuse completed preprocessing. Existing failed sorter output is retained; retry selects a new output folder.
