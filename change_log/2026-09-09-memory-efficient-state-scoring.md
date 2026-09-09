# Memory-efficient state scoring

Date: 2026-09-09. Uncommitted, based on HEAD 99a76ea.
Requested branch `feature/memory-efficient-state-scoring` could not be created:
the environment mounts `.git` read-only. Source edits remain in the working tree.

Plan: [memory-efficient state scoring](../implementation_plan/2026-09-09-memory-efficient-state-scoring.md).

Candidate selection now reads individual channels from a read-only LFP memmap,
downsamples before float64 conversion, and runs at most four concurrent candidate
evaluations. All candidates and their ordering remain intact. Workers return
compact SW histograms or theta mean spectra instead of retaining candidate time
series. Only selected SW/theta full-rate traces are copied into the output.
Final scoring also converts selected samples after downsampling. EMG, spectral
calculations, thresholds, output schemas, and existing v7.3 handling are unchanged.

## Real-data verification

Input: `sorting_temp/test_rec_260509`, complete 291,756,032-byte LFP, session
metadata, and state-scoring settings from `preprocess_run.yaml`. No pulses file
is present. Both executions requested 128 workers. A pre-edit module was archived
at `/tmp/state-memory-verification/baseline.py`; the harness at
`/tmp/state-memory-verification/run.py` ran old and new implementations in separate
directories with read-only input symlinks. Existing session outputs were untouched.

Commands executed from the repository:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v -o /tmp/state-memory-verification/baseline.time /workdir/ys2375/miniforge3/envs/phy2/bin/python /tmp/state-memory-verification/run.py baseline
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /usr/bin/time -v -o /tmp/state-memory-verification/optimized.time /workdir/ys2375/miniforge3/envs/phy2/bin/python /tmp/state-memory-verification/run.py optimized
/workdir/ys2375/miniforge3/envs/phy2/bin/python /tmp/state-memory-verification/compare.py
```

All four MAT payloads matched exactly, including shape, dtype, and NaN positions
(524 array comparisons; detection date and output-directory provenance excluded).
All four decoded JPEG images matched pixel-for-pixel. Peak RSS fell from
3,541,724 KiB to 534,056 KiB, approximately 85% lower (3.38 to 0.51 GiB).
Elapsed times were 22.98 and 16.64 seconds; these single-run times have differing
filesystem cache conditions and are not a controlled speed benchmark.

## Regression verification

```sh
/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/preprocess/test_state_scoring_memory.py tests/preprocess/test_state_scoring_neurocode_compat.py tests/preprocess/test_io_source_selection.py tests/preprocess/test_config_index_base.py
/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/state_scoring.py tests/preprocess/test_state_scoring_memory.py
git diff --check
```

Result: 42 tests passed; compilation and diff checks passed. New tests cover
non-divisible sample counts, exact downsampled inputs, requested concurrency
1/2/128, candidate ordering and ties, nonfinite histogram inputs, prohibition of
the bulk candidate loader, and selected-output backing storage size.

Independent read-only review found no numerical-semantic regression. The review
identified that the new test was ignored by the repository's broad test ignore
rule; an explicit `.gitignore` exception now includes it in the deliverable.

The original 122-hour multi-day job has not been rerun. Removal of the all-channel
heap arrays and 128 simultaneous spectral tasks directly addresses its identified
memory mechanism, but no multi-day peak-memory guarantee is claimed. Per-channel
spectral calculations and EMG still scale with recording duration; mapped file
pages can also contribute to measured resident memory.
