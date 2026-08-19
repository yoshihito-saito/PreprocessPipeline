# Automatic GPU Selection And Audit Log

Date: 2026-08-19  
Commit: uncommitted

## What changed

- Added a Sorting-worker GPU gate that inventories all GPUs visible through
  `nvidia-smi` before importing or launching the sorter.
- A GPU is eligible only when both device-memory use and compute utilization
  are below 50 percent. The worker selects the eligible device with the lowest
  combined load and exposes it to Kilosort by stable GPU UUID.
- When no GPU is eligible, the worker does not start MATLAB/Kilosort and polls
  again after 30 seconds. Normal Force stop/Slurm cancellation still terminates
  the waiting worker.
- Added an append-only `gpu-selection.jsonl` to each Sorting Attempt. Every
  observation records total device use, utilization, compute-process PID,
  username, process name and process memory, plus the original Slurm/CUDA
  allocation environment and the final selected GPU.
- Added compact selection/waiting messages to the existing Sorting stdout log
  so the GUI's live Run log explains why Kilosort has or has not started.
- Extended the same audit after selection: a background monitor records all
  visible GPU/process ownership every 10 minutes while Kilosort runs and writes
  a final `sorter_finished` or `sorter_failed` snapshot when the sorter exits.
  A telemetry-query failure is recorded as `monitor_error` and does not alter
  the sorter outcome.

## Verification

- `/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution/test_gpu_selection.py tests/execution/test_local_backend.py tests/execution/test_local_backend_windows.py tests/execution/test_worker.py tests/execution/test_slurm_backend.py tests/execution/test_controller.py tests/execution/test_session_resume.py`
  - Result after adding continuous monitoring: 64 passed, 3 skipped.
- `/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution/test_gpu_selection.py tests/execution/test_local_backend.py tests/execution/test_local_backend_windows.py`
  - Result: 13 passed, 3 skipped.
- `/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution/test_worker.py tests/execution/test_slurm_backend.py tests/execution/test_controller.py tests/execution/test_session_resume.py`
  - Result: 49 passed.
- `/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/execution/gpu_selection.py src/execution/worker.py`
  - Result: passed.
- Queried the live host through the new inventory helper. It correctly
  attributed GPU 0's 43 processes and GPU 1's 5 processes to their current
  user and classified both devices busy because GPU utilization exceeded the
  threshold.
- `git diff --check`
  - Result: passed.

## Known limitation

The live check intentionally does not guarantee exclusivity against processes
started outside Slurm. A new process can begin after selection, and on this
host the chosen physical UUID can differ from the single anonymous GPU GRES
that Slurm originally assigned. This limitation was explicitly accepted for
the requested workflow.

Implementation plan: [2026-08-19 automatic GPU selection](../implementation_plan/2026-08-19-automatic-gpu-selection.md)
