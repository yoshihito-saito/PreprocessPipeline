# Default expandable PyTorch CUDA segments

- Date: 2026-09-19
- Status: uncommitted
- Plan: [allocator defaults](../implementation_plan/2026-09-19-torch-allocator-default.md)

## Changes

Fresh pipeline processes now set `PYTORCH_ALLOC_CONF=expandable_segments:True`
at `src` package startup before dependencies import Torch. Explicit modern or
legacy allocator variables, including empty values, are left untouched.
This covers GUI and CLI entrypoints and both local and Slurm workers.
No Kilosort algorithm, threshold, recording length or numerical parameter changed.

Workers capture both allocator variable values in `started.json`; sorting
prints them to stdout, and GPU monitoring includes them in audit events.
The README describes activation, opt-out and limits of these observations.
A focused new test file is allowlisted in `.gitignore` so the regression tests
are included in version control without adding unrelated ignored tests.

## Verification

```text
/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest tests/execution/test_torch_allocator_default.py tests/execution/test_gpu_selection.py tests/execution/test_worker.py -q
git diff --check
```

Result: 32 tests passed in 1.29s; diff check passed. Fresh subprocesses verify
worker import sets the default without importing Torch, and preserve explicit
policies under either environment variable spelling. Worker failure coverage
checks that allocator settings are recorded before sorting starts; GPU audit
coverage checks the new environment fields. Final diff reviewed.

## Limits

Existing processes are unaffected. Restart the GUI to load the new default;
notebooks must configure the environment before Torch import. Logs capture
environment values, not proof that the CUDA driver supports expandable segments.
No GPU allocation stress test or full sorting rerun was performed; this change
does not establish that probe3 can finish or that every OOM is fragmentation.
