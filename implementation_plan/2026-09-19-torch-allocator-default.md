# Default expandable PyTorch CUDA segments

## Goal and observed problem

The September 18 sorting run failed on probe3 in `assign_iclust`, requesting
22.76 GiB with 22.55 GiB reserved but unallocated. Worker logs did not capture
allocator environment settings, so their propagation could not be verified.
Enable expandable segments by default without changing sorting parameters.

## Design

- Set `PYTORCH_ALLOC_CONF=expandable_segments:True` at `src` package startup,
  before pipeline dependencies can import Torch. This covers fresh GUI, CLI,
  local worker and Slurm worker processes and subsequent child processes.
- If either `PYTORCH_ALLOC_CONF` or legacy `PYTORCH_CUDA_ALLOC_CONF` exists,
  preserve it exactly, including an empty value. Do not merge, replace or
  reinterpret explicit allocator policies such as `backend:cudaMallocAsync`.
- Record both environment variables in worker `started.json`, print them for
  sorting attempts, and include them in periodic GPU audit events. These are
  environment observations, not proof of CUDA allocator activation/support.
- Document restart requirements, explicit opt-out, notebook import order,
  and the absence of a guarantee that sorting fits GPU memory.

## Files and verification

Update `src/__init__.py`, worker and GPU audit code, README, and focused tests.
Use fresh-process tests to verify defaults and precedence before Torch import;
test worker audit output even on failure and GPU audit propagation. Run the
existing worker/GPU monitor suites and inspect the final diff.

No Kilosort edits, threshold changes, GPU allocations, job submissions, or
changes to existing run records are part of this work. Real GPU completion
cannot be established by these configuration tests.
