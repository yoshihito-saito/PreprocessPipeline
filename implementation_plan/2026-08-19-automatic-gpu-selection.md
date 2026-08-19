# Automatic GPU Selection And Audit Log

## Goal

Before a Sorting Stage launches Kilosort, inspect every GPU visible to the
worker and run on the least-used GPU whose memory use and compute utilization
are both below 50 percent.
When no GPU qualifies, keep the Sorting worker waiting and poll again without
starting MATLAB/Kilosort.

## Current problem

Slurm tracks the two physical GPUs as one identical GRES type and can assign a
GPU that is already heavily used by a process outside Slurm. The execution
backend currently accepts that assignment without inspecting live GPU use and
does not retain an audit of who was using each GPU when Kilosort started.

## Changes

- Add an execution-layer NVIDIA GPU inventory helper based on `nvidia-smi`.
- Record device memory, utilization, compute-process PID, username, process
  name, and process GPU memory for every selection attempt.
- Select by GPU UUID, set `CUDA_VISIBLE_DEVICES` immediately before the sorter
  is imported/launched, and preserve the original Slurm/CUDA allocation fields
  in the audit log.
- Store the append-only audit at
  `stages/sorting/attempt-NNN/gpu-selection.jsonl` and print compact progress to
  the normal Stage stdout log.
- Treat memory use or compute utilization greater than or equal to 50 percent
  as busy and poll every 30 seconds until a candidate exists.
- After selection, keep a background inventory running every 10 minutes until
  the sorter exits. Record a final snapshot on normal or exceptional exit;
  telemetry failures are logged but never fail Kilosort.

## Verification

- Test least-used selection and UUID activation.
- Test waiting followed by selection without launching Kilosort early.
- Test per-process username/memory audit serialization and query failures.
- Test periodic and final monitoring records without coupling telemetry errors
  to the sorter result.
- Run focused execution tests, compilation, and final diff checks.

## Explicit limitation

This is a pragmatic selection policy, not an exclusivity guarantee. A process
outside Slurm can start after the final observation, and overriding the original
Slurm-visible device can differ from its GRES allocation on this host. The user
has explicitly accepted that external-process race for this workflow.
