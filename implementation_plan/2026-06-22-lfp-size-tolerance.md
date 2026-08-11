# LFP Existing File Size Tolerance

## Goal and motivation

Allow preprocessing to reuse an existing LFP binary when its frame count differs
from the analytically expected resampled length by only one frame. This avoids
blocking reruns when integer-ratio downsampling produces a floor-sized output
while the current validation computes a rounded expected length.

## Current problem

`write_lfp()` validates an existing `.lfp` file before reusing it when
`overwrite=False`. The validation currently expects
`round(raw_frames * lfp_fs / input_fs)` frames. For
`RM018_day39_260620`, the raw frame count is not divisible by the 20,000 Hz to
1,250 Hz downsampling factor:

```text
412,930,907 / 16 = 25,808,181.6875
```

The existing `.lfp` contains 25,808,181 frames, while the validation expects
25,808,182 frames, so preprocessing stops with a stale-file error even though
the difference is consistent with resampling endpoint rounding.

## Why this is needed now

After MATLAB service recovery, rerunning the full pipeline reaches LFP reuse and
fails before downstream Kilosort/postprocess can continue. The user is running
with overwrite disabled and wants this one-frame resampling discrepancy to be
accepted.

## Relevant state

- Worktree checked with `git status --short` before editing; no local changes
  were reported.
- Relevant module: `src/preprocess/recording.py`.
- Relevant tests: `tests/preprocess/test_recording_selected_transform.py`.

## Affected modules/files

- `src/preprocess/recording.py`
- `tests/preprocess/test_recording_selected_transform.py`
- `implementation_plan/README.md`
- `change_log/README.md`
- `change_log/2026-06-22-lfp-size-tolerance.md`

## Public parameters or API changes

No user-facing GUI, notebook, or CLI parameter changes are planned. The internal
binary-size validation helper will gain an optional frame tolerance argument
with a default of zero, preserving exact validation for existing callers.

## Algorithm details

Keep byte-level compatibility checks strict:

```text
file_size % (dtype_size * num_channels) == 0
```

When an expected frame count is provided, compute:

```text
abs(actual_frames - expected_frames) <= frame_tolerance
```

Use `frame_tolerance=1` only for existing LFP reuse in `write_lfp()`. Keep all
other binary reuse checks at `frame_tolerance=0`.

## Expected behavior

- Existing `.lfp` files that are exactly one frame shorter or longer than the
  current expected LFP length are reused when `overwrite=False`.
- Larger LFP mismatches still raise `ValueError`.
- Incompatible byte sizes still raise `ValueError`.
- Existing `.dat` reuse remains exact.

## Tests and checks

- Add a focused unit test that writes an existing LFP-sized file with one fewer
  frame than the rounded expectation and verifies that `write_lfp()` reuses it
  without invoking filtering, resampling, or binary writing.
- Add a focused unit test that verifies a two-frame LFP mismatch still raises.
- Run the relevant preprocess test module with `pytest`.

## Non-goals

- Do not change the actual LFP generation or resampling algorithm.
- Do not change overwrite semantics.
- Do not relax validation for AP `.dat`, digital input, or other binary files.
