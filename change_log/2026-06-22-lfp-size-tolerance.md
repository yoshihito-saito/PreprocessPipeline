# LFP Existing File Size Tolerance

## Date and git state

- Date: 2026-06-22
- Git base: `a4cb73e`
- Status: uncommitted

## What changed

- Added an optional `frame_tolerance` argument to the internal binary-size
  validation helper in `src/preprocess/recording.py`.
- Applied `frame_tolerance=1` only to existing `.lfp` reuse in `write_lfp()`.
- Kept byte-size divisibility checks strict and kept non-LFP binary reuse checks
  exact.
- Added tests for accepting a one-frame LFP mismatch and rejecting a two-frame
  mismatch.
- Added implementation documentation:
  `implementation_plan/2026-06-22-lfp-size-tolerance.md`.

## Why

The pipeline failed when reusing an existing LFP file generated from a recording
whose raw frame count is not exactly divisible by the 20,000 Hz to 1,250 Hz
downsampling factor. The existing file had one fewer frame than the validation's
rounded expected count, which is consistent with resampling endpoint rounding
rather than a meaningful stale-output condition.

## Verification

Command run:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_recording_selected_transform.py
```

Result:

```text
10 passed in 1.30s
```

An initial `pytest -q ...` command used the system Python 2.7 pytest and failed
during collection with a syntax error on modern type annotations; verification
was rerun in the repository's `phy2` environment.

## Observed behavior

`write_lfp()` now reuses an existing LFP file when the sample-count mismatch is
within one frame. A mismatch greater than one frame still raises the stale-file
`ValueError`.

## Known limitations and next steps

- This change does not alter the LFP generation algorithm.
- If future SpikeInterface versions expose a reliable output frame count before
  writing, the validation can be updated to compare against that value directly.
