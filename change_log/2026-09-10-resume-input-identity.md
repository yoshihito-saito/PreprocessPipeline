# Resume input identity and manual-session recovery

- Date: 2026-09-10; uncommitted.
- Plan: [resume input identity](../implementation_plan/2026-09-10-resume-input-identity.md).

## Changes
Configuration sidecars (.xml, .json, .oebin, .txt) now carry SHA-256 input identities. Matching recorded/live content permits timestamp-only drift; changed contents remain incompatible. Binary input checks remain unchanged. Legacy entries without hashes retain metadata checks.

With an authoritative multi-day XML, unused per-recording amplifier.xml entries no longer invalidate reuse. This follows the existing staging behavior and fixes the Day117 metadata drift blocking Day10–Day245 without changing old snapshots or output contracts manually.

GUI local-session recovery reads a manual Phy lease's previous Run, or falls back to the completed preprocess record when no previous Run exists. The lease remains intact. Invalid Run claims still fail validation.

## Verification
- `/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q tests/execution/test_input_identity.py tests/execution/test_session_resume.py tests/execution/test_worker.py tests/preprocess/test_gui_config_model.py`: 84 passed.
- Read-only call to `_input_snapshots_compatible` on the actual previous/current Day10–Day245 Run snapshots returned True.
- `load_local_session_resume` succeeded on actual Day7–Day231 and Day10–Day245 folders. The Day7 manual marker was confirmed unchanged.
- `git diff --check`: passed.
- Base `python -m pytest` could not run because base Python lacks pytest; verification used the existing phy2 environment.

## Limits
No GPU sorting or VNC GUI interaction was launched. Restart the GUI to load the updated code. Legacy metadata drift in inputs that are actually consumed still needs prior content evidence; this change does not blindly accept changed acquisition data.
