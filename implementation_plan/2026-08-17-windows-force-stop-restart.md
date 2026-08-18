# Windows Force Stop And Restart

## Goal

Allow a cancelled Windows Local persistent Run to reach a confirmed terminal state so a subsequent Run all can safely acquire the same session output.

## Current Problem

Local job identity is persisted with Linux `/proc` start ticks. On Windows that value is unavailable, and a detached cancellation controller does not own the original `Popen` object. It therefore refuses to signal the worker because it cannot distinguish the recorded PID from a reused PID. Cancellation remains unconfirmed, and `.pipeline-active-run.json` correctly blocks creation of a new Run.

The local-session resume path may retain a saved `chanmap_path`, but does not explicitly discover a `chanMap.mat` present only in the selected output folder.

## Affected Files

- `src/execution/backends/local.py`
- `src/preprocess/gui/app.py`
- `tests/execution/test_local_backend.py`
- `tests/preprocess/test_gui_config_model.py`

## Implementation

- Persist the Windows process creation timestamp alongside the PID when a Local job is submitted.
- In detached controllers, verify PID plus creation timestamp through the Windows process API before calling `taskkill`.
- Preserve an explicit unknown state for access/query failures and never treat it as process exit.
- Terminate a newly launched worker and fail submission if its reusable identity cannot be captured.
- Wait briefly for `taskkill` completion and fail closed if the verified process remains alive or becomes unverifiable.
- Keep Linux process-group identity and cancellation behavior unchanged.
- Disable Run all, Preprocess only, and Postprocess only while persistent work or cancellation is active.
- Explicitly discover a local `chanMap.mat` after local-session settings are restored, then apply it to the path field, controls, and preview.

## Verification

- Unit-test persisted Windows identity matching and PID-reuse rejection without requiring a Windows host.
- Run controller/session claim tests and the GUI resume tests.
- Run the existing move-output tests to retain the in-progress XML transfer behavior.
