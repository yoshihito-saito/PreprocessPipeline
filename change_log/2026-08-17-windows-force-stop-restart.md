# Windows Force Stop And Restart

## Date And Commit

- Date: 2026-08-17
- Commit: uncommitted
- Plan: [2026-08-17 Windows Force stop and restart](../implementation_plan/2026-08-17-windows-force-stop-restart.md)

## What Changed

- Windows Local jobs now persist the process creation FILETIME with the PID.
- A detached cancellation controller validates that identity before invoking `taskkill /T /F` and waits for confirmed exit.
- Windows process queries distinguish definitely absent, present, and unverifiable states. Access or query failures remain active and cannot falsely confirm cancellation or release the session claim.
- A new Local worker is terminated and submission fails if a reusable Windows identity cannot be captured.
- Run all, Preprocess only, and Postprocess only remain disabled while persistent work or cancellation is active, then re-enable after a conclusive terminal state.
- Browse local session to resume now explicitly discovers and loads a `chanMap.mat` from the selected local output even when its path was not saved in the recovered settings.
- Added regression coverage for Windows identity matching, PID reuse/query failure, failed identity capture, persistent GUI button state, and local chanMap restore.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_move_outputs.py tests/execution/test_local_backend.py tests/preprocess/test_gui_config_model.py tests/execution/test_controller.py tests/execution/test_session_resume.py`
  - Result: 93 passed.
- `python -m py_compile src/execution/backends/local.py src/preprocess/gui/app.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Known Limitations And Next Steps

- Existing Windows Run records created before this change do not contain a process creation token. A still-running legacy worker cannot be safely killed by a detached controller without independently verifying it; stop that worker manually once, then reconcile or request Force stop again so the Run can reach a confirmed terminal state.
- The Windows process API path is unit-tested through mocks on Linux but has not been exercised on a physical Windows host in this workspace.
- The broader `tests/execution tests/preprocess` run had 347 passes and 19 pre-existing failures unrelated to these changes.
