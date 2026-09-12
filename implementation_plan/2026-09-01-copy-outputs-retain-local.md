# Copy outputs while retaining local data

## Goal

Make the final storage action a copy by default so an unchecked cleanup option never deletes local outputs, and make the resulting storage tree readable and writable by all users.

## Current problem

The staged transfer originally deleted selected source items even when `clean_after_move=False`. After correcting that behavior, copied items received collaborative permissions but the destination root itself could retain restrictive permissions such as `2750`, preventing other users from reaching otherwise writable content.

## Affected files

- `src/preprocess/gui/app.py`
- `tests/preprocess/test_gui_move_outputs.py`
- `README.md`

## Intended behavior

- The storage action remains a staged, content-verified publication.
- With local deletion disabled, scientific outputs and the local preprocess contract remain unchanged after a successful copy.
- Extension (2026-09-10): exclude `.preprocess-output-backups` and `.preprocess-output-contract.json` from transfer. After successful publication, content verification, and permission updates, remove local preprocess backups and both items at the destination (including previously copied instances). Keep the local contract for resume checks. Never follow cleanup symlinks. Report cleanup failure separately from transfer failure, without rolling back verified outputs.
- The GUI preview excludes these items from copy size/count and explains automatic cleanup. Regression coverage includes successful cleanup, transfer/verification failure preservation, cleanup failure, and symlink safety.
- With local deletion enabled, the source session directory is removed only after destination verification succeeds.
- GUI wording describes a copy, and local deletion is disabled by default.
- After destination verification, the destination root and its complete tree are made world-readable and world-writable; directories also receive traversal permission.
- Existing internal helper names and arguments remain compatible.

## Verification

- Run the focused GUI storage-transfer tests in the `phy2` environment.
- Compile the modified Python source and inspect the final diff.

## Non-goals

- Changing destination overwrite or transactional publication behavior.
- Adding background progress or cancellation support.
