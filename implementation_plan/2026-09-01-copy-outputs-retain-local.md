# Copy outputs while retaining local data

## Goal

Make the final storage action a copy by default so an unchecked cleanup option never deletes local outputs.

## Current problem

The staged transfer copies and verifies outputs at the destination, but when `clean_after_move=False` it still deletes every selected source item. The GUI label implies that disabling local cleanup preserves the local session, which is not true.

## Affected files

- `src/preprocess/gui/app.py`
- `tests/preprocess/test_gui_move_outputs.py`
- `README.md`

## Intended behavior

- The storage action remains a staged, content-verified publication.
- With local deletion disabled, every source item remains unchanged after a successful copy.
- With local deletion enabled, the source session directory is removed only after destination verification succeeds.
- GUI wording describes a copy, and local deletion is disabled by default.
- Existing internal helper names and arguments remain compatible.

## Verification

- Run the focused GUI storage-transfer tests in the `phy2` environment.
- Compile the modified Python source and inspect the final diff.

## Non-goals

- Changing destination overwrite or transactional publication behavior.
- Adding background progress or cancellation support.
