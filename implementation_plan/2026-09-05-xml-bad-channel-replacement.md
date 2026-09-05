# Replace Bad Channels When Loading XML

## Goal and motivation

Make an explicit GUI `Load XML` action treat the newly selected XML as the
source of truth for bad channels, so stale channels from a previously loaded
configuration or chanMap do not carry into a newly generated map.

## Current problem

The GUI currently unions XML channels marked `skip="1"` with the existing bad
channel field. Loading a new XML after another session therefore retains stale
bad channels. In `multiday_Day7_to_Day231`, 13 stale channels were combined
with XML channels 16 through 47, producing 45 displayed bad channels instead
of 32.

## Affected files

- `src/preprocess/gui/app.py`: replace the bad-channel field with the selected
  XML's skipped channels during explicit XML loading.
- `tests/preprocess/test_gui_config_model.py`: cover replacement of stale GUI
  bad channels.

## Expected behavior

- Explicitly loading an XML replaces the GUI bad-channel field with the sorted,
  unique channels marked `skip="1"` in that XML.
- An XML with no skipped channels clears stale GUI bad channels.
- Editing the bad-channel field after XML loading remains supported.
- Loading an existing chanMap continues to import that chanMap's disconnected
  channels.

## Verification strategy

- Run the focused XML-load GUI regression test.
- Run the relevant GUI/config and chanMap tests.
- Inspect the final diff and run `git diff --check`.

## Non-goals

- Do not change XML parsing, flex-G5 geometry, channel indexing, or chanMap
  `connected` semantics.
- Do not modify existing data files or regenerate the user's chanMap
  automatically.
