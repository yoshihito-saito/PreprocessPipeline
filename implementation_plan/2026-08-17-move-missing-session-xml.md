# Move Missing Session XML To Storage

## Goal

Preserve an existing destination session XML, but move the local session XML when the destination does not already contain it.

## Current Problem

The final output move unconditionally excludes the canonical session XML and all other XML files on the assumption that input metadata already exists in the basepath. The fallback metadata copy runs only for an explicit custom destination with local cleanup, so a normal destination without XML never receives it.

## Affected Files

- `src/preprocess/gui/app.py`
- `tests/preprocess/test_gui_move_outputs.py`

## Expected Behavior

- If `${basename}.xml` exists at the destination, retain it unchanged and skip the local duplicate.
- Publish the staged XML with an atomic no-clobber operation so an XML created concurrently is also retained; fall back to exclusive-create copying on filesystems without hard-link support.
- If it does not exist and the local output contains it, include it in the existing staged, verified, transactional move.
- Keep non-canonical XML and RHD exclusion behavior unchanged.
- Make the confirmation preview follow the same XML selection rule.

## Verification

Cover both missing-destination and existing-destination XML cases, local cleanup modes, concurrent destination creation, hard-link fallback, and rollback after publication.
