# Legacy session recovery in the GUI

## Goal and scope
Allow Browse local session to resume to restore legacy preprocessing settings
without manufacturing persistent Run metadata or weakening output reuse checks.
Preserve the existing uncommitted sorter-config edits.

## Implementation steps
1. Add recovery from legacy parameter/manifest JSON after the existing persistent
   recovery paths, preserving recorded scientific settings and session identity.
2. Restore available session-local GUI settings and multi-day selections; explain
   legacy recovery and missing metadata in the GUI.
3. Document supported metadata and the distinction between loading settings and
   resuming validated outputs.
4. Inspect the scoped diff and perform at most one lightweight changed-file check.
   Do not add tests or run pipelines or broad validation.

## Completion
- Added legacy recovery after persistent marker/completed-record recovery. Invalid
  persistent metadata still fails without falling through to legacy settings.
- Supported metadata: session-local GUI JSON, or the legacy parameter/manifest
  pair. Restore recorded GUI preprocessing fields (including TTL aliases), worker
  counts, local XML/chanMap, and available multi-day selections. Require matching
  basenames and report missing scientific settings instead of guessing them.
- Legacy loading disables overwrite and creates no Run/contract metadata. GUI
  messages distinguish legacy settings from completed Runs and identify missing
  postprocessing history. Existing adoption/compatibility checks are unchanged.
- Preserved the pre-existing sorter-config changes. Reviewed the scoped diff.
- Check run: `python3 -m py_compile` on the two changed GUI Python files, passed;
  temporary bytecode cache was outside the repository and removed afterward.
- No tests added or run; no GUI/pipeline/data execution. The reported session
  directory is inaccessible here, so real-session recovery remains unverified.
  MAT-only records, incomplete settings, renamed legacy session directories, and
  incomplete/unrepresentable legacy outputs are not automatically migrated.
