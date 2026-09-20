# Session settings and safe sorter configuration editing

## Goal and scope

Reuse a session's GUI parameters automatically, save changes on Run, and edit a
session-local sorter configuration without modifying repository defaults or
immutable Run snapshots. Fix recovery of the observed XML/manual-exclusion
pollution without relaxing preprocess output compatibility.

## Steps

1. Add session configuration helpers to the existing GUI configuration model.
2. Integrate session-local sorter copies and settings persistence into Run
   creation, session browsing and Open config.
3. Recover legacy polluted manual exclusions only when a failed contract check,
   the unchanged XML exclusions and prior contract-matching settings prove the
   specific GUI mutation. Preserve other parameter changes.
4. Inspect the scoped diff, run one relevant existing test, and document usage
   and remaining limitations.

## Decisions

- Editable files live in `<session>/config/`; each Run still has immutable
  snapshots. Reuse existing local sorter files without overwriting edits.
- Save GUI parameters on successful Run creation, not on each widget change.
- Genuine preprocess changes retain the existing overwrite/compatibility rules.
- Existing dirty GPU selection and sorter defaults are outside this task.
- No jobs will be submitted or existing analysis snapshots changed.

## Completion and checks

- Implemented session JSON persistence in Run creation, automatic restoration
  during session browsing, and editable local sorter copies for Open config.
- Legacy recovery is attempted only before a session JSON exists, so later
  explicitly saved settings are not rewritten by the migration. Actual XML
  content hashes and prior preprocess fingerprints must agree.
- Preserved the earlier GUI fix separating XML exclusions from manual entries.
- Updated the existing controller snapshot test locally to check persisted GUI
  settings, local edits, default isolation and immutable snapshots. This existing
  test file is ignored by the repository; its update remains local.
- Ran one existing named test:
  `/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest tests/execution/test_controller.py::test_create_run_snapshots_analysis_and_sorter_config -q -p no:cacheprovider --basetemp=/tmp/pp-session-config-20260920-62ad`
  Result: 1 passed in 0.31s. Temporary test files removed.
- Reviewed the scoped diff. After the check, restricted legacy migration to
  sessions without saved JSON; reviewed that branch without another test run.
- No Slurm job, live GUI round trip, or real-session migration was executed.
  Restart the GUI and browse the session to apply the new recovery path.
- No edits to shared sorter defaults or existing Run/output records in this task.
