# Staged Metadata Same-File Copy Guard

## Date And State

- Date: 2026-07-31
- Git commit: uncommitted
- Plan: [2026-07-31 staged metadata same-file copy guard](../implementation_plan/2026-07-31-xml-samefile-guard.md)

## What Changed

- Added a shared same-file guard in `src/preprocess/io.py` so metadata ensure
  helpers return the staged target without calling `copy2` when the source and
  destination resolve to the same path.
- Applied the guard to both `ensure_xml` and `ensure_rhd`.
- Added focused tests for explicit XML, basepath XML, and discovered RHD inputs
  that are already located at the target path.

## Why

The `multiday_Day7_to_Day194` retry workflow had metadata files already staged
under `sorting_temp/`. The pipeline attempted to copy the XML and then the RHD
file onto themselves and raised `shutil.SameFileError` before preprocessing
could start.

## Verification

The default `pytest` command in this shell resolved to Python 2.7 and failed
during collection before running tests.

Successful verification:

```bash
/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_io_source_selection.py
```

Result:

```text
21 passed in 0.95s
```

## Known Limitations And Next Steps

- This change only addresses same-file metadata staging. It does not change GUI
  config persistence, Kilosort memory usage, or GPU selection.
- Runtime preprocessing should still be launched from the intended conda
  environment, preferably with the full `phy2` Python path.
