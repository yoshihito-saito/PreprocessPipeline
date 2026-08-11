# Staged Metadata Same-File Copy Guard

## Goal And Motivation

Prevent preprocessing runs from failing when selected or discovered metadata
files are already located at the pipeline's local output path.

## Current Problem

A preprocessing run for `multiday_Day7_to_Day194` selected:

```text
/local/workdir/ys2375/PreprocessPipeline/sorting_temp/multiday_Day7_to_Day194/multiday_Day7_to_Day194.xml
```

as the XML source. The resolved local output target was the same path, so
`shutil.copy2` raised `SameFileError` before preprocessing could continue.

After the XML guard was added, the same staging pattern failed for:

```text
/local/workdir/ys2375/PreprocessPipeline/sorting_temp/multiday_Day7_to_Day194/multiday_Day7_to_Day194.rhd
```

because the discovered RHD source and local output target were also the same
file.

## Why This Is Needed Now

The current retry workflow stages data under `sorting_temp/`, and the XML can
already be present in the local output directory after prior GUI or CLI runs.
The pipeline should treat that as a valid staged XML instead of failing during
setup.

## Affected Modules And Files

- `src/preprocess/io.py`
- `tests/preprocess/test_io_source_selection.py`

## Public Parameters Or API Changes

None. Existing `xml_path`, `basepath`, and `localpath` behavior remains
unchanged.

## Expected Behavior

- Explicit XML sources are still copied to the local output XML target when
  the paths differ.
- Basepath XML files are still copied to the local output XML target when the
  paths differ.
- Discovered RHD sources are still copied to the local output RHD target when
  the paths differ.
- If the resolved metadata source and target are the same file, the ensure
  helper returns the target without copying.
- Missing XML inputs still raise `FileNotFoundError`.

## Verification

Run the focused preprocessing I/O tests:

```bash
pytest -q tests/preprocess/test_io_source_selection.py
```

## Non-Goals

- Do not change XML discovery precedence.
- Do not change GUI config saving behavior.
- Do not change Kilosort or GPU scheduling behavior.
