# Lazy startup imports and XML GUI synchronization

- Date: 2026-08-13
- Commit: uncommitted
- Plan: [2026-08-13 lazy startup imports and XML GUI synchronization](../implementation_plan/2026-08-13-lazy-import-and-xml-gui-sync.md)

## What changed

- Made the top-level `src`, preprocess, and postprocess exports lazy so GUI and
  controller startup do not import the complete SpikeInterface sorter registry.
- Deferred the sorter registry import in `sorter_runner` until sorter classes
  are actually accessed.
- Made `run_pipeline` import preprocessing/postprocessing implementations only
  inside the worker execution path.
- Added XML-derived probe assignment/geometry selection and made `Load XML`
  synchronously invalidate stale chanMap preview state and render the current
  XML-driven settings.
- Added regression coverage for GUI import laziness and XML-driven preview and
  assignment updates.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_chanmap_geometry.py tests/preprocess/test_gui_config_model.py tests/preprocess/test_sorter_partitions.py tests/preprocess/test_gui_preflight.py`
  - Passed: 71 tests, including the GUI import-laziness and XML preview
    regression tests.
- `python3 -m py_compile` on all changed Python modules and tests
  - Passed.
- `git diff --check`
  - Passed.
- The broader focused command including
  `test_sorter_runner_matlab_path.py` had 62 passes and 3 pre-existing failures
  unrelated to this change: Kilosort4 parameter coercion, missing
  `_resolve_kilosort4_import_root`, and the CLI default Kilosort4 path.

## Known limitations

- XML can identify anatomical groups and a geometry hint, but cannot infer a
  user-specific multi-probe partition when that information is not encoded in
  the XML; the GUI therefore creates one assignment spanning all XML groups.
- The change is uncommitted and could not be tested on the user's Windows
  environment directly.
