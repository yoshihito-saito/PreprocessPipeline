# Move Missing Session XML

## Date And Commit

- Date: 2026-08-17
- Commit: uncommitted
- Plan: [2026-08-17 move missing session XML](../implementation_plan/2026-08-17-move-missing-session-xml.md)

## What Changed

- The final output move now transfers `${basename}.xml` through the existing staged and verified move when the destination lacks that file.
- An existing destination session XML is always retained; the local duplicate is skipped even when overwrite is enabled.
- The staged XML is published with a same-filesystem atomic no-clobber hard link, so an XML created concurrently is also preserved. Filesystems without hard-link support use an exclusive-create, flushed copy that still cannot replace an existing XML.
- Non-canonical XML and RHD input metadata retain their previous stay-local behavior.
- The GUI move confirmation inventory now uses the same conditional XML selection rule.
- Added durable regression tests for both cleanup modes, missing and existing destination XML, concurrent destination creation, hard-link fallback, and rollback after publication.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_move_outputs.py tests/execution/test_local_backend.py tests/preprocess/test_gui_config_model.py tests/execution/test_controller.py tests/execution/test_session_resume.py`
  - Result: 93 passed.
- `python -m py_compile src/execution/backends/local.py src/preprocess/gui/app.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Known Limitations And Next Steps

- Only the canonical `${basename}.xml` is transferred. Other XML files remain local because they may be unrelated input metadata.
- The broader `tests/execution tests/preprocess` run had 347 passes and 19 pre-existing failures in artifact-removal, old middle-finger preflight, pipeline integration, and Kilosort4 tests; none exercised the files or behavior changed here.
