# Replace Bad Channels When Loading XML

- Date: 2026-09-05
- Commit: uncommitted
- Plan: [2026-09-05 replace bad channels when loading XML](../implementation_plan/2026-09-05-xml-bad-channel-replacement.md)

## What changed

- Changed explicit GUI `Load XML` handling so the bad-channel field is replaced
  by the selected XML's sorted, unique `skip="1"` channels instead of being
  unioned with stale values already present in the field.
- Added a GUI regression test that starts with stale bad channels and verifies
  that loading an XML leaves only the XML-skipped channel.

This prevents bad channels from a previously loaded configuration or chanMap
from carrying into a newly generated flex-G5 map.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py::test_load_xml_updates_probe_assignments_and_chanmap_preview`
  - Passed: 1 test.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py tests/preprocess/test_chanmap_geometry.py`
  - Passed: 54 tests.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/gui/app.py tests/preprocess/test_gui_config_model.py`
  - Passed.
- `git diff --check`
  - Passed.
- A broader run including `tests/preprocess/test_gui_preflight.py` produced 64
  passes and 2 failures in pre-existing `middle_finger` geometry preflight
  expectations. Those tests do not exercise the changed XML-load path.

## Known limitations

- Existing `chanMap.mat` files are not rewritten automatically. The user must
  load the intended XML and regenerate the chanMap to replace the persisted
  disconnected-channel list.
