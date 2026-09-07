# Probe Placement And GUI Layout

- Date: 2026-09-05
- Commit: uncommitted
- Plan: [2026-09-05 probe placement and GUI layout](../implementation_plan/2026-09-05-probe-placement-gui-layout.md)

## What changed

- Standardized `x offset` for every probe geometry as the horizontal
  bounding-box center of the complete probe assignment. Internal contact and
  shank spacing is unchanged.
- Fixed flex-G5 single-group placement so left- and right-half channel IDs use
  the same center convention. Separately assigned shanks at 0 and 600 um now
  have exactly 600 um center-to-center spacing.
- Made the probe geometry combo shrink responsively and reserved an editable
  minimum width for the XML-groups field, preventing long geometry names from
  squeezing that field away.
- Enabled long-row wrapping in left-pane forms to prevent labels from colliding
  with their fields at narrow widths.
- Increased the default window and central-pane allocation, reduced the
  settings pane allocation, increased the chanMap-to-log layout weight, and
  reduced the log minimum height from 160 to 80 px.
- Removed the visible right-side `Setting config` panel so the chanMap/behavior
  preview and log use the entire center-pane width. The generated settings
  summary is retained as internal diagnostic text rather than a widget.
- Updated geometry and GUI regression coverage for the new coordinate and
  layout semantics.

## Verification

- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_chanmap_geometry.py tests/preprocess/test_gui_config_model.py::test_gui_browse_local_session_resume_is_common_and_reconnects tests/preprocess/test_gui_config_model.py::test_load_xml_updates_probe_assignments_and_chanmap_preview`
  - Passed: 10 tests.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/pytest -q tests/preprocess/test_gui_config_model.py tests/preprocess/test_chanmap_geometry.py`
  - Passed: 54 tests.
- `/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile src/preprocess/io.py src/preprocess/gui/app.py tests/preprocess/test_chanmap_geometry.py tests/preprocess/test_gui_config_model.py`
  - Passed.
- `git diff --check`
  - Passed.

## Known limitations

- Existing chanMap files retain their old coordinates until regenerated.
- The placement center is the min/max horizontal bounding-box midpoint, not the
  arithmetic mean of contact positions; this avoids channel-density bias.
