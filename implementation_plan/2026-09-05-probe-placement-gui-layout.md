# Probe Placement And GUI Layout

## Goal and motivation

Make probe assignments editable in the narrow settings pane, give `x offset`
one consistent geometric meaning across probe types, and allocate more of the
main window to the chanMap preview.

## Current problem

- The geometry combo box uses the width of the longest probe type, which can
  squeeze the XML-groups editor until it is impractical to type into.
- `x offset` is currently added to layout-specific coordinates. Depending on
  the geometry and selected XML group, it can refer to a left contact, a shank
  center, or an assignment origin. Separately positioned shanks therefore do
  not reliably have the requested center-to-center distance.
- The settings pane and log consume too much of the default window, while long
  form labels can collide with fields at narrow widths.
- The right-side `Setting config` panel duplicates settings already available
  in the left controls and reduces the chanMap/behavior preview width.

## Affected files

- `src/preprocess/io.py`: center every generated probe assignment on its
  requested `x_offset`, preserving internal shank/contact spacing.
- `src/preprocess/gui/app.py`: make the probe row responsive and editable,
  wrap narrow form rows, reduce the default left/log allocations, and enlarge
  the central preview allocation.
- `tests/preprocess/test_chanmap_geometry.py`: verify centered offsets for
  generic and flex-G5 layouts.
- `tests/preprocess/test_gui_config_model.py`: verify group-field editability
  and the revised layout constraints.

## API and parameter semantics

For every probe type, `x_offset` means the horizontal center coordinate of the
complete probe assignment. With one XML group per assignment, it is the shank
center. With multiple groups, their relative geometry remains unchanged and
the bounding-box center of the complete assignment is placed at `x_offset`.

## Expected behavior

- XML group text can be entered even when a long geometry name is selected.
- Assignments centered at 0, 600, 1200, and 1800 um have matching 600 um
  center-to-center spacing regardless of channel IDs or probe type.
- The chanMap preview receives more default horizontal and vertical space.
- The right-side `Setting config` panel is removed so the preview uses the
  complete center-pane width; its generated diagnostic text remains available
  internally for error handling and debugging.
- The log remains usable but no longer enforces a 160 px minimum height.
- Narrow left-pane form rows wrap instead of allowing labels and fields to
  overlap.

## Verification strategy

- Run focused geometry and GUI tests.
- Run Python compilation and `git diff --check`.
- Inspect the final diff and preserve the prior XML bad-channel fix.

## Non-goals

- Do not change physical contact spacing or channel wiring templates.
- Do not change XML group membership or bad-channel semantics.
- Do not regenerate user data automatically.
