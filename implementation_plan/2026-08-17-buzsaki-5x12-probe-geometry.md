# Buzsaki 5x12 Probe Geometry

## Goal And Motivation

Add a built-in chanMap geometry for the 64-channel Buzsaki 5x12 probe so it can be selected and previewed in the preprocessing GUI.

Despite the `5x12` name, the design has four additional center-shank sites, so its channel count is 64 rather than 60.

## Current Problem

The existing `middle_finger` geometry also has a specialized center shank, but its center consists of 16 linear sites with different spacing and depth. It does not represent the Buzsaki 5x12 design shown in the supplied probe drawing.

## Affected Files

- `src/preprocess/io.py`
- `src/preprocess/gui/app.py`
- `tests/test_buzsaki_5x12_geometry.py`
- `tests/preprocess/test_gui_config_model.py`

## Public Parameters Or API Changes

Add `Buzsaki 5x12` as an accepted probe assignment `type` and as a GUI geometry choice. Recognize common spaced, hyphenated, and underscored forms in XML descriptions and saved configuration.

## Geometry Semantics

- Five shanks are spaced 200 um apart.
- Each side shank uses the existing 12-channel CellExplorer `poly2` coordinates.
- The center shank has a 12-site `poly2` cluster at its tip and four single-column sites above it, for 16 center-shank channels in total.
- The four extra center sites and the top of the tip cluster are separated by 200 um center-to-center, placing the extra sites at y = 800, 600, 400, and 200 um relative to the top of the tip cluster.
- Center-shank channel order runs from the four upper linear sites to the 12-site tip cluster, consistent with the existing top-to-bottom center-shank convention.

## Expected Behavior

A standard group layout of `[12, 12, 16, 12, 12]` produces 64 coordinates. The outer four groups remain ordinary 12-site `poly2`; the center group consists of the four linear extension sites followed by a 12-site `poly2` tip cluster.

Because this is a fixed hardware design, other group counts or group sizes are rejected instead of silently producing a different number of linear sites.

## Verification

- Add focused coordinate tests for the side and center shanks.
- Add an integration test for `build_channel_map_data` and XML-description detection.
- Run the relevant chanMap geometry tests and Python syntax checks.

## Non-Goals

- No change to the existing `middle_finger` geometry.
- No channel wiring remap beyond the XML group/channel order.
- No change to the default probe geometry.
