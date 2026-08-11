# flex-G5 Probe Geometry

## Goal And Motivation

Add a built-in chanMap geometry for the high-density flex-G5 probe currently used for multi-day recordings.

## Current Problem

The GUI can generate chanMap files for existing generic layouts, but it does not expose the flexible probe layout discussed for motion-correction experiments. Users must approximate the geometry manually or provide an external chanMap.

## Why This Is Needed Now

The next validation step is to generate a correct flex-G5 chanMap from the preprocessing GUI and apply motion correction to one day of data.

## Affected Files

- `src/preprocess/io.py`
- `src/preprocess/gui/app.py`
- `tests/preprocess/test_chanmap_geometry.py`

## Public Parameters Or API Changes

Add `flex-G5` as an accepted probe assignment `type` in the GUI and chanMap builder.

## Algorithm Details

The flex-G5 geometry is modeled as a 64-channel fixed wiring template:

- lateral center-to-center spacing: 21.5 um;
- row spacing: 15 um;
- channel 15 and channel 16 are the top row contacts;
- channel 15 and channel 16 are separated by 800 um in x;
- each 32-channel half has left, center, and right columns;
- additional 64-channel blocks are tiled with a 1000 um x offset, so 128 channels become two 64-channel flex-G5 blocks.

The channel-to-coordinate mapping follows the provided probe wiring diagram instead of assuming monotonic channel order.

## Expected Behavior

Selecting `flex-G5` in a probe assignment generates `xcoords`, `ycoords`, `kcoords`, and `probe_ids` in the existing chanMap format.

## Verification

Add focused unit coverage for the standalone geometry helper and for `build_channel_map_data`.

## Non-Goals

- No motion correction implementation.
- No automatic channel-order detection from hardware metadata.
- No change to the default GUI config.
