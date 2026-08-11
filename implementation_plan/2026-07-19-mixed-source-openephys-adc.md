# Mixed Intan/Open Ephys Sessions and Open Ephys ADC Separation

## Goal and Motivation

Support real sessions where selected subepochs are a mixture of Intan and Open
Ephys recordings, and make Open Ephys `acquisition_board` streams safe when ADC
channels are embedded in the same `continuous.dat` as ephys channels.

The pipeline should sort and preprocess only ephys/probe channels, while still
preserving analog ADC channels for event export when they are present.

## Current Problem

Inspection of real Open Ephys recordings showed two cases:

- `ayadataB4/Fabrication/RSC-PFC1/Day22/.../D22_postsleep_260122_000001(ephys)/2026-01-22_11-32-44`
  - `Acquisition_Board-100.acquisition_board`
  - `num_channels=128`
  - all 128 channels have identifier
    `acq-board.rhythm.continuous.ephys`, units `uV`, names `CH1..CH128`
  - no ADC channels in the ephys stream
- `cbsuruizfs1/storage/shared/data/AutoMaze/RM018/RM018_day41_260622/RM018_ProbSwitching_2026-06-22_13-37-27`
  - `Acquisition_Board-100.acquisition_board`
  - `num_channels=200`
  - first 192 channels are ephys:
    `acq-board.rhythm.continuous.ephys`, units `uV`, names `CH1..CH192`
  - last 8 channels are ADC:
    `acq-board.rhythm.continuous.adc`, units `V`, names `ADC1..ADC8`

The current code does not distinguish these channel classes. It reads
`continuous_entries[0].num_channels` as the Open Ephys channel count, so the
RM018 example can be treated as a 200-channel ephys recording even though only
192 channels should be sent to preprocessing and sorting.

The current single-run pipeline also assumes one acquisition source type for
the whole run:

- `discover_subsessions()` first discovers Open Ephys recordings and only falls
  back to Intan-style `amplifier.dat` folders when no Open Ephys recordings are
  found.
- `build_acquisition_catalog()` branches on the first discovered item being an
  Open Ephys recording root. The resulting `AcquisitionCatalog` is either all
  Open Ephys or all Intan.
- Open Ephys catalogs currently set `board_adc_channels=0`,
  `analogin_paths=[]`, and `board_adc_native_orders=[]`, so OE ADC cannot be
  exported as analog events.

## Why This Is Needed Now

There are real multi-day/multi-subepoch datasets where Intan and Open Ephys
recordings are mixed across selected subepochs. Some Open Ephys recordings also
embed ADC channels in the acquisition-board continuous stream. Without explicit
source-aware channel classification, the pipeline may silently include ADC
channels in the ephys binary, produce a wrong `basename.dat` channel count, and
send non-ephys channels to Kilosort.

## Git State at Planning Time

- Date: 2026-07-19
- Branch: `feature-multiday-subepoch`
- Working tree: clean

## Affected Modules and Files

- `src/preprocess/io.py`
  - discovery of mixed Intan/Open Ephys subepochs;
  - Open Ephys `structure.oebin` stream/channel classification;
  - acquisition catalog fields for per-subepoch source metadata and ADC channel
    extraction.
- `src/preprocess/metafile.py`
  - `AcquisitionCatalog` shape if new per-subepoch fields are needed.
- `src/preprocess/recording.py`
  - loading Open Ephys ephys-only channel views;
  - optional extraction/writing of OE ADC sidecar data for analog events.
- `src/preprocess/events.py`
  - analog event export should accept a generated OE ADC `analogin.dat` or an
    equivalent recording-backed source.
- `src/preprocess/pipeline.py`
  - effective channel count validation across mixed sources;
  - merge point generation from ephys-only paths/counts;
  - event export using Intan `analogin.dat` and OE ADC channels together.
- `src/preprocess/multiday.py`
  - staging sample-count logic should use the source binary channel count for
    frame counting, while recording the OE ephys channel count separately for
    preprocessing and sorting validation.
  - manifest/CSV should include OE ephys and ADC channel counts when available.
- `tests/preprocess/`
  - mixed source catalog tests;
  - Open Ephys ADC classification tests;
  - end-to-end preprocess fixture for ephys-only output channel count and
    analog event availability.

## Public Parameters and API Changes

No GUI control should be required for the common path. Open Ephys ADC handling
should be automatic from `structure.oebin`.

Expected internal additions:

```python
OpenEphysStreamInfo(
    recording_root: Path,
    stream_name: str,
    continuous_dat: Path,
    sample_rate: float,
    total_channels: int,
    ephys_channel_indices: list[int],
    adc_channel_indices: list[int],
    ephys_channel_names: list[str],
    adc_channel_names: list[str],
)
```

`AcquisitionCatalog` may need per-subepoch lists such as:

```python
source_types: list[Literal["intan", "openephys"]]
ephys_channel_indices_by_subsession: list[list[int] | None]
adc_channel_indices_by_subsession: list[list[int]]
source_total_channels: list[int]
source_ephys_channels: list[int]
source_adc_channels: list[int]
```

If this can be represented cleanly without widening the public config surface,
prefer internal dataclass fields only.

## Algorithm Details

### 1. Classify Open Ephys acquisition-board channels

Read `structure.oebin` and inspect every entry under `continuous`.

Choose the ephys stream by channel metadata, not by entry order:

- ephys channel if `channel.identifier` contains `.ephys`, or channel units are
  `uV` and channel names look like `CH<number>`;
- ADC channel if `channel.identifier` contains `.adc`, or channel units are `V`
  and channel names look like `ADC<number>`;
- ignore `memory_usage` streams for ephys and ADC extraction.

For the inspected datasets this should produce:

- ayadataB4 Day22 postsleep: `ephys=128`, `adc=0`, `total=128`;
- RM018 day41: `ephys=192`, `adc=8`, `total=200`.

If multiple continuous entries contain ephys channels, fail with a clear error
unless they are an explicitly supported multi-stream layout. Silent selection
would be risky.

### 2. Discover mixed subepochs

Change `discover_subsessions()` or add a new mixed-aware discovery helper so a
basepath can return both:

- Open Ephys recording roots (`structure.oebin` under recording1); and
- Intan-style `amplifier.dat` or `continuous.dat` child folders.

The final list should still sort by the existing `_subsession_sort_key()` so
session order remains stable.

### 3. Build a mixed-source acquisition catalog

Refactor `build_acquisition_catalog()` away from the current first-item branch.
For each discovered subepoch:

- Intan:
  - ephys path is `amplifier.dat` or direct `continuous.dat`;
  - ephys channel count comes from XML `nChannels`;
  - ADC comes from `analogin.dat` plus `info.rhd` when available.
- Open Ephys:
  - ephys path is the acquisition-board `continuous.dat`;
  - ephys channel count is `len(ephys_channel_indices)`;
  - total stored channel count is `continuous.num_channels`;
  - ADC count is `len(adc_channel_indices)`;
  - TTL path remains the matching `events/<stream>/TTL` folder.

Validate that all ephys channel counts match the expected probe/chanMap count
for a single concatenated run. If counts differ, stop before writing output.

### 4. Load ephys-only recordings

For Open Ephys streams with ADC channels, load the full acquisition-board stream
through SpikeInterface, then select only ephys channel ids before concatenation.
Do this before preprocessing, raw save, LFP writing, and sorting.

The final `basename.dat`, `.lfp`, `session.mat`, sorter partitions, and
Kilosort inputs should use the ephys-only channel count.

When Open Ephys ephys channels are not already the identity range `0..N-1`,
write the local output `chanMap.mat` in final ephys-only dat column
coordinates. Probe attachment, bad-channel handling, artifact grouping, sorter
partitioning, Kilosort chanMap override, and postprocess recording
reconstruction must all consume that same canonical local `chanMap.mat`. If the
original chanMap cannot be interpreted unambiguously as either final
ephys-channel coordinates or source acquisition-board coordinates, fail before
preprocessing or sorting instead of silently mixing coordinate systems.

Do not introduce a second generated chanMap filename for normal workflows. If
the user provides an external chanMap, leave that source file untouched and
write/copy the canonical final-channel version to the local output
`chanMap.mat`.

When a source-coordinate chanMap is remapped, manual `reject_channels` supplied
with the run must be remapped through the same source-to-final channel map
before being merged with XML skipped channels and chanMap `connected=false`
channels. XML skipped channels remain final/session-channel coordinates.

### 5. Preserve ADC for analog event export

For Open Ephys subepochs with ADC channels:

- extract ADC channel traces from the same acquisition-board recording;
- write or expose an `analogin.dat`-equivalent stream for event export;
- align sample counts with the matching ephys segment;
- use ADC channel names/orders from `structure.oebin` for active channel
  metadata when possible.

For mixed Intan/OE runs:

- Intan subepochs can still use existing `analogin.dat`;
- OE subepochs can use extracted ADC traces;
- missing ADC for a subepoch should zero-fill or be represented as absent using
  the existing sidecar concatenation behavior, but this must be explicit and
  tested.

### 6. Multi-day staging metadata

Update multi-day manifest/CSV so each selected subepoch records:

- `source_type`;
- `source_total_channels`;
- `source_ephys_channels`;
- `source_adc_channels`;
- `binary_n_channels` meaning the physical source binary channel count used
  for sample counting;
- source stream name/folder when Open Ephys.

Implementation note: the original draft treated `binary_n_channels` as the
ephys channel count, but embedded-ADC Open Ephys `continuous.dat` files must be
divided by the total stored channel count to get frame counts. The final design
therefore stores both `source_total_channels`/`binary_n_channels` for source
binary accounting and `source_ephys_channels` for the channel count used by
preprocessing, sorting, merge timelines, and `session.mat`.

## Expected Behavior

- All-Intan runs remain unchanged.
- Open Ephys runs with only ephys channels remain unchanged except for richer
  manifest metadata.
- Open Ephys runs with embedded ADC produce ephys-only `basename.dat` and use
  the probe channel count for sorting.
- Open Ephys ADC channels are available for analog event export when
  `analog_inputs=True`.
- Mixed Intan/OE multi-day runs can be staged and preprocessed when all ephys
  channel counts and sampling rates are compatible.
- Mixed runs fail early with a clear error when ephys channel counts,
  sampling rates, or unsupported OE stream layouts are incompatible.

## Tests and Verification

### Worker B pipeline/recording bridge

This implementation pass is scoped to the pipeline and recording integration
points while catalog construction is being implemented separately.

- `recording.load_subsession_recordings()` will accept optional
  `ephys_channel_indices_by_subsession` metadata. Open Ephys recordings will be
  loaded through `se.read_openephys()`, renamed to integer channel ids as before,
  and sliced to the listed ephys channel ids only when metadata is present.
  Binary Intan loading remains unchanged.
- `pipeline.run_preprocess_session()` will pass the optional ephys-channel list
  from the acquisition catalog, using compatibility defaults until the catalog
  dataclass fields land.
- Open Ephys TTL export will be driven by per-subsession source metadata when
  available, so mixed Intan/Open Ephys runs can pass a list containing `None`
  entries for non-OE subepochs.
- Analog sidecar concatenation will use catalog-provided
  `analogin_source_paths` when available. This creates a reviewable boundary for
  later Open Ephys ADC extraction without coupling this worker to catalog
  internals.

### Unit tests

- Add `tests/preprocess/test_openephys_stream_classification.py` with compact
  `structure.oebin` fixtures:
  - 128 ephys, 0 ADC;
  - 192 ephys, 8 ADC;
  - memory_usage entry ignored;
  - unsupported multiple ephys streams raises a clear error.
- Add `tests/preprocess/test_io_source_selection.py` coverage for mixed
  discovery: `OE, Intan, OE` returns all three in sorted order.
- Add `tests/preprocess/test_multiday.py` coverage that multi-day staging uses
  ephys channel count for OE sample counting while recording ADC count in the
  manifest/CSV.

### Pipeline tests

- Add a small mixed-source fixture:
  - Intan epoch with `amplifier.dat` and optional `analogin.dat`;
  - OE epoch with one acquisition-board `continuous.dat` containing ephys+ADC;
  - matching ephys channel counts across both epochs.
- Assert that:
  - `basename.dat` size is divisible by ephys channel count, not total OE
    channel count;
  - `session.mat` reports ephys channel count;
  - sorter partition manifest uses ephys channel count;
  - non-prefix OE ephys selection remaps source-coordinate chanMap rows to final
    ephys-only dat columns before sorting;
  - analog event export sees ADC channels when `analog_inputs=True`.
- Add a negative test where OE ephys channel count differs from Intan/XML
  count and verify early failure before final dat/sorter output.

### Regression tests

- Existing Intan tests:
  - `pytest -q tests/preprocess/test_intan_rhd_header.py`
  - `pytest -q tests/preprocess/test_pipeline_rhd_integration.py`
- Existing Open Ephys/selection tests:
  - `pytest -q tests/preprocess/test_io_source_selection.py`
  - `pytest -q tests/preprocess/test_multiday.py`
  - `pytest -q tests/preprocess/test_events_neurocode_compat.py`
  - `pytest -q tests/preprocess/test_recording_selected_transform.py`

### End-to-end safety checks

- Run `python -m py_compile` on modified Python files.
- Run `git diff --check`.
- For the two inspected real OE sessions, run a read-only diagnostic command
  that prints:
  - stream folder;
  - total channels;
  - ephys channel count;
  - ADC channel count;
  - sample rate;
  - inferred sample count.

Do not write to the real data locations during diagnostics.

## Non-Goals

- Do not add manual GUI controls for OE ADC selection until automatic metadata
  handling is proven insufficient.
- Do not support concatenating different ephys channel counts in one sorting
  run.
- Do not treat `memory_usage` as analog input.
- Do not modify source recording folders.
- Do not change Kilosort parameters or sorter GPU behavior.

## 2026-08-10 Revision: Per-Subepoch ADC Identity and Layout

This revision supersedes the count-only ADC padding described above wherever
the two designs differ. A maximum ADC count is not a channel layout: writing
every subepoch's ADC columns into output columns `0..N-1` can silently exchange
physical inputs when enabled channels are sparse or their source order changes.

### Observed Failure and Required Outcome

The selected Intan data include these distinct layouts:

- Day23 has a one-column `analogin.dat`, while its adjacent RHD header reports
  zero ADC channels;
- Day54 and Day61 have eight-column `analogin.dat` files and adjacent RHD
  headers reporting eight ADC channels;
- the multi-day root RHD was copied from an unselected Day14 Opto epoch and
  reports one ADC channel.

The current catalog correctly infers the physical `analogin.dat` width, but
then applies one session-level `IntanRhdHeader` to every Intan subepoch and to
the aggregate output. This makes an unrelated root/copied RHD capable of
overwriting selected-subepoch counts and identities. Open Ephys already stores
per-subepoch ADC source indices and names, but concatenation left-packs the
selected columns, so names such as `ADC3, ADC1` do not determine their output
destinations.

The required result is one canonical output ADC identity axis for the selected
subepochs. Every source ADC column must be scattered to the destination column
for the same physical identity. An identity absent from a subepoch is
zero-filled for exactly that subepoch's sample interval. The aggregate channel
count is the size of the identity union, not merely the largest source count.

### Canonical Identity and Mapping Semantics

Use zero-based board ADC native order as the canonical identity:

- Intan identity comes from the selected subepoch's own RHD
  `board_adc_native_orders`, in the same order as columns in that subepoch's
  `analogin.dat`.
- Open Ephys identity comes from a case-insensitive, full-name parse of
  `ADC<number>` in `structure.oebin`; `ADC1` maps to native order `0`, `ADC8`
  maps to `7`. The existing raw acquisition-board indices remain separate and
  continue to identify which columns to extract from `continuous.dat`.
- The canonical output identities are the sorted union of identities present
  in selected subepochs. Output column `j` represents
  `canonical_adc_native_orders[j]`; it does not necessarily represent native
  order `j`.
- For each subepoch, record a source-to-output mapping parallel to its selected
  ADC columns. For example, source identities `[2, 0]` and canonical identities
  `[0, 1, 2]` produce destination indices `[2, 0]`. Native order `1` is
  zero-filled in that subepoch.
- Reject duplicate identities within a subepoch, duplicate destination
  indices, negative native orders, and any mapping whose source-column,
  identity, and destination lengths differ. Never resolve these cases by
  dropping or renumbering channels.

`board_adc_channels` remains the number of columns in the concatenated
`analogin.dat`, and `board_adc_native_orders` becomes the ordered canonical
identity list. Therefore `analogIn.channels` remains one-based physical
identity metadata (`native_order + 1`) aligned element-for-element with the
compact output columns.

### Intan Header Resolution and Fallback Contract

ADC layout metadata must be resolved independently for every selected Intan
subepoch, after discovery establishes the selected paths:

1. Look only in the recording directory adjacent to that subepoch's
   `amplifier.dat` for its local `info.rhd` or the existing supported local RHD
   filename.
2. Infer the actual source ADC width from `analogin.dat` size and the matching
   amplifier sample count. The file shape is authoritative for safe binary
   reading.
3. Use local RHD native orders only when their count equals the inferred file
   width. A header reporting zero ADC channels for a nonempty `analogin.dat` is
   treated as layout metadata unavailable, not as evidence that the file has
   zero columns. This covers Day23.
4. When local metadata are missing, unreadable, or report zero for a nonempty
   sidecar, use the legacy positional identity fallback `0..N-1`, record the
   layout provenance as inferred/positional, and emit a warning naming the
   subepoch. The Day23 one-column sidecar consequently maps to ADC1/native
   order 0.
5. If a local header reports a nonzero count or native-order list that
   conflicts with the inferred width, fail with the subepoch path, inferred
   width, reported count, and reported identities. Contradictory nonzero
   metadata must not silently fall back to position.

The root/session RHD produced or selected by `ensure_rhd()` may remain an output
provenance artifact and may continue to provide legacy session-wide metadata
that is outside this revision. It must not supply or override any selected
subepoch's ADC count, identity list, or source-to-destination mapping. An RHD
beside a directly recorded basepath-level `amplifier.dat` is local to that
recording and is not considered a fallback root header.

### Open Ephys Identity Fallback Contract

Prefer explicit `ADC<number>` names for every channel classified as ADC. Preserve
the raw names in the catalog for diagnostics.

- If all ADC names in a subepoch parse uniquely, use those identities even
  when the channels are sparse or reordered.
- If none of the ADC names provide an identity, retain the legacy positional
  fallback `0..N-1` only for compatibility, record inferred/positional
  provenance, and warn with the recording and stream name.
- If only some names parse, or parsed identities are duplicated, fail as
  ambiguous. Do not combine named and positional identities in one subepoch.

This fallback preserves older `structure.oebin` files that classify ADC by
identifier/units but lack usable names, while preventing a partially described
layout from being silently misaligned.

### Catalog and Concatenation Contract

Extend `AcquisitionCatalog` with aligned per-subepoch metadata. Concrete names
may be adjusted to repository conventions, but their semantics must remain
distinct:

```python
adc_channel_indices_by_subsession: list[list[int]]
# Source binary columns: 0..N-1 for Intan analogin.dat, acquisition-board
# column indices for Open Ephys continuous.dat.

adc_native_orders_by_subsession: list[list[int]]
# Physical ADC identities, parallel to source channel indices.

adc_output_indices_by_subsession: list[list[int]]
# Destination columns in the compact concatenated analogin.dat, parallel to
# source channel indices and native orders.

adc_layout_sources_by_subsession: list[str]
# At minimum: "intan_rhd", "openephys_name", "inferred_positional", or "none".
```

Keep `source_adc_channels` as the actual number of ADC columns extracted or
read for each subepoch. Keep `adc_channel_names_by_subsession` as source/audit
metadata; names alone are not destination positions. All per-subepoch lists
must align with `amplifier_paths`, including explicit empty lists for epochs
without ADC.

Change analog concatenation from left-padding to scatter-by-destination:

1. Read each Intan sidecar with its inferred source width, or read the Open
   Ephys acquisition-board stream with `source_total_channels` and select its
   ADC source indices.
2. Allocate zeroed chunks with `len(board_adc_native_orders)` output columns.
3. Assign each selected source column to its declared destination column.
4. Zero-fill missing identities and whole epochs without ADC, preserving the
   amplifier merge timebase and existing sample-count checks.
5. Preserve the current Intan `uint16` direct-copy behavior and the established
   Open Ephys source/output dtype conversion; voltage calibration changes are
   outside this revision.

The low-level concatenation helper should accept an optional destination-index
list per source. Calls that omit it retain the existing left-packed behavior
for backward compatibility, while catalogs built by the pipeline always pass
the explicit mappings.

### Affected Files

- `src/preprocess/io.py`
  - resolve and parse an RHD independently beside each selected Intan
    subepoch;
  - infer source ADC widths without allowing a global RHD to overwrite them;
  - normalize Intan native orders and Open Ephys ADC names to canonical
    identities;
  - build the canonical identity union and per-subepoch destination mappings.
- `src/preprocess/metafile.py`
  - add the aligned per-subepoch identity, destination, and provenance fields
    with compatibility defaults.
- `src/preprocess/recording.py`
  - validate and scatter selected source ADC columns into explicit output
    destinations while preserving zero-fill and sample-count guarantees.
- `src/preprocess/pipeline.py`
  - stop passing one session-level RHD as every subepoch's ADC layout;
  - pass destination mappings to analog concatenation;
  - export canonical native orders as active analog channel identities.
- `tests/preprocess/test_pipeline_rhd_integration.py`
  - per-subepoch Intan RHD selection, stale root RHD isolation, and fallback
    behavior.
- `tests/preprocess/test_openephys_stream_classification.py`
  - sparse, reordered, missing, partial, and duplicate Open Ephys ADC names.
- `tests/preprocess/test_recording_selected_transform.py`
  - exact scatter placement and zero-fill behavior.
- `tests/preprocess/test_pipeline_mixed_sources.py`
  - canonical mapping and event-channel metadata across Intan/Open Ephys
    boundaries.

No `implementation_plan/README.md` edit is needed because this existing plan is
already indexed.

### Backward Compatibility

- No new GUI or user configuration is required.
- Existing all-Intan runs with a consistent local RHD keep their physical
  native orders. Runs without usable local layout metadata retain positional
  `0..N-1` behavior with a warning.
- Existing Open Ephys recordings with dense `ADC1..ADCN` names produce the same
  byte-column ordering as today.
- Existing callers of the sidecar writer that do not provide destination
  mappings retain left-packed output.
- Empty default values on new catalog fields allow older test fixtures and
  external constructors to continue working; pipeline validation must reject
  only partially populated new metadata, not catalogs that omit it entirely.
- Source recording files and external RHD files remain read-only. The local
  copied basename RHD remains untouched by layout resolution.

### Tests and Verification for This Revision

Add regression coverage for these concrete cases:

- selected Intan Day23/Day54/Day61-like fixtures with inferred widths
  `[1, 8, 8]`, local header counts `[0, 8, 8]`, and an unselected/root header
  reporting one channel; assert per-subepoch counts and identities are not
  overwritten by the root header;
- Intan source columns with local native orders `[3, 1]`; assert their samples
  scatter to the canonical columns for identities 3 and 1 rather than output
  columns 0 and 1 by source position;
- a missing/unreadable or zero-ADC local header plus a nonempty sidecar; assert
  positional fallback, provenance, and warning;
- a nonzero local-header/file-width contradiction; assert an early diagnostic
  failure before output `analogin.dat` is written;
- Open Ephys layouts `[ADC3, ADC1]` and `[ADC1, ADC2, ADC3]`; assert the first
  epoch maps to canonical destinations `[2, 0]` and zero-fills ADC2;
- Open Ephys names that are wholly absent (warned positional fallback), partly
  parseable, or duplicated (the latter two fail as ambiguous);
- mixed Intan/Open Ephys epochs that refer to the same native orders; assert
  one shared output column per identity and correct one-based
  `analogIn.channels` metadata;
- an epoch with no ADC; assert full-width zeros for its exact amplifier sample
  count and unchanged merge timestamps;
- legacy low-level sidecar-writer calls without destination mappings; assert
  their existing left-packed output remains unchanged.

Run at minimum:

```text
pytest -q tests/preprocess/test_intan_rhd_header.py
pytest -q tests/preprocess/test_pipeline_rhd_integration.py
pytest -q tests/preprocess/test_openephys_stream_classification.py
pytest -q tests/preprocess/test_recording_selected_transform.py
pytest -q tests/preprocess/test_pipeline_mixed_sources.py
pytest -q tests/preprocess/test_events_neurocode_compat.py
python -m py_compile <modified Python files>
git diff --check
```

For the selected real dataset, perform a read-only catalog diagnostic before a
write-enabled pipeline run. Print, per selected subepoch: source type, source
path, inferred ADC width, local metadata path, layout provenance, source ADC
indices, native orders, and output destinations. Confirm Day23/Day54/Day61 are
`[1, 8, 8]`, the Day14 root RHD is not used for their layouts, and every
destination mapping is one-to-one.

### Additional Non-Goals

- Do not infer a sparse physical identity from signal values, TTL activity, or
  cross-correlation.
- Do not accept a global/root RHD as a substitute for missing selected-subepoch
  ADC layout metadata.
- Do not add manual channel-remapping controls in this pass.
- Do not resample or voltage-calibrate ADC data, and do not change analog event
  detection thresholds.
- Do not change ephys channel selection, probe/chanMap semantics, sorter
  behavior, digital-input identity handling, or multi-day selection order as
  part of this revision.
