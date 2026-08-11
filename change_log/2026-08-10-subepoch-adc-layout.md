# Per-subepoch ADC identity and layout

Date: 2026-08-10  
Commit: uncommitted changes on top of `7740a4d`

Implementation plan: [mixed Intan/Open Ephys ADC handling](../implementation_plan/2026-07-19-mixed-source-openephys-adc.md)

## Changes

- Resolved Intan ADC sidecar widths and native identities separately for every
  selected recording directory. Only an adjacent local RHD is consulted;
  session/root RHD metadata cannot overwrite a selected epoch layout.
- Added warned positional fallback for unavailable/zero Intan metadata and for
  entirely unnamed Open Ephys ADC layouts. Nonzero Intan width conflicts and
  partial, duplicate, or invalid Open Ephys `ADC<number>` identities now fail.
- Added aligned catalog provenance, native-order, and destination mapping
  fields. The aggregate ADC columns are the sorted union of physical native
  orders rather than a maximum source width.
- Updated analog sidecar concatenation to scatter source columns to explicit
  destinations and zero-fill missing identities/epochs. Calls without a
  destination mapping retain legacy left-packed behavior.
- Existing sidecars are not reused when any destination mapping differs from
  legacy left-packed order because binary shape alone cannot prove their layout
  provenance; callers must set `overwrite=True` to rebuild them safely. Dense
  identity mappings retain shape-validated reuse compatibility.
- Preserved truly blank Open Ephys ADC names as raw audit metadata, ensuring
  they take the warned positional fallback rather than becoming synthesized,
  falsely authoritative identities.
- Passed catalog destination mappings to the pipeline sidecar writer, while
  retaining canonical one-based analog event metadata through
  `board_adc_native_orders`. Populated mapping metadata is validated for every
  ADC-bearing subepoch; wholly omitted default metadata remains compatible.

## Verification

Successful:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q \
  tests/preprocess/test_openephys_stream_classification.py \
  tests/preprocess/test_recording_selected_transform.py \
  tests/preprocess/test_pipeline_mixed_sources.py
# 31 passed

/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q \
  tests/preprocess/test_intan_rhd_header.py \
  tests/preprocess/test_events_neurocode_compat.py \
  tests/preprocess/test_io_source_selection.py
# 28 passed

/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m py_compile \
  src/preprocess/io.py src/preprocess/metafile.py \
  src/preprocess/recording.py src/preprocess/pipeline.py

git diff --check
```

The broader required command also ran:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -m pytest -q \
  tests/preprocess/test_openephys_stream_classification.py \
  tests/preprocess/test_recording_selected_transform.py \
  tests/preprocess/test_pipeline_mixed_sources.py \
  tests/preprocess/test_pipeline_rhd_integration.py \
  tests/preprocess/test_intan_rhd_header.py \
  tests/preprocess/test_events_neurocode_compat.py
# 45 passed, 11 failed
```

The 11 failures are outside this change: the current integration tests use
removed artifact configuration fields (`remove_artifact_TTL` and
`remove_highamp_artifact`), expect the pre-existing eager-load behavior, or
expect an older high-amplitude worker count. The focused ADC paths passed.

## Limitations

The root agent ran the required read-only diagnostic against the selected real
dataset:

```text
/local/workdir/ys2375/miniforge3/envs/phy2/bin/python -c "import json; from pathlib import Path; from src.preprocess.io import build_acquisition_catalog; from src.preprocess.intan_rhd import read_intan_rhd_header; m=Path('/fs/ayadata2-afr77.nbb.cornell.edu/volume4/ayadataB4/Fabrication/PV_PFC3/multiday_Day14_to_Day217/multi_day_manifest.json'); d=json.loads(m.read_text()); paths=[Path(x['staged_subepoch_path'])/'amplifier.dat' for x in d['subepochs']]; root=Path('/fs/ayadata2-afr77.nbb.cornell.edu/volume4/ayadataB4/Fabrication/PV_PFC3/multiday_Day14_to_Day217/multiday_Day14_to_Day217.rhd'); c=build_acquisition_catalog(paths,128,'int16',read_intan_rhd_header(root)); print('canonical',c.board_adc_native_orders,'channels',c.board_adc_channels); [(print(c.subsession_names[i], 'width=',c.source_adc_channels[i], 'orders=',c.adc_native_orders_by_subsession[i], 'dest=',c.adc_output_indices_by_subsession[i], 'source=',c.adc_layout_sources_by_subsession[i])) for i in range(len(paths)) if c.source_adc_channels[i]]"
```

Result: the only warning was the expected Day23 zero-header positional
fallback. Canonical native orders were `[0, 1, 2, 3, 4, 5, 6, 7]` with eight
output channels. Day23 had width 1, orders/destinations `[0]`, and
`inferred_positional` provenance. Day54 and Day61 had width 8,
orders/destinations `[0..7]`, and `intan_rhd` provenance. No write-enabled
real-data pipeline run was performed.
