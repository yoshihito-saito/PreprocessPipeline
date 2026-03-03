# Kilosort4 Change Log

## 2026-02-27

### Pipeline integration
- Added Kilosort4 support in the Python preprocessing pipeline (`sorter='kilosort4'` / `sorter='Kilosort4'`).
- Output folder naming now supports `Kilosort4_YYYY-MM-DD_HHMMSS`.
- Sorting result discovery now supports both `Kilosort_*` and `Kilosort4_*` for backward compatibility.

### Sorter runner behavior
- Added Kilosort4 parameter normalization and unsupported-key filtering (unsupported keys are dropped with a warning).
- CLI `--config` now defaults to sorter-specific config when omitted.
- Added Kilosort4-specific error path (no MATLAB-log hint for Kilosort4 failures).

### Kilosort4 config
- Added and updated [`sorter/Kilosort4_config.yaml`](../Kilosort4_config.yaml).
- Restored tuned parameters such as `nt`, `batch_size`, `Th_*`, `n_templates`, etc.
- Added probe-geometry-driven auto parameter options:
  - `auto_geom_from_probe`
  - `auto_geom_nearest_chans_target`
  - `auto_geom_whitening_range_target`
  - `auto_geom_max_channel_distance_factor`
  - `auto_geom_gap_factor`
  - `auto_geom_max_xcenters_cap`

### Probe-based auto geometry
- Implemented auto derivation of geometry-sensitive KS4 parameters from attached probe geometry:
  - `dmin`, `dminx`, `max_channel_distance`, `x_centers`, `nearest_chans`, `whitening_range`
- Auto-geometry is applied only when enabled in `Kilosort4_config.yaml`.

### Preprocessed dat workflow
- `basename.dat` is now treated as the default sorter input (preprocessed signal).
- When `save_raw=True`, raw concatenated data is exported separately as `basename_raw.dat`.
- Preprocessing is applied to selected/good channels while preserving full channel count in output `.dat`.

### Sort output layout and params.py
- After sorting, `sorter_output/` contents are flattened into the Kilosort run root folder, and `sorter_output/` is removed.
- `params.py` `dat_path` is now always written as a relative path (relative to each `params.py` location).
- `params.py` `hp_filtered` is forced to `True` when sorter input is already preprocessed.

### Local Kilosort package patch notes
- Local environment patch to `kilosort/clustering_qr.py` is currently:
  - `swarmsplitter.split(..., meta=st0)` (restored from temporary `meta=None` workaround).

