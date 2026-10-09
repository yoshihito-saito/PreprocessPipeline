# Parameter reference

This reference describes the JSON saved by **Save config** and loaded by
`python -m src.preprocess.gui.run_pipeline`. See [CLI guide](cli.md) for commands
and example configurations. Omitted fields use the defaults below; loading a
saved configuration uses its saved values instead.

The defaults come from
[PipelineGuiSettings](../src/preprocess/gui/config_model.py). They differ from
some defaults in the lower-level Python `PreprocessConfig` and
`PostprocessConfig` classes. These configuration formats are not interchangeable.

## Configuration structure and conventions

The top level contains session paths and the `preprocess`, `postprocess`,
`behavior`, and `execution` objects. Use JSON booleans (`true`, `false`), arrays
for channel lists and numeric pairs, and strings for paths. Use absolute data
paths; relative sorter installation/configuration paths resolve from the
repository root. On Windows, use forward slashes or escaped backslashes in JSON.

- Ephys channel IDs, XML group indices in `probe_assignments`, and artifact TTL
  bit indices are **0-based**. Phy retains original binary channel IDs.
- CellExplorer channel values saved to MATLAB files are **1-based**.
- Behavior point indices and the numeric suffixes in `adc:N` and `ttl:N` are
  **1-based**. For example, `ttl:1` selects the input displayed as `TTL0`.
- Frequencies are Hz, probe distances are micrometers, artifact/duplicate windows
  are milliseconds, and behavior/state durations are seconds unless stated otherwise.

## Session paths and ordering

| Key | Default | Meaning |
| --- | --- | --- |
| `basepath` | `""` | Raw recording session directory. Required for preprocessing. Its directory name supplies the single-day basename. |
| `local_root` | `""` | Working directory; empty uses `<repository>/preprocess_tmp`. Session outputs go to `<local_root>/<basename>`. |
| `xml_path` | `""` | Session XML describing the acquisition rate, full ephys channel count, groups, and skipped channels. Empty searches for `<basename>.xml` in the session/output directories. |
| `chanmap_path` | `""` | Channel-map path. Preprocessing prepares the canonical map from XML, probe assignments, and exclusions; postprocessing can use an explicit map for an external sorting. |
| `subsession_order` | `[]` | Explicit subrecording order. Empty uses normal discovery order. Frame order must agree with any externally supplied sorting. |
| `multi_day_enabled` | `false` | Stage recordings from multiple day directories as one session. |
| `multi_day_session_paths` | `[]` | Ordered source day directories; at least two are required in multi-day mode. |
| `multi_day_selected_subepoch_paths` | `[]` | Source subepoch directories to include. Empty includes all discovered subepochs. |
| `multi_day_name` | `""` | Required output/staging basename in multi-day mode. |
| `source_basepath` | `""` | Runtime/resume source path, usually filled by the GUI or persistent Run creation. Leave empty for a new raw session. |
| `existing_session_dir` | `""` | Runtime/resume processed session directory. When set, outputs reuse this directory instead of creating `<local_root>/<basename>`. |

Intan, Open Ephys, and merged WILD recordings use acquisition-specific readers.
Mixing formats does not remove the requirement for compatible ephys channel
layout, recording rate, and channel meanings. Open Ephys ADC columns are handled
separately from the ephys channel map.

## Preprocessing

All keys in this section belong inside `preprocess`.

### Signal processing and channels

| Key | Default | Meaning |
| --- | --- | --- |
| `do_preprocess` | `true` | Apply filtering and reference to retained good channels. |
| `bandpass_min_hz` | `500.0` | Lower bandpass cutoff. |
| `bandpass_max_hz` | `8000.0` | Upper bandpass cutoff; must fit the recording's sampling rate. |
| `reference` | `"local"` | Common reference: `none`, `local`, or `global`. |
| `local_radius_um` | `[20.0, 200.0]` | Inner/outer neighbor radii for local reference, using probe coordinates. |
| `reject_channels` | `[]` | Additional original 0-based ephys channel IDs to exclude, combined with XML/map exclusions. |
| `probe_assignments` | `[]` | Probe layout and XML group membership. Empty derives the layout and groups from XML. |
| `save_raw` | `false` | Save a concatenated raw amplifier binary alongside processed outputs. |
| `analog_inputs` | `false` | Export recorded analog input events. Needed when synchronization uses an ADC input. |
| `digital_inputs` | `true` | Export digital input events. |
| `overwrite` | `false` | Reuse compatible completed outputs; `true` permits regeneration. |

The binary retains its full channel columns. The GUI conversion keeps the
lower-level `zero_bad=true` default, so bad columns are zeroed rather than
removed. Sorting and postprocessing exclude them while preserving the mapping
to original channel IDs. Filtering/reference settings here are also used by
postprocessing when `postprocess.apply_preprocess=true`.

Each `probe_assignments` entry has `type` (a supported probe layout name),
`groups` (an array of 0-based XML anatomical group indices), and `x_offset`
(micrometers). For example, `{"type": "staggered", "groups": [0, 1], "x_offset": 0}`.
Use assignments matching the actual probe; channel membership comes from XML.

### Artifact removal

Group modes are `none`, `all`, `probe`, and `shank`. `none` disables removal;
the other modes choose the scope over which events are applied/detected.

| Key | Default | Meaning |
| --- | --- | --- |
| `remove_ttl_artifacts` | `false` | Enable TTL-driven artifact removal. Otherwise the effective group mode is `none`. |
| `artifact_ttl_group_mode` | `"none"` | TTL removal scope. Set a non-`none` mode as well as enabling removal. |
| `artifact_ttl_channel` | `0` | Digital bit index, 0–15, supplying artifact triggers. |
| `export_opto_events` | `false` | Save that TTL channel as `<basename>.opto.manipulation.mat`, including when artifact removal is disabled. |
| `artifact_ttl_include_offset` | `false` | Remove windows around falling edges as well as rising edges. |
| `artifact_ttl_ms_before` | `0.5` | Removal window before each trigger, ms. |
| `artifact_ttl_ms_after` | `2.0` | Removal window after each trigger, ms. |
| `artifact_ttl_mode` | `"linear"` | Replacement method: `linear`, `cubic`, or `"0"` for zero fill. |
| `remove_highamp_artifacts` | `false` | Enable high-amplitude artifact detection/removal. |
| `artifact_highamp_group_mode` | `"shank"` | High-amplitude detection/removal scope; effective mode is `none` when disabled. |
| `highamp_threshold_sigma` | `10.0` | Detection threshold in estimated noise-standard-deviation units. |
| `highamp_ms_before` | `2.0` | Removal window before a detected event, ms. |
| `highamp_ms_after` | `2.0` | Removal window after a detected event, ms. |
| `highamp_mode` | `"linear"` | Replacement method: `linear`, `cubic`, or `"0"`. |

The JSON adapter fixes high-amplitude noise estimation to 500 one-second
windows, seed 0, one-second processing chunks, and a one-millisecond dead time.
Those controls are lower-level Python parameters, not additional GUI JSON keys.

### LFP and state scoring

| Key | Default | Meaning |
| --- | --- | --- |
| `make_lfp` | `true` | Generate the session `.lfp` binary. |
| `lfp_fs` | `1250.0` | Output LFP sampling rate, Hz. |
| `state_score` | `true` | Generate sleep-state scoring outputs from the LFP. |
| `sw_channels` | `[]` | Candidate 0-based slow-wave channels. Empty lets the scorer select candidates. |
| `theta_channels` | `[]` | Candidate 0-based theta channels. Empty lets the scorer select candidates. |
| `state_winparms` | `[2.0, 15.0]` | Spectral window length and metric smoothing duration, both seconds. |
| `emg_th_alpha` | `1.0` | Multiplier on the automatically estimated EMG threshold. |
| `useEMG_NREM` | `true` | Use EMG to reject NREM candidates and preserve EMG-driven wake intervals. |
| `state_min_state_length` | `6.0` | Minimum state-bout duration used by cleanup, seconds. |
| `state_microarousal_sec` | `100.0` | Wake-duration threshold for microarousal handling and wake-to-REM cleanup, seconds. |
| `state_block_wake_to_rem` | `true` | Apply wake-to-REM transition suppression. |
| `state_save_lfp_mat` | `true` | Save the state scorer's LFP MAT output. |
| `state_ignore_manual` | `false` | Saved as `ignoreManual` in state-scoring metadata; the Python scoring pass does not implement a separate manual override based on this flag. |
| `state_sticky_trigger` | `false` | Saved as sticky-trigger metadata; it does not add a separate hysteresis pass to Python scoring. |

Disable `state_score` when only the ephys preprocessing/sorting outputs are
needed. State scoring needs a usable LFP and eligible channels; generating an
LFP alone does not evaluate recording quality or synchronization validity.

### Sorting and worker counts

| Key | Default | Meaning |
| --- | --- | --- |
| `run_sorter` | `true` | Run sorting after preprocessing, including in `--mode preprocess`. |
| `sorter` | `"Kilosort"` | Sorter selection: `Kilosort`, `Kilosort2.5`, or `Kilosort4`; `null`/`disabled` disables sorting. |
| `sorter_partition_mode` | `"all"` | Sort all active channels together, or separately by `probe` or `shank`. |
| `sorter_path` | `"sorter/KiloSort1"` | Installation for the selected sorter. Change this along with `sorter` when switching versions. |
| `sorter_config_path` | `"sorter/Kilosort1_config.yaml"` | Separate sorter YAML/JSON file, not a pipeline GUI JSON file. |
| `matlab_path` | `""` | MATLAB executable/bin directory for MATLAB-based sorters; empty uses discovery. |
| `preprocess_worker_count` | Automatic | Worker count for direct preprocessing and high-amplitude artifact detection. |
| `sorter_worker_count` | Automatic | MATLAB worker cap for direct sorting. Persistent Runs use the sorting Stage's `cpus`. |

Automatic workers use up to 128 CPUs. Below that capacity, eight CPUs are
reserved, with a minimum of one worker. Windows first caps usable capacity at
61 process workers. Direct JSON worker counts are clamped to this automatic
limit; persistent Stage resource requests are configured separately.

Sorter algorithm options live in
[Kilosort1_config.yaml](../sorter/Kilosort1_config.yaml),
[Kilosort2.5_config.yaml](../sorter/Kilosort2.5_config.yaml), and
[Kilosort4_config.yaml](../sorter/Kilosort4_config.yaml). The Kilosort4 file
explains detection thresholds, drift correction, geometry, whitening, clustering,
and runtime options inline. These options are version-specific; changing the
pipeline's bandpass/reference does not automatically rewrite the sorter config.

## Postprocessing

These keys belong inside `postprocess`. The recording's sample order, time origin,
and channel columns must match the sorting. XML and matching Phy `params.py`
metadata supply the binary rate, channel count, dtype, and byte offset; conflicting
XML/Phy rate or channel-count metadata is rejected.

| Key | Default | Meaning |
| --- | --- | --- |
| `sorting_phy_folder` | `""` | Explicit Kilosort/Phy sorting directory. Empty resolves sorting from session/search outputs. |
| `sorting_search_root` | `""` | Explicit root for discovering sorting outputs/partition manifests. |
| `dat_path` | `""` | Explicit recording binary matching the sorting. Empty uses Phy/session path resolution. |
| `apply_preprocess` | `false` | Apply the preprocessing filter/reference settings to a raw postprocess recording. Leave false for an already processed recording. |
| `exclude_cluster_groups` | `["noise"]` | Phy labels to exclude. Add `mua` if those clusters should also be excluded. |
| `duplicate_censored_period_ms` | `0.5` | Coincidence window used to identify duplicate spike trains, ms. |
| `duplicate_threshold` | `0.5` | Duplicate-overlap threshold used by duplicate-unit removal. |
| `merge_min_spikes` | `100` | Minimum spike count for automatic merge candidates. |
| `merge_corr_diff_thresh` | `0.25` | Correlogram-difference threshold for merging. |
| `merge_template_diff_thresh` | `0.25` | Waveform-template-difference threshold for merging. |
| `split_contamination` | `0.05` | Target outlier fraction for the PCA-distance split threshold. |
| `split_threshold_mode` | `"adaptive_chi2"` | Splitter accepts `adaptive_chi2` (scaled chi-square threshold) or `empirical` (distance quantile). |
| `split_wf_threshold` | `0.2` | Minimum cosine similarity for rescuing a candidate outlier against the clean waveform template. Higher is stricter. |
| `split_wf_n_chans` | `10` | Number of best channels used for the waveform rescue gate. |
| `split_amp_mad_scale` | `10.0` | Amplitude rescue bound: clean median ± this multiplier × MAD. |
| `skip_pc_metrics` | `true` | Skip PCA-based quality metrics. Automatic splitting still computes its required PCA features. |
| `noise_label_only` | `false` | Re-label existing quality metrics without rerunning full curation. `--mode noise_label` sets this to true. |
| `noise_thresholds` | See below | Rules for assigning the Phy `noise` label. |
| `overwrite` | `false` | Permit replacing postprocess outputs when true. |
| `worker_count` | Automatic | Direct postprocess workers; persistent Runs use postprocess Stage `cpus`. |
| `cell_explorer_sorting_folders` | `[]` | GUI selection for launching CellExplorer on particular sorting folders. The ephys CLI does not launch CellExplorer. |

The current GUI also lists `chi2` and `quantile` for split threshold mode, but
the splitter rejects those names. Use `adaptive_chi2` in the GUI; JSON can also
select `empirical`.

### Noise thresholds

These keys are nested inside `postprocess.noise_thresholds`.

| Key | Default | Noise condition |
| --- | --- | --- |
| `isi_violations_ratio_gt` | `5.0` | ISI violation ratio exceeds this value. |
| `isi_violations_count_gt` | `50.0` | ISI violation count exceeds this value. |
| `presence_ratio_lt` | `0.1` | Presence ratio is below this value. |
| `snr_lt` | `2.0` | Signal-to-noise ratio is below this value. |
| `amplitude_median_lt` | `15.0` | Absolute median amplitude is below this value, µV. |
| `amplitude_median_gt` | `500.0` | Absolute median amplitude exceeds this value, µV. |
| `firing_rate_lt` | `0.01` | Firing rate is at or below this value, Hz. |

When both ISI rules and both metrics are available, **both** ISI thresholds must
be exceeded. Other available rules are combined with OR. A rule with a missing
metric is skipped. Supplying a `noise_thresholds` object replaces the default
dictionary; it does not merge just the edited keys into it.

## Behavior

These keys belong inside `behavior`. They configure the GUI's tracking,
calibration, and export actions. Ephys CLI modes use the camera acquisition
settings during preprocessing, but do **not** run behavior export.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | GUI behavior workflow setting. It does not add a behavior Stage to an ephys Run. |
| `primary_point` | `""` | Tracking bodypart/keypoint name; when set, takes precedence over `primary_coords`. |
| `primary_coords` | `2` | Fallback 1-based point index, not the number of spatial dimensions. |
| `likelihood` | `0.0` | Minimum tracking confidence; low-confidence samples are treated as invalid. |
| `pulses_delta_range` | `0.01` | Allowed camera pulse-interval deviation, seconds. |
| `calibration_distance_cm` | `100.0` | Known physical distance between calibration endpoints, cm. |
| `calibration_pixel_distance` | `0.0` | Corresponding endpoint separation, pixels. GUI calibration also keeps per-epoch ratios. |
| `interpolate_gap_sec` | `0.0` | Maximum missing-position gap to interpolate, seconds; zero disables gap filling. |
| `fallback_video_fps` | `40.0` | Frame rate used when video metadata is unavailable. |
| `clean_tracker_jumps` | `true` | GUI tracking-jump cleanup setting. |
| `dlc_batch_path` | `""` | GUI batch directory for discovering tracking files. |
| `overwrite` | `false` | Replace existing behavior export when true. |
| `camera_sync_selection` | `"auto"` | Automatic sync candidate selection, or explicit `adc:N`/`ttl:N` with 1-based suffixes. |
| `camera_adc_channel` | `0` | ADC channel for acquisition/export handling, 1-based; zero becomes unspecified. When selecting `adc:N`, keep this equal to `N`. |

The exported file is `<basename>.animal.behavior.mat`, with position in cm,
speed in cm/s, and timestamps aligned using the recording's exported events.
Enable the corresponding `preprocess.analog_inputs` or `digital_inputs` export
before selecting that synchronization source.

## Persistent execution resources

These settings are used by GUI persistent Runs and the Python creation example
in [CLI guide](cli.md#persistent-local-and-slurm-runs). The direct
`run_pipeline --config ... --mode ...` command does not submit Slurm jobs or
apply `execution` resources.

| `execution` key | Default | Meaning |
| --- | --- | --- |
| `requested_backend` | `"auto"` | `local`, `slurm`, or `auto`. Auto selects usable Slurm, otherwise local. Explicit Slurm fails if required capabilities are unavailable. |
| `workspace` | `""` | Required writable persistent Run workspace. Records are created under `<workspace>/.pipeline/<run-id>`. |
| `matlab_path` | `""` | MATLAB path for persistent workers. |
| `shared_workspace_acknowledged` | `false` | Records the shared-workspace acknowledgment; it does not mount or synchronize paths. |
| `require_sacct` | `true` | Require reachable Slurm accounting for Slurm selection. |
| `preprocess` | Resource object | Preprocessing Stage allocation. |
| `sorting` | Resource object | Sorting Stage allocation. |
| `postprocess` | Resource object | Postprocessing Stage allocation. |

Each Stage resource object accepts these keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `cpus` | Automatic | Positive Stage CPU allocation and worker count; unlike direct JSON workers, resources are not clamped by `normalize_worker_count`. |
| `memory_mb` | `262144`; sorting `524288` | Positive memory request in MB (256/512 GiB defaults). Set an appropriate allocation for the server/job. |
| `walltime_minutes` | `null` | Positive requested time limit; null omits an explicit limit, so scheduler policy can still impose one. |
| `gpu_count` | `0`; sorting `1` | Pre/postprocessing require zero GPUs; sorting requires exactly one. |
| `gpu_gres_type` | `""` | Slurm GPU GRES type, such as the site's GPU model identifier. |
| `gpu_constraint` | `""` | Slurm node constraint for GPU placement. |
| `partition` | `""` | Slurm partition; empty leaves the scheduler default. |
| `account` | `""` | Slurm account. |
| `qos` | `""` | Slurm quality-of-service selection. |
| `reservation` | `""` | Slurm reservation. |

Persistent workers take their CPU count from the Stage allocation. Use
`execution.<stage>.cpus` to tune persistent Runs, rather than only editing the
direct-run worker fields. Slurm workers need access to the same repository,
environment, inputs, sorter installations, and output workspace.

## Lower-level Python configuration

For API-only controls, read the dataclass definitions:
[PreprocessConfig](../src/preprocess/metafile.py) and
[PostprocessConfig](../src/postprocess/metafile.py). For example, binary gain,
`zero_bad`, chunking, analyzer sparsity, cache handling, and additional merge/split
controls are not accepted as arbitrary keys in GUI JSON.

In particular, the Python preprocess defaults have `digital_inputs=false` and
`state_score=false`; Python postprocess defaults exclude both `noise` and `mua`
and have `overwrite=true`. The GUI JSON defaults documented above differ.
