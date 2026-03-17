from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np


def _default_parallel_n_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 128:
        return max(1, int(cpu_count) - 4)
    return 128


@dataclass
class PreprocessConfig:
    basepath: Path
    localpath: Path | None = None
    output_dir: Path | None = None  # backward compatibility
    sort_files: bool = True
    alt_sort: list[int] | None = None
    ignore_folders: list[str] = field(default_factory=list)

    dtype: str = "int16"
    gain_to_uV: float = 0.195
    offset_to_uV: float = 0.0
    save_raw: bool = False

    do_preprocess: bool = True
    bandpass_min_hz: float = 500.0
    bandpass_max_hz: float = 8000.0
    reference: str = "local"
    local_radius_um: tuple[float, float] = (20.0, 200.0)
    artifact_ttl_group_mode: Literal["none", "all", "probe", "shank"] = "none"
    artifact_TTL_channel: int | None = None
    artifact_TTL_ms_before: float = 1.0
    artifact_TTL_ms_after: float = 1.0
    artifact_TTL_mode: str = "cubic"
    artifact_TTL_include_offset: bool = False
    artifact_highamp_group_mode: Literal["none", "all", "probe", "shank"] = "none"
    highamp_estimate_windows: int = 500
    highamp_estimate_window_s: float = 1.0
    highamp_threshold_sigma: float = 10.0
    highamp_seed: int = 0
    highamp_chunk_s: float = 10.0
    highamp_dead_time_ms: float = 2.0
    highamp_n_jobs: int = field(default_factory=_default_parallel_n_jobs)
    highamp_ms_before: float = 1.0
    highamp_ms_after: float = 1.0
    highamp_mode: str = "cubic"

    make_lfp: bool = True
    lfp_fs: float = 1250.0
    session_basepath_mode: Literal["local", "source"] = "local"
    state_score: bool = False
    sw_channels: list[int] | None = None
    theta_channels: list[int] | None = None
    state_ignore_manual: bool = False
    state_save_lfp_mat: bool = True
    state_sticky_trigger: bool = False
    state_winparms: tuple[float, float] = (2.0, 15.0)
    emg_th_alpha: float = 1.0
    state_min_state_length: float = 6.0
    state_block_wake_to_rem: bool = False

    analog_inputs: bool = False
    analog_channels: list[int] | None = None
    digital_inputs: bool = False
    digital_channels: list[int] | None = None

    chanmap_mat_path: Path | None = None
    reject_channels: list[int] = field(default_factory=list)

    export_intermediate_dat: bool = True
    zero_bad: bool = True
    bad_channels: bool = True
    rhd_use_first_child_match: bool = False

    sorter: str | None = None
    sorter_path: Path | None = None
    sorter_config_path: Path | None = None
    matlab_path: Path | None = None
    matlab_max_workers: int = field(default_factory=_default_parallel_n_jobs)
    sorter_verbose: bool = False
    cleanup_temp_wh: bool = True

    save_params_json: bool = True
    save_manifest_json: bool = True
    save_log_mat: bool = True

    overwrite: bool = False
    job_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "n_jobs": _default_parallel_n_jobs(),
            "chunk_duration": "10s",
            "progress_bar": True,
        }
    )


@dataclass
class XmlMeta:
    sr: float
    sr_lfp: float | None
    n_channels: int
    skipped_channels_0based: list[int]


@dataclass
class SessionXmlMeta:
    date: str | None
    experimenters: str | None
    notes: str
    description: str
    anatomical_groups_0based: list[list[int]]
    spike_groups_0based: list[list[int]]
    skipped_channels_0based: list[int]


@dataclass
class MergePointsData:
    timestamps_sec: np.ndarray
    timestamps_samples: np.ndarray
    firstlasttimepoints_samples: np.ndarray
    foldernames: list[str]


@dataclass
class AcquisitionCatalog:
    source_type: Literal["intan", "openephys"]
    subsession_names: list[str]
    recording_paths: list[Path]
    recording_stream_names: list[str | None]
    ttl_event_paths: list[Path]
    amplifier_paths: list[Path]
    analogin_paths: list[Path]
    digitalin_paths: list[Path]
    auxiliary_paths: list[Path]
    supply_paths: list[Path]
    time_paths: list[Path]
    sample_counts: list[int]
    sampling_frequency: float | None

    amplifier_channels: int
    auxiliary_input_channels: int
    supply_voltage_channels: int
    board_adc_channels: int
    board_digital_input_channels: int
    board_digital_word_channels: int
    board_digital_output_channels: int
    temperature_sensor_channels: int
    board_adc_native_orders: list[int]
    board_digital_input_native_orders: list[int]


@dataclass
class PreprocessResult:
    basepath: Path
    basename: str
    local_output_dir: Path

    dat_path: Path | None
    lfp_path: Path | None
    session_mat_path: Path
    mergepoints_mat_path: Path

    analog_event_paths: list[Path]
    digital_event_paths: list[Path]
    intermediate_dat_paths: dict[str, Path]

    n_channels: int
    sr: float
    sr_lfp: float | None
    bad_channels_0based: list[int]
    bad_channels_1based: list[int]

    subsession_paths: list[Path]
    subsession_sample_counts: list[int]

    sorter: str | None = None
    sorter_output_dir: Path | None = None
    state_score_paths: list[Path] = field(default_factory=list)
    state_score_figure_paths: list[Path] = field(default_factory=list)
