from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PostprocessConfig:
    sorting_phy_folder: Path

    recording: Any | None = None
    dat_path: Path | None = None
    sampling_frequency: float | None = None
    num_channels: int | None = None
    dtype: str = "int16"
    gain_to_uV: float = 0.195
    offset_to_uV: float = 0.0
    chanmap_mat_path: Path | None = None
    reject_channels: list[int] = field(default_factory=list)

    apply_preprocessing_if_dat: bool = True
    preprocess_recording_object: bool = False
    bandpass_min_hz: float = 500.0
    bandpass_max_hz: float = 8000.0
    reference: str = "local"
    local_radius_um: tuple[float, float] = (50.0, 200.0)

    exclude_cluster_groups: list[str] = field(default_factory=lambda: ["noise", "mua"])
    duplicate_censored_period_ms: float = 0.5
    duplicate_threshold: float = 0.5
    remove_strategy: str = "max_spikes"

    analyzer_format: str = "binary_folder"
    analyzer_cache_dir: Path | None = None
    delete_analyzer_cache: bool = True
    skip_curation: bool = False
    random_spikes_method: str = "all"
    n_components: int = 5
    pc_mode: str = "by_channel_local"

    merge_min_spikes: int = 100
    merge_corr_diff_thresh: float = 0.25
    merge_template_diff_thresh: float = 0.25
    merge_sparsity_overlap: float = 0.5
    merge_censor_ms: float = 0.5

    split_contamination: float = 0.05
    split_threshold_mode: str = "adaptive_chi2"
    split_min_clean_frac: float = 0.9
    split_relax_factor: float = 0.5
    split_use_waveform_gate: bool = True
    split_wf_threshold: float = 0.2
    split_wf_template_max: int | None = 1000
    split_wf_n_chans: int = 10
    split_wf_center: str = "demean"
    split_squeeze_all_outlier_to_new: bool = True
    split_min_spikes: int = 10
    split_verbose: bool = True
    verbose: bool = True

    metric_names: list[str] = field(
        default_factory=lambda: ["isi_violation", "presence_ratio", "snr", "amplitude_median", "firing_rate"]
    )
    skip_pc_metrics: bool = True

    output_folder_name: str = "sorter_output_postprocessed"
    remove_if_exists: bool = True
    copy_binary: bool = False
    phy_hp_filtered: bool = True
    use_relative_path: bool = False
    metrics_csv_name: str = "quality_metrics.csv"
    write_preprocessed_dat_for_phy: bool = False
    force_params_dat_path: Path | None = None

    noise_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "isi_violations_ratio_gt": 5.0,
            "isi_violations_count_gt": 50.0,
            "presence_ratio_lt": 0.1,
            "snr_lt": 0.3,
            "amplitude_median_lt": 25.0,
            "firing_rate_lt": 0.01,
        }
    )
    noise_backup: bool = False

    job_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "n_jobs": 4,
            "progress_bar": True,
        }
    )


@dataclass
class PostprocessResult:
    sorting_phy_folder: Path
    output_folder: Path
    preprocessed_dat_path: Path | None
    metrics_csv_path: Path
    analyzer_cache_dir: Path | None
    n_units_initial: int
    n_units_final: int
    total_spikes_initial: int
    total_spikes_final: int
    n_noise_clusters: int
