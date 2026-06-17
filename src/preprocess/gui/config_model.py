from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Literal

from src.postprocess import PostprocessConfig
from src.preprocess import PreprocessConfig
from src.preprocess.paths import find_project_root, resolve_project_path


RunMode = Literal["all", "preprocess", "postprocess", "noise_label"]

NOTEBOOK_NOISE_THRESHOLDS = {
    "isi_violations_ratio_gt": 5.0,
    "isi_violations_count_gt": 50.0,
    "presence_ratio_lt": 0.1,
    "snr_lt": 2.0,
    "amplitude_median_lt": 15.0,
    "amplitude_median_gt": 500.0,
    "firing_rate_lt": 0.01,
}

SORTING_OUTPUT_PATTERNS = (
    "Kilosort_*",
    "Kilosort2_5_*",
    "Kilosort2.5_*",
    "Kilosort4_*",
)
REPO_ROOT = find_project_root()
DEFAULT_LOCAL_WORKING_DIR = REPO_ROOT / "preprocess_tmp"


def default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return 128 if cpu_count >= 128 else max(1, cpu_count - 8)


def default_local_working_dir(*, create: bool = True) -> Path:
    path = DEFAULT_LOCAL_WORKING_DIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> list[int]:
    stripped = text.strip()
    if not stripped:
        return []
    values: list[int] = []
    for part in stripped.replace("\n", ",").split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_float_pair(text: str, *, default: tuple[float, float]) -> tuple[float, float]:
    stripped = text.strip()
    if not stripped:
        return default
    parts = [p.strip() for p in stripped.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected exactly two comma-separated numeric values.")
    return (float(parts[0]), float(parts[1]))


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text).expanduser() if text else None


def _repo_path_or_none(value: str | Path | None) -> Path | None:
    path = _path_or_none(value)
    if path is None:
        return None
    if not path.is_absolute():
        path = resolve_project_path(path, root=REPO_ROOT)
    return path.resolve()


def latest_sorting_folder(root: Path | None) -> Path | None:
    if root is None or not root.exists() or not root.is_dir():
        return None
    candidates = [
        p.resolve()
        for pattern in SORTING_OUTPUT_PATTERNS
        for p in root.glob(pattern)
        if p.is_dir() and not p.name.endswith("_spi")
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def postprocess_output_folder_for_sorting(sorting_folder: Path) -> Path:
    run_root = sorting_folder.parent if sorting_folder.name == "sorter_output" else sorting_folder
    return (run_root.parent / f"{run_root.name}_spi").resolve()


@dataclass
class PreprocessGuiSettings:
    analog_inputs: bool = False
    digital_inputs: bool = True
    save_raw: bool = False
    do_preprocess: bool = True
    bandpass_min_hz: float = 500.0
    bandpass_max_hz: float = 8000.0
    reference: str = "local"
    local_radius_um: tuple[float, float] = (20.0, 200.0)
    make_lfp: bool = True
    lfp_fs: float = 1250.0
    state_score: bool = True
    sw_channels: list[int] = field(default_factory=list)
    theta_channels: list[int] = field(default_factory=list)
    state_ignore_manual: bool = False
    state_save_lfp_mat: bool = True
    state_sticky_trigger: bool = False
    state_winparms: tuple[float, float] = (2.0, 15.0)
    emg_th_alpha: float = 1.0
    useEMG_NREM: bool = True
    state_min_state_length: float = 6.0
    state_microarousal_sec: float = 100.0
    state_block_wake_to_rem: bool = True
    remove_ttl_artifacts: bool = False
    artifact_ttl_group_mode: str = "none"
    artifact_ttl_channel: int = 0
    artifact_ttl_include_offset: bool = False
    artifact_ttl_ms_before: float = 0.5
    artifact_ttl_ms_after: float = 2.0
    artifact_ttl_mode: str = "linear"
    remove_highamp_artifacts: bool = False
    artifact_highamp_group_mode: str = "shank"
    highamp_threshold_sigma: float = 10.0
    highamp_ms_before: float = 2.0
    highamp_ms_after: float = 2.0
    highamp_mode: str = "linear"
    reject_channels: list[int] = field(default_factory=list)
    probe_assignments: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "type": "staggered",
                "groups": [0, 1, 2, 3, 4, 5, 6, 7],
                "x_offset": 0,
            }
        ]
    )
    run_sorter: bool = True
    sorter: str | None = "Kilosort"
    sorter_path: str = str(Path("sorter") / "KiloSort1")
    sorter_config_path: str = str(Path("sorter") / "Kilosort1_config.yaml")
    matlab_path: str = ""
    preprocess_worker_count: int = field(default_factory=default_worker_count)
    sorter_worker_count: int = field(default_factory=default_worker_count)
    overwrite: bool = False


@dataclass
class PostprocessGuiSettings:
    sorting_phy_folder: str = ""
    sorting_search_root: str = ""
    apply_preprocess: bool = False
    exclude_cluster_groups: list[str] = field(default_factory=lambda: ["noise"])
    duplicate_censored_period_ms: float = 0.5
    duplicate_threshold: float = 0.5
    merge_min_spikes: int = 100
    merge_corr_diff_thresh: float = 0.25
    merge_template_diff_thresh: float = 0.25
    split_contamination: float = 0.05
    split_threshold_mode: str = "adaptive_chi2"
    split_wf_threshold: float = 0.2
    split_wf_n_chans: int = 10
    split_amp_mad_scale: float = 10.0
    skip_pc_metrics: bool = True
    noise_label_only: bool = False
    noise_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(NOTEBOOK_NOISE_THRESHOLDS)
    )
    overwrite: bool = False
    worker_count: int = field(default_factory=default_worker_count)


@dataclass
class PipelineGuiSettings:
    basepath: str = ""
    local_root: str = ""
    chanmap_path: str = ""
    preprocess: PreprocessGuiSettings = field(default_factory=PreprocessGuiSettings)
    postprocess: PostprocessGuiSettings = field(default_factory=PostprocessGuiSettings)

    @property
    def basepath_path(self) -> Path | None:
        return _path_or_none(self.basepath)

    @property
    def local_root_path(self) -> Path:
        return _path_or_none(self.local_root) or default_local_working_dir()

    @property
    def basename(self) -> str:
        basepath = self.basepath_path
        return basepath.name if basepath is not None else ""

    @property
    def local_output_dir(self) -> Path | None:
        if not self.basename:
            return None
        return (self.local_root_path / self.basename).resolve()

    def resolved_chanmap_path(self) -> Path | None:
        explicit = _path_or_none(self.chanmap_path)
        if explicit is not None:
            return explicit
        output_dir = self.local_output_dir
        basepath = self.basepath_path
        if output_dir is None:
            return (basepath / "chanMap.mat") if basepath is not None else None

        local_chanmap = output_dir / "chanMap.mat"
        if local_chanmap.exists():
            return local_chanmap

        if basepath is not None:
            basepath_chanmap = basepath / "chanMap.mat"
            if basepath_chanmap.exists():
                return basepath_chanmap

        return local_chanmap

    def to_preprocess_config(self) -> PreprocessConfig:
        basepath = self.basepath_path
        if basepath is None:
            raise ValueError("basepath is required.")

        p = self.preprocess
        sorter = p.sorter if p.run_sorter and p.sorter and p.sorter.lower() != "disabled" else None
        ttl_group_mode = p.artifact_ttl_group_mode if p.remove_ttl_artifacts else "none"
        highamp_group_mode = p.artifact_highamp_group_mode if p.remove_highamp_artifacts else "none"
        return PreprocessConfig(
            basepath=basepath,
            localpath=self.local_root_path,
            save_raw=p.save_raw,
            analog_inputs=p.analog_inputs,
            digital_inputs=p.digital_inputs,
            do_preprocess=p.do_preprocess,
            bandpass_min_hz=p.bandpass_min_hz,
            bandpass_max_hz=p.bandpass_max_hz,
            reference=p.reference,
            local_radius_um=p.local_radius_um,
            artifact_ttl_group_mode=ttl_group_mode,  # type: ignore[arg-type]
            artifact_TTL_channel=p.artifact_ttl_channel,
            artifact_TTL_include_offset=p.artifact_ttl_include_offset,
            artifact_TTL_ms_before=p.artifact_ttl_ms_before,
            artifact_TTL_ms_after=p.artifact_ttl_ms_after,
            artifact_TTL_mode=p.artifact_ttl_mode,
            artifact_highamp_group_mode=highamp_group_mode,  # type: ignore[arg-type]
            highamp_threshold_sigma=p.highamp_threshold_sigma,
            highamp_estimate_windows=500,
            highamp_estimate_window_s=1.0,
            highamp_seed=0,
            highamp_chunk_s=1.0,
            highamp_dead_time_ms=1.0,
            highamp_ms_before=p.highamp_ms_before,
            highamp_ms_after=p.highamp_ms_after,
            highamp_mode=p.highamp_mode,
            highamp_n_jobs=p.preprocess_worker_count,
            make_lfp=p.make_lfp,
            lfp_fs=p.lfp_fs,
            state_score=p.state_score,
            sw_channels=list(p.sw_channels) if p.sw_channels else None,
            theta_channels=list(p.theta_channels) if p.theta_channels else None,
            state_ignore_manual=p.state_ignore_manual,
            state_save_lfp_mat=p.state_save_lfp_mat,
            state_sticky_trigger=p.state_sticky_trigger,
            state_winparms=tuple(p.state_winparms),
            state_block_wake_to_rem=p.state_block_wake_to_rem,
            state_min_state_length=p.state_min_state_length,
            state_microarousal_sec=p.state_microarousal_sec,
            emg_th_alpha=p.emg_th_alpha,
            useEMG_NREM=p.useEMG_NREM,
            chanmap_mat_path=self.resolved_chanmap_path(),
            reject_channels=list(p.reject_channels),
            matlab_path=_path_or_none(p.matlab_path),
            matlab_max_workers=p.sorter_worker_count,
            sorter=sorter,
            sorter_path=_repo_path_or_none(p.sorter_path) if sorter else None,
            sorter_config_path=_repo_path_or_none(p.sorter_config_path) if sorter else None,
            overwrite=p.overwrite,
            job_kwargs={
                "pool_engine": "process",
                "n_jobs": p.preprocess_worker_count,
                "chunk_duration": "1s",
                "progress_bar": True,
                "max_threads_per_worker": 1,
            },
        )

    def to_postprocess_config(self) -> PostprocessConfig:
        pp = self.postprocess
        output_dir = self.local_output_dir
        basename = self.basename
        dat_path = output_dir / f"{basename}.dat" if output_dir is not None and basename else None
        chanmap_path = self.resolved_chanmap_path()

        sorting_phy_folder = _path_or_none(pp.sorting_phy_folder)
        if sorting_phy_folder is None:
            sorting_phy_folder = latest_sorting_folder(output_dir)

        return PostprocessConfig(
            sorting_phy_folder=sorting_phy_folder,
            sorting_search_root=_path_or_none(pp.sorting_search_root),
            dat_path=dat_path if dat_path is not None and dat_path.exists() else None,
            chanmap_mat_path=chanmap_path if chanmap_path is not None and chanmap_path.exists() else None,
            reject_channels=list(self.preprocess.reject_channels),
            apply_preprocess=pp.apply_preprocess,
            bandpass_min_hz=self.preprocess.bandpass_min_hz,
            bandpass_max_hz=self.preprocess.bandpass_max_hz,
            reference=self.preprocess.reference,
            local_radius_um=self.preprocess.local_radius_um,
            exclude_cluster_groups=list(pp.exclude_cluster_groups),
            duplicate_censored_period_ms=pp.duplicate_censored_period_ms,
            duplicate_threshold=pp.duplicate_threshold,
            merge_min_spikes=pp.merge_min_spikes,
            merge_corr_diff_thresh=pp.merge_corr_diff_thresh,
            merge_template_diff_thresh=pp.merge_template_diff_thresh,
            split_contamination=pp.split_contamination,
            split_threshold_mode=pp.split_threshold_mode,
            split_wf_threshold=pp.split_wf_threshold,
            split_wf_n_chans=pp.split_wf_n_chans,
            split_amp_mad_scale=pp.split_amp_mad_scale,
            skip_pc_metrics=pp.skip_pc_metrics,
            noise_label_only=pp.noise_label_only,
            noise_thresholds=dict(pp.noise_thresholds),
            overwrite=pp.overwrite,
            job_kwargs={"n_jobs": pp.worker_count, "progress_bar": True},
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PipelineGuiSettings":
        data = json.loads(text)
        preprocess_data = data.pop("preprocess", {})
        if "worker_count" in preprocess_data:
            legacy_worker_count = preprocess_data.pop("worker_count")
            preprocess_data.setdefault("preprocess_worker_count", legacy_worker_count)
            preprocess_data.setdefault("sorter_worker_count", legacy_worker_count)
        preprocess = PreprocessGuiSettings(**preprocess_data)
        postprocess = PostprocessGuiSettings(**data.pop("postprocess", {}))
        return cls(**data, preprocess=preprocess, postprocess=postprocess)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineGuiSettings":
        return cls.from_json(path.read_text(encoding="utf-8"))
