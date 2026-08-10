from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
from typing import Any, Literal

from src.postprocess import PostprocessConfig
from src.preprocess import PreprocessConfig
from src.preprocess.io import load_xml_metadata
from src.preprocess.paths import find_project_root, resolve_project_path
from src.worker_defaults import default_worker_count, normalize_worker_count
from src.execution.models import BackendName, ExecutionConfig, RequestedBackend, ResourceSpec


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


def default_local_working_dir(*, create: bool = True) -> Path:
    path = DEFAULT_LOCAL_WORKING_DIR.resolve()
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
        if p.is_dir() and "_spi" not in p.name and ".preserved-" not in p.name
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def latest_sorting_folder_from_roots(roots: list[Path | None]) -> Path | None:
    for root in roots:
        candidate = latest_sorting_folder(root)
        if candidate is not None:
            return candidate
    return None


def postprocess_output_folder_for_sorting(sorting_folder: Path) -> Path:
    run_root = sorting_folder.parent if sorting_folder.name == "sorter_output" else sorting_folder
    return (run_root.parent / f"{run_root.name}_spi").resolve()


def _session_xml_path(basepath: Path | None, basename: str) -> Path | None:
    if basepath is None or not basename:
        return None
    xml_path = basepath / f"{basename}.xml"
    return xml_path if xml_path.exists() else None


def _postprocess_xml_metadata(basepath: Path | None, basename: str) -> tuple[float | None, int | None]:
    xml_path = _session_xml_path(basepath, basename)
    if xml_path is None:
        return None, None
    meta = load_xml_metadata(xml_path)
    return float(meta.sr), int(meta.n_channels)


def _xml_metadata_from_path(xml_path: Path | None) -> tuple[float | None, int | None]:
    if xml_path is None or not xml_path.exists():
        return None, None
    meta = load_xml_metadata(xml_path)
    return float(meta.sr), int(meta.n_channels)


def _looks_like_processed_session(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    basename = path.name
    primary = path / f"{basename}.dat"
    supporting = (
        path / f"{basename}.session.mat",
        path / "preprocess_run.yaml",
        path / "preprocessSession_manifest.json",
        path / "sorter_partition_manifest.json",
    )
    has_sorting = any(
        item.is_dir()
        for pattern in SORTING_OUTPUT_PATTERNS
        for item in path.glob(pattern)
    )
    return primary.exists() and (any(item.exists() for item in supporting) or has_sorting)


def _raw_recording_is_present(path: Path) -> bool:
    if (path / "structure.oebin").exists():
        return True
    if (path / "amplifier.dat").exists() or (path / "continuous.dat").exists():
        return True
    return any(
        child.is_dir()
        and (
            (child / "amplifier.dat").exists()
            or (child / "continuous.dat").exists()
            or (child / "structure.oebin").exists()
        )
        for child in path.iterdir()
    )


def _source_from_existing_session(path: Path) -> Path | None:
    final_record = path / "preprocess_run.yaml"
    if final_record.exists():
        try:
            import yaml

            payload = yaml.safe_load(final_record.read_text(encoding="utf-8")) or {}
            candidates = (
                payload.get("source_basepath"),
                (payload.get("inputs") or {}).get("source_basepath"),
                (payload.get("analysis") or {}).get("source_basepath"),
            )
            for value in candidates:
                if value and Path(str(value)).expanduser().is_dir():
                    return Path(str(value)).expanduser().resolve()
        except (OSError, TypeError, ValueError):
            pass

    active_record = path / ".pipeline-active-run.json"
    if active_record.exists():
        try:
            claim = json.loads(active_record.read_text(encoding="utf-8"))
            run_dir = Path(str(claim.get("run_dir") or ""))
            analysis_path = run_dir / "analysis_config.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            saved = analysis.get("settings") or {}
            value = str(saved.get("source_basepath") or saved.get("basepath") or "").strip()
            if value and Path(value).expanduser().is_dir():
                return Path(value).expanduser().resolve()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    legacy_manifest = path / "preprocessSession_manifest.json"
    if legacy_manifest.exists():
        try:
            payload = json.loads(legacy_manifest.read_text(encoding="utf-8"))
            value = str(payload.get("basepath") or "").strip()
            if value and Path(value).expanduser().is_dir():
                return Path(value).expanduser().resolve()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    legacy_params = path / "preprocessSession_params.json"
    if legacy_params.exists():
        try:
            payload = json.loads(legacy_params.read_text(encoding="utf-8"))
            value = str(payload.get("basepath") or "").strip()
            if value and Path(value).expanduser().is_dir():
                return Path(value).expanduser().resolve()
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if _raw_recording_is_present(path):
        return path.resolve()
    return None


def resolve_existing_session_settings(settings: "PipelineGuiSettings") -> bool:
    """Resolve a processed folder selected through the existing Basepath field."""

    selected = settings.basepath_path
    if settings.multi_day_enabled or selected is None or not _looks_like_processed_session(selected):
        settings.existing_session_dir = ""
        settings.source_basepath = ""
        return False
    settings.existing_session_dir = str(selected.resolve())
    source = _source_from_existing_session(selected)
    settings.source_basepath = str(source) if source is not None else ""
    return True


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
    sorter_partition_mode: Literal["all", "probe", "shank"] = "all"
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
    cell_explorer_sorting_folders: list[str] = field(default_factory=list)
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
class BehaviorGuiSettings:
    enabled: bool = False
    primary_coords: int = 2
    primary_point: str = ""
    likelihood: float = 0.6
    pulses_delta_range: float = 0.01
    calibration_distance_cm: float = 100.0
    calibration_pixel_distance: float = 0.0
    interpolate_gap_sec: float = 1.0
    fallback_video_fps: float = 40.0
    clean_tracker_jumps: bool = True
    dlc_batch_path: str = ""
    overwrite: bool = False


@dataclass
class StageResourceGuiSettings:
    cpus: int = field(default_factory=default_worker_count)
    memory_mb: int = 256 * 1024
    walltime_minutes: int | None = None
    gpu_count: int = 0
    gpu_gres_type: str = ""
    gpu_constraint: str = ""
    partition: str = ""
    account: str = ""
    qos: str = ""
    reservation: str = ""

    def to_resource_spec(self) -> ResourceSpec:
        spec = ResourceSpec(
            cpus=int(self.cpus),
            memory_mb=int(self.memory_mb),
            walltime_minutes=(
                None
                if self.walltime_minutes is None
                else int(self.walltime_minutes)
            ),
            gpu_count=int(self.gpu_count),
            gpu_gres_type=self.gpu_gres_type.strip(),
            gpu_constraint=self.gpu_constraint.strip(),
            partition=self.partition.strip(),
            account=self.account.strip(),
            qos=self.qos.strip(),
            reservation=self.reservation.strip(),
        )
        return spec


def _sorting_resource_defaults() -> StageResourceGuiSettings:
    return StageResourceGuiSettings(memory_mb=512 * 1024, gpu_count=1)


@dataclass
class ExecutionGuiSettings:
    requested_backend: str = RequestedBackend.AUTO.value
    workspace: str = ""
    matlab_path: str = ""
    shared_workspace_acknowledged: bool = False
    require_sacct: bool = True
    preprocess: StageResourceGuiSettings = field(default_factory=StageResourceGuiSettings)
    sorting: StageResourceGuiSettings = field(default_factory=_sorting_resource_defaults)
    postprocess: StageResourceGuiSettings = field(default_factory=StageResourceGuiSettings)

    def to_execution_config(self, *, resolved_backend: BackendName) -> ExecutionConfig:
        workspace = self.workspace.strip()
        if not workspace:
            raise ValueError("A persistent Run workspace is required")
        config = ExecutionConfig(
            requested_backend=RequestedBackend(self.requested_backend.lower()),
            resolved_backend=resolved_backend,
            workspace=str(Path(workspace).expanduser().resolve()),
            matlab_path=self.matlab_path.strip(),
            shared_workspace_acknowledged=bool(self.shared_workspace_acknowledged),
            require_sacct=bool(self.require_sacct),
            resources={
                "preprocess": self.preprocess.to_resource_spec(),
                "sorting": self.sorting.to_resource_spec(),
                "postprocess": self.postprocess.to_resource_spec(),
            },
        )
        config.validate()
        return config


@dataclass
class PipelineGuiSettings:
    basepath: str = ""
    local_root: str = ""
    # Runtime-resolved paths for an existing processed session selected in the
    # unchanged Basepath field. They are persisted in a Run snapshot but are
    # not additional GUI controls.
    source_basepath: str = ""
    existing_session_dir: str = ""
    xml_path: str = ""
    chanmap_path: str = ""
    multi_day_enabled: bool = False
    multi_day_session_paths: list[str] = field(default_factory=list)
    multi_day_selected_subepoch_paths: list[str] = field(default_factory=list)
    multi_day_name: str = ""
    preprocess: PreprocessGuiSettings = field(default_factory=PreprocessGuiSettings)
    behavior: BehaviorGuiSettings = field(default_factory=BehaviorGuiSettings)
    postprocess: PostprocessGuiSettings = field(default_factory=PostprocessGuiSettings)
    execution: ExecutionGuiSettings = field(default_factory=ExecutionGuiSettings)

    @property
    def basepath_path(self) -> Path | None:
        return _path_or_none(self.basepath)

    @property
    def preprocess_source_path(self) -> Path | None:
        source = _path_or_none(self.source_basepath)
        if source is not None:
            return source
        if self.existing_session_dir:
            return None
        return self.basepath_path

    @property
    def local_root_path(self) -> Path:
        return _path_or_none(self.local_root) or default_local_working_dir()

    @property
    def basename(self) -> str:
        if self.multi_day_enabled and self.multi_day_name.strip():
            return self.multi_day_name.strip()
        basepath = self.basepath_path
        return basepath.name if basepath is not None else ""

    @property
    def local_output_dir(self) -> Path | None:
        existing = _path_or_none(self.existing_session_dir)
        if existing is not None:
            return existing.resolve()
        if not self.basename:
            return None
        return (self.local_root_path / self.basename).resolve()

    def postprocess_dat_path(self) -> Path | None:
        basename = self.basename
        if not basename:
            return None
        candidates: list[Path] = []
        output_dir = self.local_output_dir
        if output_dir is not None:
            candidates.append(output_dir / f"{basename}.dat")
        basepath = self.basepath_path
        if basepath is not None:
            candidates.append(basepath / f"{basename}.dat")
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0] if candidates else None

    def postprocess_sorting_folder(self) -> Path | None:
        explicit = _path_or_none(self.postprocess.sorting_phy_folder)
        if explicit is not None:
            return explicit
        return latest_sorting_folder_from_roots([self.local_output_dir, self.basepath_path])

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

    def resolved_xml_path(self) -> Path | None:
        explicit = _path_or_none(self.xml_path)
        if explicit is not None:
            return explicit
        basename = self.basename
        if not basename:
            return None
        for root in (self.local_output_dir, self.preprocess_source_path, self.basepath_path):
            if root is None:
                continue
            candidate = root / f"{basename}.xml"
            if candidate.exists():
                return candidate
        return None

    def to_preprocess_config(self) -> PreprocessConfig:
        basepath = self.preprocess_source_path
        if basepath is None:
            raise ValueError("basepath is required.")

        p = self.preprocess
        sorter = p.sorter if p.run_sorter and p.sorter and p.sorter.lower() != "disabled" else None
        ttl_group_mode = p.artifact_ttl_group_mode if p.remove_ttl_artifacts else "none"
        highamp_group_mode = p.artifact_highamp_group_mode if p.remove_highamp_artifacts else "none"
        return PreprocessConfig(
            basepath=basepath,
            localpath=(
                self.local_output_dir.parent
                if self.existing_session_dir and self.local_output_dir is not None
                else self.local_root_path
            ),
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
            highamp_n_jobs=normalize_worker_count(p.preprocess_worker_count),
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
            xml_path=self.resolved_xml_path(),
            reject_channels=list(p.reject_channels),
            matlab_path=_path_or_none(p.matlab_path),
            matlab_max_workers=normalize_worker_count(p.sorter_worker_count),
            sorter=sorter,
            sorter_path=_repo_path_or_none(p.sorter_path) if sorter else None,
            sorter_config_path=_repo_path_or_none(p.sorter_config_path) if sorter else None,
            sorter_partition_mode=p.sorter_partition_mode,  # type: ignore[arg-type]
            overwrite=p.overwrite,
            job_kwargs={
                "pool_engine": "process",
                "n_jobs": normalize_worker_count(p.preprocess_worker_count),
                "chunk_duration": "1s",
                "progress_bar": True,
                "max_threads_per_worker": 1,
            },
        )

    def to_postprocess_config(self) -> PostprocessConfig:
        pp = self.postprocess
        basename = self.basename
        dat_path = self.postprocess_dat_path()
        chanmap_path = self.resolved_chanmap_path()
        sampling_frequency, num_channels = _xml_metadata_from_path(self.resolved_xml_path())
        sorting_phy_folder = self.postprocess_sorting_folder()
        sorting_search_root = _path_or_none(pp.sorting_search_root)
        local_output_dir = self.local_output_dir
        if (
            sorting_search_root is None
            and local_output_dir is not None
            and (local_output_dir / "sorter_partition_manifest.json").exists()
        ):
            sorting_search_root = local_output_dir

        return PostprocessConfig(
            sorting_phy_folder=sorting_phy_folder,
            sorting_search_root=sorting_search_root,
            dat_path=dat_path if dat_path is not None and dat_path.exists() else None,
            sampling_frequency=sampling_frequency,
            num_channels=num_channels,
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
            job_kwargs={"n_jobs": normalize_worker_count(pp.worker_count), "progress_bar": True},
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
        requested_preprocess_cpus = int(
            preprocess_data.get("preprocess_worker_count", default_worker_count())
        )
        requested_sorting_cpus = int(
            preprocess_data.get("sorter_worker_count", default_worker_count())
        )
        preprocess = PreprocessGuiSettings(**preprocess_data)
        preprocess.preprocess_worker_count = normalize_worker_count(preprocess.preprocess_worker_count)
        preprocess.sorter_worker_count = normalize_worker_count(preprocess.sorter_worker_count)
        behavior_data = data.pop("behavior", {})
        behavior_fields = {item.name for item in fields(BehaviorGuiSettings)}
        behavior = BehaviorGuiSettings(
            **{key: value for key, value in behavior_data.items() if key in behavior_fields}
        )
        postprocess_data = data.pop("postprocess", {})
        requested_postprocess_cpus = int(postprocess_data.get("worker_count", default_worker_count()))
        postprocess = PostprocessGuiSettings(**postprocess_data)
        postprocess.worker_count = normalize_worker_count(postprocess.worker_count)
        has_execution = "execution" in data
        execution_data = data.pop("execution", {})
        execution = ExecutionGuiSettings(
            requested_backend=str(execution_data.pop("requested_backend", RequestedBackend.AUTO.value)),
            workspace=str(execution_data.pop("workspace", "")),
            matlab_path=str(execution_data.pop("matlab_path", preprocess.matlab_path)),
            shared_workspace_acknowledged=bool(
                execution_data.pop("shared_workspace_acknowledged", False)
            ),
            require_sacct=bool(execution_data.pop("require_sacct", True)),
            preprocess=StageResourceGuiSettings(**execution_data.pop("preprocess", {})),
            sorting=StageResourceGuiSettings(**execution_data.pop("sorting", {"memory_mb": 512 * 1024, "gpu_count": 1})),
            postprocess=StageResourceGuiSettings(**execution_data.pop("postprocess", {})),
        )
        if not has_execution:
            execution.preprocess.cpus = max(1, requested_preprocess_cpus)
            execution.sorting.cpus = max(1, requested_sorting_cpus)
            execution.postprocess.cpus = max(1, requested_postprocess_cpus)
        return cls(
            **data,
            preprocess=preprocess,
            behavior=behavior,
            postprocess=postprocess,
            execution=execution,
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineGuiSettings":
        return cls.from_json(path.read_text(encoding="utf-8"))
