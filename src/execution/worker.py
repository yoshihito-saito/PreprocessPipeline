from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback
from typing import Any

from .models import AttemptSpec, StageName, StageStatus
from .store import RunStore, atomic_write_json, atomic_write_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _limit_thread_oversubscription() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _settings_for_run(store: RunStore):
    from src.preprocess.gui.config_model import PipelineGuiSettings

    analysis = store.load_analysis()
    settings = PipelineGuiSettings.from_json(json.dumps(analysis.settings))
    settings.preprocess.matlab_path = store.load_execution().matlab_path
    return settings


def _verify_analysis_artifacts(store: RunStore, stage: StageName) -> None:
    analysis = store.load_analysis()
    artifacts = analysis.artifact_sha256
    settings = _settings_for_run(store)

    def _verify_file(name: str, path: Path | None) -> None:
        expected = artifacts.get(name)
        if not expected:
            return
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Recorded {name} input is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Recorded {name} checksum changed after Run creation: {path}"
            )

    if stage in (StageName.PREPROCESS, StageName.POSTPROCESS):
        _verify_file("xml", settings.resolved_xml_path())
    if stage in (StageName.PREPROCESS, StageName.SORTING):
        preprocess = analysis.settings.get("preprocess", {})
        path_text = str(preprocess.get("sorter_config_path", "")).strip()
        _verify_file("sorter_config", Path(path_text) if path_text else None)
    if stage == StageName.POSTPROCESS:
        _verify_file("chanmap", settings.resolved_chanmap_path())
        from .controller import _file_metadata_identity, _sorting_input_identity

        expected_sorting = artifacts.get("postprocess_sorting_input")
        if expected_sorting and _sorting_input_identity(
            settings.postprocess_sorting_folder()
        ) != expected_sorting:
            raise ValueError("Postprocess sorting input changed after Run creation")
        expected_dat = artifacts.get("postprocess_dat_input")
        if expected_dat and _file_metadata_identity(settings.postprocess_dat_path()) != expected_dat:
            raise ValueError("Postprocess dat input changed after Run creation")


def _serialize_dataclass(value: Any) -> dict[str, Any]:
    return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}


def _preprocess_result_from_dict(value: dict[str, Any]):
    from src.preprocess import PreprocessResult

    path_fields = {
        "basepath",
        "local_output_dir",
        "dat_path",
        "lfp_path",
        "session_mat_path",
        "mergepoints_mat_path",
        "sorter_output_dir",
        "sorter_partition_manifest_path",
    }
    list_path_fields = {
        "analog_event_paths",
        "digital_event_paths",
        "subsession_paths",
        "sorter_output_dirs",
        "state_score_paths",
        "state_score_figure_paths",
    }
    data = dict(value)
    for name in path_fields:
        if data.get(name):
            data[name] = Path(data[name])
        elif name in {"dat_path", "lfp_path", "sorter_output_dir", "sorter_partition_manifest_path"}:
            data[name] = None
    for name in list_path_fields:
        data[name] = [Path(item) for item in data.get(name, [])]
    data["intermediate_dat_paths"] = {
        str(name): Path(path) for name, path in data.get("intermediate_dat_paths", {}).items()
    }
    allowed = {item.name for item in fields(PreprocessResult)}
    return PreprocessResult(**{name: item for name, item in data.items() if name in allowed})


def _load_upstream_result(store: RunStore, spec: AttemptSpec, stage: StageName) -> dict[str, Any]:
    attempt = spec.upstream_attempts.get(stage.value)
    if attempt is None:
        raise ValueError(f"{spec.stage.value} Attempt does not name a {stage.value} upstream Attempt")
    result = store.read_attempt_fact(stage, attempt, "result.json")
    if (
        result is None
        or result.get("status") != StageStatus.COMPLETED.value
        or result.get("analysis_sha256") != spec.analysis_sha256
        or not bool((result.get("validation") or {}).get("passed", False))
    ):
        raise RuntimeError(
            f"Required {stage.value} attempt-{attempt:03d} has no validated completed StageResult"
        )
    return result


def _validate_paths(paths: list[Path], *, label: str) -> list[str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{label} validation failed; missing: {', '.join(missing)}")
    return [str(path.resolve()) for path in paths]


def _validate_sorting_output(output_dir: Path) -> list[str]:
    import numpy as np

    required = [
        output_dir / "params.py",
        output_dir / "spike_times.npy",
        output_dir / "spike_clusters.npy",
        output_dir / "templates.npy",
    ]
    validated = _validate_paths(required, label=f"sorting output {output_dir.name}")
    if (output_dir / "params.py").stat().st_size == 0:
        raise RuntimeError(f"sorting output {output_dir.name} has an empty params.py")
    try:
        spike_times = np.load(output_dir / "spike_times.npy", mmap_mode="r", allow_pickle=False)
        spike_clusters = np.load(
            output_dir / "spike_clusters.npy", mmap_mode="r", allow_pickle=False
        )
        templates = np.load(output_dir / "templates.npy", mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"sorting output {output_dir.name} contains an invalid NumPy file") from exc
    if spike_times.reshape(-1).shape[0] != spike_clusters.reshape(-1).shape[0]:
        raise RuntimeError(
            f"sorting output {output_dir.name} has inconsistent spike_times/spike_clusters lengths"
        )
    if templates.ndim < 2:
        raise RuntimeError(f"sorting output {output_dir.name} has an invalid templates array")
    return validated


def _validate_sorter_manifest(path: Path, output_dirs: list[Path]) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise RuntimeError("sorter partition manifest has no partitions")
    completed_dirs = {
        str(Path(item.get("output_folder", "")).resolve())
        for item in partitions
        if isinstance(item, dict)
        and item.get("status") == StageStatus.COMPLETED.value
        and item.get("output_folder")
    }
    expected_dirs = {str(path.resolve()) for path in output_dirs}
    if completed_dirs != expected_dirs:
        raise RuntimeError(
            "sorter partition manifest does not match all completed sorting outputs"
        )


def _run_preprocess(store: RunStore, spec: AttemptSpec) -> dict[str, Any]:
    from src.preprocess import run_preprocess_session
    from src.preprocess.runtime_prep import prepare_preprocess_settings

    settings = _settings_for_run(store)
    preparation = prepare_preprocess_settings(settings)
    config = settings.to_preprocess_config()
    config.sorter = None
    config.sorter_path = None
    config.sorter_config_path = None
    # Persistent Runs keep their canonical parameters and completion record in
    # the Run store/final summary. Do not add the legacy copied-source log and
    # duplicate parameter files to the visible scientific session directory.
    config.save_params_json = False
    config.save_manifest_json = False
    config.save_log_mat = False
    output_dir = settings.local_output_dir
    if output_dir is not None and Path(output_dir).exists():
        basename = settings.basename
        stage_owned_outputs = (
            Path(output_dir) / f"{basename}.dat",
            Path(output_dir) / f"{basename}.lfp",
            Path(output_dir) / f"{basename}.session.mat",
            Path(output_dir) / f"{basename}.MergePoints.events.mat",
        )
        if any(path.exists() for path in stage_owned_outputs):
            # If this Attempt is executing, the existing Stage was not adopted
            # as compatible/complete. Do not mix its partial artifacts with a
            # new computation even when the user-level overwrite toggle is off.
            config.overwrite = True
    config.highamp_n_jobs = spec.resources.cpus
    config.job_kwargs = {
        **dict(config.job_kwargs),
        "n_jobs": spec.resources.cpus,
        "max_threads_per_worker": 1,
        "progress_bar": False,
    }
    result = run_preprocess_session(config)
    required = [
        Path(result.dat_path) if result.dat_path is not None else Path("__missing_dat__"),
        Path(result.session_mat_path),
        Path(result.mergepoints_mat_path),
        Path(result.local_output_dir) / f"{result.basename}.xml",
        Path(result.local_output_dir) / "chanMap.mat",
    ]
    validated = _validate_paths(required, label="preprocess output")
    return {
        **preparation,
        "preprocess_result": _serialize_dataclass(result),
        "xml_path": str((Path(result.local_output_dir) / f"{result.basename}.xml").resolve()),
        "chanmap_path": str((Path(result.local_output_dir) / "chanMap.mat").resolve()),
        "dtype": config.dtype,
        "validated_paths": validated,
    }


def _run_sorting(store: RunStore, spec: AttemptSpec) -> dict[str, Any]:
    from src.preprocess.sorting_stage import run_sorting_stage

    preprocess_fact = _load_upstream_result(store, spec, StageName.PREPROCESS)
    preprocess_result = _preprocess_result_from_dict(preprocess_fact["outputs"]["preprocess_result"])
    settings = _settings_for_run(store)
    config = settings.to_preprocess_config()
    if not config.sorter:
        raise ValueError("Sorting Stage is enabled but AnalysisConfig has no sorter")
    config.matlab_max_workers = spec.resources.cpus
    config.job_kwargs = {
        **dict(config.job_kwargs),
        "n_jobs": spec.resources.cpus,
        "max_threads_per_worker": 1,
        "progress_bar": False,
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    if spec.attempt > 1:
        timestamp = f"{timestamp}_a{spec.attempt:03d}"
    result = run_sorting_stage(
        config,
        preprocess_result,
        output_timestamp=timestamp,
    )
    output_dirs = [Path(path) for path in result.sorter_output_dirs]
    if not output_dirs:
        raise RuntimeError("Sorting completed without an output directory")
    validated: list[str] = []
    for output_dir in output_dirs:
        validated.extend(_validate_sorting_output(output_dir))
    manifest = Path(result.sorter_partition_manifest_path) if result.sorter_partition_manifest_path else None
    if manifest is not None:
        _validate_paths([manifest], label="sorter partition manifest")
        _validate_sorter_manifest(manifest, output_dirs)
        manifest_snapshot = store.attempt_dir(spec.stage, spec.attempt) / manifest.name
        atomic_write_text(manifest_snapshot, manifest.read_text(encoding="utf-8"))
        manifest = manifest_snapshot.resolve()
        result = replace(result, sorter_partition_manifest_path=manifest)
        validated.append(str(manifest))
    return {
        "preprocess_result": _serialize_dataclass(result),
        "sorter_output_dir": str(result.sorter_output_dir) if result.sorter_output_dir else "",
        "sorter_output_dirs": [str(path) for path in output_dirs],
        "sorter_partition_manifest_path": str(manifest) if manifest else "",
        "validated_paths": validated,
    }


def _run_postprocess(store: RunStore, spec: AttemptSpec) -> dict[str, Any]:
    from src.postprocess import run_postprocess_session

    settings = _settings_for_run(store)
    config = settings.to_postprocess_config()
    # An executing Attempt was not adopted as compatible and complete. Force a
    # real Stage rerun even when the user-level overwrite toggle is false; the
    # postprocess pipeline versions curated output before replacement.
    config.overwrite = True
    if StageName.PREPROCESS.value in spec.upstream_attempts:
        preprocess_fact = _load_upstream_result(store, spec, StageName.PREPROCESS)
        pre = preprocess_fact["outputs"]["preprocess_result"]
        config.sorting_search_root = Path(pre["local_output_dir"])
        config.dat_path = Path(pre["dat_path"])
        config.sampling_frequency = float(pre["sr"])
        config.num_channels = int(pre["n_channels"])
        config.dtype = str(preprocess_fact["outputs"].get("dtype", config.dtype))
        config.chanmap_mat_path = Path(preprocess_fact["outputs"]["chanmap_path"])
        config.reject_channels = [int(value) for value in pre.get("bad_channels_0based", [])]
    if StageName.SORTING.value in spec.upstream_attempts:
        sorting_fact = _load_upstream_result(store, spec, StageName.SORTING)
        sorting_outputs = sorting_fact["outputs"]
        output_dirs = [Path(path) for path in sorting_outputs.get("sorter_output_dirs", [])]
        config.sorting_phy_folder = output_dirs[0] if len(output_dirs) == 1 else None
    candidate_sorting_dirs: list[Path] = []
    if config.sorting_phy_folder is not None:
        candidate_sorting_dirs.append(Path(config.sorting_phy_folder))
    elif config.sorting_search_root is not None:
        candidate_sorting_dirs.extend(
            path
            for path in Path(config.sorting_search_root).glob("Kilosort*")
            if path.is_dir()
            and "_spi" not in path.name
            and ".preserved-" not in path.name
        )
    required_post_names = {
        "params.py",
        "spike_times.npy",
        "spike_clusters.npy",
        "cluster_group.tsv",
        "cluster_info.tsv",
        "quality_metrics.csv",
    }
    for sorting_dir in candidate_sorting_dirs:
        run_root = sorting_dir.parent if sorting_dir.name == "sorter_output" else sorting_dir
        post_dir = run_root.parent / f"{run_root.name}_spi"
        if post_dir.exists() and not all((post_dir / name).exists() for name in required_post_names):
            config.overwrite = True
            break
    config.job_kwargs = {
        **dict(config.job_kwargs),
        "n_jobs": spec.resources.cpus,
        "progress_bar": False,
    }
    results = list(run_postprocess_session(config))
    if not results:
        raise RuntimeError("Postprocess completed without returning a result")
    validated: list[str] = []
    for result in results:
        validated.extend(
            _validate_paths(
                [Path(result.output_folder), Path(result.metrics_csv_path)],
                label=f"postprocess output {Path(result.output_folder).name}",
            )
        )
    return {
        "postprocess_results": [_serialize_dataclass(result) for result in results],
        "validated_paths": validated,
    }


def run_stage(store: RunStore, spec: AttemptSpec) -> dict[str, Any]:
    if spec.stage == StageName.PREPROCESS:
        return _run_preprocess(store, spec)
    if spec.stage == StageName.SORTING:
        return _run_sorting(store, spec)
    if spec.stage == StageName.POSTPROCESS:
        return _run_postprocess(store, spec)
    raise ValueError(f"Unsupported Stage: {spec.stage.value}")


def execute(run_dir: Path, stage: StageName, attempt: int) -> int:
    _limit_thread_oversubscription()
    store = RunStore(run_dir)
    spec = store.load_attempt_spec(stage, attempt)
    analysis = store.load_analysis()
    attempt_dir = store.attempt_dir(stage, attempt)
    if any((attempt_dir / name).exists() for name in ("started.json", "result.json", "failure.json")):
        raise RuntimeError(
            f"Refusing to execute {stage.value} attempt-{attempt:03d} more than once"
        )
    started_at = _utc_now()
    atomic_write_json(
        attempt_dir / "started.json",
        {
            "stage": stage.value,
            "attempt": attempt,
            "started_at": started_at,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
    )
    try:
        if spec.analysis_sha256 != analysis.sha256:
            raise ValueError("AttemptSpec does not match the immutable AnalysisConfig")
        _verify_analysis_artifacts(store, stage)
        outputs = run_stage(store, spec)
        finished_at = _utc_now()
        atomic_write_json(
            attempt_dir / "result.json",
            {
                "schema_version": 1,
                "stage": stage.value,
                "attempt": attempt,
                "status": StageStatus.COMPLETED.value,
                "analysis_sha256": analysis.sha256,
                "started_at": started_at,
                "finished_at": finished_at,
                "outputs": _json_safe(outputs),
                "validation": {
                    "passed": True,
                    "validated_paths": list(outputs.get("validated_paths", [])),
                },
            },
        )
        return 0
    except BaseException as exc:
        finished_at = _utc_now()
        atomic_write_json(
            attempt_dir / "failure.json",
            {
                "schema_version": 1,
                "stage": stage.value,
                "attempt": attempt,
                "status": StageStatus.FAILED.value,
                "analysis_sha256": analysis.sha256,
                "started_at": started_at,
                "finished_at": finished_at,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "validation": {"passed": False},
            },
        )
        traceback.print_exc()
        return 1
    finally:
        try:
            store.rebuild_state()
        except BaseException:
            # result.json/failure.json is the atomic worker commitment. A
            # derived-view refresh failure must not change the worker exit code.
            traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exactly one persistent pipeline Stage")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", required=True, choices=[stage.value for stage in StageName])
    parser.add_argument("--attempt", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(Path(args.run_dir), StageName(args.stage), int(args.attempt))
    except BaseException:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
