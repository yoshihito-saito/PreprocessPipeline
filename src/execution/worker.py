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
        _verify_file("chanmap", settings.postprocess_chanmap_path())
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
    from src.sorting_manifest import all_partitions_skipped
    manifest = json.loads(path.read_text(encoding="utf-8"))
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list) or not partitions:
        raise RuntimeError("sorter partition manifest has no partitions")
    if not output_dirs and not all_partitions_skipped(path):
        raise RuntimeError("Empty sorting outputs require an explicitly all-skipped manifest")
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
    from .session import prepare_preprocess_output_contract

    # This must happen before runtime preparation or pipeline execution can
    # reuse/create a single canonical scientific output.
    contract_path = prepare_preprocess_output_contract(
        store, settings, overwrite=bool(settings.preprocess.overwrite)
    )
    preparation = prepare_preprocess_settings(settings)
    # Runtime preparation may replace the source basepath with a staged
    # multi-day combined input; construct the scientific config only after
    # that rewrite while retaining the contract's canonical output identity.
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
    # `overwrite` is part of the immutable analysis snapshot.  In particular,
    # an interrupted attempt must not turn a later resume into destructive
    # recomputation of already published canonical outputs.
    config.highamp_n_jobs = spec.resources.cpus
    config.job_kwargs = {
        **dict(config.job_kwargs),
        "n_jobs": spec.resources.cpus,
        "max_threads_per_worker": 1,
        "progress_bar": False,
    }
    result = run_preprocess_session(config)
    from .session import preprocess_output_inventory, validate_output_inventory

    inventory = preprocess_output_inventory(result, config)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        contract = {}
    if contract.get("producer_schema") == "legacy-preprocess-v1":
        inventory["producer_schema"] = "preprocess-output-validated-legacy-resume-v1"
    validated = validate_output_inventory(inventory, label="preprocess output")
    return {
        **preparation,
        "preprocess_result": _serialize_dataclass(result),
        "xml_path": str((Path(result.local_output_dir) / f"{result.basename}.xml").resolve()),
        "chanmap_path": str((Path(result.local_output_dir) / "chanMap.mat").resolve()),
        "dtype": config.dtype,
        "validated_paths": validated,
        "output_inventory": inventory,
    }


def _run_sorting(store: RunStore, spec: AttemptSpec) -> dict[str, Any]:
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
    from .gpu_selection import GpuUsageMonitor, activate_least_used_gpu
    from src.preprocess.channel_layout import load_channel_layout
    from src.sorting_manifest import all_partitions_skipped

    map_path = Path(preprocess_fact["outputs"].get("chanmap_path") or
                    config.chanmap_mat_path or Path(preprocess_result.local_output_dir) / "chanMap.mat")
    active = set(range(preprocess_result.n_channels))
    if map_path.exists():
        layout = load_channel_layout(map_path, preprocess_result.n_channels)
        active = set(layout["chanMap0ind"][layout["connected"]].tolist())
    if config.bad_channels:
        active.difference_update(preprocess_result.bad_channels_0based)

    gpu_log_path = store.attempt_dir(spec.stage, spec.attempt) / "gpu-selection.jsonl"
    gpu_selection = activate_least_used_gpu(
        log_path=gpu_log_path
    ) if active else {"selected_gpu": None, "skip_reason": "no_active_channels"}
    # Import the sorter only after CUDA visibility has been finalized. Some
    # sorter dependencies initialize CUDA as part of their import path.
    from src.preprocess.sorting_stage import run_sorting_stage

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    if spec.attempt > 1:
        timestamp = f"{timestamp}_a{spec.attempt:03d}"
    selected_gpu = gpu_selection["selected_gpu"]
    if active and (not isinstance(selected_gpu, dict) or not selected_gpu.get("uuid")):
        raise RuntimeError("GPU selection returned no selected GPU UUID")
    gpu_monitor = GpuUsageMonitor(
        log_path=gpu_log_path,
        selected_gpu_uuid=str(selected_gpu["uuid"]),
    ) if active else None
    if gpu_monitor is not None:
        gpu_monitor.start()
    try:
        result = run_sorting_stage(
            config,
            preprocess_result,
            output_timestamp=timestamp,
        )
    except BaseException:
        if gpu_monitor is not None:
            gpu_monitor.stop(final_status="sorter_failed")
        raise
    else:
        if gpu_monitor is not None:
            gpu_monitor.stop(final_status="sorter_finished")
    output_dirs = [Path(path) for path in result.sorter_output_dirs]
    if not output_dirs and not all_partitions_skipped(result.sorter_partition_manifest_path):
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
        "gpu_selection": gpu_selection,
        "skip_reason": "no_active_channels" if not output_dirs else None,
        "validated_paths": validated,
    }


def _run_postprocess(store: RunStore, spec: AttemptSpec) -> dict[str, Any]:
    from src.postprocess import run_postprocess_session

    settings = _settings_for_run(store)
    config = settings.to_postprocess_config()
    # Preserve the saved overwrite contract.  The postprocess implementation
    # remains responsible for its explicitly allowed atomic cluster-group
    # update, but this worker must not broaden that authority.
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
        from src.sorting_manifest import all_partitions_skipped
        manifest = sorting_outputs.get("sorter_partition_manifest_path")
        if all_partitions_skipped(Path(manifest) if manifest else None):
            return {"postprocess_results": [], "skip_reason": "no_active_channels",
                    "sorter_partition_manifest_path": manifest, "validated_paths": [manifest]}
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
    config.job_kwargs = {
        **dict(config.job_kwargs),
        "n_jobs": spec.resources.cpus,
        "progress_bar": False,
    }
    results = list(run_postprocess_session(config))
    if not results:
        from src.postprocess.pipeline import _resolve_recording_for_postprocess, _resolve_postprocess_targets
        from src.preprocess.channel_layout import NoActiveChannels
        from src.sorting_manifest import all_partitions_skipped
        manifest = (Path(config.sorting_search_root) / "sorter_partition_manifest.json"
                    if config.sorting_search_root is not None else None)
        if all_partitions_skipped(manifest):
            return {"postprocess_results": [], "skip_reason": "no_active_channels",
                    "sorter_partition_manifest_path": str(manifest), "validated_paths": [str(manifest)]}
        targets = _resolve_postprocess_targets(config)
        all_bad = bool(targets)
        for target in targets:
            try:
                _resolve_recording_for_postprocess(replace(config, sorting_phy_folder=target))
            except NoActiveChannels:
                continue
            all_bad = False
            break
        if all_bad:
            return {"postprocess_results": [], "skip_reason": "no_active_channels",
                    "validated_paths": [str(path) for path in (config.dat_path, config.chanmap_mat_path, config.xml_path) if path]}
        raise RuntimeError("Postprocess completed without returning a result")
    validated: list[str] = []
    for result in results:
        from .session import _validate_post_output

        validated.extend(_validate_post_output(Path(result.output_folder)))
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
    allocator_environment = {
        name: os.environ.get(name)
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    }
    atomic_write_json(
        attempt_dir / "started.json",
        {
            "stage": stage.value,
            "attempt": attempt,
            "started_at": started_at,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "allocator_environment": allocator_environment,
        },
    )
    try:
        if stage == StageName.SORTING:
            print(
                "[Torch allocator environment] " + json.dumps(allocator_environment, sort_keys=True),
                flush=True,
            )
        if spec.analysis_sha256 != analysis.sha256:
            raise ValueError("AttemptSpec does not match the immutable AnalysisConfig")
        _verify_analysis_artifacts(store, stage)
        outputs = run_stage(store, spec)
        from .session import OUTPUT_INVENTORY_SCHEMA, PRODUCER_SCHEMAS

        # Stages that already supply a richer inventory (preprocess) retain
        # it.  Sorting/postprocess inventories derive from their deep
        # validation list, so completion always records all validated files.
        inventory = outputs.get("output_inventory")
        if inventory is None:
            inventory = {
                "schema": OUTPUT_INVENTORY_SCHEMA,
                "producer_schema": PRODUCER_SCHEMAS[stage.value],
                "entries": [
                    {"path": str(Path(path).resolve()), "role": "validated_output"}
                    for path in outputs.get("validated_paths", [])
                    if Path(path).is_file()
                ],
            }
            outputs["output_inventory"] = inventory
        allowed_schemas = {PRODUCER_SCHEMAS[stage.value]}
        if stage == StageName.PREPROCESS:
            allowed_schemas.add("preprocess-output-validated-legacy-resume-v1")
        if inventory.get("producer_schema") not in allowed_schemas:
            raise RuntimeError(f"{stage.value} output inventory has an incompatible producer schema")
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
                "producer_schema": inventory["producer_schema"],
                "validation": {
                    "passed": True,
                    "validated_paths": list(outputs.get("validated_paths", [])),
                    "output_inventory": _json_safe(inventory),
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
