from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

import numpy as np

from .models import StageName, StageStatus
from .store import (
    RunStore,
    atomic_write_json,
    atomic_write_text,
    read_json,
    short_file_lock,
    utc_now,
)


_ACTIVE_RUN_STATES = {"pending", "submitted", "running", "lost"}
_MUTATING_STAGE_STATES = {"submitted", "running", "lost"}
_PHY_REQUIRED = ("params.py", "spike_times.npy", "spike_clusters.npy", "templates.npy")
_POST_REQUIRED = (
    "params.py",
    "spike_times.npy",
    "spike_clusters.npy",
    "cluster_group.tsv",
    "cluster_info.tsv",
    "quality_metrics.csv",
)


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stage_fingerprints(analysis: Any) -> dict[str, str]:
    settings = analysis.settings if hasattr(analysis, "settings") else dict(analysis["settings"])
    artifacts = (
        analysis.artifact_sha256
        if hasattr(analysis, "artifact_sha256")
        else dict(analysis.get("artifact_sha256", {}))
    )
    preprocess = dict(settings.get("preprocess", {}))
    sorting_keys = {
        "run_sorter",
        "sorter",
        "sorter_path",
        "sorter_config_path",
        "sorter_partition_mode",
        "sorter_verbose",
        "cleanup_temp_wh",
    }
    runtime_keys = {"overwrite", "preprocess_worker_count", "sorter_worker_count", "matlab_path"}
    preprocess_payload = {
        key: value
        for key, value in preprocess.items()
        if key not in sorting_keys | runtime_keys
    }
    preprocess_payload["multi_day_enabled"] = settings.get("multi_day_enabled", False)
    preprocess_payload["multi_day_session_paths"] = settings.get("multi_day_session_paths", [])
    preprocess_payload["multi_day_selected_subepoch_paths"] = settings.get(
        "multi_day_selected_subepoch_paths", []
    )
    preprocess_payload["xml_sha256"] = artifacts.get("xml", "")
    sorting_payload = {
        key: preprocess.get(key)
        for key in sorted(sorting_keys)
        if key in preprocess
    }
    sorting_payload["sorter_config_sha256"] = artifacts.get("sorter_config", "")
    postprocess = dict(settings.get("postprocess", {}))
    for key in (
        "overwrite",
        "worker_count",
        "cell_explorer_sorting_folders",
    ):
        postprocess.pop(key, None)
    postprocess["chanmap_sha256"] = artifacts.get("chanmap", "")
    postprocess["xml_sha256"] = artifacts.get("xml", "")
    postprocess["sorting_input_identity"] = artifacts.get(
        "postprocess_sorting_input", ""
    )
    postprocess["dat_input_identity"] = artifacts.get("postprocess_dat_input", "")
    return {
        StageName.PREPROCESS.value: _hash_payload(preprocess_payload),
        StageName.SORTING.value: _hash_payload(sorting_payload),
        StageName.POSTPROCESS.value: _hash_payload(postprocess),
    }


def _prior_stage_fingerprints(session_dir: Path) -> dict[str, str] | None:
    path = session_dir / "preprocess_run.yaml"
    if not path.exists():
        return None
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        recorded = payload.get("stage_fingerprints")
        if isinstance(recorded, dict):
            return {str(key): str(value) for key, value in recorded.items()}
        analysis = payload.get("analysis")
        if isinstance(analysis, dict):
            return stage_fingerprints(analysis)
    except (OSError, TypeError, ValueError):
        return None
    return None


def _persistent_input_provenance_compatible(
    session_dir: Path, store: RunStore, settings: Any
) -> bool:
    prior_path = session_dir / "preprocess_run.yaml"
    current_path = store.run_dir / "snapshots" / "inputs.json"
    if not prior_path.exists() or not current_path.exists():
        return False
    try:
        import yaml

        prior_record = yaml.safe_load(prior_path.read_text(encoding="utf-8")) or {}
        prior_snapshot = (prior_record.get("provenance") or {}).get("inputs") or {}
        current_snapshot = read_json(current_path)
        return _input_snapshots_compatible(prior_snapshot, current_snapshot, settings)
    except (OSError, TypeError, ValueError):
        return False


def _input_snapshots_compatible(
    prior_snapshot: dict[str, Any], current_snapshot: dict[str, Any], settings: Any
) -> bool:
    prior_entries = {
        str(item.get("path")): item for item in prior_snapshot.get("inputs", [])
    }
    current_entries = {
        str(item.get("path")): item for item in current_snapshot.get("inputs", [])
    }
    relevant = [settings.preprocess_source_path]
    relevant.extend(
        Path(value).expanduser()
        for value in settings.multi_day_session_paths
        if str(value).strip()
    )
    relevant.extend(
        Path(value).expanduser()
        for value in settings.multi_day_selected_subepoch_paths
        if str(value).strip()
    )
    relevant_roots = {str(Path(path).resolve()) for path in relevant if path is not None}
    if not relevant_roots:
        return False
    relevant_paths = {
        path
        for path in current_entries
        if any(path == root or path.startswith(root + os.sep) for root in relevant_roots)
    }
    prior_relevant_paths = {
        path
        for path in prior_entries
        if any(path == root or path.startswith(root + os.sep) for root in relevant_roots)
    }
    if relevant_paths != prior_relevant_paths:
        return False
    keys = ("exists", "size", "mtime_ns", "is_dir")
    return all(
        path in prior_entries
        and path in current_entries
        and all(prior_entries[path].get(key) == current_entries[path].get(key) for key in keys)
        for path in relevant_paths
    )


def _legacy_preprocess_compatible(session_dir: Path, settings: Any) -> bool:
    path = session_dir / "preprocessSession_params.json"
    if not path.exists():
        return False
    try:
        prior = read_json(path)
        current = _json_safe(asdict(settings.to_preprocess_config()))
    except (OSError, TypeError, ValueError):
        return False
    excluded = {
        "basepath",
        "localpath",
        "output_dir",
        "chanmap_mat_path",
        "xml_path",
        "sorter",
        "sorter_path",
        "sorter_config_path",
        "sorter_partition_mode",
        "matlab_path",
        "matlab_max_workers",
        "sorter_verbose",
        "cleanup_temp_wh",
        "highamp_n_jobs",
        "overwrite",
        "job_kwargs",
        "save_params_json",
        "save_manifest_json",
        "save_log_mat",
    }
    comparable_keys = set(current) - excluded
    if not comparable_keys.issubset(prior):
        return False
    return all(_json_safe(prior[key]) == _json_safe(current[key]) for key in comparable_keys)


def _legacy_sorting_compatible(session_dir: Path, store: RunStore, settings: Any) -> bool:
    expected = store.load_analysis().artifact_sha256.get("sorter_config")
    params_path = session_dir / "preprocessSession_params.json"
    candidates = [
        path
        for output in session_dir.glob("Kilosort*")
        if output.is_dir()
        and "_spi" not in output.name
        and ".preserved-" not in output.name
        for path in output.glob("sorter_config_source.*")
        if path.is_file()
    ]
    if not expected or not candidates or not params_path.exists():
        return False
    try:
        prior = read_json(params_path)
        current = _json_safe(asdict(settings.to_preprocess_config()))
        for key in ("sorter", "sorter_partition_mode"):
            if key not in prior or _json_safe(prior[key]) != current.get(key):
                return False
        return any(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path in candidates)
    except (OSError, TypeError, ValueError):
        return False


def _json_safe(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def settings_for_store(store: RunStore):
    from src.preprocess.gui.config_model import PipelineGuiSettings

    return PipelineGuiSettings.from_json(json.dumps(store.load_analysis().settings))


def session_output_dir(settings: Any) -> Path:
    output = settings.local_output_dir
    if output is None:
        raise ValueError("The session output directory cannot be resolved")
    return Path(output).expanduser().resolve()


def _claim_path(pipeline_root: Path, session_dir: Path) -> Path:
    del pipeline_root
    return session_dir.resolve() / ".pipeline-active-run.json"


def _claim_blocks_new_run(claim: dict[str, Any]) -> bool:
    if claim.get("kind") == "manual":
        try:
            pid = int(claim.get("owner_pid") or 0)
            if pid <= 0:
                return True
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except (OSError, TypeError, ValueError):
            return True
    run_dir_text = str(claim.get("run_dir") or "").strip()
    if not run_dir_text:
        return True
    run_dir = Path(run_dir_text)
    if not (run_dir / "run.json").exists():
        created_text = str(claim.get("created_at") or "")
        try:
            created = datetime.fromisoformat(created_text)
            return (datetime.now(timezone.utc) - created).total_seconds() < 60.0
        except (TypeError, ValueError):
            return True
    try:
        store = RunStore(run_dir)
        state = store.derive_state()
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    for stage in StageName:
        for attempt in store.list_attempt_numbers(stage):
            submitted = store.read_attempt_fact(stage, attempt, "submitted.json")
            if submitted is not None:
                observation = store.latest_observation(stage, attempt) or {}
                backend_status = observation.get("status") or {}
                conclusive = (
                    backend_status.get("terminal") is True
                    and backend_status.get("successful") is not None
                )
                cancelled = (
                    store.read_attempt_fact(stage, attempt, "cancel_confirmed.json") is not None
                )
                if not conclusive and not cancelled:
                    return True
            started = store.read_attempt_fact(stage, attempt, "started.json")
            if started is not None and not any(
                store.read_attempt_fact(stage, attempt, name) is not None
                for name in ("result.json", "failure.json", "cancel_confirmed.json")
            ):
                return True
            intent = store.read_attempt_fact(stage, attempt, "submission_intent.json")
            submission_failure = store.read_attempt_fact(
                stage, attempt, "submission_failure.json"
            )
            if intent is not None and submitted is None and submission_failure is None:
                return True
    if str(state.get("status", "unknown")) in _ACTIVE_RUN_STATES:
        return True
    return any(
        str(view.get("status", "unknown")) in _MUTATING_STAGE_STATES
        for view in (state.get("stages") or {}).values()
        if isinstance(view, dict) and view.get("enabled")
    )


def acquire_session_claim(
    *, pipeline_root: Path, session_dir: Path, run_id: str, run_dir: Path
) -> Path:
    claim_path = _claim_path(pipeline_root, session_dir)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        previous_run_dir = ""
        if claim_path.exists():
            claim = read_json(claim_path)
            if _claim_blocks_new_run(claim):
                raise RuntimeError(
                    "Another persistent Run still owns this session output: "
                    f"{claim.get('run_id', 'unknown')} ({claim.get('run_dir', 'unknown')})"
                )
            if claim.get("kind", "run") == "run":
                previous_run_dir = str(claim.get("run_dir") or "")
        atomic_write_json(
            claim_path,
            {
                "schema_version": 1,
                "kind": "run",
                "run_id": run_id,
                "run_dir": str(run_dir.resolve()),
                "session_dir": str(session_dir.resolve()),
                "previous_run_dir": previous_run_dir,
                "created_at": utc_now(),
            },
        )
    return claim_path


def abandon_session_claim(*, claim_path: Path, run_id: str) -> None:
    """Remove an unsubmitted claim while preserving a recoverable prior Run pointer."""

    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        if not claim_path.exists():
            return
        claim = read_json(claim_path)
        if claim.get("kind", "run") != "run" or claim.get("run_id") != run_id:
            return
        previous_dir = Path(str(claim.get("previous_run_dir") or ""))
        if (previous_dir / "run.json").exists():
            previous_run = read_json(previous_dir / "run.json")
            atomic_write_json(
                claim_path,
                {
                    "schema_version": 1,
                    "kind": "run",
                    "run_id": previous_run.get("run_id", previous_dir.name),
                    "run_dir": str(previous_dir.resolve()),
                    "session_dir": str(claim.get("session_dir") or claim_path.parent),
                    "created_at": previous_run.get("created_at", utc_now()),
                },
            )
        else:
            claim_path.unlink(missing_ok=True)


def acquire_manual_session_claim(*, session_dir: Path, owner: str) -> str:
    claim_path = _claim_path(Path(), session_dir)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    token = uuid.uuid4().hex
    with short_file_lock(guard, timeout=60.0):
        previous_claim: dict[str, Any] | None = None
        if claim_path.exists():
            existing = read_json(claim_path)
            if _claim_blocks_new_run(existing):
                raise RuntimeError("A persistent Run or manual application still owns this session")
            if existing.get("kind", "run") == "run":
                previous_claim = existing
        atomic_write_json(
            claim_path,
            {
                "schema_version": 1,
                "kind": "manual",
                "owner": owner,
                "owner_pid": os.getpid(),
                "token": token,
                "previous_claim": previous_claim,
                "session_dir": str(session_dir.resolve()),
                "created_at": utc_now(),
            },
        )
    return token


def release_manual_session_claim(*, session_dir: Path, token: str) -> None:
    claim_path = _claim_path(Path(), session_dir)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        if not claim_path.exists():
            return
        claim = read_json(claim_path)
        if claim.get("kind") == "manual" and claim.get("token") == token:
            previous = claim.get("previous_claim")
            if isinstance(previous, dict) and previous.get("run_dir"):
                atomic_write_json(claim_path, previous)
            else:
                claim_path.unlink(missing_ok=True)


def update_manual_session_claim_pid(*, session_dir: Path, token: str, child_pid: int) -> None:
    if child_pid <= 0:
        raise ValueError("A positive child PID is required for a manual session lease")
    claim_path = _claim_path(Path(), session_dir)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        claim = read_json(claim_path)
        if claim.get("kind") != "manual" or claim.get("token") != token:
            raise RuntimeError("Manual session lease changed before child launch completed")
        claim["owner_pid"] = int(child_pid)
        claim["child_started_at"] = utc_now()
        atomic_write_json(claim_path, claim)


def release_session_claim(store: RunStore) -> None:
    claim_text = str(store.load_run().get("session_claim_path") or "").strip()
    if not claim_text:
        return
    claim_path = Path(claim_text)
    guard = claim_path.parent / f".{claim_path.name}.lock"
    with short_file_lock(guard, timeout=60.0):
        if not claim_path.exists():
            return
        claim = read_json(claim_path)
        if claim.get("run_id") == store.load_run().get("run_id"):
            claim_path.unlink(missing_ok=True)


def active_session_claim(*, pipeline_root: Path, session_dir: Path) -> dict[str, Any] | None:
    claim_path = _claim_path(pipeline_root, session_dir)
    if not claim_path.exists():
        return None
    claim = read_json(claim_path)
    return claim if _claim_blocks_new_run(claim) else None


def _validate_phy_output(output_dir: Path) -> list[str]:
    missing = [str(output_dir / name) for name in _PHY_REQUIRED if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError("sorting output is incomplete: " + ", ".join(missing))
    spike_times = np.load(output_dir / "spike_times.npy", mmap_mode="r", allow_pickle=False)
    spike_clusters = np.load(output_dir / "spike_clusters.npy", mmap_mode="r", allow_pickle=False)
    templates = np.load(output_dir / "templates.npy", mmap_mode="r", allow_pickle=False)
    if spike_times.reshape(-1).shape[0] != spike_clusters.reshape(-1).shape[0]:
        raise RuntimeError("sorting output has inconsistent spike_times/spike_clusters lengths")
    if templates.ndim < 2 or (output_dir / "params.py").stat().st_size == 0:
        raise RuntimeError("sorting output has invalid templates or params.py")
    return [str((output_dir / name).resolve()) for name in _PHY_REQUIRED]


def _validate_post_output(output_dir: Path) -> list[str]:
    required = [output_dir / name for name in _POST_REQUIRED]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("postprocess output is incomplete: " + ", ".join(missing))
    spike_times = np.load(output_dir / "spike_times.npy", mmap_mode="r", allow_pickle=False)
    spike_clusters = np.load(
        output_dir / "spike_clusters.npy", mmap_mode="r", allow_pickle=False
    )
    if spike_times.reshape(-1).shape[0] != spike_clusters.reshape(-1).shape[0]:
        raise RuntimeError("postprocess output has inconsistent spike array lengths")
    return [str(path.resolve()) for path in required]


def _legacy_preprocess_result(session_dir: Path, settings: Any) -> tuple[dict[str, Any], list[str]] | None:
    manifest_path = session_dir / "preprocessSession_manifest.json"
    manifest: dict[str, Any]
    final_record = session_dir / "preprocess_run.yaml"
    if final_record.exists():
        try:
            import yaml

            payload = yaml.safe_load(final_record.read_text(encoding="utf-8")) or {}
            stage_result = (payload.get("stages") or {}).get("preprocess") or {}
            stage_result = stage_result.get("result") or stage_result
            manifest = dict((stage_result.get("outputs") or {}).get("preprocess_result") or {})
        except (OSError, TypeError, ValueError):
            return None
    elif manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        return None
    if not manifest:
        return None
    basename = str(manifest.get("basename") or session_dir.name)
    dat_path = Path(str(manifest.get("dat_path") or session_dir / f"{basename}.dat"))
    required = [
        dat_path,
        Path(str(manifest.get("session_mat_path") or session_dir / f"{basename}.session.mat")),
        Path(
            str(
                manifest.get("mergepoints_mat_path")
                or session_dir / f"{basename}.MergePoints.events.mat"
            )
        ),
        session_dir / f"{basename}.xml",
        session_dir / "chanMap.mat",
    ]
    if any(not path.exists() or (path.is_file() and path.stat().st_size == 0) for path in required):
        return None
    n_channels = int(manifest.get("n_channels") or 0)
    if n_channels < 1 or dat_path.stat().st_size % (n_channels * np.dtype("int16").itemsize):
        return None
    sample_counts = [int(value) for value in manifest.get("subsession_sample_counts", [])]
    if sample_counts:
        actual_frames = dat_path.stat().st_size // (n_channels * np.dtype("int16").itemsize)
        if actual_frames != sum(sample_counts):
            return None
    source_paths = [Path(str(value)).expanduser() for value in manifest.get("subsession_paths", [])]
    if not source_paths or len(source_paths) != len(sample_counts):
        return None
    if any(
        not path.exists()
        or not path.is_file()
        or count <= 0
        or path.stat().st_size % (count * np.dtype("int16").itemsize) != 0
        or path.stat().st_size // (count * np.dtype("int16").itemsize) < n_channels
        for path, count in zip(source_paths, sample_counts, strict=True)
    ):
        return None
    source = settings.preprocess_source_path or Path(str(manifest.get("basepath") or session_dir))
    result = {
        "basepath": str(Path(source).resolve()),
        "basename": basename,
        "local_output_dir": str(session_dir.resolve()),
        "dat_path": str(dat_path.resolve()),
        "lfp_path": str(manifest.get("lfp_path") or "") or None,
        "session_mat_path": str(required[1].resolve()),
        "mergepoints_mat_path": str(required[2].resolve()),
        "analog_event_paths": list(manifest.get("analog_event_paths", [])),
        "digital_event_paths": list(manifest.get("digital_event_paths", [])),
        "intermediate_dat_paths": dict(manifest.get("intermediate_dat_paths", {})),
        "n_channels": n_channels,
        "sr": float(manifest.get("sr") or 0.0),
        "sr_lfp": manifest.get("sr_lfp"),
        "bad_channels_0based": [int(value) for value in manifest.get("bad_channels_0based", [])],
        "bad_channels_1based": [int(value) for value in manifest.get("bad_channels_1based", [])],
        "subsession_paths": list(manifest.get("subsession_paths", [])),
        "subsession_sample_counts": sample_counts,
        "sorter": None,
        "sorter_output_dir": None,
        "sorter_output_dirs": [],
        "sorter_partition_manifest_path": None,
        "state_score_paths": list(manifest.get("state_score_paths", [])),
        "state_score_figure_paths": list(manifest.get("state_score_figure_paths", [])),
    }
    return result, [str(path.resolve()) for path in required]


def _sorting_outputs(session_dir: Path) -> tuple[list[Path], Path | None]:
    manifest_path = session_dir / "sorter_partition_manifest.json"
    outputs: list[Path] = []
    if manifest_path.exists():
        payload = read_json(manifest_path)
        for item in payload.get("partitions", []):
            if not isinstance(item, dict) or item.get("status") not in {None, "completed"}:
                continue
            value = str(item.get("output_folder") or "").strip()
            if value:
                outputs.append(Path(value).expanduser().resolve())
    if not outputs:
        outputs = sorted(
            (
                path.resolve()
                for path in session_dir.glob("Kilosort*")
                if path.is_dir()
                and "_spi" not in path.name
                and ".preserved-" not in path.name
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:1]
    return outputs, manifest_path if manifest_path.exists() else None


def _write_adopted_result(
    store: RunStore,
    *,
    stage: StageName,
    attempt: int,
    outputs: dict[str, Any],
    validated_paths: list[str],
    source: str,
) -> None:
    now = utc_now()
    store.write_attempt_fact(
        stage,
        attempt,
        "adopted.json",
        {"adopted_at": now, "source": source, "validated_paths": validated_paths},
    )
    store.write_attempt_fact(
        stage,
        attempt,
        "result.json",
        {
            "schema_version": 1,
            "stage": stage.value,
            "attempt": attempt,
            "status": StageStatus.COMPLETED.value,
            "analysis_sha256": store.load_analysis().sha256,
            "started_at": now,
            "finished_at": now,
            "adopted": True,
            "outputs": _json_safe(outputs),
            "validation": {"passed": True, "validated_paths": validated_paths},
        },
    )


def _previous_persistent_store(store: RunStore) -> RunStore | None:
    claim_text = str(store.load_run().get("session_claim_path") or "").strip()
    if not claim_text or not Path(claim_text).exists():
        return None
    try:
        claim = read_json(Path(claim_text))
        previous = Path(str(claim.get("previous_run_dir") or ""))
        if previous == store.run_dir or not (previous / "run.json").exists():
            return None
        return RunStore(previous)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_previous_result(
    previous: RunStore, stage: StageName
) -> tuple[dict[str, Any], list[str]] | None:
    view = previous.derive_state()["stages"][stage.value]
    if view.get("status") != StageStatus.COMPLETED.value:
        return None
    attempt = view.get("selected_attempt")
    if attempt is None:
        return None
    result = previous.read_attempt_fact(stage, int(attempt), "result.json")
    if (
        result is None
        or result.get("status") != StageStatus.COMPLETED.value
        or result.get("analysis_sha256") != previous.load_analysis().sha256
        or (result.get("validation") or {}).get("passed") is not True
    ):
        return None
    try:
        validated = _revalidate_previous_outputs(stage, result)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return result, validated


def _revalidate_previous_outputs(stage: StageName, result: dict[str, Any]) -> list[str]:
    outputs = dict(result.get("outputs") or {})
    if stage == StageName.PREPROCESS:
        pre = dict(outputs.get("preprocess_result") or {})
        output_dir = Path(str(pre.get("local_output_dir") or ""))
        basename = str(pre.get("basename") or output_dir.name)
        dat_path = Path(str(pre.get("dat_path") or output_dir / f"{basename}.dat"))
        required = [
            dat_path,
            Path(str(pre.get("session_mat_path") or output_dir / f"{basename}.session.mat")),
            Path(
                str(
                    pre.get("mergepoints_mat_path")
                    or output_dir / f"{basename}.MergePoints.events.mat"
                )
            ),
            output_dir / f"{basename}.xml",
            output_dir / "chanMap.mat",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("previous preprocess output is incomplete")
        n_channels = int(pre.get("n_channels") or 0)
        if n_channels < 1 or dat_path.stat().st_size % (n_channels * np.dtype("int16").itemsize):
            raise RuntimeError("previous preprocess dat has an invalid frame shape")
        sample_counts = [int(value) for value in pre.get("subsession_sample_counts", [])]
        if sample_counts:
            frames = dat_path.stat().st_size // (n_channels * np.dtype("int16").itemsize)
            if frames != sum(sample_counts):
                raise RuntimeError("previous preprocess dat has a stale sample count")
        return [str(path.resolve()) for path in required]
    if stage == StageName.SORTING:
        output_dirs = [
            Path(value).expanduser().resolve()
            for value in outputs.get("sorter_output_dirs", [])
            if str(value).strip()
        ]
        if not output_dirs:
            raise RuntimeError("previous sorting result has no output directories")
        validated = [item for path in output_dirs for item in _validate_phy_output(path)]
        manifest_text = str(outputs.get("sorter_partition_manifest_path") or "").strip()
        if manifest_text:
            manifest_path = Path(manifest_text)
            manifest = read_json(manifest_path)
            completed = {
                str(Path(item.get("output_folder", "")).resolve())
                for item in manifest.get("partitions", [])
                if isinstance(item, dict)
                and item.get("status") == StageStatus.COMPLETED.value
                and item.get("output_folder")
            }
            if completed != {str(path) for path in output_dirs}:
                raise RuntimeError("previous sorter manifest disagrees with output directories")
            validated.append(str(manifest_path.resolve()))
        return validated
    post_results = outputs.get("postprocess_results", [])
    output_dirs = [
        Path(str(item.get("output_folder"))).resolve()
        for item in post_results
        if isinstance(item, dict) and item.get("output_folder")
    ]
    if not output_dirs:
        raise RuntimeError("previous postprocess result has no output directories")
    return [item for path in output_dirs for item in _validate_post_output(path)]


def adopt_existing_outputs(store: RunStore) -> dict[str, str]:
    """Adopt deeply validated legacy outputs for overwrite-disabled Stages."""

    settings = settings_for_store(store)
    session_dir = session_output_dir(settings)
    enabled = [StageName(value) for value in store.load_run().get("enabled_stages", [])]
    decisions: dict[str, str] = {stage.value: "run" for stage in enabled}
    current_fingerprints = stage_fingerprints(store.load_analysis())
    prior_fingerprints = _prior_stage_fingerprints(session_dir)
    previous = _previous_persistent_store(store)
    previous_fingerprints = (
        dict(previous.load_run().get("stage_fingerprints") or {}) if previous is not None else {}
    )
    if previous is not None and not previous_fingerprints:
        previous_fingerprints = stage_fingerprints(previous.load_analysis())

    def previous_compatible(stage: StageName) -> bool:
        if previous is None:
            return False
        if previous_fingerprints.get(stage.value) != current_fingerprints[stage.value]:
            return False
        if stage != StageName.PREPROCESS:
            return True
        try:
            prior_inputs = read_json(previous.run_dir / "snapshots" / "inputs.json")
            current_inputs = read_json(store.run_dir / "snapshots" / "inputs.json")
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return _input_snapshots_compatible(prior_inputs, current_inputs, settings)

    def compatible(stage: StageName) -> bool:
        if prior_fingerprints is not None:
            matches = prior_fingerprints.get(stage.value) == current_fingerprints[stage.value]
            if stage == StageName.PREPROCESS:
                matches = matches and _persistent_input_provenance_compatible(
                    session_dir, store, settings
                )
            return matches
        if stage == StageName.PREPROCESS:
            return _legacy_preprocess_compatible(session_dir, settings)
        if stage == StageName.SORTING:
            return _legacy_sorting_compatible(session_dir, store, settings)
        # Legacy postprocess folders do not contain a complete, canonical copy
        # of the scientific postprocess settings. Shape validation alone is not
        # evidence of parameter compatibility.
        return False

    preprocess_result: dict[str, Any] | None = None
    preprocess_attempt = store.list_attempt_numbers(StageName.PREPROCESS)
    if (
        StageName.PREPROCESS in enabled
        and preprocess_attempt
        and not settings.preprocess.overwrite
        and previous_compatible(StageName.PREPROCESS)
        and previous is not None
    ):
        adopted_previous = _completed_previous_result(previous, StageName.PREPROCESS)
        if adopted_previous is not None:
            prior_result, validated = adopted_previous
            _write_adopted_result(
                store,
                stage=StageName.PREPROCESS,
                attempt=preprocess_attempt[-1],
                outputs=dict(prior_result.get("outputs") or {}),
                validated_paths=validated,
                source=f"persistent Run {previous.run_dir.name}",
            )
            preprocess_result = dict(
                (prior_result.get("outputs") or {}).get("preprocess_result") or {}
            )
            decisions[StageName.PREPROCESS.value] = "reuse"
    if (
        StageName.PREPROCESS in enabled
        and preprocess_attempt
        and decisions[StageName.PREPROCESS.value] != "reuse"
        and not settings.preprocess.overwrite
        and compatible(StageName.PREPROCESS)
    ):
        adopted = _legacy_preprocess_result(session_dir, settings)
        if adopted is not None:
            preprocess_result, validated = adopted
            _write_adopted_result(
                store,
                stage=StageName.PREPROCESS,
                attempt=preprocess_attempt[-1],
                outputs={
                    "preprocess_result": preprocess_result,
                    "xml_path": str((session_dir / f"{session_dir.name}.xml").resolve()),
                    "chanmap_path": str((session_dir / "chanMap.mat").resolve()),
                    "dtype": "int16",
                    "validated_paths": validated,
                },
                validated_paths=validated,
                source="legacy preprocess manifest",
            )
            decisions[StageName.PREPROCESS.value] = "reuse"

    sorting_attempt = store.list_attempt_numbers(StageName.SORTING)
    sorting_outputs: list[Path] = []
    if (
        StageName.SORTING in enabled
        and sorting_attempt
        and not settings.preprocess.overwrite
        and previous_compatible(StageName.SORTING)
        and previous is not None
        and (
            StageName.PREPROCESS not in enabled
            or decisions.get(StageName.PREPROCESS.value) == "reuse"
        )
    ):
        adopted_previous = _completed_previous_result(previous, StageName.SORTING)
        if adopted_previous is not None:
            prior_result, validated = adopted_previous
            _write_adopted_result(
                store,
                stage=StageName.SORTING,
                attempt=sorting_attempt[-1],
                outputs=dict(prior_result.get("outputs") or {}),
                validated_paths=validated,
                source=f"persistent Run {previous.run_dir.name}",
            )
            sorting_outputs = [
                Path(value).resolve()
                for value in (prior_result.get("outputs") or {}).get(
                    "sorter_output_dirs", []
                )
            ]
            decisions[StageName.SORTING.value] = "reuse"
    if (
        StageName.SORTING in enabled
        and sorting_attempt
        and decisions[StageName.SORTING.value] != "reuse"
        and not settings.preprocess.overwrite
        and compatible(StageName.SORTING)
        and (
            StageName.PREPROCESS not in enabled
            or decisions.get(StageName.PREPROCESS.value) == "reuse"
        )
    ):
        candidates, manifest_path = _sorting_outputs(session_dir)
        try:
            validated = [item for path in candidates for item in _validate_phy_output(path)]
        except (OSError, RuntimeError, ValueError):
            candidates = []
        if candidates:
            sorting_outputs = candidates
            if manifest_path is not None:
                validated.append(str(manifest_path.resolve()))
            sorting_preprocess = dict(preprocess_result or {})
            sorting_preprocess.update(
                {
                    "sorter": settings.preprocess.sorter,
                    "sorter_output_dir": str(candidates[0]) if len(candidates) == 1 else None,
                    "sorter_output_dirs": [str(path) for path in candidates],
                    "sorter_partition_manifest_path": (
                        str(manifest_path.resolve()) if manifest_path is not None else None
                    ),
                }
            )
            _write_adopted_result(
                store,
                stage=StageName.SORTING,
                attempt=sorting_attempt[-1],
                outputs={
                    "preprocess_result": sorting_preprocess,
                    "sorter_output_dir": str(candidates[0]) if len(candidates) == 1 else "",
                    "sorter_output_dirs": [str(path) for path in candidates],
                    "sorter_partition_manifest_path": (
                        str(manifest_path.resolve()) if manifest_path is not None else ""
                    ),
                    "validated_paths": validated,
                },
                validated_paths=validated,
                source="validated existing Kilosort output",
            )
            decisions[StageName.SORTING.value] = "reuse"

    post_attempt = store.list_attempt_numbers(StageName.POSTPROCESS)
    if (
        StageName.POSTPROCESS in enabled
        and post_attempt
        and not settings.postprocess.overwrite
        and previous_compatible(StageName.POSTPROCESS)
        and previous is not None
        and (
            StageName.SORTING not in enabled
            or decisions.get(StageName.SORTING.value) == "reuse"
        )
    ):
        adopted_previous = _completed_previous_result(previous, StageName.POSTPROCESS)
        if adopted_previous is not None:
            prior_result, validated = adopted_previous
            _write_adopted_result(
                store,
                stage=StageName.POSTPROCESS,
                attempt=post_attempt[-1],
                outputs=dict(prior_result.get("outputs") or {}),
                validated_paths=validated,
                source=f"persistent Run {previous.run_dir.name}",
            )
            decisions[StageName.POSTPROCESS.value] = "reuse"
    if (
        StageName.POSTPROCESS in enabled
        and post_attempt
        and decisions[StageName.POSTPROCESS.value] != "reuse"
        and not settings.postprocess.overwrite
        and compatible(StageName.POSTPROCESS)
        and (
            StageName.SORTING not in enabled
            or decisions.get(StageName.SORTING.value) == "reuse"
        )
    ):
        if not sorting_outputs:
            sorting_outputs, _ = _sorting_outputs(session_dir)
        post_outputs = [path.parent / f"{path.name}_spi" for path in sorting_outputs]
        try:
            validated = [item for output in post_outputs for item in _validate_post_output(output)]
        except (OSError, RuntimeError, ValueError):
            validated = []
        if post_outputs and validated:
            _write_adopted_result(
                store,
                stage=StageName.POSTPROCESS,
                attempt=post_attempt[-1],
                outputs={
                    "postprocess_results": [
                        {
                            "sorting_phy_folder": str(sorting),
                            "output_folder": str(output),
                            "metrics_csv_path": str(output / "quality_metrics.csv"),
                        }
                        for sorting, output in zip(sorting_outputs, post_outputs, strict=True)
                    ],
                    "validated_paths": validated,
                },
                validated_paths=validated,
                source="validated existing postprocess output",
            )
            decisions[StageName.POSTPROCESS.value] = "reuse"

    atomic_write_json(
        store.run_dir / "resume_plan.json",
        {"created_at": utc_now(), "session_dir": str(session_dir), "stages": decisions},
    )
    store.rebuild_state()
    return decisions


def _selected_result(store: RunStore, stage: StageName) -> dict[str, Any] | None:
    view = store.derive_state()["stages"][stage.value]
    attempt = view.get("selected_attempt")
    if attempt is None:
        return None
    return store.read_attempt_fact(stage, int(attempt), "result.json")


def _selected_stage_summary(store: RunStore, stage: StageName) -> dict[str, Any] | None:
    view = store.derive_state()["stages"][stage.value]
    attempt = view.get("selected_attempt")
    if attempt is None:
        return None
    attempt_number = int(attempt)
    submitted = store.read_attempt_fact(stage, attempt_number, "submitted.json")
    return {
        "attempt": attempt_number,
        "status": view.get("status"),
        "spec": store.load_attempt_spec(stage, attempt_number).to_dict(),
        "submitted": submitted,
        "terminal_observation": store.latest_observation(stage, attempt_number),
        "result": store.read_attempt_fact(stage, attempt_number, "result.json"),
    }


def _conclusive_success(store: RunStore) -> bool:
    state = store.derive_state()
    if state.get("status") != StageStatus.COMPLETED.value:
        return False
    for stage in StageName:
        view = state["stages"][stage.value]
        if not view.get("enabled"):
            continue
        attempt = view.get("selected_attempt")
        if attempt is None:
            return False
        submitted = store.read_attempt_fact(stage, int(attempt), "submitted.json")
        if submitted is None:
            continue
        observation = store.latest_observation(stage, int(attempt))
        status = (observation or {}).get("status") or {}
        if not (status.get("terminal") is True and status.get("successful") is True):
            return False
    return True


def _log_sources(store: RunStore, sorting_dirs: list[Path]) -> list[tuple[str, Path]]:
    values: list[tuple[str, Path]] = []
    controller_log = store.run_dir / "controller.log"
    if controller_log.exists():
        values.append(("controller", controller_log))
    for stage in StageName:
        for attempt in store.list_attempt_numbers(stage):
            attempt_dir = store.attempt_dir(stage, attempt)
            for name in ("stdout.log", "stderr.log"):
                path = attempt_dir / name
                if path.exists():
                    values.append((f"{stage.value}/attempt-{attempt:03d}/{name}", path))
    for output in sorting_dirs:
        for name in ("kilosort.log", "matlab_run.log", "spikeinterface_log.json"):
            path = output / name
            if path.exists():
                values.append((f"{output.name}/{name}", path))
    return values


_WARNING_RE = re.compile(r"(?:^|\b)(?:warning|warn)(?:\b|:)", re.IGNORECASE)
_ERROR_RE = re.compile(
    r"(?:traceback \(most recent call last\)|\b(?:fatal|exception|segmentation fault|out of memory)\b|\berror:)",
    re.IGNORECASE,
)
_NEGATED_WARNING_RE = re.compile(
    r"(?:without|no|zero|0)\s+(?:detected\s+)?warnings?\b", re.IGNORECASE
)


def _diagnostics(sources: list[tuple[str, Path]]) -> tuple[list[str], list[str]]:
    warnings_found: list[str] = []
    errors_found: list[str] = []
    seen_warning_lines: set[str] = set()
    seen_error_lines: set[str] = set()
    for label, path in sources:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rendered = f"{label}: {stripped[:500]}"
            if _ERROR_RE.search(stripped):
                if stripped not in seen_error_lines:
                    errors_found.append(rendered)
                    seen_error_lines.add(stripped)
            elif _WARNING_RE.search(stripped) and not _NEGATED_WARNING_RE.search(stripped):
                if stripped not in seen_warning_lines:
                    warnings_found.append(rendered)
                    seen_warning_lines.add(stripped)
    return warnings_found, errors_found


def _sorting_dirs_from_results(store: RunStore) -> list[Path]:
    result = _selected_result(store, StageName.SORTING) or {}
    return [
        Path(value).expanduser().resolve()
        for value in (result.get("outputs") or {}).get("sorter_output_dirs", [])
        if str(value).strip()
    ]


def _write_pipeline_log(
    *,
    store: RunStore,
    destination: Path,
    sources: list[tuple[str, Path]],
    warnings_found: list[str],
    errors_found: list[str],
) -> None:
    state = store.derive_state()
    run = store.load_run()
    lines = [
        f"RUN {run.get('run_id')} status=completed",
        f"started_at={run.get('created_at', '')}",
        f"finished_at={utc_now()}",
        f"backend={run.get('resolved_backend', '')}",
        f"warnings_detected={len(warnings_found)} errors_detected={len(errors_found)}",
        "",
    ]
    for stage in StageName:
        view = state["stages"][stage.value]
        if view.get("enabled"):
            result = _selected_result(store, stage) or {}
            submitted = store.read_attempt_fact(
                stage, int(view["selected_attempt"]), "submitted.json"
            )
            jobs = [str(job.get("job_id")) for job in (submitted or {}).get("jobs", [])]
            observation = store.latest_observation(stage, int(view["selected_attempt"])) or {}
            telemetry = (observation.get("status") or {}).get("telemetry") or {}
            lines.append(
                f"STAGE {stage.value} status={view.get('status')} "
                f"attempt={view.get('selected_attempt')} "
                f"started_at={result.get('started_at', '')} "
                f"finished_at={result.get('finished_at', '')} "
                f"jobs={','.join(jobs) or '-'} "
                f"elapsed={telemetry.get('elapsed', '-')} "
                f"max_rss={telemetry.get('max_rss', '-')}"
            )
    for label, path in sources:
        if path.name in {"kilosort.log", "matlab_run.log", "spikeinterface_log.json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"[log read failed: {exc}]\n"
        lines.extend(("", f"===== {label} =====", text.rstrip()))
    if warnings_found:
        lines.extend(("", "===== DETECTED WARNINGS =====", *warnings_found))
    if errors_found:
        lines.extend(("", "===== DETECTED ERRORS =====", *errors_found))
    atomic_write_text(destination, "\n".join(lines).rstrip() + "\n")


def _remove_success_helpers(store: RunStore, sorting_dirs: list[Path]) -> None:
    for output in sorting_dirs:
        matlab_log = output / "matlab_run.log"
        kilosort_log = output / "kilosort.log"
        if matlab_log.exists() and kilosort_log.exists():
            try:
                if hashlib.sha256(matlab_log.read_bytes()).digest() == hashlib.sha256(
                    kilosort_log.read_bytes()
                ).digest():
                    matlab_log.unlink()
            except OSError:
                pass
        for name in (
            "spikeinterface_log.json",
            "spikeinterface_params.json",
            "spikeinterface_recording.json",
            "run_kilosort.sh",
            "sorter_config_source.yaml",
            "sorter_config_source.yml",
            "sorter_config_source.json",
        ):
            (output / name).unlink(missing_ok=True)
    state = store.derive_state()
    has_historical_attempts = False
    for stage in StageName:
        selected = state["stages"][stage.value].get("selected_attempt")
        attempts = store.list_attempt_numbers(stage)
        has_historical_attempts = has_historical_attempts or any(
            selected is None or attempt != int(selected) for attempt in attempts
        )
        if selected is None:
            continue
        attempt_dir = store.attempt_dir(stage, int(selected))
        for name in ("stdout.log", "stderr.log"):
            (attempt_dir / name).unlink(missing_ok=True)
    if not has_historical_attempts:
        atomic_write_text(
            store.run_dir / "controller.log", "Run finalized; see session preprocess.log.\n"
        )


def finalize_successful_run(store: RunStore) -> dict[str, Any] | None:
    """Write the compact final record once all enabled jobs conclusively succeed."""

    final_path = store.run_dir / "final.json"
    if final_path.exists():
        return read_json(final_path)
    if not _conclusive_success(store):
        return None
    settings = settings_for_store(store)
    session_dir = session_output_dir(settings)
    session_dir.mkdir(parents=True, exist_ok=True)
    prior_record: dict[str, Any] = {}
    prior_record_path = session_dir / "preprocess_run.yaml"
    if prior_record_path.exists():
        try:
            import yaml

            prior_record = yaml.safe_load(prior_record_path.read_text(encoding="utf-8")) or {}
        except (OSError, TypeError, ValueError):
            prior_record = {}
    sorting_dirs = _sorting_dirs_from_results(store)
    if not sorting_dirs:
        sorting_dirs = [
            Path(value).expanduser().resolve()
            for value in prior_record.get("sorting_output_dirs", [])
            if str(value).strip()
        ]
    sources = _log_sources(store, sorting_dirs)
    warnings_found, errors_found = _diagnostics(sources)
    stage_results = dict(prior_record.get("stages") or {})
    enabled_stage_names: list[str] = []
    for stage in StageName:
        if store.derive_state()["stages"][stage.value].get("enabled"):
            stage_results[stage.value] = _selected_stage_summary(store, stage)
            enabled_stage_names.append(stage.value)
    fingerprints = dict(prior_record.get("stage_fingerprints") or {})
    current_fingerprints = stage_fingerprints(store.load_analysis())
    for stage_name in enabled_stage_names:
        fingerprints[stage_name] = current_fingerprints[stage_name]
    record = {
        "schema_version": 1,
        "run_id": store.load_run().get("run_id"),
        "status": (
            "completed_with_log_findings"
            if warnings_found or errors_found
            else "completed_clean"
        ),
        "started_at": store.load_run().get("created_at"),
        "finished_at": utc_now(),
        "backend": store.load_run().get("resolved_backend"),
        "source_basepath": str(
            settings.preprocess_source_path or prior_record.get("source_basepath") or ""
        ),
        "session_output_dir": str(session_dir),
        "analysis": store.load_analysis().to_dict(),
        "stage_fingerprints": fingerprints,
        "execution": store.load_execution().to_dict(),
        "provenance": {
            name: read_json(store.run_dir / "snapshots" / f"{name}.json")
            for name in ("git", "environment", "inputs")
            if (store.run_dir / "snapshots" / f"{name}.json").exists()
        },
        "stages": stage_results,
        "warnings": {"count": len(warnings_found), "items": warnings_found},
        "errors": {"count": len(errors_found), "items": errors_found},
        "sorting_output_dirs": [str(path) for path in sorting_dirs],
    }
    import yaml

    atomic_write_text(
        session_dir / "preprocess_run.yaml",
        yaml.safe_dump(record, sort_keys=False, allow_unicode=True),
    )
    _write_pipeline_log(
        store=store,
        destination=session_dir / "preprocess.log",
        sources=sources,
        warnings_found=warnings_found,
        errors_found=errors_found,
    )
    atomic_write_json(final_path, _json_safe(record))
    _remove_success_helpers(store, sorting_dirs)
    release_session_claim(store)
    return record
