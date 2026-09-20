from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
import getpass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import traceback
from typing import Any
import uuid

from .backends import LocalBackend, SlurmBackend, SlurmCapabilities, detect_slurm_capabilities
from .models import (
    AnalysisConfig,
    AttemptSpec,
    BackendName,
    ExecutionConfig,
    JobRef,
    RequestedBackend,
    ResourceSpec,
    StageName,
    StageStatus,
)
from .store import (
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json,
    short_file_lock,
    utc_now,
)


STAGE_ORDER = (StageName.PREPROCESS, StageName.SORTING, StageName.POSTPROCESS)


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"


def resolve_backend(
    requested: RequestedBackend,
    *,
    require_sacct: bool,
    capabilities: SlurmCapabilities | None = None,
) -> tuple[BackendName, SlurmCapabilities]:
    if requested == RequestedBackend.LOCAL:
        capabilities = capabilities or SlurmCapabilities(
            sbatch=None,
            squeue=None,
            scancel=None,
            sacct=None,
            controller_reachable=False,
            accounting_reachable=False,
            errors=(),
        )
        return BackendName.LOCAL, capabilities
    capabilities = capabilities or detect_slurm_capabilities()
    if capabilities.usable(require_sacct=require_sacct):
        return BackendName.SLURM, capabilities
    if requested == RequestedBackend.AUTO:
        return BackendName.LOCAL, capabilities
    detail = "; ".join(capabilities.errors) or "required Slurm capabilities are unavailable"
    raise RuntimeError(f"Explicit Slurm backend is unavailable: {detail}")


def _analysis_settings_snapshot(settings: Any, *, sorter_snapshot_path: Path) -> dict[str, Any]:
    payload = json.loads(settings.to_json())
    preprocess = dict(payload.get("preprocess", {}))
    preprocess.pop("preprocess_worker_count", None)
    preprocess.pop("sorter_worker_count", None)
    preprocess.pop("matlab_path", None)
    if preprocess.get("run_sorter") and preprocess.get("sorter_config_path"):
        preprocess["sorter_config_path"] = str(sorter_snapshot_path.resolve())
    payload["preprocess"] = preprocess
    postprocess = dict(payload.get("postprocess", {}))
    postprocess.pop("worker_count", None)
    payload["postprocess"] = postprocess
    payload.pop("execution", None)
    return payload


def _resolve_sorter_config_source(settings: Any) -> Path | None:
    if not settings.preprocess.run_sorter or not settings.preprocess.sorter_config_path:
        return None
    from src.preprocess.gui.config_model import REPO_ROOT

    path = Path(settings.preprocess.sorter_config_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Sorter config snapshot source does not exist: {path}")
    return path


def _git_provenance(repo_root: Path) -> dict[str, Any]:
    def _git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "recorded_at": utc_now(),
        "repository": str(repo_root.resolve()),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _environment_provenance() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "spikeinterface", "PySide6", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "recorded_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "packages": packages,
    }


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sorting_input_identity(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.expanduser().resolve()
    payload: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_dir()}
    def _add_output(folder: Path, label: str) -> None:
        files: dict[str, Any] = {"path": str(folder.resolve())}
        for name in ("params.py", "spike_times.npy", "spike_clusters.npy", "templates.npy"):
            child = folder / name
            if child.exists():
                stat = child.stat()
                files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            else:
                files[name] = None
        payload[label] = files

    if resolved.is_dir():
        _add_output(resolved, "primary_output")
        manifest = resolved.parent / "sorter_partition_manifest.json"
        if manifest.exists():
            stat = manifest.stat()
            payload["partition_manifest"] = {
                "path": str(manifest.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(manifest),
            }
            try:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                for index, item in enumerate(manifest_payload.get("partitions", [])):
                    if isinstance(item, dict) and item.get("output_folder"):
                        _add_output(
                            Path(str(item["output_folder"])).expanduser(),
                            f"partition_{index:03d}",
                        )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                payload["partition_manifest_invalid"] = True
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_metadata_identity(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return ""
    stat = resolved.stat()
    payload = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _recursive_input_roots(settings: Any) -> list[Path]:
    selected = [
        Path(value).expanduser()
        for value in getattr(settings, "multi_day_selected_subepoch_paths", [])
        if str(value).strip()
    ]
    sessions = [
        Path(value).expanduser()
        for value in getattr(settings, "multi_day_session_paths", [])
        if str(value).strip()
    ]
    if bool(getattr(settings, "multi_day_enabled", False)):
        return selected or sessions
    source = getattr(settings, "preprocess_source_path", None)
    return [Path(source).expanduser()] if source is not None else []


def _input_provenance(settings: Any) -> dict[str, Any]:
    scan_started_at = utc_now()
    recursive_roots = _recursive_input_roots(settings)
    paths: list[Path | None] = [
        settings.resolved_xml_path(),
        settings.resolved_chanmap_path(),
        *recursive_roots,
    ]
    if not bool(getattr(settings, "multi_day_enabled", False)):
        paths[:0] = [settings.basepath_path, settings.preprocess_source_path]
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        resolved_text = str(Path(path).resolve())
        if resolved_text in seen:
            continue
        seen.add(resolved_text)
        item: dict[str, Any] = {"path": resolved_text, "exists": Path(path).exists()}
        if Path(path).exists():
            stat = Path(path).stat()
            item.update(
                {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "is_dir": Path(path).is_dir(),
                }
            )
        values.append(item)
    # Acquisition sidecars affect event extraction, artifact removal and time
    # alignment just as materially as amplifier samples.  Record their
    # metadata so a changed TTL/analog/time input cannot silently reuse a
    # preprocess result.
    input_names = {
        "structure.oebin",
        "amplifier.dat",
        "continuous.dat",
        "analogin.dat",
        "digitalin.dat",
        "auxiliary.dat",
        "supply.dat",
        "time.dat",
        "timestamps.npy",
        "sample_numbers.npy",
        "states.npy",
        "full_words.npy",
        "sync_messages.txt",
        "events.json",
        "settings.xml",
    }
    input_suffixes = {".rhd", ".xml"}
    for root in recursive_roots:
        if root is None or not Path(root).is_dir():
            continue
        try:
            children = (
                child
                for child in Path(root).rglob("*")
                if child.is_file()
                and (
                    child.name.lower() in input_names
                    or "ttl" in child.name.lower()
                    or child.suffix.lower() in input_suffixes
                )
            )
            for child in children:
                resolved_text = str(child.resolve())
                if resolved_text in seen:
                    continue
                seen.add(resolved_text)
                stat = child.stat()
                values.append(
                    {
                        "path": resolved_text,
                        "exists": True,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "ctime_ns": stat.st_ctime_ns,
                        "is_dir": False,
                    }
                )
        except OSError:
            continue
    for item in values:
        path = Path(item["path"])
        if item.get("is_dir") is False and path.suffix.lower() in {".xml", ".json", ".oebin", ".txt"}:
            item["sha256"] = _file_sha256(path)
    return {
        "scan_started_at": scan_started_at,
        "recorded_at": utc_now(),
        "inputs": values,
    }


def enabled_stages_for(settings: Any, mode: str) -> list[StageName]:
    if mode == "all":
        stages = [StageName.PREPROCESS]
        if settings.preprocess.run_sorter and str(settings.preprocess.sorter or "").lower() != "disabled":
            stages.append(StageName.SORTING)
            stages.append(StageName.POSTPROCESS)
        return stages
    if mode == "preprocess":
        stages = [StageName.PREPROCESS]
        if settings.preprocess.run_sorter and str(settings.preprocess.sorter or "").lower() != "disabled":
            stages.append(StageName.SORTING)
        return stages
    if mode == "postprocess":
        return [StageName.POSTPROCESS]
    raise ValueError(f"Persistent execution does not support mode={mode!r}")


def create_run(
    *,
    settings: Any,
    execution: ExecutionConfig,
    mode: str,
    capabilities: SlurmCapabilities | None = None,
) -> Path:
    from src.preprocess.gui.config_model import (
        resolve_existing_session_settings, prepare_session_sorter_config, save_session_settings,
    )

    resolve_existing_session_settings(settings)
    resolved_xml = settings.resolved_xml_path()
    resolved_chanmap = settings.postprocess_chanmap_path() if mode == "postprocess" else settings.resolved_chanmap_path()
    if resolved_xml is not None:
        settings.xml_path = str(resolved_xml.resolve())
    if resolved_chanmap is not None and resolved_chanmap.exists():
        settings.chanmap_path = str(resolved_chanmap.resolve())
    if mode == "postprocess":
        resolved_dat = settings.postprocess_dat_path()
        if resolved_dat is not None:
            settings.postprocess.dat_path = str(resolved_dat.resolve())
        resolved_sorting = settings.postprocess_sorting_folder()
        if resolved_sorting is not None:
            settings.postprocess.sorting_phy_folder = str(resolved_sorting.resolve())
            candidate_roots = [settings.local_output_dir, resolved_sorting.resolve().parent]
            manifest_root = next(
                (
                    root.resolve()
                    for root in candidate_roots
                    if root is not None and (root / "sorter_partition_manifest.json").exists()
                ),
                None,
            )
            if manifest_root is not None:
                settings.postprocess.sorting_search_root = str(manifest_root)
    elif mode == "all":
        settings.postprocess.sorting_phy_folder = ""
        settings.postprocess.sorting_search_root = ""
    execution.validate()
    resolved, capabilities = resolve_backend(
        execution.requested_backend,
        require_sacct=execution.require_sacct,
        capabilities=capabilities,
    )
    if resolved != execution.resolved_backend:
        raise ValueError(
            f"ExecutionConfig resolved_backend={execution.resolved_backend.value} does not match "
            f"capability resolution {resolved.value}"
        )
    workspace = Path(execution.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not os.access(workspace, os.W_OK):
        raise PermissionError(f"Persistent workspace is not writable: {workspace}")
    pipeline_root = workspace if workspace.name == ".pipeline" else workspace / ".pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    run_id = _run_id()
    run_dir = pipeline_root / run_id
    from .session import abandon_session_claim, acquire_session_claim, session_output_dir

    claim_path = acquire_session_claim(
        pipeline_root=pipeline_root,
        session_dir=session_output_dir(settings),
        run_id=run_id,
        run_dir=run_dir,
    )
    try:
        claim = read_json(claim_path)
        prepare_session_sorter_config(settings)
        sorter_source = _resolve_sorter_config_source(settings)
        sorter_source_bytes = sorter_source.read_bytes() if sorter_source is not None else None
    except BaseException:
        abandon_session_claim(claim_path=claim_path, run_id=run_id)
        raise
    sorter_snapshot = run_dir / "snapshots" / (
        sorter_source.name if sorter_source is not None else "sorter-config.yaml"
    )
    from src.preprocess.gui.config_model import REPO_ROOT

    artifact_sha256 = {"xml": _file_sha256(settings.resolved_xml_path())}
    if sorter_source_bytes is not None:
        artifact_sha256["sorter_config"] = hashlib.sha256(sorter_source_bytes).hexdigest()
    if mode == "postprocess":
        chanmap_sha256 = _file_sha256(settings.postprocess_chanmap_path())
        if chanmap_sha256:
            artifact_sha256["chanmap"] = chanmap_sha256
        artifact_sha256["postprocess_sorting_input"] = _sorting_input_identity(
            settings.postprocess_sorting_folder()
        )
        dat_identity = _file_metadata_identity(settings.postprocess_dat_path())
        if dat_identity:
            artifact_sha256["postprocess_dat_input"] = dat_identity
    artifact_sha256 = {key: value for key, value in artifact_sha256.items() if value}
    analysis = AnalysisConfig.create(
        _analysis_settings_snapshot(settings, sorter_snapshot_path=sorter_snapshot),
        artifact_sha256=artifact_sha256,
    )
    enabled = enabled_stages_for(settings, mode)
    from .session import stage_fingerprints

    run = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "mode": mode,
        "requested_backend": execution.requested_backend.value,
        "resolved_backend": execution.resolved_backend.value,
        "enabled_stages": [stage.value for stage in enabled],
        "analysis_sha256": analysis.sha256,
        "stage_fingerprints": stage_fingerprints(analysis),
        "slurm_capabilities": capabilities.to_dict(),
        "repository_root": str(REPO_ROOT.resolve()),
        "session_output_dir": str(session_output_dir(settings)),
        "session_claim_path": str(claim_path),
        "previous_run_dir": str(claim.get("previous_run_dir") or ""),
        "previous_run_dirs": list(claim.get("previous_run_dirs") or []),
    }
    store = RunStore(run_dir)
    try:
        store.initialize(run=run, analysis=analysis, execution=execution)
        if sorter_source_bytes is not None:
            sorter_snapshot.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(sorter_snapshot, sorter_source_bytes)

        atomic_write_json(run_dir / "snapshots" / "git.json", _git_provenance(REPO_ROOT))
        atomic_write_json(run_dir / "snapshots" / "environment.json", _environment_provenance())
        atomic_write_json(run_dir / "snapshots" / "inputs.json", _input_provenance(settings))
        create_initial_attempts(store)
        if StageName.PREPROCESS in enabled:
            _cleanup_preprocess_partials_for_store(
                store,
                fact_name="startup_partial_cleanup.json",
                strict=True,
            )
        from .session import adopt_existing_outputs

        decisions = adopt_existing_outputs(store)
        if (
            StageName.PREPROCESS in enabled
            and decisions.get(StageName.PREPROCESS.value) != "reuse"
            and settings.preprocess_source_path is None
        ):
            raise ValueError(
                "The selected processed session needs a preprocess rerun, but its raw "
                "source basepath cannot be recovered from persistent or legacy metadata."
            )
        save_session_settings(settings, execution=execution)
    except BaseException:
        abandon_session_claim(claim_path=claim_path, run_id=run_id)
        raise
    return run_dir


def _selected_attempt_number(store: RunStore, stage: StageName) -> int | None:
    state = store.derive_state()
    value = state["stages"][stage.value]["selected_attempt"]
    return int(value) if value is not None else None


def create_initial_attempts(store: RunStore) -> None:
    run = store.load_run()
    analysis = store.load_analysis()
    execution = store.load_execution()
    enabled = [StageName(value) for value in run["enabled_stages"]]
    created: dict[str, int] = {}
    for stage in STAGE_ORDER:
        if stage not in enabled:
            continue
        upstream: dict[str, int] = {}
        if stage == StageName.SORTING and StageName.PREPROCESS.value in created:
            upstream[StageName.PREPROCESS.value] = created[StageName.PREPROCESS.value]
        elif stage == StageName.POSTPROCESS:
            if StageName.PREPROCESS.value in created:
                upstream[StageName.PREPROCESS.value] = created[StageName.PREPROCESS.value]
            if StageName.SORTING.value in created:
                upstream[StageName.SORTING.value] = created[StageName.SORTING.value]
        attempt = store.next_attempt_number(stage)
        spec = AttemptSpec(
            run_id=run["run_id"],
            stage=stage,
            attempt=attempt,
            backend=execution.resolved_backend,
            resources=execution.resource_for(stage),
            analysis_sha256=analysis.sha256,
            upstream_attempts=upstream,
        )
        store.create_attempt(spec)
        created[stage.value] = attempt


def _backend_for(store: RunStore):
    execution = store.load_execution()
    if execution.resolved_backend == BackendName.LOCAL:
        return LocalBackend()
    capabilities_data = store.load_run().get("slurm_capabilities", {})
    capabilities = SlurmCapabilities(
        sbatch=capabilities_data.get("sbatch"),
        squeue=capabilities_data.get("squeue"),
        scancel=capabilities_data.get("scancel"),
        sacct=capabilities_data.get("sacct"),
        controller_reachable=bool(capabilities_data.get("controller_reachable")),
        accounting_reachable=bool(capabilities_data.get("accounting_reachable")),
        errors=tuple(capabilities_data.get("errors", [])),
    )
    return SlurmBackend(capabilities=capabilities)


def _submitted_jobs(store: RunStore, stage: StageName, attempt: int) -> list[JobRef]:
    fact = store.read_attempt_fact(stage, attempt, "submitted.json")
    if fact is None:
        return []
    return [JobRef.from_dict(value) for value in fact.get("jobs", [])]


def _preprocess_jobs_are_inactive(store: RunStore, backend: Any) -> bool:
    for attempt in store.list_attempt_numbers(StageName.PREPROCESS):
        for job in _submitted_jobs(store, StageName.PREPROCESS, attempt):
            try:
                status = backend.status(job)
            except BaseException:
                return False
            if not status.terminal:
                return False
            if (
                job.backend != BackendName.LOCAL
                and status.successful is None
                and status.state.lower() not in {"cancelled", "canceled"}
            ):
                return False
    return True


def _cleanup_preprocess_partials_for_store(
    store: RunStore,
    *,
    fact_name: str,
    strict: bool,
) -> dict[str, Any]:
    from .session import (
        cleanup_preprocess_binary_partials,
        session_output_dir,
        settings_for_store,
    )

    settings = settings_for_store(store)
    report = cleanup_preprocess_binary_partials(
        session_output_dir(settings),
        settings.basename,
    )
    report = {**report, "cleaned_at": utc_now()}
    if report["removed"]:
        print(
            "Removed incomplete preprocess binaries after confirmed worker stop: "
            f"files={len(report['removed'])}, bytes={report['removed_bytes']}"
        )
        for removed_path in report["removed"]:
            print(f"  removed: {removed_path}")
    for cleanup_error in report["errors"]:
        print(
            "Warning: could not remove incomplete preprocess binary: "
            f"{cleanup_error['path']}: {cleanup_error['error']}"
        )
    attempt = _selected_attempt_number(store, StageName.PREPROCESS)
    if attempt is not None:
        path = store.attempt_dir(StageName.PREPROCESS, attempt) / fact_name
        if not path.exists():
            store.write_attempt_fact(
                StageName.PREPROCESS,
                attempt,
                fact_name,
                report,
            )
    if strict and report["errors"]:
        details = "; ".join(
            f"{item['path']}: {item['error']}" for item in report["errors"]
        )
        raise RuntimeError(
            "Cannot safely start preprocessing while incomplete binary outputs "
            f"cannot be removed: {details}"
        )
    return report


def _dependency_jobs(store: RunStore, spec: AttemptSpec) -> list[JobRef]:
    jobs: list[JobRef] = []
    for stage_name, attempt in spec.upstream_attempts.items():
        completed = store.read_attempt_fact(StageName(stage_name), attempt, "result.json")
        if completed is not None and completed.get("status") == StageStatus.COMPLETED.value:
            continue
        jobs.extend(_submitted_jobs(store, StageName(stage_name), attempt))
    return jobs


def _mark_downstream_blocked(store: RunStore, failed_stage: StageName, reason: str) -> None:
    failed_index = STAGE_ORDER.index(failed_stage)
    for stage in STAGE_ORDER[failed_index + 1 :]:
        attempt = _selected_attempt_number(store, stage)
        if attempt is None:
            continue
        directory = store.attempt_dir(stage, attempt)
        if any((directory / name).exists() for name in ("result.json", "failure.json", "blocked.json")):
            continue
        store.write_attempt_fact(
            stage,
            attempt,
            "blocked.json",
            {"blocked_at": utc_now(), "upstream_stage": failed_stage.value, "reason": reason},
        )


def submit_run(run_dir: Path | str) -> dict[str, Any]:
    store = RunStore(run_dir)
    backend = _backend_for(store)
    run = store.load_run()
    enabled = [StageName(value) for value in run["enabled_stages"]]
    is_local = store.load_execution().resolved_backend == BackendName.LOCAL
    for stage in STAGE_ORDER:
        if stage not in enabled:
            continue
        submitted_here = False
        jobs: list[JobRef] = []
        attempt: int | None = None
        action = "continue"
        with short_file_lock(store.run_dir / ".controller.lock", timeout=60.0):
            attempt = _selected_attempt_number(store, stage)
            if attempt is None:
                action = "continue"
            else:
                stage_status = store.derive_state()["stages"][stage.value]["status"]
                directory = store.attempt_dir(stage, attempt)
                if stage_status == StageStatus.COMPLETED.value:
                    action = "continue"
                elif stage_status in {
                    StageStatus.FAILED.value,
                    StageStatus.CANCELLED.value,
                    StageStatus.BLOCKED.value,
                    StageStatus.LOST.value,
                }:
                    action = "stop"
                elif (directory / "submitted.json").exists():
                    jobs = _submitted_jobs(store, stage, attempt)
                    action = "existing"
                elif (directory / "submission_intent.json").exists():
                    raise RuntimeError(
                        f"Refusing to resubmit {stage.value} attempt-{attempt:03d}: "
                        "a prior Slurm submission intent has an ambiguous or incomplete outcome"
                    )
                else:
                    spec = store.load_attempt_spec(stage, attempt)
                    dependencies = _dependency_jobs(store, spec)
                    if is_local and spec.upstream_attempts:
                        upstream_states = store.derive_state()["stages"]
                        incomplete = [
                            upstream
                            for upstream in spec.upstream_attempts
                            if upstream_states[upstream]["status"]
                            != StageStatus.COMPLETED.value
                        ]
                        if incomplete:
                            terminal_failure = any(
                                upstream_states[upstream]["status"]
                                in {
                                    StageStatus.FAILED.value,
                                    StageStatus.CANCELLED.value,
                                    StageStatus.BLOCKED.value,
                                    StageStatus.LOST.value,
                                }
                                for upstream in incomplete
                            )
                            if terminal_failure:
                                store.write_attempt_fact(
                                    stage,
                                    attempt,
                                    "blocked.json",
                                    {
                                        "blocked_at": utc_now(),
                                        "reason": "Upstream Stage did not complete successfully",
                                    },
                                )
                            action = "stop"
                        else:
                            action = "submit"
                    else:
                        action = "submit"
                    if action == "submit":
                        try:
                            jobs = backend.submit(
                                run_dir=store.run_dir,
                                attempt_spec=spec,
                                dependency_job_refs=dependencies,
                            )
                        except BaseException as exc:
                            submission_failure = store.read_attempt_fact(
                                stage, attempt, "submission_failure.json"
                            )
                            acceptance_ambiguous = bool(
                                (submission_failure or {}).get(
                                    "acceptance_ambiguous", False
                                )
                            )
                            if not acceptance_ambiguous and not (
                                directory / "failure.json"
                            ).exists():
                                store.write_attempt_fact(
                                    stage,
                                    attempt,
                                    "failure.json",
                                    {
                                        "stage": stage.value,
                                        "attempt": attempt,
                                        "status": StageStatus.FAILED.value,
                                        "finished_at": utc_now(),
                                        "type": type(exc).__name__,
                                        "message": str(exc),
                                        "traceback": traceback.format_exc(),
                                    },
                                )
                            _mark_downstream_blocked(store, stage, str(exc))
                            store.rebuild_state()
                            raise
                        store.write_attempt_fact(
                            stage,
                            attempt,
                            "submitted.json",
                            {
                                "stage": stage.value,
                                "attempt": attempt,
                                "submitted_at": utc_now(),
                                "jobs": [job.to_dict() for job in jobs],
                            },
                        )
                        store.rebuild_state()
                        submitted_here = True
        if action == "stop":
            break
        if action == "continue":
            continue
        if is_local:
            assert attempt is not None
            if not submitted_here:
                for existing_job in jobs:
                    status = backend.status(existing_job)
                    store.record_observation(
                        stage, attempt, job=existing_job, status=status.to_dict()
                    )
                refreshed = store.rebuild_state()["stages"][stage.value]["status"]
                if refreshed == StageStatus.COMPLETED.value:
                    continue
                # Another detached controller may still own this worker. Do not
                # submit downstream Local work until its result is canonical.
                return store.rebuild_state()
            job = jobs[0]
            return_code = backend.wait(job)  # type: ignore[attr-defined]
            status = backend.status(job)
            store.record_observation(stage, attempt, job=job, status=status.to_dict())
            store.rebuild_state()
            result = store.read_attempt_fact(stage, attempt, "result.json")
            if return_code != 0 or result is None:
                reason = (
                    (store.read_attempt_fact(stage, attempt, "failure.json") or {}).get("message")
                    or f"Local Stage worker exited with code {return_code}"
                )
                _mark_downstream_blocked(store, stage, str(reason))
                break
    state = store.rebuild_state()
    if state.get("status") == StageStatus.COMPLETED.value:
        from .session import finalize_successful_run

        finalize_successful_run(store)
    return store.rebuild_state()


def _attempt_has_terminal_fact(directory: Path) -> bool:
    return any(
        (directory / filename).exists()
        for filename in ("result.json", "failure.json", "cancel_confirmed.json")
    )


def _write_cancel_confirmation(
    store: RunStore, stage: StageName, attempt: int, *, state: str
) -> None:
    directory = store.attempt_dir(stage, attempt)
    if not (directory / "cancel_confirmed.json").exists():
        store.write_attempt_fact(
            stage,
            attempt,
            "cancel_confirmed.json",
            {"finished_at": utc_now(), "state": state},
        )


def _cancel_attempt_jobs(
    store: RunStore,
    backend: Any,
    stage: StageName,
    attempt: int,
) -> bool:
    """Request cancellation and return whether any job may still be active."""
    directory = store.attempt_dir(stage, attempt)
    if _attempt_has_terminal_fact(directory):
        return False
    jobs = _submitted_jobs(store, stage, attempt)
    if not (directory / "cancel_requested.json").exists():
        store.write_attempt_fact(
            stage,
            attempt,
            "cancel_requested.json",
            {"requested_at": utc_now(), "requested_by": getpass.getuser()},
        )
    if not jobs:
        submission_failure = store.read_attempt_fact(
            stage, attempt, "submission_failure.json"
        )
        if (directory / "submission_intent.json").exists() or bool(
            (submission_failure or {}).get("acceptance_ambiguous", False)
        ):
            return True
        if not (directory / "started.json").exists():
            _write_cancel_confirmation(
                store, stage, attempt, state="Attempt was never submitted"
            )
            return False
        return True

    active_remaining = False
    for job in jobs:
        try:
            backend.cancel(job)
        except BaseException as exc:
            active_remaining = True
            store.record_observation(
                stage,
                attempt,
                job=job,
                status={
                    "state": "cancel_failed",
                    "terminal": False,
                    "successful": None,
                    "reason": str(exc),
                    "telemetry": {},
                },
            )
            continue
        try:
            status = backend.status(job)
        except BaseException as exc:
            active_remaining = True
            store.record_observation(
                stage,
                attempt,
                job=job,
                status={
                    "state": "cancel_pending",
                    "terminal": False,
                    "successful": None,
                    "reason": f"Cancellation status query failed: {exc}",
                    "telemetry": {},
                },
            )
            continue
        store.record_observation(stage, attempt, job=job, status=status.to_dict())
        if job.backend == BackendName.LOCAL and status.state == "unknown":
            _write_cancel_confirmation(
                store, stage, attempt, state="local process terminated"
            )
        elif status.terminal and status.state.lower() in {"cancelled", "canceled"}:
            _write_cancel_confirmation(store, stage, attempt, state=status.state)
        elif not status.terminal or status.successful is None:
            active_remaining = True
    return active_remaining


def reconcile_run(run_dir: Path | str) -> dict[str, Any]:
    store = RunStore(run_dir)
    with short_file_lock(store.run_dir / ".controller.lock", timeout=60.0):
        return _reconcile_run_locked(store)


def _reconcile_run_locked(store: RunStore) -> dict[str, Any]:
    backend = _backend_for(store)
    state = store.derive_state()
    for stage in STAGE_ORDER:
        stage_view = state["stages"][stage.value]
        selected_attempt = stage_view["selected_attempt"]
        for attempt_view in stage_view["attempts"]:
            attempt = int(attempt_view["attempt"])
            directory = store.attempt_dir(stage, attempt)
            jobs = _submitted_jobs(store, stage, attempt)
            latest_observation = store.latest_observation(stage, attempt)
            latest_status = (latest_observation or {}).get("status") or {}
            conclusive_terminal = bool(latest_status.get("terminal", False)) and (
                latest_status.get("successful") is not None
                or str(latest_status.get("state", "")).lower()
                in {"cancelled", "canceled"}
            )
            if not jobs or conclusive_terminal:
                continue
            for job in jobs:
                try:
                    status = backend.status(job)
                except BaseException as exc:
                    status_dict = {
                        "state": "unknown",
                        "terminal": True,
                        "successful": None,
                        "reason": f"Backend status query failed: {exc}",
                        "telemetry": {},
                    }
                else:
                    status_dict = status.to_dict()
                store.record_observation(stage, attempt, job=job, status=status_dict)
                if (
                    (directory / "cancel_requested.json").exists()
                    and status_dict.get("terminal")
                    and str(status_dict.get("state", "")).lower()
                    in {"cancelled", "canceled"}
                ):
                    _write_cancel_confirmation(
                        store, stage, attempt, state=str(status_dict["state"])
                    )
        refreshed_stage_view = store.derive_state()["stages"][stage.value]
        if selected_attempt is not None and refreshed_stage_view["status"] in {
            StageStatus.FAILED.value,
            StageStatus.CANCELLED.value,
            StageStatus.LOST.value,
        }:
            _mark_downstream_blocked(store, stage, f"{stage.value} did not complete successfully")
    preprocess_attempt = _selected_attempt_number(store, StageName.PREPROCESS)
    if preprocess_attempt is not None:
        preprocess_dir = store.attempt_dir(StageName.PREPROCESS, preprocess_attempt)
        if (
            (preprocess_dir / "cancel_requested.json").exists()
            and _preprocess_jobs_are_inactive(store, backend)
        ):
            _cleanup_preprocess_partials_for_store(
                store,
                fact_name="reconcile_partial_cleanup.json",
                strict=False,
            )
    state = store.rebuild_state()
    if state.get("status") == StageStatus.COMPLETED.value:
        from .session import finalize_successful_run

        finalize_successful_run(store)
    return store.rebuild_state()


def cancel_run(run_dir: Path | str, *, stage: StageName | None = None) -> dict[str, Any]:
    store = RunStore(run_dir)
    with short_file_lock(store.run_dir / ".controller.lock", timeout=60.0):
        return _cancel_run_locked(store, stage=stage)


def _cancel_run_locked(
    store: RunStore, *, stage: StageName | None = None
) -> dict[str, Any]:
    backend = _backend_for(store)
    state = store.derive_state()
    start_index = STAGE_ORDER.index(stage) if stage is not None else 0
    cancellation_pending = False
    for target in STAGE_ORDER[start_index:]:
        for attempt_view in state["stages"][target.value]["attempts"]:
            cancellation_pending = (
                _cancel_attempt_jobs(
                    store,
                    backend,
                    target,
                    int(attempt_view["attempt"]),
                )
                or cancellation_pending
            )
    preprocess_targeted = stage is None or stage == StageName.PREPROCESS
    if (
        preprocess_targeted
        and not cancellation_pending
        and _preprocess_jobs_are_inactive(store, backend)
    ):
        _cleanup_preprocess_partials_for_store(
            store,
            fact_name="cancel_partial_cleanup.json",
            strict=False,
        )
    return store.rebuild_state()


def prepare_retry(
    run_dir: Path | str,
    *,
    stage: StageName,
    resource_override: ResourceSpec | None = None,
) -> list[AttemptSpec]:
    store = RunStore(run_dir)
    with short_file_lock(store.run_dir / ".controller.lock", timeout=60.0):
        return _prepare_retry_locked(
            store,
            stage=stage,
            resource_override=resource_override,
        )


def _prepare_retry_locked(
    store: RunStore,
    *,
    stage: StageName,
    resource_override: ResourceSpec | None = None,
) -> list[AttemptSpec]:
    run = store.load_run()
    analysis = store.load_analysis()
    execution = store.load_execution()
    enabled = [StageName(value) for value in run["enabled_stages"]]
    if stage not in enabled:
        raise ValueError(f"Stage {stage.value} is not enabled for this Run")
    if resource_override is not None:
        resource_override.validate(stage=stage)
    state = store.derive_state()
    current = state["stages"][stage.value]
    if current["status"] not in {
        StageStatus.FAILED.value,
        StageStatus.CANCELLED.value,
        StageStatus.BLOCKED.value,
    }:
        raise ValueError(f"Stage {stage.value} is not retryable from state {current['status']}")

    start_index = STAGE_ORDER.index(stage)
    backend = _backend_for(store)
    cancellation_pending: list[str] = []
    for target in STAGE_ORDER[start_index:]:
        if target not in enabled:
            continue
        selected = state["stages"][target.value]["selected_attempt"]
        if selected is None:
            continue
        if _cancel_attempt_jobs(store, backend, target, int(selected)):
            cancellation_pending.append(f"{target.value} attempt-{int(selected):03d}")
    if cancellation_pending:
        store.rebuild_state()
        raise RuntimeError(
            "Retry is waiting for cancellation confirmation of: "
            + ", ".join(cancellation_pending)
        )
    if stage == StageName.PREPROCESS:
        if not _preprocess_jobs_are_inactive(store, backend):
            raise RuntimeError(
                "Retry is waiting for the previous preprocess worker to terminate "
                "before incomplete outputs can be removed"
            )
        _cleanup_preprocess_partials_for_store(
            store,
            fact_name="retry_partial_cleanup.json",
            strict=True,
        )

    for target in STAGE_ORDER[start_index:]:
        if target not in enabled:
            continue
        selected = state["stages"][target.value]["selected_attempt"]
        if selected is None:
            continue
        directory = store.attempt_dir(target, int(selected))
        if not (directory / "superseded.json").exists():
            store.write_attempt_fact(
                target,
                int(selected),
                "superseded.json",
                {"superseded_at": utc_now(), "retry_from": stage.value},
            )

    selected_attempts: dict[str, int] = {}
    for upstream in STAGE_ORDER[:start_index]:
        if upstream not in enabled:
            continue
        selected = state["stages"][upstream.value]["selected_attempt"]
        if selected is None or state["stages"][upstream.value]["status"] != StageStatus.COMPLETED.value:
            raise RuntimeError(f"Cannot retry {stage.value}: upstream {upstream.value} is not completed")
        selected_attempts[upstream.value] = int(selected)

    created: list[AttemptSpec] = []
    for target in STAGE_ORDER[start_index:]:
        if target not in enabled:
            continue
        upstream_attempts: dict[str, int] = {}
        if target == StageName.SORTING and StageName.PREPROCESS.value in selected_attempts:
            upstream_attempts[StageName.PREPROCESS.value] = selected_attempts[StageName.PREPROCESS.value]
        elif target == StageName.POSTPROCESS:
            if StageName.PREPROCESS.value in selected_attempts:
                upstream_attempts[StageName.PREPROCESS.value] = selected_attempts[StageName.PREPROCESS.value]
            if StageName.SORTING.value in selected_attempts:
                upstream_attempts[StageName.SORTING.value] = selected_attempts[StageName.SORTING.value]
        attempt = store.next_attempt_number(target)
        resource = resource_override if target == stage and resource_override is not None else execution.resource_for(target)
        spec = AttemptSpec(
            run_id=run["run_id"],
            stage=target,
            attempt=attempt,
            backend=execution.resolved_backend,
            resources=resource,
            analysis_sha256=analysis.sha256,
            upstream_attempts=upstream_attempts,
        )
        store.create_attempt(spec)
        selected_attempts[target.value] = attempt
        created.append(spec)
    store.rebuild_state()
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent pipeline Run controller")
    parser.add_argument("--run-dir", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--reconcile", action="store_true")
    action.add_argument("--cancel", action="store_true")
    action.add_argument("--cancel-stage", choices=[stage.value for stage in StageName])
    action.add_argument("--retry", choices=[stage.value for stage in StageName])
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--memory-mb", type=int)
    walltime = parser.add_mutually_exclusive_group()
    walltime.add_argument("--walltime-minutes", type=int)
    walltime.add_argument("--unlimited-walltime", action="store_true")
    parser.add_argument("--gpu-gres-type")
    parser.add_argument("--gpu-constraint")
    parser.add_argument("--partition")
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--reservation")
    return parser


def _execute_controller_action(args: argparse.Namespace) -> int:
    try:
        if args.reconcile:
            state = reconcile_run(args.run_dir)
        elif args.cancel:
            state = cancel_run(args.run_dir)
        elif args.cancel_stage:
            state = cancel_run(args.run_dir, stage=StageName(args.cancel_stage))
        elif args.retry:
            retry_stage = StageName(args.retry)
            default_resource = RunStore(args.run_dir).load_execution().resource_for(retry_stage)
            override = replace(
                default_resource,
                cpus=args.cpus if args.cpus is not None else default_resource.cpus,
                memory_mb=(
                    args.memory_mb if args.memory_mb is not None else default_resource.memory_mb
                ),
                walltime_minutes=(
                    None
                    if args.unlimited_walltime
                    else (
                        args.walltime_minutes
                        if args.walltime_minutes is not None
                        else default_resource.walltime_minutes
                    )
                ),
                gpu_gres_type=(
                    args.gpu_gres_type
                    if args.gpu_gres_type is not None
                    else default_resource.gpu_gres_type
                ),
                gpu_constraint=(
                    args.gpu_constraint
                    if args.gpu_constraint is not None
                    else default_resource.gpu_constraint
                ),
                partition=args.partition if args.partition is not None else default_resource.partition,
                account=args.account if args.account is not None else default_resource.account,
                qos=args.qos if args.qos is not None else default_resource.qos,
                reservation=(
                    args.reservation if args.reservation is not None else default_resource.reservation
                ),
            )
            prepare_retry(args.run_dir, stage=retry_stage, resource_override=override)
            state = submit_run(args.run_dir)
        else:
            state = submit_run(args.run_dir)
        if not args.reconcile:
            print(
                f"Run status={state.get('status', 'unknown')} "
                f"backend={state.get('resolved_backend', 'unknown')}"
            )
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "controller.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"\n[{utc_now()}] controller pid={os.getpid()} action={argv or sys.argv[1:]}\n")
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            return _execute_controller_action(args)


if __name__ == "__main__":
    raise SystemExit(main())
