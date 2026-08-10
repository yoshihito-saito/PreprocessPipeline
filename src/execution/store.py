from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator
import uuid

from .models import AnalysisConfig, AttemptSpec, ExecutionConfig, JobRef, StageName, StageStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def short_file_lock(path: Path, *, timeout: float = 10.0) -> Iterator[None]:
    lock_path = Path(path)
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                contents = lock_path.read_text(encoding="utf-8", errors="replace")
                match = next(
                    (part for part in contents.split() if part.startswith("pid=")),
                    "",
                )
                owner_pid = int(match.removeprefix("pid=")) if match else -1
                owner_alive = owner_pid > 0
                if owner_alive:
                    try:
                        os.kill(owner_pid, 0)
                    except ProcessLookupError:
                        owner_alive = False
                    except PermissionError:
                        owner_alive = True
                if age > timeout and not owner_alive:
                    lock_path.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Run lock: {lock_path}")
            time.sleep(0.05)
    try:
        os.write(fd, f"pid={os.getpid()} created={utc_now()}\n".encode("utf-8"))
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


class RunStore:
    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.stages_dir = self.run_dir / "stages"

    @property
    def run_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def analysis_path(self) -> Path:
        return self.run_dir / "analysis_config.json"

    @property
    def execution_path(self) -> Path:
        return self.run_dir / "execution_config.json"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def lock_path(self) -> Path:
        return self.run_dir / ".state.lock"

    def initialize(
        self,
        *,
        run: dict[str, Any],
        analysis: AnalysisConfig,
        execution: ExecutionConfig,
    ) -> None:
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "snapshots").mkdir(exist_ok=True)
        atomic_write_json(self.run_path, run)
        atomic_write_json(self.analysis_path, analysis.to_dict())
        atomic_write_json(self.execution_path, execution.to_dict())
        self.rebuild_state()

    def load_run(self) -> dict[str, Any]:
        return read_json(self.run_path)

    def load_analysis(self) -> AnalysisConfig:
        return AnalysisConfig.from_dict(read_json(self.analysis_path))

    def load_execution(self) -> ExecutionConfig:
        return ExecutionConfig.from_dict(read_json(self.execution_path))

    def attempt_dir(self, stage: StageName | str, attempt: int) -> Path:
        stage_name = StageName(stage).value
        return self.stages_dir / stage_name / f"attempt-{int(attempt):03d}"

    def list_attempt_numbers(self, stage: StageName | str) -> list[int]:
        stage_dir = self.stages_dir / StageName(stage).value
        if not stage_dir.exists():
            return []
        values: list[int] = []
        for path in stage_dir.glob("attempt-*"):
            try:
                values.append(int(path.name.removeprefix("attempt-")))
            except ValueError:
                continue
        return sorted(values)

    def next_attempt_number(self, stage: StageName | str) -> int:
        existing = self.list_attempt_numbers(stage)
        return (existing[-1] + 1) if existing else 1

    def create_attempt(self, spec: AttemptSpec) -> Path:
        spec.validate()
        attempt_dir = self.attempt_dir(spec.stage, spec.attempt)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(attempt_dir / "spec.json", spec.to_dict())
        self.rebuild_state()
        return attempt_dir

    def load_attempt_spec(self, stage: StageName | str, attempt: int) -> AttemptSpec:
        return AttemptSpec.from_dict(read_json(self.attempt_dir(stage, attempt) / "spec.json"))

    def write_attempt_fact(
        self,
        stage: StageName | str,
        attempt: int,
        filename: str,
        value: dict[str, Any],
        *,
        overwrite: bool = False,
    ) -> Path:
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("Attempt fact filename must be a simple .json filename")
        path = self.attempt_dir(stage, attempt) / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Attempt fact is immutable and already exists: {path}")
        atomic_write_json(path, value)
        return path

    def record_observation(
        self,
        stage: StageName | str,
        attempt: int,
        *,
        job: JobRef,
        status: dict[str, Any],
    ) -> Path:
        directory = self.attempt_dir(stage, attempt) / "observations"
        directory.mkdir(parents=True, exist_ok=True)
        conclusive_terminal = bool(status.get("terminal", False)) and (
            status.get("successful") is not None
            or str(status.get("state", "")).lower() in {"cancelled", "canceled"}
        )
        live_path = directory / "latest.json"
        if conclusive_terminal:
            live_path.unlink(missing_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            path = directory / f"{timestamp}-{uuid.uuid4().hex[:8]}.json"
        else:
            path = live_path
        atomic_write_json(
            path,
            {"observed_at": utc_now(), "job": job.to_dict(), "status": status},
        )
        return path

    def latest_observation(self, stage: StageName | str, attempt: int) -> dict[str, Any] | None:
        directory = self.attempt_dir(stage, attempt) / "observations"
        paths = sorted(directory.glob("*.json")) if directory.exists() else []
        return read_json(paths[-1]) if paths else None

    def read_attempt_fact(
        self, stage: StageName | str, attempt: int, filename: str
    ) -> dict[str, Any] | None:
        path = self.attempt_dir(stage, attempt) / filename
        return read_json(path) if path.exists() else None

    def _attempt_view(self, stage: StageName, attempt: int) -> dict[str, Any]:
        directory = self.attempt_dir(stage, attempt)
        spec = read_json(directory / "spec.json")
        result = self.read_attempt_fact(stage, attempt, "result.json")
        failure = self.read_attempt_fact(stage, attempt, "failure.json")
        submission_failure = self.read_attempt_fact(stage, attempt, "submission_failure.json")
        submission_intent = self.read_attempt_fact(stage, attempt, "submission_intent.json")
        submitted = self.read_attempt_fact(stage, attempt, "submitted.json")
        started = self.read_attempt_fact(stage, attempt, "started.json")
        blocked = self.read_attempt_fact(stage, attempt, "blocked.json")
        superseded = self.read_attempt_fact(stage, attempt, "superseded.json")
        cancel_requested = self.read_attempt_fact(stage, attempt, "cancel_requested.json")
        cancel_confirmed = self.read_attempt_fact(stage, attempt, "cancel_confirmed.json")
        observation = self.latest_observation(stage, attempt)
        result_is_valid = bool(
            result is not None
            and result.get("status") == StageStatus.COMPLETED.value
            and result.get("analysis_sha256") == spec.get("analysis_sha256")
            and bool((result.get("validation") or {}).get("passed", False))
        )
        observed_status = (observation or {}).get("status", {})
        backend_terminal = bool(observed_status.get("terminal", False))
        backend_successful = observed_status.get("successful")

        status = StageStatus.PENDING
        reason = ""
        if superseded is not None:
            status = StageStatus.SUPERSEDED
        elif result_is_valid:
            if backend_terminal and backend_successful is False:
                status = StageStatus.FAILED
                reason = str(
                    observed_status.get(
                        "reason", "Backend exit status contradicted the validated StageResult"
                    )
                )
            elif submitted is not None and not backend_terminal:
                status = StageStatus.RUNNING
                reason = "Validated StageResult is awaiting terminal backend confirmation"
            else:
                # A validated result plus a successful terminal observation is
                # canonical. If accounting is unavailable (successful=None),
                # the worker's final atomic result is the recovery commitment.
                status = StageStatus.COMPLETED
                if started is not None and submitted is None:
                    reason = "Validated StageResult recovered without a persisted JobRef"
        elif result is not None:
            status = StageStatus.FAILED
            reason = "StageResult failed identity or validation checks"
        elif failure is not None or (
            submission_failure is not None
            and not bool(submission_failure.get("acceptance_ambiguous", False))
        ):
            status = StageStatus.FAILED
            reason = str((failure or submission_failure or {}).get("message", ""))
        elif cancel_confirmed is not None:
            status = StageStatus.CANCELLED
        elif blocked is not None:
            status = StageStatus.BLOCKED
            reason = str(blocked.get("reason", ""))
        elif submission_intent is not None and submitted is None:
            created_text = str(submission_intent.get("created_at", ""))
            try:
                created_at = datetime.fromisoformat(created_text)
                age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
            except (TypeError, ValueError):
                age_seconds = 60.0
            if age_seconds >= 60.0:
                status = StageStatus.LOST
                reason = "Slurm submission intent exists but no JobRef outcome was persisted"
            else:
                status = StageStatus.PENDING
                reason = "Slurm submission is in progress"
        elif observation is not None:
            backend_status = observation.get("status", {})
            state = str(backend_status.get("state", "")).lower()
            terminal = bool(backend_status.get("terminal", False))
            successful = backend_status.get("successful")
            if terminal and successful is True:
                status = StageStatus.LOST
                reason = "Backend completed, but no validated StageResult exists"
            elif terminal and state in {"cancelled", "canceled"}:
                status = StageStatus.CANCELLED
            elif terminal and (successful is None or state in {"unknown", "lost"}):
                status = StageStatus.LOST
                reason = str(
                    backend_status.get("reason", "Backend outcome is unknown")
                )
            elif terminal:
                status = StageStatus.FAILED
                reason = str(backend_status.get("reason", state))
            elif state in {"running", "completing", "configuring"}:
                status = StageStatus.RUNNING
            elif state in {"unknown", "lost"}:
                status = StageStatus.LOST
                reason = str(backend_status.get("reason", "Backend outcome is unknown"))
            else:
                status = StageStatus.SUBMITTED
        elif started is not None:
            status = StageStatus.RUNNING
        elif submission_failure is not None and bool(
            submission_failure.get("acceptance_ambiguous", False)
        ):
            status = StageStatus.LOST
            reason = str(
                submission_failure.get(
                    "message", "Slurm may have accepted the job; automatic retry is unsafe"
                )
            )
        elif submitted is not None:
            status = StageStatus.SUBMITTED

        return {
            "attempt": attempt,
            "status": status.value,
            "reason": reason,
            "spec": spec,
            "jobs": list((submitted or {}).get("jobs", [])),
            "submitted_at": (submitted or {}).get("submitted_at"),
            "started_at": (started or {}).get("started_at"),
            "finished_at": (
                (result or failure or cancel_confirmed or {}).get("finished_at")
                or (submission_failure or {}).get("failed_at")
            ),
            "cancel_requested": cancel_requested is not None,
            "result": result,
            "failure": failure or submission_failure,
            "latest_observation": observation,
        }

    def derive_state(self) -> dict[str, Any]:
        run = self.load_run()
        enabled = [StageName(value) for value in run.get("enabled_stages", [])]
        stages: dict[str, Any] = {}
        selected_statuses: list[str] = []
        for stage in StageName:
            attempts = [self._attempt_view(stage, number) for number in self.list_attempt_numbers(stage)]
            selected = next(
                (item for item in reversed(attempts) if item["status"] != StageStatus.SUPERSEDED.value),
                attempts[-1] if attempts else None,
            )
            if stage in enabled:
                selected_statuses.append(
                    selected["status"] if selected is not None else StageStatus.PENDING.value
                )
            stages[stage.value] = {
                "enabled": stage in enabled,
                "selected_attempt": selected["attempt"] if selected else None,
                "status": selected["status"] if selected else StageStatus.PENDING.value,
                "attempts": attempts,
            }

        if selected_statuses and all(value == StageStatus.COMPLETED.value for value in selected_statuses):
            overall = StageStatus.COMPLETED.value
        elif any(value == StageStatus.FAILED.value for value in selected_statuses):
            overall = StageStatus.FAILED.value
        elif any(value == StageStatus.LOST.value for value in selected_statuses):
            overall = StageStatus.LOST.value
        elif any(value == StageStatus.CANCELLED.value for value in selected_statuses):
            overall = StageStatus.CANCELLED.value
        elif any(value == StageStatus.BLOCKED.value for value in selected_statuses):
            overall = StageStatus.BLOCKED.value
        elif any(value == StageStatus.RUNNING.value for value in selected_statuses):
            overall = StageStatus.RUNNING.value
        elif any(value == StageStatus.SUBMITTED.value for value in selected_statuses):
            overall = StageStatus.SUBMITTED.value
        else:
            overall = StageStatus.PENDING.value

        return {
            "schema_version": 1,
            "run_id": run["run_id"],
            "updated_at": utc_now(),
            "status": overall,
            "requested_backend": run["requested_backend"],
            "resolved_backend": run["resolved_backend"],
            "stages": stages,
        }

    def rebuild_state(self) -> dict[str, Any]:
        if not self.run_path.exists():
            return {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with short_file_lock(self.lock_path):
            state = self.derive_state()
            atomic_write_json(self.state_path, state)
        return state
