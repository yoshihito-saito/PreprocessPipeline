from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from ..models import AttemptSpec, BackendName, BackendStatus, JobRef
from .base import ExecutionBackend


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _linux_process_start_ticks(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8", errors="replace")
        right = text.rfind(")")
        fields = text[right + 2 :].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8", errors="replace")
        right = text.rfind(")")
        if right >= 0 and text[right + 2 :].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_exists(process_group_id: int) -> bool:
    if process_group_id <= 0 or os.name == "nt":
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class LocalBackend(ExecutionBackend):
    def __init__(self, *, python_executable: str | None = None) -> None:
        self.python_executable = python_executable or sys.executable
        self._children: dict[str, subprocess.Popen[Any]] = {}

    def submit(
        self,
        *,
        run_dir: Path,
        attempt_spec: AttemptSpec,
        dependency_job_refs: list[JobRef] | None = None,
    ) -> list[JobRef]:
        if dependency_job_refs:
            for dependency in dependency_job_refs:
                status = self.status(dependency)
                if not status.terminal or status.successful is not True:
                    raise RuntimeError(
                        f"Local dependency {dependency.job_id} has not completed successfully"
                    )
        attempt_dir = Path(run_dir) / "stages" / attempt_spec.stage.value / f"attempt-{attempt_spec.attempt:03d}"
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        command = [
            self.python_executable,
            "-m",
            "src.execution.worker",
            "--run-dir",
            str(Path(run_dir).resolve()),
            "--stage",
            attempt_spec.stage.value,
            "--attempt",
            str(attempt_spec.attempt),
        ]
        stdout_handle = stdout_path.open("ab", buffering=0)
        stderr_handle = stderr_path.open("ab", buffering=0)
        worker_environment = dict(os.environ)
        worker_environment["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                env=worker_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                close_fds=True,
                start_new_session=(os.name != "nt"),
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        job_id = str(process.pid)
        self._children[job_id] = process
        metadata = {
            "pid": process.pid,
            "process_group_id": process.pid if os.name != "nt" else None,
            "process_start_ticks": _linux_process_start_ticks(process.pid),
            "command": command,
        }
        return [
            JobRef(
                backend=BackendName.LOCAL,
                job_id=job_id,
                submitted_at=_utc_now(),
                metadata=metadata,
            )
        ]

    def _identity_matches(self, job_ref: JobRef) -> bool:
        pid = int(job_ref.metadata.get("pid", job_ref.job_id))
        if not _process_exists(pid):
            return False
        owned = self._children.get(job_ref.job_id)
        if owned is not None and owned.pid == pid and owned.poll() is None:
            return True
        expected = job_ref.metadata.get("process_start_ticks")
        actual = _linux_process_start_ticks(pid)
        # Persistent cancellation must fail closed when process identity cannot
        # be verified. This avoids signalling an unrelated process after PID
        # reuse on platforms without Linux /proc start ticks.
        return (
            expected is not None
            and actual is not None
            and str(expected) == str(actual)
        )

    def status(self, job_ref: JobRef) -> BackendStatus:
        process = self._children.get(job_ref.job_id)
        if process is not None:
            return_code = process.poll()
            if return_code is None:
                return BackendStatus(state="running", terminal=False)
            return BackendStatus(
                state="completed" if return_code == 0 else "failed",
                terminal=True,
                successful=return_code == 0,
                exit_code=str(return_code),
            )
        if self._identity_matches(job_ref):
            return BackendStatus(state="running", terminal=False)
        return BackendStatus(
            state="unknown",
            terminal=True,
            successful=None,
            reason="Local process is no longer observable; use the persisted worker result/failure fact",
        )

    def wait(self, job_ref: JobRef) -> int:
        process = self._children.get(job_ref.job_id)
        if process is None:
            raise RuntimeError(f"Local job {job_ref.job_id} is not owned by this controller")
        return int(process.wait())

    def cancel(self, job_ref: JobRef) -> None:
        pid = int(job_ref.metadata.get("pid", job_ref.job_id))
        if not _process_exists(pid):
            return
        if not self._identity_matches(job_ref):
            raise RuntimeError(
                f"Refusing to cancel Local job {job_ref.job_id}: "
                "the live process identity cannot be verified"
            )
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            return
        try:
            process_group_id = int(job_ref.metadata.get("process_group_id") or pid)
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(30):
            # The worker leader may exit before MATLAB or another descendant.
            # Escalation is complete only when the entire originally verified
            # process group is gone, not merely when the leader disappears.
            if not _process_group_exists(process_group_id):
                return
            time.sleep(0.1)
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return
