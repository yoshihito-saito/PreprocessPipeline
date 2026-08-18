from __future__ import annotations

from datetime import datetime, timezone
import ctypes
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


def _is_windows() -> bool:
    return os.name == "nt"


def _linux_process_start_ticks(pid: int) -> str | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8", errors="replace")
        right = text.rfind(")")
        fields = text[right + 2 :].split()
        return fields[19]
    except (OSError, IndexError):
        return None


def _windows_process_identity(pid: int) -> tuple[bool | None, str | None]:
    """Return Windows existence and creation time without treating access errors as exit."""
    if not _is_windows() or pid <= 0:
        return False, None
    try:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        process_query_limited_information = 0x1000
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            # ERROR_INVALID_PARAMETER is the documented result for a PID that
            # does not identify a process. Access/query failures are unknown,
            # not evidence that the worker exited.
            return (False, None) if ctypes.get_last_error() == 87 else (None, None)
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return None, None
            value = (int(creation.dwHighDateTime) << 32) | int(
                creation.dwLowDateTime
            )
            return True, str(value)
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None


def _windows_process_creation_time(pid: int) -> str | None:
    exists, creation_time = _windows_process_identity(pid)
    return creation_time if exists is True else None


def _process_start_token(pid: int) -> str | None:
    """Return a platform-qualified token that survives controller restarts."""

    if _is_windows():
        created = _windows_process_creation_time(pid)
        return f"windows-filetime:{created}" if created is not None else None
    ticks = _linux_process_start_ticks(pid)
    return f"linux-ticks:{ticks}" if ticks is not None else None


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if _is_windows():
        exists, _creation_time = _windows_process_identity(pid)
        return exists is not False
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
    if process_group_id <= 0 or _is_windows():
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _taskkill_windows_process_tree(pid: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        check=False,
        capture_output=True,
        text=True,
    )


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
                start_new_session=not _is_windows(),
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()
        process_creation_time = _windows_process_creation_time(process.pid)
        if _is_windows() and process_creation_time is None:
            result = _taskkill_windows_process_tree(process.pid)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "Could not capture a reusable Windows process identity; "
                "the Local worker was terminated before submission"
                + (f": {detail}" if detail else "")
            )
        job_id = str(process.pid)
        self._children[job_id] = process
        metadata = {
            "pid": process.pid,
            "process_group_id": process.pid if not _is_windows() else None,
            "process_start_ticks": _linux_process_start_ticks(process.pid),
            "process_start_token": _process_start_token(process.pid),
            "process_creation_time": process_creation_time,
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
        expected_creation_time = job_ref.metadata.get("process_creation_time")
        if expected_creation_time is not None:
            actual_creation_time = _windows_process_creation_time(pid)
            return (
                actual_creation_time is not None
                and str(expected_creation_time) == str(actual_creation_time)
            )
        expected_token = job_ref.metadata.get("process_start_token")
        if expected_token is not None:
            actual_token = _process_start_token(pid)
            return actual_token is not None and str(expected_token) == str(actual_token)
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
        if _is_windows():
            result = _taskkill_windows_process_tree(pid)
            expected_creation_time = str(
                job_ref.metadata.get("process_creation_time") or ""
            )
            if not expected_creation_time:
                legacy_token = str(job_ref.metadata.get("process_start_token") or "")
                prefix = "windows-filetime:"
                if legacy_token.startswith(prefix):
                    expected_creation_time = legacy_token[len(prefix) :]
            for _ in range(50):
                exists, actual_creation_time = _windows_process_identity(pid)
                if exists is False or (
                    exists is True
                    and expected_creation_time
                    and actual_creation_time is not None
                    and actual_creation_time != expected_creation_time
                ):
                    return
                time.sleep(0.1)
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"taskkill did not terminate Local job {job_ref.job_id}"
                + (f": {detail}" if detail else "")
            )
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
