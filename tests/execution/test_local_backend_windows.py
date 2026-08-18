from __future__ import annotations

import os
import subprocess
import sys

import pytest

from src.execution.backends.local import LocalBackend, _process_start_token
from src.execution.models import BackendName, JobRef


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows process identity tests")


def _sleeping_process() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_windows_process_start_token_is_stable() -> None:
    process = _sleeping_process()
    try:
        first = _process_start_token(process.pid)
        second = _process_start_token(process.pid)

        assert first is not None
        assert first.startswith("windows-filetime:")
        assert second == first
    finally:
        process.kill()
        process.wait(timeout=10)


def test_new_backend_recognizes_persisted_windows_job() -> None:
    process = _sleeping_process()
    try:
        token = _process_start_token(process.pid)
        assert token is not None
        job = JobRef(
            backend=BackendName.LOCAL,
            job_id=str(process.pid),
            submitted_at="test",
            metadata={"pid": process.pid, "process_start_token": token},
        )

        # This backend did not launch the process. It represents the detached
        # controller started later by the GUI's Force stop action.
        assert LocalBackend()._identity_matches(job) is True
    finally:
        process.kill()
        process.wait(timeout=10)


def test_windows_cancel_refuses_reused_or_mismatched_pid() -> None:
    process = _sleeping_process()
    try:
        job = JobRef(
            backend=BackendName.LOCAL,
            job_id=str(process.pid),
            submitted_at="test",
            metadata={
                "pid": process.pid,
                "process_start_token": "windows-filetime:not-this-process",
            },
        )

        with pytest.raises(RuntimeError, match="identity cannot be verified"):
            LocalBackend().cancel(job)

        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=10)
