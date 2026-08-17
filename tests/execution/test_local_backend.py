from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from src.execution.backends.local import LocalBackend
import src.execution.backends.local as local_backend
from src.execution.models import AttemptSpec, BackendName, JobRef, ResourceSpec, StageName


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell-script worker test")
def test_local_worker_output_is_unbuffered(tmp_path: Path) -> None:
    executable = tmp_path / "worker-env"
    executable.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$PYTHONUNBUFFERED\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "stages" / "preprocess" / "attempt-001"
    attempt_dir.mkdir(parents=True)
    spec = AttemptSpec(
        run_id="run-local-unbuffered",
        stage=StageName.PREPROCESS,
        attempt=1,
        backend=BackendName.LOCAL,
        resources=ResourceSpec(cpus=1, memory_mb=1024, walltime_minutes=None),
        analysis_sha256="abc",
    )
    backend = LocalBackend(python_executable=str(executable))

    job = backend.submit(run_dir=run_dir, attempt_spec=spec)[0]
    for _ in range(50):
        if backend.status(job).terminal:
            break
        time.sleep(0.02)

    assert backend.status(job).terminal is True
    assert (attempt_dir / "stdout.log").read_text(encoding="utf-8") == "1\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group persistence test")
def test_local_job_is_observable_and_cancellable_from_a_new_backend_instance(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "detached-worker"
    executable.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    executable.chmod(0o700)
    run_dir = tmp_path / "run"
    (run_dir / "stages" / "preprocess" / "attempt-001").mkdir(parents=True)
    spec = AttemptSpec(
        run_id="run-local-persistence",
        stage=StageName.PREPROCESS,
        attempt=1,
        backend=BackendName.LOCAL,
        resources=ResourceSpec(cpus=1, memory_mb=1024, walltime_minutes=10),
        analysis_sha256="abc",
    )

    job = LocalBackend(python_executable=str(executable)).submit(
        run_dir=run_dir,
        attempt_spec=spec,
    )[0]
    reopened = LocalBackend()
    try:
        assert reopened.status(job).state == "running"
        reopened.cancel(job)
        for _ in range(50):
            if reopened.status(job).terminal:
                break
            time.sleep(0.02)
        assert reopened.status(job).terminal is True
    finally:
        reopened.cancel(job)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group escalation test")
def test_local_cancel_kills_term_ignoring_descendant_after_leader_exits(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    executable = tmp_path / "leader-with-stubborn-child"
    executable.write_text(
        "#!/bin/sh\n"
        "trap 'exit 0' TERM\n"
        "sh -c 'trap \"\" TERM; while :; do sleep 1; done' &\n"
        "child=$!\n"
        f"printf '%s\\n' \"$child\" > '{child_pid_path}'\n"
        "wait \"$child\"\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    run_dir = tmp_path / "run"
    (run_dir / "stages" / "preprocess" / "attempt-001").mkdir(parents=True)
    spec = AttemptSpec(
        run_id="run-local-descendant-cancel",
        stage=StageName.PREPROCESS,
        attempt=1,
        backend=BackendName.LOCAL,
        resources=ResourceSpec(cpus=1, memory_mb=1024, walltime_minutes=10),
        analysis_sha256="abc",
    )
    backend = LocalBackend(python_executable=str(executable))
    job = backend.submit(run_dir=run_dir, attempt_spec=spec)[0]
    try:
        for _ in range(100):
            if child_pid_path.exists():
                break
            time.sleep(0.02)
        assert child_pid_path.exists()
        child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())

        backend.cancel(job)

        for _ in range(100):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("TERM-ignoring Local descendant remained alive after escalation")
    finally:
        try:
            os.killpg(int(job.metadata["process_group_id"]), 9)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX process identity test")
def test_reconnected_local_cancel_refuses_unverifiable_process_identity(
    monkeypatch,
) -> None:
    job = JobRef(
        backend=BackendName.LOCAL,
        job_id="4242",
        submitted_at="2026-08-10T00:00:00+00:00",
        metadata={"pid": 4242, "process_group_id": 4242, "process_start_ticks": None},
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr("src.execution.backends.local._process_exists", lambda _pid: True)
    monkeypatch.setattr(
        "src.execution.backends.local._linux_process_start_ticks", lambda _pid: None
    )
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))

    backend = LocalBackend()
    assert backend.status(job).state == "unknown"
    with pytest.raises(RuntimeError, match="identity cannot be verified"):
        backend.cancel(job)

    assert signals == []


def _windows_job(*, creation_time: str = "123456") -> JobRef:
    return JobRef(
        backend=BackendName.LOCAL,
        job_id="4242",
        submitted_at="2026-08-17T00:00:00+00:00",
        metadata={"pid": 4242, "process_creation_time": creation_time},
    )


def test_reconnected_windows_local_cancel_uses_persisted_creation_time(
    monkeypatch,
) -> None:
    observations = iter(
        [
            (True, "123456"),
            (True, "123456"),
            (True, "123456"),
            (False, None),
        ]
    )
    taskkill_pids: list[int] = []
    monkeypatch.setattr(local_backend, "_is_windows", lambda: True)
    monkeypatch.setattr(
        local_backend, "_windows_process_identity", lambda _pid: next(observations)
    )
    monkeypatch.setattr(
        local_backend,
        "_taskkill_windows_process_tree",
        lambda pid: (
            taskkill_pids.append(pid)
            or SimpleNamespace(stdout="", stderr="", returncode=0)
        ),
    )

    LocalBackend().cancel(_windows_job())

    assert taskkill_pids == [4242]


@pytest.mark.parametrize(
    "observed_identity",
    [(True, "different-process"), (None, None)],
)
def test_reconnected_windows_local_cancel_fails_closed_on_unverifiable_identity(
    monkeypatch, observed_identity
) -> None:
    taskkill_pids: list[int] = []
    monkeypatch.setattr(local_backend, "_is_windows", lambda: True)
    monkeypatch.setattr(
        local_backend,
        "_windows_process_identity",
        lambda _pid: observed_identity,
    )
    monkeypatch.setattr(
        local_backend,
        "_taskkill_windows_process_tree",
        lambda pid: taskkill_pids.append(pid),
    )

    with pytest.raises(RuntimeError, match="identity cannot be verified"):
        LocalBackend().cancel(_windows_job())

    assert taskkill_pids == []


def test_windows_cancel_does_not_confirm_when_identity_becomes_unverifiable(
    monkeypatch,
) -> None:
    calls = 0

    def process_identity(_pid):
        nonlocal calls
        calls += 1
        if calls <= 3:
            return True, "123456"
        return None, None

    monkeypatch.setattr(local_backend, "_is_windows", lambda: True)
    monkeypatch.setattr(local_backend, "_windows_process_identity", process_identity)
    monkeypatch.setattr(local_backend.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        local_backend,
        "_taskkill_windows_process_tree",
        lambda _pid: SimpleNamespace(stdout="", stderr="access denied", returncode=1),
    )

    with pytest.raises(RuntimeError, match="taskkill did not terminate"):
        LocalBackend().cancel(_windows_job())


def test_windows_submit_terminates_worker_when_creation_time_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeProcess:
        pid = 4242

        def __init__(self) -> None:
            self.killed = False

        def wait(self, timeout=None) -> int:
            return -9

        def kill(self) -> None:
            self.killed = True

    fake_process = FakeProcess()
    taskkill_pids: list[int] = []
    monkeypatch.setattr(local_backend, "_is_windows", lambda: True)
    monkeypatch.setattr(local_backend, "_windows_process_creation_time", lambda _pid: None)
    monkeypatch.setattr(local_backend.subprocess, "Popen", lambda *_args, **_kwargs: fake_process)
    monkeypatch.setattr(
        local_backend,
        "_taskkill_windows_process_tree",
        lambda pid: (
            taskkill_pids.append(pid)
            or SimpleNamespace(stdout="", stderr="", returncode=0)
        ),
    )
    run_dir = tmp_path / "run"
    (run_dir / "stages" / "preprocess" / "attempt-001").mkdir(parents=True)
    spec = AttemptSpec(
        run_id="run-windows-no-identity",
        stage=StageName.PREPROCESS,
        attempt=1,
        backend=BackendName.LOCAL,
        resources=ResourceSpec(cpus=1, memory_mb=1024, walltime_minutes=None),
        analysis_sha256="abc",
    )
    backend = LocalBackend()

    with pytest.raises(RuntimeError, match="worker was terminated before submission"):
        backend.submit(run_dir=run_dir, attempt_spec=spec)

    assert taskkill_pids == [4242]
    assert backend._children == {}
