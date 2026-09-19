from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from src.execution import gpu_selection
from src.execution.gpu_selection import GpuDevice, GpuProcess


def _device(
    index: int,
    memory_percent: float,
    *,
    utilization: float = 0.0,
    processes: tuple[GpuProcess, ...] = (),
) -> GpuDevice:
    total = 100_000.0
    return GpuDevice(
        index=index,
        uuid=f"GPU-uuid-{index}",
        name="Test GPU",
        memory_used_mib=total * memory_percent / 100.0,
        memory_total_mib=total,
        memory_percent=memory_percent,
        utilization_percent=utilization,
        processes=processes,
    )


def test_query_gpu_devices_records_process_user_and_memory(monkeypatch) -> None:
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout="GPU-b, 321, /opt/MATLAB, 8192\n",
                stderr="",
            ),
            SimpleNamespace(
                returncode=0,
                stdout=(
                    "0, GPU-a, RTX Test, 60000, 100000, 75\n"
                    "1, GPU-b, RTX Test, 10000, 100000, 20\n"
                ),
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(gpu_selection.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(gpu_selection, "_process_username", lambda pid: f"user-{pid}")

    devices = gpu_selection.query_gpu_devices()

    assert [device.index for device in devices] == [0, 1]
    assert devices[0].memory_percent == pytest.approx(60.0)
    assert devices[1].processes == (
        GpuProcess(
            pid=321,
            username="user-321",
            process_name="/opt/MATLAB",
            used_memory_mib=8192.0,
        ),
    )


def test_gpu_selection_waits_then_activates_least_used_uuid(
    tmp_path: Path, monkeypatch
) -> None:
    busy_process = GpuProcess(123, "other-user", "python", 60_000.0)
    snapshots = iter(
        [
            [
                _device(0, 80.0, processes=(busy_process,)),
                _device(1, 10.0, utilization=80.0),
            ],
            [_device(0, 70.0, processes=(busy_process,)), _device(1, 12.0, utilization=5.0)],
        ]
    )
    sleeps: list[float] = []
    log_path = tmp_path / "gpu-selection.jsonl"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    selection = gpu_selection.activate_least_used_gpu(
        log_path=log_path,
        query=lambda: next(snapshots),
        sleeper=sleeps.append,
        poll_seconds=0.25,
    )

    assert sleeps == [0.25]
    assert selection["selected_gpu"]["index"] == 1
    assert selection["allocation_environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert gpu_selection.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-uuid-1"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["waiting", "selected"]
    assert events[0]["gpus"][0]["processes"][0]["username"] == "other-user"
    assert events[0]["gpus"][0]["processes"][0]["used_memory_mib"] == 60_000.0
    assert events[0]["allocation_environment"]["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert events[0]["allocation_environment"]["PYTORCH_CUDA_ALLOC_CONF"] is None


def test_gpu_selection_uses_utilization_and_index_as_tiebreakers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    devices = [
        _device(0, 10.0, utilization=30.0),
        _device(1, 10.0, utilization=5.0),
    ]

    selection = gpu_selection.activate_least_used_gpu(
        log_path=tmp_path / "gpu-selection.jsonl",
        query=lambda: devices,
        sleeper=lambda _seconds: None,
    )

    assert selection["selected_gpu"]["index"] == 1
    assert gpu_selection.os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-uuid-1"


def test_gpu_query_failure_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(
        gpu_selection.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="driver unavailable",
        ),
    )

    with pytest.raises(RuntimeError, match="driver unavailable"):
        gpu_selection.query_gpu_devices()


def test_gpu_monitor_records_periodic_and_final_snapshots(tmp_path: Path) -> None:
    periodic_recorded = threading.Event()
    query_count = 0

    def query() -> list[GpuDevice]:
        nonlocal query_count
        query_count += 1
        periodic_recorded.set()
        return [_device(0, 20.0, utilization=10.0)]

    log_path = tmp_path / "gpu-selection.jsonl"
    monitor = gpu_selection.GpuUsageMonitor(
        log_path=log_path,
        selected_gpu_uuid="GPU-uuid-0",
        interval_seconds=0.01,
        query=query,
    )
    monitor.start()
    assert periodic_recorded.wait(timeout=1.0)
    monitor.stop(final_status="sorter_finished")

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["status"] == "monitoring"
    assert events[-1]["status"] == "sorter_finished"
    assert events[-1]["selected_gpu"]["uuid"] == "GPU-uuid-0"
    assert query_count >= 2


def test_gpu_monitor_logs_query_error_without_raising(tmp_path: Path) -> None:
    query_attempted = threading.Event()

    def failing_query() -> list[GpuDevice]:
        query_attempted.set()
        raise RuntimeError("temporary telemetry failure")

    log_path = tmp_path / "gpu-selection.jsonl"
    monitor = gpu_selection.GpuUsageMonitor(
        log_path=log_path,
        selected_gpu_uuid="GPU-uuid-1",
        interval_seconds=0.01,
        query=failing_query,
    )
    monitor.start()
    assert query_attempted.wait(timeout=1.0)
    monitor.stop(final_status="sorter_failed")

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert events
    assert all(event["status"] == "monitor_error" for event in events)
    assert events[-1]["monitor_phase"] == "sorter_failed"
    assert "temporary telemetry failure" in events[-1]["error"]["message"]
