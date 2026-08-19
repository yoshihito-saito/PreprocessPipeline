from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Callable


GPU_BUSY_PERCENT = 50.0
GPU_POLL_SECONDS = 30.0
GPU_MONITOR_SECONDS = 10.0 * 60.0


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    username: str
    process_name: str
    used_memory_mib: float | None


@dataclass(frozen=True)
class GpuDevice:
    index: int
    uuid: str
    name: str
    memory_used_mib: float
    memory_total_mib: float
    memory_percent: float
    utilization_percent: float | None
    processes: tuple[GpuProcess, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_float(value: str) -> float | None:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _run_nvidia_query(fields: str) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-{fields.split(':', 1)[0]}={fields.split(':', 1)[1]}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("GPU selection requires nvidia-smi, but it was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("GPU selection timed out while querying nvidia-smi") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "GPU selection could not query nvidia-smi"
            + (f": {detail}" if detail else "")
        )
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True)
        if row
    ]


def _process_username(pid: int) -> str:
    if os.name != "posix":
        return "unknown"
    try:
        import pwd

        uid = Path(f"/proc/{int(pid)}").stat().st_uid
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError, ValueError):
        return "unknown"


def query_gpu_devices() -> list[GpuDevice]:
    process_rows = _run_nvidia_query(
        "compute-apps:gpu_uuid,pid,process_name,used_gpu_memory"
    )
    processes_by_uuid: dict[str, list[GpuProcess]] = {}
    for row in process_rows:
        if len(row) < 4:
            continue
        try:
            pid = int(row[1])
        except ValueError:
            continue
        processes_by_uuid.setdefault(row[0], []).append(
            GpuProcess(
                pid=pid,
                username=_process_username(pid),
                process_name=row[2],
                used_memory_mib=_parse_float(row[3]),
            )
        )

    gpu_rows = _run_nvidia_query(
        "gpu:index,uuid,name,memory.used,memory.total,utilization.gpu"
    )
    devices: list[GpuDevice] = []
    for row in gpu_rows:
        if len(row) < 6:
            continue
        used = _parse_float(row[3])
        total = _parse_float(row[4])
        if used is None or total is None or total <= 0:
            continue
        try:
            index = int(row[0])
        except ValueError:
            continue
        devices.append(
            GpuDevice(
                index=index,
                uuid=row[1],
                name=row[2],
                memory_used_mib=used,
                memory_total_mib=total,
                memory_percent=100.0 * used / total,
                utilization_percent=_parse_float(row[5]),
                processes=tuple(
                    sorted(processes_by_uuid.get(row[1], []), key=lambda item: item.pid)
                ),
            )
        )
    if not devices:
        raise RuntimeError("GPU selection found no queryable NVIDIA GPUs")
    return sorted(devices, key=lambda item: item.index)


def _append_event(log_path: Path, event: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _device_line(device: GpuDevice) -> str:
    utilization = (
        "unknown" if device.utilization_percent is None else f"{device.utilization_percent:.0f}%"
    )
    users = sorted({process.username for process in device.processes})
    user_text = ",".join(users) if users else "none"
    return (
        f"GPU {device.index}: memory={device.memory_percent:.1f}% "
        f"({device.memory_used_mib:.0f}/{device.memory_total_mib:.0f} MiB), "
        f"utilization={utilization}, users={user_text}"
    )


def _allocation_environment() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
        )
    }


class GpuUsageMonitor:
    def __init__(
        self,
        *,
        log_path: Path,
        selected_gpu_uuid: str,
        interval_seconds: float = GPU_MONITOR_SECONDS,
        query: Callable[[], list[GpuDevice]] = query_gpu_devices,
    ) -> None:
        if float(interval_seconds) <= 0:
            raise ValueError("interval_seconds must be positive")
        self.log_path = Path(log_path)
        self.selected_gpu_uuid = str(selected_gpu_uuid)
        self.interval_seconds = float(interval_seconds)
        self.query = query
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _record(self, status: str) -> None:
        try:
            devices = self.query()
            selected = next(
                (device for device in devices if device.uuid == self.selected_gpu_uuid),
                None,
            )
            event: dict[str, object] = {
                "schema": "preprocess-pipeline-gpu-selection-v1",
                "observed_at": _utc_now(),
                "hostname": socket.gethostname(),
                "status": status,
                "monitor_interval_seconds": self.interval_seconds,
                "allocation_environment": _allocation_environment(),
                "selected_gpu_uuid": self.selected_gpu_uuid,
                "selected_gpu": asdict(selected) if selected is not None else None,
                "gpus": [asdict(device) for device in devices],
            }
            _append_event(self.log_path, event)
            print(
                "[GPU monitoring] "
                + " | ".join(_device_line(device) for device in devices),
                flush=True,
            )
        except Exception as exc:
            # GPU telemetry must never change the scientific sorter outcome.
            # Preserve the audit gap and continue monitoring on the next tick.
            _append_event(
                self.log_path,
                {
                    "schema": "preprocess-pipeline-gpu-selection-v1",
                    "observed_at": _utc_now(),
                    "hostname": socket.gethostname(),
                    "status": "monitor_error",
                    "monitor_phase": status,
                    "monitor_interval_seconds": self.interval_seconds,
                    "allocation_environment": _allocation_environment(),
                    "selected_gpu_uuid": self.selected_gpu_uuid,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "gpus": [],
                },
            )
            print(f"[GPU monitoring] Telemetry query failed: {exc}", flush=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._record("monitoring")

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU usage monitor has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="gpu-usage-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, final_status: str) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._record(final_status)
        self._thread = None


def activate_least_used_gpu(
    *,
    log_path: Path,
    busy_percent: float = GPU_BUSY_PERCENT,
    poll_seconds: float = GPU_POLL_SECONDS,
    query: Callable[[], list[GpuDevice]] = query_gpu_devices,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if not 0.0 < float(busy_percent) <= 100.0:
        raise ValueError("busy_percent must be in (0, 100]")
    if float(poll_seconds) <= 0:
        raise ValueError("poll_seconds must be positive")

    allocation_environment = _allocation_environment()
    while True:
        devices = query()
        candidates = [
            device
            for device in devices
            if device.memory_percent < float(busy_percent)
            and device.utilization_percent is not None
            and device.utilization_percent < float(busy_percent)
        ]
        selected = min(
            candidates,
            key=lambda device: (
                max(device.memory_percent, float(device.utilization_percent or 0.0)),
                device.memory_percent,
                device.index,
            ),
            default=None,
        )
        event: dict[str, object] = {
            "schema": "preprocess-pipeline-gpu-selection-v1",
            "observed_at": _utc_now(),
            "hostname": socket.gethostname(),
            "busy_percent": float(busy_percent),
            "poll_seconds": float(poll_seconds),
            "status": "selected" if selected is not None else "waiting",
            "allocation_environment": allocation_environment,
            "selected_gpu": asdict(selected) if selected is not None else None,
            "gpus": [asdict(device) for device in devices],
        }
        _append_event(Path(log_path), event)
        print("[GPU selection] " + " | ".join(_device_line(device) for device in devices), flush=True)

        if selected is not None:
            # A UUID is stable even when CUDA's integer enumeration differs
            # from nvidia-smi or Slurm's original CUDA_VISIBLE_DEVICES value.
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = selected.uuid
            print(
                f"[GPU selection] Selected physical GPU {selected.index} "
                f"({selected.uuid}); audit log: {Path(log_path)}",
                flush=True,
            )
            return {
                "selected_gpu": asdict(selected),
                "busy_percent": float(busy_percent),
                "allocation_environment": allocation_environment,
                "log_path": str(Path(log_path).resolve()),
            }

        print(
            f"[GPU selection] No GPU is below {float(busy_percent):.1f}% for both "
            f"memory and utilization; waiting {float(poll_seconds):g} seconds before retry.",
            flush=True,
        )
        sleeper(float(poll_seconds))
