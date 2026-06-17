from __future__ import annotations

import os
import platform


WINDOWS_PROCESS_WORKER_LIMIT = 61
REQUESTED_WORKER_COUNT = 128
WORKER_RESERVE = 8


def usable_worker_capacity() -> int:
    cpu_count = os.cpu_count() or 1
    if platform.system().lower() == "windows":
        return min(cpu_count, WINDOWS_PROCESS_WORKER_LIMIT)
    return cpu_count


def default_worker_count() -> int:
    usable_count = usable_worker_capacity()
    if usable_count >= REQUESTED_WORKER_COUNT:
        return REQUESTED_WORKER_COUNT
    return max(1, usable_count - WORKER_RESERVE)


def normalize_worker_count(value: int | float | str | None) -> int:
    default_count = default_worker_count()
    if value is None:
        return default_count
    try:
        requested = int(value)
    except (TypeError, ValueError):
        return default_count
    return max(1, min(requested, default_count))
