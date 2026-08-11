from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import AttemptSpec, BackendStatus, JobRef


class ExecutionBackend(ABC):
    @abstractmethod
    def submit(
        self,
        *,
        run_dir: Path,
        attempt_spec: AttemptSpec,
        dependency_job_refs: list[JobRef] | None = None,
    ) -> list[JobRef]:
        raise NotImplementedError

    @abstractmethod
    def status(self, job_ref: JobRef) -> BackendStatus:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_ref: JobRef) -> None:
        raise NotImplementedError
