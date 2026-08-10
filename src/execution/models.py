from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1


class RequestedBackend(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    SLURM = "slurm"


class BackendName(str, Enum):
    LOCAL = "local"
    SLURM = "slurm"


class StageName(str, Enum):
    PREPROCESS = "preprocess"
    SORTING = "sorting"
    POSTPROCESS = "postprocess"


class StageStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    LOST = "lost"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class ResourceSpec:
    cpus: int
    memory_mb: int
    walltime_minutes: int | None
    gpu_count: int = 0
    gpu_gres_type: str = ""
    gpu_constraint: str = ""
    partition: str = ""
    account: str = ""
    qos: str = ""
    reservation: str = ""

    def validate(self, *, stage: StageName | None = None) -> None:
        if int(self.cpus) < 1:
            raise ValueError("ResourceSpec.cpus must be at least 1")
        if int(self.memory_mb) < 1:
            raise ValueError("ResourceSpec.memory_mb must be at least 1")
        if self.walltime_minutes is not None and int(self.walltime_minutes) < 1:
            raise ValueError(
                "ResourceSpec.walltime_minutes must be None or at least 1"
            )
        if int(self.gpu_count) < 0:
            raise ValueError("ResourceSpec.gpu_count cannot be negative")
        if stage in {StageName.PREPROCESS, StageName.POSTPROCESS} and self.gpu_count != 0:
            raise ValueError(f"{stage.value} must not request a GPU")
        if stage == StageName.SORTING and self.gpu_count != 1:
            raise ValueError("sorting must request exactly one GPU")
        for name in (
            "gpu_gres_type",
            "gpu_constraint",
            "partition",
            "account",
            "qos",
            "reservation",
        ):
            value = str(getattr(self, name))
            if any(ch in value for ch in ("\n", "\r", "\x00")):
                raise ValueError(f"ResourceSpec.{name} contains an invalid control character")
            if value and not re.fullmatch(r"[A-Za-z0-9_.:/+@-]+", value):
                raise ValueError(f"ResourceSpec.{name} contains invalid Slurm syntax")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResourceSpec":
        return cls(**value)


@dataclass(frozen=True)
class AnalysisConfig:
    settings: dict[str, Any]
    sha256: str
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def calculate_hash(
        settings: dict[str, Any], artifact_sha256: dict[str, str] | None = None
    ) -> str:
        encoded = json.dumps(
            {"settings": settings, "artifact_sha256": artifact_sha256 or {}},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        settings: dict[str, Any],
        *,
        artifact_sha256: dict[str, str] | None = None,
    ) -> "AnalysisConfig":
        artifacts = dict(artifact_sha256 or {})
        return cls(
            settings=settings,
            artifact_sha256=artifacts,
            sha256=cls.calculate_hash(settings, artifacts),
        )

    def validate(self) -> None:
        actual = self.calculate_hash(self.settings, self.artifact_sha256)
        if actual != self.sha256:
            raise ValueError("AnalysisConfig hash does not match its immutable settings snapshot")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalysisConfig":
        config = cls(
            settings=dict(value["settings"]),
            sha256=str(value["sha256"]),
            artifact_sha256={
                str(name): str(checksum)
                for name, checksum in value.get("artifact_sha256", {}).items()
            },
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class ExecutionConfig:
    requested_backend: RequestedBackend
    resolved_backend: BackendName
    workspace: str
    resources: dict[str, ResourceSpec]
    matlab_path: str = ""
    shared_workspace_acknowledged: bool = False
    require_sacct: bool = True
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if not str(self.workspace).strip():
            raise ValueError("ExecutionConfig.workspace is required")
        for stage in StageName:
            if stage.value not in self.resources:
                raise ValueError(f"Missing ResourceSpec for {stage.value}")
            self.resources[stage.value].validate(stage=stage)

    def resource_for(self, stage: StageName) -> ResourceSpec:
        return self.resources[stage.value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested_backend": self.requested_backend.value,
            "resolved_backend": self.resolved_backend.value,
            "workspace": self.workspace,
            "matlab_path": self.matlab_path,
            "shared_workspace_acknowledged": self.shared_workspace_acknowledged,
            "require_sacct": self.require_sacct,
            "resources": {name: spec.to_dict() for name, spec in self.resources.items()},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExecutionConfig":
        config = cls(
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            requested_backend=RequestedBackend(value["requested_backend"]),
            resolved_backend=BackendName(value["resolved_backend"]),
            workspace=str(value["workspace"]),
            matlab_path=str(value.get("matlab_path", "")),
            shared_workspace_acknowledged=bool(
                value.get("shared_workspace_acknowledged", False)
            ),
            require_sacct=bool(value.get("require_sacct", True)),
            resources={
                str(name): ResourceSpec.from_dict(spec)
                for name, spec in dict(value["resources"]).items()
            },
        )
        config.validate()
        return config


@dataclass(frozen=True)
class AttemptSpec:
    run_id: str
    stage: StageName
    attempt: int
    backend: BackendName
    resources: ResourceSpec
    analysis_sha256: str
    upstream_attempts: dict[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.attempt < 1:
            raise ValueError("Attempt number must be at least 1")
        self.resources.validate(stage=self.stage)
        for stage, attempt in self.upstream_attempts.items():
            StageName(stage)
            if int(attempt) < 1:
                raise ValueError("Upstream attempt number must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage.value,
            "attempt": self.attempt,
            "backend": self.backend.value,
            "resources": self.resources.to_dict(),
            "analysis_sha256": self.analysis_sha256,
            "upstream_attempts": dict(self.upstream_attempts),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttemptSpec":
        spec = cls(
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            run_id=str(value["run_id"]),
            stage=StageName(value["stage"]),
            attempt=int(value["attempt"]),
            backend=BackendName(value["backend"]),
            resources=ResourceSpec.from_dict(value["resources"]),
            analysis_sha256=str(value["analysis_sha256"]),
            upstream_attempts={str(k): int(v) for k, v in value.get("upstream_attempts", {}).items()},
        )
        spec.validate()
        return spec


@dataclass(frozen=True)
class JobRef:
    backend: BackendName
    job_id: str
    submitted_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "job_id": self.job_id,
            "submitted_at": self.submitted_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobRef":
        return cls(
            backend=BackendName(value["backend"]),
            job_id=str(value["job_id"]),
            submitted_at=str(value["submitted_at"]),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class BackendStatus:
    state: str
    terminal: bool
    successful: bool | None = None
    exit_code: str | None = None
    reason: str = ""
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BackendStatus":
        return cls(**value)
