from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Callable

from ..models import AttemptSpec, BackendName, BackendStatus, JobRef, StageName
from ..store import atomic_write_json, atomic_write_text, read_json
from .base import ExecutionBackend


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SlurmCapabilities:
    sbatch: str | None
    squeue: str | None
    scancel: str | None
    sacct: str | None
    controller_reachable: bool
    accounting_reachable: bool
    errors: tuple[str, ...] = ()

    def usable(self, *, require_sacct: bool) -> bool:
        required = self.sbatch and self.squeue and self.scancel and self.controller_reachable
        if not required:
            return False
        return bool(self.sacct and self.accounting_reachable) if require_sacct else True

    def to_dict(self) -> dict[str, Any]:
        return {
            "sbatch": self.sbatch,
            "squeue": self.squeue,
            "scancel": self.scancel,
            "sacct": self.sacct,
            "controller_reachable": self.controller_reachable,
            "accounting_reachable": self.accounting_reachable,
            "errors": list(self.errors),
        }


def _probe(command: list[str], runner: CommandRunner, timeout: float) -> tuple[bool, str]:
    try:
        result = runner(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    return True, ""


def detect_slurm_capabilities(
    *, runner: CommandRunner = subprocess.run, timeout: float = 5.0
) -> SlurmCapabilities:
    commands = {name: shutil.which(name) for name in ("sbatch", "squeue", "scancel", "sacct")}
    errors: list[str] = []
    controller_reachable = False
    accounting_reachable = False
    if commands["squeue"]:
        controller_reachable, error = _probe(
            [str(commands["squeue"]), "--noheader", "--user", getpass.getuser()], runner, timeout
        )
        if error:
            errors.append(f"squeue: {error}")
    else:
        errors.append("squeue not found")
    for required in ("sbatch", "scancel"):
        if not commands[required]:
            errors.append(f"{required} not found")
    if commands["sacct"]:
        accounting_reachable, error = _probe(
            [str(commands["sacct"]), "--noheader", "--allocations", "--starttime", "now"],
            runner,
            timeout,
        )
        if error:
            errors.append(f"sacct: {error}")
    else:
        errors.append("sacct not found")
    return SlurmCapabilities(
        sbatch=commands["sbatch"],
        squeue=commands["squeue"],
        scancel=commands["scancel"],
        sacct=commands["sacct"],
        controller_reachable=controller_reachable,
        accounting_reachable=accounting_reachable,
        errors=tuple(errors),
    )


def _walltime_text(minutes: int) -> str:
    days, remaining = divmod(int(minutes), 24 * 60)
    hours, mins = divmod(remaining, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{mins:02d}:00"


def _directive_value(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.:/+@-]+", text):
        raise ValueError(f"Invalid Slurm {name}: {text!r}")
    return text


def _repository_root(run_dir: Path) -> Path:
    run_path = Path(run_dir) / "run.json"
    if run_path.exists():
        configured = read_json(run_path).get("repository_root")
        if configured:
            return Path(str(configured)).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


class SlurmBackend(ExecutionBackend):
    def __init__(
        self,
        *,
        capabilities: SlurmCapabilities,
        runner: CommandRunner = subprocess.run,
        python_executable: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.capabilities = capabilities
        self.runner = runner
        self.python_executable = python_executable or sys.executable
        self.timeout = timeout

    def render_script(
        self,
        *,
        run_dir: Path,
        attempt_spec: AttemptSpec,
        dependency_job_refs: list[JobRef] | None = None,
    ) -> str:
        resource = attempt_spec.resources
        resource.validate(stage=attempt_spec.stage)
        attempt_dir = Path(run_dir) / "stages" / attempt_spec.stage.value / f"attempt-{attempt_spec.attempt:03d}"
        job_name = f"pp-{attempt_spec.run_id[-10:]}-{attempt_spec.stage.value}-a{attempt_spec.attempt}"
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={_directive_value(job_name, name='job name')}",
            f"#SBATCH --cpus-per-task={resource.cpus}",
            f"#SBATCH --mem={resource.memory_mb}M",
            f"#SBATCH --output={shlex.quote(str(attempt_dir / 'stdout.log'))}",
            f"#SBATCH --error={shlex.quote(str(attempt_dir / 'stderr.log'))}",
        ]
        if resource.walltime_minutes is not None:
            lines.append(f"#SBATCH --time={_walltime_text(resource.walltime_minutes)}")
        optional = {
            "partition": resource.partition,
            "account": resource.account,
            "qos": resource.qos,
            "reservation": resource.reservation,
            "constraint": resource.gpu_constraint,
        }
        for name, value in optional.items():
            checked = _directive_value(value, name=name)
            if checked:
                lines.append(f"#SBATCH --{name}={checked}")
        if resource.gpu_count:
            gres_type = _directive_value(resource.gpu_gres_type, name="GPU GRES type")
            if gres_type:
                lines.append(f"#SBATCH --gres=gpu:{gres_type}:{resource.gpu_count}")
            else:
                lines.append(f"#SBATCH --gres=gpu:{resource.gpu_count}")
        if dependency_job_refs:
            dependency_ids = ":".join(
                _directive_value(job.job_id.split(";", 1)[0], name="dependency job ID")
                for job in dependency_job_refs
            )
            lines.append(f"#SBATCH --dependency=afterok:{dependency_ids}")
            lines.append("#SBATCH --kill-on-invalid-dep=yes")
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
        resolved_run_dir = str(Path(run_dir).resolve())
        analysis_config_path = str(Path(run_dir).resolve() / "analysis_config.json")
        lines.extend(
            [
                "set -euo pipefail",
                "export PYTHONUNBUFFERED=1",
                "export OMP_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
                f"if [[ ! -r {shlex.quote(analysis_config_path)} || ! -w {shlex.quote(resolved_run_dir)} ]]; then",
                f"  echo {shlex.quote('Persistent Run workspace is not readable and writable on this Slurm compute node: ' + resolved_run_dir)} >&2",
                "  exit 73",
                "fi",
                f"cd {shlex.quote(str(_repository_root(Path(run_dir))))}",
                "exec " + shlex.join(command),
                "",
            ]
        )
        return "\n".join(lines)

    def submit(
        self,
        *,
        run_dir: Path,
        attempt_spec: AttemptSpec,
        dependency_job_refs: list[JobRef] | None = None,
    ) -> list[JobRef]:
        if not self.capabilities.sbatch:
            raise RuntimeError("sbatch is unavailable")
        attempt_dir = Path(run_dir) / "stages" / attempt_spec.stage.value / f"attempt-{attempt_spec.attempt:03d}"
        script_path = attempt_dir / "job.sbatch"
        script_text = self.render_script(
            run_dir=run_dir,
            attempt_spec=attempt_spec,
            dependency_job_refs=dependency_job_refs,
        )
        atomic_write_text(script_path, script_text)
        intent_path = attempt_dir / "submission_intent.json"
        atomic_write_json(
            intent_path,
            {
                "created_at": _utc_now(),
                "script": str(script_path),
                "dependency_job_refs": [job.to_dict() for job in dependency_job_refs or []],
            },
        )
        try:
            result = self.runner(
                [str(self.capabilities.sbatch), "--parsable", str(script_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            atomic_write_json(
                attempt_dir / "submission_failure.json",
                {
                    "failed_at": _utc_now(),
                    "message": str(exc),
                    "acceptance_ambiguous": True,
                },
            )
            raise RuntimeError(
                "Slurm submission outcome is ambiguous; the Attempt will not fall back to Local"
            ) from exc
        output = result.stdout.strip()
        if result.returncode != 0 or not output:
            atomic_write_json(
                attempt_dir / "submission_failure.json",
                {
                    "failed_at": _utc_now(),
                    "message": (result.stderr or output or f"sbatch exit {result.returncode}").strip(),
                    "acceptance_ambiguous": False,
                },
            )
            raise RuntimeError(
                f"sbatch failed; the Attempt will not fall back to Local: "
                f"{(result.stderr or output).strip()}"
            )
        raw_job_id = output.splitlines()[-1].strip()
        primary_id = raw_job_id.split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", primary_id):
            atomic_write_json(
                attempt_dir / "submission_failure.json",
                {
                    "failed_at": _utc_now(),
                    "message": f"Unrecognized sbatch --parsable output: {raw_job_id}",
                    "acceptance_ambiguous": True,
                },
            )
            raise RuntimeError(
                "Slurm returned an unrecognized job identifier; the Attempt will not fall back to Local"
            )
        return [
            JobRef(
                backend=BackendName.SLURM,
                job_id=raw_job_id,
                submitted_at=_utc_now(),
                metadata={"script": str(script_path), "primary_job_id": primary_id},
            )
        ]

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return self.runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

    def status(self, job_ref: JobRef) -> BackendStatus:
        job_id = str(job_ref.metadata.get("primary_job_id", job_ref.job_id.split(";", 1)[0]))
        if not self.capabilities.squeue:
            return BackendStatus(state="unknown", terminal=True, reason="squeue is unavailable")
        live = self._run(
            [str(self.capabilities.squeue), "--noheader", "--jobs", job_id, "--format=%T|%R"]
        )
        if live.returncode == 0 and live.stdout.strip():
            state, _, reason = live.stdout.strip().splitlines()[0].partition("|")
            normalized = state.strip().lower()
            return BackendStatus(
                state=normalized,
                terminal=False,
                reason=reason.strip(),
            )
        if self.capabilities.sacct and self.capabilities.accounting_reachable:
            accounting = self._run(
                [
                    str(self.capabilities.sacct),
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    job_id,
                    "--format=JobIDRaw,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,ReqMem",
                ]
            )
            if accounting.returncode == 0:
                rows = [line for line in accounting.stdout.splitlines() if line.strip()]
                row = next(
                    (line for line in rows if line.split("|", 1)[0] == job_id),
                    rows[0] if rows else "",
                )
                if row:
                    fields = row.split("|")
                    fields += [""] * (7 - len(fields))
                    parsed_rows = [line.split("|") for line in rows]
                    max_rss = next(
                        (parts[4] for parts in parsed_rows if len(parts) > 4 and parts[4]),
                        fields[4],
                    )
                    state_parts = fields[1].split("+", 1)[0].strip().split()
                    state = state_parts[0].lower() if state_parts else "unknown"
                    active_states = {
                        "pending",
                        "running",
                        "configuring",
                        "completing",
                        "resizing",
                        "requeued",
                        "requeue_fed",
                        "requeue_hold",
                        "signaling",
                        "stage_out",
                        "suspended",
                    }
                    terminal_states = {
                        "boot_fail",
                        "cancelled",
                        "deadline",
                        "failed",
                        "node_fail",
                        "out_of_memory",
                        "preempted",
                        "revoked",
                        "special_exit",
                        "timeout",
                        "completed",
                    }
                    terminal = state in terminal_states
                    if state in active_states or not terminal:
                        terminal = False
                    success = terminal and state == "completed" and fields[2].startswith("0:")
                    return BackendStatus(
                        state=state,
                        terminal=terminal,
                        successful=success if terminal else None,
                        exit_code=fields[2] or None,
                        reason="" if success else state,
                        telemetry={
                            "elapsed": fields[3],
                            "max_rss": max_rss,
                            "allocated_cpus": fields[5],
                            "requested_memory": fields[6],
                        },
                    )
        return BackendStatus(
            state="unknown",
            terminal=True,
            successful=None,
            reason="Job is absent from squeue and no terminal sacct record is available",
        )

    def cancel(self, job_ref: JobRef) -> None:
        if not self.capabilities.scancel:
            raise RuntimeError("scancel is unavailable")
        job_id = str(job_ref.metadata.get("primary_job_id", job_ref.job_id.split(";", 1)[0]))
        result = self._run([str(self.capabilities.scancel), job_id])
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "scancel failed").strip())
