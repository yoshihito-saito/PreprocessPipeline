from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "explicit",
    [
        {},
        {"PYTORCH_ALLOC_CONF": "expandable_segments:False"},
        {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"},
        {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
        {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
        {"PYTORCH_ALLOC_CONF": ""},
        {"PYTORCH_CUDA_ALLOC_CONF": ""},
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        },
    ],
)
def test_fresh_worker_import_configures_allocator_before_torch(explicit) -> None:
    env = os.environ.copy()
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        env.pop(key, None)
    env.update(explicit)
    # Exercise the same import path used by both local and Slurm workers.
    script = """
import json, os, sys
import src.execution.worker
print(json.dumps({
    'allocator': {k: os.environ[k] for k in
                  ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF') if k in os.environ},
    'torch_imported': 'torch' in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    assert observed["allocator"] == (
        explicit or {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    )
    assert observed["torch_imported"] is False


def _sorting_store(tmp_path):
    from src.execution.models import (
        AnalysisConfig, AttemptSpec, BackendName, ExecutionConfig,
        RequestedBackend, ResourceSpec, StageName,
    )
    from src.execution.store import RunStore

    store = RunStore(tmp_path / "run")
    analysis = AnalysisConfig.create({})
    resources = ResourceSpec(1, 1024, 10, gpu_count=1)
    store.initialize(
        run={"run_id": "allocator-test", "enabled_stages": ["sorting"],
             "requested_backend": "local", "resolved_backend": "local"},
        analysis=analysis,
        execution=ExecutionConfig(
            requested_backend=RequestedBackend.LOCAL,
            resolved_backend=BackendName.LOCAL,
            workspace=str(tmp_path), resources={"sorting": resources},
        ),
    )
    store.create_attempt(AttemptSpec(
        run_id="allocator-test", stage=StageName.SORTING, attempt=1,
        backend=BackendName.LOCAL, resources=resources, analysis_sha256=analysis.sha256,
    ))
    return store


def test_sorting_worker_records_allocator_even_when_sorting_fails(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from src.execution import worker
    from src.execution.models import StageName

    store = _sorting_store(tmp_path)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(worker, "_verify_analysis_artifacts", lambda *_args: None)

    def fail_sorting(*_args):
        started = store.read_attempt_fact(StageName.SORTING, 1, "started.json")
        assert started["allocator_environment"] == {
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": None,
        }
        raise RuntimeError("simulated sorter failure")

    monkeypatch.setattr(worker, "run_stage", fail_sorting)
    assert worker.execute(store.run_dir, StageName.SORTING, 1) == 1
    failure = store.read_attempt_fact(StageName.SORTING, 1, "failure.json")
    assert "simulated sorter failure" in str(failure)
    assert '[Torch allocator environment]' in capsys.readouterr().out

