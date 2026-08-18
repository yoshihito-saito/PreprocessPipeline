from __future__ import annotations

from pathlib import Path

from src.execution.session import (
    _claim_blocks_new_run,
    cleanup_preprocess_binary_partials,
)
from src.execution.store import atomic_write_json


def _write_submitted_attempt(run_dir: Path, *, with_failure: bool) -> dict[str, object]:
    atomic_write_json(
        run_dir / "run.json",
        {
            "run_id": "run-test",
            "enabled_stages": ["preprocess"],
            "requested_backend": "local",
            "resolved_backend": "local",
        },
    )
    attempt_dir = run_dir / "stages" / "preprocess" / "attempt-001"
    atomic_write_json(
        attempt_dir / "spec.json",
        {
            "run_id": "run-test",
            "stage": "preprocess",
            "attempt": 1,
            "backend": "local",
            "analysis_sha256": "test",
            "resources": {},
            "upstream_attempts": {},
        },
    )
    job = {
        "backend": "local",
        "job_id": "1234",
        "submitted_at": "test",
        "metadata": {"pid": 1234, "process_start_token": "windows-filetime:test"},
    }
    atomic_write_json(
        attempt_dir / "submitted.json",
        {"stage": "preprocess", "attempt": 1, "submitted_at": "test", "jobs": [job]},
    )
    atomic_write_json(attempt_dir / "started.json", {"started_at": "test"})
    atomic_write_json(
        attempt_dir / "observations" / "latest.json",
        {
            "observed_at": "test",
            "job": job,
            "status": {
                "state": "cancel_failed",
                "terminal": False,
                "successful": None,
                "reason": "identity could not be verified",
            },
        },
    )
    if with_failure:
        atomic_write_json(
            attempt_dir / "failure.json",
            {
                "status": "failed",
                "finished_at": "test",
                "message": "interrupted",
                "validation": {"passed": False},
            },
        )
    return {"kind": "run", "run_id": "run-test", "run_dir": str(run_dir)}


def test_claim_blocks_submitted_attempt_with_nonterminal_observation(tmp_path: Path) -> None:
    claim = _write_submitted_attempt(tmp_path / "active", with_failure=False)

    assert _claim_blocks_new_run(claim) is True


def test_failure_fact_releases_stale_nonterminal_claim(tmp_path: Path) -> None:
    claim = _write_submitted_attempt(tmp_path / "failed", with_failure=True)

    assert _claim_blocks_new_run(claim) is False


def test_preprocess_partial_cleanup_is_nonrecursive_and_target_scoped(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    nested = session / "nested"
    nested.mkdir(parents=True)
    removable = [
        session / "session.dat.partial-0123456789ab",
        session / "session_raw.dat.partial-abcdef012345",
        session / "session.lfp.partial-1234567890ab",
        session / ".analogin.dat.partial-0123456789abcdef0123456789abcdef",
    ]
    for path in removable:
        path.write_bytes(b"partial")
    canonical = session / "session.dat"
    canonical.write_bytes(b"complete")
    unrelated = session / "notes.dat.partial-0123456789ab"
    unrelated.write_bytes(b"user")
    malformed = session / "session.dat.partial-not-a-token"
    malformed.write_bytes(b"user")
    nested_partial = nested / "session.dat.partial-0123456789ab"
    nested_partial.write_bytes(b"nested")

    report = cleanup_preprocess_binary_partials(session, "session")

    assert {Path(value).name for value in report["removed"]} == {
        path.name for path in removable
    }
    assert report["removed_bytes"] == sum(len(b"partial") for _ in removable)
    assert report["errors"] == []
    assert all(not path.exists() for path in removable)
    assert canonical.read_bytes() == b"complete"
    assert unrelated.read_bytes() == b"user"
    assert malformed.read_bytes() == b"user"
    assert nested_partial.read_bytes() == b"nested"
