from __future__ import annotations

from pathlib import Path
import json

from src.execution.backends import SlurmCapabilities
from src.execution.models import (
    AnalysisConfig,
    BackendName,
    ExecutionConfig,
    RequestedBackend,
    ResourceSpec,
    StageName,
)
from src.preprocess.gui.app import (
    CONFIG_DIR,
    MainWindow,
    _default_config_has_backend_choice,
    _has_slurm_server_commands,
    _move_local_output_to_storage,
    _resolve_cell_explorer_source_basepath,
)
from src.preprocess.gui.config_model import (
    PipelineGuiSettings,
    load_local_session_resume,
    resolve_existing_session_settings,
)
from src.preprocess.gui.run_pipeline import _json_safe


def test_only_an_explicit_saved_backend_suppresses_environment_default(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"preprocess": {}}), encoding="utf-8")
    explicit = tmp_path / "explicit.json"
    explicit.write_text(
        json.dumps({"execution": {"requested_backend": "auto"}}), encoding="utf-8"
    )

    assert _default_config_has_backend_choice(tmp_path / "missing.json") is False
    assert _default_config_has_backend_choice(legacy) is False
    assert _default_config_has_backend_choice(explicit) is True


def _write_multi_day_manifest(
    local_output: Path,
    server_basepath: Path,
    *,
    name: str | None = None,
    staged_names: tuple[str, ...] = ("001_Day14_epoch", "002_Day15_epoch"),
) -> None:
    payload = {
        "schema_version": 1,
        "name": name or local_output.name,
        "server_basepath": str(server_basepath),
        "subepochs": [
            {"staged_subepoch_path": str(server_basepath / staged_name)}
            for staged_name in staged_names
        ],
    }
    (local_output / "multi_day_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_cell_explorer_source_prefers_multi_day_manifest_over_first_day(
    tmp_path: Path,
) -> None:
    first_day = tmp_path / "raw" / "Day14"
    first_day.mkdir(parents=True)
    local_output = tmp_path / "sorting_temp" / "multiday_Day14_to_Day15"
    local_output.mkdir(parents=True)
    staged_root = tmp_path / "staged" / local_output.name
    for staged_name in ("001_Day14_epoch", "002_Day15_epoch"):
        (staged_root / staged_name).mkdir(parents=True)
    _write_multi_day_manifest(local_output, staged_root)
    settings = PipelineGuiSettings(
        basepath=str(first_day),
        local_root=str(local_output.parent),
        multi_day_enabled=True,
        multi_day_name=local_output.name,
    )

    resolved = _resolve_cell_explorer_source_basepath(settings, local_output)

    assert resolved == staged_root.resolve()


def test_cell_explorer_source_uses_manifest_when_existing_multiday_is_reopened(
    tmp_path: Path,
) -> None:
    first_day = tmp_path / "raw" / "Day14"
    first_day.mkdir(parents=True)
    local_output = tmp_path / "sorting_temp" / "multiday_Day14_to_Day15"
    local_output.mkdir(parents=True)
    staged_root = tmp_path / "staged" / local_output.name
    for staged_name in ("001_Day14_epoch", "002_Day15_epoch"):
        (staged_root / staged_name).mkdir(parents=True)
    _write_multi_day_manifest(local_output, staged_root)
    settings = PipelineGuiSettings(
        basepath=str(local_output),
        source_basepath=str(first_day),
        existing_session_dir=str(local_output),
        multi_day_enabled=False,
    )

    resolved = _resolve_cell_explorer_source_basepath(settings, local_output)

    assert resolved == staged_root.resolve()


def test_cell_explorer_source_without_manifest_preserves_single_day_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw" / "sessionA"
    source.mkdir(parents=True)
    local_output = tmp_path / "sorting_temp" / source.name
    local_output.mkdir(parents=True)
    settings = PipelineGuiSettings(basepath=str(source), local_root=str(local_output.parent))

    resolved = _resolve_cell_explorer_source_basepath(settings, local_output)

    assert resolved == source.resolve()


def test_cell_explorer_source_rejects_manifest_for_another_session(
    tmp_path: Path,
) -> None:
    import pytest

    local_output = tmp_path / "sorting_temp" / "multiday_Day14_to_Day15"
    local_output.mkdir(parents=True)
    staged_root = tmp_path / "staged" / local_output.name
    for staged_name in ("001_Day14_epoch", "002_Day15_epoch"):
        (staged_root / staged_name).mkdir(parents=True)
    _write_multi_day_manifest(local_output, staged_root, name="different_session")

    with pytest.raises(ValueError, match="does not match the selected local session"):
        _resolve_cell_explorer_source_basepath(PipelineGuiSettings(), local_output)


def test_cell_explorer_source_rejects_unsupported_manifest_schema(
    tmp_path: Path,
) -> None:
    import pytest

    local_output = tmp_path / "sorting_temp" / "multiday_Day14_to_Day15"
    local_output.mkdir(parents=True)
    staged_root = tmp_path / "staged" / local_output.name
    for staged_name in ("001_Day14_epoch", "002_Day15_epoch"):
        (staged_root / staged_name).mkdir(parents=True)
    _write_multi_day_manifest(local_output, staged_root)
    manifest_path = local_output / "multi_day_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported multi-day manifest schema"):
        _resolve_cell_explorer_source_basepath(PipelineGuiSettings(), local_output)


def test_cell_explorer_source_rejects_missing_staged_subepoch(tmp_path: Path) -> None:
    import pytest

    local_output = tmp_path / "sorting_temp" / "multiday_Day14_to_Day15"
    local_output.mkdir(parents=True)
    staged_root = tmp_path / "staged" / local_output.name
    (staged_root / "001_Day14_epoch").mkdir(parents=True)
    _write_multi_day_manifest(local_output, staged_root)

    with pytest.raises(FileNotFoundError, match="002_Day15_epoch"):
        _resolve_cell_explorer_source_basepath(PipelineGuiSettings(), local_output)


def test_execution_controls_are_nested_under_ephys(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    unavailable = SlurmCapabilities(
        None, None, None, None, False, False, ("Slurm unavailable",)
    )
    monkeypatch.setattr(
        "src.preprocess.gui.app.detect_slurm_capabilities",
        lambda **_kwargs: unavailable,
    )
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
            "Ephys",
            "Behavior",
        ]
        assert [
            window.ephys_tabs.tabText(index) for index in range(window.ephys_tabs.count())
        ] == ["Preprocess", "Postprocess", "Run"]
        run_panel = window.ephys_tabs.widget(window._ephys_run_tab_index).widget()
        assert run_panel.isAncestorOf(window.run_all)
        assert run_panel.isAncestorOf(window.run_pre)
        assert run_panel.isAncestorOf(window.run_post)
        assert window.run_pre.text() == "Preprocess only"
        assert window.run_post.text() == "Postprocess only"
        assert not hasattr(window, "execution_workspace")
        assert not hasattr(window, "execution_matlab_path")
        assert not hasattr(window, "execution_shared_workspace_ack")
        assert not hasattr(window, "execution_require_sacct")
        assert set(window._resource_widgets["sorting"]) == {
            "cpus",
            "memory_gib",
            "gpu_count",
        }
        assert window._resource_widgets["sorting"]["gpu_count"].text() == "1 (auto)"
        assert (
            window._resource_widgets["postprocess"]["cpus"].accessibleName()
            == "Postprocess CPU cores"
        )
        assert (
            window._resource_widgets["postprocess"]["memory_gib"].accessibleName()
            == "Postprocess host memory in GiB"
        )
        assert not hasattr(window, "execution_stage_table")
        assert not hasattr(window, "execution_log_select")
        assert not hasattr(window, "execution_cancel_run")
        assert not hasattr(window, "execution_cancel_stage")
        assert not hasattr(window, "execution_retry_stage")
        assert set(window.execution_stage_status_labels) == {
            "preprocess",
            "sorting",
            "postprocess",
        }

        window.local_root.setText(str(tmp_path / "sorting_temp"))
        settings = window._collect_settings()
        assert settings.execution.workspace == str((tmp_path / "sorting_temp").resolve())
        assert settings.execution.matlab_path == ""
        assert settings.execution.require_sacct is True
        assert settings.execution.sorting.walltime_minutes is None
        assert settings.execution.sorting.gpu_count == 1

        window.resize(1200, 800)
        window.ephys_tabs.setCurrentIndex(window._ephys_run_tab_index)
        window.show()
        application.processEvents()
        viewport = window.execution_scroll_area.viewport()
        assert window.execution_scroll_area.horizontalScrollBar().maximum() == 0
        assert window.execution_scroll_area.widget().width() <= viewport.width()
        postprocess_cpu = window._resource_widgets["postprocess"]["cpus"]
        window.execution_scroll_area.ensureWidgetVisible(postprocess_cpu)
        application.processEvents()
        postprocess_position = postprocess_cpu.mapTo(viewport, QPoint(0, 0))
        assert 0 <= postprocess_position.y() < viewport.height()
        assert (
            postprocess_position.x() + postprocess_cpu.width()
            <= viewport.width()
        )

        legacy = PipelineGuiSettings(local_root=str(tmp_path / "legacy-sorting-temp"))
        legacy.execution.sorting.walltime_minutes = 90
        window._apply_settings(legacy)
        assert window._collect_settings().execution.sorting.walltime_minutes is None

        window._fit_to_available_screen()
        available = window.screen().availableGeometry()
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert window.center_scroll_area.verticalScrollBar().maximum() == 0
        log_position = window.log.mapTo(window, QPoint(0, 0))
        assert 0 <= log_position.y() < window.height()
        assert log_position.y() + window.log.height() <= window.height()
    finally:
        window.close()
        application.processEvents()


def test_persistent_progress_uses_stage_rows_main_log_and_force_stop(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    run_dir = tmp_path / "workspace" / ".pipeline" / "run-test"
    session_dir = tmp_path / "workspace" / "session"
    output_dir = session_dir / "Kilosort_test"
    attempt_dirs = {
        stage: run_dir / "stages" / stage / "attempt-001"
        for stage in ("preprocess", "sorting", "postprocess")
    }
    for attempt_dir in attempt_dirs.values():
        attempt_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    (attempt_dirs["preprocess"] / "stdout.log").write_text(
        "preprocess finished\n", encoding="utf-8"
    )
    (attempt_dirs["sorting"] / "stdout.log").write_text(
        "sorting started\n", encoding="utf-8"
    )
    (attempt_dirs["sorting"] / "stderr.log").write_text(
        "sorting warning\n", encoding="utf-8"
    )
    kilosort_log = output_dir / "sorter_output" / "kilosort.log"
    kilosort_log.parent.mkdir(parents=True)
    kilosort_log.write_text("batch 10/100\n", encoding="utf-8")
    (session_dir / "sorter_partition_manifest.json").write_text(
        json.dumps(
            {
                "partitions": [
                    {"output_folder": str(output_dir), "status": "running"}
                ]
            }
        ),
        encoding="utf-8",
    )
    state = {
        "run_id": "run-test",
        "status": "running",
        "resolved_backend": "slurm",
        "stages": {
            "preprocess": {
                "enabled": True,
                "selected_attempt": 1,
                "status": "completed",
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "completed",
                        "jobs": [{"job_id": "100"}],
                        "result": {
                            "outputs": {
                                "preprocess_result": {
                                    "local_output_dir": str(session_dir)
                                }
                            }
                        },
                    }
                ],
            },
            "sorting": {
                "enabled": True,
                "selected_attempt": 1,
                "status": "running",
                "attempts": [
                    {"attempt": 1, "status": "running", "jobs": [{"job_id": "101"}]}
                ],
            },
            "postprocess": {
                "enabled": True,
                "selected_attempt": 1,
                "status": "submitted",
                "attempts": [
                    {
                        "attempt": 1,
                        "status": "submitted",
                        "jobs": [{"job_id": "102"}],
                        "latest_observation": {
                            "status": {"terminal": False, "reason": "(Dependency)"}
                        },
                    }
                ],
            },
        },
    }
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    window = MainWindow()
    try:
        window._set_active_run(run_dir)
        window._flush_log_buffer()
        application.processEvents()
        assert window.execution_stage_status_labels["preprocess"].text() == "Completed"
        assert window.execution_stage_status_labels["sorting"].text() == "Running"
        assert (
            window.execution_stage_status_labels["postprocess"].text()
            == "Waiting for previous Stage"
        )
        assert window.execution_stage_job_labels["sorting"].text() == "101"
        assert window.force_stop.isEnabled()
        assert window.move_outputs.isEnabled() is False
        assert window.browse_move_storage.isEnabled() is False
        log_text = window.log.toPlainText()
        assert "preprocess finished" in log_text
        assert "sorting started" in log_text
        assert "sorting warning" in log_text
        assert "batch 10/100" in log_text

        with (attempt_dirs["sorting"] / "stdout.log").open("a", encoding="utf-8") as handle:
            handle.write("sorting continues\n")
        window._refresh_persistent_run_monitor()
        window._flush_log_buffer()
        application.processEvents()
        log_text = window.log.toPlainText()
        assert log_text.count("sorting started") == 1
        assert log_text.count("sorting continues") == 1

        raw_session = tmp_path / "raw" / "session"
        raw_session.mkdir(parents=True)
        window.basepath.setText(str(raw_session))
        window.local_root.setText(str(tmp_path / "workspace"))
        (session_dir / ".pipeline-active-run.json").write_text(
            json.dumps({"kind": "run", "run_dir": str(run_dir)}),
            encoding="utf-8",
        )
        assert window._discover_active_persistent_run() == run_dir.resolve()

        stop_requests: list[bool] = []
        monkeypatch.setattr(
            window,
            "_request_persistent_run_cancel",
            lambda: stop_requests.append(True),
        )
        window._force_stop_process()
        assert stop_requests == [True]
    finally:
        window.close()
        application.processEvents()


def test_persistent_monitor_handles_confirmed_cancel_and_session_switch(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    workspace = tmp_path / "workspace"
    session_a = workspace / "session-a"
    session_b = workspace / "session-b"
    run_a = workspace / ".pipeline" / "run-a"
    run_b = workspace / ".pipeline" / "run-b"
    for session, run in ((session_a, run_a), (session_b, run_b)):
        session.mkdir(parents=True)
        run.mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps(
                {
                    "session_claim_path": str(session / ".pipeline-active-run.json"),
                    "session_output_dir": str(session),
                }
            ),
            encoding="utf-8",
        )

    cancelled_attempt = {
        "attempt": 1,
        "status": "cancelled",
        "cancel_requested": True,
        "jobs": [{"job_id": "200"}],
        "latest_observation": {
            "status": {
                "state": "unknown",
                "terminal": True,
                "successful": None,
            }
        },
    }
    cancelled_stage = {
        "enabled": True,
        "selected_attempt": 1,
        "status": "cancelled",
        "attempts": [cancelled_attempt],
    }
    cancelled_state = {
        "run_id": "run-a",
        "status": "cancelled",
        "resolved_backend": "local",
        "stages": {
            "preprocess": cancelled_stage,
            "sorting": {"enabled": False, "status": "pending", "attempts": []},
            "postprocess": {"enabled": False, "status": "pending", "attempts": []},
        },
    }
    (run_a / "state.json").write_text(json.dumps(cancelled_state), encoding="utf-8")
    active_b_state = {
        "run_id": "run-b",
        "status": "running",
        "resolved_backend": "slurm",
        "stages": {
            "preprocess": {
                "enabled": True,
                "selected_attempt": 1,
                "status": "running",
                "attempts": [
                    {"attempt": 1, "status": "running", "jobs": [{"job_id": "300"}]}
                ],
            },
            "sorting": {"enabled": False, "status": "pending", "attempts": []},
            "postprocess": {"enabled": False, "status": "pending", "attempts": []},
        },
    }
    (run_b / "state.json").write_text(json.dumps(active_b_state), encoding="utf-8")
    (session_b / ".pipeline-active-run.json").write_text(
        json.dumps({"kind": "run", "run_dir": str(run_b)}), encoding="utf-8"
    )

    window = MainWindow()
    try:
        window.basepath.setText(str(tmp_path / "raw" / "session-a"))
        window.local_root.setText(str(workspace))
        window._set_active_run(run_a, session_dir=session_a)
        assert window._human_stage_status(cancelled_stage) == "Cancelled"
        assert window._persistent_state_has_active_work(cancelled_state) is False
        assert window.force_stop.isEnabled() is False
        assert window.move_outputs.isEnabled() is True

        stop_targets: list[Path | None] = []
        monkeypatch.setattr(
            window,
            "_request_persistent_run_cancel",
            lambda: stop_targets.append(window._active_run_dir),
        )
        window.basepath.setText(str(tmp_path / "raw" / "session-b"))
        window._force_stop_process()

        assert window._active_run_dir == run_b.resolve()
        assert window.execution_run_status.text().startswith("Running on Slurm")
        assert window.force_stop.isEnabled() is True
        assert stop_targets == [run_b.resolve()]
    finally:
        window.close()
        application.processEvents()


def test_fresh_server_gui_defaults_to_explicit_slurm(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    available = SlurmCapabilities(
        "/usr/bin/sbatch",
        "/usr/bin/squeue",
        "/usr/bin/scancel",
        "/usr/bin/sacct",
        False,
        False,
        ("controller temporarily unavailable",),
    )
    monkeypatch.setattr(
        "src.preprocess.gui.app.detect_slurm_capabilities",
        lambda **_kwargs: available,
    )
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert _has_slurm_server_commands(available, require_sacct=True) is True
        window.execution_backend.setCurrentIndex(
            window.execution_backend.findData(RequestedBackend.AUTO.value)
        )
        window._environment_backend_default_pending = True
        window._apply_environment_backend_default(available)
        assert window.execution_backend.currentData() == RequestedBackend.SLURM.value

        # Once settings have been loaded or the environment default consumed,
        # later capability refreshes must not replace the user's selection.
        window.execution_backend.setCurrentIndex(
            window.execution_backend.findData(RequestedBackend.AUTO.value)
        )
        window._apply_environment_backend_default(available)
        assert window.execution_backend.currentData() == RequestedBackend.AUTO.value

        no_clients = SlurmCapabilities(
            None, None, None, None, False, False, ("Slurm unavailable",)
        )
        window._environment_backend_default_pending = True
        window._apply_environment_backend_default(no_clients)
        assert window.execution_backend.currentData() == RequestedBackend.AUTO.value
    finally:
        window.close()
        application.processEvents()


def test_loading_legacy_config_reapplies_server_slurm_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    server_capabilities = SlurmCapabilities(
        "/usr/bin/sbatch",
        "/usr/bin/squeue",
        "/usr/bin/scancel",
        "/usr/bin/sacct",
        False,
        False,
        ("controller temporarily unavailable",),
    )
    monkeypatch.setattr(
        "src.preprocess.gui.app.detect_slurm_capabilities",
        lambda **_kwargs: server_capabilities,
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"preprocess": {}}), encoding="utf-8")
    explicit_auto = tmp_path / "explicit-auto.json"
    explicit_auto.write_text(
        json.dumps({"execution": {"requested_backend": "auto"}}), encoding="utf-8"
    )
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window._slurm_capabilities = server_capabilities
        selected_paths = iter((str(legacy), str(explicit_auto)))
        monkeypatch.setattr(window, "_select_open_file", lambda *_args: next(selected_paths))

        window._load_config()
        assert window.execution_backend.currentData() == RequestedBackend.SLURM.value

        window._load_config()
        assert window.execution_backend.currentData() == RequestedBackend.AUTO.value
    finally:
        window.close()
        application.processEvents()


def test_load_config_opens_project_config_directory(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(
        "src.preprocess.gui.app.detect_slurm_capabilities",
        lambda **_kwargs: SlurmCapabilities(
            None, None, None, None, False, False, ("Slurm unavailable",)
        ),
    )
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    captured: list[str] = []
    try:
        monkeypatch.setattr(
            window,
            "_select_open_file",
            lambda _title, start, _filter: captured.append(start) or "",
        )
        window._load_config()
        assert captured == [str(CONFIG_DIR)]
    finally:
        window.close()
        application.processEvents()


def _write_session_xml(basepath: Path, *, sample_rate: float = 20000.0, n_channels: int = 128) -> None:
    (basepath / f"{basepath.name}.xml").write_text(
        f"""
<session>
  <acquisitionSystem>
    <nChannels>{n_channels}</nChannels>
    <sampleRate>{sample_rate}</sampleRate>
  </acquisitionSystem>
  <anatomicalDescription><channelGroups /></anatomicalDescription>
</session>
""".strip(),
        encoding="utf-8",
    )


def test_partial_persistent_session_recovers_raw_source_from_run_claim(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "sessionA"
    source.mkdir(parents=True)
    session = tmp_path / "sorting_temp" / "sessionA"
    session.mkdir(parents=True)
    (session / "sessionA.dat").write_bytes(b"partial")
    (session / "sessionA.session.mat").write_bytes(b"partial")
    run_dir = tmp_path / "workspace" / ".pipeline" / "run-test"
    run_dir.mkdir(parents=True)
    (run_dir / "analysis_config.json").write_text(
        json.dumps({"settings": {"basepath": str(source), "source_basepath": ""}}),
        encoding="utf-8",
    )
    (session / ".pipeline-active-run.json").write_text(
        json.dumps({"run_dir": str(run_dir)}), encoding="utf-8"
    )
    settings = PipelineGuiSettings(basepath=str(session), local_root=str(tmp_path / "unused"))

    assert resolve_existing_session_settings(settings) is True
    assert settings.preprocess_source_path == source.resolve()
    assert settings.local_output_dir == session.resolve()


def _resume_execution(workspace: Path) -> ExecutionConfig:
    resources = {
        StageName.PREPROCESS.value: ResourceSpec(3, 8192, None),
        StageName.SORTING.value: ResourceSpec(4, 16384, None, gpu_count=1),
        StageName.POSTPROCESS.value: ResourceSpec(2, 4096, None),
    }
    return ExecutionConfig(
        requested_backend=RequestedBackend.SLURM,
        resolved_backend=BackendName.SLURM,
        workspace=str(workspace),
        resources=resources,
        matlab_path="/opt/matlab",
        shared_workspace_acknowledged=True,
        require_sacct=False,
    )


def _write_active_resume_session(
    session: Path,
    settings: PipelineGuiSettings,
) -> Path:
    session.mkdir(parents=True)
    workspace = session.parent
    run_dir = workspace / ".pipeline" / "run-resume-test"
    run_dir.mkdir(parents=True)
    analysis = AnalysisConfig.create(json.loads(settings.to_json()))
    execution = _resume_execution(workspace)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "session_output_dir": str(session.resolve()),
                "analysis_sha256": analysis.sha256,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis_config.json").write_text(
        json.dumps(analysis.to_dict()), encoding="utf-8"
    )
    (run_dir / "execution_config.json").write_text(
        json.dumps(execution.to_dict()), encoding="utf-8"
    )
    (session / ".pipeline-active-run.json").write_text(
        json.dumps(
            {
                "kind": "run",
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "session_dir": str(session.resolve()),
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_local_session_resume_restores_multiday_analysis_and_execution(
    tmp_path: Path,
) -> None:
    raw_days = [tmp_path / "raw" / "day1", tmp_path / "raw" / "day2"]
    for path in raw_days:
        path.mkdir(parents=True)
    session = tmp_path / "sorting_temp" / "multiday_day1_to_day2"
    settings = PipelineGuiSettings(
        basepath=str(raw_days[0]),
        local_root=str(session.parent),
        multi_day_enabled=True,
        multi_day_session_paths=[str(path) for path in raw_days],
        multi_day_selected_subepoch_paths=[
            str(raw_days[0] / "Record Node 101"),
            str(raw_days[1] / "Record Node 101"),
        ],
        multi_day_name=session.name,
    )
    settings.preprocess.analog_inputs = True
    run_dir = _write_active_resume_session(session, settings)

    recovered = load_local_session_resume(session)

    assert recovered.run_dir == run_dir.resolve()
    assert recovered.metadata_source == "persistent_run"
    assert recovered.settings.multi_day_enabled is True
    assert recovered.settings.multi_day_session_paths == [str(path) for path in raw_days]
    assert recovered.settings.multi_day_selected_subepoch_paths == [
        str(raw_days[0] / "Record Node 101"),
        str(raw_days[1] / "Record Node 101"),
    ]
    assert recovered.settings.preprocess.analog_inputs is True
    assert recovered.settings.execution.requested_backend == RequestedBackend.SLURM.value
    assert recovered.settings.execution.require_sacct is False
    assert recovered.settings.execution.sorting.memory_mb == 16384
    assert recovered.settings.preprocess.preprocess_worker_count == 3
    assert recovered.settings.preprocess.sorter_worker_count == 4
    assert recovered.settings.postprocess.worker_count == 2
    assert recovered.settings.local_output_dir == session.resolve()


def test_local_session_resume_uses_completed_record_for_single_day(tmp_path: Path) -> None:
    import yaml

    raw = tmp_path / "raw" / "day1"
    raw.mkdir(parents=True)
    session = tmp_path / "sorting_temp" / raw.name
    session.mkdir(parents=True)
    settings = PipelineGuiSettings(basepath=str(raw), local_root=str(session.parent))
    settings.preprocess.digital_inputs = False
    analysis = AnalysisConfig.create(json.loads(settings.to_json()))
    execution = _resume_execution(session.parent)
    (session / "preprocess_run.yaml").write_text(
        yaml.safe_dump(
            {
                "session_output_dir": str(session.resolve()),
                "analysis": analysis.to_dict(),
                "execution": execution.to_dict(),
            }
        ),
        encoding="utf-8",
    )

    recovered = load_local_session_resume(session)

    assert recovered.run_dir is None
    assert recovered.metadata_source == "preprocess_run.yaml"
    assert recovered.settings.multi_day_enabled is False
    assert recovered.settings.basename == "day1"
    assert recovered.settings.preprocess.digital_inputs is False
    assert recovered.settings.local_output_dir == session.resolve()


def test_move_completed_session_resume_uses_custom_selected_folder(tmp_path: Path) -> None:
    import yaml

    local_root = tmp_path / "local"
    source = local_root / "temporary-output"
    destination = tmp_path / "storage" / "custom-output-folder"
    raw = tmp_path / "raw" / "scientific-session"
    external_workspace = tmp_path / "external-workspace"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    raw.mkdir(parents=True)
    external_workspace.mkdir()
    (source / "scientific-session.xml").write_text("<session />", encoding="utf-8")
    (source / "scientific-session.rhd").write_bytes(b"rhd")
    settings = PipelineGuiSettings(basepath=str(raw), local_root=str(local_root))
    analysis = AnalysisConfig.create(json.loads(settings.to_json()))
    execution = _resume_execution(external_workspace)
    (source / "preprocess_run.yaml").write_text(
        yaml.safe_dump(
            {
                "session_output_dir": str(source.resolve()),
                "analysis": analysis.to_dict(),
                "execution": execution.to_dict(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _move_local_output_to_storage(
        settings,
        destination_dir=destination,
        source_dir=source,
        source_basename="scientific-session",
        move_dat=False,
        overwrite=False,
        clean_after_move=True,
    )

    recovered = load_local_session_resume(destination)

    assert recovered.settings.basename == "scientific-session"
    assert recovered.settings.existing_session_dir == str(destination.resolve())
    assert recovered.settings.local_output_dir == destination.resolve()
    assert recovered.settings.execution.workspace == str(external_workspace.resolve())


def test_local_session_resume_fails_closed_on_invalid_active_marker(tmp_path: Path) -> None:
    import pytest
    import yaml

    raw = tmp_path / "raw" / "day1"
    raw.mkdir(parents=True)
    session = tmp_path / "sorting_temp" / raw.name
    session.mkdir(parents=True)
    settings = PipelineGuiSettings(basepath=str(raw), local_root=str(session.parent))
    analysis = AnalysisConfig.create(json.loads(settings.to_json()))
    execution = _resume_execution(session.parent)
    (session / "preprocess_run.yaml").write_text(
        yaml.safe_dump(
            {
                "session_output_dir": str(session.resolve()),
                "analysis": analysis.to_dict(),
                "execution": execution.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    (session / ".pipeline-active-run.json").write_text(
        json.dumps({"kind": "run", "run_dir": str(tmp_path / "missing-run")}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Cannot recover the persistent Run"):
        load_local_session_resume(session)


def test_gui_browse_local_session_resume_is_common_and_reconnects(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    unavailable = SlurmCapabilities(
        None, None, None, None, False, False, ("Slurm unavailable",)
    )
    monkeypatch.setattr(
        "src.preprocess.gui.app.detect_slurm_capabilities",
        lambda **_kwargs: unavailable,
    )
    from PySide6.QtWidgets import QApplication

    raw = tmp_path / "raw" / "day1"
    raw.mkdir(parents=True)
    session = tmp_path / "sorting_temp" / raw.name
    settings = PipelineGuiSettings(basepath=str(raw), local_root=str(session.parent))
    settings.xml_path = str(tmp_path / "unmounted-server" / "day1.xml")
    run_dir = _write_active_resume_session(session, settings)
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    set_active_calls: list[tuple[Path, Path | None]] = []
    reconcile_calls: list[bool] = []
    try:
        monkeypatch.setattr(window, "_select_directory", lambda *_args: str(session))
        monkeypatch.setattr(
            window,
            "_set_active_run",
            lambda path, *, session_dir=None: set_active_calls.append(
                (Path(path), session_dir)
            ),
        )
        monkeypatch.setattr(
            window, "_request_run_reconcile", lambda: reconcile_calls.append(True)
        )

        window._browse_local_session_to_resume()
        window.resize(1200, 800)
        window.show()
        application.processEvents()

        assert window.browse_local_session_resume.text() == "Browse local session to resume"
        assert window.basepath.text() == str(raw)
        assert window.local_root.text() == str(session.parent.resolve())
        assert window.xml_path.text() == settings.xml_path
        assert set_active_calls == [(run_dir.resolve(), session.resolve())]
        assert reconcile_calls == [True]
        assert window.browse_local_session_resume.width() >= (
            window.browse_local_session_resume.sizeHint().width()
        )
        assert "font-size: 11px" in window.styleSheet()
        assert window.main_splitter.sizes()[0] < 500
        assert window.execution_scroll_area.horizontalScrollBar().maximum() == 0
    finally:
        window.close()
        application.processEvents()


def test_gui_browse_invalid_local_session_does_not_mutate_settings(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    unavailable = SlurmCapabilities(
        None, None, None, None, False, False, ("Slurm unavailable",)
    )
    monkeypatch.setattr(
        "src.preprocess.gui.app.detect_slurm_capabilities",
        lambda **_kwargs: unavailable,
    )
    from PySide6.QtWidgets import QApplication, QMessageBox

    invalid = tmp_path / "not-a-session"
    invalid.mkdir()
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.basepath.setText("unchanged-basepath")
        window.multi_day_name.setText("unchanged-name")
        monkeypatch.setattr(window, "_select_directory", lambda *_args: str(invalid))
        errors: list[str] = []
        monkeypatch.setattr(
            QMessageBox,
            "critical",
            lambda _parent, _title, message: errors.append(str(message)),
        )

        window._browse_local_session_to_resume()

        assert errors and "neither .pipeline-active-run.json" in errors[0]
        assert window.basepath.text() == "unchanged-basepath"
        assert window.multi_day_name.text() == "unchanged-name"
    finally:
        window.close()
        application.processEvents()


def test_local_session_resume_rejects_output_mismatch(tmp_path: Path) -> None:
    import pytest

    raw = tmp_path / "raw" / "day1"
    raw.mkdir(parents=True)
    session = tmp_path / "sorting_temp" / raw.name
    settings = PipelineGuiSettings(basepath=str(raw), local_root=str(session.parent))
    run_dir = _write_active_resume_session(session, settings)
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["session_output_dir"] = str(tmp_path / "different-session")
    (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the selected local session"):
        load_local_session_resume(session)


def test_postprocess_config_infers_recording_metadata_from_basepath_xml(tmp_path: Path) -> None:
    basepath = tmp_path / "test_rec"
    local_root = tmp_path / "preprocess_tmp"
    local_session = local_root / basepath.name
    sorting_folder = local_session / "Kilosort_20260101"
    basepath.mkdir()
    sorting_folder.mkdir(parents=True)
    (local_session / f"{basepath.name}.dat").write_bytes(b"\x00\x00")
    _write_session_xml(basepath, sample_rate=30000.0, n_channels=64)

    settings = PipelineGuiSettings(basepath=str(basepath), local_root=str(local_root))
    settings.postprocess.sorting_phy_folder = str(sorting_folder)

    config = settings.to_postprocess_config()

    assert config.dat_path == local_session / f"{basepath.name}.dat"
    assert config.sampling_frequency == 30000.0
    assert config.num_channels == 64


def test_postprocess_config_falls_back_to_basepath_dat_and_sorting_folder(tmp_path: Path) -> None:
    basepath = tmp_path / "test_rec"
    local_root = tmp_path / "preprocess_tmp"
    sorting_folder = basepath / "Kilosort_20260101"
    basepath.mkdir()
    sorting_folder.mkdir()
    (basepath / f"{basepath.name}.dat").write_bytes(b"\x00\x00")
    _write_session_xml(basepath, sample_rate=20000.0, n_channels=32)

    settings = PipelineGuiSettings(basepath=str(basepath), local_root=str(local_root))

    config = settings.to_postprocess_config()

    assert config.sorting_phy_folder == sorting_folder.resolve()
    assert config.dat_path == (basepath / f"{basepath.name}.dat").resolve()
    assert config.sampling_frequency == 20000.0
    assert config.num_channels == 32


def test_preprocess_config_preserves_sorter_partition_mode(tmp_path: Path) -> None:
    basepath = tmp_path / "test_rec"
    basepath.mkdir()
    _write_session_xml(basepath)

    settings = PipelineGuiSettings(basepath=str(basepath), local_root=str(tmp_path / "local"))
    settings.preprocess.sorter_partition_mode = "shank"

    config = settings.to_preprocess_config()
    loaded = PipelineGuiSettings.from_json(settings.to_json())

    assert config.sorter_partition_mode == "shank"
    assert loaded.preprocess.sorter_partition_mode == "shank"


def test_execution_resources_round_trip_exact_slurm_cpu_count_without_local_cap(
    tmp_path: Path,
) -> None:
    settings = PipelineGuiSettings(
        basepath=str(tmp_path / "session"),
        local_root=str(tmp_path / "local"),
    )
    settings.execution.workspace = str(tmp_path / "shared")
    settings.execution.matlab_path = "/opt/matlab/bin/matlab"
    settings.execution.shared_workspace_acknowledged = True
    settings.execution.preprocess.cpus = 128
    settings.execution.sorting.cpus = 128
    settings.execution.postprocess.cpus = 128

    loaded = PipelineGuiSettings.from_json(settings.to_json())

    assert loaded.execution.preprocess.cpus == 128
    assert loaded.execution.sorting.cpus == 128
    assert loaded.execution.postprocess.cpus == 128
    assert loaded.execution.sorting.gpu_count == 1
    assert loaded.execution.preprocess.gpu_count == 0
    assert loaded.execution.postprocess.gpu_count == 0
    assert loaded.execution.matlab_path == "/opt/matlab/bin/matlab"
    assert loaded.execution.shared_workspace_acknowledged is True


def test_legacy_worker_counts_seed_execution_resources() -> None:
    loaded = PipelineGuiSettings.from_json(
        json.dumps(
            {
                "preprocess": {
                    "preprocess_worker_count": 91,
                    "sorter_worker_count": 73,
                },
                "postprocess": {"worker_count": 55},
            }
        )
    )

    assert loaded.execution.preprocess.cpus == 91
    assert loaded.execution.sorting.cpus == 73
    assert loaded.execution.postprocess.cpus == 55


def test_preprocess_config_uses_explicit_xml_path(tmp_path: Path) -> None:
    basepath = tmp_path / "test_rec"
    selected_xml = tmp_path / "selected.xml"
    basepath.mkdir()
    selected_xml.write_text(
        """
<session>
  <acquisitionSystem>
    <nChannels>16</nChannels>
    <sampleRate>20000</sampleRate>
  </acquisitionSystem>
</session>
""".strip(),
        encoding="utf-8",
    )

    settings = PipelineGuiSettings(
        basepath=str(basepath),
        local_root=str(tmp_path / "local"),
        xml_path=str(selected_xml),
    )

    config = settings.to_preprocess_config()
    loaded = PipelineGuiSettings.from_json(settings.to_json())

    assert config.xml_path == selected_xml
    assert loaded.xml_path == str(selected_xml)


def test_pipeline_settings_round_trips_multi_day_fields(tmp_path: Path) -> None:
    day1 = tmp_path / "day1"
    day2 = tmp_path / "day2"
    day1.mkdir()
    day2.mkdir()

    settings = PipelineGuiSettings(
        basepath=str(day1),
        local_root=str(tmp_path / "local"),
        multi_day_enabled=True,
        multi_day_session_paths=[str(day1), str(day2)],
        multi_day_name="animal_multiday",
    )

    loaded = PipelineGuiSettings.from_json(settings.to_json())

    assert loaded.multi_day_enabled is True
    assert loaded.multi_day_session_paths == [str(day1), str(day2)]
    assert loaded.multi_day_name == "animal_multiday"
    assert loaded.basename == "animal_multiday"


def test_pipeline_settings_round_trips_cell_explorer_folders(tmp_path: Path) -> None:
    folder1 = tmp_path / "Kilosort_probe1_spi"
    folder2 = tmp_path / "Kilosort_probe2_spi"
    folder1.mkdir()
    folder2.mkdir()

    settings = PipelineGuiSettings(basepath=str(tmp_path / "session"), local_root=str(tmp_path / "local"))
    settings.postprocess.cell_explorer_sorting_folders = [str(folder1), str(folder2)]

    loaded = PipelineGuiSettings.from_json(settings.to_json())

    assert loaded.postprocess.cell_explorer_sorting_folders == [str(folder1), str(folder2)]


def test_postprocess_config_uses_local_manifest_as_search_root(tmp_path: Path) -> None:
    basepath = tmp_path / "session"
    local_root = tmp_path / "local"
    local_session = local_root / "session"
    sorting_folder = local_session / "Kilosort_20260624_probe3"
    basepath.mkdir()
    sorting_folder.mkdir(parents=True)
    (local_session / "sorter_partition_manifest.json").write_text("{}", encoding="utf-8")
    (local_session / "session.dat").write_bytes(b"\x00\x00")
    _write_session_xml(basepath, sample_rate=20000.0, n_channels=4)

    settings = PipelineGuiSettings(basepath=str(basepath), local_root=str(local_root))

    config = settings.to_postprocess_config()

    assert config.sorting_search_root == local_session
    assert config.sorting_phy_folder == sorting_folder.resolve()


def test_gui_run_pipeline_result_json_safe_converts_paths(tmp_path: Path) -> None:
    payload = {
        "path": tmp_path / "Kilosort_probe1",
        "nested": {"paths": [tmp_path / "Kilosort_probe2", {tmp_path / "Kilosort_probe3"}]},
    }

    encoded = json.dumps(_json_safe(payload), sort_keys=True)

    assert str(tmp_path / "Kilosort_probe1") in encoded
    assert str(tmp_path / "Kilosort_probe2") in encoded
    assert str(tmp_path / "Kilosort_probe3") in encoded


def _write_phy_params(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    params_path = folder / "params.py"
    params_path.write_text("# test params\n", encoding="utf-8")
    return params_path


def test_launch_phy_respects_selected_non_spi_folder_when_params_exists(tmp_path: Path) -> None:
    sorting_folder = tmp_path / "Kilosort_2026-07-05_120000"
    spi_folder = tmp_path / "Kilosort_2026-07-05_120000_spi"
    selected_params = _write_phy_params(sorting_folder)
    _write_phy_params(spi_folder)

    resolved = MainWindow._resolve_phy_params_path(object(), sorting_folder)

    assert resolved == selected_params.resolve()


def test_launch_phy_respects_selected_spi_folder(tmp_path: Path) -> None:
    sorting_folder = tmp_path / "Kilosort_2026-07-05_120000"
    spi_folder = tmp_path / "Kilosort_2026-07-05_120000_spi"
    _write_phy_params(sorting_folder)
    selected_params = _write_phy_params(spi_folder)

    resolved = MainWindow._resolve_phy_params_path(object(), spi_folder)

    assert resolved == selected_params.resolve()


def test_launch_phy_falls_back_to_spi_when_selected_folder_has_no_params(tmp_path: Path) -> None:
    sorting_folder = tmp_path / "Kilosort_2026-07-05_120000"
    spi_folder = tmp_path / "Kilosort_2026-07-05_120000_spi"
    sorting_folder.mkdir()
    spi_params = _write_phy_params(spi_folder)

    resolved = MainWindow._resolve_phy_params_path(object(), sorting_folder)

    assert resolved == spi_params.resolve()


def test_postprocessed_preference_still_resolves_spi_first(tmp_path: Path) -> None:
    sorting_folder = tmp_path / "Kilosort_2026-07-05_120000"
    spi_folder = tmp_path / "Kilosort_2026-07-05_120000_spi"
    _write_phy_params(sorting_folder)
    spi_params = _write_phy_params(spi_folder)

    resolved = MainWindow._resolve_phy_params_path(
        object(),
        sorting_folder,
        prefer_postprocessed=True,
    )

    assert resolved == spi_params.resolve()


def test_legacy_noise_label_rejects_an_active_session_claim(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from src.execution.session import acquire_manual_session_claim, release_manual_session_claim

    application = QApplication.instance() or QApplication([])
    session_dir = tmp_path / "local" / "session"
    session_dir.mkdir(parents=True)
    settings = PipelineGuiSettings(
        basepath=str(tmp_path / "raw" / "session"), local_root=str(tmp_path / "local")
    )
    token = acquire_manual_session_claim(session_dir=session_dir, owner="other application")
    window = MainWindow()
    messages: list[str] = []
    try:
        monkeypatch.setattr(window, "_collect_settings", lambda: settings)
        monkeypatch.setattr("src.preprocess.gui.app.run_preflight", lambda *_args: [])
        monkeypatch.setattr(
            QMessageBox, "critical", lambda _parent, _title, message: messages.append(message)
        )

        window._start_run("noise_label")

        assert window._process is None
        assert messages and "session is active" in messages[0].lower()
    finally:
        release_manual_session_claim(session_dir=session_dir, token=token)
        window.close()
        application.processEvents()


def test_legacy_noise_label_claim_releases_on_process_completion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from src.execution.session import active_session_claim, acquire_manual_session_claim

    application = QApplication.instance() or QApplication([])
    session_dir = tmp_path / "local" / "session"
    session_dir.mkdir(parents=True)
    token = acquire_manual_session_claim(session_dir=session_dir, owner="Legacy noise labeling")
    window = MainWindow()
    try:
        window._legacy_process_session_claim = (session_dir, token)
        window._process_finished(0, None)

        assert active_session_claim(pipeline_root=Path(), session_dir=session_dir) is None
    finally:
        window.close()
        application.processEvents()


def test_legacy_noise_label_claim_waits_for_process_group_exit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from src.execution.session import active_session_claim, acquire_manual_session_claim

    application = QApplication.instance() or QApplication([])
    session_dir = tmp_path / "local" / "session"
    session_dir.mkdir(parents=True)
    token = acquire_manual_session_claim(session_dir=session_dir, owner="Legacy noise labeling")
    window = MainWindow()
    try:
        window._legacy_process_session_claim = (session_dir, token)
        window._legacy_process_group_pid = 12345
        monkeypatch.setattr("src.preprocess.gui.app.os.killpg", lambda _pid, _sig: None)

        assert window._release_legacy_process_session_claim() is False
        assert active_session_claim(pipeline_root=Path(), session_dir=session_dir) is not None

        def missing_group(_pid, _sig):
            raise ProcessLookupError

        monkeypatch.setattr("src.preprocess.gui.app.os.killpg", missing_group)
        assert window._release_legacy_process_session_claim() is True
        assert active_session_claim(pipeline_root=Path(), session_dir=session_dir) is None
    finally:
        window.close()
        application.processEvents()


def test_legacy_noise_label_pid_claim_failure_cleans_up_startup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from src.execution.session import active_session_claim
    import src.preprocess.gui.app as app

    class _Signal:
        def connect(self, _callback) -> None:
            pass

    class _FakeProcess:
        class ProcessChannelMode:
            SeparateChannels = object()

        def __init__(self, _parent) -> None:
            self.readyReadStandardOutput = _Signal()
            self.readyReadStandardError = _Signal()
            self.finished = _Signal()
            self.errorOccurred = _Signal()
            self.killed = False

        def setProgram(self, _program) -> None:
            pass

        def setArguments(self, _arguments) -> None:
            pass

        def setWorkingDirectory(self, _path) -> None:
            pass

        def setProcessChannelMode(self, _mode) -> None:
            pass

        def setChildProcessModifier(self, _modifier) -> None:
            pass

        def start(self) -> None:
            pass

        def waitForStarted(self, _timeout) -> bool:
            return True

        def processId(self) -> int:
            return 123

        def kill(self) -> None:
            self.killed = True

        def waitForFinished(self, _timeout) -> bool:
            return True

    application = QApplication.instance() or QApplication([])
    session_dir = tmp_path / "local" / "session"
    session_dir.mkdir(parents=True)
    settings = PipelineGuiSettings(
        basepath=str(tmp_path / "raw" / "session"), local_root=str(tmp_path / "local")
    )
    window = MainWindow()
    messages: list[str] = []
    try:
        monkeypatch.setattr(window, "_collect_settings", lambda: settings)
        monkeypatch.setattr("src.preprocess.gui.app.run_preflight", lambda *_args: [])
        monkeypatch.setattr(app, "QProcess", _FakeProcess)
        monkeypatch.setattr(
            "src.execution.session.update_manual_session_claim_pid",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected PID update failure")),
        )
        monkeypatch.setattr(
            QMessageBox, "critical", lambda _parent, _title, message: messages.append(message)
        )

        window._start_run("noise_label")

        assert window._process is None
        assert window._process_config_path is None
        assert active_session_claim(pipeline_root=Path(), session_dir=session_dir) is None
        assert messages and "claim could not be updated" in messages[-1]
    finally:
        window.close()
        application.processEvents()
