from __future__ import annotations

from pathlib import Path
import json

from src.execution.backends import SlurmCapabilities
from src.execution.models import RequestedBackend
from src.preprocess.gui.app import (
    MainWindow,
    _default_config_has_backend_choice,
    _has_slurm_server_commands,
)
from src.preprocess.gui.config_model import (
    PipelineGuiSettings,
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
