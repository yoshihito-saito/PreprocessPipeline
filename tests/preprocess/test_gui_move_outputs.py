from __future__ import annotations

import errno
from pathlib import Path
import json
import os

import pytest
import yaml

from src.execution.models import (
    AnalysisConfig,
    BackendName,
    ExecutionConfig,
    RequestedBackend,
    ResourceSpec,
    StageName,
)

from src.preprocess.gui.app import (
    MainWindow,
    _default_output_storage_dir,
    _move_local_output_to_basepath,
    _move_local_output_to_storage,
)
from src.preprocess.gui.config_model import PipelineGuiSettings, load_local_session_resume


def test_move_outputs_can_clean_local_session_folder(tmp_path: Path) -> None:
    basepath = tmp_path / "sessionA"
    local_root = tmp_path / "preprocess_tmp"
    local_session = local_root / "sessionA"
    basepath.mkdir()
    local_session.mkdir(parents=True)
    (local_session / "sessionA.dat").write_bytes(b"dat")
    (local_session / "sessionA.rhd").write_bytes(b"rhd")
    (local_session / "sessionA.xml").write_text("<session />", encoding="utf-8")
    (local_session / "quality_metrics.csv").write_text("cluster_id\n1\n", encoding="utf-8")

    result = _move_local_output_to_basepath(
        PipelineGuiSettings(basepath=str(basepath), local_root=str(local_root)),
        move_dat=False,
        overwrite=False,
        clean_after_move=True,
    )

    assert (basepath / "quality_metrics.csv").exists()
    assert (basepath / "sessionA.xml").read_text(encoding="utf-8") == "<session />"
    assert not local_session.exists()
    assert result["cleaned"] is True
    assert result["clean_after_move"] is True
    assert {item["name"] for item in result["skipped"]} == {
        "sessionA.dat",
        "sessionA.rhd",
    }


def test_multiday_default_storage_creates_named_sibling_of_basepath(
    tmp_path: Path,
) -> None:
    day1 = tmp_path / "storage" / "animal_day1"
    day2 = tmp_path / "storage" / "animal_day2"
    local_root = tmp_path / "preprocess_tmp"
    local_session = local_root / "animal_multiday"
    day1.mkdir(parents=True)
    day2.mkdir()
    local_session.mkdir(parents=True)
    (local_session / "quality_metrics.csv").write_text("cluster_id\n1\n", encoding="utf-8")
    settings = PipelineGuiSettings(
        basepath=str(day1),
        local_root=str(local_root),
        multi_day_enabled=True,
        multi_day_session_paths=[str(day1), str(day2)],
        multi_day_name="animal_multiday",
    )

    expected = tmp_path / "storage" / "animal_multiday"
    assert _default_output_storage_dir(settings) == expected.resolve()

    result = _move_local_output_to_storage(
        settings,
        move_dat=False,
        overwrite=False,
        clean_after_move=False,
    )

    assert (expected / "quality_metrics.csv").exists()
    assert (local_session / "quality_metrics.csv").exists()
    assert result["storage_dir"] == str(expected.resolve())
    assert result["cleaned"] is False


def test_copy_without_local_cleanup_preserves_all_source_items(tmp_path: Path) -> None:
    basepath = tmp_path / "storage" / "session"
    local_root = tmp_path / "local"
    source = local_root / "session"
    basepath.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "quality_metrics.csv").write_text("cluster_id\n1\n", encoding="utf-8")
    (source / "session.dat").write_bytes(b"dat")
    (source / "session.xml").write_text("<session />", encoding="utf-8")
    (source / "session.rhd").write_bytes(b"rhd")

    _move_local_output_to_storage(
        PipelineGuiSettings(basepath=str(basepath), local_root=str(local_root)),
        move_dat=False,
        overwrite=False,
        clean_after_move=False,
    )

    assert (basepath / "quality_metrics.csv").exists()
    assert (basepath / "session.xml").read_text(encoding="utf-8") == "<session />"
    assert (source / "quality_metrics.csv").exists()
    assert (source / "session.dat").exists()
    assert (source / "session.xml").read_text(encoding="utf-8") == "<session />"
    assert (source / "session.rhd").exists()


@pytest.mark.parametrize("clean_after_move", [False, True])
def test_move_preserves_existing_destination_session_xml(
    tmp_path: Path, clean_after_move: bool
) -> None:
    destination = tmp_path / "storage" / "session"
    source = tmp_path / "local" / "session"
    destination.mkdir(parents=True)
    source.mkdir(parents=True)
    (destination / "session.xml").write_text("<destination />", encoding="utf-8")
    (source / "session.xml").write_text("<source />", encoding="utf-8")
    (source / "result.mat").write_bytes(b"result")

    result = _move_local_output_to_storage(
        PipelineGuiSettings(basepath=str(destination), local_root=str(source.parent)),
        move_dat=False,
        overwrite=False,
        clean_after_move=clean_after_move,
    )

    assert (destination / "session.xml").read_text(encoding="utf-8") == "<destination />"
    if clean_after_move:
        assert not source.exists()
    else:
        assert (source / "session.xml").read_text(encoding="utf-8") == "<source />"
        assert (source / "result.mat").read_bytes() == b"result"
    assert {item["name"]: item["reason"] for item in result["skipped"]}[
        "session.xml"
    ] == "destination session XML already exists"


def test_move_atomically_preserves_destination_xml_created_during_publication(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "storage" / "session"
    source = tmp_path / "local" / "session"
    destination.mkdir(parents=True)
    source.mkdir(parents=True)
    source_xml = source / "session.xml"
    destination_xml = destination / "session.xml"
    source_xml.write_text("<source />", encoding="utf-8")
    (source / "result.mat").write_bytes(b"result")
    original_link = os.link

    def link_with_late_destination(src, dst, *args, **kwargs):
        if Path(dst) == destination_xml:
            destination_xml.write_text("<late-destination />", encoding="utf-8")
        return original_link(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.preprocess.gui.app.os.link", link_with_late_destination)

    result = _move_local_output_to_storage(
        PipelineGuiSettings(basepath=str(destination), local_root=str(source.parent)),
        move_dat=False,
        overwrite=True,
        clean_after_move=False,
    )

    assert destination_xml.read_text(encoding="utf-8") == "<late-destination />"
    assert source_xml.read_text(encoding="utf-8") == "<source />"
    assert (destination / "result.mat").read_bytes() == b"result"
    assert (source / "result.mat").read_bytes() == b"result"
    assert {item["name"]: item["reason"] for item in result["skipped"]}[
        "session.xml"
    ] == "destination session XML appeared during transfer"
    assert "session.xml" not in {item["name"] for item in result["inventory"]["move"]}


def test_move_rolls_back_xml_when_staged_link_cleanup_fails(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "storage" / "session"
    source = tmp_path / "local" / "session"
    destination.mkdir(parents=True)
    source.mkdir(parents=True)
    source_xml = source / "session.xml"
    source_xml.write_text("<source />", encoding="utf-8")
    original_unlink = Path.unlink
    failed = False

    def unlink_with_staging_failure(path, *args, **kwargs):
        nonlocal failed
        if (
            not failed
            and path.name == "session.xml"
            and ".move-staging-" in path.parent.name
        ):
            failed = True
            raise OSError("injected staged-link cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_with_staging_failure)

    with pytest.raises(OSError, match="injected staged-link cleanup failure"):
        _move_local_output_to_storage(
            PipelineGuiSettings(basepath=str(destination), local_root=str(source.parent)),
            move_dat=False,
            overwrite=False,
            clean_after_move=False,
        )

    assert source_xml.read_text(encoding="utf-8") == "<source />"
    assert not (destination / "session.xml").exists()
    assert not list(destination.glob(".session.move-staging-*"))


def test_move_xml_uses_no_clobber_copy_when_hard_links_are_unsupported(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "storage" / "session"
    source = tmp_path / "local" / "session"
    destination.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "session.xml").write_text("<source />", encoding="utf-8")
    monkeypatch.setattr(
        "src.preprocess.gui.app.os.link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(errno.EOPNOTSUPP, "hard links unsupported")
        ),
    )

    _move_local_output_to_storage(
        PipelineGuiSettings(basepath=str(destination), local_root=str(source.parent)),
        move_dat=False,
        overwrite=False,
        clean_after_move=False,
    )

    assert (destination / "session.xml").read_text(encoding="utf-8") == "<source />"
    assert (source / "session.xml").read_text(encoding="utf-8") == "<source />"


def test_explicit_storage_destination_overrides_multiday_default(
    tmp_path: Path,
) -> None:
    day1 = tmp_path / "raw" / "day1"
    day2 = tmp_path / "raw" / "day2"
    local_root = tmp_path / "preprocess_tmp"
    local_session = local_root / "combined"
    custom = tmp_path / "selected-storage"
    day1.mkdir(parents=True)
    day2.mkdir()
    local_session.mkdir(parents=True)
    custom.mkdir()
    (local_session / "result.mat").write_bytes(b"result")
    settings = PipelineGuiSettings(
        basepath=str(day1),
        local_root=str(local_root),
        multi_day_enabled=True,
        multi_day_session_paths=[str(day1), str(day2)],
        multi_day_name="combined",
    )

    result = _move_local_output_to_storage(
        settings,
        destination_dir=custom,
        move_dat=False,
        overwrite=False,
        clean_after_move=False,
    )

    assert (custom / "result.mat").exists()
    assert not (day1.parent / "combined").exists()
    assert result["storage_dir"] == str(custom.resolve())


def test_browse_save_dir_updates_basepath_but_retains_local_move_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    basepath = tmp_path / "raw" / "session-a"
    local_root = tmp_path / "preprocess_tmp"
    custom = tmp_path / "selected-storage"
    basepath.mkdir(parents=True)
    custom.mkdir()
    window = MainWindow()
    try:
        window.basepath.setText(str(basepath))
        window.local_root.setText(str(local_root))
        assert window.move_outputs.text() == "Copy outputs to storage"
        assert window.move_clean_local.text() == "Delete local after verified copy"
        assert window.move_clean_local.isChecked() is False
        assert window.move_storage_dir.text() == str(basepath.resolve())
        monkeypatch.setattr(window, "_select_directory", lambda *_args: str(custom))

        window._browse_move_storage_dir()

        assert window.basepath.text() == str(custom.resolve())
        assert window.move_storage_dir.text() == str(custom.resolve())
        assert window._move_source_snapshot == (
            (local_root / "session-a").resolve(),
            "session-a",
        )
    finally:
        window.close()
        application.processEvents()


def test_move_rewrites_recovery_paths_and_copies_custom_destination_metadata(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    source = local_root / "session"
    destination = tmp_path / "storage"
    source.mkdir(parents=True)
    destination.mkdir()
    (source / "session.dat").write_bytes(b"dat")
    (source / "session.xml").write_text("<session />", encoding="utf-8")
    (source / "session.rhd").write_bytes(b"rhd")
    record = source / "preprocess_run.yaml"
    record.write_text(f"session_output_dir: {source}\nsource_basepath: {tmp_path / 'raw'}\n", encoding="utf-8")
    (source / "sorter_partition_manifest.json").write_text(
        '{"partitions": [{"output_folder": "' + str(source / "Kilosort4_x") + '"}]}',
        encoding="utf-8",
    )
    settings = PipelineGuiSettings(basepath=str(tmp_path / "raw"), local_root=str(local_root))

    result = _move_local_output_to_storage(
        settings,
        destination_dir=destination,
        source_dir=source,
        source_basename="session",
        move_dat=False,
        overwrite=False,
        clean_after_move=True,
    )

    assert result["cleaned"] is True
    assert not source.exists()
    assert (destination / "session.xml").exists()
    assert (destination / "session.rhd").exists()
    assert str(destination) in (destination / "preprocess_run.yaml").read_text(encoding="utf-8")
    assert str(destination) in (destination / "sorter_partition_manifest.json").read_text(encoding="utf-8")


def test_move_completed_session_resumes_from_custom_named_destination(tmp_path: Path) -> None:
    local_root = tmp_path / "local"
    source = local_root / "temporary-output"
    destination = tmp_path / "storage" / "chosen-folder-name"
    raw = tmp_path / "raw" / "scientific-session"
    external_workspace = tmp_path / "external-run-workspace"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    raw.mkdir(parents=True)
    external_workspace.mkdir()
    (source / "scientific-session.dat").write_bytes(b"dat")
    (source / "scientific-session.xml").write_text("<session />", encoding="utf-8")
    (source / "scientific-session.rhd").write_bytes(b"rhd")
    settings = PipelineGuiSettings(basepath=str(raw), local_root=str(local_root))
    analysis = AnalysisConfig.create(json.loads(settings.to_json()))
    execution = ExecutionConfig(
        requested_backend=RequestedBackend.LOCAL,
        resolved_backend=BackendName.LOCAL,
        workspace=str(external_workspace),
        resources={
            StageName.PREPROCESS.value: ResourceSpec(1, 1024, None),
            StageName.SORTING.value: ResourceSpec(1, 1024, None, gpu_count=1),
            StageName.POSTPROCESS.value: ResourceSpec(1, 1024, None),
        },
    )
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


def test_move_rejects_claim_backed_session_without_mutating_either_tree(tmp_path: Path) -> None:
    source = tmp_path / "local" / "session"
    destination = tmp_path / "storage"
    source.mkdir(parents=True)
    destination.mkdir()
    result = source / "result.mat"
    result.write_bytes(b"result")
    claim = source / ".pipeline-active-run.json"
    claim.write_text('{"kind": "run"}', encoding="utf-8")
    settings = PipelineGuiSettings(basepath=str(tmp_path / "raw"), local_root=str(source.parent))

    with pytest.raises(ValueError, match="Cannot move an output folder with .pipeline-active-run.json"):
        _move_local_output_to_storage(
            settings,
            destination_dir=destination,
            source_dir=source,
            source_basename="session",
            move_dat=False,
            overwrite=False,
            clean_after_move=True,
        )

    assert result.read_bytes() == b"result"
    assert claim.exists()
    assert not (destination / "result.mat").exists()


def test_move_failure_keeps_source_and_destination_unchanged(tmp_path: Path, monkeypatch) -> None:
    local_root = tmp_path / "local"
    source = local_root / "session"
    destination = tmp_path / "storage"
    source.mkdir(parents=True)
    destination.mkdir()
    source_file = source / "result.mat"
    source_file.write_bytes(b"result")
    settings = PipelineGuiSettings(basepath=str(destination), local_root=str(local_root))

    import src.preprocess.gui.app as app

    real_copy2 = app.shutil.copy2

    def fail_copy(source_path, destination_path, *args, **kwargs):
        if Path(source_path).name == "result.mat":
            raise OSError("injected copy failure")
        return real_copy2(source_path, destination_path, *args, **kwargs)

    monkeypatch.setattr(app.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        _move_local_output_to_storage(
            settings,
            source_dir=source,
            source_basename="session",
            move_dat=False,
            overwrite=False,
            clean_after_move=True,
        )

    assert source_file.read_bytes() == b"result"
    assert not (destination / "result.mat").exists()


def test_move_rejects_equal_size_staging_corruption_before_source_deletion(
    tmp_path: Path, monkeypatch
) -> None:
    local_root = tmp_path / "local"
    source = local_root / "session"
    destination = tmp_path / "storage"
    source.mkdir(parents=True)
    destination.mkdir()
    source_file = source / "result.mat"
    source_file.write_bytes(b"original")
    settings = PipelineGuiSettings(basepath=str(destination), local_root=str(local_root))

    import src.preprocess.gui.app as app

    real_copy2 = app.shutil.copy2

    def corrupt_copy(source_path, destination_path, *args, **kwargs):
        result = real_copy2(source_path, destination_path, *args, **kwargs)
        if Path(source_path).name == "result.mat":
            Path(destination_path).write_bytes(b"corrupt!")
        return result

    monkeypatch.setattr(app.shutil, "copy2", corrupt_copy)
    with pytest.raises(IOError, match="Staged transfer validation failed"):
        _move_local_output_to_storage(
            settings,
            source_dir=source,
            source_basename="session",
            move_dat=False,
            overwrite=False,
            clean_after_move=True,
        )

    assert source_file.read_bytes() == b"original"
    assert not (destination / "result.mat").exists()
