from __future__ import annotations

from pathlib import Path
import json

from src.preprocess.gui.config_model import PipelineGuiSettings
from src.preprocess.gui.run_pipeline import _json_safe


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
