from __future__ import annotations

from pathlib import Path

from src.preprocess.gui.config_model import PipelineGuiSettings


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
