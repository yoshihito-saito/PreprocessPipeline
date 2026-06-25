from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.io import discover_subsessions
from src.preprocess.gui.config_model import PipelineGuiSettings
from src.preprocess.gui.run_pipeline import run_pipeline
from src.preprocess.multiday import prepare_multi_day_basepath


def _write_xml(basepath: Path, *, n_channels: int = 4, sample_rate: float = 20000.0) -> None:
    (basepath / f"{basepath.name}.xml").write_text(
        f"""
<session>
  <acquisitionSystem>
    <nChannels>{n_channels}</nChannels>
    <sampleRate>{sample_rate}</sampleRate>
  </acquisitionSystem>
  <anatomicalDescription>
    <channelGroups>
      <group><channels><channel>0</channel><channel>1</channel><channel>2</channel><channel>3</channel></channels></group>
    </channelGroups>
  </anatomicalDescription>
</session>
""".strip(),
        encoding="utf-8",
    )


def _write_epoch(session: Path, name: str, *, n_channels: int = 4, n_samples: int = 5) -> Path:
    epoch = session / name
    epoch.mkdir(parents=True)
    data = np.arange(n_samples * n_channels, dtype=np.int16)
    (epoch / "amplifier.dat").write_bytes(data.tobytes())
    return epoch


def _write_openephys_epoch(
    session: Path,
    name: str,
    *,
    n_channels: int,
    n_samples: int,
    sample_rate: float = 20000.0,
) -> Path:
    recording_root = session / name / "Record Node 101" / "experiment1" / "recording1"
    continuous_dir = recording_root / "continuous" / "Acquisition_Board-100.acquisition_board"
    continuous_dir.mkdir(parents=True)
    structure = {
        "continuous": [
            {
                "folder_name": "Acquisition_Board-100.acquisition_board/",
                "sample_rate": sample_rate,
                "recorded_processor": "Record Node",
                "recorded_processor_id": 101,
                "num_channels": n_channels,
            }
        ]
    }
    (recording_root / "structure.oebin").write_text(json.dumps(structure), encoding="utf-8")
    np.zeros((n_samples, n_channels), dtype=np.int16).tofile(continuous_dir / "continuous.dat")
    return session / name


def test_prepare_multi_day_basepath_stages_sessions_and_manifest(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    ep1 = _write_epoch(day1, "ep_240101_120000", n_samples=3)
    ep2 = _write_epoch(day1, "ep_240101_130000", n_samples=4)
    ep3 = _write_epoch(day2, "ep_240102_120000", n_samples=5)

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        overwrite=True,
    )

    assert staged.server_basepath == tmp_path / "animal_multiday"
    assert staged.local_basepath == tmp_path / "local" / "animal_multiday"
    assert (staged.server_basepath / "animal_multiday.xml").exists()
    assert len(staged.subepochs) == 3
    assert [item.sample_count for item in staged.subepochs] == [3, 4, 5]

    discovered = discover_subsessions(
        basepath=staged.server_basepath,
        sort_files=True,
        alt_sort=None,
        ignore_folders=[],
    )
    assert [path.resolve() for path in discovered] == [
        (ep1 / "amplifier.dat").resolve(),
        (ep2 / "amplifier.dat").resolve(),
        (ep3 / "amplifier.dat").resolve(),
    ]

    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "animal_multiday"
    assert manifest["source_sessions"] == [str(day1.resolve()), str(day2.resolve())]
    assert [entry["session_name"] for entry in manifest["subepochs"]] == [
        "animal_day1",
        "animal_day1",
        "animal_day2",
    ]


def test_prepare_multi_day_basepath_uses_openephys_stream_channels_for_sample_counts(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    selected_xml_dir = tmp_path / "metadata"
    day1.mkdir()
    day2.mkdir()
    selected_xml_dir.mkdir()
    selected_xml = selected_xml_dir / "selected.xml"
    selected_xml.write_text(
        ("""
<session>
  <acquisitionSystem>
    <nChannels>192</nChannels>
    <sampleRate>20000</sampleRate>
  </acquisitionSystem>
</session>
""").strip(),
        encoding="utf-8",
    )
    _write_openephys_epoch(day1, "animal_2026-06-24_16-38-56", n_channels=200, n_samples=11)
    _write_openephys_epoch(day2, "animal_2026-06-25_16-38-56", n_channels=200, n_samples=13)

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        xml_path=selected_xml,
        overwrite=True,
    )

    assert [item.sample_count for item in staged.subepochs] == [11, 13]
    assert [item.binary_n_channels for item in staged.subepochs] == [200, 200]
    assert [item.source_type for item in staged.subepochs] == ["openephys", "openephys"]
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_channels"] == 192
    assert [entry["binary_n_channels"] for entry in manifest["subepochs"]] == [200, 200]


def test_prepare_multi_day_basepath_rejects_mismatched_openephys_stream_channels(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    selected_xml_dir = tmp_path / "metadata"
    day1.mkdir()
    day2.mkdir()
    selected_xml_dir.mkdir()
    selected_xml = selected_xml_dir / "selected.xml"
    selected_xml.write_text(
        ("""
<session>
  <acquisitionSystem>
    <nChannels>192</nChannels>
    <sampleRate>20000</sampleRate>
  </acquisitionSystem>
</session>
""").strip(),
        encoding="utf-8",
    )
    _write_openephys_epoch(day1, "animal_2026-06-24_16-38-56", n_channels=200, n_samples=11)
    _write_openephys_epoch(day2, "animal_2026-06-25_16-38-56", n_channels=208, n_samples=13)

    with pytest.raises(ValueError, match="mismatched channel counts"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            local_root=tmp_path / "local",
            name="animal_multiday",
            xml_path=selected_xml,
            overwrite=True,
        )


def test_prepare_multi_day_basepath_rejects_mismatched_xml(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1, n_channels=4)
    _write_xml(day2, n_channels=8)
    _write_epoch(day1, "ep1")
    _write_epoch(day2, "ep1", n_channels=8)

    with pytest.raises(ValueError, match="share channel count and sampling rate"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            local_root=tmp_path / "local",
            name="bad_multiday",
        )


def test_prepare_multi_day_basepath_ignores_parent_xml_without_loaded_xml(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    day1 = parent / "animal_day1"
    day2 = parent / "animal_day2"
    parent.mkdir()
    day1.mkdir()
    day2.mkdir()
    (parent / "parent.xml").write_text(
        ("""
<session>
  <acquisitionSystem>
    <nChannels>4</nChannels>
    <sampleRate>20000</sampleRate>
  </acquisitionSystem>
</session>
""").strip(),
        encoding="utf-8",
    )
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")

    with pytest.raises(FileNotFoundError, match="No XML file selected"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            local_root=tmp_path / "local",
            name="animal_multiday",
            overwrite=True,
        )


def test_prepare_multi_day_basepath_uses_existing_multiday_xml(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    multiday = tmp_path / "animal_multiday"
    day1.mkdir()
    day2.mkdir()
    multiday.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    _write_xml(multiday)
    existing_xml = multiday / "animal_multiday.xml"
    original_xml_text = existing_xml.read_text(encoding="utf-8")
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        overwrite=True,
    )

    assert (staged.server_basepath / "animal_multiday.xml").read_text(encoding="utf-8") == original_xml_text
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["xml_path"] == str(existing_xml.resolve())


def test_prepare_multi_day_basepath_explicit_xml_allows_missing_session_xml(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    selected_xml_dir = tmp_path / "metadata"
    day1.mkdir()
    day2.mkdir()
    selected_xml_dir.mkdir()
    selected_xml = selected_xml_dir / "selected.xml"
    selected_xml.write_text(
        ("""
<session>
  <acquisitionSystem>
    <nChannels>4</nChannels>
    <sampleRate>20000</sampleRate>
  </acquisitionSystem>
</session>
""").strip(),
        encoding="utf-8",
    )
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        xml_path=selected_xml,
        overwrite=True,
    )

    assert (staged.server_basepath / "animal_multiday.xml").read_text(encoding="utf-8") == selected_xml.read_text(
        encoding="utf-8"
    )


def test_prepare_multi_day_basepath_explicit_xml_updates_staged_xml(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    multiday = tmp_path / "animal_multiday"
    selected_xml_dir = tmp_path / "metadata"
    day1.mkdir()
    day2.mkdir()
    multiday.mkdir()
    selected_xml_dir.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    _write_xml(multiday)
    selected_xml = selected_xml_dir / "selected.xml"
    selected_xml.write_text((day1 / "animal_day1.xml").read_text(encoding="utf-8"), encoding="utf-8")
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        xml_path=selected_xml,
        overwrite=True,
    )

    assert (staged.server_basepath / "animal_multiday.xml").read_text(encoding="utf-8") == selected_xml.read_text(
        encoding="utf-8"
    )
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["xml_path"] == str((staged.server_basepath / "animal_multiday.xml").resolve())


def test_prepare_multi_day_basepath_accepts_staged_xml_as_explicit_xml(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    multiday = tmp_path / "animal_multiday"
    day1.mkdir()
    day2.mkdir()
    multiday.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    _write_xml(multiday)
    staged_xml = multiday / "animal_multiday.xml"
    original_xml_text = staged_xml.read_text(encoding="utf-8")
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        xml_path=staged_xml,
        overwrite=True,
    )

    assert (staged.server_basepath / "animal_multiday.xml").read_text(encoding="utf-8") == original_xml_text
    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert manifest["xml_path"] == str(staged_xml.resolve())


def test_run_pipeline_multi_day_requires_selected_sessions(tmp_path: Path) -> None:
    settings = PipelineGuiSettings(
        local_root=str(tmp_path / "local"),
        multi_day_enabled=True,
        multi_day_session_paths=[],
        multi_day_name="animal_multiday",
    )

    with pytest.raises(ValueError, match="requires at least two session folders"):
        run_pipeline(settings, "preprocess")


def test_run_pipeline_multi_day_requires_basepath_name(tmp_path: Path) -> None:
    day1 = tmp_path / "day1"
    day2 = tmp_path / "day2"
    day1.mkdir()
    day2.mkdir()
    settings = PipelineGuiSettings(
        local_root=str(tmp_path / "local"),
        multi_day_enabled=True,
        multi_day_session_paths=[str(day1), str(day2)],
        multi_day_name="",
    )

    with pytest.raises(ValueError, match="Multi-day basepath name is required"):
        run_pipeline(settings, "preprocess")
