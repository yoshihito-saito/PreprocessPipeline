from __future__ import annotations

import csv
import errno
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.io import build_acquisition_catalog, discover_subsessions
from src.preprocess.gui.config_model import PipelineGuiSettings
from src.preprocess.gui.run_pipeline import run_pipeline
from src.preprocess.multiday import prepare_multi_day_basepath
from src.preprocess.recording import (
    write_concatenated_dat,
    write_concatenated_dat_analogin,
)
import src.preprocess.multiday as multiday


def test_cleanup_staged_subepochs_rejects_manifest_path_escape(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    manifest_path = staging_root / "multi_day_manifest.json"
    manifest_path.write_text(
        json.dumps({"subepochs": [{"staged_subepoch_path": str(external)}]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="outside multi-day staging root"):
        multiday._cleanup_stale_staged_subepochs(
            manifest_path=manifest_path,
            active_staged_folders=set(),
            staging_root=staging_root,
            overwrite=True,
        )
    assert external.exists()


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


def test_explicit_subsession_order_uses_portable_relative_paths(tmp_path: Path) -> None:
    first = _write_epoch(tmp_path, "session_1") / "amplifier.dat"
    second = _write_epoch(tmp_path, "session_2") / "amplifier.dat"
    third = _write_epoch(tmp_path, "session_3") / "amplifier.dat"

    discovered = discover_subsessions(
        basepath=tmp_path,
        sort_files=True,
        alt_sort=None,
        ignore_folders=[],
        subsession_order=[
            "session_3/amplifier.dat",
            "session_1/amplifier.dat",
            "session_2/amplifier.dat",
        ],
    )

    assert discovered == [third, first, second]


def test_explicit_subsession_order_rejects_stale_or_incomplete_selection(
    tmp_path: Path,
) -> None:
    _write_epoch(tmp_path, "session_1")
    _write_epoch(tmp_path, "session_2")

    with pytest.raises(ValueError, match="every discovered recording exactly once"):
        discover_subsessions(
            basepath=tmp_path,
            sort_files=True,
            alt_sort=None,
            ignore_folders=[],
            subsession_order=["session_1/amplifier.dat"],
        )


def test_explicit_subsession_order_rejects_duplicates(tmp_path: Path) -> None:
    _write_epoch(tmp_path, "session_1")
    _write_epoch(tmp_path, "session_2")

    with pytest.raises(ValueError, match="duplicate paths"):
        discover_subsessions(
            basepath=tmp_path,
            sort_files=True,
            alt_sort=None,
            ignore_folders=[],
            subsession_order=[
                "session_1/amplifier.dat",
                "session_1/amplifier.dat",
            ],
        )


def _write_openephys_epoch(
    session: Path,
    name: str,
    *,
    n_channels: int,
    n_samples: int,
    sample_rate: float = 20000.0,
    n_ephys_channels: int | None = None,
    n_adc_channels: int = 0,
) -> Path:
    recording_root = session / name / "Record Node 101" / "experiment1" / "recording1"
    continuous_dir = recording_root / "continuous" / "Acquisition_Board-100.acquisition_board"
    continuous_dir.mkdir(parents=True)
    if n_ephys_channels is None:
        n_ephys_channels = n_channels - n_adc_channels
    channels = [
        {
            "channel_name": f"CH{channel + 1}",
            "identifier": "acq-board.rhythm.continuous.ephys",
            "units": "uV",
        }
        for channel in range(n_ephys_channels)
    ]
    channels.extend(
        {
            "channel_name": f"ADC{channel + 1}",
            "identifier": "acq-board.rhythm.continuous.adc",
            "units": "V",
        }
        for channel in range(n_adc_channels)
    )
    structure = {
        "continuous": [
            {
                "folder_name": "Acquisition_Board-100.acquisition_board/",
                "sample_rate": sample_rate,
                "recorded_processor": "Record Node",
                "recorded_processor_id": 101,
                "num_channels": n_channels,
                "channels": channels,
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


def test_prepare_multi_day_basepath_filters_selected_subepochs_and_removes_stale_links(
    tmp_path: Path,
) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    ep1 = _write_epoch(day1, "ep_240101_120000", n_samples=3)
    ep2 = _write_epoch(day1, "ep_240101_130000", n_samples=4)
    ep3 = _write_epoch(day2, "ep_240102_120000", n_samples=5)

    first = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        overwrite=True,
    )
    assert len(first.subepochs) == 3
    old_staged_paths = [Path(item.staged_subepoch_path) for item in first.subepochs]

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        selected_subepoch_paths=[ep2, ep3],
        local_root=tmp_path / "local",
        name="animal_multiday",
        overwrite=True,
    )

    assert [item.sample_count for item in staged.subepochs] == [4, 5]
    assert [Path(item.source_subepoch_path).resolve() for item in staged.subepochs] == [
        ep2.resolve(),
        ep3.resolve(),
    ]
    assert [Path(item.staged_subepoch_path).name for item in staged.subepochs] == [
        "001_animal_day1_ep_240101_130000",
        "002_animal_day2_ep_240102_120000",
    ]
    assert not old_staged_paths[0].exists()
    assert not old_staged_paths[1].exists()
    assert not old_staged_paths[2].exists()

    discovered = discover_subsessions(
        basepath=staged.server_basepath,
        sort_files=True,
        alt_sort=None,
        ignore_folders=[],
    )
    assert [path.resolve() for path in discovered] == [
        (ep2 / "amplifier.dat").resolve(),
        (ep3 / "amplifier.dat").resolve(),
    ]

    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert [Path(entry["source_subepoch_path"]).resolve() for entry in manifest["subepochs"]] == [
        ep2.resolve(),
        ep3.resolve(),
    ]
    with staged.selected_subepochs_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["staged_order"] for row in rows] == ["1", "2"]
    assert [Path(row["source_subepoch_path"]).resolve() for row in rows] == [
        ep2.resolve(),
        ep3.resolve(),
    ]
    assert [int(row["sample_count"]) for row in rows] == [4, 5]
    assert [row["source_type"] for row in rows] == ["intan", "intan"]
    assert [int(row["source_total_channels"]) for row in rows] == [4, 4]
    assert [int(row["source_ephys_channels"]) for row in rows] == [4, 4]
    assert [int(row["source_adc_channels"]) for row in rows] == [0, 0]
    assert [int(row["binary_n_channels"]) for row in rows] == [4, 4]
    assert (staged.local_basepath / "multi_day_selected_subepochs.csv").exists()


def test_prepare_multi_day_basepath_ignores_unused_session_without_subepochs(
    tmp_path: Path,
) -> None:
    unused = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day3 = tmp_path / "animal_day3"
    unused.mkdir()
    day2.mkdir()
    day3.mkdir()
    (unused / "raw_only").mkdir()
    (unused / "raw_only" / "recording.rhd").write_bytes(b"raw")
    _write_xml(day2)
    _write_xml(day3)
    ep2 = _write_epoch(day2, "ep_240102_120000", n_samples=4)
    ep3 = _write_epoch(day3, "ep_240103_120000", n_samples=5)

    staged = prepare_multi_day_basepath(
        session_paths=[unused, day2, day3],
        selected_subepoch_paths=[ep2, ep3],
        local_root=tmp_path / "local",
        name="animal_multiday",
        overwrite=True,
    )

    assert [item.session_index for item in staged.subepochs] == [2, 3]
    assert [Path(item.source_subepoch_path).resolve() for item in staged.subepochs] == [
        ep2.resolve(),
        ep3.resolve(),
    ]


def test_prepare_multi_day_basepath_rejects_used_session_without_subepochs(
    tmp_path: Path,
) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    raw_only = day1 / "raw_only"
    raw_only.mkdir()
    (raw_only / "recording.rhd").write_bytes(b"raw")
    _write_xml(day1)
    _write_xml(day2)
    ep2 = _write_epoch(day2, "ep_240102_120000")

    with pytest.raises(FileNotFoundError, match="No subepochs found.*animal_day1"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            selected_subepoch_paths=[raw_only, ep2],
            local_root=tmp_path / "local",
            name="animal_multiday",
            overwrite=True,
        )


def test_prepare_multi_day_basepath_rejects_unknown_selected_subepoch(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")

    with pytest.raises(ValueError, match="Selected multi-day subepoch paths were not found"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            selected_subepoch_paths=[tmp_path / "missing_epoch"],
            local_root=tmp_path / "local",
            name="animal_multiday",
            overwrite=True,
        )


def test_pipeline_gui_settings_round_trips_multi_day_selected_subepochs(tmp_path: Path) -> None:
    settings = PipelineGuiSettings(
        basepath=str(tmp_path / "day1"),
        local_root=str(tmp_path / "local"),
        multi_day_enabled=True,
        multi_day_session_paths=[str(tmp_path / "day1"), str(tmp_path / "day2")],
        multi_day_selected_subepoch_paths=[
            str(tmp_path / "day1" / "ep1"),
            str(tmp_path / "day2" / "ep2"),
        ],
        multi_day_name="animal_multiday",
    )

    loaded = PipelineGuiSettings.from_json(settings.to_json())

    assert loaded.multi_day_selected_subepoch_paths == settings.multi_day_selected_subepoch_paths


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
    assert [entry["source_total_channels"] for entry in manifest["subepochs"]] == [200, 200]
    assert [entry["source_ephys_channels"] for entry in manifest["subepochs"]] == [200, 200]
    assert [entry["source_adc_channels"] for entry in manifest["subepochs"]] == [0, 0]


def test_prepare_multi_day_basepath_records_openephys_embedded_adc_metadata(tmp_path: Path) -> None:
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
    _write_openephys_epoch(
        day1,
        "animal_2026-06-24_16-38-56",
        n_channels=200,
        n_ephys_channels=192,
        n_adc_channels=8,
        n_samples=11,
    )
    _write_openephys_epoch(
        day2,
        "animal_2026-06-25_16-38-56",
        n_channels=200,
        n_ephys_channels=192,
        n_adc_channels=8,
        n_samples=13,
    )

    staged = prepare_multi_day_basepath(
        session_paths=[day1, day2],
        local_root=tmp_path / "local",
        name="animal_multiday",
        xml_path=selected_xml,
        overwrite=True,
    )

    assert [item.sample_count for item in staged.subepochs] == [11, 13]
    assert [item.binary_n_channels for item in staged.subepochs] == [200, 200]
    assert [item.source_total_channels for item in staged.subepochs] == [200, 200]
    assert [item.source_ephys_channels for item in staged.subepochs] == [192, 192]
    assert [item.source_adc_channels for item in staged.subepochs] == [8, 8]

    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert [entry["source_type"] for entry in manifest["subepochs"]] == ["openephys", "openephys"]
    assert [entry["source_total_channels"] for entry in manifest["subepochs"]] == [200, 200]
    assert [entry["source_ephys_channels"] for entry in manifest["subepochs"]] == [192, 192]
    assert [entry["source_adc_channels"] for entry in manifest["subepochs"]] == [8, 8]
    assert [entry["binary_n_channels"] for entry in manifest["subepochs"]] == [200, 200]

    with staged.selected_subepochs_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["source_type"] for row in rows] == ["openephys", "openephys"]
    assert [int(row["source_total_channels"]) for row in rows] == [200, 200]
    assert [int(row["source_ephys_channels"]) for row in rows] == [192, 192]
    assert [int(row["source_adc_channels"]) for row in rows] == [8, 8]
    assert [int(row["binary_n_channels"]) for row in rows] == [200, 200]


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


def test_multiday_overwrite_false_rejects_conflicting_xml_before_link_mutation(tmp_path: Path) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    multiday = tmp_path / "animal_multiday"
    day1.mkdir()
    day2.mkdir()
    multiday.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")
    existing_xml = multiday / "animal_multiday.xml"
    existing_xml.write_text("<different />", encoding="utf-8")

    with pytest.raises(FileExistsError, match="XML already exists"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            local_root=tmp_path / "local",
            name="animal_multiday",
            xml_path=day1 / "animal_day1.xml",
            overwrite=False,
        )

    assert existing_xml.read_text(encoding="utf-8") == "<different />"
    assert not list(multiday.glob("001_*"))


def test_multiday_publish_failure_rolls_back_server_and_local_views(tmp_path: Path, monkeypatch) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    epoch1 = _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")
    initial = prepare_multi_day_basepath(
        session_paths=[day1, day2], local_root=tmp_path / "local", name="animal_multiday"
    )
    server_manifest_before = initial.manifest_path.read_bytes()
    local_manifest = initial.local_basepath / "multi_day_manifest.json"
    local_manifest_before = local_manifest.read_bytes()
    links_before = {
        Path(entry.staged_subepoch_path).name: os.readlink(entry.staged_subepoch_path)
        for entry in initial.subepochs
    }

    import src.preprocess.multiday as multiday

    real_replace = multiday.os.replace

    def fail_local_manifest(source, destination):
        if Path(destination) == local_manifest and ".multiday-publish-" in str(source):
            raise OSError("injected local manifest publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(multiday.os, "replace", fail_local_manifest)
    with pytest.raises(OSError, match="injected local manifest"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            selected_subepoch_paths=[epoch1],
            local_root=tmp_path / "local",
            name="animal_multiday",
            overwrite=True,
        )

    assert initial.manifest_path.read_bytes() == server_manifest_before
    assert local_manifest.read_bytes() == local_manifest_before
    assert {
        Path(entry.staged_subepoch_path).name: os.readlink(entry.staged_subepoch_path)
        for entry in initial.subepochs
    } == links_before


def test_multiday_interrupt_rolls_back_before_transaction_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    epoch1 = _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")
    initial = prepare_multi_day_basepath(
        session_paths=[day1, day2], local_root=tmp_path / "local", name="animal_multiday"
    )
    server_manifest_before = initial.manifest_path.read_bytes()
    local_manifest = initial.local_basepath / "multi_day_manifest.json"
    local_manifest_before = local_manifest.read_bytes()
    links_before = {
        Path(entry.staged_subepoch_path).name: os.readlink(entry.staged_subepoch_path)
        for entry in initial.subepochs
    }

    import src.preprocess.multiday as multiday

    real_replace = multiday.os.replace

    def interrupt_local_manifest(source, destination):
        if Path(destination) == local_manifest and ".multiday-publish-" in str(source):
            raise KeyboardInterrupt("injected cancellation")
        return real_replace(source, destination)

    monkeypatch.setattr(multiday.os, "replace", interrupt_local_manifest)
    with pytest.raises(KeyboardInterrupt, match="injected cancellation"):
        prepare_multi_day_basepath(
            session_paths=[day1, day2],
            selected_subepoch_paths=[epoch1],
            local_root=tmp_path / "local",
            name="animal_multiday",
            overwrite=True,
        )

    assert initial.manifest_path.read_bytes() == server_manifest_before
    assert local_manifest.read_bytes() == local_manifest_before
    assert {
        Path(entry.staged_subepoch_path).name: os.readlink(entry.staged_subepoch_path)
        for entry in initial.subepochs
    } == links_before


def test_multiday_uses_same_filesystem_staging_for_local_publication(tmp_path: Path, monkeypatch) -> None:
    day1 = tmp_path / "animal_day1"
    day2 = tmp_path / "animal_day2"
    day1.mkdir()
    day2.mkdir()
    _write_xml(day1)
    _write_xml(day2)
    _write_epoch(day1, "ep_240101_120000")
    _write_epoch(day2, "ep_240102_120000")
    local_root = tmp_path / "local"
    local_basepath = local_root / "animal_multiday"
    import src.preprocess.multiday as multiday

    real_replace = multiday.os.replace

    def reject_cross_mount_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if ".multiday-publish-" in str(source_path) and destination_path.parent == local_basepath:
            if not source_path.is_relative_to(local_basepath):
                raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(source, destination)

    monkeypatch.setattr(multiday.os, "replace", reject_cross_mount_replace)
    result = prepare_multi_day_basepath(
        session_paths=[day1, day2], local_root=local_root, name="animal_multiday"
    )

    assert result.manifest_path.exists()
    assert (local_basepath / "multi_day_manifest.json").exists()


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


@pytest.mark.parametrize(
    ("ephys_channels", "analog_channels"),
    [(64, 16), (128, 32), (192, 48)],
)
def test_wild_catalog_uses_manifest_analog_width_and_1250_hz(
    tmp_path: Path,
    ephys_channels: int,
    analog_channels: int,
) -> None:
    epoch = tmp_path / f"wild_{ephys_channels}ch"
    epoch.mkdir()
    ephys_samples = 32
    analog_samples = 2
    amplifier_path = epoch / "amplifier.dat"
    amplifier_path.write_bytes(
        np.zeros((ephys_samples, ephys_channels), dtype=np.int16).tobytes()
    )
    (epoch / "analogin.dat").write_bytes(
        np.zeros((analog_samples, analog_channels), dtype=np.int16).tobytes()
    )
    (epoch / "wild_preprocess_run.json").write_text(
        json.dumps(
            {
                "merge": {
                    "fs": 20000,
                    "n_channels": ephys_channels,
                    "n_samples": ephys_samples,
                    "analog_channels": analog_channels,
                    "analog_samples": analog_samples,
                }
            }
        ),
        encoding="utf-8",
    )

    catalog = build_acquisition_catalog(
        amplifier_paths=[amplifier_path],
        n_amplifier_channels=ephys_channels,
        dtype="int16",
    )

    assert catalog.source_type == "wild"
    assert catalog.source_adc_channels == [analog_channels]
    assert catalog.board_adc_channels == analog_channels
    assert catalog.analog_sample_counts_by_subsession == [analog_samples]
    assert catalog.analog_sampling_frequencies_by_subsession == [1250.0]
    assert catalog.ephys_sampling_frequencies_by_subsession == [20000.0]


def test_analog_concat_downsamples_non_wild_epochs_to_wild_rate(tmp_path: Path) -> None:
    wild = np.arange(12, dtype=np.int16).reshape(3, 4)
    wild_path = tmp_path / "wild-analogin.dat"
    wild_path.write_bytes(wild.tobytes())

    other = np.arange(32 * 4, dtype=np.int16).reshape(32, 4)
    other_path = tmp_path / "other-continuous.dat"
    other_path.write_bytes(other.tobytes())
    output_path = tmp_path / "analogin.dat"

    result = write_concatenated_dat_analogin(
        dat_paths=[wild_path, other_path],
        output_dat_path=output_path,
        sampling_frequency=1250.0,
        num_channels=4,
        overwrite=False,
        job_kwargs={},
        sample_counts=[3, 2],
        source_sample_counts=[3, 32],
        source_sampling_frequencies=[1250.0, 20000.0],
        source_num_channels=[4, 4],
        source_channel_indices=[None, [2, 3]],
        destination_channel_indices=[[0, 1, 2, 3], [0, 1]],
    )

    assert result == output_path
    actual = np.fromfile(output_path, dtype=np.uint16).reshape(-1, 4)
    expected_other = np.zeros((2, 4), dtype=np.uint16)
    expected_other[:, :2] = other[[0, 16]][:, [2, 3]].astype(np.uint16)
    np.testing.assert_array_equal(
        actual,
        np.vstack((wild.astype(np.uint16), expected_other)),
    )
    layout = json.loads((tmp_path / "analogin.dat.layout.json").read_text(encoding="utf-8"))
    assert layout["sampling_frequency"] == 1250.0
    assert layout["sample_counts"] == [3, 2]
    assert layout["source_sample_counts"] == [3, 32]
    assert layout["source_sampling_frequencies"] == [1250.0, 20000.0]


def test_concatenated_dat_removes_partial_after_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Recording:
        def get_num_channels(self) -> int:
            return 4

    def interrupted_write(_recording, *, file_paths, **_kwargs) -> None:
        Path(file_paths).write_bytes(b"incomplete")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "src.preprocess.recording.si.write_binary_recording",
        interrupted_write,
    )
    output = tmp_path / "session.dat"

    with pytest.raises(KeyboardInterrupt):
        write_concatenated_dat(
            recording=Recording(),
            output_dat_path=output,
            dtype="int16",
            overwrite=False,
            job_kwargs={},
        )

    assert not output.exists()
    assert list(tmp_path.glob("session.dat.partial-*")) == []
