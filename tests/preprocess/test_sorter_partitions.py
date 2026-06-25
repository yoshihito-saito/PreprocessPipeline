from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

import src.preprocess.sorter_runner as sr
from src.preprocess.sorter_runner import build_sorter_partitions, write_sorter_partition_manifest


def _write_chanmap(path: Path) -> None:
    savemat(
        path,
        {
            "chanMap0ind": np.arange(8, dtype=np.int64),
            "chanMap": np.arange(1, 9, dtype=np.int64),
            "connected": np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=np.uint8),
            "probe_ids": np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64),
            "kcoords": np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.int64),
            "xcoords": np.arange(8, dtype=np.float64),
            "ycoords": np.arange(8, dtype=np.float64),
        },
    )


def test_build_sorter_partitions_by_probe_excludes_disconnected_and_rejected(tmp_path: Path) -> None:
    chanmap = tmp_path / "chanMap.mat"
    _write_chanmap(chanmap)

    partitions = build_sorter_partitions(
        mode="probe",
        chanmap_mat_path=chanmap,
        num_channels=8,
        excluded_channels_0based=[1],
    )

    assert [p.name for p in partitions] == ["probe1", "probe2"]
    assert partitions[0].channels_0based == [0, 2]
    assert partitions[0].channels_1based == [1, 3]
    assert partitions[1].channels_0based == [4, 5, 6, 7]
    assert partitions[0].excluded_channels_0based == [1]


def test_build_sorter_partitions_by_shank_keys_by_probe_and_shank(tmp_path: Path) -> None:
    chanmap = tmp_path / "chanMap.mat"
    _write_chanmap(chanmap)

    partitions = build_sorter_partitions(
        mode="shank",
        chanmap_mat_path=chanmap,
        num_channels=8,
        excluded_channels_0based=[],
    )

    assert [p.name for p in partitions] == [
        "probe1_shank1",
        "probe1_shank2",
        "probe2_shank1",
        "probe2_shank2",
    ]
    assert [p.channels_0based for p in partitions] == [[0, 1], [2], [4, 5], [6, 7]]


def test_write_sorter_partition_manifest_records_outputs(tmp_path: Path) -> None:
    chanmap = tmp_path / "chanMap.mat"
    _write_chanmap(chanmap)
    partitions = build_sorter_partitions(
        mode="probe",
        chanmap_mat_path=chanmap,
        num_channels=8,
        excluded_channels_0based=[1],
    )

    manifest = write_sorter_partition_manifest(
        output_dir=tmp_path,
        mode="probe",
        sorter="Kilosort",
        partitions=partitions,
    )

    text = manifest.read_text(encoding="utf-8")
    assert '"mode": "probe"' in text
    assert '"name": "probe1"' in text
    assert '"channels_0based"' in text


def test_run_sorter_cli_passes_active_and_excluded_channels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_sorting_job(**kwargs):
        captured.update(kwargs)
        return Path(kwargs["output_folder"])

    monkeypatch.setattr(sr, "execute_sorting_job", _fake_execute_sorting_job)

    parser = sr.build_parser()
    args = parser.parse_args(
        [
            "--sorter",
            "kilosort4",
            "--dat-path",
            str(tmp_path / "input.dat"),
            "--xml-path",
            str(tmp_path / "input.xml"),
            "--output-folder",
            str(tmp_path / "out"),
            "--active-channels",
            "0, 2,3",
            "--exclude-channels",
            "7; 8",
        ]
    )
    sr.run_sorter_cli(args)

    assert captured["active_channels_0based"] == [0, 2, 3]
    assert captured["exclude_channels_0based"] == [7, 8]


def test_run_sorter_cli_defaults_channel_lists_to_none(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_sorting_job(**kwargs):
        captured.update(kwargs)
        return Path(kwargs["output_folder"])

    monkeypatch.setattr(sr, "execute_sorting_job", _fake_execute_sorting_job)

    parser = sr.build_parser()
    args = parser.parse_args(
        [
            "--sorter",
            "kilosort4",
            "--dat-path",
            str(tmp_path / "input.dat"),
            "--xml-path",
            str(tmp_path / "input.xml"),
            "--output-folder",
            str(tmp_path / "out"),
        ]
    )
    sr.run_sorter_cli(args)

    assert captured["active_channels_0based"] is None
    assert captured["exclude_channels_0based"] is None
