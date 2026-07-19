from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat, savemat

import src.preprocess.pipeline as pipeline
import src.preprocess.sorter_runner as sr
from src.preprocess.recording import _load_bad_channels_from_chanmap
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


@pytest.mark.parametrize("ephys_indices", [None, [None], [[0, 1, 2, 3]]])
def test_prepare_effective_chanmap_preserves_identity_bad_channels(
    tmp_path: Path,
    ephys_indices,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_chanmap = source_dir / "chanMap.mat"
    savemat(
        source_chanmap,
        {
            "chanMap0ind": np.asarray([0, 1, 2, 3], dtype=np.int64),
            "chanMap": np.asarray([1, 2, 3, 4], dtype=np.int64),
            "connected": np.asarray([1, 0, 1, 0], dtype=np.uint8),
            "probe_ids": np.asarray([1, 1, 1, 1], dtype=np.int64),
            "kcoords": np.asarray([1, 1, 2, 2], dtype=np.int64),
            "xcoords": np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
            "ycoords": np.asarray([0.0, 0.0, 50.0, 50.0], dtype=np.float64),
        },
    )

    effective, reject_remap = pipeline._prepare_effective_chanmap_for_final_channel_space(
        chanmap_mat_path=source_chanmap,
        output_dir=tmp_path,
        final_num_channels=4,
        ephys_channel_indices_by_subsession=ephys_indices,
    )

    assert effective == tmp_path / "chanMap.mat"
    assert reject_remap is None
    assert _load_bad_channels_from_chanmap(effective) == [1, 3]
    mat = loadmat(effective, simplify_cells=True)
    assert np.array_equal(np.asarray(mat["chanMap0ind"]).reshape(-1), np.asarray([0, 1, 2, 3]))


def test_prepare_effective_chanmap_remaps_source_oe_indices_to_final_columns(
    tmp_path: Path,
) -> None:
    chanmap = tmp_path / "source_chanMap.mat"
    savemat(
        chanmap,
        {
            "chanMap0ind": np.asarray([0, 2, 4], dtype=np.int64),
            "chanMap": np.asarray([1, 3, 5], dtype=np.int64),
            "connected": np.asarray([1, 0, 1], dtype=np.uint8),
            "probe_ids": np.asarray([1, 1, 2], dtype=np.int64),
            "kcoords": np.asarray([1, 2, 1], dtype=np.int64),
            "xcoords": np.asarray([10.0, 20.0, 30.0], dtype=np.float64),
            "ycoords": np.asarray([0.0, 50.0, 100.0], dtype=np.float64),
        },
    )

    effective, reject_remap = pipeline._prepare_effective_chanmap_for_final_channel_space(
        chanmap_mat_path=chanmap,
        output_dir=tmp_path,
        final_num_channels=3,
        ephys_channel_indices_by_subsession=[None, [0, 2, 4]],
    )

    assert effective == tmp_path / "chanMap.mat"
    assert reject_remap == {0: 0, 2: 1, 4: 2}
    assert pipeline._normalize_reject_channels_for_final_channel_space([2], reject_remap) == [1]
    mat = loadmat(effective, simplify_cells=True)
    assert np.array_equal(np.asarray(mat["chanMap0ind"]).reshape(-1), np.asarray([0, 1, 2]))
    assert np.array_equal(np.asarray(mat["chanMap"]).reshape(-1), np.asarray([1, 2, 3]))
    assert np.array_equal(np.asarray(mat["connected"]).reshape(-1), np.asarray([1, 0, 1]))
    assert np.array_equal(np.asarray(mat["probe_ids"]).reshape(-1), np.asarray([1, 1, 2]))
    assert np.array_equal(np.asarray(mat["kcoords"]).reshape(-1), np.asarray([1, 2, 1]))
    assert np.array_equal(np.asarray(mat["xcoords"]).reshape(-1), np.asarray([10.0, 20.0, 30.0]))
    assert _load_bad_channels_from_chanmap(effective) == [1]

    partitions = build_sorter_partitions(
        mode="probe",
        chanmap_mat_path=effective,
        num_channels=3,
        excluded_channels_0based=[],
    )
    assert [partition.name for partition in partitions] == ["probe1", "probe2"]
    assert [partition.channels_0based for partition in partitions] == [[0], [2]]

    source_mat = loadmat(chanmap, simplify_cells=True)
    assert np.array_equal(np.asarray(source_mat["chanMap0ind"]).reshape(-1), np.asarray([0, 2, 4]))


def test_copy_chanmap_for_kilosort_preserves_final_channel_ids(tmp_path: Path) -> None:
    chanmap = tmp_path / "compact_subset_chanMap.mat"
    dst = tmp_path / "copied_chanMap.mat"
    savemat(
        chanmap,
        {
            "chanMap0ind": np.asarray([0, 2], dtype=np.int64),
            "chanMap": np.asarray([1, 3], dtype=np.int64),
            "connected": np.asarray([1, 1], dtype=np.uint8),
            "xcoords": np.asarray([10.0, 30.0], dtype=np.float64),
            "ycoords": np.asarray([0.0, 100.0], dtype=np.float64),
            "kcoords": np.asarray([1, 1], dtype=np.int64),
        },
    )

    sr._copy_chanmap_for_kilosort(chanmap, dst, n_channels_total=3)

    copied = loadmat(dst, simplify_cells=True)
    assert np.array_equal(np.asarray(copied["chanMap0ind"]).reshape(-1), np.asarray([0, 2]))
    assert np.array_equal(np.asarray(copied["chanMap"]).reshape(-1), np.asarray([1, 3]))
