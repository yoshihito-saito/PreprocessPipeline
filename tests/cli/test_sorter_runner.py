from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import savemat

import src.preprocess.sorter_runner as sr
from src.preprocess.terasort_runner import execute_terasort_job
import pytest

from src.preprocess.channel_layout import NoActiveChannels


def test_kilosort1_defaults_match_vendored_directory_case():
    root = Path(__file__).resolve().parents[2]
    expected = root / "sorter" / "KiloSort1"
    # Check the spelling explicitly: exists() alone accepts incorrect case
    # on Windows and would miss the Linux default-path failure.
    assert sr._default_kilosort1_path().as_posix() == expected.as_posix()
    args = sr.build_parser().parse_args([
        "--dat-path", "input.dat", "--output-folder", "out",
    ])
    assert args.kilosort1_path == "sorter/KiloSort1"
    assert expected.is_dir()


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


def test_sorter_cli_reports_all_excluded_channels_as_success(tmp_path, monkeypatch, capsys):
    def no_active_channels(**kwargs):
        raise NoActiveChannels("All recording channels are excluded")

    monkeypatch.setattr(sr, "execute_sorting_job", no_active_channels)
    args = sr.build_parser().parse_args([
        "--dat-path", str(tmp_path / "input.dat"),
        "--xml-path", str(tmp_path / "input.xml"),
        "--output-folder", str(tmp_path / "out"),
    ])
    sr.run_sorter_cli(args)
    assert "reason=no_active_channels" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_sorter_cli_requires_xml_before_sorter_execution(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(sr, "execute_sorting_job", lambda **kwargs: called.append(kwargs))
    args = sr.build_parser().parse_args([
        "--dat-path", str(tmp_path / "input.dat"),
        "--output-folder", str(tmp_path / "out"),
    ])
    with pytest.raises(ValueError, match="--xml-path is required"):
        sr.run_sorter_cli(args)
    assert called == []


def test_nt41_cublas_reaches_session_sort_and_resume_identity(tmp_path, monkeypatch):
    source = tmp_path / "input.dat"
    np.zeros((8, 2), dtype="<i2").tofile(source)
    chanmap = tmp_path / "chanMap.mat"
    savemat(chanmap, {"chanMap0ind": [0, 1], "xcoords": [0, 20],
                      "ycoords": [0, 20], "kcoords": [0, 0]})
    output = tmp_path / "Terasort_test"
    calls = []

    def finish_short_job(command, check):
        assert check is True
        calls.append(command)
        native = Path(command[command.index("--output-root") + 1])
        native.mkdir(parents=True, exist_ok=True)
        (native / "run.json").write_text("{}", encoding="utf-8")
        output.mkdir(exist_ok=True)
        (output / "params.py").write_text("", encoding="utf-8")
        for name in ("spike_times", "spike_clusters", "templates"):
            np.save(output / f"{name}.npy", np.empty(0))
        np.save(output / "channel_map.npy", [0, 1])

    monkeypatch.setattr("src.preprocess.terasort_runner.subprocess.run", finish_short_job)
    options = {"nt": 41, "seed_backend": "cublas", "core_seconds": 2.0}
    kwargs = dict(dat_path=source, chanmap_path=chanmap, output_folder=output,
                  terasort_path=Path(sys.executable), sampling_frequency=20000,
                  num_channels=2, gain_to_uV=0.2, offset_to_uV=0, dtype="int16",
                  active_channels_0based=[0, 1], input_is_preprocessed=True,
                  params=options)
    execute_terasort_job(**kwargs)
    assert calls[0][calls[0].index("--nt") + 1] == "41"
    assert calls[0][calls[0].index("--seed-backend") + 1] == "cublas"
    assert calls[0][calls[0].index("--core-seconds") + 1] == "2.0"
    job = json.loads((output.parent / ".terasort/Terasort_test.job.json").read_text())
    assert job["params"] == options

    execute_terasort_job(**kwargs, resume=True)
    assert "--resume" in calls[1]
    with pytest.raises(ValueError, match="changed or already exist"):
        execute_terasort_job(**(kwargs | {"params": {"nt": 61, "seed_backend": "cublas"}}),
                             resume=True)
    assert len(calls) == 2
