from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
import spikeinterface as si

from src.stability import channel_mad


matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _write_selected_subepochs_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "staged_order",
        "session_index",
        "subepoch_index",
        "session_name",
        "source_session_path",
        "source_subepoch_path",
        "source_dat_path",
        "staged_subepoch_path",
        "source_type",
        "source_total_channels",
        "source_ephys_channels",
        "source_adc_channels",
        "binary_n_channels",
        "binary_sampling_frequency",
        "sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _intan_row(
    *,
    staged_order: int,
    dat_path: Path,
    sampling_frequency: float,
    n_channels: int,
    n_samples: int,
) -> dict[str, object]:
    return {
        "staged_order": staged_order,
        "session_index": staged_order,
        "subepoch_index": 1,
        "session_name": f"day{staged_order}",
        "source_session_path": str(dat_path.parent),
        "source_subepoch_path": str(dat_path.parent),
        "source_dat_path": str(dat_path),
        "staged_subepoch_path": str(dat_path.parent),
        "source_type": "intan",
        "source_total_channels": n_channels,
        "source_ephys_channels": n_channels,
        "source_adc_channels": 0,
        "binary_n_channels": n_channels,
        "binary_sampling_frequency": sampling_frequency,
        "sample_count": n_samples,
    }


def test_compute_mad_noise_uses_centered_gaussian_equivalent_mad() -> None:
    traces = np.asarray(
        [
            [-1.0, 4.0],
            [0.0, 4.0],
            [1.0, 4.0],
        ]
    )

    noise = channel_mad.compute_mad_noise(traces, axis=0)

    np.testing.assert_allclose(
        noise,
        np.asarray([channel_mad.MAD_TO_GAUSSIAN_SIGMA, 0.0]),
    )


def test_compute_mad_noise_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        channel_mad.compute_mad_noise(np.asarray([[0.0], [np.nan]]))


def test_evenly_spaced_windows_are_non_overlapping_and_cover_short_recording() -> None:
    windows = channel_mad.evenly_spaced_windows(
        num_frames=1000,
        sampling_frequency_hz=100.0,
        window_duration_s=2.0,
        num_windows=3,
    )
    assert windows == [(67, 267), (400, 600), (733, 933)]
    assert all(left_end <= right_start for (_, left_end), (right_start, _) in zip(windows, windows[1:]))

    assert channel_mad.evenly_spaced_windows(
        num_frames=50,
        sampling_frequency_hz=100.0,
        window_duration_s=1.0,
        num_windows=20,
    ) == [(0, 50)]


def test_analyze_multi_day_mad_reads_intan_rows_and_writes_csv_and_plot(tmp_path: Path) -> None:
    sampling_frequency = 2000.0
    n_samples = 4000
    time = np.arange(n_samples, dtype=np.float64) / sampling_frequency
    rows: list[dict[str, object]] = []
    for staged_order, scale in ((1, 1.0), (2, 1.5)):
        folder = tmp_path / f"day{staged_order}"
        folder.mkdir()
        dat_path = folder / "amplifier.dat"
        traces = np.column_stack(
            [
                scale * (500.0 * np.sin(2 * np.pi * 20.0 * time) + 20.0 * np.sin(2 * np.pi * 200.0 * time)),
                scale * 100.0 * np.sin(2 * np.pi * 200.0 * time),
            ]
        ).astype(np.int16)
        traces.tofile(dat_path)
        rows.append(
            _intan_row(
                staged_order=staged_order,
                dat_path=dat_path,
                sampling_frequency=sampling_frequency,
                n_channels=2,
                n_samples=n_samples,
            )
        )

    selected_csv = tmp_path / "multi_day_selected_subepochs.csv"
    _write_selected_subepochs_csv(selected_csv, rows)

    result = channel_mad.analyze_multi_day_mad(
        selected_csv,
        bandpass_min_hz=50.0,
        bandpass_max_hz=500.0,
        window_duration_s=0.25,
        num_windows=4,
    )

    assert result.csv_path == tmp_path / "multi_day_mad.csv"
    assert result.figure_path == tmp_path / "multi_day_mad.png"
    assert result.csv_path.exists()
    assert result.figure_path.exists()
    assert result.metrics.shape[0] == 4
    assert result.metrics["staged_order"].tolist() == [1, 1, 2, 2]
    assert result.metrics["channel_index"].tolist() == [0, 1, 0, 1]
    assert set(result.metrics["n_windows"]) == {4}
    assert set(result.metrics["sampled_duration_s"]) == {1.0}

    channel_zero = result.metrics[result.metrics["channel_index"] == 0]
    assert np.all(
        channel_zero["pre_filter_mad_uv"]
        > 5.0 * channel_zero["post_filter_mad_uv"]
    )
    saved = pd.read_csv(result.csv_path)
    pd.testing.assert_frame_equal(saved, result.metrics, check_dtype=False)
    figure, axes = channel_mad.plot_multi_day_mad(result.csv_path)
    assert len(axes) == 2
    plt.close(figure)

    with pytest.raises(FileExistsError, match="enable overwrite"):
        channel_mad.analyze_multi_day_mad(
            selected_csv,
            bandpass_min_hz=50.0,
            bandpass_max_hz=500.0,
        )

    with pytest.raises(ValueError, match="must differ"):
        channel_mad.analyze_multi_day_mad(
            selected_csv,
            output_csv_path=selected_csv,
            save_figure=False,
            bandpass_min_hz=50.0,
            bandpass_max_hz=500.0,
            overwrite=True,
        )


def test_plot_multi_day_mad_by_channel_overlays_colored_pre_and_post_traces() -> None:
    rows = []
    for staged_order in (1, 2, 3):
        for channel_index in range(5):
            rows.append(
                {
                    "staged_order": staged_order,
                    "session_name": f"Day{staged_order}",
                    "subepoch_index": 1,
                    "channel_index": channel_index,
                    "source_channel_name": f"CH{channel_index + 1}",
                    "pre_filter_mad_uv": float(10 * staged_order + channel_index),
                    "post_filter_mad_uv": float(staged_order + channel_index),
                }
            )
    figure, axes = channel_mad.plot_multi_day_mad_by_channel(pd.DataFrame(rows))

    assert len(axes) == 2
    assert axes[0].get_title() == "Pre-filter MAD by channel"
    assert axes[1].get_title() == "Post-bandpass MAD by channel"
    assert len(axes[0].lines) == 5
    assert len(axes[1].lines) == 5
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [1.0, 2.0, 3.0])
    assert axes[0].lines[0].get_color() != axes[0].lines[-1].get_color()
    plt.close(figure)

    with pytest.raises(ValueError, match="alpha"):
        channel_mad.plot_multi_day_mad_by_channel(
            pd.DataFrame(rows), alpha=0.0
        )


def test_compute_multi_day_mad_selects_only_open_ephys_channels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sampling_frequency = 1000.0
    n_samples = 1000
    recording_root = tmp_path / "recording1"
    recording_root.mkdir()
    dat_path = recording_root / "continuous.dat"
    dat_path.write_bytes(b"placeholder")
    selected_csv = tmp_path / "multi_day_selected_subepochs.csv"
    row = _intan_row(
        staged_order=1,
        dat_path=dat_path,
        sampling_frequency=sampling_frequency,
        n_channels=3,
        n_samples=n_samples,
    )
    row.update(
        {
            "source_type": "openephys",
            "source_subepoch_path": str(recording_root),
            "source_total_channels": 3,
            "source_ephys_channels": 2,
            "source_adc_channels": 1,
            "binary_n_channels": 3,
        }
    )
    _write_selected_subepochs_csv(selected_csv, [row])

    info = SimpleNamespace(
        continuous_dat=dat_path,
        stream_name="Record Node 101#Acquisition_Board",
        total_channels=3,
        sampling_frequency=sampling_frequency,
        ephys_channel_indices=[0, 2],
        ephys_channel_names=["CH1", "CH3"],
    )
    monkeypatch.setattr(channel_mad, "_resolve_openephys_stream_info", lambda _path: info)

    time = np.arange(n_samples, dtype=np.float32) / sampling_frequency
    traces = np.column_stack(
        [
            np.sin(2 * np.pi * 100.0 * time),
            np.full(n_samples, 50.0, dtype=np.float32),
            2.0 * np.sin(2 * np.pi * 100.0 * time),
        ]
    ).astype(np.float32)
    recording = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=sampling_frequency,
        channel_ids=[10, 11, 12],
    )
    recording.set_channel_gains([1.0, 1.0, 1.0])
    recording.set_channel_offsets([0.0, 0.0, 0.0])
    monkeypatch.setattr(channel_mad.se, "read_openephys", lambda **_kwargs: recording)

    metrics = channel_mad.compute_multi_day_mad(
        selected_csv,
        bandpass_min_hz=50.0,
        bandpass_max_hz=300.0,
        window_duration_s=0.25,
        num_windows=2,
    )

    assert metrics["channel_index"].tolist() == [0, 1]
    assert metrics["source_channel_index"].tolist() == [0, 2]
    assert metrics["source_channel_name"].tolist() == ["CH1", "CH3"]


def test_compute_multi_day_mad_rejects_bandpass_at_nyquist(tmp_path: Path) -> None:
    dat_path = tmp_path / "amplifier.dat"
    np.zeros((100, 1), dtype=np.int16).tofile(dat_path)
    selected_csv = tmp_path / "multi_day_selected_subepochs.csv"
    _write_selected_subepochs_csv(
        selected_csv,
        [
            _intan_row(
                staged_order=1,
                dat_path=dat_path,
                sampling_frequency=1000.0,
                n_channels=1,
                n_samples=100,
            )
        ],
    )

    with pytest.raises(ValueError, match="below Nyquist"):
        channel_mad.compute_multi_day_mad(
            selected_csv,
            bandpass_min_hz=100.0,
            bandpass_max_hz=500.0,
        )


def test_compute_multi_day_mad_records_nondefault_intan_scaling(tmp_path: Path) -> None:
    pattern = np.asarray([-1, 0, 1, 0], dtype=np.int16)
    traces = np.tile(pattern, 250).reshape(-1, 1)
    dat_path = tmp_path / "amplifier.dat"
    traces.tofile(dat_path)
    selected_csv = tmp_path / "multi_day_selected_subepochs.csv"
    _write_selected_subepochs_csv(
        selected_csv,
        [
            _intan_row(
                staged_order=1,
                dat_path=dat_path,
                sampling_frequency=1000.0,
                n_channels=1,
                n_samples=1000,
            )
        ],
    )

    metrics = channel_mad.compute_multi_day_mad(
        selected_csv,
        bandpass_min_hz=50.0,
        bandpass_max_hz=300.0,
        window_duration_s=1.0,
        num_windows=1,
        gain_to_uV=0.5,
        offset_to_uV=3.0,
    )

    assert metrics.loc[0, "source_dtype"] == "int16"
    assert metrics.loc[0, "channel_gain_to_uv"] == 0.5
    assert metrics.loc[0, "channel_offset_to_uv"] == 3.0
    assert metrics.loc[0, "scaling_source"] == "analysis_arguments"
    assert metrics.loc[0, "pre_filter_mad_uv"] == pytest.approx(
        0.25 * channel_mad.MAD_TO_GAUSSIAN_SIGMA
    )


def test_compute_multi_day_mad_rejects_inconsistent_physical_channel_maps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_csv = tmp_path / "multi_day_selected_subepochs.csv"
    rows = []
    for staged_order in (1, 2):
        dat_path = tmp_path / f"day{staged_order}.dat"
        dat_path.write_bytes(b"placeholder")
        rows.append(
            _intan_row(
                staged_order=staged_order,
                dat_path=dat_path,
                sampling_frequency=1000.0,
                n_channels=2,
                n_samples=1000,
            )
        )
    _write_selected_subepochs_csv(selected_csv, rows)

    recording = si.NumpyRecording(
        traces_list=[np.zeros((1000, 2), dtype=np.float32)],
        sampling_frequency=1000.0,
        channel_ids=[0, 1],
    )
    recording.set_channel_gains([1.0, 1.0])
    recording.set_channel_offsets([0.0, 0.0])

    def _load(row, **_kwargs):
        names = ["CH1", "CH2"] if int(row["staged_order"]) == 1 else ["CH2", "CH3"]
        return channel_mad._LoadedSubepoch(
            recording=recording,
            source_channel_indices=[0, 1],
            source_channel_names=names,
            scaling_source="test",
        )

    monkeypatch.setattr(channel_mad, "_load_subepoch_recording", _load)

    with pytest.raises(ValueError, match="Physical ephys channel map differs"):
        channel_mad.compute_multi_day_mad(
            selected_csv,
            bandpass_min_hz=50.0,
            bandpass_max_hz=300.0,
        )


def test_compute_multi_day_mad_reports_short_filter_input_clearly(tmp_path: Path) -> None:
    dat_path = tmp_path / "amplifier.dat"
    np.zeros((20, 1), dtype=np.int16).tofile(dat_path)
    selected_csv = tmp_path / "multi_day_selected_subepochs.csv"
    _write_selected_subepochs_csv(
        selected_csv,
        [
            _intan_row(
                staged_order=1,
                dat_path=dat_path,
                sampling_frequency=1000.0,
                n_channels=1,
                n_samples=20,
            )
        ],
    )

    with pytest.raises(ValueError, match="too short.*bandpass filter"):
        channel_mad.compute_multi_day_mad(
            selected_csv,
            bandpass_min_hz=50.0,
            bandpass_max_hz=300.0,
        )
