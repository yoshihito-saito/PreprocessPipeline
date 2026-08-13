from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from src.preprocess.io import _resolve_openephys_stream_info


MAD_TO_GAUSSIAN_SIGMA = 1.482602218505602
DEFAULT_OUTPUT_CSV_NAME = "multi_day_mad.csv"
DEFAULT_OUTPUT_FIGURE_NAME = "multi_day_mad.png"

_REQUIRED_SELECTED_SUBEPOCH_COLUMNS = {
    "staged_order",
    "session_index",
    "subepoch_index",
    "session_name",
    "source_subepoch_path",
    "source_dat_path",
    "source_type",
    "source_ephys_channels",
    "binary_n_channels",
    "binary_sampling_frequency",
    "sample_count",
}

_OUTPUT_COLUMNS = [
    "staged_order",
    "session_index",
    "subepoch_index",
    "session_name",
    "source_subepoch_path",
    "source_dat_path",
    "source_type",
    "channel_index",
    "source_channel_index",
    "source_channel_name",
    "source_dtype",
    "channel_gain_to_uv",
    "channel_offset_to_uv",
    "scaling_source",
    "pre_filter_mad_uv",
    "post_filter_mad_uv",
    "sampling_frequency_hz",
    "bandpass_min_hz",
    "bandpass_max_hz",
    "n_windows",
    "window_duration_s",
    "sampled_duration_s",
]


@dataclass(frozen=True)
class MultiDayMadResult:
    """Files and in-memory metrics produced by :func:`analyze_multi_day_mad`."""

    metrics: pd.DataFrame
    csv_path: Path
    figure_path: Path | None


@dataclass(frozen=True)
class _LoadedSubepoch:
    recording: Any
    source_channel_indices: list[int]
    source_channel_names: list[str]
    scaling_source: str


def compute_mad_noise(traces: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Estimate Gaussian-equivalent background scale with a centered MAD.

    The returned value is ``1.482602218505602 * median(abs(x - median(x)))``.
    Input and output units are the same; the multi-day API reads traces in µV.
    """

    values = np.asarray(traces)
    if values.ndim == 0 or values.shape[axis] == 0:
        raise ValueError("MAD noise requires at least one sample along the requested axis.")
    if not np.all(np.isfinite(values)):
        raise ValueError("MAD noise cannot be computed from non-finite trace values.")
    center = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - center), axis=axis)
    return np.asarray(mad, dtype=np.float64) * MAD_TO_GAUSSIAN_SIGMA


def evenly_spaced_windows(
    *,
    num_frames: int,
    sampling_frequency_hz: float,
    window_duration_s: float,
    num_windows: int,
) -> list[tuple[int, int]]:
    """Return non-overlapping analysis windows distributed across a recording."""

    n_frames = int(num_frames)
    sampling_frequency = float(sampling_frequency_hz)
    duration = float(window_duration_s)
    requested_windows = int(num_windows)
    if n_frames <= 0:
        raise ValueError("num_frames must be greater than zero.")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency_hz must be finite and greater than zero.")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("window_duration_s must be finite and greater than zero.")
    if requested_windows <= 0:
        raise ValueError("num_windows must be greater than zero.")

    requested_window_frames = max(1, int(round(duration * sampling_frequency)))
    window_frames = min(requested_window_frames, n_frames)
    actual_windows = min(requested_windows, max(1, n_frames // window_frames))
    edges = np.linspace(0.0, float(n_frames), actual_windows + 1)

    windows: list[tuple[int, int]] = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        start = int(round((left + right - window_frames) / 2.0))
        start = min(max(0, start), n_frames - window_frames)
        windows.append((start, start + window_frames))
    return windows


def _read_selected_subepochs_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Selected-subepochs CSV does not exist: {path}")
    rows = pd.read_csv(path)
    if rows.empty:
        raise ValueError(f"Selected-subepochs CSV contains no rows: {path}")
    missing = sorted(_REQUIRED_SELECTED_SUBEPOCH_COLUMNS.difference(rows.columns))
    if missing:
        raise ValueError(
            "Selected-subepochs CSV is missing required columns: " + ", ".join(missing)
        )

    integer_columns = [
        "staged_order",
        "session_index",
        "subepoch_index",
        "source_ephys_channels",
        "binary_n_channels",
        "sample_count",
    ]
    for column in integer_columns:
        numeric = pd.to_numeric(rows[column], errors="raise")
        numeric_values = numeric.to_numpy(dtype=float)
        if (
            numeric.isna().any()
            or not np.all(np.isfinite(numeric_values))
            or not np.all(np.equal(numeric_values, np.floor(numeric_values)))
        ):
            raise ValueError(f"CSV column {column!r} must contain finite integers.")
        rows[column] = numeric.astype(np.int64)
    rows["binary_sampling_frequency"] = pd.to_numeric(
        rows["binary_sampling_frequency"], errors="raise"
    )

    if rows["staged_order"].duplicated().any():
        raise ValueError("Selected-subepochs CSV contains duplicate staged_order values.")
    for column in ("staged_order", "source_ephys_channels", "binary_n_channels", "sample_count"):
        if (rows[column] <= 0).any():
            raise ValueError(f"CSV column {column!r} must contain values greater than zero.")
    if (
        ~np.isfinite(rows["binary_sampling_frequency"].to_numpy(dtype=float))
    ).any() or (rows["binary_sampling_frequency"] <= 0).any():
        raise ValueError("CSV column 'binary_sampling_frequency' must be finite and positive.")

    rows = rows.sort_values("staged_order", kind="stable").reset_index(drop=True)
    return rows


def _load_subepoch_recording(
    row: pd.Series,
    *,
    dtype: str,
    gain_to_uV: float,
    offset_to_uV: float,
) -> _LoadedSubepoch:
    source_type = str(row["source_type"]).strip().lower()
    dat_path = Path(str(row["source_dat_path"])).expanduser().resolve()
    if not dat_path.exists() or not dat_path.is_file():
        raise FileNotFoundError(f"Source electrophysiology .dat file does not exist: {dat_path}")

    sampling_frequency = float(row["binary_sampling_frequency"])
    expected_ephys_channels = int(row["source_ephys_channels"])
    expected_samples = int(row["sample_count"])

    if source_type == "intan":
        binary_channels = int(row["binary_n_channels"])
        if binary_channels != expected_ephys_channels:
            raise ValueError(
                "Intan binary_n_channels and source_ephys_channels differ for "
                f"{dat_path}: {binary_channels} != {expected_ephys_channels}"
            )
        recording = se.read_binary(
            str(dat_path),
            sampling_frequency=sampling_frequency,
            dtype=dtype,
            num_channels=binary_channels,
            gain_to_uV=float(gain_to_uV),
            offset_to_uV=float(offset_to_uV),
        )
        source_channel_indices = list(range(binary_channels))
        source_channel_names = [f"CH{index + 1}" for index in source_channel_indices]
        scaling_source = "analysis_arguments"
    elif source_type == "openephys":
        recording_root = Path(str(row["source_subepoch_path"])).expanduser().resolve()
        info = _resolve_openephys_stream_info(recording_root)
        if info.continuous_dat.resolve() != dat_path:
            raise ValueError(
                "Open Ephys stream resolved to a different .dat file than the CSV row: "
                f"{info.continuous_dat} != {dat_path}"
            )
        if int(info.total_channels) != int(row["binary_n_channels"]):
            raise ValueError(
                "Open Ephys binary channel count does not match the CSV row for "
                f"{dat_path}: {info.total_channels} != {int(row['binary_n_channels'])}"
            )
        if len(info.ephys_channel_indices) != expected_ephys_channels:
            raise ValueError(
                "Open Ephys ephys-channel count does not match the CSV row for "
                f"{dat_path}: {len(info.ephys_channel_indices)} != {expected_ephys_channels}"
            )
        if not np.isclose(float(info.sampling_frequency), sampling_frequency):
            raise ValueError(
                "Open Ephys sampling frequency does not match the CSV row for "
                f"{dat_path}: {info.sampling_frequency} != {sampling_frequency}"
            )
        recording = se.read_openephys(
            folder_path=str(recording_root),
            stream_name=info.stream_name,
        )
        all_channel_ids = list(recording.get_channel_ids())
        selected_channel_ids = [all_channel_ids[index] for index in info.ephys_channel_indices]
        if hasattr(recording, "channel_slice"):
            recording = recording.channel_slice(channel_ids=selected_channel_ids)
        elif hasattr(recording, "select_channels"):
            recording = recording.select_channels(channel_ids=selected_channel_ids)
        else:
            raise AttributeError(
                "Open Ephys recording does not support channel_slice or select_channels."
            )
        recording = recording.rename_channels(list(range(recording.get_num_channels())))
        source_channel_indices = [int(index) for index in info.ephys_channel_indices]
        source_channel_names = [str(name) for name in info.ephys_channel_names]
        scaling_source = "open_ephys_metadata"
    else:
        raise ValueError(
            f"Unsupported source_type {row['source_type']!r}; expected 'intan' or 'openephys'."
        )

    if int(recording.get_num_segments()) != 1:
        raise ValueError(f"Expected one recording segment for subepoch {dat_path}.")
    actual_channels = int(recording.get_num_channels())
    if actual_channels != expected_ephys_channels:
        raise ValueError(
            f"Loaded ephys-channel count does not match CSV for {dat_path}: "
            f"{actual_channels} != {expected_ephys_channels}"
        )
    if len(source_channel_names) != actual_channels or len(set(source_channel_names)) != actual_channels:
        raise ValueError(
            f"Physical ephys channel names must be present and unique for {dat_path}."
        )
    actual_samples = int(recording.get_num_frames(segment_index=0))
    if actual_samples != expected_samples:
        raise ValueError(
            f"Loaded sample count does not match CSV for {dat_path}: "
            f"{actual_samples} != {expected_samples}"
        )
    return _LoadedSubepoch(
        recording=recording,
        source_channel_indices=source_channel_indices,
        source_channel_names=source_channel_names,
        scaling_source=scaling_source,
    )


def _get_traces_uV(recording: Any, *, start_frame: int, end_frame: int) -> np.ndarray:
    try:
        traces = recording.get_traces(
            segment_index=0,
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            return_in_uV=True,
        )
    except Exception as exc:
        raise ValueError(
            "Recording traces could not be read in µV. Check source scaling metadata or "
            "the Intan gain_to_uV/offset_to_uV arguments."
        ) from exc
    return np.asarray(traces)


def _get_filtered_traces_uV(recording: Any, *, start_frame: int, end_frame: int) -> np.ndarray:
    try:
        return np.asarray(
            recording.get_traces(
                segment_index=0,
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                return_in_uV=True,
            )
        )
    except ValueError as exc:
        raise ValueError(
            "Recording or analysis window is too short for the requested zero-phase "
            "bandpass filter. Increase window_duration_s or use a longer subepoch."
        ) from exc
    except Exception as exc:
        raise RuntimeError("Bandpass-filtered traces could not be read.") from exc


def _validate_analysis_parameters(
    *,
    bandpass_min_hz: float,
    bandpass_max_hz: float,
    window_duration_s: float,
    num_windows: int,
    gain_to_uV: float,
    offset_to_uV: float,
) -> None:
    low = float(bandpass_min_hz)
    high = float(bandpass_max_hz)
    if not np.isfinite(low) or low <= 0:
        raise ValueError("bandpass_min_hz must be finite and greater than zero.")
    if not np.isfinite(high) or high <= low:
        raise ValueError("bandpass_max_hz must be finite and greater than bandpass_min_hz.")
    if not np.isfinite(float(window_duration_s)) or float(window_duration_s) <= 0:
        raise ValueError("window_duration_s must be finite and greater than zero.")
    if int(num_windows) <= 0:
        raise ValueError("num_windows must be greater than zero.")
    if not np.isfinite(float(gain_to_uV)) or float(gain_to_uV) == 0:
        raise ValueError("gain_to_uV must be finite and non-zero.")
    if not np.isfinite(float(offset_to_uV)):
        raise ValueError("offset_to_uV must be finite.")


def compute_multi_day_mad(
    selected_subepochs_csv: Path | str,
    *,
    bandpass_min_hz: float = 500.0,
    bandpass_max_hz: float = 8000.0,
    window_duration_s: float = 1.0,
    num_windows: int = 20,
    dtype: str = "int16",
    gain_to_uV: float = 0.195,
    offset_to_uV: float = 0.0,
) -> pd.DataFrame:
    """Compute pre/post-bandpass per-channel MAD noise for staged subepochs."""

    _validate_analysis_parameters(
        bandpass_min_hz=bandpass_min_hz,
        bandpass_max_hz=bandpass_max_hz,
        window_duration_s=window_duration_s,
        num_windows=num_windows,
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
    )
    csv_path = Path(selected_subepochs_csv).expanduser().resolve()
    subepochs = _read_selected_subepochs_csv(csv_path)

    output_rows: list[dict[str, Any]] = []
    expected_channel_names: tuple[str, ...] | None = None
    for _, row in subepochs.iterrows():
        sampling_frequency = float(row["binary_sampling_frequency"])
        nyquist = sampling_frequency / 2.0
        if float(bandpass_max_hz) >= nyquist:
            raise ValueError(
                "bandpass_max_hz must be below Nyquist for every subepoch: "
                f"{bandpass_max_hz} >= {nyquist} Hz for staged_order={int(row['staged_order'])}"
            )

        loaded = _load_subepoch_recording(
            row,
            dtype=dtype,
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
        )
        channel_names = tuple(loaded.source_channel_names)
        if expected_channel_names is None:
            expected_channel_names = channel_names
        elif channel_names != expected_channel_names:
            raise ValueError(
                "Physical ephys channel map differs across selected subepochs. "
                f"Expected {list(expected_channel_names)}, got {list(channel_names)} "
                f"for staged_order={int(row['staged_order'])}."
            )
        recording = loaded.recording
        channel_gains = np.asarray(recording.get_channel_gains(), dtype=float)
        channel_offsets = np.asarray(recording.get_channel_offsets(), dtype=float)
        windows = evenly_spaced_windows(
            num_frames=int(recording.get_num_frames(segment_index=0)),
            sampling_frequency_hz=sampling_frequency,
            window_duration_s=window_duration_s,
            num_windows=num_windows,
        )
        filtered = spre.bandpass_filter(
            recording,
            freq_min=float(bandpass_min_hz),
            freq_max=float(bandpass_max_hz),
            dtype="float32",
        )

        pre_window_values: list[np.ndarray] = []
        post_window_values: list[np.ndarray] = []
        for start_frame, end_frame in windows:
            pre_window_values.append(
                compute_mad_noise(
                    _get_traces_uV(recording, start_frame=start_frame, end_frame=end_frame),
                    axis=0,
                )
            )
            post_window_values.append(
                compute_mad_noise(
                    _get_filtered_traces_uV(
                        filtered,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    ),
                    axis=0,
                )
            )

        pre_mad = np.median(np.vstack(pre_window_values), axis=0)
        post_mad = np.median(np.vstack(post_window_values), axis=0)
        sampled_frames = sum(end - start for start, end in windows)
        effective_window_duration = (windows[0][1] - windows[0][0]) / sampling_frequency

        for channel_index in range(int(recording.get_num_channels())):
            output_rows.append(
                {
                    "staged_order": int(row["staged_order"]),
                    "session_index": int(row["session_index"]),
                    "subepoch_index": int(row["subepoch_index"]),
                    "session_name": str(row["session_name"]),
                    "source_subepoch_path": str(row["source_subepoch_path"]),
                    "source_dat_path": str(row["source_dat_path"]),
                    "source_type": str(row["source_type"]),
                    "channel_index": channel_index,
                    "source_channel_index": loaded.source_channel_indices[channel_index],
                    "source_channel_name": loaded.source_channel_names[channel_index],
                    "source_dtype": str(recording.get_dtype()),
                    "channel_gain_to_uv": float(channel_gains[channel_index]),
                    "channel_offset_to_uv": float(channel_offsets[channel_index]),
                    "scaling_source": loaded.scaling_source,
                    "pre_filter_mad_uv": float(pre_mad[channel_index]),
                    "post_filter_mad_uv": float(post_mad[channel_index]),
                    "sampling_frequency_hz": sampling_frequency,
                    "bandpass_min_hz": float(bandpass_min_hz),
                    "bandpass_max_hz": float(bandpass_max_hz),
                    "n_windows": len(windows),
                    "window_duration_s": float(effective_window_duration),
                    "sampled_duration_s": float(sampled_frames / sampling_frequency),
                }
            )

    return pd.DataFrame(output_rows, columns=_OUTPUT_COLUMNS)


def _load_metrics(metrics: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        path = Path(metrics).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Multi-day MAD CSV does not exist: {path}")
        frame = pd.read_csv(path)
    required = {
        "staged_order",
        "session_name",
        "subepoch_index",
        "channel_index",
        "pre_filter_mad_uv",
        "post_filter_mad_uv",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("MAD metrics are missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("MAD metrics contain no rows to plot.")
    return frame


def _sparse_ticks(size: int, *, maximum: int) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, maximum, dtype=int))


def plot_multi_day_mad(
    metrics: pd.DataFrame | Path | str,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
):
    """Plot pre-filter and post-bandpass channel-by-subepoch MAD heatmaps."""

    import matplotlib.pyplot as plt

    frame = _load_metrics(metrics)
    orders = sorted(int(value) for value in frame["staged_order"].unique())
    channels = sorted(int(value) for value in frame["channel_index"].unique())
    labels_by_order: dict[int, str] = {}
    for order in orders:
        first = frame.loc[frame["staged_order"] == order].iloc[0]
        labels_by_order[order] = (
            f"{order}: {first['session_name']} / subepoch {int(first['subepoch_index'])}"
        )

    if figsize is None:
        figsize = (max(8.0, min(18.0, 0.75 * len(orders) + 5.0)), 8.0)
    fig, axes = plt.subplots(2, 1, figsize=figsize, constrained_layout=True, sharex=True)
    stages = [
        ("pre_filter_mad_uv", "Pre-filter MAD noise"),
        ("post_filter_mad_uv", "Post-bandpass MAD noise"),
    ]

    for axis, (value_column, title) in zip(axes, stages, strict=True):
        matrix = frame.pivot(
            index="channel_index",
            columns="staged_order",
            values=value_column,
        ).reindex(index=channels, columns=orders)
        image = axis.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
        )
        axis.set_title(title)
        axis.set_ylabel("Channel (0-based)")
        y_ticks = _sparse_ticks(len(channels), maximum=20)
        axis.set_yticks(y_ticks, [str(channels[index]) for index in y_ticks])
        colorbar = fig.colorbar(image, ax=axis, pad=0.01)
        colorbar.set_label("MAD-derived noise (µV)")

    x_ticks = _sparse_ticks(len(orders), maximum=24)
    axes[-1].set_xticks(
        x_ticks,
        [labels_by_order[orders[index]] for index in x_ticks],
        rotation=45,
        ha="right",
    )
    axes[-1].set_xlabel("Selected subepoch")
    return fig, axes


def plot_multi_day_mad_by_channel(
    metrics: pd.DataFrame | Path | str,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    linewidth: float = 0.8,
    alpha: float = 0.75,
):
    """Plot all channel MAD time series in pre/post-filter panels.

    Channel identity is encoded continuously by ``cmap`` and reported by a shared
    zero-based channel-index colorbar.
    """

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    line_width = float(linewidth)
    line_alpha = float(alpha)
    if not np.isfinite(line_width) or line_width <= 0:
        raise ValueError("linewidth must be finite and greater than zero.")
    if not np.isfinite(line_alpha) or not 0 < line_alpha <= 1:
        raise ValueError("alpha must be finite and in the interval (0, 1].")

    frame = _load_metrics(metrics)
    orders = sorted(int(value) for value in frame["staged_order"].unique())
    channels = sorted(int(value) for value in frame["channel_index"].unique())
    x_values = np.arange(len(orders), dtype=int)
    first_by_order = frame.sort_values("staged_order", kind="stable").drop_duplicates(
        "staged_order"
    ).set_index("staged_order")
    order_labels = [
        f"{first_by_order.loc[order, 'session_name']}\ns{int(first_by_order.loc[order, 'subepoch_index'])}"
        for order in orders
    ]
    color_map = plt.get_cmap(cmap)
    color_min = float(channels[0])
    color_max = float(channels[-1])
    if color_min == color_max:
        color_min -= 0.5
        color_max += 0.5
    normalization = Normalize(vmin=color_min, vmax=color_max)

    if figsize is None:
        figsize = (max(12.0, min(22.0, 0.32 * len(orders) + 6.0)), 10.0)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
    )
    stages = [
        ("pre_filter_mad_uv", "Pre-filter MAD by channel"),
        ("post_filter_mad_uv", "Post-bandpass MAD by channel"),
    ]
    for axis, (value_column, title) in zip(axes, stages, strict=True):
        for channel in channels:
            channel_values = (
                frame.loc[frame["channel_index"] == channel]
                .set_index("staged_order")[value_column]
                .reindex(orders)
                .to_numpy(dtype=float)
            )
            axis.plot(
                x_values,
                channel_values,
                color=color_map(normalization(channel)),
                linewidth=line_width,
                alpha=line_alpha,
            )
        axis.set_title(title)
        axis.set_ylabel("MAD-derived noise (µV)")
        axis.grid(alpha=0.2, linewidth=0.6)

    tick_indices = _sparse_ticks(len(orders), maximum=16)
    axes[-1].set_xticks(
        tick_indices,
        [order_labels[index] for index in tick_indices],
        rotation=45,
        ha="right",
    )
    axes[-1].set_xlabel("Selected subepoch")
    colorbar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=color_map),
        ax=axes,
        pad=0.015,
        aspect=35,
    )
    colorbar.set_label("Channel index (0-based)")
    colorbar.set_ticks(
        [
            channels[index]
            for index in _sparse_ticks(len(channels), maximum=9)
        ]
    )
    figure.suptitle("Multi-day MAD noise across channels")
    return figure, axes


def _check_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; enable overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial-{uuid.uuid4().hex}{path.suffix}")


def analyze_multi_day_mad(
    selected_subepochs_csv: Path | str,
    *,
    output_csv_path: Path | str | None = None,
    output_figure_path: Path | str | None = None,
    save_figure: bool = True,
    bandpass_min_hz: float = 500.0,
    bandpass_max_hz: float = 8000.0,
    window_duration_s: float = 1.0,
    num_windows: int = 20,
    dtype: str = "int16",
    gain_to_uV: float = 0.195,
    offset_to_uV: float = 0.0,
    overwrite: bool = False,
) -> MultiDayMadResult:
    """Compute, save, and plot standalone multi-day channel MAD noise metrics."""

    input_path = Path(selected_subepochs_csv).expanduser().resolve()
    csv_path = (
        Path(output_csv_path).expanduser().resolve()
        if output_csv_path is not None
        else input_path.with_name(DEFAULT_OUTPUT_CSV_NAME)
    )
    figure_path = None
    if save_figure:
        figure_path = (
            Path(output_figure_path).expanduser().resolve()
            if output_figure_path is not None
            else input_path.with_name(DEFAULT_OUTPUT_FIGURE_NAME)
        )

    if csv_path == input_path:
        raise ValueError("output_csv_path must differ from selected_subepochs_csv.")
    if figure_path is not None and figure_path in {input_path, csv_path}:
        raise ValueError(
            "output_figure_path must differ from the input CSV and output CSV paths."
        )

    _check_output_path(csv_path, overwrite=overwrite)
    if figure_path is not None:
        _check_output_path(figure_path, overwrite=overwrite)

    metrics = compute_multi_day_mad(
        input_path,
        bandpass_min_hz=bandpass_min_hz,
        bandpass_max_hz=bandpass_max_hz,
        window_duration_s=window_duration_s,
        num_windows=num_windows,
        dtype=dtype,
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
    )

    csv_temporary = _temporary_output_path(csv_path)
    figure_temporary = _temporary_output_path(figure_path) if figure_path is not None else None
    figure = None
    try:
        metrics.to_csv(csv_temporary, index=False)
        if figure_temporary is not None:
            figure, _ = plot_multi_day_mad(metrics)
            figure_format = figure_path.suffix.lstrip(".") if figure_path is not None else "png"
            figure.savefig(figure_temporary, dpi=150, format=figure_format or "png")
        os.replace(csv_temporary, csv_path)
        if figure_temporary is not None and figure_path is not None:
            os.replace(figure_temporary, figure_path)
    finally:
        if csv_temporary.exists():
            csv_temporary.unlink()
        if figure_temporary is not None and figure_temporary.exists():
            figure_temporary.unlink()
        if figure is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)

    return MultiDayMadResult(metrics=metrics, csv_path=csv_path, figure_path=figure_path)
