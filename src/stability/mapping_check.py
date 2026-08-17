from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd


DEFAULT_MAPPING_OUTPUT_CSV_NAME = "multi_day_mapping_check.csv"
DEFAULT_MAPPING_OUTPUT_FIGURE_NAME = "multi_day_mapping_check.png"

_REQUIRED_MAD_COLUMNS = {
    "staged_order",
    "session_name",
    "subepoch_index",
    "channel_index",
    "pre_filter_mad_uv",
    "post_filter_mad_uv",
}

_OUTPUT_COLUMNS = [
    "previous_staged_order",
    "current_staged_order",
    "previous_session_name",
    "current_session_name",
    "previous_subepoch_index",
    "current_subepoch_index",
    "channel_start",
    "channel_end",
    "block_size",
    "candidate_permutation",
    "pre_identity_similarity",
    "post_identity_similarity",
    "identity_similarity",
    "profile_discontinuity",
    "pre_permuted_similarity",
    "post_permuted_similarity",
    "permuted_similarity",
    "pre_permutation_gain",
    "post_permutation_gain",
    "consensus_permutation_gain",
    "candidate_priority_score",
    "support_count",
    "best_for_transition",
    "candidate_rank",
]


@dataclass(frozen=True)
class MultiDayMappingCheckResult:
    """Files and metrics produced by :func:`analyze_multi_day_mapping_check`."""

    metrics: pd.DataFrame
    csv_path: Path
    figure_path: Path | None


@dataclass(frozen=True)
class _PermutationCandidate:
    channel_start: int
    channel_end: int
    name: str
    indices: np.ndarray


def _load_mad_metrics(metrics: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        path = Path(metrics).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Multi-day MAD CSV does not exist: {path}")
        frame = pd.read_csv(path)
    missing = sorted(_REQUIRED_MAD_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError("MAD metrics are missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("MAD metrics contain no rows for mapping analysis.")

    numeric_columns = [
        "staged_order",
        "subepoch_index",
        "channel_index",
        "pre_filter_mad_uv",
        "post_filter_mad_uv",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("staged_order", "subepoch_index", "channel_index"):
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"MAD mapping column {column!r} must contain finite integers.")
        frame[column] = frame[column].astype(np.int64)
    if (frame[["staged_order", "subepoch_index"]] <= 0).any().any():
        raise ValueError("MAD mapping staged_order and subepoch_index must be positive.")
    if not np.all(
        np.isfinite(frame[["pre_filter_mad_uv", "post_filter_mad_uv"]].to_numpy(float))
    ):
        raise ValueError("MAD mapping analysis requires finite pre/post MAD values.")
    if frame.duplicated(["staged_order", "channel_index"]).any():
        raise ValueError("MAD metrics contain duplicate staged-order/channel rows.")
    metadata_counts = frame.groupby("staged_order")[["session_name", "subepoch_index"]].nunique()
    if (metadata_counts > 1).any().any():
        raise ValueError("MAD metrics contain inconsistent session metadata within a subepoch.")

    orders = sorted(int(value) for value in frame["staged_order"].unique())
    if len(orders) < 2:
        raise ValueError("MAD mapping analysis requires at least two selected subepochs.")
    channels = sorted(int(value) for value in frame["channel_index"].unique())
    if channels != list(range(len(channels))):
        raise ValueError(
            "MAD mapping analysis requires contiguous zero-based channel indices."
        )
    expected_pairs = pd.MultiIndex.from_product(
        [orders, channels], names=["staged_order", "channel_index"]
    )
    actual_pairs = pd.MultiIndex.from_frame(
        frame[["staged_order", "channel_index"]].astype(int)
    )
    if len(actual_pairs) != len(expected_pairs) or not expected_pairs.isin(
        actual_pairs
    ).all():
        raise ValueError(
            "MAD metrics must contain the same complete channel set for every subepoch."
        )
    if len(channels) < 64 or len(channels) % 64 != 0:
        raise ValueError(
            "MAD mapping analysis requires a positive multiple of 64 channels "
            "to represent complete Intan headstage blocks."
        )
    return frame


def _candidate_permutations(num_channels: int) -> list[_PermutationCandidate]:
    candidates: list[_PermutationCandidate] = []
    for start in range(0, num_channels, 64):
        stop = start + 64
        candidates.extend(
            [
                _PermutationCandidate(
                    channel_start=start,
                    channel_end=stop - 1,
                    name="reverse_block",
                    indices=np.arange(63, -1, -1),
                ),
                _PermutationCandidate(
                    start,
                    stop - 1,
                    "swap_32_halves",
                    np.r_[np.arange(32, 64), np.arange(0, 32)],
                ),
                _PermutationCandidate(
                    start,
                    stop - 1,
                    "reverse_each_32",
                    np.r_[np.arange(31, -1, -1), np.arange(63, 31, -1)],
                ),
            ]
        )
    return candidates


def _spearman_similarity(
    left: np.ndarray, right: np.ndarray, *, context: str
) -> float:
    left_ranks = (
        pd.Series(np.asarray(left, dtype=float)).rank(method="average").to_numpy()
    )
    right_ranks = (
        pd.Series(np.asarray(right, dtype=float)).rank(method="average").to_numpy()
    )
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        raise ValueError(
            f"Channel MAD profile is constant and cannot identify mapping: {context}"
        )
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def compute_multi_day_mapping_check(
    mad_metrics: pd.DataFrame | Path | str,
) -> pd.DataFrame:
    """Score adjacent subepochs for plausible blockwise channel permutations.

    Similarities are Spearman correlations of channel MAD profiles. The conservative
    identity similarity is the smaller of the pre/post-filter identity correlations.
    The consensus permutation gain is the smaller pre/post improvement, so a
    candidate is positive only when both profiles support the same permutation.
    """

    frame = _load_mad_metrics(mad_metrics)
    orders = sorted(int(value) for value in frame["staged_order"].unique())
    channels = sorted(int(value) for value in frame["channel_index"].unique())
    candidates = _candidate_permutations(len(channels))
    metadata = (
        frame.sort_values(["staged_order", "channel_index"], kind="stable")
        .drop_duplicates("staged_order")
        .set_index("staged_order")
    )
    profiles = {
        column: frame.pivot(
            index="staged_order", columns="channel_index", values=column
        ).reindex(index=orders, columns=channels).to_numpy(dtype=float)
        for column in ("pre_filter_mad_uv", "post_filter_mad_uv")
    }

    output_rows: list[dict[str, Any]] = []
    for order_index in range(1, len(orders)):
        previous_order = orders[order_index - 1]
        current_order = orders[order_index]
        previous_meta = metadata.loc[previous_order]
        current_meta = metadata.loc[current_order]
        for candidate in candidates:
            start = candidate.channel_start
            stop = candidate.channel_end + 1
            similarities: dict[str, tuple[float, float]] = {}
            try:
                for stage, column in (
                    ("pre", "pre_filter_mad_uv"),
                    ("post", "post_filter_mad_uv"),
                ):
                    previous_values = profiles[column][order_index - 1, start:stop]
                    current_values = profiles[column][order_index, start:stop]
                    context = (
                        f"{previous_order}->{current_order}, "
                        f"channels {start}-{stop - 1}, {stage}"
                    )
                    identity = _spearman_similarity(
                        previous_values, current_values, context=context
                    )
                    permuted = _spearman_similarity(
                        previous_values,
                        current_values[candidate.indices],
                        context=context,
                    )
                    similarities[stage] = (identity, permuted)
            except ValueError as exc:
                if "profile is constant" not in str(exc):
                    raise
                continue

            pre_identity, pre_permuted = similarities["pre"]
            post_identity, post_permuted = similarities["post"]
            pre_gain = pre_permuted - pre_identity
            post_gain = post_permuted - post_identity
            identity_similarity = min(pre_identity, post_identity)
            permuted_similarity = min(pre_permuted, post_permuted)
            output_rows.append(
                {
                    "previous_staged_order": previous_order,
                    "current_staged_order": current_order,
                    "previous_session_name": str(previous_meta["session_name"]),
                    "current_session_name": str(current_meta["session_name"]),
                    "previous_subepoch_index": int(previous_meta["subepoch_index"]),
                    "current_subepoch_index": int(current_meta["subepoch_index"]),
                    "channel_start": start,
                    "channel_end": stop - 1,
                    "block_size": stop - start,
                    "candidate_permutation": candidate.name,
                    "pre_identity_similarity": pre_identity,
                    "post_identity_similarity": post_identity,
                    "identity_similarity": identity_similarity,
                    "profile_discontinuity": 1.0 - identity_similarity,
                    "pre_permuted_similarity": pre_permuted,
                    "post_permuted_similarity": post_permuted,
                    "permuted_similarity": permuted_similarity,
                    "pre_permutation_gain": pre_gain,
                    "post_permutation_gain": post_gain,
                    "consensus_permutation_gain": min(pre_gain, post_gain),
                    "support_count": int(pre_gain > 0) + int(post_gain > 0),
                }
            )

    if not output_rows:
        raise ValueError(
            "No informative mapping candidates remain; all candidate blocks have "
            "constant pre/post MAD profiles."
        )
    result = pd.DataFrame(output_rows)
    result["candidate_priority_score"] = np.maximum(
        result["consensus_permutation_gain"], 0.0
    ) * np.maximum(result["permuted_similarity"], 0.0)
    priority_order = result.sort_values(
        [
            "current_staged_order",
            "candidate_priority_score",
            "consensus_permutation_gain",
            "permuted_similarity",
        ],
        ascending=[True, False, False, False],
        kind="stable",
    )
    supported_order = priority_order.loc[
        priority_order["candidate_priority_score"] > 0
    ]
    best_indices = supported_order.groupby("current_staged_order", sort=False).head(1).index
    result["best_for_transition"] = False
    result.loc[best_indices, "best_for_transition"] = True
    global_order = result.sort_values(
        [
            "candidate_priority_score",
            "consensus_permutation_gain",
            "permuted_similarity",
        ],
        ascending=[False, False, False],
        kind="stable",
    )
    result["candidate_rank"] = 0
    result.loc[global_order.index, "candidate_rank"] = np.arange(1, len(result) + 1)
    result["candidate_rank"] = result["candidate_rank"].astype(np.int64)
    return (
        result[_OUTPUT_COLUMNS]
        .sort_values("candidate_rank", kind="stable")
        .reset_index(drop=True)
    )


def _load_mapping_metrics(metrics: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        path = Path(metrics).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Multi-day mapping-check CSV does not exist: {path}")
        frame = pd.read_csv(path)
    missing = sorted(set(_OUTPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            "Mapping-check metrics are missing required columns: " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Mapping-check metrics contain no rows to plot.")
    return frame


def plot_multi_day_mapping_check(
    metrics: pd.DataFrame | Path | str,
    *,
    figsize: tuple[float, float] | None = None,
    annotate_top: int = 5,
):
    """Plot the highest consensus permutation candidate at each transition."""

    import matplotlib.pyplot as plt

    top_count = int(annotate_top)
    if top_count < 0:
        raise ValueError("annotate_top must be zero or greater.")
    frame = _load_mapping_metrics(metrics)
    best = frame.loc[frame["best_for_transition"].astype(bool)].sort_values(
        "current_staged_order", kind="stable"
    )
    if best.empty:
        if figsize is None:
            figsize = (12.0, 4.0)
        figure, axis = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        axis.set_axis_off()
        axis.text(
            0.5,
            0.5,
            "No supported 64-channel headstage permutation candidates",
            ha="center",
            va="center",
            fontsize=14,
        )
        figure.suptitle("Multi-day channel-mapping candidate check")
        return figure, np.asarray([axis])
    x_values = np.arange(len(best), dtype=int)
    labels = [
        f"{row.previous_session_name}\n→{row.current_session_name}"
        for row in best.itertuples()
    ]
    if figsize is None:
        figsize = (max(12.0, min(22.0, 0.35 * len(best) + 5.0)), 8.0)
    figure, axes = plt.subplots(
        2, 1, figsize=figsize, constrained_layout=True, sharex=True
    )
    axes[0].plot(
        x_values,
        best["identity_similarity"].to_numpy(float),
        color="tab:blue",
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    axes[0].set_title("Identity similarity of the best candidate block")
    axes[0].set_ylabel("min(pre, post) Spearman r")
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    gains = best["consensus_permutation_gain"].to_numpy(float)
    axes[1].plot(
        x_values,
        gains,
        color="tab:red",
        marker="o",
        markersize=3,
        linewidth=1.2,
    )
    axes[1].set_title("Consensus permutation gain of the priority-ranked candidate")
    axes[1].set_ylabel("min(pre gain, post gain)")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].grid(alpha=0.25, linewidth=0.6)
    tick_indices = np.unique(
        np.linspace(0, len(best) - 1, min(16, len(best)), dtype=int)
    )
    axes[1].set_xticks(
        tick_indices,
        [labels[index] for index in tick_indices],
        rotation=45,
        ha="right",
    )
    axes[1].set_xlabel("Adjacent selected subepochs")

    if top_count > 0:
        priorities = best["candidate_priority_score"].to_numpy(float)
        positive = np.flatnonzero(priorities > 0)
        ranked = positive[np.argsort(priorities[positive])[::-1]][:top_count]
        for index in ranked:
            row = best.iloc[index]
            at_right_edge = index == len(best) - 1
            axes[1].annotate(
                f"ch{int(row['channel_start'])}-{int(row['channel_end'])}\n"
                f"{row['candidate_permutation']}",
                (x_values[index], gains[index]),
                xytext=(-4 if at_right_edge else 4, -8),
                textcoords="offset points",
                fontsize=7,
                rotation=20,
                va="top",
                ha="right" if at_right_edge else "left",
            )
    figure.suptitle("Multi-day channel-mapping candidate check")
    return figure, axes


def _check_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; enable overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _temporary_output_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.partial-{uuid.uuid4().hex}{path.suffix}")


def analyze_multi_day_mapping_check(
    mad_metrics_csv: Path | str,
    *,
    output_csv_path: Path | str | None = None,
    output_figure_path: Path | str | None = None,
    save_figure: bool = True,
    annotate_top: int = 5,
    overwrite: bool = False,
) -> MultiDayMappingCheckResult:
    """Compute and save standalone MAD-profile channel-mapping candidate scores."""

    input_path = Path(mad_metrics_csv).expanduser().resolve()
    csv_path = (
        Path(output_csv_path).expanduser().resolve()
        if output_csv_path is not None
        else input_path.with_name(DEFAULT_MAPPING_OUTPUT_CSV_NAME)
    )
    figure_path = None
    if save_figure:
        figure_path = (
            Path(output_figure_path).expanduser().resolve()
            if output_figure_path is not None
            else input_path.with_name(DEFAULT_MAPPING_OUTPUT_FIGURE_NAME)
        )
    if csv_path == input_path:
        raise ValueError("output_csv_path must differ from mad_metrics_csv.")
    if figure_path is not None and figure_path in {input_path, csv_path}:
        raise ValueError(
            "output_figure_path must differ from the input and output CSV paths."
        )
    _check_output_path(csv_path, overwrite=overwrite)
    if figure_path is not None:
        _check_output_path(figure_path, overwrite=overwrite)

    metrics = compute_multi_day_mapping_check(input_path)
    csv_temporary = _temporary_output_path(csv_path)
    figure_temporary = (
        _temporary_output_path(figure_path) if figure_path is not None else None
    )
    figure = None
    try:
        metrics.to_csv(csv_temporary, index=False)
        if figure_temporary is not None:
            figure, _ = plot_multi_day_mapping_check(metrics, annotate_top=annotate_top)
            figure_format = figure_path.suffix.lstrip(".") if figure_path is not None else "png"
            figure.savefig(figure_temporary, dpi=180, format=figure_format or "png")
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
    return MultiDayMappingCheckResult(metrics=metrics, csv_path=csv_path, figure_path=figure_path)
