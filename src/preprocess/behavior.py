from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from .io import load_xml_metadata


DLC_DISCOVERY_PATTERNS = ("*_filtered.h5", "*_filtered.csv", "*.h5", "*.csv")
VIDEO_PATTERNS = ("*.avi", "*.mp4", "*.mov", "*.m4v")


@dataclass(frozen=True)
class DlcFile:
    folder_name: str
    folder_path: Path
    path: Path
    priority: int
    video_path: Path | None = None


@dataclass
class DlcTrackingTable:
    frame_index: np.ndarray
    x: np.ndarray
    y: np.ndarray
    likelihood: np.ndarray
    point_names: list[str]
    x_field_names: list[str]
    y_field_names: list[str]
    likelihood_field_names: list[str]


@dataclass
class BehaviorProcessingResult:
    behavior: dict[str, Any]
    output_path: Path
    warnings: list[str] = field(default_factory=list)
    dlc_files: list[DlcFile] = field(default_factory=list)
    pixel_to_cm_ratio: float | None = None
    clean_mask: np.ndarray | None = None
    sub_session_mask: np.ndarray | None = None


def _basename(basepath: Path) -> str:
    return Path(basepath).resolve().name


def _find_existing_mat(basepath: Path, basename: str, output_dir: Path | None, suffix: str) -> Path | None:
    candidates: list[Path] = []
    if output_dir is not None:
        candidates.append(output_dir / f"{basename}{suffix}")
    candidates.append(basepath / f"{basename}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_mergepoints(
    basepath: Path,
    basename: str,
    output_dir: Path | None,
) -> tuple[list[str], np.ndarray] | tuple[None, None]:
    path = _find_existing_mat(basepath, basename, output_dir, ".MergePoints.events.mat")
    if path is None:
        return None, None
    loaded = loadmat(path, simplify_cells=True)
    merge = loaded.get("MergePoints")
    if not isinstance(merge, dict):
        raise ValueError(f"Invalid MergePoints structure in {path}")
    foldernames_raw = merge.get("foldernames", [])
    if isinstance(foldernames_raw, str):
        foldernames = [foldernames_raw]
    else:
        foldernames = [str(v) for v in np.asarray(foldernames_raw, dtype=object).reshape(-1).tolist()]
    timestamps = np.asarray(merge.get("timestamps", []), dtype=np.float64)
    if timestamps.ndim == 1 and timestamps.size == 2:
        timestamps = timestamps.reshape(1, 2)
    if timestamps.ndim != 2 or timestamps.shape[1] != 2:
        raise ValueError(f"MergePoints.timestamps must be N x 2 in {path}")
    return foldernames, timestamps


def _candidate_subsession_folders(basepath: Path, foldernames: list[str] | None) -> list[tuple[str, Path]]:
    if foldernames:
        return [(name, basepath / name) for name in foldernames]
    return [(child.name, child) for child in sorted(basepath.iterdir()) if child.is_dir()]


def _find_video_file(folder: Path) -> Path | None:
    for pattern in VIDEO_PATTERNS:
        matches = sorted(p for p in folder.glob(pattern) if p.is_file())
        if matches:
            return matches[0]
    return None


def discover_dlc_files(
    basepath: Path,
    *,
    output_dir: Path | None = None,
    basename: str | None = None,
) -> list[DlcFile]:
    """Discover DLC outputs in recording order, preferring filtered files."""
    basepath = Path(basepath).resolve()
    basename = basename or _basename(basepath)
    foldernames, _timestamps = _load_mergepoints(basepath, basename, output_dir)
    found: list[DlcFile] = []
    for folder_name, folder_path in _candidate_subsession_folders(basepath, foldernames):
        if not folder_path.exists() or not folder_path.is_dir():
            continue
        for priority, pattern in enumerate(DLC_DISCOVERY_PATTERNS):
            matches = sorted(p for p in folder_path.glob(pattern) if p.is_file())
            if matches:
                found.append(
                    DlcFile(
                        folder_name=folder_name,
                        folder_path=folder_path,
                        path=matches[0],
                        priority=priority,
                        video_path=_find_video_file(folder_path),
                    )
                )
                break
    return found


def _flatten_column_parts(parts: tuple[Any, ...] | Any) -> str:
    if not isinstance(parts, tuple):
        return str(parts)
    cleaned = [
        str(part)
        for part in parts
        if str(part) and not str(part).startswith("Unnamed:")
    ]
    return "_".join(cleaned)


def _point_name_from_column(parts: tuple[Any, ...] | Any, coord: str) -> str:
    if not isinstance(parts, tuple):
        text = str(parts)
        suffix = f"_{coord}"
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
        tokens = [token for token in text.split("_") if token]
        if len(tokens) >= 2 and tokens[0].lower().startswith("dlc"):
            # Best effort fallback for already-flattened DLC names. Prefer the
            # final two tokens so body_1 or ear_2 are not collapsed to 1/2.
            return "_".join(tokens[-2:])
        return text
    cleaned = [
        str(part)
        for part in parts
        if str(part) and not str(part).startswith("Unnamed:")
    ]
    if cleaned and cleaned[-1].lower() == coord.lower():
        cleaned = cleaned[:-1]
    if len(cleaned) >= 2 and cleaned[0].lower().startswith("dlc"):
        cleaned = cleaned[1:]
    return "_".join(cleaned) if cleaned else ""


def _coord_columns(field_names: list[str], coord: str) -> list[int]:
    coord_l = coord.lower()
    cols: list[int] = []
    for idx, name in enumerate(field_names):
        parts = [p.lower() for p in str(name).split("_") if p]
        if not parts:
            continue
        if parts[-1] == coord_l or str(name).lower().endswith(f"_{coord_l}"):
            cols.append(idx)
    return cols


def _dlc_point_name(field_name: str, coord: str) -> str:
    text = str(field_name)
    suffix = f"_{coord}"
    if text.lower().endswith(suffix):
        return text[: -len(suffix)]
    parts = [part for part in text.split("_") if part]
    if parts and parts[-1].lower() == coord.lower():
        return "_".join(parts[:-1])
    return text


def dlc_point_names(table: DlcTrackingTable) -> list[str]:
    return list(table.point_names)


def _load_dlc_csv(path: Path) -> DlcTrackingTable:
    last_error: Exception | None = None
    for header_rows in range(1, 51):
        try:
            df = pd.read_csv(path, header=list(range(header_rows)))
        except Exception as exc:
            last_error = exc
            continue
        field_names = [_flatten_column_parts(col) for col in df.columns]
        x_cols = _coord_columns(field_names, "x")
        y_cols = _coord_columns(field_names, "y")
        likelihood_cols = _coord_columns(field_names, "likelihood")
        if x_cols and y_cols and likelihood_cols:
            return _table_from_dataframe(df, field_names, x_cols, y_cols, likelihood_cols)
    if last_error is not None:
        raise ValueError(f"{path} is not a readable DLC CSV file: {last_error}") from last_error
    raise ValueError(f"{path} is not a DLC CSV file")


def _load_dlc_h5(path: Path) -> DlcTrackingTable:
    df = pd.read_hdf(path)
    field_names = [_flatten_column_parts(col) for col in df.columns]
    x_cols = _coord_columns(field_names, "x")
    y_cols = _coord_columns(field_names, "y")
    likelihood_cols = _coord_columns(field_names, "likelihood")
    if not (x_cols and y_cols and likelihood_cols):
        raise ValueError(f"{path} is not a DLC H5 file with x/y/likelihood columns")
    return _table_from_dataframe(df, field_names, x_cols, y_cols, likelihood_cols)


def _table_from_dataframe(
    df: pd.DataFrame,
    field_names: list[str],
    x_cols: list[int],
    y_cols: list[int],
    likelihood_cols: list[int],
) -> DlcTrackingTable:
    n_points = min(len(x_cols), len(y_cols), len(likelihood_cols))
    x_cols = x_cols[:n_points]
    y_cols = y_cols[:n_points]
    likelihood_cols = likelihood_cols[:n_points]
    x = df.iloc[:, x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    y = df.iloc[:, y_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    likelihood = df.iloc[:, likelihood_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    frame_index = pd.to_numeric(pd.Series(df.index), errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(frame_index)) or frame_index.size != x.shape[0]:
        frame_index = np.arange(x.shape[0], dtype=np.float64)
    return DlcTrackingTable(
        frame_index=frame_index,
        x=x,
        y=y,
        likelihood=likelihood,
        point_names=[_point_name_from_column(df.columns[i], "x") for i in x_cols],
        x_field_names=[field_names[i] for i in x_cols],
        y_field_names=[field_names[i] for i in y_cols],
        likelihood_field_names=[field_names[i] for i in likelihood_cols],
    )


def load_dlc_tracking(path: Path) -> DlcTrackingTable:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        return _load_dlc_h5(path)
    if suffix == ".csv":
        return _load_dlc_csv(path)
    raise ValueError(f"Unsupported DLC file type: {path}")


def resolve_dlc_primary_index(table: DlcTrackingTable, primary_coords: int = 1, primary_point: str | None = None) -> int:
    names = dlc_point_names(table)
    if primary_point:
        wanted = str(primary_point).strip()
        for idx, name in enumerate(names):
            if name == wanted:
                return idx
        wanted_l = wanted.lower()
        for idx, name in enumerate(names):
            if name.lower() == wanted_l:
                return idx
        raise ValueError(
            f"DLC point '{primary_point}' was not found. Available points: "
            + ", ".join(names)
        )
    primary_idx = int(primary_coords) - 1
    if primary_idx < 0:
        raise ValueError("primary_coords is MATLAB-style 1-based and must be >= 1.")
    if primary_idx >= len(names):
        raise ValueError(
            f"primary_coords={primary_coords} is out of range; available points={len(names)}"
        )
    return primary_idx


def reorder_dlc_primary(table: DlcTrackingTable, primary_idx: int) -> DlcTrackingTable:
    n_points = table.x.shape[1]
    if primary_idx < 0 or primary_idx >= n_points:
        raise ValueError(f"primary_idx={primary_idx} is out of range for {n_points} DLC point(s)")
    order = [primary_idx] + [idx for idx in range(n_points) if idx != primary_idx]
    return DlcTrackingTable(
        frame_index=table.frame_index,
        x=table.x[:, order],
        y=table.y[:, order],
        likelihood=table.likelihood[:, order],
        point_names=[table.point_names[idx] for idx in order],
        x_field_names=[table.x_field_names[idx] for idx in order],
        y_field_names=[table.y_field_names[idx] for idx in order],
        likelihood_field_names=[table.likelihood_field_names[idx] for idx in order],
    )


def _apply_likelihood_filter(table: DlcTrackingTable, threshold: float) -> DlcTrackingTable:
    x = table.x.copy()
    y = table.y.copy()
    bad = table.likelihood < float(threshold)
    x[bad] = np.nan
    y[bad] = np.nan
    return DlcTrackingTable(
        frame_index=table.frame_index,
        x=x,
        y=y,
        likelihood=table.likelihood,
        point_names=table.point_names,
        x_field_names=table.x_field_names,
        y_field_names=table.y_field_names,
        likelihood_field_names=table.likelihood_field_names,
    )


def _read_video_fps(video_path: Path | None) -> float | None:
    if video_path is None:
        return None
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video_path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if fps > 0:
            return fps
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore

        meta = iio.immeta(video_path)
        fps = meta.get("fps")
        if isinstance(fps, (int, float)) and float(fps) > 0:
            return float(fps)
    except Exception:
        return None
    return None


def load_representative_frame(video_path: Path) -> np.ndarray:
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if ok and frame is not None:
            return cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception as exc:
        raise RuntimeError("Reading video frames requires OpenCV or imageio.") from exc
    try:
        frame = iio.imread(video_path, index=0)
    except Exception as exc:
        raise RuntimeError(f"Could not load first frame from {video_path}: {exc}") from exc
    return np.asarray(frame)


def _cell_lengths(cell: Any) -> list[int]:
    if isinstance(cell, np.ndarray) and cell.dtype == object:
        values = cell.reshape(-1).tolist()
    elif isinstance(cell, (list, tuple)):
        values = list(cell)
    else:
        values = [cell]
    return [np.asarray(v).size for v in values]


def _cell_item(cell: Any, idx: int) -> np.ndarray:
    if isinstance(cell, np.ndarray) and cell.dtype == object:
        values = cell.reshape(-1).tolist()
        return np.asarray(values[idx], dtype=np.float64).reshape(-1)
    if isinstance(cell, (list, tuple)):
        return np.asarray(cell[idx], dtype=np.float64).reshape(-1)
    if idx == 0:
        return np.asarray(cell, dtype=np.float64).reshape(-1)
    return np.empty((0,), dtype=np.float64)


def _load_digital_in_timestamps(path: Path) -> np.ndarray:
    loaded = loadmat(path, simplify_cells=True)
    digital = loaded.get("digitalIn")
    if not isinstance(digital, dict):
        raise ValueError(f"Invalid digitalIn structure in {path}")
    timestamps_on = digital.get("timestampsOn")
    if timestamps_on is None:
        raise ValueError(f"digitalIn.timestampsOn is missing in {path}")
    lengths = _cell_lengths(timestamps_on)
    if not lengths or max(lengths) == 0:
        return np.empty((0,), dtype=np.float64)
    idx = int(np.argmax(lengths))
    return _cell_item(timestamps_on, idx)


def _digitalin_edges_from_dat(path: Path, sampling_rate: float) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size == 0:
        return np.empty((0,), dtype=np.float64)
    best_edges = np.empty((0,), dtype=np.int64)
    for bit_idx in range(16):
        bit = ((raw & np.uint16(1 << bit_idx)) != 0).astype(np.int8)
        edges = np.flatnonzero(np.diff(bit) == 1).astype(np.int64)
        if edges.size > best_edges.size:
            best_edges = edges
    return best_edges.astype(np.float64) / float(sampling_rate)


def _sampling_rate_from_xml(basepath: Path, basename: str) -> float:
    xml_path = basepath / f"{basename}.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Cannot infer digitalin.dat sampling rate without XML: {xml_path}")
    return float(load_xml_metadata(xml_path).sr)


def _load_local_or_global_ttl(
    *,
    basepath: Path,
    basename: str,
    output_dir: Path | None,
    dlc_file: DlcFile,
    merge_interval: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    local_events = dlc_file.folder_path / "digitalIn.events.mat"
    if local_events.exists():
        return _load_digital_in_timestamps(local_events), False

    local_dat = dlc_file.folder_path / "digitalin.dat"
    if local_dat.exists():
        return _digitalin_edges_from_dat(local_dat, _sampling_rate_from_xml(basepath, basename)), False

    global_candidates: list[Path] = []
    if output_dir is not None:
        global_candidates.append(output_dir / "digitalIn.events.mat")
    global_candidates.extend([basepath / "digitalIn.events.mat", basepath / f"{basename}.DigitalIn.events.mat"])
    for candidate in global_candidates:
        if candidate.exists():
            ttl = _load_digital_in_timestamps(candidate)
            if merge_interval is not None and ttl.size:
                start, stop = float(merge_interval[0]), float(merge_interval[1])
                ttl = ttl[(ttl >= start) & (ttl <= stop)]
            return ttl, True

    global_dat_candidates: list[Path] = []
    if output_dir is not None:
        global_dat_candidates.append(output_dir / "digitalin.dat")
    global_dat_candidates.append(basepath / "digitalin.dat")
    for candidate in global_dat_candidates:
        if candidate.exists():
            ttl = _digitalin_edges_from_dat(candidate, _sampling_rate_from_xml(basepath, basename))
            if merge_interval is not None and ttl.size:
                start, stop = float(merge_interval[0]), float(merge_interval[1])
                ttl = ttl[(ttl >= start) & (ttl <= stop)]
            return ttl, True

    raise FileNotFoundError(f"No digitalIn.events.mat or digitalin.dat found for {dlc_file.folder_path}")


def match_frames_to_ttl(
    ttl: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fps: float,
    pulses_delta_range: float = 0.01,
    context: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[str]]:
    ttl = np.asarray(ttl, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    warnings_out: list[str] = []
    if ttl.size == 0:
        raise ValueError("No camera TTL pulses were found.")
    if fps <= 0:
        raise ValueError(f"Video FPS must be positive, got {fps}")

    def _with_context(message: str) -> str:
        return f"{context}: {message}" if context else message

    def _warn(message: str) -> None:
        full_message = _with_context(message)
        warnings_out.append(full_message)

    def _format_indices(indices: np.ndarray) -> str:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return "none"
        if indices.size == 1:
            return str(int(indices[0]))
        if np.all(np.diff(indices) == 1):
            return f"{int(indices[0])}-{int(indices[-1])}"
        shown = ", ".join(str(int(value)) for value in indices[:10])
        if indices.size > 10:
            shown += f", ... ({indices.size} total)"
        return shown

    def _mismatch_details() -> str:
        seconds = abs(diff) / float(fps)
        n_common = min(n_ttl, n_frames)
        parts = [
            f"TTL pulses={n_ttl}",
            f"video frames={n_frames}",
            f"fps={float(fps):.6g} Hz",
            f"approx mismatch={seconds:.6g} s",
        ]
        if diff > 0:
            truncated = np.arange(n_common + 1, n_ttl + 1, dtype=np.int64)
            parts.append(f"current alignment truncates TTL pulse indices {_format_indices(truncated)}")
        elif diff < 0:
            truncated = np.arange(n_common + 1, n_frames + 1, dtype=np.int64)
            parts.append(f"current alignment truncates video frame indices {_format_indices(truncated)}")
        return "; ".join(parts)

    expected_frame_interval = 1.0 / float(fps)
    ttl_intervals = np.diff(ttl)
    extra_pulses = ttl_intervals < (
        expected_frame_interval - expected_frame_interval * float(pulses_delta_range)
    )
    if np.any(extra_pulses):
        removed_indices = np.flatnonzero(extra_pulses).astype(np.int64) + 2
        short_intervals = ttl_intervals[extra_pulses]
        msg = (
            f"Removed {int(np.sum(extra_pulses))} camera TTL pulses shorter than possible "
            f"(removed TTL pulse indices {_format_indices(removed_indices)}; "
            f"min interval={float(np.nanmin(short_intervals)):.6g} s; "
            f"expected frame interval={expected_frame_interval:.6g} s)."
        )
        _warn(msg)
        keep = np.ones(ttl.size, dtype=bool)
        keep[1:][extra_pulses] = False
        ttl = ttl[keep]

    n_ttl = int(ttl.size)
    n_frames = int(x.shape[0])
    diff = n_ttl - n_frames
    note: str
    if diff == 0:
        note = "N of frames match."
    elif abs(diff) <= 2:
        note = (
            f"{abs(diff)} frame TTL mismatch within tolerance; "
            f"aligned by truncating to common length. {_mismatch_details()}."
        )
        _warn(note)
        n = min(n_ttl, n_frames)
        ttl = ttl[:n]
        x = x[:n, :]
        y = y[:n, :]
    elif diff > 0 and abs(diff) < fps:
        note = (
            f"{abs(diff)} extra TTL pulses; probably at the end of the recording. "
            f"Truncated TTL. {_mismatch_details()}."
        )
        _warn(note)
        ttl = ttl[:n_frames]
    elif diff < 0 and abs(diff) < fps:
        note = (
            f"{abs(diff)} extra video frames; probably at the beginning/end of recording. "
            f"Truncated tracking. {_mismatch_details()}."
        )
        _warn(note)
        x = x[:n_ttl, :]
        y = y[:n_ttl, :]
    elif abs(diff) > 60 * fps:
        note = f"More than 1 minute frame/TTL mismatch; interpolated timestamps over TTL span. {_mismatch_details()}."
        _warn(note)
        ttl = np.linspace(float(np.nanmin(ttl)), float(np.nanmax(ttl)), n_frames)
    elif abs(diff) > 2 * fps:
        note = f"More than 2 seconds frame/TTL mismatch; truncated the longer stream. {_mismatch_details()}."
        _warn(note)
        n = min(n_ttl, n_frames)
        ttl = ttl[:n]
        x = x[:n, :]
        y = y[:n, :]
    elif diff > 0 and ttl[0] > 1.0:
        note = f"{diff} extra TTL pulses and first TTL > 1 s; assuming mismatch is at the end. {_mismatch_details()}."
        _warn(note)
        ttl = ttl[:n_frames]
    else:
        note = f"Unclassified camera/Intan mismatch; truncated to common length. {_mismatch_details()}."
        _warn(note)
        n = min(n_ttl, n_frames)
        ttl = ttl[:n]
        x = x[:n, :]
        y = y[:n, :]

    n_final = min(ttl.size, x.shape[0], y.shape[0])
    return ttl[:n_final], x[:n_final, :], y[:n_final, :], note, warnings_out


def inspect_dlc_ttl_sync(
    basepath: Path,
    *,
    output_dir: Path | None = None,
    basename: str | None = None,
    dlc_files: list[DlcFile] | None = None,
    pulses_delta_range: float = 0.01,
    fallback_video_fps: float = 40.0,
) -> list[str]:
    """Check DLC row counts against camera TTL pulses without producing behavior output."""
    basepath = Path(basepath).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else None
    basename = basename or _basename(basepath)
    files = list(dlc_files) if dlc_files is not None else discover_dlc_files(
        basepath,
        output_dir=output_dir,
        basename=basename,
    )

    foldernames, merge_timestamps = _load_mergepoints(basepath, basename, output_dir)
    merge_by_folder: dict[str, np.ndarray] = {}
    if foldernames is not None and merge_timestamps is not None:
        for idx, name in enumerate(foldernames):
            if idx < merge_timestamps.shape[0]:
                merge_by_folder[name] = merge_timestamps[idx, :]

    warnings_out: list[str] = []
    for dlc_file in files:
        table = load_dlc_tracking(dlc_file.path)
        fps = _read_video_fps(dlc_file.video_path) or fallback_video_fps
        if fps is None or fps <= 0:
            raise ValueError(
                f"Could not determine video FPS for {dlc_file.folder_name}. "
                "Set the Behavior tab fallback video FPS."
            )
        ttl, _ttl_is_global = _load_local_or_global_ttl(
            basepath=basepath,
            basename=basename,
            output_dir=output_dir,
            dlc_file=dlc_file,
            merge_interval=merge_by_folder.get(dlc_file.folder_name),
        )
        _matched_t, _matched_x, _matched_y, _note, warn = match_frames_to_ttl(
            ttl,
            table.x,
            table.y,
            fps=fps,
            pulses_delta_range=pulses_delta_range,
            context=dlc_file.folder_name,
        )
        warnings_out.extend(warn)
    return warnings_out


def _interpolate_short_gaps_1d(values: np.ndarray, timestamps: np.ndarray, max_gap_sec: float) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if max_gap_sec <= 0 or out.size == 0:
        return out
    isnan = ~np.isfinite(out)
    if not np.any(isnan):
        return out
    n = out.size
    idx = 0
    while idx < n:
        if not isnan[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and isnan[idx]:
            idx += 1
        stop = idx
        left = start - 1
        right = stop
        if left < 0 or right >= n or not np.isfinite(out[left]) or not np.isfinite(out[right]):
            continue
        gap_duration = float(timestamps[right] - timestamps[left])
        if gap_duration <= max_gap_sec:
            out[start:stop] = np.interp(timestamps[start:stop], [timestamps[left], timestamps[right]], [out[left], out[right]])
    return out


def interpolate_short_gaps(values: np.ndarray, timestamps: np.ndarray, max_gap_sec: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return _interpolate_short_gaps_1d(arr, timestamps, max_gap_sec)
    out = arr.copy()
    for col in range(out.shape[1]):
        out[:, col] = _interpolate_short_gaps_1d(out[:, col], timestamps, max_gap_sec)
    return out


def interpolate_short_gaps_by_epoch(
    values: np.ndarray,
    timestamps: np.ndarray,
    max_gap_sec: float,
    epoch_mask: np.ndarray | None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if epoch_mask is None:
        return interpolate_short_gaps(arr, timestamps, max_gap_sec)
    mask = np.asarray(epoch_mask).reshape(-1)
    if mask.size != np.asarray(timestamps).reshape(-1).size:
        return interpolate_short_gaps(arr, timestamps, max_gap_sec)
    out = arr.copy()
    for epoch_id in np.unique(mask):
        idx = np.flatnonzero(mask == epoch_id)
        if idx.size:
            out[idx] = interpolate_short_gaps(out[idx], np.asarray(timestamps)[idx], max_gap_sec)
    return out


def _speed_from_xy(timestamps: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    speed = np.full(t.shape, np.nan, dtype=np.float64)
    if t.size < 2:
        return np.zeros_like(t)
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    valid = np.isfinite(dt) & (dt > 0) & np.isfinite(dx) & np.isfinite(dy)
    step_speed = np.full(dt.shape, np.nan, dtype=np.float64)
    step_speed[valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / dt[valid]
    speed[0] = 0.0 if np.isfinite(step_speed[0]) else np.nan
    speed[1:] = step_speed
    return speed


def _session_epochs(basepath: Path, basename: str, output_dir: Path | None) -> Any:
    path = _find_existing_mat(basepath, basename, output_dir, ".session.mat")
    if path is None:
        return np.empty((0,), dtype=object)
    loaded = loadmat(path, simplify_cells=True)
    session = loaded.get("session")
    if isinstance(session, dict) and "epochs" in session:
        return session["epochs"]
    return np.empty((0,), dtype=object)


def _as_row(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def _matlab_field_name(name: str, *, max_len: int = 63) -> str:
    text = re.sub(r"\W", "_", str(name))
    if not text or not (text[0].isalpha() or text[0] == "_"):
        text = "x" + text
    return text[:max_len]


def _build_behavior_struct(
    *,
    basename: str,
    timestamps: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    z_cm: np.ndarray,
    speed: np.ndarray,
    trials: np.ndarray,
    epochs: Any,
    notes: list[str],
    source: str,
    settings: dict[str, Any],
    x_field_names: list[str],
    y_field_names: list[str],
    pixel_to_cm_ratio: float,
) -> dict[str, Any]:
    position: dict[str, Any] = {
        "x": _as_row(x_cm[:, 0] if x_cm.ndim == 2 and x_cm.shape[1] else np.empty((0,))),
        "y": _as_row(y_cm[:, 0] if y_cm.ndim == 2 and y_cm.shape[1] else np.empty((0,))),
        "z": _as_row(z_cm[:, 0] if z_cm.ndim == 2 and z_cm.shape[1] else np.empty((0,))),
        "linearized": np.empty((1, 0), dtype=np.float64),
        "units": "cm",
    }
    for idx, name in enumerate(x_field_names):
        if idx < x_cm.shape[1]:
            position[_matlab_field_name(f"{name}_point")] = _as_row(x_cm[:, idx])
    for idx, name in enumerate(y_field_names):
        if idx < y_cm.shape[1]:
            position[_matlab_field_name(f"{name}_point")] = _as_row(y_cm[:, idx])

    return {
        "sr": float(1.0 / np.nanmedian(np.diff(timestamps))) if timestamps.size > 1 else np.nan,
        "timestamps": _as_row(timestamps),
        "position": position,
        "speed": _as_row(speed),
        "acceleration": _as_row(np.concatenate(([0.0], np.diff(speed))) if speed.size else speed),
        "trials": np.asarray(trials, dtype=np.float64).reshape(-1, 2) if np.asarray(trials).size else np.empty((0, 2)),
        "trialID": np.empty((0, 0), dtype=np.float64),
        "states": np.empty((1, 0), dtype=np.float64),
        "stateNames": np.empty((0,), dtype=object),
        "notes": np.asarray(notes, dtype=object),
        "epochs": epochs,
        "processinginfo": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "function": "behavior.py",
            "source": source,
            "basename": basename,
            "pixel_to_cm_ratio": float(pixel_to_cm_ratio),
            "dlc_x_field_names": np.asarray(x_field_names, dtype=object),
            "dlc_y_field_names": np.asarray(y_field_names, dtype=object),
            "dlc_position_fields": np.asarray(
                [_matlab_field_name(f"{name}_point") for name in x_field_names]
                + [_matlab_field_name(f"{name}_point") for name in y_field_names],
                dtype=object,
            ),
            "varargin": settings,
        },
    }


def process_dlc_behavior(
    *,
    basepath: Path,
    output_dir: Path | None = None,
    basename: str | None = None,
    primary_coords: int = 1,
    primary_point: str | None = None,
    likelihood: float = 0.95,
    pulses_delta_range: float = 0.01,
    calibration_distance_cm: float | None = None,
    calibration_pixel_distance: float | None = None,
    pixel_to_cm_ratio: float | None = None,
    calibration_pixel_distances_by_folder: dict[str, float] | None = None,
    pixel_to_cm_ratios_by_folder: dict[str, float] | None = None,
    interpolate_gap_sec: float = 0.0,
    clean_mask: np.ndarray | None = None,
    fallback_video_fps: float | None = None,
    overwrite: bool = False,
    save_mat: bool = True,
) -> BehaviorProcessingResult:
    basepath = Path(basepath).resolve()
    basename = basename or _basename(basepath)
    if output_dir is None:
        raise ValueError("output_dir is required; behavior output should be written to the local output directory.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{basename}.animal.behavior.mat"
    if out_path.exists() and not overwrite and save_mat:
        return BehaviorProcessingResult(
            behavior=loadmat(out_path, simplify_cells=True).get("behavior", {}),
            output_path=out_path,
        )

    foldernames, merge_timestamps = _load_mergepoints(basepath, basename, output_dir)
    dlc_files = discover_dlc_files(basepath, output_dir=output_dir, basename=basename)
    if not dlc_files:
        raise FileNotFoundError(f"No DLC CSV/H5 files found under {basepath}")

    ratio_by_folder: dict[str, float] | None = None
    if pixel_to_cm_ratios_by_folder is not None:
        ratio_by_folder = {
            str(folder): float(ratio)
            for folder, ratio in pixel_to_cm_ratios_by_folder.items()
            if float(ratio) > 0
        }
    elif calibration_pixel_distances_by_folder is not None:
        if calibration_distance_cm is None or calibration_distance_cm <= 0:
            raise ValueError("calibration_distance_cm is required for centimeter behavior export.")
        ratio_by_folder = {
            str(folder): float(pixel_distance) / float(calibration_distance_cm)
            for folder, pixel_distance in calibration_pixel_distances_by_folder.items()
            if float(pixel_distance) > 0
        }
    else:
        if pixel_to_cm_ratio is None:
            if calibration_distance_cm is None or calibration_distance_cm <= 0:
                raise ValueError("calibration_distance_cm is required for centimeter behavior export.")
            if calibration_pixel_distance is None or calibration_pixel_distance <= 0:
                raise ValueError("calibration_pixel_distance is required for centimeter behavior export.")
            pixel_to_cm_ratio = float(calibration_pixel_distance) / float(calibration_distance_cm)
        if pixel_to_cm_ratio <= 0:
            raise ValueError(f"pixel_to_cm_ratio must be positive, got {pixel_to_cm_ratio}")
        ratio_by_folder = {dlc_file.folder_name: float(pixel_to_cm_ratio) for dlc_file in dlc_files}

    missing_ratios = [dlc_file.folder_name for dlc_file in dlc_files if dlc_file.folder_name not in ratio_by_folder]
    if missing_ratios:
        raise ValueError(
            "Calibration is missing for DLC epoch(s): "
            + ", ".join(missing_ratios)
        )
    ratio_values = np.asarray([ratio_by_folder[dlc_file.folder_name] for dlc_file in dlc_files], dtype=np.float64)
    if np.any(~np.isfinite(ratio_values)) or np.any(ratio_values <= 0):
        raise ValueError("All pixel-to-centimeter ratios must be positive finite values.")
    summary_pixel_to_cm_ratio = float(ratio_values[0]) if np.allclose(ratio_values, ratio_values[0]) else float(np.nanmedian(ratio_values))

    merge_by_folder: dict[str, np.ndarray] = {}
    if foldernames is not None and merge_timestamps is not None:
        for idx, name in enumerate(foldernames):
            if idx < merge_timestamps.shape[0]:
                merge_by_folder[name] = merge_timestamps[idx, :]

    all_t: list[np.ndarray] = []
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_ratio_rows: list[np.ndarray] = []
    all_subsession_mask: list[np.ndarray] = []
    all_trials: list[np.ndarray] = []
    notes: list[str] = []
    warnings_out: list[str] = []
    x_field_names: list[str] = []
    y_field_names: list[str] = []

    for dlc_file in dlc_files:
        table = _apply_likelihood_filter(load_dlc_tracking(dlc_file.path), likelihood)
        primary_idx = resolve_dlc_primary_index(table, primary_coords=primary_coords, primary_point=primary_point)
        table = reorder_dlc_primary(table, primary_idx)
        fps = _read_video_fps(dlc_file.video_path) or fallback_video_fps
        if fps is None or fps <= 0:
            raise ValueError(
                f"Could not determine video FPS for {dlc_file.folder_name}. "
                "Set the Behavior tab fallback video FPS."
            )
        merge_interval = merge_by_folder.get(dlc_file.folder_name)
        ttl, ttl_is_global = _load_local_or_global_ttl(
            basepath=basepath,
            basename=basename,
            output_dir=output_dir,
            dlc_file=dlc_file,
            merge_interval=merge_interval,
        )
        matched_t, matched_x, matched_y, note, warn = match_frames_to_ttl(
            ttl,
            table.x,
            table.y,
            fps=fps,
            pulses_delta_range=pulses_delta_range,
            context=dlc_file.folder_name,
        )
        if not ttl_is_global and merge_interval is not None:
            matched_t = matched_t + float(merge_interval[0])
        all_t.append(matched_t)
        all_x.append(matched_x)
        all_y.append(matched_y)
        all_ratio_rows.append(
            np.full(matched_t.shape, ratio_by_folder[dlc_file.folder_name], dtype=np.float64)
        )
        all_subsession_mask.append(np.full(matched_t.shape, len(all_subsession_mask) + 1, dtype=np.int32))
        if matched_t.size:
            if merge_interval is not None:
                all_trials.append(np.asarray(merge_interval, dtype=np.float64).reshape(1, 2))
            else:
                all_trials.append(np.asarray([[float(matched_t[0]), float(matched_t[-1])]], dtype=np.float64))
        notes.append(f"{dlc_file.folder_name}: {note}")
        warnings_out.extend(warn)
        if not x_field_names or len(table.x_field_names) > len(x_field_names):
            x_field_names = list(table.x_field_names)
            y_field_names = list(table.y_field_names)

    max_cols = max(arr.shape[1] for arr in all_x)
    total_rows = sum(arr.shape[0] for arr in all_x)
    timestamps = np.concatenate(all_t) if all_t else np.empty((0,), dtype=np.float64)
    pixel_to_cm_rows = np.concatenate(all_ratio_rows) if all_ratio_rows else np.empty((0,), dtype=np.float64)
    sub_session_mask = np.concatenate(all_subsession_mask) if all_subsession_mask else np.empty((0,), dtype=np.int32)
    x_pixels = np.full((total_rows, max_cols), np.nan, dtype=np.float64)
    y_pixels = np.full((total_rows, max_cols), np.nan, dtype=np.float64)
    row = 0
    for x_arr, y_arr in zip(all_x, all_y):
        n_rows, n_cols = x_arr.shape
        x_pixels[row : row + n_rows, :n_cols] = x_arr
        y_pixels[row : row + n_rows, :n_cols] = y_arr
        row += n_rows

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    pixel_to_cm_rows = pixel_to_cm_rows[order]
    sub_session_mask = sub_session_mask[order]
    x_cm = x_pixels[order, :] / pixel_to_cm_rows[:, None]
    y_cm = y_pixels[order, :] / pixel_to_cm_rows[:, None]
    z_cm = np.empty((timestamps.size, 0), dtype=np.float64)

    if clean_mask is not None:
        mask = np.asarray(clean_mask, dtype=bool).reshape(-1)
        if mask.size != timestamps.size:
            raise ValueError(f"clean_mask length {mask.size} does not match tracking length {timestamps.size}")
        x_cm[~mask, :] = np.nan
        y_cm[~mask, :] = np.nan
        notes.append(f"outlier_clean_mask: rejected_frames={int(np.sum(~mask))}")
    else:
        mask = np.ones(timestamps.size, dtype=bool)

    if interpolate_gap_sec > 0:
        before_nan = int(np.sum(~np.isfinite(x_cm)) + np.sum(~np.isfinite(y_cm)))
        x_cm = interpolate_short_gaps_by_epoch(x_cm, timestamps, interpolate_gap_sec, sub_session_mask)
        y_cm = interpolate_short_gaps_by_epoch(y_cm, timestamps, interpolate_gap_sec, sub_session_mask)
        after_nan = int(np.sum(~np.isfinite(x_cm)) + np.sum(~np.isfinite(y_cm)))
        notes.append(f"interpolated_short_gaps: filled_values={before_nan - after_nan}, threshold_sec={interpolate_gap_sec}")

    speed = _speed_from_xy(timestamps, x_cm[:, 0], y_cm[:, 0])
    trials = np.vstack(all_trials) if all_trials else np.empty((0, 2), dtype=np.float64)
    epochs = _session_epochs(basepath, basename, output_dir)
    settings = {
        "basepath": str(basepath),
        "output_dir": str(output_dir),
        "primary_coords": int(primary_coords),
        "primary_point": "" if primary_point is None else str(primary_point),
        "likelihood": float(likelihood),
        "pulses_delta_range": float(pulses_delta_range),
        "pixel_to_cm_ratio": summary_pixel_to_cm_ratio,
        "pixel_to_cm_ratios_by_folder": {folder: float(ratio) for folder, ratio in ratio_by_folder.items()},
        "interpolate_gap_sec": float(interpolate_gap_sec),
        "fallback_video_fps": None if fallback_video_fps is None else float(fallback_video_fps),
        "subSessionMask": sub_session_mask.reshape(1, -1),
        "subSessionNames": np.asarray([item.folder_name for item in dlc_files], dtype=object),
    }
    behavior = _build_behavior_struct(
        basename=basename,
        timestamps=timestamps,
        x_cm=x_cm,
        y_cm=y_cm,
        z_cm=z_cm,
        speed=speed,
        trials=trials,
        epochs=epochs,
        notes=notes,
        source="deeplabcut",
        settings=settings,
        x_field_names=x_field_names,
        y_field_names=y_field_names,
        pixel_to_cm_ratio=summary_pixel_to_cm_ratio,
    )
    if save_mat:
        savemat(out_path, {"behavior": behavior}, do_compression=True, long_field_names=True)
    return BehaviorProcessingResult(
        behavior=behavior,
        output_path=out_path,
        warnings=warnings_out,
        dlc_files=dlc_files,
        pixel_to_cm_ratio=summary_pixel_to_cm_ratio,
        clean_mask=mask,
        sub_session_mask=sub_session_mask,
    )


def calibration_ratio_from_line(pixel_distance: float, physical_distance_cm: float) -> float:
    if not math.isfinite(pixel_distance) or pixel_distance <= 0:
        raise ValueError("pixel_distance must be positive.")
    if not math.isfinite(physical_distance_cm) or physical_distance_cm <= 0:
        raise ValueError("physical_distance_cm must be positive.")
    return float(pixel_distance) / float(physical_distance_cm)
