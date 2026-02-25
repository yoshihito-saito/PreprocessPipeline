from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from .metafile import MergePointsData, SessionXmlMeta

SESSION_SCHEMA = "neurocode_strict"


def _matlab_empty_char_array() -> np.ndarray:
    return np.asarray([], dtype="U1")


def _channel_dtype(n_channels: int) -> np.dtype:
    if n_channels <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if n_channels <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _as_1based_group_arrays(groups_0based: list[list[int]], n_channels: int) -> np.ndarray:
    cell = np.empty((1, len(groups_0based)), dtype=object)
    for i, group in enumerate(groups_0based):
        # Keep channel lists as 1xM double row-vectors to match MATLAB sessionTemplate output.
        cell[0, i] = (np.asarray(group, dtype=np.float64) + 1.0).reshape(1, -1)
    return cell


def _normalized_scalar(value: float | int | None) -> float | int | np.ndarray:
    if value is None:
        return _matlab_empty_char_array()
    as_float = float(value)
    if as_float.is_integer():
        return int(as_float)
    return as_float


def _compute_sample_count(dat_path: Path, n_channels: int, dtype: str) -> int:
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing dat file for neurocode_strict session: {dat_path}")

    frame_bytes = n_channels * np.dtype(dtype).itemsize
    size_bytes = dat_path.stat().st_size
    remainder = size_bytes % frame_bytes
    if remainder != 0:
        raise ValueError(
            f"Dat size {size_bytes} is not divisible by frame size {frame_bytes}. "
            f"Cannot infer exact nSamples for {dat_path}."
        )
    return int(size_bytes // frame_bytes)


def _build_epochs(merge_data: MergePointsData, basename: str) -> np.ndarray:
    epochs: list[dict[str, Any]] = []
    timestamps = np.asarray(merge_data.timestamps_sec)
    foldernames = list(merge_data.foldernames)

    if timestamps.ndim == 2 and timestamps.shape[1] == 2 and foldernames:
        n = min(len(foldernames), timestamps.shape[0])
        for i in range(n):
            epochs.append(
                {
                    "name": str(foldernames[i]),
                    "startTime": float(timestamps[i, 0]),
                    "stopTime": float(timestamps[i, 1]),
                }
            )
    else:
        epochs.append({"name": basename, "startTime": 0.0})
    return np.asarray(epochs, dtype=object)


def _build_spike_sorting_entry(
    *,
    relative_path: str,
    format_name: str,
    method_name: str,
    n_channels: int,
) -> dict[str, Any]:
    return {
        "relativePath": relative_path,
        "format": format_name,
        "method": method_name,
        "channels": np.asarray([], dtype=np.float64).reshape(1, 0),
        "manuallyCurated": 1,
        "notes": _matlab_empty_char_array(),
    }


def _detect_spike_sorting(basepath: Path, basename: str, n_channels: int) -> dict[str, Any] | None:
    relative_path = ""
    kilo_folders = sorted([p for p in basepath.glob("Kilosort_*") if p.is_dir()], key=lambda p: p.name)
    if kilo_folders:
        relative_path = kilo_folders[0].name

    search_root = basepath / relative_path if relative_path else basepath

    if (search_root / "spike_times.npy").is_file():
        return _build_spike_sorting_entry(
            relative_path=relative_path,
            format_name="Phy",
            method_name="KiloSort",
            n_channels=n_channels,
        )
    if any(search_root.glob(f"{basename}.res.*")):
        return _build_spike_sorting_entry(
            relative_path=relative_path,
            format_name="Klustakwik",
            method_name="Klustakwik",
            n_channels=n_channels,
        )
    if (search_root / f"{basename}.kwik").is_file():
        return _build_spike_sorting_entry(
            relative_path=relative_path,
            format_name="KlustaViewa",
            method_name="KlustaViewa",
            n_channels=n_channels,
        )
    if any(search_root.glob("times_raw_elec_CH*.mat")):
        return _build_spike_sorting_entry(
            relative_path=relative_path,
            format_name="UltraMegaSort2000",
            method_name="UltraMegaSort2000",
            n_channels=n_channels,
        )
    if any(search_root.glob("TT*.mat")):
        return _build_spike_sorting_entry(
            relative_path=relative_path,
            format_name="MClust",
            method_name="MClust",
            n_channels=n_channels,
        )
    return None


def build_session_struct(
    *,
    source_basepath: Path,
    local_basepath: Path,
    session_basepath_mode: str,
    basename: str,
    dat_path: Path,
    dat_dtype: str = "int16",
    sr: float,
    sr_lfp: float | None,
    n_channels: int,
    bad_channels_1based: list[int],
    merge_data: MergePointsData,
    xml_meta: SessionXmlMeta,
) -> dict[str, Any]:
    if SESSION_SCHEMA != "neurocode_strict":
        raise RuntimeError(f"Unsupported session schema: {SESSION_SCHEMA}")
    if session_basepath_mode not in {"local", "source"}:
        raise ValueError(
            f"session_basepath_mode must be 'local' or 'source', got: {session_basepath_mode}"
        )

    session_basepath = local_basepath if session_basepath_mode == "local" else source_basepath
    if source_basepath.parent != source_basepath:
        animal_name = source_basepath.parent.name
    else:
        animal_name = source_basepath.name

    n_samples = _compute_sample_count(dat_path=dat_path, n_channels=n_channels, dtype=dat_dtype)
    duration = float(n_samples) / float(sr)

    anat_groups = _as_1based_group_arrays(xml_meta.anatomical_groups_0based, n_channels)
    spike_groups_0based = (
        xml_meta.spike_groups_0based
        if xml_meta.spike_groups_0based
        else xml_meta.anatomical_groups_0based
    )
    spike_groups = _as_1based_group_arrays(spike_groups_0based, n_channels)

    bad_channels = np.asarray(
        sorted(set(int(ch) for ch in bad_channels_1based)),
        dtype=np.float64,
    ).reshape(1, -1)

    notes = f"Notes: {xml_meta.notes}   Description from xml: {xml_meta.description}"
    session: dict[str, Any] = {
        "general": {
            "version": 5,
            "name": basename,
            "basePath": str(session_basepath),
            "sessionType": "Unknown",
            "date": xml_meta.date if xml_meta.date is not None else _matlab_empty_char_array(),
            "notes": notes,
            "experimenters": (
                xml_meta.experimenters
                if xml_meta.experimenters is not None
                else _matlab_empty_char_array()
            ),
            "duration": duration,
        },
        "animal": {
            "name": animal_name,
            "sex": "Unknown",
            "species": "Unknown",
            "strain": "Unknown",
            "geneticLine": _matlab_empty_char_array(),
        },
        "extracellular": {
            "sr": _normalized_scalar(sr),
            "srLfp": _normalized_scalar(sr_lfp),
            "nChannels": int(n_channels),
            "fileName": _matlab_empty_char_array(),
            "electrodeGroups": {"channels": anat_groups},
            "spikeGroups": {"channels": spike_groups},
            "nElectrodeGroups": int(len(xml_meta.anatomical_groups_0based)),
            "nSpikeGroups": int(len(spike_groups_0based)),
            "nSamples": int(n_samples),
            "chanCoords": {"layout": "poly2", "verticalSpacing": 10},
            "probeDepths": 0,
            "precision": str(dat_dtype),
            "leastSignificantBit": 0.195,
        },
        "channelTags": {
            "Bad": {
                "channels": bad_channels,
            }
        },
        "epochs": _build_epochs(merge_data, basename),
    }

    spike_sorting = _detect_spike_sorting(source_basepath, basename, n_channels)
    if spike_sorting is not None:
        session["spikeSorting"] = np.asarray([spike_sorting], dtype=object)

    return session


def save_session_mat(path: Path, session_struct: dict[str, Any]) -> Path:
    savemat(path, {"session": session_struct}, do_compression=True)
    return path
