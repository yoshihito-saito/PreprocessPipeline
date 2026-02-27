from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from .metafile import MergePointsData, SessionXmlMeta


def _matlab_empty_char_array() -> np.ndarray:
    return np.asarray([], dtype="U1")


def _channel_dtype(n_channels: int) -> np.dtype:
    if n_channels <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if n_channels <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _signed_channel_dtype(n_channels: int) -> np.dtype:
    if n_channels <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def _as_integer_column(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return np.zeros((0, 1), dtype=np.int16)
    if not np.all(np.isfinite(flat)):
        return flat.reshape(-1, 1)
    rounded = np.rint(flat)
    if np.allclose(flat, rounded, rtol=0, atol=0):
        as_int = rounded.astype(np.int64, copy=False)
        if np.min(as_int) >= 0:
            if np.max(as_int) <= np.iinfo(np.uint8).max:
                return as_int.astype(np.uint8).reshape(-1, 1)
            if np.max(as_int) <= np.iinfo(np.uint16).max:
                return as_int.astype(np.uint16).reshape(-1, 1)
            return as_int.astype(np.uint32).reshape(-1, 1)
        if np.min(as_int) >= np.iinfo(np.int16).min and np.max(as_int) <= np.iinfo(np.int16).max:
            return as_int.astype(np.int16).reshape(-1, 1)
        return as_int.astype(np.int32).reshape(-1, 1)
    return flat.reshape(-1, 1)


def _as_1based_group_arrays(groups_0based: list[list[int]], n_channels: int) -> np.ndarray:
    cell = np.empty((1, len(groups_0based)), dtype=object)
    chan_dtype = _channel_dtype(n_channels)
    for i, group in enumerate(groups_0based):
        cell[0, i] = (np.asarray(group, dtype=np.int64) + 1).astype(chan_dtype, copy=False).reshape(1, -1)
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
        "channels": np.asarray([], dtype=_signed_channel_dtype(n_channels)).reshape(1, 0),
        "manuallyCurated": 1,
        "notes": _matlab_empty_char_array(),
    }


def _first_kilosort_folder(basepath: Path) -> Path | None:
    kilo_folders = sorted(
        [
            p
            for pattern in ("Kilosort_*", "Kilosort4_*")
            for p in basepath.glob(pattern)
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )
    if kilo_folders:
        return kilo_folders[0]
    return None


def _detect_spike_sorting(basepath: Path, basename: str, n_channels: int) -> dict[str, Any] | None:
    relative_path = ""
    kilosort_dir = _first_kilosort_folder(basepath)
    if kilosort_dir is not None:
        relative_path = kilosort_dir.name

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


def _extract_kilosort_metadata(
    source_basepath: Path, n_channels: int
) -> tuple[dict[str, np.ndarray | str] | None, int | None]:
    kilosort_dir = _first_kilosort_folder(source_basepath)
    if kilosort_dir is None:
        return None, None
    rez_path = kilosort_dir / "rez.mat"
    if not rez_path.exists():
        return None, None

    try:
        import h5py  # type: ignore
    except Exception:
        return None, None

    try:
        with h5py.File(rez_path, "r") as f:
            rez_group = f.get("rez")
            if rez_group is None:
                return None, None

            xcoords = np.asarray(rez_group.get("xcoords")) if "xcoords" in rez_group else None
            ycoords = np.asarray(rez_group.get("ycoords")) if "ycoords" in rez_group else None
            kcoords_unique_count: int | None = None
            if "ops" in rez_group and "kcoords" in rez_group["ops"]:
                kcoords = np.asarray(rez_group["ops"]["kcoords"]).reshape(-1)
                if kcoords.size > 0:
                    kcoords_unique_count = int(np.unique(np.rint(kcoords).astype(np.int64)).size)

            if xcoords is None or ycoords is None:
                return None, kcoords_unique_count

            x_flat = xcoords.reshape(-1)
            y_flat = ycoords.reshape(-1)
            if x_flat.size < n_channels or y_flat.size < n_channels:
                return None, kcoords_unique_count

            chan_coords = {
                "x": _as_integer_column(x_flat[:n_channels]),
                "y": _as_integer_column(y_flat[:n_channels]),
                "source": "KiloSort",
            }
            return chan_coords, kcoords_unique_count
    except Exception:
        return None, None


def _channels_union(channels_1based: list[int], extra_1based: np.ndarray, chan_dtype: np.dtype) -> np.ndarray:
    merged = sorted(set(int(ch) for ch in channels_1based).union(int(ch) for ch in extra_1based.reshape(-1)))
    return np.asarray(merged, dtype=chan_dtype).reshape(1, -1)


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

    chan_dtype = _channel_dtype(n_channels)
    bad_channels = np.asarray(
        sorted(set(int(ch) for ch in bad_channels_1based)),
        dtype=chan_dtype,
    ).reshape(1, -1)

    notes = (
        f"Notes: {xml_meta.notes}   Description from xml: {xml_meta.description}"
        f"   Notes from xml: {xml_meta.notes}   Description: {xml_meta.description}"
    )

    chan_coords: dict[str, Any] = {"layout": "poly2", "verticalSpacing": 10}
    kilosort_setdiff_channels: np.ndarray | None = None
    chan_coords_from_kilosort, kilosort_group_count = _extract_kilosort_metadata(source_basepath, n_channels)
    if chan_coords_from_kilosort is not None:
        chan_coords = chan_coords_from_kilosort
    if kilosort_group_count is not None and kilosort_group_count < len(spike_groups_0based) and spike_groups.shape[1] > 0:
        kilosort_setdiff_channels = np.asarray(spike_groups[0, -1], dtype=chan_dtype).reshape(1, -1)
        bad_channels = _channels_union(
            channels_1based=[int(ch) for ch in bad_channels.reshape(-1)],
            extra_1based=kilosort_setdiff_channels,
            chan_dtype=chan_dtype,
        )
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
            "chanCoords": chan_coords,
            "probeDepths": 0,
            "precision": str(dat_dtype),
            "leastSignificantBit": 0.195,
        },
        "channelTags": {"Bad": {"channels": bad_channels}},
        "epochs": _build_epochs(merge_data, basename),
    }
    if kilosort_setdiff_channels is not None:
        session["channelTags"]["KiloSort_setdiff"] = {"channels": kilosort_setdiff_channels}

    spike_sorting = _detect_spike_sorting(source_basepath, basename, n_channels)
    if spike_sorting is not None:
        session["spikeSorting"] = np.asarray([spike_sorting], dtype=object)

    return session


def save_session_mat(path: Path, session_struct: dict[str, Any]) -> Path:
    savemat(path, {"session": session_struct}, do_compression=True)
    return path
