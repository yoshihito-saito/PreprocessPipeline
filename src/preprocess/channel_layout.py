"""Validated original-column channel metadata shared by analysis and sorting."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.io import loadmat


def validate_channel_ids(values, num_channels: int, *, name: str) -> list[int]:
    ids = np.asarray(values).reshape(-1)
    if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
        raise ValueError(f"{name} must contain finite integer channel IDs")
    if np.any(ids < 0) or np.any(ids >= num_channels):
        raise ValueError(f"{name} contains IDs outside binary column range [0, {num_channels - 1}]")
    return ids.astype(np.int64).tolist()


def load_channel_layout(path: Path, num_channels: int) -> dict:
    mat = loadmat(str(path), simplify_cells=True)
    if "chanMap0ind" in mat:
        ids = np.asarray(mat["chanMap0ind"]).reshape(-1)
    elif "chanMap" in mat:
        ids = np.asarray(mat["chanMap"]).reshape(-1) - 1
    else:
        raise ValueError(f"{path}: chanMap or chanMap0ind is required")
    if not np.all(np.isfinite(ids)) or not np.all(ids == np.floor(ids)):
        raise ValueError(f"{path}: channel IDs must be finite integers")
    ids = ids.astype(np.int64)
    if ids.size == 0 or len(np.unique(ids)) != ids.size:
        raise ValueError(f"{path}: channel IDs must be nonempty and unique")
    if np.any(ids < 0) or np.any(ids >= num_channels):
        raise ValueError(f"{path}: channel IDs outside binary column range [0, {num_channels - 1}]")
    if "chanMap" in mat and not np.array_equal(np.asarray(mat["chanMap"]).reshape(-1) - 1, ids):
        raise ValueError(f"{path}: chanMap and chanMap0ind disagree")
    for key in ("connected", "xcoords", "ycoords", "zcoords", "kcoords", "probe_ids", "sidecoords"):
        if key in mat and np.asarray(mat[key]).size != ids.size:
            raise ValueError(f"{path}: {key} length must match channel IDs")
    connected = np.asarray(mat.get("connected", np.ones(ids.size))).reshape(-1)
    if not np.all(np.isfinite(connected)):
        raise ValueError(f"{path}: connected must be finite")
    return dict(mat, chanMap=ids + 1, chanMap0ind=ids, connected=connected > 0)


def channel_map_geometry(data: dict) -> dict[str, list]:
    """Canonical physical metadata, independent of vector orientation and row order.

    Exclusions are deliberately separate from geometry. Optional fields retain
    their presence so dropping z, probe identity, or face identity is detected.
    """
    raw_ids = data.get("chanMap0ind")
    if raw_ids is None:
        raw_ids = np.asarray(data["chanMap"]) - 1
    ids = np.asarray(raw_ids).reshape(-1)
    if ids.size == 0 or not np.all(np.isfinite(ids) & (ids == np.floor(ids)) & (ids >= 0)):
        raise ValueError("chanMap channel IDs must be nonempty nonnegative integers")
    if len(np.unique(ids)) != ids.size:
        raise ValueError("chanMap channel IDs must be unique")
    if "chanMap" in data and not np.array_equal(np.asarray(data["chanMap"]).reshape(-1) - 1, ids):
        raise ValueError("chanMap and chanMap0ind disagree")
    order = np.argsort(ids)
    geometry = {"chanMap0ind": ids[order].astype(int).tolist()}
    recording_format = np.asarray(data.get("recording_format", "")).reshape(-1)
    inactive_np = None
    if recording_format.size == 1 and str(recording_format[0]) == "wildx_np":
        connected = np.asarray(data["connected"]).reshape(-1)
        if connected.size != ids.size or not np.all(np.isfinite(connected)):
            raise ValueError("chanMap connected must have one finite value per channel")
        inactive_np = connected <= 0
    for key in ("xcoords", "ycoords", "zcoords", "kcoords", "probe_ids", "sidecoords"):
        if key not in data:
            if key in ("xcoords", "ycoords", "kcoords"):
                raise ValueError(f"chanMap requires {key}")
            continue
        values = np.asarray(data[key]).reshape(-1)
        if values.size != ids.size:
            raise ValueError(f"chanMap {key} must have one finite value per channel")
        valid = np.isfinite(values)
        if inactive_np is not None and key in ("xcoords", "ycoords"):
            # Native NP maps retain binary columns with no assigned physical site.
            valid |= np.isnan(values) & inactive_np
        if not np.all(valid):
            raise ValueError(f"chanMap {key} must have one finite value per channel")
        if key in ("kcoords", "probe_ids") and not np.all((values == np.floor(values)) & (values >= 0)):
            raise ValueError(f"chanMap {key} must contain nonnegative integer IDs")
        # A stable null sentinel makes unassigned NP sites comparable and hashable.
        geometry[key] = [None if np.isnan(value) else value for value in values[order].tolist()]
    return geometry


def channel_map_geometry_sha256(data: dict) -> str:
    encoded = json.dumps(channel_map_geometry(data), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assert_channel_map_geometry(expected: dict, actual: dict, *, context: str) -> None:
    left, right = channel_map_geometry(expected), channel_map_geometry(actual)
    changed = [key for key in left.keys() | right.keys() if left.get(key) != right.get(key)]
    if changed:
        raise ValueError(
            f"{context}: channel-map geometry differs ({', '.join(sorted(changed))}). "
            "Previously processed data may use the wrong reference neighborhoods. "
            "Regenerate preprocessing in a new output folder or explicitly enable overwrite; "
            "replacing only chanMap.mat is insufficient."
        )


class NoActiveChannels(ValueError):
    """Valid metadata deliberately excludes every channel in an analysis target."""
