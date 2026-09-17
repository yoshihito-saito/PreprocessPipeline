"""Validated original-column channel metadata shared by analysis and sorting."""
from __future__ import annotations

from pathlib import Path
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
    for key in ("connected", "xcoords", "ycoords", "kcoords", "probe_ids"):
        if key in mat and np.asarray(mat[key]).size != ids.size:
            raise ValueError(f"{path}: {key} length must match channel IDs")
    connected = np.asarray(mat.get("connected", np.ones(ids.size))).reshape(-1)
    if not np.all(np.isfinite(connected)):
        raise ValueError(f"{path}: connected must be finite")
    return dict(mat, chanMap=ids + 1, chanMap0ind=ids, connected=connected > 0)


class NoActiveChannels(ValueError):
    """Valid metadata deliberately excludes every channel in an analysis target."""
