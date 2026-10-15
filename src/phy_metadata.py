"""Read declarative Phy params and write channel metadata without executing code."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def phy_params_path(folder_or_params: Path) -> Path:
    path = Path(folder_or_params)
    if path.name == "params.py":
        return path
    direct = path / "params.py"
    return direct if direct.exists() else path / "sorter_output" / "params.py"


def read_phy_params(folder_or_params: Path) -> dict[str, Any]:
    path = phy_params_path(folder_or_params)
    if not path.exists():
        return {}
    values: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path)).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if target.id in {"dat_path", "n_channels_dat", "sample_rate", "dtype", "offset"}:
                    raise ValueError(f"{path}: {target.id} must be a literal value") from None
    return values


def resolve_phy_dat_path(folder: Path) -> Path | None:
    value = read_phy_params(folder).get("dat_path")
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if value is None or value == "None" or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("Phy dat_path must name one binary file")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = phy_params_path(folder).parent / path
    return path.resolve()


def write_phy_shank_metadata(
    folder: Path, *, source_channel_ids: Any, shank_ids: Any,
    probe_ids: Any, exported_channel_ids: Any,
    separate_overlapping_positions: bool = False,
) -> None:
    """Write zero-based probe/shank labels in the exported channel-map order.

    Shanks are numbered by sorted (probe, shank) pairs so repeated local shank
    IDs cannot mix channels from different probes in Phy's waveform selection.
    Channel maps and sorting are unchanged. With the optional display correction,
    overlapping probe/shank groups are translated horizontally for Phy while
    their original coordinates are retained in channel_positions_local.npy.
    """
    import numpy as np

    def integer_vector(values: Any, name: str) -> np.ndarray:
        values = np.asarray(values).reshape(-1).astype(np.float64)
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"{name} must contain finite integer IDs")
        if np.any(values < 0) or np.any(values >= 2**31):
            raise ValueError(f"{name} must contain nonnegative int32 IDs")
        return values.astype(np.int32)

    channels = integer_vector(source_channel_ids, "source channels")
    shanks = integer_vector(shank_ids, "shanks")
    probes = integer_vector(probe_ids, "probes")
    exported = integer_vector(exported_channel_ids, "exported channels")
    if not (len(channels) == len(shanks) == len(probes)):
        raise ValueError("Probe/shank metadata length must match source channels")
    if len(set(channels.tolist())) != len(channels):
        raise ValueError("Source channel IDs must be unique")
    rows = {int(channel): i for i, channel in enumerate(channels)}
    if any(int(channel) not in rows for channel in exported):
        raise ValueError("Probe/shank metadata does not cover exported channels")
    indices = [rows[int(channel)] for channel in exported]
    pairs = list(zip(probes.tolist(), shanks.tolist()))
    pair_labels = {pair: i for i, pair in enumerate(sorted(set(pairs)))}
    probe_labels = {probe: i for i, probe in enumerate(sorted(set(probes.tolist())))}
    # Validate and align both arrays before publishing either file.
    exported_shanks = np.asarray([pair_labels[pairs[i]] for i in indices], dtype=np.int32)
    exported_probes = np.asarray([probe_labels[int(probes[i])] for i in indices], dtype=np.int32)
    folder = Path(folder)
    local_positions = display_positions = None
    offsets = {}
    positions_path = folder / "channel_positions.npy"
    if separate_overlapping_positions and positions_path.exists():
        positions = np.load(positions_path)
        if positions.shape != (len(exported), 2) or not np.all(np.isfinite(positions)):
            raise ValueError("Phy positions must be finite and match exported channels")
        if len(np.unique(positions, axis=0)) != len(positions):
            groups = np.unique(exported_shanks)
            if any(len(np.unique(positions[exported_shanks == group], axis=0))
                   != np.count_nonzero(exported_shanks == group) for group in groups):
                raise ValueError("Duplicate Phy positions within one probe/shank")
            local_positions = positions
            display_positions = positions.astype(np.float64, copy=True)
            # The 100 um gap is a display convention, not physical probe spacing.
            step = float(np.ptp(positions[:, 0])) + 100.0
            for rank, group in enumerate(groups):
                offset = rank * step
                display_positions[exported_shanks == group, 0] += offset
                offsets[str(int(group))] = offset
            local_path = folder / "channel_positions_local.npy"
            if local_path.exists() and not np.array_equal(np.load(local_path), local_positions):
                raise ValueError("Saved local Phy coordinates disagree with this export")
    np.save(folder / "channel_shanks.npy", exported_shanks)
    np.save(folder / "channel_probe.npy", exported_probes)
    if local_positions is not None:
        import json

        np.save(folder / "channel_positions_local.npy", local_positions)
        (folder / "channel_positions_display.json").write_text(json.dumps({
            "format": "PreprocessPipeline Phy display geometry", "version": 1,
            "units": "um", "source_positions": "channel_positions_local.npy",
            "display_positions": "channel_positions.npy", "grouping": "channel_shanks.npy",
            "x_offsets_um": offsets,
            "purpose": "Separate overlapping groups for Phy; offsets are not physical probe spacing",
        }, indent=2) + "\n", encoding="utf-8")
        np.save(positions_path, display_positions)
