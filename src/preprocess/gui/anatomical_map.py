from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


class AnatomicalMapError(ValueError):
    """Raised when an anatomical map cannot be parsed or generated."""


@dataclass(frozen=True)
class AnatomicalChannel:
    channel: int
    x: float
    y: float


@dataclass(frozen=True)
class AnatomicalChannelGroup:
    group_id: int
    channels: tuple[AnatomicalChannel, ...]


def channel_groups_from_chanmap_data(
    data: dict[str, Any],
    *,
    anatomical_groups_0based: list[list[int]] | None = None,
) -> list[AnatomicalChannelGroup]:
    """Use XML electrode order for CSV cells, and chanMap for coordinates only.

    The legacy one-argument form retains kcoords grouping for callers without
    XML. The GUI always supplies XML groups, including disconnected channels.
    """
    try:
        x = np.asarray(data["xcoords"], dtype=float).reshape(-1)
        y = np.asarray(data["ycoords"], dtype=float).reshape(-1)
        device_ch = np.asarray(
            data["chanMap0ind"] if "chanMap0ind" in data else np.asarray(data["chanMap"]) - 1,
            dtype=float,
        ).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise AnatomicalMapError(f"Invalid chanMap coordinates or channel IDs: {exc}") from exc
    if len(x) != len(y) or len(x) != len(device_ch):
        raise AnatomicalMapError("chanMap coordinates and channel IDs must have equal lengths.")
    if not np.all(np.isfinite(device_ch) & (device_ch >= 0) & (device_ch == np.floor(device_ch))):
        raise AnatomicalMapError("chanMap channel IDs must be nonnegative integers.")
    if len(np.unique(device_ch)) != len(device_ch):
        raise AnatomicalMapError("chanMap contains duplicate channel IDs.")
    if "chanMap" in data and not np.array_equal(np.asarray(data["chanMap"]).reshape(-1) - 1, device_ch):
        raise AnatomicalMapError("chanMap and chanMap0ind channel IDs disagree.")
    lookup = {
        int(channel): AnatomicalChannel(int(channel) + 1, float(x[index]), float(y[index]))
        for index, channel in enumerate(device_ch)
    }
    if anatomical_groups_0based is None:
        kcoords = np.asarray(data.get("kcoords", np.ones_like(x))).reshape(-1)
        if len(kcoords) != len(device_ch):
            raise AnatomicalMapError("chanMap kcoords and channel IDs must have equal lengths.")
        ordered_groups = [
            (int(group_id), [int(ch) for ch in device_ch[kcoords == group_id]])
            for group_id in sorted(set(kcoords.tolist()))
        ]
    else:
        ordered_groups = list(enumerate(anatomical_groups_0based, start=1))
    seen: set[int] = set()
    groups: list[AnatomicalChannelGroup] = []
    for group_id, ids in ordered_groups:
        channels = []
        for channel in ids:
            if channel in seen:
                raise AnatomicalMapError(f"Anatomical groups contain duplicate channel {channel} (0-based).")
            if channel not in lookup:
                raise AnatomicalMapError(f"XML channel {channel} (0-based) is missing from chanMap.")
            point = lookup[channel]
            if not np.isfinite(point.x) or not np.isfinite(point.y):
                raise AnatomicalMapError(f"Channel {channel} (0-based) has no finite coordinates.")
            channels.append(point)
            seen.add(channel)
        groups.append(AnatomicalChannelGroup(group_id=group_id, channels=tuple(channels)))
    return groups


def build_anatomical_map_rows(
    groups: list[AnatomicalChannelGroup],
    channel_regions: dict[int, str],
) -> list[list[str]]:
    known_channels = {channel.channel for group in groups for channel in group.channels}
    unknown = sorted(channel for channel, label in channel_regions.items() if label.strip() and channel not in known_channels)
    if unknown:
        raise AnatomicalMapError(f"Labels refer to channels outside the current XML groups: {unknown}")
    if not groups:
        return []
    max_rows = max((len(group.channels) for group in groups), default=0)
    rows: list[list[str]] = []
    for row_index in range(max_rows):
        row: list[str] = []
        for group in groups:
            if row_index < len(group.channels):
                label = channel_regions.get(group.channels[row_index].channel, "").strip()
                row.append(label)
            else:
                row.append("")
        rows.append(row)
    return rows


def build_anatomical_map_csv(
    groups: list[AnatomicalChannelGroup],
    channel_regions: dict[int, str],
) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in build_anatomical_map_rows(groups, channel_regions):
        writer.writerow(row)
    return buffer.getvalue()


def parse_anatomical_map_csv(
    text: str,
    groups: list[AnatomicalChannelGroup],
) -> dict[int, str]:
    try:
        rows = list(csv.reader(StringIO(text), strict=True))
    except csv.Error as exc:
        raise AnatomicalMapError(f"Invalid anatomical CSV: {exc}") from exc
    channel_regions: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        for group_index, value in enumerate(row):
            label = value.strip()
            if group_index >= len(groups):
                if label:
                    raise AnatomicalMapError(f"CSV row {row_index + 1}, column {group_index + 1} exceeds the XML group count.")
                continue
            group = groups[group_index]
            if row_index >= len(group.channels):
                if label:
                    raise AnatomicalMapError(f"CSV row {row_index + 1}, column {group_index + 1} exceeds that XML group's channel count.")
                continue
            if label:
                channel_regions[group.channels[row_index].channel] = label
    return channel_regions


def load_anatomical_map_csv(
    path: str | Path,
    groups: list[AnatomicalChannelGroup],
) -> dict[int, str]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise AnatomicalMapError(f"Anatomical map not found: {csv_path}")
    return parse_anatomical_map_csv(csv_path.read_text(encoding="utf-8"), groups)


def save_anatomical_map_csv(
    path: str | Path,
    groups: list[AnatomicalChannelGroup],
    channel_regions: dict[int, str],
) -> Path:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(build_anatomical_map_csv(groups, channel_regions), encoding="utf-8")
    return csv_path
