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


def channel_groups_from_chanmap_data(data: dict[str, Any]) -> list[AnatomicalChannelGroup]:
    x = np.asarray(data["xcoords"]).reshape(-1)
    y = np.asarray(data["ycoords"]).reshape(-1)
    kcoords = np.asarray(data.get("kcoords", np.ones_like(x))).reshape(-1)
    device_ch = np.asarray(
        data.get("chanMap0ind", np.asarray(data["chanMap"]).reshape(-1) - 1)
    ).reshape(-1).astype(int)

    n = min(len(x), len(y), len(kcoords), len(device_ch))
    groups: list[AnatomicalChannelGroup] = []
    for group_id in sorted(set(int(v) for v in kcoords[:n].tolist())):
        channels: list[AnatomicalChannel] = []
        for idx in range(n):
            if int(kcoords[idx]) != group_id:
                continue
            channels.append(
                AnatomicalChannel(
                    channel=int(device_ch[idx]) + 1,
                    x=float(x[idx]),
                    y=float(y[idx]),
                )
            )
        if channels:
            groups.append(AnatomicalChannelGroup(group_id=group_id, channels=tuple(channels)))
    return groups


def build_anatomical_map_rows(
    groups: list[AnatomicalChannelGroup],
    channel_regions: dict[int, str],
) -> list[list[str]]:
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
    reader = csv.reader(StringIO(text))
    rows = list(reader)
    channel_regions: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        for group_index, value in enumerate(row):
            if group_index >= len(groups):
                continue
            group = groups[group_index]
            if row_index >= len(group.channels):
                continue
            label = value.strip()
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
