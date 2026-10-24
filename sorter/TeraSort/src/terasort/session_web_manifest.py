"""Build a bounded-sort source clock from a validated Neuroscope web request."""

from __future__ import annotations

import json
import math
from pathlib import Path


def build_xml_session_manifest(request: dict, destination: Path, *, session_id: str):
    """Pin source order, geometry and gain without copying or reading raw data."""
    metadata = request.get("xml_import")
    if not isinstance(metadata, dict):
        raise ValueError("Bounded session sorting requires Neuroscope XML")
    settings = metadata.get("settings", {})
    count = settings.get("n_chan_bin")
    rate = settings.get("fs")
    gain = settings.get("scale")
    if (type(count) is not int or count < 1 or
            not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0 or
            not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 0):
        raise ValueError("Bounded session sorting needs XML channel count, sample rate and explicit gain")
    groups = metadata.get("anatomical_groups") or metadata.get("channel_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("XML channel groups are required for bounded sorting")
    x, y, shank = [0.] * count, [0.] * count, [0] * count
    placed = set()
    for group_id, group in enumerate(groups):
        for slot, channel in enumerate(group):
            if type(channel) is not int or not 0 <= channel < count or channel in placed:
                raise ValueError("XML geometry has duplicate or invalid channels")
            placed.add(channel)
            x[channel] = float(200 * (group_id + 1) + (-20 if slot % 2 == 0 else 20))
            y[channel] = float(-20 * (slot + 1))
            shank[channel] = group_id
    bad = set(metadata.get("bad_channels", [])) | (set(range(count)) - placed)
    if len(bad) == count:
        raise ValueError("XML excludes every channel")
    for channel in set(range(count)) - placed:
        x[channel] = float(1_000_000 + channel)
    files = request.get("filenames", [request.get("filename")])
    if not isinstance(files, list) or not files:
        raise ValueError("Bounded session needs at least one source file")
    segments = []
    first = 0
    for name in files:
        source = Path(name)
        size = source.stat().st_size
        if size < 2 * count or size % (2 * count):
            raise ValueError(f"Invalid INT16 source frame count: {source}")
        frames = size // (2 * count)
        segments.append({"path": str(source), "start_sample": first,
                         "n_samples": frames, "day_id": "day1"})
        first += frames
    payload = {"schema_version": 1, "session_id": session_id,
               "probes": [{"probe_id": "probeA", "sample_rate_hz": rate,
                           "gain_uv_per_count": gain,
                           "geometry": {"x_um": x, "y_um": y, "shank": shank},
                           "bad_channels": sorted(bad), "segments": segments,
                           "gaps": []}]}
    destination = Path(destination)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return destination
