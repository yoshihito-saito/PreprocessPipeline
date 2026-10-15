"""Sparse cross-day evidence without forced global neuron identities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from .session_models import LocalModels


def _read_model(path):
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"Incomplete shard in link graph: {path}")
        group = handle["model_after"]
        return LocalModels(group["waveforms"][:], group["channels"][:],
                           group["anchors"][:], group["assigned"][:],
                           int(group.attrs["version"]))


def _local_cosine(left, right, left_channels, right_channels, *, shifts=range(-2, 3)):
    common, li, ri = np.intersect1d(
        left_channels[left_channels >= 0], right_channels[right_channels >= 0],
        return_indices=True)
    if not len(common):
        return -1.
    a = left[:, li]
    b = right[:, ri]
    total_a = float(np.sum(left[:, left_channels >= 0] ** 2))
    total_b = float(np.sum(right[:, right_channels >= 0] ** 2))
    if (total_a <= 0 or total_b <= 0 or
            np.sum(a * a) < .7 * total_a or
            np.sum(b * b) < .7 * total_b):
        return -1.
    best = -1.
    for shift in shifts:
        aa, bb = ((a[:len(a)-shift], b[shift:]) if shift >= 0
                  else (a[-shift:], b[:len(b)+shift]))
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        if denom > 0:
            best = max(best, float(np.sum(aa * bb) / denom))
    return best


def build_cross_day_links(session, output_root, *, radius_um=200.,
                          max_neighbors=256):
    """Persist bounded candidate edges; all cross-day outcomes are unresolved.

    There is no probability calibration in this experimental path. Identical
    unit indices on different days are not treated as the same neuron.
    """
    if radius_um <= 0 or max_neighbors < 1:
        raise ValueError("Invalid link search bounds")
    root = Path(output_root)
    edges = []
    for probe in session.probes:
        by_day = {}
        for segment in probe.segments:
            by_day.setdefault(segment.day_id, []).append(segment)
        days = list(by_day)
        folder = root / probe.probe_id
        shards = []
        for path in folder.glob("*.h5"):
            first, stop = map(int, path.stem.split("-"))
            shards.append((first, stop, path))
        shards.sort()
        for earlier, later in zip(days, days[1:]):
            left_stop = max(seg.stop_sample for seg in by_day[earlier])
            right_start = min(seg.start_sample for seg in by_day[later])
            left_paths = [path for first, stop, path in shards if stop == left_stop]
            right_paths = [path for first, stop, path in shards if first == right_start]
            if not left_paths or not right_paths:
                continue
            left = _read_model(left_paths[0])
            right = _read_model(right_paths[0])
            position = probe.geometry
            for left_id, channel in enumerate(left.anchors):
                if left.assigned[left_id] == 0:
                    continue
                eligible = np.flatnonzero(
                    (probe.shank[right.anchors] == probe.shank[channel]) &
                    (np.linalg.norm(position[right.anchors] -
                                    position[channel], axis=1) <= radius_um))
                if len(eligible) > max_neighbors:
                    raise OverflowError("Cross-day link partition exceeds neighbor cap")
                for right_id in eligible:
                    if right.assigned[right_id] == 0:
                        continue
                    score = _local_cosine(left.waveforms[left_id],
                                          right.waveforms[right_id],
                                          left.channels[left_id],
                                          right.channels[right_id])
                    if score <= 0:
                        continue
                    distance = float(np.linalg.norm(
                        position[channel] - position[right.anchors[right_id]]))
                    edges.append({
                        "probe_id": probe.probe_id,
                        "left_local_id": f"{probe.probe_id}/{earlier}/{left_id}",
                        "right_local_id": f"{probe.probe_id}/{later}/{int(right_id)}",
                        "template_cosine": score,
                        "anchor_distance_um": distance,
                        "left_spikes": int(left.assigned[left_id]),
                        "right_spikes": int(right.assigned[right_id]),
                        "confidence": None,
                        "status": "unresolved",
                    })
    payload = {
        "schema_version": 1,
        "meaning": "candidate cross-day evidence only; no global IDs or calibrated probabilities",
        "edges": edges,
    }
    path = root / "cross_day_links.json"
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    return len(edges)
