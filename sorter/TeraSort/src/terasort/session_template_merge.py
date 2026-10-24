"""Spatially sparse grouping for duplicate templates from one calibration epoch."""

from __future__ import annotations

import numpy as np


def build_template_map(waveforms, channels, anchors, geometry, shanks, *,
                       cosine_threshold=0.93, radius_um=32.0,
                       mode="components"):
    """Return a compact template-to-unit map without changing fitted templates.

    Similarity is a signed zero-lag cosine over shared physical contacts. The
    denominator uses each template's full energy, equivalent to cosine
    similarity after placing each local waveform into a zero-filled dense
    recording-channel array. Only anchors on the same shank and within the
    spatial radius are compared. The work list is local-pair sparse rather than
    a dense all-template similarity matrix.
    """
    waveforms = np.asarray(waveforms, np.float32)
    channels = np.asarray(channels, np.int32)
    anchors = np.asarray(anchors, np.int32)
    geometry = np.asarray(geometry, np.float64)
    shanks = np.asarray(shanks)
    n_units = len(waveforms)
    if (waveforms.ndim != 3 or (waveforms.shape[1] < 21 or waveforms.shape[1] % 2 != 1)
            or channels.shape != (n_units, waveforms.shape[2])
            or anchors.shape != (n_units,)
            or geometry.ndim != 2 or geometry.shape[1] != 2
            or shanks.shape != (len(geometry),)
            or np.any(channels < -1) or np.any(channels >= len(geometry))
            or np.any(anchors < 0) or np.any(anchors >= len(geometry))
            or not np.isfinite(waveforms).all()
            or not np.isfinite(geometry).all()
            or not np.isfinite(cosine_threshold)
            or not 0 < cosine_threshold <= 1
            or not np.isfinite(radius_um) or radius_um <= 0):
        raise ValueError("Invalid model templates or template-link parameters")
    if mode not in ("components", "reciprocal"):
        raise ValueError("Template link mode must be components or reciprocal")

    norms = np.sqrt(np.sum(waveforms.astype(np.float64) ** 2,
                           axis=(1, 2), dtype=np.float64))
    units_by_anchor = {}
    channel_slots = []
    for unit in range(n_units):
        units_by_anchor.setdefault(int(anchors[unit]), []).append(unit)
        channel_slots.append({
            int(channel): slot
            for slot, channel in enumerate(channels[unit]) if channel >= 0
        })

    # Cache only physical-contact neighborhood lists, not U x U template data.
    neighbors_by_anchor = {}
    for anchor in np.unique(anchors):
        delta = geometry - geometry[int(anchor)]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        mask = ((shanks == shanks[int(anchor)]) &
                (distance2 <= radius_um * radius_um))
        neighbors_by_anchor[int(anchor)] = np.flatnonzero(mask)

    parent = np.arange(n_units, dtype=np.int32)
    best_score = np.full(n_units, -np.inf, np.float64)
    best_partner = np.full(n_units, -1, np.int32)

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    accepted = compared = 0
    for left in range(n_units):
        if norms[left] <= 1e-12:
            continue
        left_slots = channel_slots[left]
        for anchor in neighbors_by_anchor[int(anchors[left])]:
            for right in units_by_anchor.get(int(anchor), ()):
                if right <= left or norms[right] <= 1e-12:
                    continue
                right_slots = channel_slots[right]
                common = sorted(left_slots.keys() & right_slots.keys())
                if not common:
                    continue
                left_idx = np.fromiter((left_slots[c] for c in common), np.int32)
                right_idx = np.fromiter((right_slots[c] for c in common), np.int32)
                dot = np.sum(
                    waveforms[left, :, left_idx].astype(np.float64) *
                    waveforms[right, :, right_idx].astype(np.float64),
                    dtype=np.float64,
                )
                compared += 1
                similarity = dot / (norms[left] * norms[right])
                if mode == "reciprocal":
                    for unit, partner in ((left, right), (right, left)):
                        if (similarity > best_score[unit] or
                                (similarity == best_score[unit] and
                                 partner < best_partner[unit])):
                            best_score[unit] = similarity
                            best_partner[unit] = partner
                if mode != "components" or similarity < cosine_threshold:
                    continue
                a, b = find(left), find(right)
                if a != b:
                    parent[b] = a
                    accepted += 1

    if mode == "reciprocal":
        for left, right in enumerate(best_partner):
            right = int(right)
            if (right <= left or best_partner[right] != left
                    or best_score[left] < cosine_threshold
                    or best_score[right] < cosine_threshold):
                continue
            parent[right] = left
            accepted += 1

    roots = np.asarray([find(unit) for unit in range(n_units)], np.int32)
    unique_roots = np.unique(roots)
    compact = {int(root): i for i, root in enumerate(unique_roots)}
    mapping = np.asarray([compact[int(root)] for root in roots], np.int64)
    return mapping, {
        "input_templates": int(n_units),
        "output_groups": int(len(unique_roots)),
        "candidate_pairs_compared": int(compared),
        "links_accepted": int(accepted),
        "cosine_threshold": float(cosine_threshold),
        "anchor_radius_um": float(radius_um),
        "link_mode": mode,
    }
