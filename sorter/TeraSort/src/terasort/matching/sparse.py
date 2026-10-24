"""Spatially sparse, bounded-batch matching of per-unit waveform templates.

This module produces candidate pairs and one template-cosine metric. It does not
assign identities, implement UnitMatch probabilities, or infer cross-day drift.
"""

import numpy as np
from scipy.spatial import cKDTree


def template_centers(templates, positions, *, batch_size=128):
    """Compute energy-weighted physical centroids without loading all templates."""
    positions = np.asarray(positions, np.float32)
    if (templates.ndim != 3 or positions.ndim != 2
            or templates.shape[2] != len(positions) or batch_size <= 0
            or not np.isfinite(positions).all()):
        raise ValueError("Invalid template/channel geometry or batch size")
    centers = np.empty((len(templates), positions.shape[1]), np.float32)
    for start in range(0, len(templates), batch_size):
        stop = min(start + batch_size, len(templates))
        waveform = np.asarray(templates[start:stop], np.float32)
        if not np.isfinite(waveform).all():
            raise ValueError("Template values must be finite")
        energy = np.square(waveform).sum(axis=1)
        centers[start:stop] = (energy @ positions) / np.maximum(
            energy.sum(axis=1, keepdims=True), 1e-12)
    return centers


def iter_spatial_pairs(left, right, radius_um, *, left_probe=None,
                       right_probe=None, pair_batch=4096,
                       max_neighbors_per_unit=256):
    """Yield same-probe candidate pairs in deterministic bounded batches.

    Raises on dense neighborhoods instead of silently dropping plausible pairs.
    Callers can then split the geometry or increase the explicit neighbor cap.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if (left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]
            or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.isfinite(radius_um) or radius_um < 0
            or pair_batch <= 0 or max_neighbors_per_unit <= 0):
        raise ValueError("Finite coordinates, radius and positive limits required")
    lp = np.zeros(len(left), np.int64) if left_probe is None else np.asarray(left_probe)
    rp = np.zeros(len(right), np.int64) if right_probe is None else np.asarray(right_probe)
    if lp.shape != (len(left),) or rp.shape != (len(right),):
        raise ValueError("Probe labels must match unit counts")
    for probe in np.intersect1d(lp, rp):
        left_ids = np.flatnonzero(lp == probe)
        right_ids = np.flatnonzero(rp == probe)
        tree = cKDTree(right[right_ids])
        pending = []
        for left_id in left_ids:
            neighbors = tree.query_ball_point(left[left_id], radius_um)
            if len(neighbors) > max_neighbors_per_unit:
                raise OverflowError(f"Unit {left_id} has {len(neighbors)} spatial candidates; "
                                    f"limit is {max_neighbors_per_unit}")
            for right_id in np.sort(right_ids[neighbors]):
                pending.append((int(left_id), int(right_id)))
                if len(pending) == pair_batch:
                    yield np.asarray(pending, np.int64)
                    pending = []
        if pending:
            yield np.asarray(pending, np.int64)


def score_template_pairs(left, right, pairs, *, shifts=(-2, -1, 0, 1, 2),
                         edge=2, batch_size=64):
    """Five-shift cosine for supplied pairs, with a fixed waveform batch cap."""
    pairs = np.asarray(pairs, np.int64)
    if (left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]
            or pairs.ndim != 2 or pairs.shape[1] != 2 or batch_size <= 0
            or edge < 0 or left.shape[1] <= 2 * edge or not shifts
            or any(abs(shift) > edge for shift in shifts)
            or (len(pairs) and (np.any(pairs < 0)
                               or np.any(pairs[:, 0] >= len(left))
                               or np.any(pairs[:, 1] >= len(right))))):
        raise ValueError("Invalid templates, pairs, shifts or batch size")
    scores = np.empty(len(pairs), np.float32)
    stop = left.shape[1] - edge
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        rows, cols = pairs[start:end, 0], pairs[start:end, 1]
        a = np.asarray(left[rows, edge:stop, :], np.float32)
        a_norm = np.sqrt(np.einsum("ijk,ijk->i", a, a)).clip(1e-12)
        best = np.full(len(rows), -1., np.float32)
        for shift in shifts:
            b = np.asarray(right[cols, edge + shift:stop + shift, :], np.float32)
            b_norm = np.sqrt(np.einsum("ijk,ijk->i", b, b)).clip(1e-12)
            dot = np.einsum("ijk,ijk->i", a, b)
            best = np.maximum(best, dot / (a_norm * b_norm))
        scores[start:end] = best
    return scores
