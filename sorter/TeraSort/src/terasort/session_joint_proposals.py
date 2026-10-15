"""Ground-truth-free, bounded close-overlap proposals for offline evaluation.

This module proposes fits only. It does not accept spikes or change the
streaming sorter. Inputs and outputs use int64 source-sample coordinates;
callers translate one core to local int32 offsets for the CUDA scorer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointProposalBatch:
    pairs: np.ndarray  # rows: candidate time, target unit, blocker time, unit
    candidate_indices: np.ndarray  # indices in the original candidate array


def _triplet(value, label, index_type):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{label} must be three separate arrays")
    sample, index, strength = (np.asarray(part) for part in value)
    if (sample.ndim != 1 or index.shape != sample.shape
            or strength.shape != sample.shape
            or not np.issubdtype(sample.dtype, np.integer)
            or not np.issubdtype(index.dtype, np.integer)
            or not np.issubdtype(strength.dtype, np.number)
            or np.any(sample < 0) or np.any(index < 0)
            or not np.isfinite(strength).all()):
        raise ValueError(f"Invalid {label} arrays")
    if np.issubdtype(sample.dtype, np.unsignedinteger):
        if np.any(sample > np.iinfo(np.int64).max):
            raise ValueError(f"{label} source sample exceeds int64")
    if np.any(index >= index_type):
        raise ValueError(f"{label} index out of range")
    return sample.astype(np.int64, copy=False), index.astype(np.int32, copy=False), strength


def iter_joint_pair_batches(candidates, accepted, channels, anchors,
                            channel_positions_um, *,
                            near_samples=15, same_event_samples=2,
                            max_blockers=2, max_local_templates=8,
                            pair_batch=65_536):
    """Yield local spatial proposals in fixed-size batches.

    ``candidates`` is (source samples, physical channels, SNRs), and
    ``accepted`` is (source samples, unit indices, residual gains). Each is
    three separate arrays, keeping 64-bit sample coordinates exact. One exact
    sample/contact duplicate is retained at highest SNR; nearby distinct
    candidates remain available for recall. Candidate rows within
    ``same_event_samples`` of a spatially overlapping accepted spike are
    excluded because a same-time fit can falsely split a single event. Other
    probe sites can fire simultaneously. Caller must retain those
    excluded candidates as unknowns for later identity analysis.
    """
    contacts = np.asarray(channels)
    anchor_rows = np.asarray(anchors)
    positions = np.asarray(channel_positions_um)
    if (contacts.ndim != 2 or not np.issubdtype(contacts.dtype, np.integer)
            or anchor_rows.shape != (len(contacts),)
            or not np.issubdtype(anchor_rows.dtype, np.integer)
            or positions.ndim != 2 or positions.shape[1] not in (1, 2, 3)
            or not np.isfinite(positions).all()
            or np.any(contacts < -1) or np.any(contacts >= len(positions))
            or np.any(anchor_rows < 0) or np.any(anchor_rows >= len(positions))
            or np.any(contacts[:, 0] != anchor_rows)
            or not 3 <= near_samples <= 60
            or not 0 <= same_event_samples < near_samples
            or not 1 <= max_blockers <= 8
            or not 1 <= max_local_templates <= 32
            or not 1 <= pair_batch <= 2_000_000):
        raise ValueError("Invalid bounded joint-proposal inputs")
    candidate_time, candidate_channel, candidate_snr = _triplet(
        candidates, "candidate", len(positions))
    spike_time, spike_unit, spike_gain = _triplet(
        accepted, "accepted", len(contacts))

    # Spatial list is immutable for this model version. A proposal must use
    # the candidate's actual contact, never an inferred GT unit or anchor.
    contact_sets = [set(map(int, row[row >= 0])) for row in contacts]
    by_contact = []
    for channel in range(len(positions)):
        local = [unit for unit, support in enumerate(contact_sets)
                 if channel in support]
        local.sort(key=lambda unit: (
            float(np.sum((positions[anchor_rows[unit]] - positions[channel]) ** 2)),
            unit))
        by_contact.append(local[:max_local_templates])

    spike_order = np.lexsort((spike_unit, spike_time))
    spike_times = spike_time[spike_order]
    spike_units = spike_unit[spike_order]
    spike_gains = spike_gain[spike_order]
    # Highest-SNR exact duplicate wins; remaining rows are in source order.
    candidate_order = np.lexsort((np.arange(len(candidate_time)),
                                  -candidate_snr,
                                  candidate_channel, candidate_time))
    seen = set()
    rows = []
    indices = []
    for index in candidate_order:
        t = int(candidate_time[index])
        channel = int(candidate_channel[index])
        key = (t, channel)
        if key in seen:
            continue
        seen.add(key)
        left = np.searchsorted(spike_times, t - near_samples, side="left")
        right = np.searchsorted(spike_times, t + near_samples, side="right")
        nearby_times = spike_times[left:right]
        if any(channel in contact_sets[int(spike_units[j])]
               for j in range(left, right)
               if abs(int(spike_times[j]) - t) <= same_event_samples):
            continue
        targets = by_contact[channel]
        blocker_rows = [j for j in range(left, right) if
                        same_event_samples < abs(int(spike_times[j]) - t) <= near_samples
                        and any(unit != int(spike_units[j]) and
                                contact_sets[unit] & contact_sets[int(spike_units[j])]
                                for unit in targets)]
        blocker_rows.sort(key=lambda j: (abs(int(spike_times[j]) - t),
                                         -float(spike_gains[j]),
                                         int(spike_times[j]), int(spike_units[j])))
        for blocker in blocker_rows[:max_blockers]:
            blocker_time, blocker_unit = int(spike_times[blocker]), int(spike_units[blocker])
            for unit in targets:
                if unit == blocker_unit or not (
                        contact_sets[unit] & contact_sets[blocker_unit]):
                    continue
                rows.append((t, unit, blocker_time, blocker_unit))
                indices.append(int(index))
                if len(rows) == pair_batch:
                    yield JointProposalBatch(np.asarray(rows, np.int64),
                                             np.asarray(indices, np.int64))
                    rows, indices = [], []
    if rows:
        yield JointProposalBatch(np.asarray(rows, np.int64),
                                 np.asarray(indices, np.int64))
