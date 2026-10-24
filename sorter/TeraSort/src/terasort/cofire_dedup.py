"""Bounded, ground-truth-free screening of repeated template co-firing.

This is an experimental post-fit component. A pilot learns template pairs from
accepted assignments; the suppressor then marks one assignment from each
near-synchronous pair while preserving stable unit identities. Neither class
reads raw voltage or retains recording-length event arrays.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CofirePolicy:
    pair_winners: dict[tuple[int, int], int]
    pair_counts: dict[tuple[int, int], int]
    unit_counts: dict[int, int]


class CofirePairLearner:
    """Learn conservative duplicate pairs from chronological assignment chunks."""

    def __init__(self, geometry_um, *, tolerance_samples: int,
                 amplitude_min: float = .7, min_cofires: int = 20,
                 min_cofire_fraction: float = .95, max_distance_um: float = 250.,
                 max_channel_index_distance: int = 12,
                 max_active_events: int = 8192, max_pairs: int = 100_000):
        geometry = np.asarray(geometry_um, np.float32)
        if geometry.ndim != 2 or geometry.shape[1] != 2 or not np.isfinite(geometry).all():
            raise ValueError("geometry_um must be a finite (channels, 2) array")
        if (tolerance_samples < 0 or not np.isfinite(amplitude_min)
                or amplitude_min < 0 or min_cofires < 1
                or not 0 < min_cofire_fraction <= 1
                or not np.isfinite(max_distance_um) or max_distance_um < 0
                or max_channel_index_distance < 0
                or max_active_events < 1 or max_pairs < 1):
            raise ValueError("Invalid co-firing pilot parameters")
        self.geometry = geometry
        self.tolerance = int(tolerance_samples)
        self.amplitude_min = float(amplitude_min)
        self.min_cofires = int(min_cofires)
        self.min_fraction = float(min_cofire_fraction)
        self.max_distance_sq = float(max_distance_um) ** 2
        self.max_channel_index_distance = int(max_channel_index_distance)
        self.max_active_events = int(max_active_events)
        self.max_pairs = int(max_pairs)
        self.recent = deque()
        self.unit_counts = Counter()
        self.pair_counts = Counter()
        self.near_counts = Counter()
        self.last_time = None

    def observe(self, spikes) -> None:
        """Consume rows sorted by sample_index, without storing the chunk."""
        required = {"sample_index", "unit_id", "channel_index", "fitted_amplitude"}
        if not required.issubset(spikes.dtype.names or ()):
            raise ValueError("Assignments lack fields required for co-firing calibration")
        for row in spikes:
            time = int(row["sample_index"])
            if self.last_time is not None and time < self.last_time:
                raise ValueError("Pilot assignments must be chronological")
            self.last_time = time
            while self.recent and time - self.recent[0][0] > self.tolerance:
                self.recent.popleft()
            if float(row["fitted_amplitude"]) < self.amplitude_min:
                continue
            unit, channel = int(row["unit_id"]), int(row["channel_index"])
            if not 0 <= channel < len(self.geometry):
                raise ValueError("Assignment channel is outside probe geometry")
            self.unit_counts[unit] += 1
            for _, other_unit, other_channel in self.recent:
                if unit == other_unit:
                    continue
                pair = (min(unit, other_unit), max(unit, other_unit))
                if pair not in self.pair_counts and len(self.pair_counts) >= self.max_pairs:
                    raise MemoryError("Co-firing pilot exceeded its pair budget")
                self.pair_counts[pair] += 1
                displacement = self.geometry[channel] - self.geometry[other_channel]
                if (abs(channel - other_channel) <= self.max_channel_index_distance
                        and float(displacement @ displacement) <= self.max_distance_sq):
                    self.near_counts[pair] += 1
            if len(self.recent) >= self.max_active_events:
                raise MemoryError("Co-firing pilot exceeded its active-event budget")
            self.recent.append((time, unit, channel))

    def policy(self) -> CofirePolicy:
        winners = {}
        for pair, count in self.pair_counts.items():
            a, b = pair
            if (count < self.min_cofires
                    or count < self.min_fraction * min(self.unit_counts[a],
                                                        self.unit_counts[b])
                    or 2 * self.near_counts[pair] < count):
                continue
            winners[pair] = a if self.unit_counts[a] >= self.unit_counts[b] else b
        return CofirePolicy(winners, dict(self.pair_counts), dict(self.unit_counts))


class CofireSuppressor:
    """Return refs to suppress; refs may refer to an earlier chunk or shard."""

    def __init__(self, pair_winners: dict[tuple[int, int], int], *,
                 tolerance_samples: int, max_active_events: int = 8192):
        if tolerance_samples < 0 or max_active_events < 1:
            raise ValueError("Invalid co-firing suppression parameters")
        self.pair_winners = dict(pair_winners)
        for pair, winner in self.pair_winners.items():
            if len(pair) != 2 or pair[0] >= pair[1] or winner not in pair:
                raise ValueError("Each pair must be ordered and name its winner")
        self.tolerance = int(tolerance_samples)
        self.max_active_events = int(max_active_events)
        self.recent = deque()
        self.last_time = None

    def observe(self, time: int, unit: int, ref) -> list:
        time, unit = int(time), int(unit)
        if self.last_time is not None and time < self.last_time:
            raise ValueError("Assignments must be chronological")
        self.last_time = time
        while self.recent and time - self.recent[0][0] > self.tolerance:
            self.recent.popleft()
        victims = []
        for _, other_unit, other_ref in self.recent:
            winner = self.pair_winners.get((min(unit, other_unit),
                                            max(unit, other_unit)))
            if winner is None:
                continue
            victims.append(ref if unit != winner else other_ref)
        if len(self.recent) >= self.max_active_events:
            raise MemoryError("Co-firing suppression exceeded its active-event budget")
        self.recent.append((time, unit, ref))
        return victims
