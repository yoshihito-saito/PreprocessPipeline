"""Causal, bounded rolling waveform state for an online template matcher.

Only post-assignment, high-confidence, motion-aligned *summaries* enter this
state. A caller must assign an interval with a frozen snapshot and commit it
after output is durable; this class does not detect or assign spikes.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TemplateSnapshot:
    waveform: np.ndarray
    anchor: np.ndarray
    version: int
    through_sample: int
    recent_spike_count: int
    active: bool
    coordinate_frame: str


class RollingTemplate:
    """One unit's bin-aligned rolling template; memory is O(window/bin × shape).

    The current template is a mean of recent bin means, weighted by each bin's
    spike count capped at ``max_effective_spikes_per_bin``. The cap prevents a
    brief high-rate period from dominating the time window. A 30-minute window
    with 5-minute bins retains at most six bin summaries. On dormancy the last
    waveform is frozen, never decayed toward zero.
    """

    def __init__(self, anchor, *, sample_rate_hz, coordinate_frame,
                 start_sample=0, window_seconds=1800, bin_seconds=300,
                 min_confidence=.95, min_spikes_per_summary=1,
                 max_effective_spikes_per_bin=100):
        anchor = np.asarray(anchor, np.float32)
        if (anchor.ndim != 2 or not anchor.size or not np.isfinite(anchor).all()
                or not isinstance(coordinate_frame, str) or not coordinate_frame
                or not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0
                or not math.isfinite(window_seconds) or window_seconds <= 0
                or not math.isfinite(bin_seconds) or bin_seconds <= 0
                or bin_seconds > window_seconds
                or not 0 <= min_confidence <= 1
                or min_spikes_per_summary < 1
                or max_effective_spikes_per_bin < 1
                or start_sample < 0):
            raise ValueError("Invalid rolling template configuration")
        self.sample_rate_hz = float(sample_rate_hz)
        self.bin_samples = round(bin_seconds * sample_rate_hz)
        self.window_bins = math.ceil(window_seconds / bin_seconds)
        if self.bin_samples < 1:
            raise ValueError("Bin duration must span at least one sample")
        self.coordinate_frame = coordinate_frame
        self.min_confidence = float(min_confidence)
        self.min_spikes_per_summary = int(min_spikes_per_summary)
        self.max_effective_spikes_per_bin = int(max_effective_spikes_per_bin)
        self.anchor = anchor.copy()
        self._last = anchor.copy()
        self._bins = {}  # bin ID -> [sum of aligned waveforms (float64), count]
        self.version = 0
        self.through_sample = int(start_sample)

    def snapshot(self):
        """Return copies that remain frozen while a chunk is assigned."""
        recent = sum(count for _, count in self._bins.values())
        return TemplateSnapshot(self._last.copy(), self.anchor.copy(),
                                self.version, self.through_sample, recent,
                                bool(recent), self.coordinate_frame)

    def commit(self, start_sample, stop_sample, *, mean_waveform=None,
               spike_count=0, confidence=0., coordinate_frame,
               expected_version):
        """Advance one committed interval; return whether its summary was used.

        A nonempty summary must be fully inside one time bin. Split a longer
        interval upstream so the rolling window retains its time meaning.
        A rejected or empty summary still advances the watermark and expires
        old bins. No operation can silently rewind or update a stale snapshot.
        """
        if (expected_version != self.version or coordinate_frame != self.coordinate_frame
                or not isinstance(start_sample, (int, np.integer))
                or not isinstance(stop_sample, (int, np.integer))
                or start_sample < self.through_sample or stop_sample <= start_sample
                or not isinstance(spike_count, (int, np.integer)) or spike_count < 0
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Stale, overlapping or incompatible template update")
        accepted = bool(spike_count >= self.min_spikes_per_summary
                        and confidence >= self.min_confidence)
        if accepted:
            if start_sample // self.bin_samples != (stop_sample - 1) // self.bin_samples:
                raise ValueError("Summary crosses a rolling time-bin boundary")
            mean = np.asarray(mean_waveform, np.float32)
            if mean.shape != self.anchor.shape or not np.isfinite(mean).all():
                raise ValueError("Waveform shape or values changed")
            bin_id = start_sample // self.bin_samples
            if bin_id not in self._bins:
                self._bins[bin_id] = [np.zeros_like(self.anchor, np.float64), 0]
            self._bins[bin_id][0] += mean.astype(np.float64) * int(spike_count)
            self._bins[bin_id][1] += int(spike_count)
        current_bin = (stop_sample - 1) // self.bin_samples
        oldest = current_bin - self.window_bins + 1
        for bin_id in list(self._bins):
            if bin_id < oldest:
                del self._bins[bin_id]
        if self._bins:
            weighted_sum = np.zeros_like(self.anchor, np.float64)
            weight_total = 0
            for total, count in self._bins.values():
                weight = min(count, self.max_effective_spikes_per_bin)
                weighted_sum += (total / count) * weight
                weight_total += weight
            self._last = (weighted_sum / weight_total).astype(np.float32)
        self.through_sample = int(stop_sample)
        self.version += 1
        return accepted

    def state_dict(self):
        """Small, copy-safe state for an external atomic checkpoint writer."""
        ids = np.asarray(sorted(self._bins), np.int64)
        return {
            "anchor": self.anchor.copy(), "last": self._last.copy(),
            "bin_ids": ids,
            "bin_sums": np.stack([self._bins[i][0] for i in ids]) if len(ids)
                        else np.empty((0, *self.anchor.shape), np.float64),
            "bin_counts": np.asarray([self._bins[i][1] for i in ids], np.int64),
            "sample_rate_hz": self.sample_rate_hz,
            "bin_samples": self.bin_samples, "window_bins": self.window_bins,
            "coordinate_frame": self.coordinate_frame,
            "min_confidence": self.min_confidence,
            "min_spikes_per_summary": self.min_spikes_per_summary,
            "max_effective_spikes_per_bin": self.max_effective_spikes_per_bin,
            "version": self.version, "through_sample": self.through_sample,
        }

    @classmethod
    def from_state_dict(cls, state):
        """Restore a checkpoint produced by :meth:`state_dict`."""
        obj = cls(state["anchor"], sample_rate_hz=state["sample_rate_hz"],
                  coordinate_frame=state["coordinate_frame"],
                  start_sample=state["through_sample"],
                  window_seconds=state["window_bins"] * state["bin_samples"]
                                 / state["sample_rate_hz"],
                  bin_seconds=state["bin_samples"] / state["sample_rate_hz"],
                  min_confidence=state["min_confidence"],
                  min_spikes_per_summary=state["min_spikes_per_summary"],
                  max_effective_spikes_per_bin=state["max_effective_spikes_per_bin"])
        ids = np.asarray(state["bin_ids"], np.int64)
        sums = np.asarray(state["bin_sums"], np.float64)
        counts = np.asarray(state["bin_counts"], np.int64)
        if (sums.shape != (len(ids), *obj.anchor.shape) or counts.shape != ids.shape
                or np.any(counts <= 0) or not np.isfinite(sums).all()
                or not np.array_equal(ids, np.unique(ids))):
            raise ValueError("Invalid rolling template checkpoint")
        obj._bins = {int(i): [summation.copy(), int(count)]
                     for i, summation, count in zip(ids, sums, counts)}
        obj._last = np.asarray(state["last"], np.float32).copy()
        if obj._last.shape != obj.anchor.shape or not np.isfinite(obj._last).all():
            raise ValueError("Invalid last template in checkpoint")
        obj.version = int(state["version"])
        return obj
