"""Bounded per-unit timing-search adaptation for the CUDA session matcher."""

from __future__ import annotations

import numpy as np


class AdaptiveShiftRadius:
    """Pilot one expanded search window, then specialize by unit.

    During a short pilot interval all templates get one extra sample of timing
    search. Units whose accepted fits use that boundary repeatedly retain the
    wider search; other units return to the configured base radius. State is
    small, deterministic, and checkpointable at shard boundaries.
    """

    VERSION = 1
    PILOT_SECONDS = 30.0
    MIN_OBSERVATIONS = 20
    BOUNDARY_FRACTION = 0.05

    def __init__(self, unit_count, sample_rate_hz, base_radius, *,
                 boundary_fraction=BOUNDARY_FRACTION):
        try:
            boundary_fraction = float(boundary_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid adaptive timing-search configuration") from exc
        if (unit_count < 0 or not np.isfinite(sample_rate_hz)
                or sample_rate_hz <= 0 or not isinstance(base_radius, int)
                or not 0 <= base_radius < 8
                or not np.isfinite(boundary_fraction)
                or not 0 < boundary_fraction <= 1):
            raise ValueError("Invalid adaptive timing-search configuration")
        self.sample_rate_hz = float(sample_rate_hz)
        self.base_radius = base_radius
        self.boundary_fraction = float(boundary_fraction)
        self.pilot_samples = max(1, round(self.sample_rate_hz * self.PILOT_SECONDS))
        self.elapsed_samples = 0
        self.observations = np.zeros(unit_count, np.uint32)
        self.boundary_fits = np.zeros(unit_count, np.uint32)
        self.radii = np.full(unit_count, base_radius, np.int32)
        self.fitted = False

    @property
    def expanded_radius(self):
        return self.base_radius + 1

    def for_core(self, unit_count=None):
        if unit_count is not None:
            self.extend_units(unit_count)
        if not self.fitted:
            return np.full(len(self.radii), self.expanded_radius, np.int32)
        return self.radii.copy()

    def extend_units(self, unit_count):
        old = len(self.radii)
        if unit_count < old:
            raise ValueError("Adaptive timing state cannot remove units")
        if unit_count == old:
            return
        extra = unit_count - old
        self.observations = np.pad(self.observations, (0, extra))
        self.boundary_fits = np.pad(self.boundary_fits, (0, extra))
        self.radii = np.pad(self.radii, (0, extra),
                            constant_values=self.base_radius)

    def observe(self, matches, core_samples, *, interval_good=True):
        if not isinstance(core_samples, int) or core_samples < 0:
            raise ValueError("Core sample count must be a nonnegative integer")
        if self.fitted:
            return False
        if interval_good:
            boundary = self.expanded_radius
            for match in matches:
                unit = int(match.unit)
                if not 0 <= unit < len(self.radii):
                    continue
                candidate = match.candidate_sample
                if candidate is None:
                    continue
                shift = int(match.source_sample) - int(candidate)
                self.observations[unit] += 1
                self.boundary_fits[unit] += abs(shift) >= boundary
        self.elapsed_samples = min(self.pilot_samples,
                                   self.elapsed_samples + core_samples)
        if self.elapsed_samples < self.pilot_samples:
            return False
        enough = self.observations >= self.MIN_OBSERVATIONS
        frequent = (self.boundary_fits /
                    np.maximum(self.observations, 1)) >= self.boundary_fraction
        self.radii[:] = self.base_radius
        self.radii[enough & frequent] = self.expanded_radius
        self.fitted = True
        return True

    def save(self, group):
        group.attrs.update(
            version=self.VERSION,
            sample_rate_hz=self.sample_rate_hz,
            base_radius=self.base_radius,
            pilot_samples=self.pilot_samples,
            elapsed_samples=self.elapsed_samples,
            fitted=self.fitted,
            min_observations=self.MIN_OBSERVATIONS,
            boundary_fraction=self.boundary_fraction,
        )
        group.create_dataset("observations", data=self.observations)
        group.create_dataset("boundary_fits", data=self.boundary_fits)
        group.create_dataset("radii", data=self.radii)

    def restore(self, group, unit_count):
        attrs = group.attrs
        if (int(attrs["version"]) != self.VERSION
                or float(attrs["sample_rate_hz"]) != self.sample_rate_hz
                or int(attrs["base_radius"]) != self.base_radius
                or int(attrs["pilot_samples"]) != self.pilot_samples
                or int(attrs["min_observations"]) != self.MIN_OBSERVATIONS
                or float(attrs["boundary_fraction"]) != self.boundary_fraction):
            raise ValueError("Adaptive timing checkpoint configuration mismatch")
        observations = group["observations"][:]
        boundary_fits = group["boundary_fits"][:]
        radii = group["radii"][:]
        if (observations.shape != (unit_count,)
                or boundary_fits.shape != (unit_count,)
                or radii.shape != (unit_count,)
                or np.any(boundary_fits > observations)
                or np.any((radii < self.base_radius) |
                          (radii > self.expanded_radius))):
            raise ValueError("Corrupt adaptive timing checkpoint")
        elapsed = int(attrs["elapsed_samples"])
        if not 0 <= elapsed <= self.pilot_samples:
            raise ValueError("Invalid adaptive timing progress")
        self.observations = observations.astype(np.uint32, copy=False)
        self.boundary_fits = boundary_fits.astype(np.uint32, copy=False)
        self.radii = radii.astype(np.int32, copy=False)
        self.elapsed_samples = elapsed
        self.fitted = bool(attrs["fitted"])
        if self.fitted != (elapsed >= self.pilot_samples):
            raise ValueError("Adaptive timing checkpoint completion mismatch")
