"""Bounded local models and a residual-based reference matcher.

This matcher is intentionally an experimental quality path. Its thresholds
must pass the separate ground-truth gate before replacing Kilosort4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidates.waveforms import geometry_channel_map


def validate_nt(nt):
    """Odd waveform length covering the 21-sample raw-anchor holdout."""
    if isinstance(nt, (bool, np.bool_)) or not isinstance(nt, (int, np.integer)) or nt < 21 or nt % 2 != 1:
        raise ValueError("Bounded nt must be an odd integer >= 21 samples")
    return int(nt)


@dataclass
class LocalModels:
    waveforms: np.ndarray     # unit x nt time x local contact
    channels: np.ndarray      # unit x local contact, -1 padded
    anchors: np.ndarray       # unit x physical recording channel
    assigned: np.ndarray      # unit x cumulative count
    version: int = 0

    def __post_init__(self):
        self.waveforms = np.asarray(self.waveforms, np.float32)
        self.channels = np.asarray(self.channels, np.int32)
        self.anchors = np.asarray(self.anchors, np.int32)
        self.assigned = np.asarray(self.assigned, np.int64)
        if (self.waveforms.ndim != 3 or self.waveforms.shape[1] < 21 or self.waveforms.shape[1] % 2 != 1
                or self.channels.shape != (len(self.waveforms), self.waveforms.shape[2])
                or self.anchors.shape != (len(self.waveforms),)
                or self.assigned.shape != (len(self.waveforms),)
                or not np.isfinite(self.waveforms).all()
                or np.any(self.channels < -1)):
            raise ValueError("Invalid local template model")
        if np.any(self.channels[:, 0] != self.anchors):
            raise ValueError("Template anchors must occupy local slot zero")
        if self.version < 0:
            raise ValueError("Negative model version")
        self.by_channel = {}
        for unit, contacts in enumerate(self.channels):
            for channel in contacts:
                if channel >= 0:
                    self.by_channel.setdefault(int(channel), []).append(unit)

    @property
    def nt(self):
        return self.waveforms.shape[1]

    @classmethod
    def from_dense(cls, dense, probe, *, radius_um=75., width=16):
        dense = np.asarray(dense, np.float32)
        if (dense.ndim != 3 or (dense.shape[1] < 21 or dense.shape[1] % 2 != 1 or dense.shape[2] != probe.n_channels)
                or not np.isfinite(dense).all()):
            raise ValueError("Seeds must be unit x odd nt samples x recording channel")
        neighbor = geometry_channel_map(probe.geometry, probe.shank,
                                        radius_um, width)
        waveforms = np.zeros((len(dense), dense.shape[1], neighbor.shape[1]), np.float32)
        channels = np.empty((len(dense), neighbor.shape[1]), np.int32)
        anchors = np.empty(len(dense), np.int32)
        for unit, template in enumerate(dense):
            anchor = int(np.argmax(np.max(np.abs(template), axis=0)))
            if not np.any(template):
                raise ValueError("Zero calibration template")
            contacts = neighbor[anchor]
            local = np.zeros_like(waveforms[unit])
            valid = contacts >= 0
            local[:, valid] = template[:, contacts[valid]]
            peak_time = int(np.argmax(np.abs(local[:, 0])))
            shift = dense.shape[1] // 2 - peak_time
            if shift > 0:
                local[shift:] = local[:-shift].copy()
                local[:shift] = 0
            elif shift < 0:
                local[:shift] = local[-shift:].copy()
                local[shift:] = 0
            waveforms[unit] = local
            channels[unit] = contacts
            anchors[unit] = anchor
        return cls(waveforms, channels, anchors,
                   np.zeros(len(dense), np.int64))

    @classmethod
    def load(cls, path, probe):
        path = Path(path)
        if path.suffix == ".npy":
            return cls.from_dense(np.load(path, mmap_mode="r"), probe)
        with np.load(path, allow_pickle=False) as data:
            model = cls(data["waveforms"], data["channels"],
                        data["anchors"], data["assigned"],
                        int(data["version"]))
        if np.any(model.channels >= probe.n_channels):
            raise ValueError("Seed model references unavailable contacts")
        return model

    def exclude_contacts(self, excluded_channels):
        """Mask static bad contacts without changing template identities or width.

        Work on copies first so an unusable external seed cannot partly mutate
        a live model bank. The next canonicalization selects a valid anchor.
        """
        bad = np.asarray(excluded_channels, np.int32)
        if not bad.size:
            return 0
        waveform = self.waveforms.copy()
        channels = self.channels.copy()
        affected = 0
        for unit, row in enumerate(self.channels):
            remove = np.isin(row, bad)
            if not remove.any():
                continue
            keep = np.flatnonzero((row >= 0) & ~remove)
            if not len(keep) or not np.any(self.waveforms[unit][:, keep]):
                raise ValueError('External seed has no signal on usable contacts')
            channels[unit].fill(-1)
            waveform[unit].fill(0)
            channels[unit, :len(keep)] = row[keep]
            waveform[unit, :, :len(keep)] = self.waveforms[unit][:, keep]
            affected += 1
        if affected:
            self.waveforms = waveform
            self.channels = channels
            self.anchors = channels[:, 0].copy()
            self.version += 1
            self.__post_init__()
        return affected

    def snapshot(self):
        return (self.waveforms.copy(), self.channels.copy(),
                self.anchors.copy(), self.assigned.copy(), self.version)

    def canonicalize(self):
        """Align each seed's dominant contact and peak; preserve unit IDs."""
        changed = 0
        for unit in range(len(self.waveforms)):
            valid = self.channels[unit] >= 0
            magnitude = np.abs(self.waveforms[unit]).copy()
            magnitude[:, ~valid] = -1
            peak, slot = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            shift = self.nt // 2 - peak
            if slot == 0 and shift == 0:
                continue
            order = np.r_[slot, np.arange(self.waveforms.shape[2])[
                np.arange(self.waveforms.shape[2]) != slot]]
            waveform = self.waveforms[unit][:, order].copy()
            if shift > 0:
                waveform[shift:] = waveform[:-shift].copy()
                waveform[:shift] = 0
            elif shift < 0:
                waveform[:shift] = waveform[-shift:].copy()
                waveform[shift:] = 0
            self.waveforms[unit] = waveform
            self.channels[unit] = self.channels[unit, order]
            self.anchors[unit] = self.channels[unit, 0]
            changed += 1
        if changed:
            self.version += 1
            self.__post_init__()
        return changed

    def restore(self, state):
        other = LocalModels(*state)
        if other.nt != self.nt:
            raise ValueError("Checkpoint waveform length differs from live nt")
        self.__dict__.update(other.__dict__)

    def update(self, accepted_waveforms, *, fraction=.05):
        """Small bounded update from isolated, high-confidence waveforms."""
        changed = 0
        for unit, snippets in accepted_waveforms.items():
            if not snippets:
                continue
            mean = np.mean(snippets, axis=0, dtype=np.float64).astype(np.float32)
            if mean.shape != self.waveforms[unit].shape or not np.isfinite(mean).all():
                raise ValueError("Invalid update waveform")
            self.waveforms[unit] = ((1 - fraction) * self.waveforms[unit]
                                    + fraction * mean)
            changed += len(snippets)
        if changed:
            self.version += 1
        return changed


def _score_one(residual, center, model, contacts, *, half_width=8, noise_uv=None):
    """Center-weighted proposal with an amplitude and residual-gain estimate."""
    half = len(model) // 2
    valid_contacts = contacts >= 0
    indices = contacts[valid_contacts]
    valid = (center - half >= 0 and center + half < len(residual))
    if not valid or not len(indices):
        return -1., 0., 0.
    span = slice(half - half_width, half + 1 + half_width)
    waveform = model[span][:, valid_contacts]
    signal = residual[center - half_width:center + half_width + 1,
                      indices]
    weight = (np.ones(len(indices), np.float32) if noise_uv is None else
              1. / np.maximum(np.asarray(noise_uv)[indices], .01) ** 2)
    energy = float(np.sum(waveform * waveform * weight, dtype=np.float64))
    observed = float(np.sum(signal * signal * weight, dtype=np.float64))
    if energy <= 1e-9 or observed <= 1e-9:
        return -1., 0., 0.
    dot = float(np.sum(waveform * signal * weight, dtype=np.float64))
    full_model = model[:, valid_contacts]
    full_signal = residual[center-half:center+half+1, indices]
    full_energy = float(np.sum(full_model ** 2 * weight, dtype=np.float64))
    full_dot = float(np.sum(full_model * full_signal * weight, dtype=np.float64))
    amplitude = full_dot / max(full_energy, 1e-9)
    if amplitude <= 0:
        return -1., amplitude, 0.
    cosine = dot / np.sqrt(energy * observed)
    return float(cosine), float(amplitude), float(
        2 * amplitude * full_dot - amplitude ** 2 * full_energy)


@dataclass(frozen=True)
class Match:
    source_sample: int
    channel: int
    unit: int
    score: float
    runner_up: float
    amplitude: float
    residual_gain: float
    pass_index: int
    candidate_sample: int | None = None
    relative_margin: float = 1.


def normalize_template_identity_map(identity_map, n_templates):
    if identity_map is None:
        return None
    raw = np.asarray(identity_map)
    if (raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer)
            or np.issubdtype(raw.dtype, np.bool_)
            or np.any(raw < 0) or len(raw) > n_templates):
        raise ValueError("Invalid template identity map")
    result = raw.astype(np.int64, copy=False)
    if len(result) < n_templates:
        first_new = int(result.max()) + 1 if len(result) else 0
        result = np.concatenate((
            result,
            np.arange(first_new, first_new + n_templates - len(result),
                      dtype=np.int64),
        ))
    return result


def collapse_identity_scores(scored, identity_map):
    """Keep one best waveform representation per stable unit identity."""
    if identity_map is None:
        return list(scored)
    best = {}
    for row in scored:
        gain, score, template_id = row[:3]
        identity = int(identity_map[template_id])
        previous = best.get(identity)
        if previous is None or row > previous:
            best[identity] = row
    return list(best.values())


def match_residual_cpu(residual, events, models: LocalModels, *,
                       score_floor=.65, min_margin=.03,
                       amplitude_min=.3, amplitude_max=3.,
                       half_width=8, max_passes=3, detector=None,
                       noise_uv=None, floor_snr=4.5,
                       core_start=0, core_stop=None, all_events=None,
                       shift_radius=2, refractory_samples=2,
                       template_identity_map=None):
    """Greedy reference: fit, subtract and redetect overlaps in bounded cores.

    The detector receives the residual and must return (sample, channel, snr)
    tuples relative to this haloed array. Only center samples in the core are
    returned. Input events may include halo events, which provide subtraction
    context without creating duplicate cross-core output rows.
    """
    residual = np.array(residual, dtype=np.float32, copy=True)
    identity_map = normalize_template_identity_map(
        template_identity_map, len(models.waveforms))
    if core_stop is None:
        core_stop = len(residual)
    accepted = []
    occupied = set()
    current = list(events)
    for pass_index in range(max_passes):
        if not current:
            break
        if all_events is not None:
            all_events.extend(current)
        # High-amplitude events are fitted first. Repeated detections on
        # neighboring contacts disappear after their shared signal is removed.
        current.sort(key=lambda e: (-e[2], e[0], e[1]))
        for center, channel, snr in current:
            if center < models.nt // 2 or center + models.nt // 2 + 1 > len(residual):
                continue
            ids = models.by_channel.get(int(channel), ())
            if not ids:
                continue
            scored = []
            for unit in ids:
                options = []
                for shift in range(-shift_radius, shift_radius + 1):
                    score, amplitude, gain = _score_one(
                        residual, center + shift, models.waveforms[unit],
                        models.channels[unit], half_width=half_width,
                        noise_uv=noise_uv)
                    effective_floor = max(score_floor, .85) if pass_index >= 3 else score_floor
                    if amplitude_min <= amplitude <= amplitude_max and score >= effective_floor:
                        options.append((gain, score, unit, amplitude, shift))
                if options:
                    scored.append(max(options, key=lambda row: (row[0], -abs(row[4]), -row[4])))
            scored = collapse_identity_scores(scored, identity_map)
            if not scored:
                continue
            scored.sort(reverse=True)
            gain, score, unit, amplitude, shift = scored[0]
            relative_margin = (1 - scored[1][0] / gain if len(scored) > 1 else 1.)
            runner = scored[1][1] if len(scored) > 1 else -1.
            if gain < (floor_snr ** 2 if noise_uv is not None else 1e-9) or relative_margin < min_margin:
                continue
            candidate_sample = int(center)
            center = int(center) + shift
            # Do not assign the same source event twice on one local patch.
            unit_identity = int(unit) if identity_map is None else int(identity_map[unit])
            identity = (int(center), unit_identity)
            if any((int(center) + offset, unit_identity) in occupied
                   for offset in range(-refractory_samples, refractory_samples + 1)):
                continue
            contacts = models.channels[unit]
            good = contacts >= 0
            target = residual[center - models.nt // 2:center + models.nt // 2 + 1]
            target[:, contacts[good]] -= (amplitude *
                                           models.waveforms[unit, :, good].T)
            occupied.add(identity)
            if core_start <= center < core_stop:
                accepted.append(Match(center, channel, unit, score, runner,
                                      amplitude, gain, pass_index,
                                      candidate_sample, relative_margin))
        if detector is None or pass_index + 1 == max_passes:
            break
        current = detector(residual, noise_uv, floor_snr)
    accepted.sort(key=lambda match: (match.source_sample, match.channel,
                                      match.pass_index))
    return accepted, residual
