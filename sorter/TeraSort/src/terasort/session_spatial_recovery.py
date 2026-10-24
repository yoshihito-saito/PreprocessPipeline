"""Bounded original-signal proposals fitted after unchanged baseline learning.

Supplemental assignments remain separate from adaptation/enrollment observations.
The fixed epoch basis and spatial geometry are persisted before the first core.
"""
from bisect import bisect_left, insort
import hashlib
import os
from pathlib import Path

import h5py
import numpy as np

from .candidates.waveforms import geometry_channel_map
from .session_models import Match, normalize_template_identity_map
from .session_temporal import (CudaTemporalProposals, RULE_VERSION as TEMPORAL_RULE,
                               seed_anchor_coefficients)


RULE_VERSION = 'original_spatial_temporal_delayed_frozen_learning_v1'
FIELD_RULE = 'positive_gaussian48_sigma24_mad_weighted_integer_average_v1'
RESPONSE_BYTES = 256 * 1024**2
MAX_CORE_SAMPLES = 8 * 65535
MAX_EXTRA_FITS = 4096
CORE_GUARD = 61


def policy(nt=61):
    """Scientific identity; performance-only changes retain this version."""
    return dict(algorithm=RULE_VERSION, learning='baseline_observations_only',
        proposal_floor_snr=4.5, score_floor=.76, amplitude_min=.7, minimum_margin=.1,
        response_planes_bytes=RESPONSE_BYTES, maximum_extra_fits=MAX_EXTRA_FITS,
        core_guard_samples=nt, temporal_rule=TEMPORAL_RULE, field_rule=FIELD_RULE,
        spatial_radius_um=48., spatial_sigma_um=24., spatial_width=16,
        empty_bank_basis='gaussian17_sigma2_q10_v1',
        core_frame='bounded_per_probe_including_preprocessing_halos_v1')


def bounded_core_samples(probe, core_seconds, halo_samples):
    """Bound both response planes before reading/uploading a wide core."""
    if not 1 <= probe.n_channels <= 4096:
        raise ValueError('Spatial recovery supports 1-4096 contacts per probe')
    budget = min(RESPONSE_BYTES // (8 * probe.n_channels), MAX_CORE_SAMPLES)
    available = budget - 2 * halo_samples
    if available < 1:
        raise ValueError('Recovery response budget cannot contain preprocessing halos')
    return min(max(1, round(core_seconds * probe.sample_rate_hz)), available)


def field_geometry(probe):
    neighbors = geometry_channel_map(probe.geometry, probe.shank, 48., 16)
    distance2 = np.sum((probe.geometry[np.maximum(neighbors, 0)]
                       - probe.geometry[:, None, :])**2, axis=2)
    weights = np.exp(-.5 * distance2 / 24.**2).astype(np.float32)
    weights[neighbors < 0] = 0
    return neighbors, weights


class CudaSpatialTemporalProposals(CudaTemporalProposals):
    """INT16 signal, two bounded INT32 responses and INT64 weighted sums."""

    def __init__(self, coefficients, neighbors, field_neighbors, field_weights,
                 *, max_response_bytes=RESPONSE_BYTES):
        field_neighbors = np.asarray(field_neighbors)
        field_weights = np.asarray(field_weights)
        if (field_neighbors.ndim != 2 or field_neighbors.shape[0] != len(neighbors)
                or not 1 <= field_neighbors.shape[1] <= 16
                or field_neighbors.dtype.kind not in 'iu'
                or np.any(field_neighbors < -1) or np.any(field_neighbors >= len(neighbors))
                or np.any(field_neighbors[:, 0] != np.arange(len(neighbors)))
                or field_weights.shape != field_neighbors.shape
                or not np.isfinite(field_weights).all()
                or np.any(field_weights < 0) or np.any(field_weights > 1)
                or np.any(field_weights[field_neighbors < 0] != 0)):
            raise ValueError('Invalid bounded spatial field geometry')
        for row in field_neighbors:
            valid = row[row >= 0]
            if len(valid) != len(np.unique(valid)) or np.any(row[:len(valid)] < 0):
                raise ValueError('Spatial contacts must be unique with trailing padding')
        super().__init__(coefficients, neighbors, max_response_bytes=max_response_bytes)
        cp = self.cp
        self.field_neighbors = field_neighbors.astype(np.int32, copy=True)
        self.field_weights = field_weights.astype(np.float32, copy=True)
        self.field_neighbors.flags.writeable = False
        self.field_weights.flags.writeable = False
        self.device_field_neighbors = cp.asarray(self.field_neighbors)
        module = cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),
                              options=('--std=c++11', '--fmad=false'))
        self.spatial_kernel = module.get_function('spatial_temporal_average')
        self.field_response = None
        self.field_output = None
        self.field_count = cp.zeros(1, cp.uint64)

    def filter(self, residual):
        if int(residual.size) * 8 > self.max_response_bytes:
            raise MemoryError('Two temporal/spatial response planes exceed the shared budget')
        return super().filter(residual)

    def _pool(self, response, weights):
        cp = self.cp
        if (response.dtype != cp.int32 or response.ndim != 2
                or not response.flags.c_contiguous
                or response.shape[1] != len(self.neighbors)
                or weights.dtype != np.int16 or weights.shape != self.field_neighbors.shape
                or np.any(weights < 0) or np.any(weights[self.field_neighbors < 0] != 0)):
            raise ValueError('Invalid contiguous INT32 field response or INT16 weights')
        if response.size * 8 > self.max_response_bytes:
            raise MemoryError('Two temporal/spatial response planes exceed the shared budget')
        if self.field_response is None or self.field_response.shape != response.shape:
            self.field_response = cp.empty(response.shape, cp.int32)
        denominators = np.sum(weights.astype(np.int32), axis=1, dtype=np.int32)
        self.spatial_kernel(((response.size + 255)//256,), (256,),
            (response, self.device_field_neighbors, cp.asarray(weights), cp.asarray(denominators),
             np.int64(response.size), np.int32(response.shape[1]), np.int32(weights.shape[1]),
             self.field_response))
        return self.field_response

    def _detect_indices(self, residual, noise_uv, floor_snr, *, noise_token, max_candidates):
        cp = self.cp
        original = super()._detect_indices(residual, noise_uv, floor_snr,
            noise_token=noise_token, max_candidates=max_candidates).copy()
        nt, nc = residual.shape
        indices = self.field_neighbors
        valid = ((indices >= 0)
                 & np.isfinite(np.asarray(noise_uv))[np.maximum(indices, 0)])
        noise = self.noise_estimate.astype(np.float64)
        ratio = np.divide(noise[:, None], noise[np.maximum(indices, 0)],
            out=np.zeros(indices.shape, np.float64), where=valid & np.isfinite(noise[:, None]))
        weights = np.rint(np.clip(self.field_weights * ratio * 1024, 0, 32767)).astype(np.int16)
        weights[~valid] = 0
        field = self._pool(self.response, weights)
        sample = cp.asnumpy(field[8:nt-8:max(1, nt//4000)]).astype(np.float32)
        center = np.median(sample, axis=0)
        estimate = np.median(np.abs(sample-center), axis=0) / .67448975
        field_noise = np.where(np.isfinite(noise_uv), np.maximum(estimate, 1.), np.inf).astype(np.float32)
        capacity = min(max_candidates, field.size)
        if self.field_output is None or len(self.field_output) != capacity:
            self.field_output = cp.empty(capacity, cp.int64)
        self.field_count.fill(0)
        self.peaks(((nt * nc + 255)//256,), (256,),
            (field, cp.asarray(field_noise), self.device_neighbors, np.int32(self.neighbors.shape[1]),
             np.int64(nt), np.int32(nc), np.float32(floor_snr), self.field_count,
             self.field_output, np.uint64(capacity)))
        count = int(self.field_count.get()[0])
        if count > capacity:
            raise OverflowError('Spatial detector exceeds the per-core candidate cap')
        union = cp.union1d(original, self.field_output[:count])
        if len(union) > max_candidates:
            raise OverflowError('Temporal/spatial union exceeds the per-core candidate cap')
        return union

    def save(self, group):
        super().save(group)
        group.attrs['field_rule_version'] = FIELD_RULE
        group.create_dataset('field_neighbors', data=self.field_neighbors)
        group.create_dataset('field_weights', data=self.field_weights)

    @classmethod
    def restore(cls, group):
        # Check dimensions/dtypes before reading any potentially oversized data.
        if (group.attrs.get('rule_version') != TEMPORAL_RULE
                or group.attrs.get('field_rule_version') != FIELD_RULE
                or not {'coefficients', 'neighbors', 'field_neighbors', 'field_weights'} <= set(group)
                or group['coefficients'].shape != (17,)
                or group['coefficients'].dtype != np.dtype('int16')):
            raise ValueError('Invalid spatial-temporal detector checkpoint')
        neighbors, fields, weights = (group[n] for n in ('neighbors', 'field_neighbors', 'field_weights'))
        if (neighbors.ndim != 2 or not 1 <= neighbors.shape[0] <= 4096
                or not 1 <= neighbors.shape[1] <= 16 or neighbors.dtype != np.dtype('int32')
                or fields.ndim != 2 or fields.shape[0] != neighbors.shape[0]
                or not 1 <= fields.shape[1] <= 16 or fields.dtype != np.dtype('int32')
                or weights.shape != fields.shape or weights.dtype != np.dtype('float32')):
            raise ValueError('Invalid bounded spatial-temporal checkpoint arrays')
        return cls(group['coefficients'][:], neighbors[:], fields[:], weights[:],
                   max_response_bytes=int(group.attrs['max_response_bytes']))


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenSpatialRecovery:
    """Fixed epoch proposals and one bounded final residual fit pass."""

    def __init__(self, detector):
        self.detector = detector

    @classmethod
    def for_epoch(cls, probe, day_id, calibration_root, models, *, resume=False):
        root = Path(calibration_root)
        seed_path = root/f'{day_id}.npz'
        seed_digest = _file_digest(seed_path)
        geometry = hashlib.sha256(probe.geometry.tobytes() + probe.shank.tobytes()).hexdigest()
        path = root/f'{day_id}.spatial_recovery.h5'
        if path.exists():
            with h5py.File(path, 'r') as group:
                if (group.attrs.get('recovery_rule') != RULE_VERSION
                        or group.attrs.get('calibration_sha256') != seed_digest
                        or group.attrs.get('geometry_sha256') != geometry
                        or group.attrs.get('max_response_bytes') != RESPONSE_BYTES):
                    raise ValueError('Recovery parameters differ from calibration or geometry')
                detector = CudaSpatialTemporalProposals.restore(group)
            return cls(detector)
        if len(models.waveforms) and np.any(models.waveforms[:, models.nt//2-8:models.nt//2+9, 0]):
            coefficients = seed_anchor_coefficients(models)
        else:
            coefficients = np.rint(1024*np.exp(-((np.arange(17)-8)/2.)**2)).astype(np.int16)
        neighbors = geometry_channel_map(probe.geometry, probe.shank, 32., 16)
        fields, weights = field_geometry(probe)
        detector = CudaSpatialTemporalProposals(coefficients, neighbors, fields, weights)
        partial = path.with_suffix('.h5.partial')
        if partial.exists():
            if not resume:
                raise FileExistsError(partial)
            partial.unlink()
        with h5py.File(partial, 'x') as group:
            detector.save(group)
            group.attrs.update(recovery_rule=RULE_VERSION,
                calibration_sha256=seed_digest, geometry_sha256=geometry)
            group.flush()
        os.replace(partial, path)
        return cls(detector)

    def propose(self, matcher, residual, noise, floor_snr, physical_noise, max_candidates):
        if len(residual) < 17:
            return []
        return self.detector.detect_union(residual, [], noise, max(4.5, floor_snr),
            matcher.residual_scales, physical_noise, noise_token=object(),
            max_candidates=max_candidates)

    def fit(self, matcher, residual, noise, additional, baseline, *, core_start, core_stop,
            last_pass, score_floor, amplitude_min, amplitude_max, min_margin,
            floor_snr, half_width, shift_radius, refractory_samples):
        cp = matcher.cp
        final = [m for m in baseline if m.pass_index == last_pass]
        if final:
            matcher.subtract_kernel((len(final),), (256,),
                (residual, matcher.templates, matcher.channels,
                 cp.asarray([m.source_sample for m in final], cp.int32),
                 cp.asarray([m.unit for m in final], cp.int32),
                 cp.asarray([m.amplitude for m in final], cp.float32),
                 np.int32(len(final)), np.int32(len(residual)), np.int32(residual.shape[1]),
                 np.int32(matcher.models.waveforms.shape[2]), *matcher._extra_kernel_args()))
            matcher._check_subtraction()
        choices = matcher._score(residual, additional, noise_uv=noise,
            score_floor=max(.76, score_floor), amplitude_min=max(.7, amplitude_min),
            amplitude_max=amplitude_max, half_width=half_width, shift_radius=shift_radius)
        diagnostics = dict(candidate_events=len(additional),
            **matcher.last_score_diagnostics,
            events_with_fit=sum(c is not None for c in choices),
            events_without_eligible_fit=sum(c is None for c in choices),
            recovery_gain_rejections=0, recovery_margin_rejections=0,
            recovery_edge_rejections=0, recovery_refractory_rejections=0,
            recovery_contact_rejections=0)
        identities = normalize_template_identity_map(matcher.template_identity_map, len(matcher.models.waveforms))
        if identities is None:
            identities = np.arange(len(matcher.models.waveforms), dtype=np.int64)
        occupied = {}
        for match in baseline:
            occupied.setdefault(int(identities[match.unit]), []).append(match.source_sample)
        for rows in occupied.values():
            rows.sort()
        contact_history = {}
        contacts = [tuple(int(c) for c in row if c >= 0) for row in matcher.models.channels]
        proposals = []
        for event, choice in zip(additional, choices):
            if choice is None:
                continue
            score, gain, unit, amplitude, runner, shift, margin = choice
            t, channel, _ = event
            fitted = int(t) + shift
            if gain < floor_snr**2:
                diagnostics['recovery_gain_rejections'] += 1
                continue
            if margin < max(.1, min_margin):
                diagnostics['recovery_margin_rejections'] += 1
                continue
            if not core_start + matcher.models.nt <= fitted < core_stop - matcher.models.nt:
                diagnostics['recovery_edge_rejections'] += 1
                continue
            proposals.append((gain, fitted, int(channel), unit, score, runner, amplitude, int(t), margin))
        proposals.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        accepted = []
        for gain, t, channel, unit, score, runner, amplitude, candidate, margin in proposals:
            rows = occupied.setdefault(int(identities[unit]), [])
            index = bisect_left(rows, t)
            if ((index and t-rows[index-1] <= refractory_samples)
                    or (index < len(rows) and rows[index]-t <= refractory_samples)):
                diagnostics['recovery_refractory_rejections'] += 1
                continue
            conflict = False
            for contact in contacts[unit]:
                history = contact_history.get(contact, ())
                pos = bisect_left(history, t)
                if (pos and t-history[pos-1] < matcher.models.nt) or (pos < len(history) and history[pos]-t < matcher.models.nt):
                    conflict = True
                    break
            if conflict:
                diagnostics['recovery_contact_rejections'] += 1
                continue
            if len(accepted) == MAX_EXTRA_FITS:
                raise OverflowError('Recovery exceeds its fixed per-core fit budget')
            insort(rows, t)
            for contact in contacts[unit]:
                insort(contact_history.setdefault(contact, []), t)
            accepted.append(Match(t, channel, unit, score, runner, amplitude, gain,
                                  last_pass + 1, candidate, margin))
        diagnostics.update(accepted_fits=len(accepted), emitted_matches=len(accepted),
                           recovery_candidates=len(additional), recovery_fits=len(accepted))
        return accepted, diagnostics
