"""Bounded fixed temporal proposals for experimental spatial recovery.

Raw cores remain INT16 and FIR responses are INT32. Only a bounded noise sample
and event scores become FP32. A caller must preserve the original detector
pool, use a fixed calibration basis, and reset the noise cache for each core.
Ground-truth qualification and session-level replay remain separate gates.
"""
from pathlib import Path

import numpy as np

from .cuda_environment import configure_cupy_cache


RULE_VERSION = 'seed_anchor_pc1_center17_q10_spatial_union_v1'
MAX_CANDIDATES = 500_000
DEFAULT_RESPONSE_BYTES = 256 * 1024**2


def seed_anchor_coefficients(models):
    """Fixed temporal PC from seed anchors, with bounded covariance slabs."""
    if not len(models.waveforms):
        raise ValueError('Temporal calibration requires seed templates')
    covariance = np.zeros((17, 17), np.float64)
    for first in range(0, len(models.waveforms), 16):
        traces = models.waveforms[first:first+16, models.nt//2-8:models.nt//2+9, 0].astype(np.float64)
        covariance += traces.T @ traces
    if not np.isfinite(covariance).all() or not np.any(covariance):
        raise ValueError('Temporal calibration requires finite nonzero anchors')
    _, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, -1]
    return np.rint(basis / np.max(np.abs(basis)) * 1024).astype(np.int16)


def union_candidates(original, additional, n_channels, *, max_candidates=MAX_CANDIDATES):
    """Retain every original coordinate; keep the stronger score on duplicates.

    Cross-filter callers must recompute a common physical strength frame before
    feeding these records to amplitude-sensitive enrollment or caching.
    """
    if (isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer))
            or not 1 <= n_channels <= 4096 or isinstance(max_candidates, bool)
            or not isinstance(max_candidates, (int, np.integer))
            or not 1 <= max_candidates <= MAX_CANDIDATES):
        raise ValueError('Invalid bounded candidate-union configuration')
    if max(len(original), len(additional)) > max_candidates:
        raise OverflowError('Input detector exceeds the per-core candidate cap')
    indices, strengths = [], []
    for events in (original, additional):
        times = np.fromiter((e[0] for e in events), np.int64, count=len(events))
        contacts = np.fromiter((e[1] for e in events), np.int64, count=len(events))
        scores = np.fromiter((e[2] for e in events), np.float32, count=len(events))
        if (np.any((times < 0) | (times > np.iinfo(np.int32).max))
                or np.any((contacts < 0) | (contacts >= n_channels))
                or not np.all(np.isfinite(scores) & (scores > 0))):
            raise ValueError('Invalid local candidate coordinates or scores')
        indices.append(times*n_channels + contacts)
        strengths.append(scores)
    combined = np.union1d(*indices)
    if len(combined) > max_candidates:
        raise OverflowError('Combined detector exceeds the per-core candidate cap')
    scores = np.zeros(len(combined), np.float32)
    for coordinates, values in zip(indices, strengths):
        np.maximum.at(scores, np.searchsorted(combined, coordinates), values)
    return list(zip((combined//n_channels).astype(np.int32),
                    (combined%n_channels).astype(np.int32), scores))


class CudaTemporalProposals:
    """A fixed 17-tap integer FIR, geometry-local peaks and bounded packing.

    Coefficients and geometry are fixed for an epoch. Save them explicitly;
    rebuilding a basis from adapted templates would invalidate exact replay.
    ``noise_token`` may only be reused for residual passes of the same core.
    """
    def __init__(self, coefficients, neighbors, *, max_response_bytes=DEFAULT_RESPONSE_BYTES):
        coefficients = np.asarray(coefficients)
        neighbors = np.asarray(neighbors)
        if (isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int)
                or not 1 <= max_response_bytes <= 512*1024**2):
            raise ValueError('Temporal response budget must be 1 byte to 512 MiB')
        self.max_response_bytes = max_response_bytes
        if (coefficients.shape != (17,) or coefficients.dtype.kind not in 'iu'
                or np.any(coefficients < -1024) or np.any(coefficients > 1024)
                or not np.any(coefficients)):
            raise ValueError('Expected 17 nonzero bounded integer FIR coefficients')
        if (neighbors.ndim != 2 or neighbors.dtype.kind not in 'iu'
                or not 1 <= len(neighbors) <= 4096 or not 1 <= neighbors.shape[1] <= 16
                or np.any(neighbors < -1) or np.any(neighbors >= len(neighbors))
                or np.any(neighbors[:, 0] != np.arange(len(neighbors)))):
            raise ValueError('Invalid bounded spatial neighbor index')
        for row in neighbors:
            valid = row[row >= 0]
            if len(valid) != len(np.unique(valid)) or np.any(row[:len(valid)] < 0):
                raise ValueError('Neighbor contacts must be unique with padding at the end')
        self.coefficients = coefficients.astype(np.int16, copy=True)
        self.neighbors = neighbors.astype(np.int32, copy=True)
        self.coefficients.flags.writeable = False
        self.neighbors.flags.writeable = False
        configure_cupy_cache()
        import cupy as cp
        self.cp = cp
        module = cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),
                              options=('--std=c++11', '--fmad=false'))
        self.fir = module.get_function('matched_fir17')
        self.peaks = module.get_function('matched_peaks')
        self.strength_kernel = module.get_function('smooth3_candidate_strengths')
        self.device_coefficients = cp.asarray(self.coefficients)
        self.device_neighbors = cp.asarray(self.neighbors)
        self.response = None
        self.count = cp.zeros(1, cp.uint64)
        self.output = None
        self.reset_noise()

    def reset_noise(self):
        self.noise_token = None
        self.noise_estimate = None
        self.device_noise = None

    def filter(self, residual):
        cp = self.cp
        if (not isinstance(residual, cp.ndarray) or residual.ndim != 2
                or residual.dtype != cp.int16 or not residual.flags.c_contiguous
                or not 17 <= len(residual) <= 8*65535
                or residual.shape[1] != len(self.neighbors)):
            raise ValueError('Expected a bounded contiguous INT16 core matching geometry')
        if residual.size * np.dtype('int32').itemsize > self.max_response_bytes:
            raise MemoryError('Temporal response exceeds its fixed GPU budget; shorten the core')
        nt, nc = residual.shape
        if self.response is None or self.response.shape != residual.shape:
            self.response = cp.empty(residual.shape, cp.int32)
        self.fir(((nc+31)//32, (nt+7)//8), (32, 8),
                 (residual, self.device_coefficients, np.int64(nt), np.int32(nc), self.response))
        return self.response

    def detect(self, residual, noise_uv, floor_snr, *, noise_token=None,
               max_candidates=MAX_CANDIDATES):
        indices = self._detect_indices(residual, noise_uv, floor_snr,
                                       noise_token=noise_token, max_candidates=max_candidates)
        cp = self.cp
        indices = cp.sort(indices)
        nc = residual.shape[1]
        flat = cp.asnumpy(indices)
        scores = cp.asnumpy(cp.abs(self.response.ravel()[indices].astype(cp.float32))/self.device_noise[indices%nc])
        return list(zip((flat//nc).astype(np.int32), (flat%nc).astype(np.int32), scores))

    def _detect_indices(self, residual, noise_uv, floor_snr, *, noise_token,
                        max_candidates):
        if (isinstance(floor_snr, (bool, np.bool_)) or not np.isfinite(floor_snr)
                or floor_snr <= 0 or isinstance(max_candidates, bool)
                or not isinstance(max_candidates, (int, np.integer))
                or not 1 <= max_candidates <= MAX_CANDIDATES):
            raise ValueError('Invalid temporal detection threshold or event cap')
        noise = np.asarray(noise_uv, np.float32)
        if (noise.shape != (len(self.neighbors),)
                or np.any(np.isnan(noise) | (noise <= 0))):
            raise ValueError('Positive contact noise required; infinity masks contacts')
        response = self.filter(residual)
        cp = self.cp
        nt, nc = response.shape
        if noise_token is None or self.noise_token is not noise_token or self.device_noise is None:
            # At most 7,983 rows, independent of recording duration. Keep this
            # sampling clock identical to the evaluated prototype.
            sample = cp.asnumpy(response[8:nt-8:max(1, nt//4000)]).astype(np.float32)
            center = np.median(sample, axis=0)
            estimate = np.median(np.abs(sample-center), axis=0)/.67448975
            self.noise_estimate = np.where(np.isfinite(noise), np.maximum(estimate, 1.), np.inf).astype(np.float32)
            self.device_noise = cp.asarray(self.noise_estimate)
            self.noise_token = noise_token
        capacity = min(max_candidates, response.size)
        if self.output is None or len(self.output) != capacity:
            self.output = cp.empty(capacity, cp.int64)
        self.count.fill(0)
        self.peaks(((nt*nc+255)//256,), (256,),
                   (response, self.device_noise, self.device_neighbors,
                    np.int32(self.neighbors.shape[1]), np.int64(nt), np.int32(nc),
                    np.float32(floor_snr), self.count, self.output, np.uint64(capacity)))
        count = int(self.count.get()[0])
        if count > capacity:
            raise OverflowError('Temporal detector exceeds the per-core candidate cap')
        return self.output[:count]

    def detect_union(self, residual, original, noise_uv, floor_snr,
                     contact_scales, physical_noise_uv, *, noise_token=None,
                     max_candidates=MAX_CANDIDATES):
        """Union on the GPU, then pack physical metadata once for the caller.

        The original coordinates all survive. Supplemental packing, union and
        strength computation stay on the GPU, avoiding an intermediate Python
        candidate list and two additional host coordinate conversions. Peak
        input storage is bounded by two capped candidate arrays per core.
        """
        if len(original) > max_candidates:
            raise OverflowError('Input detector exceeds the per-core candidate cap')
        extra = self._detect_indices(residual, noise_uv, floor_snr,
                                     noise_token=noise_token, max_candidates=max_candidates)
        cp = self.cp
        nc = residual.shape[1]
        coordinates = np.fromiter((event[:2] for event in original),
            dtype=np.dtype([('time', np.int64), ('contact', np.int64)]), count=len(original))
        times, contacts = coordinates['time'], coordinates['contact']
        if (np.any((times < 0) | (times >= len(residual)))
                or np.any((contacts < 0) | (contacts >= nc))):
            raise ValueError('Original candidate coordinates are outside the core')
        indices = cp.union1d(cp.asarray(times*nc+contacts), extra)
        if len(indices) > max_candidates:
            raise OverflowError('Combined detector exceeds the per-core candidate cap')
        scores = self._physical_scores(residual, indices, contact_scales, physical_noise_uv)
        flat, scores = cp.asnumpy(indices), cp.asnumpy(scores)
        return list(zip((flat//nc).astype(np.int32), (flat%nc).astype(np.int32), scores))

    def physical_strengths(self, residual, events, contact_scales, noise_uv):
        """Re-express union metadata in the original smooth3 physical SNR frame.

        Filter-trigger scores have different noise gains. Comparing their maxima
        in enrollment's factor-of-two SNR gate can delay real unit discovery.
        This expands three selected samples per event, never a whole FP32 core.
        Zero physical strength is valid for a temporal-energy-triggered event.
        """
        cp = self.cp
        if (not isinstance(residual, cp.ndarray) or residual.ndim != 2
                or residual.dtype != cp.int16 or not residual.flags.c_contiguous
                or residual.shape[1] != len(self.neighbors)
                or not 17 <= len(residual) <= 8*65535):
            raise ValueError('Expected a bounded INT16 core for physical strengths')
        if len(events) > MAX_CANDIDATES:
            raise OverflowError('Physical strengths exceed the per-core event cap')
        times = np.fromiter((e[0] for e in events), np.int64, count=len(events))
        contacts = np.fromiter((e[1] for e in events), np.int64, count=len(events))
        if (np.any((times < 0) | (times >= len(residual)))
                or np.any((contacts < 0) | (contacts >= residual.shape[1]))):
            raise ValueError('Physical strength coordinates are outside the core')
        indices = cp.asarray(times*residual.shape[1]+contacts)
        scores = self._physical_scores(residual, indices, contact_scales, noise_uv)
        return list(zip(times.astype(np.int32), contacts.astype(np.int32), cp.asnumpy(scores)))

    def _physical_scores(self, residual, indices, contact_scales, noise_uv):
        cp = self.cp
        scales = cp.asarray(contact_scales, dtype=cp.float32)
        noise = np.asarray(noise_uv, np.float32)
        if (scales.shape != (residual.shape[1],) or noise.shape != scales.shape
                or np.any(np.isnan(noise) | (noise <= 0))
                or not bool(cp.all(cp.isfinite(scales) & (scales > 0)))):
            raise ValueError('Positive finite scales and positive contact noise required')
        scores = cp.empty(len(indices), cp.float32)
        if not len(indices):
            return scores
        self.strength_kernel(((len(indices)+255)//256,), (256,),
            (residual, indices, scales, cp.asarray(noise), np.int64(len(residual)),
             np.int32(residual.shape[1]), np.int32(len(indices)), scores))
        return scores

    def save(self, group):
        """Persist fixed parameters only; core-local MAD is recomputed on replay."""
        group.attrs['rule_version'] = RULE_VERSION
        group.attrs['max_response_bytes'] = self.max_response_bytes
        group.create_dataset('coefficients', data=self.coefficients)
        group.create_dataset('neighbors', data=self.neighbors)

    @classmethod
    def restore(cls, group):
        if (group.attrs.get('rule_version') != RULE_VERSION
                or 'coefficients' not in group or 'neighbors' not in group
                or group['coefficients'].shape != (17,)
                or group['coefficients'].dtype != np.dtype('int16')
                or len(group['neighbors'].shape) != 2
                or not 1 <= group['neighbors'].shape[0] <= 4096
                or not 1 <= group['neighbors'].shape[1] <= 16
                or group['neighbors'].dtype != np.dtype('int32')):
            raise ValueError('Invalid bounded temporal detector checkpoint')
        return cls(group['coefficients'][:], group['neighbors'][:],
                   max_response_bytes=int(group.attrs.get('max_response_bytes', DEFAULT_RESPONSE_BYTES)))
