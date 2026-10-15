"""Equivalent float32 peak rules implemented in NumPy, Torch, and CUDA.

The permissive bank uses radius=1 and neighbors=None. Spatial exclusion is a
separate, optional operation. These rules are explicit; they are not claimed
to reproduce every version of SpikeInterface or Kilosort.
"""
from pathlib import Path
import os
import numpy as np

from ..cuda_environment import configure_cupy_cache


def validate(q, radius, floor, valid_start=0, valid_stop=None, neighbors=None):
    if len(q.shape) != 2 or not all(q.shape):
        raise ValueError('Expected nonempty time x channel scores')
    if not isinstance(radius, (int, np.integer)) or radius < 1 or not np.isfinite(floor) or floor <= 0:
        raise ValueError('Positive finite floor and radius >= 1 required')
    stop = q.shape[0] if valid_stop is None else int(valid_stop)
    if not 0 <= valid_start <= stop <= q.shape[0]:
        raise ValueError('Invalid output interval')
    if neighbors is not None:
        n = np.asarray(neighbors)
        if n.ndim != 2 or n.shape[0] != q.shape[1] or np.any(n < -1) or np.any(n >= q.shape[1]):
            raise ValueError('Invalid channel neighborhood')
    return stop


def neighbor_table(positions, radius_um):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or not np.isfinite(positions).all():
        raise ValueError('Finite positions required')
    near = np.linalg.norm(positions[:, None] - positions[None, :], axis=2) <= radius_um
    table = np.full((len(positions), int(near.sum(1).max())), -1, np.int32)
    for c, row in enumerate(near):
        ids = np.flatnonzero(row)
        table[c, :len(ids)] = ids
    return table


def numpy_detect(q, radius=1, floor=3., neighbors=None, valid_start=0, valid_stop=None):
    """Independent CPU reference; returns sorted flattened indices."""
    q = np.asarray(q, np.float32)
    stop = validate(q, radius, floor, valid_start, valid_stop, neighbors)
    if not np.isfinite(q).all():
        raise ValueError('Scores must be finite')
    keep = q > np.float32(floor)
    for d in range(1, radius + 1):
        if d >= len(q):
            break
        keep[d:] &= q[d:] > q[:-d]
        keep[:-d] &= q[:-d] >= q[d:]
    if neighbors is not None:
        candidates = np.where(keep, q, np.float32(0))
        maxima = candidates.copy()
        for d in range(1, 2 * radius + 1):
            if d >= len(q):
                break
            maxima[d:] = np.maximum(maxima[d:], candidates[:-d])
            maxima[:-d] = np.maximum(maxima[:-d], candidates[d:])
        spatial = np.zeros_like(q)
        for c, ns in enumerate(neighbors):
            ns = ns[ns >= 0]
            if len(ns):
                spatial[:, c] = maxima[:, ns].max(axis=1)
        keep &= q >= spatial - np.float32(1e-8)
    keep[:valid_start] = False
    keep[stop:] = False
    return np.flatnonzero(keep.reshape(-1))


def torch_detect(q, radius=1, floor=3., neighbors=None, valid_start=0, valid_stop=None):
    """Vectorized Torch reference with exactly the same candidate definition.

    Returns CUDA indices; transfer/synchronization is the caller's responsibility.
    Neighbor reduction is tiled to avoid a full time x channel x neighbor array.
    """
    import torch
    import torch.nn.functional as F
    stop = validate(q, radius, floor, valid_start, valid_stop, neighbors)
    with torch.inference_mode():
        values, indices = F.max_pool2d(q[None, None], (2 * radius + 1, 1), 1,
                                     (radius, 0), return_indices=True)
        flat_positions = torch.arange(q.numel(), device=q.device).reshape(q.shape)
        keep = (indices[0, 0] == flat_positions) & (q > floor)
        if neighbors is not None:
            candidate_scores = torch.where(keep, q, 0.)
            temporal = F.max_pool2d(candidate_scores[None, None], (4 * radius + 1, 1),
                                   1, (2 * radius, 0))[0, 0]
            ns = np.where(np.asarray(neighbors) < 0, q.shape[1], neighbors)
            ns = torch.as_tensor(ns, device=q.device, dtype=torch.int64)
            for start in range(0, len(q), 2048):
                end = min(start + 2048, len(q))
                pooled = F.pad(temporal[start:end], (0, 1))[:, ns].amax(2)
                keep[start:end] &= q[start:end] >= pooled - 1e-8
        keep[:valid_start] = False
        keep[stop:] = False
        return torch.nonzero(keep.reshape(-1), as_tuple=True)[0]


class CudaDetector:
    """Custom CUDA C++ kernels compiled by NVRTC, with reusable work buffers."""
    def __init__(self, capacity_fraction=.02, minimum_capacity=4096):
        configure_cupy_cache()
        import cupy as cp
        self.cp = cp
        self.capacity_fraction = capacity_fraction
        self.minimum_capacity = minimum_capacity
        code = Path(__file__).with_name('detect.cu').read_text()
        self.module = cp.RawModule(code=code, options=('--std=c++11', '--fmad=false'))
        self.temporal = self.module.get_function('temporal_flags')
        self.pack = self.module.get_function('pack_candidates')
        self.fused = self.module.get_function('temporal_pack')
        self.buffers = {}

    def detect(self, q, radius=1, floor=3., neighbors=None, valid_start=0,
               valid_stop=None, capacity=None):
        """Return unsorted device indices; fail loudly rather than drop overflow.

        The final count transfer synchronizes once. The caller can sort on CPU
        after transfer. Inputs must be finite (checked by the ingestion layer).
        """
        cp = self.cp
        stop = validate(q, radius, floor, valid_start, valid_stop, neighbors)
        if q.dtype != cp.float32 or not q.flags.c_contiguous:
            raise ValueError('CUDA scores must be contiguous float32')
        nt, nc = q.shape
        cap = max(self.minimum_capacity, int(q.size * self.capacity_fraction)) if capacity is None else int(capacity)
        if cap < 1:
            raise ValueError('Positive output capacity required')
        key = (q.shape, cap, neighbors is not None)
        # One active shape only: running many different shapes must not accumulate buffers.
        if key not in self.buffers:
            self.buffers.clear()
            self.buffers[key] = (cp.empty(q.size, cp.uint8) if neighbors is not None else None,
                                 cp.zeros(1, cp.uint64), cp.empty(cap, cp.int64))
        flags, count, output = self.buffers[key]
        count.fill(0)
        grid = ((q.size + 255) // 256,)
        common = (q, np.int64(nt), np.int32(nc), np.int32(radius), np.float32(floor),
                  np.int64(valid_start), np.int64(stop), count, output, np.uint64(cap))
        if neighbors is None:
            self.fused(grid, (256,), common)
        else:
            ns = cp.asarray(neighbors, dtype=cp.int32)
            self.temporal(grid, (256,), (q, flags, np.int64(nt), np.int32(nc),
                                        np.int32(radius), np.float32(floor)))
            self.pack(grid, (256,), (q, flags, ns, np.int64(nt), np.int32(nc),
                                    np.int32(ns.shape[1]), np.int32(2 * radius),
                                    np.int64(valid_start), np.int64(stop), count, output, np.uint64(cap)))
        n = int(count.get()[0])
        if n > cap:
            raise OverflowError(f'Candidate buffer needs {n} entries; capacity={cap}. Retry with a larger buffer or smaller chunk.')
        return output[:n]
