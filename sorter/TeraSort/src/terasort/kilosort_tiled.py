"""Bound score/NMS workspace while preserving the validated fused detector.

The full temporal convolution is retained to preserve its arithmetic. Spatial
scores, signed choices and neighborhood maxima live only for a time tile plus
the temporal suppression halo. This bounds workspace within a Kilosort batch;
it does not make Kilosort's downstream global clustering bounded-memory.

Selection semantics follow MouseLand/Kilosort 4.1.7 (GPL-3.0).
"""
from contextlib import contextmanager
import inspect
import operator

from .kilosort_native import NativeReductions, make_native_template_match


def make_tiled_template_match(backend, tile_samples=4096):
    import torch
    from torch.nn.functional import conv1d, max_pool1d

    tile_samples = operator.index(tile_samples)
    if tile_samples < 1:
        raise ValueError('tile_samples must be positive')
    # Also verifies the installed version and the original function's hash.
    # Retain its arithmetic for geometries without our specialized CUDA kernel.
    fallback, _ = make_native_template_match(backend, 'fused')

    def template_match(X, ops, iC, iC2, weigh, device=torch.device('cuda')):
        if weigh.shape[:2] != (5, 10) or iC.shape[1] > 65535:
            return fallback(X, ops, iC, iC2, weigh, device=device)
        nt = operator.index(ops['nt'])
        radius = operator.index(ops['settings']['nt0min'])
        nk = operator.index(ops['settings']['n_templates'])
        if nt < 1 or nt % 2 == 0 or radius < 0 or X.shape[-1] < 1:
            raise ValueError('Expected odd positive nt, nonnegative nt0min and nonempty time axis')
        if ops['wTEMP'].shape != (nk, nt):
            raise ValueError('Temporal template shape disagrees with settings')
        NT = X.shape[-1]
        B = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=nt // 2)
        locations, signed_choices, amplitudes = [], [], []
        for start in range(0, NT, tile_samples):
            stop = min(start + tile_samples, NT)
            lo, hi = max(0, start - radius), min(NT, stop + radius)
            scores, choices = backend.template_scores(B, iC, weigh, lo, hi)
            spatial_max = backend.neighbor_max(scores, iC2)
            # Match the original's global border zeroing BEFORE temporal NMS.
            if lo < nt:
                spatial_max[:, :min(nt, hi) - lo] = 0
            if hi > NT - nt:
                spatial_max[:, max(NT - nt - lo, 0):] = 0
            temporal_max = max_pool1d(spatial_max.unsqueeze(0), 2 * radius + 1,
                                     stride=1, padding=radius).squeeze(0)
            core = slice(start - lo, stop - lo)
            core_scores = scores[:, core]
            xy = ((temporal_max[:, core] == core_scores) &
                  (core_scores > ops['Th_universal'])).nonzero()
            amplitudes.append(core_scores[xy[:, 0], xy[:, 1]])
            signed_choices.append(choices[:, core][xy[:, 0], xy[:, 1]])
            xy[:, 1] += start
            locations.append(xy)
            # Release each tile before allocating the next. Candidate vectors
            # remain bounded by the caller's batch, with no truncation/cap.
            del scores, choices, spatial_max, temporal_max, core_scores, xy

        xy = torch.cat(locations)
        signed = torch.cat(signed_choices)
        amp = torch.cat(amplitudes)
        # Original nonzero() orders by center, then time. Tiling orders by time
        # tile first; restore the exact original order before feature extraction.
        order = torch.argsort(xy[:, 0] * NT + xy[:, 1])
        xy, signed, amp = xy[order], signed[order], amp[order]
        imax = signed.abs() - 1
        adist = B[iC[:, xy[:, 0]], imax % nk, xy[:, 1]] * signed.sign()
        return xy, imax, amp, adist

    return template_match


@contextmanager
def tiled_kilosort(tile_samples=4096, deep=False):
    """Opt-in scoped patch, optionally combined with the learned CUDA matcher."""
    from kilosort import spikedetect
    from contextlib import nullcontext
    backend = NativeReductions()
    replacement = make_tiled_template_match(backend, tile_samples)
    if deep:
        from .kilosort_deep import deep_kilosort
        outer = deep_kilosort()
    else:
        outer = nullcontext({})
    with outer as details:
        original = spikedetect.template_match
        spikedetect.template_match = replacement
        try:
            yield dict(details, mode='deep_tiled' if deep else 'tiled',
                       generated_source=inspect.getsource(replacement),
                       universal_tile_samples=tile_samples,
                       universal_workspace='time-tiled scores with NMS halo')
        finally:
            spikedetect.template_match = original
