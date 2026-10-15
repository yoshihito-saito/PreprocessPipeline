"""Bounded two-template fit for experimental close-overlap evaluation.

This module does not change the streaming matcher. A caller must first propose
two distinct templates and still enforce event identity, refractory, QC, and
output policies. Only a local waveform patch and at most 32 contacts are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointFit:
    amplitude_a: float
    amplitude_b: float
    residual_gain: float
    incremental_gain: float
    template_correlation: float


def fit_two_templates(signal_uv, noise_uv, waveform_a, channels_a, center_a,
                      waveform_b, channels_b, center_b, *,
                      amplitude_min=.3, amplitude_max=3.,
                      minimum_incremental_gain=20.25,
                      minimum_incremental_fraction=.015,
                      min_center_separation_samples=3,
                      max_template_correlation=.98) -> JointFit | None:
    """Fit two overlapping equal-length templates with bounded positive gains.

    The comparison is against the better of the two single-template fits on
    the *same union patch*. The fractional gain guard limits splitting a
    slightly mismodeled single spike into two units. Both defaults are
    experimental and require real-data calibration before matcher use. The
    default three-sample center gap avoids an observed same-event false split;
    exact co-firing is left unresolved until identity evidence is stronger.
    None means the pair lacks sufficient independent evidence; it says nothing
    about whether either single-template fit should be accepted.
    """
    signal = np.asarray(signal_uv)
    noise = np.asarray(noise_uv)
    templates = (np.asarray(waveform_a), np.asarray(waveform_b))
    nt = templates[0].shape[0] if templates[0].ndim == 2 else 0
    channel_rows = (np.asarray(channels_a), np.asarray(channels_b))
    centers = (center_a, center_b)
    if (nt < 21 or nt % 2 != 1 or signal.ndim != 2 or not np.issubdtype(signal.dtype, np.number)
            or noise.shape != (signal.shape[1],)
            or np.any(np.isnan(noise) | (noise <= 0))
            or not np.isfinite(amplitude_min) or not np.isfinite(amplitude_max)
            or not 0 < amplitude_min <= amplitude_max
            or not np.isfinite(minimum_incremental_gain)
            or minimum_incremental_gain < 0
            or not np.isfinite(minimum_incremental_fraction)
            or not 0 <= minimum_incremental_fraction < 1
            or isinstance(min_center_separation_samples, (bool, np.bool_))
            or not isinstance(min_center_separation_samples, (int, np.integer))
            or not 0 <= min_center_separation_samples <= nt-1
            or not np.isfinite(max_template_correlation)
            or not 0 <= max_template_correlation < 1):
        raise ValueError("Invalid local signal, noise, or pair-fit thresholds")
    for template, channels, center in zip(templates, channel_rows, centers):
        if (template.ndim != 2 or template.shape[0] != nt
                or channels.shape != (template.shape[1],)
                or not np.issubdtype(channels.dtype, np.integer)
                or np.any(channels < -1) or np.any(channels >= signal.shape[1])
                or not np.isfinite(template).all()
                or not isinstance(center, (int, np.integer))
                or center < (nt//2) or center + (nt//2) >= len(signal)):
            raise ValueError("Invalid local template or waveform center")
        valid = channels[channels >= 0]
        if not len(valid) or len(np.unique(valid)) != len(valid):
            raise ValueError("Each template needs distinct valid contacts")
    separation = abs(int(center_a) - int(center_b))
    if separation < min_center_separation_samples or separation > nt-1:
        return None

    first = min(int(center_a), int(center_b)) - (nt//2)
    stop = max(int(center_a), int(center_b)) + (nt//2+1)
    contacts = np.union1d(channel_rows[0][channel_rows[0] >= 0],
                          channel_rows[1][channel_rows[1] >= 0])
    if len(contacts) > 32:
        raise ValueError("Pair exceeds the 32-contact fit budget")
    positions = {int(channel): slot for slot, channel in enumerate(contacts)}
    weights = 1. / noise[contacts].astype(np.float64) ** 2
    observed = np.asarray(signal[first:stop][:, contacts], np.float64)
    if not np.isfinite(observed).all():
        raise ValueError("Nonfinite local signal patch")
    design = []
    for template, channels, center in zip(templates, channel_rows, centers):
        aligned = np.zeros_like(observed)
        offset = int(center) - (nt//2) - first
        for slot, channel in enumerate(channels):
            if channel >= 0:
                aligned[offset:offset + nt, positions[int(channel)]] = template[:, slot]
        design.append(aligned)
    a, b = design
    weighted = weights[None, :]
    gram_aa = float(np.sum(a * a * weighted, dtype=np.float64))
    gram_bb = float(np.sum(b * b * weighted, dtype=np.float64))
    if min(gram_aa, gram_bb) <= 1e-9:
        return None
    gram_ab = float(np.sum(a * b * weighted, dtype=np.float64))
    correlation = gram_ab / np.sqrt(gram_aa * gram_bb)
    if abs(correlation) >= max_template_correlation:
        return None
    dot_a = float(np.sum(a * observed * weighted, dtype=np.float64))
    dot_b = float(np.sum(b * observed * weighted, dtype=np.float64))

    def gain(x, y):
        return (2*x*dot_a + 2*y*dot_b - gram_aa*x*x
                - 2*gram_ab*x*y - gram_bb*y*y)

    # Exact maximum of a concave quadratic on a two-dimensional box:
    # unconstrained stationary point, four stationary edge points, corners.
    proposals = []
    determinant = gram_aa * gram_bb - gram_ab * gram_ab
    if determinant > 1e-12 * gram_aa * gram_bb:
        x = (dot_a * gram_bb - dot_b * gram_ab) / determinant
        y = (dot_b * gram_aa - dot_a * gram_ab) / determinant
        if amplitude_min <= x <= amplitude_max and amplitude_min <= y <= amplitude_max:
            proposals.append((x, y))
    for x in (amplitude_min, amplitude_max):
        y = np.clip((dot_b - gram_ab*x) / gram_bb,
                    amplitude_min, amplitude_max)
        proposals.append((x, float(y)))
    for y in (amplitude_min, amplitude_max):
        x = np.clip((dot_a - gram_ab*y) / gram_aa,
                    amplitude_min, amplitude_max)
        proposals.append((float(x), y))
    x, y = max(proposals, key=lambda pair: gain(*pair))
    joint_gain = gain(x, y)
    single_a = np.clip(dot_a / gram_aa, amplitude_min, amplitude_max)
    single_b = np.clip(dot_b / gram_bb, amplitude_min, amplitude_max)
    best_single = max(0., gain(float(single_a), 0.),
                      gain(0., float(single_b)))
    increment = joint_gain - best_single
    if (increment < minimum_incremental_gain or joint_gain <= 0
            or increment / joint_gain < minimum_incremental_fraction):
        return None
    return JointFit(float(x), float(y), float(joint_gain), float(increment),
                    float(correlation))
