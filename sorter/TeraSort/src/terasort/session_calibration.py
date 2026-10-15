"""Deterministic, bounded calibration across a probe's complete time span."""

from __future__ import annotations

from collections import defaultdict
from math import ceil

import numpy as np
from scipy.spatial import cKDTree

from .candidates.bank import events_from_indices
from .candidates.detectors import numpy_detect
from .candidates.waveforms import (WaveformSpec, extract_waveforms,
                                   geometry_channel_map)
from .session_models import LocalModels
from .session_signal import assess_quality, iter_cores, preprocess


def select_calibration_windows(probe, day_id, *, window_seconds=60.,
                               spacing_seconds=7_200., target_windows=None):
    """At least start/middle/end, then one minute per two recorded hours.

    On shorter days the first window is at most one minute. An explicit
    target_windows overrides the count while preserving bounded window
    length. Windows never cross source files or explicit gaps. This is a
    sparse preview, not a whole-recording training array.
    """
    if window_seconds <= 0 or spacing_seconds <= 0:
        raise ValueError("Positive calibration window and spacing required")
    if target_windows is not None and (
            isinstance(target_windows, bool) or not isinstance(target_windows, int)
            or not 3 <= target_windows <= 64):
        raise ValueError("Calibration target window count must be 3–64")
    segments = [seg for seg in probe.segments if seg.day_id == day_id]
    if not segments:
        raise ValueError(f"No source segments for {day_id}")
    total = sum(seg.n_samples for seg in segments)
    target = (target_windows if target_windows is not None else
              max(3, ceil(total / (spacing_seconds * probe.sample_rate_hz))))
    length = max(1, round(window_seconds * probe.sample_rate_hz))
    targets = np.linspace(0, max(0, total - 1), target,
                          endpoint=True, dtype=np.int64)
    windows = []
    for offset in targets:
        remaining = int(offset)
        for seg in segments:
            if remaining < seg.n_samples:
                start = seg.start_sample + remaining
                stop = min(seg.stop_sample, start + length)
                # Shift the last window backward instead of producing a
                # several-sample calibration near a file boundary.
                start = max(seg.start_sample, stop - length)
                item = (start, stop)
                if item not in windows:
                    windows.append(item)
                break
            remaining -= seg.n_samples
    return sorted(windows)


def _keys(samples, channels):
    x = np.asarray(samples, np.uint64) ^ (
        np.asarray(channels, np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _spatial_neighbors(probe, radius_um):
    """Sparse KD-tree neighborhoods without an all-channel distance matrix."""
    positions = probe.geometry
    tree = cKDTree(positions)
    rows = []
    for channel, nearby in enumerate(tree.query_ball_point(positions, radius_um)):
        rows.append(sorted(other for other in nearby
                           if probe.shank[other] == probe.shank[channel]))
    neighbors = np.full((probe.n_channels, max(map(len, rows))), -1,
                        np.int32)
    for channel, nearby in enumerate(rows):
        neighbors[channel, :len(nearby)] = nearby
    return neighbors


def _farthest_means(waveforms, n_clusters, *, template_estimator="mean"):
    """Small deterministic, bounded k-means for one anchor's snippets."""
    if template_estimator not in ("mean", "median_rank3"):
        raise ValueError("Template estimator must be mean or median_rank3")
    shape = waveforms.shape[1:]
    features = waveforms[:, waveforms.shape[1]//2-8:waveforms.shape[1]//2+9].reshape(len(waveforms), -1)
    norms = np.linalg.norm(features, axis=1)
    valid = norms > 1e-6
    features = features[valid] / norms[valid, None]
    source = waveforms[valid]
    if len(source) < 10:
        return []
    seeds = [int(np.argmax(norms[valid]))]
    distance = np.sum((features - features[seeds[0]]) ** 2, axis=1)
    for _ in range(1, min(n_clusters, len(source))):
        next_seed = int(np.argmax(distance))
        seeds.append(next_seed)
        distance = np.minimum(distance,
                              np.sum((features - features[next_seed]) ** 2, axis=1))
    centers = features[seeds].copy()
    labels = np.zeros(len(features), np.int32)
    for _ in range(8):
        distances = np.sum((features[:, None] - centers[None]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        for cluster in range(len(centers)):
            chosen = features[labels == cluster]
            if len(chosen):
                centers[cluster] = chosen.mean(axis=0)
                centers[cluster] /= max(np.linalg.norm(centers[cluster]), 1e-6)
    result = []
    for cluster in range(len(centers)):
        selected = source[labels == cluster]
        if len(selected) >= 10:
            if template_estimator == "median_rank3":
                waveform = np.median(selected, axis=0).astype(np.float32)
                u, singular, vt = np.linalg.svd(waveform, full_matrices=False)
                waveform = ((u[:, :3] * singular[:3]) @ vt[:3]).astype(np.float32)
            else:
                waveform = selected.mean(axis=0)
            result.append((len(selected), waveform.reshape(shape)))
    return result


def calibrate_day(probe, day_id, *, floor_snr=5.5, width=16,
                  radius_um=75., reservoir_per_anchor=64,
                  max_total_waveforms=32_768, preprocessor=None,
                  template_estimator="mean", exclude_bad_contacts=False, nt=61):
    """Learn provisional local templates from high-SNR, clean preview cores.

    This fallback is useful when aligned Kilosort seed templates are not
    available. It is not a claim of Kilosort-equivalent model quality.
    """
    from .session_models import validate_nt
    nt = validate_nt(nt)
    if reservoir_per_anchor < 10 or max_total_waveforms < reservoir_per_anchor:
        raise ValueError("Invalid calibration reservoir budget")
    if template_estimator not in ("mean", "median_rank3"):
        raise ValueError("Template estimator must be mean or median_rank3")
    windows = select_calibration_windows(probe, day_id)
    local_map = geometry_channel_map(probe.geometry, probe.shank, radius_um, width,
        excluded_channels=probe.bad_channels if exclude_bad_contacts else ())
    spec = WaveformSpec(local_map, nt//2, nt//2, 1., "float32")
    neighbors = _spatial_neighbors(probe, radius_um)
    reservoir = defaultdict(list)
    admitted = 0
    skipped_qc_cores = 0
    for first, stop in windows:
        for core in iter_cores(probe, start_sample=first, stop_sample=stop,
                               halo_samples=max(2_000, nt//2+1)):
            voltage = (preprocessor or preprocess)(core, probe)
            quality = assess_quality(core, voltage, probe)
            if quality.interval_bad:
                skipped_qc_cores += 1
                continue
            q = np.abs(voltage) / quality.noise_uv
            q[:, ~quality.usable_channels] = 0
            indices = numpy_detect(q, floor=floor_snr, neighbors=neighbors,
                                   valid_start=core.valid_slice.start,
                                   valid_stop=core.valid_slice.stop)
            if not len(indices):
                continue
            events = events_from_indices(indices, voltage, quality.noise_uv,
                                         sample_offset=core.data_start)
            selected = []
            for channel in np.unique(events["channel_index"]):
                current = np.flatnonzero(events["channel_index"] == channel)
                hashes = _keys(events["sample_index"][current],
                               events["channel_index"][current])
                # Only snippets that can enter this anchor's fixed reservoir
                # are extracted from the haloed voltage.
                take = current[np.argsort(hashes, kind="stable")
                               [:reservoir_per_anchor]]
                selected.extend(take.tolist())
            if not selected:
                continue
            selected = np.asarray(sorted(selected), np.int64)
            snippets, _, _ = extract_waveforms(
                events[selected], voltage, core.data_start, spec,
                core.data_start, core.data_start + len(voltage))
            hashes = _keys(events["sample_index"][selected],
                           events["channel_index"][selected])
            for index, snippet, key in zip(selected, snippets, hashes):
                channel = int(events["channel_index"][index])
                if not np.isfinite(snippet[:, local_map[channel] >= 0]).all():
                    continue
                row = reservoir[channel]
                # Own this one snippet: a view would retain the entire
                # extracted batch from every sampled source core.
                row.append((int(key), snippet.copy()))
                row.sort(key=lambda item: item[0])
                if len(row) > reservoir_per_anchor:
                    row.pop()
                else:
                    admitted += 1
                if admitted > max_total_waveforms:
                    # Keep one bounded global population; no silent
                    # per-event truncation in the dense sorting pass.
                    largest = max((entries[-1][0], anchor)
                                  for anchor, entries in reservoir.items()
                                  if entries)
                    reservoir[largest[1]].pop()
                    admitted -= 1
    templates, channels, anchors, counts = [], [], [], []
    for anchor in sorted(reservoir):
        entries = reservoir[anchor]
        if len(entries) < 20:
            continue
        waveforms = np.stack([snippet for _, snippet in entries])
        n_clusters = max(1, min(4, len(waveforms) // 20))
        for count, waveform in _farthest_means(
                waveforms, n_clusters, template_estimator=template_estimator):
            templates.append(waveform)
            channels.append(local_map[anchor])
            anchors.append(anchor)
            counts.append(count)
    if not templates:
        raise RuntimeError("No clean calibration templates; supply aligned Kilosort seeds")
    models = LocalModels(np.stack(templates), np.stack(channels),
                         np.asarray(anchors, np.int32),
                         np.asarray(counts, np.int64))
    aligned = models.canonicalize()
    return models, {"day_id": day_id, "windows": windows,
                    "canonicalized_units": aligned,
                    "sampled_waveforms": admitted, "seed_units": len(models.waveforms),
                    "skipped_qc_cores": skipped_qc_cores,
                    "method": ("bounded_high_snr_local_kmeans_v1"
                               if template_estimator == "mean" else
                               "bounded_high_snr_local_kmeans_median_rank3_v1")}
