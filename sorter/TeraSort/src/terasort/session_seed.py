"""Bounded seed previews and waveform reconstruction using local preview data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .candidates.waveforms import geometry_channel_map
from .session_manifest import Segment
from .session_models import LocalModels
from .session_signal import Core, _read_range_into, assess_quality, iter_cores, preprocess


@dataclass(frozen=True)
class SeedPreview:
    path: Path
    prefix: Path
    suffix: Path
    segment: Segment
    first: int
    stop: int
    data_start: int
    data_stop: int
    preview_start: int


def stage_seed_previews(probe, windows, root, *, scratch_limit_bytes=5*1024**3, nt=61):
    """Stage unchanged KS input plus boundary halos in one source-read pass.

    Halo sidecars are excluded from the KS input. Keeping them separately
    preserves both the KS preview clock and the original session filter frame.
    The scratch budget counts all raw preview and halo bytes before writing.
    """
    root = Path(root)
    halo = max(nt//2+1, round(.1*probe.sample_rate_hz))
    previews, offset, scratch_bytes = [], 0, 0
    for index, (first, stop) in enumerate(windows):
        first, stop = int(first), int(stop)
        segment = next((s for s in probe.segments
                        if s.start_sample <= first < stop <= s.stop_sample), None)
        if segment is None:
            raise ValueError("Seed preview must lie inside one source segment")
        data_start, data_stop = max(segment.start_sample, first-halo), min(segment.stop_sample, stop+halo)
        previews.append(SeedPreview(root/f"preview-{index}.bin",
            root/f"preview-{index}.prefix.bin", root/f"preview-{index}.suffix.bin",
            segment, first, stop, data_start, data_stop, offset))
        offset += stop-first
        scratch_bytes += (data_stop-data_start)*probe.n_channels*2
    if not previews or len(previews) > 12 or scratch_bytes > scratch_limit_bytes:
        raise ValueError("Seed previews and halos exceed the bounded scratch/window budget")
    # One block holds a full two-second core and halos. Very wide cores use
    # direct per-core reads instead of growing the two-block queue past 512 MiB.
    minimum_mb = int(np.ceil((round(2*probe.sample_rate_hz)+2*halo)*getattr(probe, 'source_n_channels', probe.n_channels)*2/1024**2))
    buffer_mb = max(64, minimum_mb) if minimum_mb <= 256 else 0
    source_bytes = 0
    for preview in previews:
        with preview.path.open("xb") as output, preview.prefix.open("xb") as prefix, preview.suffix.open("xb") as suffix:
            cores = iter_cores(probe, start_sample=preview.first, stop_sample=preview.stop,
                               halo_samples=halo, read_buffer_mb=buffer_mb)
            try:
                for core in cores:
                    source_bytes += core.source_bytes_read
                    core.raw[core.valid_slice].tofile(output)
                    if core.core_start == preview.first:
                        core.raw[:core.valid_slice.start].tofile(prefix)
                    if core.core_stop == preview.stop:
                        core.raw[core.valid_slice.stop:].tofile(suffix)
            finally:
                cores.close()
    return previews, dict(raw_preview_bytes=offset*probe.n_channels*2,
                          raw_scratch_bytes=scratch_bytes, source_bytes_read=source_bytes,
                          staging_read_buffer_mb=buffer_mb)


def iter_seed_preview_cores(probe, preview, *, nt=61):
    """Reproduce original two-second source-clock cores from local files only."""
    halo = max(nt//2+1, round(.1*probe.sample_rate_hz))
    core_samples = max(1, round(2*probe.sample_rate_hz))
    regions = [(preview.prefix, preview.data_start, preview.first),
               (preview.path, preview.first, preview.stop),
               (preview.suffix, preview.stop, preview.data_stop)]
    handles = []
    try:
        for path, lo, hi in regions:
            if path.stat().st_size != (hi-lo)*probe.n_channels*2:
                raise ValueError("Staged seed preview size changed")
            handles.append(path.open("rb", buffering=0))
        for first in range(preview.first, preview.stop, core_samples):
            stop = min(first+core_samples, preview.stop)
            data_start, data_stop = max(preview.data_start, first-halo), min(preview.data_stop, stop+halo)
            raw = np.empty((data_stop-data_start, probe.n_channels), dtype=np.int16)
            target, received = memoryview(raw).cast("B"), 0
            for index, (path, lo, hi) in enumerate(regions):
                start, end = max(lo, data_start), min(hi, data_stop)
                if start >= end:
                    continue
                frame = probe.n_channels*2
                handles[index], count = _read_range_into(handles[index], path, (start-lo)*frame,
                    target[(start-data_start)*frame:(end-data_start)*frame])
                received += count
            del target
            raw.setflags(write=False)
            yield Core(probe.probe_id, preview.segment.day_id, preview.segment.path,
                       first, stop, data_start, raw, received)
    finally:
        for handle in handles:
            handle.close()


def rebuild_seed_models(probe, previews, ksroot, *, preprocessor=None, exclude_bad_contacts=False, nt=61):
    """Preserve the bounded stratified sample and median/rank-three estimator."""
    from .session_models import validate_nt
    nt = validate_nt(nt)
    ksroot = Path(ksroot)
    times = np.load(ksroot/"spike_times.npy", mmap_mode="r").ravel()
    labels = np.load(ksroot/"spike_clusters.npy", mmap_mode="r").ravel()
    templates = np.load(ksroot/"templates.npy", mmap_mode="r")
    if templates.shape[1] != nt:
        raise ValueError("KS4 seed waveform length differs from bounded nt")
    channel_map = np.load(ksroot/"channel_map.npy").ravel()
    inverse = np.load(ksroot/"whitening_mat_inv.npy")
    units, counts = np.unique(labels, return_counts=True)
    units = units[counts >= 20]
    if not len(units) or len(units) > 4096:
        raise RuntimeError("Calibration unit count outside bounded seed budget")
    anchors = np.empty(len(templates), np.int64)
    # Unwhiten only small unit slabs, not a full unit/time/channel bank.
    for first in range(0, len(templates), 16):
        physical = templates[first:first+16] @ inverse
        anchors[first:first+16] = channel_map[np.argmax(np.max(np.abs(physical), axis=1), axis=1)]
        del physical
    # Keep the typical 32,768-snippet target, but never allocate fewer than
    # the ten observations required to retain a unit. Otherwise a bank of
    # 3,277-4,096 identities is guaranteed to lose every seed. The 4,096-unit
    # limit above bounds the maximum at 40,960 snippets (152.5 MiB FP32 at
    # 61 samples and 16 contacts; proportional to nt), independently of recording duration.
    cap = min(128, max(10, 32768//len(units)))
    selected_times, selected_units = [], []
    for unit in units:
        spikes = np.sort(times[labels == unit])
        chosen = spikes[np.linspace(0, len(spikes)-1, min(cap, len(spikes)), dtype=int)]
        selected_times.append(chosen)
        selected_units.append(np.full(len(chosen), unit, dtype=np.int64))
    selected_times, selected_units = np.concatenate(selected_times), np.concatenate(selected_units)
    order = np.argsort(selected_times, kind="stable")
    selected_times, selected_units = selected_times[order], selected_units[order]
    local_map = geometry_channel_map(probe.geometry, probe.shank, 75., 16,
        excluded_channels=probe.bad_channels if exclude_bad_contacts else ())
    samples = {int(unit): [] for unit in units}
    timing = dict(preprocessing_seconds=0., qc_seconds=0., local_rebuild_bytes_read=0)
    for preview in previews:
        for core in iter_seed_preview_cores(probe, preview, nt=nt):
            timing["local_rebuild_bytes_read"] += core.source_bytes_read
            begin = time.perf_counter()
            voltage = (preprocessor or preprocess)(core, probe)
            after = time.perf_counter()
            quality = assess_quality(core, voltage, probe)
            timing["preprocessing_seconds"] += after-begin
            timing["qc_seconds"] += time.perf_counter()-after
            if not quality.interval_bad:
                # Binary-search only the selected events belonging to this core.
                # Endpoint exclusions and per-unit time order match the old loop.
                lo = max(preview.preview_start+core.core_start-preview.first, preview.preview_start+nt//2+1)
                hi = min(preview.preview_start+core.core_stop-preview.first,
                         preview.preview_start+preview.stop-preview.first-nt//2-1)
                left, right = np.searchsorted(selected_times, [lo, hi])
                for unit in np.unique(selected_units[left:right]):
                    channels = local_map[anchors[unit]]
                    valid = channels >= 0
                    if not np.all(quality.usable_channels[channels[valid]]):
                        continue
                    chosen = selected_times[left:right][selected_units[left:right] == unit]
                    centers = chosen-preview.preview_start+preview.first-core.data_start
                    snippets = np.zeros((len(centers), nt, local_map.shape[1]), np.float32)
                    rows = centers[:, None]+np.arange(-(nt//2), nt//2+1)
                    snippets[:, :, valid] = voltage[rows[:, :, None], channels[valid][None, None, :]]
                    samples[int(unit)].append(snippets)
            del voltage
    waveforms, contacts, retained, supports = [], [], [], []
    for unit in units:
        count = sum(len(block) for block in samples[int(unit)])
        if count < 10:
            continue
        waveform = np.median(np.concatenate(samples[int(unit)]), axis=0)
        u, s, vt = np.linalg.svd(waveform, full_matrices=False)
        waveforms.append((u[:, :3]*s[:3]) @ vt[:3])
        contacts.append(local_map[anchors[unit]])
        retained.append(int(unit))
        supports.append(count)
    if not retained:
        raise RuntimeError("No calibration units retained after waveform/QC checks")
    models = LocalModels(np.array(waveforms), np.array(contacts), anchors[retained],
                         np.zeros(len(retained), np.int64))
    aligned = models.canonicalize()
    return models, dict(calibration_unit_ids=retained, waveform_support=supports,
        seed_units=len(retained), canonicalized_units=aligned,
        rebuild_sampler='stratified_min10_target32768_hard40960_v2',
        selected_waveform_quota_per_unit=cap,
        selected_waveform_count=len(selected_times), **timing)
