"""Display templates without changing the analyzer's scientific features."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from spikeinterface.core import estimate_templates_with_accumulator


def write_centered_native_templates(analyzer, output_folder: Path, job_kwargs: dict) -> None:
    path = Path(output_folder) / "templates.npy"
    if not path.exists():
        return
    shape = np.load(path, mmap_mode="r").shape
    size = shape[1]
    before, after = size // 2, size - size // 2
    recording = analyzer.recording
    spikes = analyzer.get_extension("random_spikes").get_random_spikes()
    mask = analyzer.sparsity.mask if analyzer.is_sparse() else None
    n_channels = int(mask.sum(axis=1).max()) if mask is not None else recording.get_num_channels()
    sums = np.zeros((len(analyzer.unit_ids), size, n_channels), dtype=np.float64)
    interior = np.zeros(len(spikes), dtype=bool)
    for segment in range(recording.get_num_segments()):
        # SI's accumulator excludes the rightmost exactly fitting window too.
        interior |= ((spikes["segment_index"] == segment)
                     & (spikes["sample_index"] >= before)
                     & (spikes["sample_index"] < recording.get_num_samples(segment) - after))
    if np.any(interior):
        means = estimate_templates_with_accumulator(
            recording, spikes[interior], analyzer.unit_ids, before, after,
            return_in_uV=False, sparsity_mask=mask, **job_kwargs,
        )
        counts = np.bincount(spikes[interior]["unit_index"], minlength=len(sums))
        sums += means * counts[:, None, None]
    # Match Phylib's clipped/zero-padded boundary convention without allocating
    # the full spike x samples x channels waveform array.
    for spike in spikes[~interior]:
        segment, sample, unit = (int(spike[key]) for key in ("segment_index", "sample_index", "unit_index"))
        start, stop = max(0, sample - before), min(recording.get_num_samples(segment), sample + after)
        if stop <= start:
            continue
        indices = np.flatnonzero(mask[unit]) if mask is not None else np.arange(n_channels)
        traces = recording.get_traces(
            segment_index=segment, start_frame=start, end_frame=stop,
            channel_ids=analyzer.channel_ids[indices], return_in_uV=False,
        )
        offset = start - (sample - before)
        sums[unit, offset:offset + len(traces), :len(indices)] += traces
    counts = np.bincount(spikes["unit_index"], minlength=len(sums))
    nonzero = counts > 0
    sums[nonzero] /= counts[nonzero, None, None]
    # SI's exporter omits empty units, using this same order.
    exported = [i for i, unit in enumerate(analyzer.unit_ids)
                if len(analyzer.sorting.get_unit_spike_train(unit)) > 0]
    templates = sums[exported]
    if templates.shape != shape:
        raise ValueError(f"Phy display template layout mismatch: {templates.shape} != {shape}")
    np.save(path, templates)
