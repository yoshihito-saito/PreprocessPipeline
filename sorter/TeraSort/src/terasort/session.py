"""Ordered binary sources and the virtual sample coordinates used by Kilosort."""

import json
import os
from pathlib import Path

import numpy as np


MANIFEST_NAME = "session_sources.json"


def prepare_session(filenames, settings, dtype="int16"):
    """Validate source framing without reading the recordings into memory."""
    if not isinstance(filenames, (list, tuple)) or len(filenames) < 2:
        raise ValueError("A session requires at least two ordered source files")
    channels = settings.get("n_chan_bin")
    rate = settings.get("fs")
    if type(channels) is not int or channels <= 0:
        raise ValueError("Session settings require positive integer n_chan_bin")
    if not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("Session settings require positive fs")
    itemsize = np.dtype(dtype).itemsize
    frame_bytes = channels * itemsize
    sources = []
    seen = set()
    offset = 0
    for index, filename in enumerate(filenames):
        path = Path(filename).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Source {index + 1} is not a file")
        identity = os.path.normcase(str(path))
        if identity in seen:
            raise ValueError("The same source file was supplied more than once")
        seen.add(identity)
        size = path.stat().st_size
        if not size or size % frame_bytes:
            raise ValueError(f"Source {index + 1} is empty or has an incomplete {channels}-channel frame: {path}")
        samples = size // frame_bytes
        sources.append({"index": index, "path": str(path), "bytes": size,
                        "samples": samples, "virtual_start_sample": offset,
                        "virtual_stop_sample": offset + samples,
                        "has_next_source": index < len(filenames) - 1,
                        "gap_after_samples": None})
        offset += samples
    return {"schema_version": 1, "format": "terasort.ordered_binary_session",
            "sample_rate_hz": rate, "n_chan_bin": channels, "dtype": np.dtype(dtype).name,
            "virtual_total_samples": offset,
            "timebase": "virtual_concatenation_for_sorting_only",
            "boundary_note": "Adjacent files may have an unknown acquisition gap. No samples are inserted. "
                             "Kilosort may process data across a file boundary; inspect or exclude events near boundaries.",
            "sources": sources}


def write_session_manifest(results_dir, manifest):
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MANIFEST_NAME
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def locate_spikes(spike_times, manifest):
    """Map virtual Kilosort sample indices to source index and local sample."""
    times = np.asarray(spike_times)
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("Spike times must be integer sample indices")
    stops = np.asarray([s["virtual_stop_sample"] for s in manifest["sources"]], dtype=np.int64)
    if np.any(times < 0) or np.any(times >= stops[-1]):
        raise ValueError("Spike time is outside the session")
    indices = np.searchsorted(stops, times, side="right")
    starts = np.asarray([s["virtual_start_sample"] for s in manifest["sources"]], dtype=np.int64)
    return indices, times - starts[indices]
