"""Bounded-memory, resumable INT16 LFP export from interleaved INT16 voltage."""

from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly


def _ceil_div(a, b):
    return (a + b - 1) // b


def _write_json_atomic(path, payload):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source_identity(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        head = stream.read(min(size, 1024 * 1024))
        stream.seek(max(0, size - 1024 * 1024))
        tail = stream.read(1024 * 1024)
    return dict(path=str(path.resolve()), bytes=size,
                mtime_ns=path.stat().st_mtime_ns,
                head_sha256=hashlib.sha256(head).hexdigest(),
                tail_sha256=hashlib.sha256(tail).hexdigest())


def _read_frames(stream, start, count, channels):
    frame_bytes = channels * 2
    result = np.empty((count, channels), dtype="<i2")
    buffer = memoryview(result).cast("B")
    stream.seek(start * frame_bytes)
    offset = 0
    while offset < len(buffer):
        read = stream.readinto(buffer[offset:])
        if not read:
            raise EOFError("Source truncated during LFP export")
        offset += read
    return result


def _filter(sample_rate_hz, output_rate_hz, passband_hz):
    if (not isinstance(sample_rate_hz, int) or not isinstance(output_rate_hz, int)
        or sample_rate_hz <= output_rate_hz or output_rate_hz <= 0):
        raise ValueError("Sample rates must be positive integers with output below source")
    stopband_hz = output_rate_hz / 2
    if not 0 < passband_hz < stopband_hz:
        raise ValueError("Passband must be positive and below output Nyquist")
    ratio = Fraction(output_rate_hz, sample_rate_hz)
    up, down = ratio.numerator, ratio.denominator
    upsampled_hz = sample_rate_hz * up
    transition_hz = stopband_hz - passband_hz
    taps, beta = kaiserord(60, transition_hz / (upsampled_hz / 2))
    taps += 1 - taps % 2
    if taps > 100_001:
        raise ValueError("Requested passband needs an impractically long FIR; widen transition")
    kernel = firwin(taps, (passband_hz + stopband_hz) / 2,
                    window=("kaiser", beta), fs=upsampled_hz).astype(np.float32)
    halo = _ceil_div(_ceil_div(taps // 2, up) + down, down) * down
    return up, down, kernel, halo


def export_lfp(filename, output, *, sample_rate_hz, n_channels,
               output_rate_hz=1250, passband_hz=500.0,
               scale_uv_per_count=None, chunk_seconds=5.0, workers=8,
               resume=False):
    """Write time-major INT16 LFP and a JSON sidecar using bounded RAM.

    The filter is a Kaiser-window FIR designed for nominal 60 dB stopband
    rejection beginning at output Nyquist. A source-aligned halo makes each
    chunk identical to a whole-array SciPy resample_poly call with this FIR.
    Original INT16 calibration is retained in the sidecar; output samples are
    rounded to nearest-even and saturated only if filtering overshoots INT16.
    """
    source, target = Path(filename).resolve(), Path(output).resolve()
    if source == target or not source.is_file():
        raise ValueError("Source must exist and differ from output")
    if not isinstance(n_channels, int) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive and finite")
    if scale_uv_per_count is not None and (
        not math.isfinite(scale_uv_per_count) or scale_uv_per_count <= 0
    ):
        raise ValueError("scale_uv_per_count must be positive and finite")
    up, down, kernel, halo = _filter(sample_rate_hz, output_rate_hz, passband_hz)
    source_id = _source_identity(source)
    frame_bytes = n_channels * 2
    if source_id["bytes"] == 0 or source_id["bytes"] % frame_bytes:
        raise ValueError("Input byte count is not a nonempty whole number of INT16 frames")
    source_samples = source_id["bytes"] // frame_bytes
    output_samples = _ceil_div(source_samples * up, down)
    core_samples = max(down, int(sample_rate_hz * chunk_seconds) // down * down)
    config = dict(format="TeraSort-LFP", version=1, source=source_id,
                  source_dtype="<i2", source_sample_rate_hz=sample_rate_hz,
                  source_samples=source_samples, n_channels=n_channels,
                  output_dtype="<i2", output_sample_rate_hz=output_rate_hz,
                  output_samples=output_samples, layout="time_major_interleaved",
                  scale_uv_per_count=scale_uv_per_count,
                  resample_up=up, resample_down=down,
                  passband_hz=passband_hz,
                  stopband_start_hz=output_rate_hz / 2,
                  nominal_stopband_attenuation_db=60,
                  fir_taps=len(kernel), fir_window="kaiser",
                  fir_half_halo_input_samples=halo,
                  core_samples=core_samples,
                  edge_padding="zero outside source",
                  time_origin="output sample k is centered at k / output_sample_rate_hz seconds from source start")
    partial = Path(str(target) + ".partial")
    progress_path = Path(str(target) + ".progress.json")
    metadata_path = Path(str(target) + ".json")
    metadata_partial = Path(str(metadata_path) + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or metadata_path.exists():
        raise FileExistsError("Output or metadata already exists")
    if resume:
        if not partial.exists() or not progress_path.exists():
            raise FileNotFoundError("No partial LFP export to resume")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("config") != config:
            raise ValueError("Source or LFP parameters changed since partial export")
        start = progress["completed_input_samples"]
        completed_output = progress["completed_output_samples"]
        if (not isinstance(start, int) or not 0 <= start <= source_samples or
            (start != source_samples and start % core_samples) or
            completed_output != _ceil_div(start * up, down) or
            partial.stat().st_size < completed_output * frame_bytes):
            raise ValueError("Partial LFP checkpoint is inconsistent")
        clipped = progress["clipped_values"]
        output_stream = partial.open("r+b")
        output_stream.truncate(completed_output * frame_bytes)
        output_stream.seek(0, os.SEEK_END)
    else:
        if any(path.exists() for path in (partial, progress_path, metadata_partial)):
            raise FileExistsError("Partial export exists; use resume=True")
        output_stream = partial.open("xb")
        start = completed_output = clipped = 0
        _write_json_atomic(progress_path, dict(config=config,
            completed_input_samples=0, completed_output_samples=0, clipped_values=0))
    workers = min(workers, n_channels)
    channel_spans = [(i * n_channels // workers, (i + 1) * n_channels // workers)
                     for i in range(workers)]
    worker_context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
    with source.open("rb") as input_stream, output_stream, worker_context as pool:
        for core_start in range(start, source_samples, core_samples):
            core_stop = min(core_start + core_samples, source_samples)
            read_start = max(0, core_start - halo)
            read_stop = min(source_samples, core_stop + halo)
            raw = _read_frames(input_stream, read_start, read_stop - read_start, n_channels)
            values_f32 = raw.astype(np.float32)
            if pool is None:
                filtered = resample_poly(values_f32, up, down, axis=0,
                                         window=kernel, padtype="constant")
            else:
                def filter_channels(span):
                    left, right = span
                    return resample_poly(values_f32[:, left:right], up, down,
                                         axis=0, window=kernel, padtype="constant")
                filtered = np.concatenate(list(pool.map(filter_channels, channel_spans)), axis=1)
            local_start = (core_start - read_start) * up // down
            local_stop = _ceil_div((core_stop - read_start) * up, down)
            values = np.rint(filtered[local_start:local_stop])
            expected = _ceil_div(core_stop * up, down) - _ceil_div(core_start * up, down)
            if values.shape != (expected, n_channels):
                raise RuntimeError("LFP chunk alignment failed")
            clipped += int(np.count_nonzero((values < -32768) | (values > 32767)))
            np.clip(values, -32768, 32767).astype("<i2").tofile(output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
            completed_output += expected
            _write_json_atomic(progress_path, dict(config=config,
                completed_input_samples=core_stop,
                completed_output_samples=completed_output,
                clipped_values=clipped))
    if partial.stat().st_size != output_samples * frame_bytes:
        raise RuntimeError("LFP output byte count differs from expected sample count")
    result = dict(config, clipped_values=clipped,
                  filter_workers=workers,
                  output_bytes=partial.stat().st_size,
                  complete=True,
                  provenance="Source size/mtime and head/tail hashes; full source hash omitted to avoid another pass")
    _write_json_atomic(metadata_partial, result)
    os.replace(partial, target)
    os.replace(metadata_partial, metadata_path)
    progress_path.unlink()
    return result
