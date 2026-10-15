"""Bounded source reads, one preprocessing frame, and recording-quality flags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Event, Thread
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt


QC_SATURATED = 1
QC_FLATLINE = 2
QC_DROPOUT = 4
QC_NOISY = 8
QC_ARTIFACT = 16
QC_USER_BAD = 32

# Temporary matrices are sized independently of the source duration. Keep
# operations channel/time independent so changing tiles preserves the frame.
_WORK_TILE_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class Core:
    probe_id: str
    day_id: str
    segment_path: Path
    core_start: int
    core_stop: int
    data_start: int
    raw: np.ndarray
    source_bytes_read: int

    @property
    def valid_slice(self):
        return slice(self.core_start - self.data_start,
                     self.core_stop - self.data_start)


@dataclass(frozen=True)
class Quality:
    noise_uv: np.ndarray
    flags: np.ndarray
    interval_bad: bool
    saturated_fraction: np.ndarray
    flatline_fraction: np.ndarray
    zero_fraction: np.ndarray

    @property
    def usable_channels(self):
        # A synchronous artifact is an interval-level warning. It must not
        # erase every contact from the candidate stream.
        return (self.flags & np.uint8(0xFF ^ QC_ARTIFACT)) == 0


def _read_range(handle, source_path, byte_offset, byte_count, *, attempts=3):
    """Retry a bounded byte range after a transient SMB read or reconnect."""
    bytes_received = 0
    for attempt in range(attempts):
        try:
            handle.seek(byte_offset)
            payload = handle.read(byte_count)
            bytes_received += len(payload)
            if len(payload) != byte_count:
                raise OSError("Short source read")
            return handle, payload, bytes_received
        except OSError:
            handle.close()
            if attempt + 1 == attempts:
                raise
            time.sleep(min(.25 * (2 ** attempt), 1.))
            handle = source_path.open("rb", buffering=0)
    raise AssertionError("Unreachable source retry")


def _read_range_into(handle, source_path, byte_offset, destination, *, attempts=3):
    """Fill a bounded buffer without another block-sized allocation.

    Resume at the last confirmed byte after a short read or SMB reconnect.
    A read returning zero before the requested end is an error, not EOF.
    """
    done = failures = 0
    while done < len(destination):
        try:
            handle.seek(byte_offset + done)
            while done < len(destination):
                stop = min(len(destination), done + 8 * 1024**2)
                count = handle.readinto(destination[done:stop])
                if not count:
                    raise OSError("Short source read")
                done += count
        except OSError:
            handle.close()
            failures += 1
            if failures >= attempts:
                raise
            time.sleep(min(.25 * (2 ** (failures - 1)), 1.))
            handle = source_path.open("rb", buffering=0)
    return handle, done


def _prefetch_blocks(iterable):
    """Own at most two blocks: current plus one ready/in-flight lookahead.

    Acquire capacity *before* requesting another allocation from the source.
    Releasing a core queue slot is deliberately unrelated to block capacity.
    """
    slots = BoundedSemaphore(2)
    queue = Queue(maxsize=2)
    stopped = Event()
    sentinel = object()

    def produce():
        source = iter(iterable)
        try:
            while not stopped.is_set():
                if not slots.acquire(timeout=.1):
                    continue
                if stopped.is_set():
                    slots.release()
                    break
                try:
                    block = next(source)
                except StopIteration:
                    slots.release()
                    queue.put(sentinel)
                    break
                except BaseException as exc:
                    slots.release()
                    queue.put((True, exc))
                    break
                queue.put((False, block))
                del block
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    worker = Thread(target=produce, name="terasort-block-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            item = queue.get()
            if item is sentinel:
                return
            failed, block = item
            del item
            if failed:
                raise block
            yield block
            del block
            slots.release()
    finally:
        stopped.set()
        # A stalled network read can finish later; its buffers and handle are
        # still owned by the producer, never reused by the caller.
        worker.join(timeout=5)


def _source_blocks(probe, first_sample, stop_sample, core_samples,
                   grouped_cores, halo_samples):
    frame_bytes = getattr(probe, 'source_n_channels', probe.n_channels) * 2
    for segment in probe.segments:
        first = max(first_sample, segment.start_sample)
        last = min(stop_sample, segment.stop_sample)
        if first >= last:
            continue
        handle = segment.path.open("rb", buffering=0)
        tail = b""
        previous_stop = None
        try:
            for start in range(first, last, grouped_cores * core_samples):
                stop = min(last, start + grouped_cores * core_samples)
                read_start = max(segment.start_sample, start - halo_samples)
                read_stop = min(segment.stop_sample, stop + halo_samples)
                overlap = max(0, (previous_stop - read_start) * frame_bytes) if tail else 0
                payload = bytearray((read_stop - read_start) * frame_bytes)
                if overlap:
                    payload[:overlap] = tail[-overlap:]
                tail = b""
                target = memoryview(payload)[overlap:]
                handle, received = _read_range_into(
                    handle, segment.path,
                    (read_start - segment.start_sample) * frame_bytes + overlap,
                    target)
                del target
                # Retain only the overlap, not a view pinning this whole block.
                next_start = max(segment.start_sample, stop - halo_samples)
                tail = bytes(memoryview(payload)[(next_start - read_start) * frame_bytes:])
                previous_stop = read_stop
                yield segment, start, stop, read_start, payload, received
                del payload
        finally:
            handle.close()


def iter_cores(probe, *, core_seconds=2., halo_samples=2_000,
               start_sample=0, stop_sample=None, read_buffer_mb=0):
    """Yield source-clock cores, never spanning a file boundary or a gap.

    Optional MiB-sized sequential reads amortize requests and shared halos.
    Yielded cores own small copies so queued cores cannot pin old large buffers.
    Sources are read-only; neither path maps or copies the whole recording.
    """
    if not np.isfinite(core_seconds) or core_seconds <= 0:
        raise ValueError("Positive core duration required")
    if not isinstance(halo_samples, int) or halo_samples < 0:
        raise ValueError("Nonnegative integer halo required")
    core_samples = max(1, int(round(core_seconds * probe.sample_rate_hz)))
    if (isinstance(read_buffer_mb, bool) or not isinstance(read_buffer_mb, int)
            or not 0 <= read_buffer_mb <= 1024):
        raise ValueError("read_buffer_mb must be an integer from 0 to 1024 MiB")
    frame_bytes = getattr(probe, 'source_n_channels', probe.n_channels) * 2
    grouped_cores = 1
    if read_buffer_mb:
        room = read_buffer_mb * 1024**2 // frame_bytes - 2*halo_samples
        if room < core_samples:
            raise ValueError("Read buffer must fit one core plus both halos")
        grouped_cores = room // core_samples
    if stop_sample is None:
        stop_sample = probe.stop_sample
    if start_sample < 0 or stop_sample <= start_sample:
        raise ValueError("Invalid source interval")
    if read_buffer_mb:
        blocks = _prefetch_blocks(_source_blocks(
            probe, start_sample, stop_sample, core_samples,
            grouped_cores, halo_samples))
        try:
            for block in blocks:
                segment, block_start, block_stop, read_start, payload, received = block
                for core_start in range(block_start, block_stop, core_samples):
                    core_stop = min(core_start + core_samples, block_stop)
                    data_start = max(segment.start_sample, core_start - halo_samples)
                    data_stop = min(segment.stop_sample, core_stop + halo_samples)
                    raw = np.frombuffer(payload, dtype="<i2",
                                        offset=(data_start - read_start) * frame_bytes,
                                        count=(data_stop - data_start) * (frame_bytes // 2))
                    raw = raw.reshape(-1, frame_bytes // 2)
                    if getattr(probe, "channel_map", None) is not None:
                        raw = raw[:, probe.channel_map]
                    raw = np.array(raw, dtype=np.int16, order="C", copy=True)
                    raw.setflags(write=False)
                    yield Core(probe.probe_id, segment.day_id, segment.path,
                               core_start, core_stop, data_start, raw, received)
                    received = 0
                del block, payload
        finally:
            blocks.close()
        return
    for segment in probe.segments:
        first = max(start_sample, segment.start_sample)
        last = min(stop_sample, segment.stop_sample)
        if first >= last:
            continue
        handle = segment.path.open("rb", buffering=0)
        try:
            for core_start in range(first, last, core_samples):
                core_stop = min(core_start + core_samples, last)
                data_start = max(segment.start_sample, core_start - halo_samples)
                data_stop = min(segment.stop_sample, core_stop + halo_samples)
                offset = (data_start - segment.start_sample) * frame_bytes
                count = (data_stop - data_start) * frame_bytes
                handle, payload, received = _read_range(
                    handle, segment.path, offset, count)
                raw = np.frombuffer(payload, dtype="<i2").reshape(
                    data_stop - data_start, frame_bytes // 2)
                if getattr(probe, "channel_map", None) is not None:
                    raw = np.ascontiguousarray(raw[:, probe.channel_map])
                    raw.setflags(write=False)
                yield Core(probe.probe_id, segment.day_id, segment.path,
                           core_start, core_stop, data_start, raw, received)
        finally:
            handle.close()


def prefetch_cores(iterable, *, depth=2):
    """Bounded asynchronous source reader with backpressure."""
    if depth < 1:
        raise ValueError("Positive prefetch depth required")
    queue = Queue(maxsize=depth)
    stopped = Event()
    sentinel = object()

    def put(value):
        while not stopped.is_set():
            try:
                queue.put(value, timeout=.1)
                return
            except Full:
                pass

    def producer():
        try:
            for core in iterable:
                if stopped.is_set():
                    break
                put((False, core))
        except BaseException as exc:
            put((True, exc))
        finally:
            try:
                if hasattr(iterable, "close"):
                    iterable.close()
            finally:
                put(sentinel)

    worker = Thread(target=producer, name="terasort-source-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = queue.get(timeout=.1)
            except Empty:
                if not worker.is_alive() and queue.empty():
                    raise RuntimeError("Source prefetch stopped without a sentinel")
                continue
            if item is sentinel:
                return
            failed, payload = item
            if failed:
                raise payload
            yield payload
    finally:
        stopped.set()
        worker.join(timeout=5)


def preprocess(core: Core, probe):
    """Per-shank median reference and zero-phase 300-6000 Hz filtering.

    Calibration and dense assignment call this same function. The halo is
    clipped only at a true segment boundary; an internal core has >=100 ms
    context with the default reader settings.
    """
    voltage = np.array(core.raw, dtype=np.float32, copy=True)
    voltage *= np.float32(probe.gain_uv_per_count)
    if getattr(probe, "input_is_preprocessed", False):
        return voltage
    time_rows = max(1, _WORK_TILE_BYTES // (voltage.shape[1] * 4))
    for shank in np.unique(probe.shank):
        indices = np.flatnonzero(probe.shank == shank)
        good = np.setdiff1d(indices, probe.bad_channels, assume_unique=True)
        if len(good) >= 2:
            for first in range(0, len(voltage), time_rows):
                block = voltage[first:first+time_rows]
                reference = np.median(block[:, good], axis=1)
                block[:, indices] -= reference[:, None]
    high = min(6_000., .4 * probe.sample_rate_hz)
    if high <= 300.:
        raise ValueError("Sample rate too low for the 300 Hz spike band")
    sos = butter(3, [300., high], btype="bandpass",
                 fs=probe.sample_rate_hz, output="sos")
    if len(voltage) < 64:
        raise ValueError("Segment too short for zero-phase spike filtering")
    # Filter each channel over its complete haloed time span. Independent
    # channel slabs avoid whole-core FP64 output/padding/filter temporaries;
    # splitting time here would change the zero-phase boundary conditions.
    filter_channels = max(1, _WORK_TILE_BYTES // ((len(voltage) + 64) * 8))
    for first in range(0, voltage.shape[1], filter_channels):
        columns = slice(first, first+filter_channels)
        voltage[:, columns] = sosfiltfilt(sos, voltage[:, columns], axis=0)
    return voltage


def assess_quality(core: Core, voltage_uv, probe):
    """Record bad contacts and intervals without erasing source-clock time."""
    raw = core.raw[core.valid_slice]
    voltage = voltage_uv[core.valid_slice]
    if len(raw) == 0:
        raise ValueError("Empty or nonfinite preprocessed core")
    time_rows = max(1, _WORK_TILE_BYTES // (voltage.shape[1] * 4))
    saturated_count = np.zeros(probe.n_channels, np.int64)
    flat_count = np.zeros(probe.n_channels, np.int64)
    zero_count = np.zeros(probe.n_channels, np.int64)
    for first in range(0, len(raw), time_rows):
        stop = min(len(raw), first+time_rows)
        block = raw[first:stop]
        if not np.isfinite(voltage[first:stop]).all():
            raise ValueError("Empty or nonfinite preprocessed core")
        saturated_count += np.count_nonzero((block >= 32760) | (block <= -32760), axis=0)
        zero_count += np.count_nonzero(block == 0, axis=0)
        # Include the pair spanning a tile boundary; compare ADC counts
        # directly instead of expanding the entire raw core to INT32.
        lo = max(1, first)
        flat_count += np.count_nonzero(raw[lo:stop] == raw[lo-1:stop-1], axis=0)
    median = np.median(voltage[::max(1, len(voltage)//4000)], axis=0)
    noise = np.median(np.abs(voltage[::max(1, len(voltage)//4000)] - median),
                      axis=0).astype(np.float32) / np.float32(.67448975)
    noise = np.maximum(noise, np.float32(.01))
    saturated = saturated_count / len(raw)
    flatline = (flat_count / (len(raw)-1) if len(raw) > 1 else np.ones(probe.n_channels))
    zero = zero_count / len(raw)
    flags = np.zeros(probe.n_channels, np.uint8)
    flags[saturated > 1e-4] |= QC_SATURATED
    flags[flatline > .98] |= QC_FLATLINE
    flags[zero > .99] |= QC_DROPOUT
    for shank in np.unique(probe.shank):
        members = np.flatnonzero(probe.shank == shank)
        baseline = float(np.median(noise[members]))
        flags[members[noise[members] > max(5 * baseline, 100.)]] |= QC_NOISY
    if probe.bad_channels:
        flags[list(probe.bad_channels)] |= QC_USER_BAD
    usable = flags == 0
    # Synchronous high-amplitude voltage on many contacts is an artifact;
    # such cores still produce rows but do not update the model.
    if np.sum(usable) >= 8:
        synchronous = False
        threshold = 10 * noise[usable]
        for first in range(0, len(voltage), time_rows):
            active = np.abs(voltage[first:first+time_rows, usable]) > threshold
            if np.any(np.mean(active, axis=1) > .3):
                synchronous = True
                break
    else:
        synchronous = False
    bad_interval = bool(synchronous or np.mean(~usable) > .2)
    if synchronous:
        flags |= QC_ARTIFACT
    return Quality(noise, flags, bad_interval, saturated, flatline, zero)
