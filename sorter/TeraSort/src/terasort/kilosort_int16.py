"""Scoped INT16 disk/H2D reader for Kilosort 4.1.7; FP32 processing unchanged."""
from contextlib import contextmanager
from collections import deque
from bisect import bisect_right
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import time
import weakref

import numpy as np

from .cuda_environment import configure_cupy_cache

EXPECTED_READER_SHA256 = '69f3209529c20618126774b3ac2b6c65d800005c8b379b4fcc200fc186fdc055'


def channel_calibration(value, channels, default):
    array = np.asarray(default if value is None else value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full(channels, array, dtype=np.float32)
    elif array.shape == (channels, 1):
        array = array[:, 0]
    if array.shape != (channels,) or not np.isfinite(array).all():
        raise ValueError('Calibration must be finite, scalar or one value per channel')
    return np.ascontiguousarray(array)


def _cache_path_key(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def source_slice(reader, start, stop, cache=None, sequential=False, cache_sources=None):
    """Read only the files overlapping a batch, even in a long file list."""
    source = reader.file
    if not hasattr(source, "split_indices") or source._filenames is None:
        if cache is not None:
            filename = reader.filename
            # Kilosort normalizes even a single named input to a one-item
            # list on some reader paths. The non-group reader still maps one
            # file, so pass its path (rather than the list) to the cache.
            if isinstance(filename, (list, tuple)):
                if len(filename) != 1:
                    raise ValueError('Non-group Kilosort reader has multiple source filenames')
                filename = filename[0]
            if cache_sources is None or _cache_path_key(filename) in cache_sources:
                payload = cache.read(filename, start*reader.n_chan_bin*2,
                                     stop*reader.n_chan_bin*2, sequential=sequential)
                return np.frombuffer(payload,dtype=np.int16).reshape(-1,reader.n_chan_bin)
        return source[start:stop]
    pieces = []
    while start < stop:
        index = bisect_right(source.split_indices, start)
        previous = 0 if index == 0 else source.split_indices[index - 1]
        end = min(stop, source.split_indices[index])
        if cache is None or (cache_sources is not None and
                             _cache_path_key(source._filenames[index]) not in cache_sources):
            pieces.append(source.get_file(index)[start - previous:end - previous])
        else:
            payload = cache.read(source._filenames[index], (start-previous)*reader.n_chan_bin*2,
                                 (end-previous)*reader.n_chan_bin*2,sequential=sequential)
            pieces.append(np.frombuffer(payload,dtype=np.int16).reshape(-1,reader.n_chan_bin))
        start = end
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


class Int16Reader:
    def __init__(self, read_cache=None, cache_sources=None):
        configure_cupy_cache()
        import cupy as cp
        self.cp = cp
        self.module = cp.RawModule(code=Path(__file__).with_suffix('.cu').read_text(),
                                   options=('--std=c++11', '--fmad=false'))
        self.kernel = self.module.get_function('decode_transpose_pad_i16')
        self.states = weakref.WeakKeyDictionary()
        self.events = deque()
        self.timing_event_limit = 64
        self.peak_timing_batches = 0
        self.h2d_cuda_ms = self.decode_cuda_ms = 0.
        self.calls = self.transferred_bytes = self.buffers_created = 0
        self.host_read_copy_seconds = 0.
        self.max_pinned_buffer_bytes = 0
        self.read_cache = read_cache
        self.cache_sources = cache_sources

    def read(self, reader, ibatch, return_inds=False):
        import torch
        if (np.dtype(reader.dtype) != np.dtype('int16') or
            torch.device(reader.device).type != 'cuda' or reader.writable):
            raise ValueError('Native reader requires read-only INT16 data and CUDA')
        if not isinstance(ibatch, (int, np.integer)) or not 0 <= ibatch < reader.n_batches:
            raise IndexError('Batch index out of bounds')
        original_batch = int(ibatch)*int(reader.batch_downsampling)
        bstart, bend = reader._get_batch_edges(original_batch)
        channels = int(reader.n_chan_bin)
        output_samples = int(reader.NT+2*reader.nt)
        scale = channel_calibration(reader.scale, channels, 1.)
        offset = channel_calibration(reader.shift, channels, 0.)
        state = self.states.get(reader)
        stream = torch.cuda.current_stream(reader.device)
        if state is None:
            host = torch.empty((output_samples, channels), dtype=torch.int16, pin_memory=True)
            state = dict(host=host, device=torch.empty_like(host, device=reader.device),
                         scale=torch.as_tensor(scale, device=reader.device),
                         offset=torch.as_tensor(offset, device=reader.device),
                         scale_cpu=scale.copy(), offset_cpu=offset.copy(), transfer_done=None, decode_done=None)
            self.states[reader] = state
            self.buffers_created += 1
            self.max_pinned_buffer_bytes = max(self.max_pinned_buffer_bytes, host.numel()*2)
        elif not np.array_equal(scale, state['scale_cpu']) or not np.array_equal(offset, state['offset_cpu']):
            raise ValueError('Calibration changed while a buffered reader was active')
        if state['transfer_done'] is not None:
            # The CPU may reuse pinned memory only after the last H2D completes.
            state['transfer_done'].synchronize()
            # Covers callers switching CUDA streams between reads.
            stream.wait_event(state['decode_done'])
        begin = time.perf_counter()
        sequential = original_batch == state.get('previous_batch', -2)+1
        raw = source_slice(reader, bstart, bend, self.read_cache, sequential,
                           self.cache_sources)
        state['previous_batch'] = original_batch
        nsamp = len(raw)
        left_pad = int(reader.nt) if original_batch == 0 else 0
        if nsamp <= 0 or nsamp+left_pad > output_samples:
            raise ValueError('Unexpected source batch shape')
        if original_batch not in (0, reader.n_batches-1) and nsamp != output_samples:
            raise ValueError('Incomplete interior batch')
        valid_stop = nsamp+left_pad if original_batch == 0 else output_samples
        # No CPU floating-point expansion and no retained whole-file mapping.
        np.copyto(state['host'].numpy()[:nsamp], raw, casting='no')
        del raw
        self.host_read_copy_seconds += time.perf_counter()-begin
        transfer_start = torch.cuda.Event(enable_timing=True)
        transfer_done = torch.cuda.Event(enable_timing=True)
        decode_done = torch.cuda.Event(enable_timing=True)
        transfer_start.record(stream)
        state['device'][:nsamp].copy_(state['host'][:nsamp], non_blocking=True)
        for tensor in (state['device'], state['scale'], state['offset']):
            tensor.record_stream(stream)
        transfer_done.record(stream)
        X = torch.empty((channels, output_samples), dtype=torch.float32, device=reader.device)
        arguments = (*[np.uint64(v.data_ptr()) for v in (state['device'], X, state['scale'], state['offset'])],
                     *map(np.int64, (nsamp, channels, output_samples, left_pad, valid_stop)))
        with self.cp.cuda.Device(X.device.index):
            self.kernel(((channels+31)//32, (output_samples+31)//32), (32, 8), arguments,
                        stream=self.cp.cuda.Stream.from_external(stream))
        decode_done.record(stream)
        state['transfer_done'], state['decode_done'] = transfer_done, decode_done
        self._record_timing(transfer_start, transfer_done, decode_done)
        self.calls += 1
        self.transferred_bytes += nsamp*channels*2
        # Preserve installed Kilosort's returned coordinate convention.
        if original_batch == 0:
            bstart = reader.imin-reader.nt
        elif original_batch == reader.n_batches-1:
            bend += reader.nt
        return (X, [bstart, bend]) if return_inds else X

    def _drain_timings(self, *, wait=False):
        while self.events:
            start, transferred, decoded = self.events[0]
            if wait:
                decoded.synchronize()
            elif not decoded.query():
                break
            self.h2d_cuda_ms += start.elapsed_time(transferred)
            self.decode_cuda_ms += transferred.elapsed_time(decoded)
            self.events.popleft()

    def _record_timing(self, start, transferred, decoded):
        self._drain_timings()
        if len(self.events) >= self.timing_event_limit:
            # Bound diagnostic handles even if callers enqueue many batches
            # across different CUDA streams before prior events complete.
            self.events[0][2].synchronize()
            self._drain_timings()
        self.events.append((start, transferred, decoded))
        self.peak_timing_batches = max(self.peak_timing_batches, len(self.events))

    def stats(self):
        self._drain_timings(wait=True)
        return dict(calls=self.calls, transferred_bytes=self.transferred_bytes,
                    buffers_created=self.buffers_created, max_pinned_buffer_bytes=self.max_pinned_buffer_bytes,
                    host_read_copy_seconds=self.host_read_copy_seconds,
                    h2d_cuda_seconds=self.h2d_cuda_ms/1000,
                    decode_cuda_seconds=self.decode_cuda_ms/1000,
                    timing_event_limit=self.timing_event_limit,
                    peak_pending_timing_batches=self.peak_timing_batches,
                    scope='Summed CUDA events for H2D/decode; host mmap read/copy excludes buffer-reuse waits. No disk/compute prefetch.')

    def close(self):
        self._drain_timings(wait=True)
        self.states.clear()
        if self.read_cache is not None:
            import json
            self.read_cache.close()
            print('TERASORT_READ_CACHE ' + json.dumps(self.read_cache.stats()), flush=True)


@contextmanager
def native_int16_reader(filename, *, read_cache_dir=None, read_cache_mb=4096,
                        read_cache_slots=3, read_cache_workers=4, read_cache_retain=False,
                        read_cache_retain_gb=0.):
    from kilosort import io
    if importlib.metadata.version('kilosort') != '4.1.7':
        raise RuntimeError('INT16 adapter validated only against Kilosort 4.1.7')
    original = io.BinaryRWFile.padded_batch_to_torch
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != EXPECTED_READER_SHA256:
        raise RuntimeError('Kilosort reader changed; refusing unverified substitution')
    named = tuple(Path(name) for name in
                  (filename if isinstance(filename, (list, tuple)) else [filename]))
    target = tuple(path.resolve() for path in named)
    cache = None
    cache_sources = None
    if read_cache_dir is not None:
        from .source_location import is_network_source
        remote_indices = [index for index, (name, path) in enumerate(zip(named, target))
                          if is_network_source(name) or is_network_source(path)]
        if remote_indices:
            from .read_cache import ReadAheadCache
            # Kilosort may retain a mapped-drive spelling even when resolve()
            # yields UNC. Accept both spellings without resolving every batch.
            cache_sources = {_cache_path_key(path)
                             for index in remote_indices
                             for path in (named[index], target[index])}
            cache = ReadAheadCache(read_cache_dir, block_mb=read_cache_mb,
                                   slots=read_cache_slots, workers=read_cache_workers,
                                   retain_sources=[target[i] for i in remote_indices] if read_cache_retain else None,
                                   retain_gb=read_cache_retain_gb,
                                   wrap_source=target[0] if len(target) == 1 else None)
            print(f'TERASORT_READ_CACHE_NETWORK_SOURCES {len(remote_indices)}/{len(target)}', flush=True)
            if cache.retention is not None:
                import json
                print('TERASORT_READ_CACHE_RETENTION ' + json.dumps(cache.retention), flush=True)
        else:
            print('TERASORT_READ_CACHE_BYPASS all sources are local', flush=True)
    try:
        backend = Int16Reader(cache, cache_sources)
    except BaseException:
        if cache is not None:
            cache.close()
        raise

    def replacement(reader, ibatch, return_inds=False):
        path = reader.filename
        paths = path if isinstance(path, (list, tuple)) else [path]
        if path is not None and tuple(Path(name).resolve() for name in paths) == target:
            return backend.read(reader, ibatch, return_inds)
        return original(reader, ibatch, return_inds)

    io.BinaryRWFile.padded_batch_to_torch = replacement
    try:
        yield backend
    finally:
        io.BinaryRWFile.padded_batch_to_torch = original
        backend.close()
