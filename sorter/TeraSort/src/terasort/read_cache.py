"""Bounded SSD read-ahead for sequential passes, exact reads for sparse passes."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
import shutil
import threading
import json
import math


DEFAULT_READ_CACHE_SLOTS = 3


def retention_plan(source_sizes, block_bytes, slots, free_bytes, retain_bytes=0):
    """Reserve a source prefix plus rolling tail, leaving 10% (at least 64 MiB) free."""
    reserve = max(64*1024**2, free_bytes//10)
    available = max(0, free_bytes-reserve)
    total = sum(source_sizes)
    requested = min(total, retain_bytes) if retain_bytes else total
    if requested == total and total <= available:
        budget = total
        rolling = 0
    else:
        rolling = block_bytes*slots
        if available < rolling:
            raise OSError('Insufficient free space for rolling cache plus free-space reserve')
        budget = min(requested, available-rolling)
    counts, pinned, remaining = [], 0, budget
    for size in source_sizes:
        count = (size+block_bytes-1)//block_bytes if size <= remaining else remaining//block_bytes
        used = min(size, count*block_bytes)
        counts.append(count)
        pinned += used
        # Preserve acquisition order; do not skip an unretained block to keep
        # a smaller block from a later source.
        remaining = remaining-used if used == size else 0
    return dict(prefix_blocks=counts, retained_bytes=pinned,
                disk_budget_bytes=pinned+rolling, reserve_bytes=reserve,
                all_sources_retained=pinned == total)


def retry_with_current_cache_defaults(request):
    """Apply newer non-scientific read-ahead defaults to a retry."""
    updated = dict(request)
    slots = updated.get('read_cache_slots')
    if updated.get('read_cache_dir') and type(slots) is int and slots < DEFAULT_READ_CACHE_SLOTS:
        updated['read_cache_slots'] = DEFAULT_READ_CACHE_SLOTS
    return updated


class ReadAheadCache:
    def __init__(self, directory, *, block_mb=256, slots=DEFAULT_READ_CACHE_SLOTS,
                 workers=4, wrap_source=None, retain_sources=None, retain_gb=0.):
        if type(block_mb) is not int or not 1 <= block_mb <= 4096:
            raise ValueError('Cache block must be 1–4096 MiB')
        if type(slots) is not int or not 2 <= slots <= 16:
            raise ValueError('Cache slots must be 2–16')
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('Download workers must be 1–16')
        if type(retain_gb) not in (int, float) or not math.isfinite(retain_gb) or retain_gb < 0:
            raise ValueError('Retention budget must be finite nonnegative GiB; zero means automatic')
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.retained_sources = {}
        if retain_sources is not None:
            for name in retain_sources:
                source = Path(name).resolve(strict=True)
                info = source.stat()
                if not source.is_file() or info.st_size <= 0:
                    raise ValueError('Retained cache requires nonempty source files')
                self.retained_sources[source] = (info.st_size, info.st_mtime_ns)
            if not self.retained_sources:
                raise ValueError('Retained cache requires source paths')
        free = shutil.disk_usage(parent).free
        self.retain_prefix = {}
        self.retention = None
        self.disk_budget = block_mb*1024**2*slots
        if self.retained_sources:
            self.retention = retention_plan([size for size, _ in self.retained_sources.values()],
                                           block_mb*1024**2, slots, free, int(retain_gb*1024**3))
            self.retain_prefix = dict(zip(self.retained_sources, self.retention['prefix_blocks']))
            self.disk_budget = self.retention['disk_budget_bytes']
        elif free < self.disk_budget + 64*1024**2:
            raise OSError('Insufficient free space for the bounded read cache')
        self.root = Path(tempfile.mkdtemp(prefix='terasort-read-', dir=parent)).resolve()
        self.block_bytes, self.slots, self.workers = block_mb*1024**2, slots, workers
        # Kilosort revisits a single source in later passes. Near EOF, reuse
        # slots freed by consumed blocks to stage the start of the next pass.
        # Multi-file sessions cannot safely infer the next source here.
        self.wrap_source = (Path(wrap_source).resolve()
                            if wrap_source is not None and not self.retained_sources else None)
        self.wrap_keys = set()
        self.wrap_scheduled = 0
        self.last_sequential_index = None
        self.last_sequential_path = None
        self.primed_path = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='terasort-download')
        self.files, self.pending = OrderedDict(), OrderedDict()
        # A single block downloader runs continuously through the queued work.
        # The third slot holds a ready block while the next one downloads.
        self.prefetch_depth = slots-1

        # Whitening samples distant batches. Keep those exact ranges in RAM
        # while concurrent SMB reads overlap the current GPU batch. Never
        # prefetch a whole 4 GiB sequential cache block for a sparse request.
        self.sparse_depth = min(workers, 4)
        self.sparse_max_range_bytes = 64*1024**2
        self.sparse_pool = ThreadPoolExecutor(max_workers=self.sparse_depth,
                                              thread_name_prefix='terasort-sparse')
        self.sparse_futures = OrderedDict()
        self.sparse_history = []
        self.sparse_served_bytes = 0
        self.sparse_wait_seconds = 0.
        self.direct, self.direct_path = None, None
        self.counter = 0
        self.network_bytes = self.direct_bytes = self.cache_bytes = 0
        self.wait_seconds = 0.
        self.download_lock = threading.Lock()
        self.stop_downloads = threading.Event()
        self.report_time = time.monotonic()
        self.report_bytes = 0
        self.reporter = threading.Thread(target=self._report_loop,
                                         name='terasort-cache-status', daemon=True)
        self.reporter.start()

    def _record_download(self, count):
        with self.download_lock:
            self.network_bytes += count

    def _retained(self, key):
        return key[1] < self.retain_prefix.get(key[0], 0)

    def _rolling_count(self):
        return sum(not self._retained(key) for key in self.files) + sum(
            not self._retained(key) for key in self.pending)

    def _report_loop(self):
        while not self.stop_downloads.wait(10):
            with self.download_lock:
                now = time.monotonic()
                downloaded = self.network_bytes
                elapsed = max(now-self.report_time, .001)
                rate = (downloaded-self.report_bytes)/elapsed/1024**2
                self.report_time, self.report_bytes = now, downloaded
            try:
                pending = tuple(self.pending.values())
            except RuntimeError:
                pending = ()
            ready = len(self.files) + sum(future.done() and not future.cancelled()
                                          for future in pending)
            try:
                print('TERASORT_READ_CACHE_PROGRESS ' + json.dumps({
                    'downloaded_bytes': downloaded,
                    'download_mib_s': round(rate, 1),
                    'workers': self.workers,
                    'slots_used': self._rolling_count(),
                    'slots_total': self.slots,
                    'retained_blocks': sum(self._retained(key) for key in self.files),
                    'retain_for_passes': bool(self.retained_sources),
                    'retained_budget_bytes': self.retention['retained_bytes'] if self.retention else 0,
                    'ready_blocks': ready,
                    'consumer_wait_seconds': round(self.wait_seconds, 1)}), flush=True)
            except (OSError, RuntimeError):
                pass

    def prepare_pass(self, path):
        """Start the first pass blocks before the first dense GPU batch."""
        path = Path(path).resolve()
        while self.future is not None and self.future[1].done():
            self._settle()
        blocks = (path.stat().st_size+self.block_bytes-1)//self.block_bytes
        prefix = self.retain_prefix.get(path, 0)
        # On later passes, fill the rolling tail while computation replays the
        # retained prefix. Do not wait until the prefix has been consumed.
        first = prefix if prefix and all((path, i) in self.files for i in range(prefix)) else 0
        count = min(self.slots if first else self.prefetch_depth, blocks-first)
        if count <= 0:
            return
        desired = {(path, index) for index in range(first, first+count)}
        for key, future in list(self.pending.items()):
            if key not in desired and future.cancel():
                self.pending.pop(key)
                self.wrap_keys.discard(key)
        while self.future is not None and self.future[1].done():
            self._settle()
        for old in list(self.files):
            if old not in desired and not self._retained(old):
                self.files.pop(old).unlink()
                self.wrap_keys.discard(old)
        for index in range(first, first+count):
            self._schedule((path, index), first_needed=first, current=first)
        self.primed_path = path

    def _download(self, key, target):
        path, index = key
        before = path.stat()
        if self.retained_sources and self.retained_sources[path] != (before.st_size, before.st_mtime_ns):
            raise OSError('Source changed since retained cache was created')
        offset = index*self.block_bytes
        remaining = min(self.block_bytes, before.st_size-offset)
        if remaining <= 0:
            raise ValueError('Cache block outside source')
        partial = target.with_suffix('.partial')
        try:
            if self.stop_downloads.is_set():
                raise RuntimeError('Read cache is closing')
            with partial.open('xb') as dest:
                dest.truncate(remaining)

            chunk_bytes = 8*1024**2
            worker_count = min(self.workers, max(1, (remaining+chunk_bytes-1)//chunk_bytes))
            segment_bytes = (remaining+worker_count-1)//worker_count

            def copy_range(start, stop):
                with path.open('rb', buffering=0) as source, partial.open('r+b', buffering=0) as dest:
                    source.seek(offset+start)
                    dest.seek(start)
                    todo = stop-start
                    while todo:
                        if self.stop_downloads.is_set():
                            raise RuntimeError('Read cache is closing')
                        payload = source.read(min(chunk_bytes, todo))
                        if not payload:
                            raise OSError('Short network read')
                        view = memoryview(payload)
                        while view:
                            written = dest.write(view)
                            if not written:
                                raise OSError('Short local cache write')
                            view = view[written:]
                        todo -= len(payload)
                        self._record_download(len(payload))

            ranges = [(start, min(start+segment_bytes, remaining))
                      for start in range(0, remaining, segment_bytes)]
            if len(ranges) == 1:
                copy_range(*ranges[0])
            else:
                with ThreadPoolExecutor(max_workers=worker_count,
                                        thread_name_prefix='terasort-range') as workers:
                    futures = [workers.submit(copy_range, start, stop) for start, stop in ranges]
                    for future in futures:
                        future.result()
            after = path.stat()
            if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
                raise OSError('Source changed while downloading cache block')
            partial.replace(target)
            return target
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    @property
    def future(self):
        # Retained for cache diagnostics and callers inspecting the active job.
        return next(iter(self.pending.items()), None)

    def _settle(self):
        if not self.pending:
            return
        key, future = next(iter(self.pending.items()))
        begin = time.perf_counter()
        try:
            self.files[key] = future.result()
        except BaseException:
            self.wrap_keys.discard(key)
            raise
        finally:
            self.wait_seconds += time.perf_counter()-begin
            self.pending.pop(key)

    def _cancel_queued_downloads(self):
        # A sparse read or a new source invalidates speculative sequential
        # work. A download already running may finish; queued blocks are
        # cancelled before they consume more network bytes or disk space.
        for key, future in list(self.pending.items()):
            if future.cancel():
                self.pending.pop(key)
                self.wrap_keys.discard(key)

    def _schedule(self, key, *, first_needed=None, current=None, required=False):
        if key in self.pending or key in self.files:
            return True
        if self.retained_sources and len(self.pending) >= self.slots:
            return False
        while not self._retained(key) and self._rolling_count() >= self.slots:
            if required:
                victim = next((old for old in self.files if not self._retained(old)), None)
            else:
                # Never evict a block that this request or the next halo may
                # still read. Old blocks become reusable only after the next
                # request begins entirely beyond their boundary.
                victim = next((old for old in self.files if not self._retained(old) and old not in self.wrap_keys and (
                    old[0] != key[0] or old[1] < first_needed or
                    old[1] > current + self.prefetch_depth)), None)
            if victim is None:
                return False
            self.files.pop(victim).unlink()
            self.wrap_keys.discard(victim)
        target = self.root / f'{self.counter:012d}.block'
        self.counter += 1
        self.pending[key] = self.pool.submit(self._download, key, target)
        return True

    def _cached_range(self, path, start, stop):
        """Return an exact range only when every covering block is already ready."""
        keys = [(path, index) for index in range(start//self.block_bytes,
                                                (stop-1)//self.block_bytes+1)]
        if any(key not in self.files for key in keys):
            return None
        output = bytearray(stop-start)
        position = start
        for key in keys:
            offset = position % self.block_bytes
            count = min(stop-position, self.files[key].stat().st_size-offset)
            if count <= 0:
                raise OSError('Source range exceeds cached file')
            with self.files[key].open('rb') as handle:
                handle.seek(offset)
                payload = handle.read(count)
            if len(payload) != count:
                raise OSError('Incomplete local cache read')
            output[position-start:position-start+count] = payload
            position += count
        if position != stop:
            raise OSError('Source range exceeds cached file')
        self.cache_bytes += len(output)
        return output

    def _read_sparse_range(self, path, start, stop):
        before = path.stat()
        with path.open('rb', buffering=0) as source:
            source.seek(start)
            payload = source.read(stop-start)
        if len(payload) != stop-start:
            raise OSError('Short sparse source read')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError('Source changed during sparse prefetch')
        self._record_download(len(payload))
        return payload

    def _reset_sparse(self):
        for future in self.sparse_futures.values():
            future.cancel()
        self.sparse_futures.clear()
        self.sparse_history.clear()

    def _schedule_sparse(self, path, start, stop):
        current = (path, start, stop)
        self.sparse_history.append(current)
        self.sparse_history = self.sparse_history[-3:]
        if len(self.sparse_history) < 3:
            return
        first, previous, current = self.sparse_history
        length = stop-start
        stride = start-previous[1]
        if (first[0] != path or previous[0] != path or
            previous[2]-previous[1] != length or
            stride != previous[1]-first[1] or
            stride <= length or length > self.sparse_max_range_bytes):
            return
        size = path.stat().st_size
        for step in range(1, self.sparse_depth+1):
            future_start = start+step*stride
            future_stop = future_start+length
            if future_stop > size:
                break
            key = (path, future_start, future_stop)
            if key not in self.sparse_futures and len(self.sparse_futures) < self.sparse_depth:
                self.sparse_futures[key] = self.sparse_pool.submit(
                    self._read_sparse_range, *key)

    def read(self, path, start, stop, *, sequential=False):
        path = Path(path).resolve()
        # Kilosort's concatenated-file offsets may be NumPy integer scalars.
        start, stop = int(start), int(stop)
        if not 0 <= start < stop:
            raise ValueError('Invalid byte range')
        if self.retained_sources:
            info = path.stat()
            if self.retained_sources.get(path) != (info.st_size, info.st_mtime_ns):
                raise OSError('Unknown or changed retained-cache source')
            if stop > info.st_size:
                raise OSError('Source range exceeds retained source')
        while self.future is not None and self.future[1].done():
            self._settle()
        if not sequential and not self._retained((path, start//self.block_bytes)):
            # Kilosort reads batch zero directly at a pass boundary. Keep
            # completed and queued wrapped blocks for the following sequential
            # batches; unrelated sparse reads still cancel speculation.
            next_pass_start = (
                path == self.wrap_source and bool(self.wrap_keys)
                and self.last_sequential_path == path
                and self.last_sequential_index is not None
                and self.last_sequential_index >=
                    (path.stat().st_size-1)//self.block_bytes
                and start < self.block_bytes)
            primed_start = path == self.primed_path and start < self.block_bytes
            if not next_pass_start and not primed_start:
                self._cancel_queued_downloads()
            if primed_start:
                self.primed_path = None
            self.wrap_keys.clear()
            self.last_sequential_index = None
            self.last_sequential_path = None
            cached = self._cached_range(path, start, stop)
            if cached is not None:
                self._reset_sparse()
                return cached
            # Strided calibration uses only requested batches, plus a bounded
            # prediction of the next exact batch ranges after stride stabilizes.
            key = (path, start, stop)
            future = self.sparse_futures.pop(key, None)
            if future is not None:
                begin = time.perf_counter()
                payload = future.result()
                self.sparse_wait_seconds += time.perf_counter()-begin
                self.sparse_served_bytes += len(payload)
            else:
                if self.sparse_futures:
                    self._reset_sparse()
                if self.direct_path != path:
                    if self.direct is not None:
                        self.direct.close()
                    self.direct = path.open('rb',buffering=0)
                    self.direct_path = path
                self.direct.seek(start)
                payload = self.direct.read(stop-start)
                self.direct_bytes += len(payload)
            if len(payload) != stop-start:
                raise OSError('Short direct source read')
            self._schedule_sparse(path, start, stop)
            return payload
        if self.sparse_history or self.sparse_futures:
            self._reset_sparse()
        if path != self.last_sequential_path:
            self.wrap_keys.clear()
            self.last_sequential_index = None
            self.last_sequential_path = path
        output = bytearray(stop-start)
        position = start
        first_needed = start//self.block_bytes
        source_size = path.stat().st_size
        while position < stop:
            index = position//self.block_bytes
            if path == self.wrap_source and self.last_sequential_index is not None:
                if index == 0 and self.last_sequential_index > self.prefetch_depth:
                    self.wrap_keys.clear()
            self.last_sequential_index = index
            key = (path,index)
            if key not in self.files:
                if key in self.pending:
                    while key not in self.files:
                        self._settle()
                else:
                    # A new pass or file may leave unrelated downloads in
                    # flight. Cancel queued work, then finish at most the
                    # currently running block before the demanded one.
                    self._cancel_queued_downloads()
                    while self.pending:
                        self._settle()
                    self._schedule(key, required=True)
                    self._settle()
            local = self.files[key]
            self.files.move_to_end(key)
            count = min(stop-position,local.stat().st_size-position%self.block_bytes)
            if count <= 0:
                raise OSError('Source range exceeds cached file')
            with local.open('rb') as handle:
                handle.seek(position%self.block_bytes)
                payload = handle.read(count)
            if len(payload) != count:
                raise OSError('Incomplete local cache read')
            output[position-start:position-start+count] = payload
            self.cache_bytes += count
            position += count
            for ahead in (range(1, self.prefetch_depth+1) if sequential else ()):
                next_index = index+ahead
                if next_index*self.block_bytes >= source_size:
                    break
                self._schedule((path,next_index), first_needed=first_needed,
                               current=index)
            if path == self.wrap_source:
                last_index = (source_size-1)//self.block_bytes
                # Only speculative work within the same bounded slot budget.
                # Keep wrapped blocks protected until this pass ends so later
                # EOF reads cannot evict the next pass's start.
                wrap_count = min(last_index + 1, self.prefetch_depth,
                                 max(0, index+self.prefetch_depth-last_index))
                for next_index in range(wrap_count):
                    wrapped = (path, next_index)
                    if wrapped in self.files or wrapped in self.pending:
                        self.wrap_keys.add(wrapped)
                    elif self._schedule(wrapped, first_needed=first_needed,
                                        current=index):
                        self.wrap_keys.add(wrapped)
                        self.wrap_scheduled += 1
        return output

    def stats(self):
        return dict(downloaded_bytes=self.network_bytes,direct_bytes=self.direct_bytes,
                    cache_served_bytes=self.cache_bytes,wait_seconds=self.wait_seconds,
                    disk_budget_bytes=self.disk_budget,
                    retain_for_passes=bool(self.retained_sources),
                    retention=self.retention,
                    read_buffer_bytes=8*1024**2,download_workers=self.workers,
                    sequential_prefetch_depth=self.prefetch_depth,
                    sparse_prefetch_served_bytes=self.sparse_served_bytes,
                    sparse_prefetch_wait_seconds=self.sparse_wait_seconds,
                    sparse_prefetch_budget_bytes=self.sparse_depth*self.sparse_max_range_bytes,
                    wrap_prefetch_enabled=self.wrap_source is not None,
                    wrap_blocks_scheduled=self.wrap_scheduled)

    def close(self):
        self.stop_downloads.set()
        self.reporter.join()
        self._reset_sparse()
        self.sparse_pool.shutdown(wait=True,cancel_futures=True)
        if self.direct is not None:
            self.direct.close()
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.pending.clear()
        # Only delete files created inside this instance's unique cache directory.
        for path in self.root.iterdir():
            if path.suffix in ('.block','.partial') and path.stem.isdigit():
                path.unlink()
        self.root.rmdir()
