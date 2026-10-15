"""SCB v0.1/v0.2: indexed candidates, optionally with local waveform caches.

Only detector-defined local extrema above the manifest floor are retained.
There is no spatial exclusion and no unit assignment. Lower floors, changed
filtering/noise calibration, and full-recording fitting require the raw source.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import h5py
from .waveforms import WaveformSpec, WaveformBatch, encode_waveforms, waveform_mask

EVENT_DTYPE = np.dtype([('sample_index', '<i8'), ('channel_index', '<u4'),
                        ('amplitude_uv', '<f4'), ('snr', '<f4')])
BLOCK_DTYPE = np.dtype([('start_sample', '<i8'), ('stop_sample', '<i8'),
                       ('row_start', '<i8'), ('row_stop', '<i8')])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class BankWriter:
    """Single writer, atomic publication; incomplete files have .partial suffix.

    One file per source segment/probe. A collection can reference many files.
    No append-to-completed-file or concurrent readers of incomplete files.
    """
    def __init__(self, path, manifest, noise_uv, channel_positions=None, waveform_spec=None):
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + '.partial')
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        noise_uv = np.asarray(noise_uv, np.float32)
        if noise_uv.ndim != 1 or not len(noise_uv) or not np.isfinite(noise_uv).all() or np.any(noise_uv <= 0):
            raise ValueError('Positive finite per-channel noise required')
        manifest = dict(manifest)
        if not np.isfinite(manifest['floor_snr']) or manifest['floor_snr'] <= 0:
            raise ValueError('Invalid detection floor')
        if channel_positions is None:
            channel_positions = np.full((len(noise_uv), 3), np.nan, np.float32)
        channel_positions = np.asarray(channel_positions, np.float32)
        if channel_positions.shape != (len(noise_uv), 3):
            raise ValueError('Expected channel x 3 coordinates')
        if waveform_spec is not None and len(waveform_spec.channel_index) != len(noise_uv):
            raise ValueError('Waveform map must cover all recording channels')
        self.waveform_spec = waveform_spec
        manifest.pop('waveforms', None)
        if waveform_spec is not None:
            manifest['waveforms'] = waveform_spec.metadata()
        manifest.update(format='SCB', version='0.2' if waveform_spec is not None else '0.1', spatial_exclusion=False,
                        event_definition='absolute normalized temporal local maximum; earliest temporal tie; strict score > floor',
                        amplitude_definition='signed preprocessed voltage in microvolts',
                        location_definition='recorded channel index; channel coordinates are not neuron localization',
                        completeness='detector-defined events only; no guarantee of biological spike completeness')
        self.manifest = manifest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.partial, 'x')
        self.f.attrs['manifest_json'] = json.dumps(manifest, sort_keys=True)
        self.f.attrs['complete'] = False
        self.f.create_dataset('noise_uv', data=noise_uv)
        self.f.create_dataset('channel_positions_um', data=channel_positions)
        self.events = self.f.create_dataset('events', shape=(0,), maxshape=(None,),
                                           dtype=EVENT_DTYPE, chunks=(65536,), compression='lzf', shuffle=True)
        self.blocks = self.f.create_dataset('blocks', shape=(0,), maxshape=(None,),
                                           dtype=BLOCK_DTYPE, chunks=(1024,))
        if waveform_spec is not None:
            spec = waveform_spec
            g = self.f.create_group('waveforms')
            g.create_dataset('channel_index', data=spec.channel_index)
            # At most ~1 MiB per waveform chunk, capped at 512 event rows.
            row_bytes = spec.n_samples*spec.width*spec.dtype.itemsize
            chunk_rows = max(1, min(512, 1048576//row_bytes))
            g.create_dataset('data', shape=(0, spec.n_samples, spec.width),
                             maxshape=(None, spec.n_samples, spec.width), dtype=spec.dtype,
                             chunks=(chunk_rows, spec.n_samples, spec.width), compression='lzf', shuffle=True)
            for name in ('valid_start', 'valid_stop'):
                g.create_dataset(name, shape=(0,), maxshape=(None,), dtype='<i4', chunks=(65536,))
        self.stop = None

    def append(self, events, start_sample, stop_sample, waveforms=None,
               waveform_valid_start=None, waveform_valid_stop=None):
        events = np.asarray(events, dtype=EVENT_DTYPE)
        if not start_sample < stop_sample or (self.stop is not None and self.stop != start_sample):
            raise ValueError('Blocks must be positive, contiguous and ordered within one segment')
        if len(events):
            t, c = events['sample_index'], events['channel_index']
            if t.min() < start_sample or t.max() >= stop_sample or c.max() >= len(self.f['noise_uv']):
                raise ValueError('Event outside block or channel map')
            if np.any(t[1:] < t[:-1]) or np.any((t[1:] == t[:-1]) & (c[1:] <= c[:-1])):
                raise ValueError('Events must be sorted and unique by sample/channel')
            if not np.isfinite(events['snr']).all() or not np.isfinite(events['amplitude_uv']).all():
                raise ValueError('Nonfinite event data')
            if np.any(events['snr'] <= self.manifest['floor_snr']):
                raise ValueError('Event at or below detector floor')
            expected = np.abs(events['amplitude_uv']) / self.f['noise_uv'][:][c]
            if not np.allclose(events['snr'], expected, rtol=2e-6, atol=1e-6):
                raise ValueError('Amplitude/SNR calibration mismatch')
        if self.waveform_spec is None:
            if any(v is not None for v in (waveforms, waveform_valid_start, waveform_valid_stop)):
                raise ValueError('Waveform payload requires a waveform spec')
        else:
            spec = self.waveform_spec
            values = np.asarray(waveforms, np.float32)
            valid_start, valid_stop = np.asarray(waveform_valid_start), np.asarray(waveform_valid_stop)
            if values.shape != (len(events), spec.n_samples, spec.width):
                raise ValueError('One local waveform per event row required')
            if (valid_start.shape != (len(events),) or valid_stop.shape != (len(events),)
                or valid_start.dtype.kind not in 'iu' or valid_stop.dtype.kind not in 'iu'
                or np.any(valid_start < 0) or np.any(valid_start > spec.pre_samples)
                or np.any(valid_stop <= spec.pre_samples) or np.any(valid_stop > spec.n_samples)):
                raise ValueError('Invalid waveform sample bounds; center must be present')
            if not np.array_equal(values[:, spec.pre_samples, 0], events['amplitude_uv']):
                raise ValueError('Waveform anchor voltage differs from saved event amplitude')
            valid = waveform_mask(valid_start, valid_stop,
                                  spec.channel_index[events['channel_index']], spec.n_samples)
            if np.any(values[~valid] != 0):
                raise ValueError('Missing waveform samples/channels must store zero')
            encoded = encode_waveforms(values, spec)
        a, b = len(self.events), len(self.events) + len(events)
        self.events.resize((b,))
        self.events[a:b] = events
        if self.waveform_spec is not None:
            g = self.f['waveforms']
            g['data'].resize((b, spec.n_samples, spec.width))
            g['data'][a:b] = encoded
            for name, value in [('valid_start', valid_start), ('valid_stop', valid_stop)]:
                g[name].resize((b,))
                g[name][a:b] = value
        n = len(self.blocks)
        self.blocks.resize((n + 1,))
        self.blocks[n] = (start_sample, stop_sample, a, b)
        self.f.flush()
        self.stop = stop_sample

    def close(self, success=True):
        if self.f:
            self.f.attrs['complete'] = bool(success)
            self.f.flush()
            self.f.close()
            self.f = None
            if success:
                os.replace(self.partial, self.path)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(success=kind is None)


class CandidateBank:
    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(path, 'r') as f:
            if not f.attrs.get('complete', False):
                raise ValueError('Incomplete candidate bank')
            self.manifest = json.loads(f.attrs['manifest_json'])
            if self.manifest.get('format') != 'SCB' or self.manifest.get('version') not in ('0.1', '0.2'):
                raise ValueError('Unsupported candidate format/version')
            self.n_events = len(f['events'])
            self.n_channels = len(f['noise_uv'])
            self.blocks = f['blocks'][:]
            if f['events'].dtype != EVENT_DTYPE or f['blocks'].dtype != BLOCK_DTYPE:
                raise ValueError('Unexpected event/index schema')
            b = self.blocks
            if len(b):
                if (b['row_start'][0] != 0 or b['row_stop'][-1] != self.n_events
                    or np.any(b['row_stop'] < b['row_start'])
                    or np.any(b['stop_sample'] <= b['start_sample'])
                    or np.any(b['row_start'][1:] != b['row_stop'][:-1])
                    or np.any(b['start_sample'][1:] != b['stop_sample'][:-1])):
                    raise ValueError('Invalid block index')
            elif self.n_events:
                raise ValueError('Missing event index')
            self.waveform_spec = None
            if self.manifest['version'] == '0.2':
                if 'waveforms' not in f or 'waveforms' not in self.manifest:
                    raise ValueError('Missing waveform cache/schema')
                g, m = f['waveforms'], self.manifest['waveforms']
                if any(k not in g for k in ['data', 'channel_index', 'valid_start', 'valid_stop']):
                    raise ValueError('Missing waveform dataset')
                spec = WaveformSpec(g['channel_index'][:], m['pre_samples'], m['post_samples'],
                                    m['scale_uv'], m['storage_dtype'])
                if (m != spec.metadata() or len(spec.channel_index) != self.n_channels
                    or g['data'].shape != (self.n_events, spec.n_samples, spec.width)
                    or g['data'].dtype != spec.dtype
                    or any(g[k].shape != (self.n_events,) or g[k].dtype != np.dtype('<i4')
                           for k in ['valid_start', 'valid_stop'])):
                    raise ValueError('Invalid waveform schema or event-row alignment')
                self.waveform_spec = spec
            elif 'waveforms' in f or 'waveforms' in self.manifest:
                raise ValueError('Waveforms require SCB 0.2')

    def iter_indexed_query(self, snr, start_sample=None, stop_sample=None, channels=None, polarity='both'):
        """Stream (event row IDs, events); IDs join the optional waveform cache."""
        if not np.isfinite(snr) or snr < self.manifest['floor_snr']:
            raise ValueError('Requested threshold is below saved floor or nonfinite; reread raw voltage')
        if polarity not in ('both', 'neg', 'pos'):
            raise ValueError('Unknown polarity')
        if start_sample is not None and stop_sample is not None and start_sample > stop_sample:
            raise ValueError('Reversed time interval')
        if channels is not None:
            channels = np.asarray(channels, dtype=int)
            if np.any(channels < 0) or np.any(channels >= self.n_channels):
                raise ValueError('Unknown channel')
        with h5py.File(self.path, 'r') as f:
            for block in self.blocks:
                if start_sample is not None and block['stop_sample'] <= start_sample:
                    continue
                if stop_sample is not None and block['start_sample'] >= stop_sample:
                    continue
                events = f['events'][int(block['row_start']):int(block['row_stop'])]
                keep = events['snr'] > snr
                if start_sample is not None:
                    keep &= events['sample_index'] >= start_sample
                if stop_sample is not None:
                    keep &= events['sample_index'] < stop_sample
                if channels is not None:
                    keep &= np.isin(events['channel_index'], channels)
                if polarity != 'both':
                    keep &= events['amplitude_uv'] < 0 if polarity == 'neg' else events['amplitude_uv'] > 0
                yield np.flatnonzero(keep).astype(np.int64)+block['row_start'], events[keep]

    def iter_query(self, snr, **kwargs):
        """Stream selected blocks without loading the full bank into memory."""
        for _, events in self.iter_indexed_query(snr, **kwargs):
            yield events

    def iter_waveforms(self, snr, batch_size=512, **kwargs):
        """CPU-only bounded waveform query; invalid samples/slots decode to NaN.

        Channel filters select candidate anchors, not the cached neighbor slots.
        Returned row IDs refer to immutable event rows within this bank.
        """
        spec = self.waveform_spec
        if spec is None:
            raise ValueError('This bank has no waveform cache')
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('Positive integer waveform batch size required')
        with h5py.File(self.path, 'r') as f:
            g = f['waveforms']
            for rows, events in self.iter_indexed_query(snr, **kwargs):
                i = 0
                while i < len(rows):
                    # Bound the source row span as well as the returned row count.
                    j = int(np.searchsorted(rows, (rows[i]//batch_size+1)*batch_size))
                    a, b = int(rows[i]), int(rows[j-1])+1
                    selected = rows[i:j]-a
                    values = g['data'][a:b][selected].astype(np.float32)*np.float32(spec.scale_uv)
                    starts = g['valid_start'][a:b][selected]
                    stops = g['valid_stop'][a:b][selected]
                    channels = spec.channel_index[events['channel_index'][i:j]]
                    if (np.any(starts < 0) or np.any(starts > spec.pre_samples)
                        or np.any(stops <= spec.pre_samples) or np.any(stops > spec.n_samples)):
                        raise ValueError('Invalid cached waveform bounds')
                    values[~waveform_mask(starts, stops, channels, spec.n_samples)] = np.nan
                    yield WaveformBatch(rows[i:j], events[i:j], values, channels, starts, stops)
                    i = j

    def query(self, snr, **kwargs):
        """Convenience materialization for small selections; use iter_query at scale."""
        chunks = list(self.iter_query(snr, **kwargs))
        return np.concatenate(chunks) if chunks else np.empty(0, EVENT_DTYPE)


def events_from_indices(indices, voltage, noise_uv, sample_offset=0):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    nt, nc = voltage.shape
    t, c = np.divmod(indices, nc)
    out = np.empty(len(indices), EVENT_DTYPE)
    out['sample_index'] = t + np.int64(sample_offset)
    out['channel_index'] = c
    out['amplitude_uv'] = voltage[t, c]
    out['snr'] = np.abs(voltage[t, c]) / noise_uv[c]
    return out
