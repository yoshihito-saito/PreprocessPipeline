"""Local multichannel waveform layout, extraction, and bounded encoding.

No deduplication or alignment is inferred: snippets stay centered on the saved
candidate sample, and all neighbors retain the same sample clock.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WaveformSpec:
    channel_index: np.ndarray  # recording channel x local slot, -1 padding
    pre_samples: int
    post_samples: int
    scale_uv: float = 0.05
    storage_dtype: str = 'int16'

    def __post_init__(self):
        raw = np.asarray(self.channel_index)
        if raw.ndim != 2 or not raw.size or raw.dtype.kind not in 'iu':
            raise ValueError('Expected integer recording-channel x local-slot map')
        nc, width = raw.shape
        if np.any(raw < -1) or np.any(raw >= nc):
            raise ValueError('Unknown waveform channel')
        for c, row in enumerate(raw):
            valid = row[row >= 0]
            if row[0] != c or len(valid) != len(np.unique(valid)):
                raise ValueError('Anchor must be first; local channels must be unique')
            if np.any(row[:len(valid)] < 0):
                raise ValueError('Missing channels must pad the end of each row')
        for n in (self.pre_samples, self.post_samples):
            if not isinstance(n, (int, np.integer)) or n < 0:
                raise ValueError('Waveform window must use nonnegative integer samples')
        if self.n_samples > np.iinfo(np.int32).max:
            raise ValueError('Waveform window too long')
        if self.storage_dtype not in ('int16', 'float32'):
            raise ValueError('Waveform storage must be int16 or float32')
        if not np.isfinite(self.scale_uv) or self.scale_uv <= 0:
            raise ValueError('Positive finite waveform scale required')
        if self.storage_dtype == 'float32' and self.scale_uv != 1:
            raise ValueError('Float32 waveforms use scale_uv=1')
        # Use the exact decoder scale during both quantization and validation.
        scale = np.float32(self.scale_uv)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Waveform scale cannot be represented in float32')
        channels = raw.astype('<i4', copy=True)
        channels.flags.writeable = False
        object.__setattr__(self, 'channel_index', channels)
        object.__setattr__(self, 'scale_uv', float(scale))

    @property
    def n_samples(self):
        return self.pre_samples + 1 + self.post_samples

    @property
    def width(self):
        return self.channel_index.shape[1]

    @property
    def dtype(self):
        return np.dtype('<i2' if self.storage_dtype == 'int16' else '<f4')

    def metadata(self):
        return dict(pre_samples=int(self.pre_samples), post_samples=int(self.post_samples),
                    sample_count=int(self.n_samples), center_index=int(self.pre_samples),
                    storage_dtype=self.storage_dtype, scale_uv=self.scale_uv,
                    quantization_error_uv=self.scale_uv/2 if self.storage_dtype == 'int16' else 0.,
                    axes=['event_row', 'sample_offset', 'local_channel_slot'],
                    channel_map='channel_index[events.channel_index]; anchor in slot zero; -1 pads missing contacts',
                    boundary_policy='zero stored outside valid_start:valid_stop or at channel -1; decoded to NaN',
                    alignment='saved candidate sample; no recentering or per-channel alignment',
                    coverage='one waveform per event row, including low-SNR candidates; no spatial deduplication')


def grouped_channel_map(group_ids):
    """Explicit acquisition groups, e.g. tetrodes; no physical geometry inferred."""
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or not len(groups):
        raise ValueError('One group ID per recording channel required')
    rows = []
    for c in range(len(groups)):
        others = np.flatnonzero(groups == groups[c])
        rows.append(np.r_[c, others[others != c]])
    result = np.full((len(groups), max(map(len, rows))), -1, np.int32)
    for c, row in enumerate(rows):
        result[c, :len(row)] = row
    return result


def geometry_channel_map(positions_um, group_ids, radius_um, max_channels, *, excluded_channels=()):
    """Nearest contacts within a radius AND explicit probe/shank group.

    Geometry must be known. Explicitly excluded contacts are absent from good
    neighborhoods; recording-channel IDs and flagged anchor rows stay intact.
    """
    pos, groups = np.asarray(positions_um, float), np.asarray(group_ids)
    if (pos.ndim != 2 or pos.shape[1] not in (2, 3) or len(pos) == 0
        or not np.isfinite(pos).all() or groups.shape != (len(pos),)
        or not np.isfinite(radius_um) or radius_um < 0
        or not isinstance(max_channels, (int, np.integer)) or max_channels < 1):
        raise ValueError('Finite geometry, matching group IDs, radius, and channel cap required')
    excluded = np.asarray(excluded_channels)
    if (excluded.ndim != 1 or (excluded.size and excluded.dtype.kind not in 'iu')
            or np.any(excluded < 0) or np.any(excluded >= len(pos))):
        raise ValueError('Excluded channels must be recording-channel indices')
    allowed = np.ones(len(pos), bool)
    allowed[excluded.astype(np.int64)] = False
    result = np.full((len(pos), min(max_channels, len(pos))), -1, np.int32)
    for c in range(len(pos)):
        distance = np.linalg.norm(pos-pos[c], axis=1)
        # Keep the anchor in slot zero, including flagged anchors, for bank
        # coordinate compatibility. Such anchors are never used for learning.
        if not allowed[c]:
            result[c, 0] = c
            continue
        valid = np.flatnonzero(allowed & (groups == groups[c]) & (distance <= radius_um))
        valid = valid[valid != c]
        valid = valid[np.lexsort((valid, distance[valid]))]
        chosen = np.r_[c, valid][:result.shape[1]]
        result[c, :len(chosen)] = chosen
    return result


def extract_waveforms(events, voltage, voltage_start_sample, spec, source_start, source_stop):
    """Extract from an explicitly haloed, preprocessed source block.

    Only source boundaries may truncate a waveform. A missing processing halo
    is an error, never silently treated as missing recording data.
    """
    # Integer session cores expose selected voltage only. Calling asarray on
    # one would rebuild a full FP32 host core just to cache a few snippets.
    from ..scaled_signal import ScaledInt16Signal
    if not isinstance(voltage, ScaledInt16Signal):
        voltage = np.asarray(voltage)
    if voltage.ndim != 2 or voltage.shape[1] != len(spec.channel_index) or not len(voltage):
        raise ValueError('Voltage shape disagrees with waveform channel map')
    t = events['sample_index'].astype(np.int64)
    if source_start >= source_stop or np.any(t < source_start) or np.any(t >= source_stop):
        raise ValueError('Candidate outside source bounds')
    c = events['channel_index'].astype(np.int64)
    if np.any(c < 0) or np.any(c >= len(spec.channel_index)):
        raise ValueError('Unknown anchor channel')
    lo = t - spec.pre_samples
    valid_start = np.maximum(source_start-lo, 0).astype(np.int32)
    valid_stop = np.minimum(source_stop-lo, spec.n_samples).astype(np.int32)
    needed_start = np.maximum(lo, source_start)
    needed_stop = np.minimum(t+spec.post_samples+1, source_stop)
    if (np.any(needed_start < voltage_start_sample)
        or np.any(needed_stop > voltage_start_sample+len(voltage))):
        raise ValueError('Insufficient voltage halo for requested waveforms')
    samples = lo[:, None] + np.arange(spec.n_samples, dtype=np.int64)
    local_channels = spec.channel_index[c]
    values = voltage[np.clip(samples-voltage_start_sample, 0, len(voltage)-1)[:, :, None],
                     np.maximum(local_channels, 0)[:, None, :]].astype(np.float32)
    valid = waveform_mask(valid_start, valid_stop, local_channels, spec.n_samples)
    if not np.isfinite(values[valid]).all():
        raise ValueError('Nonfinite waveform voltage')
    values[~valid] = 0
    return values, valid_start, valid_stop


def waveform_mask(valid_start, valid_stop, channels, n_samples):
    offsets = np.arange(n_samples)[None, :]
    return ((offsets >= np.asarray(valid_start)[:, None])
            & (offsets < np.asarray(valid_stop)[:, None]))[:, :, None] & (channels >= 0)[:, None, :]


def encode_waveforms(values, spec):
    """Round to nearest count. Out-of-range input fails rather than clipping."""
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite stored waveform')
    if spec.storage_dtype == 'float32':
        return np.asarray(values, dtype='<f4')
    counts = np.rint(np.asarray(values, np.float64)/spec.scale_uv)
    if np.any(counts < -32768) or np.any(counts > 32767):
        raise OverflowError('Waveform exceeds int16 range; increase scale_uv or use float32')
    return counts.astype('<i2')


@dataclass
class WaveformBatch:
    row_indices: np.ndarray
    events: np.ndarray
    waveforms_uv: np.ndarray
    channel_indices: np.ndarray
    valid_start: np.ndarray
    valid_stop: np.ndarray
