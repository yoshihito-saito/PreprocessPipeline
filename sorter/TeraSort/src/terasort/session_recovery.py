"""Bounded retrospective fits of collision-free novelty observations.

Only selected observations already read by the dense pass are retained. Recovery
is published with enrollment; earlier assignment shards are never rewritten.
"""
import json

import numpy as np

from .session_store import SPIKE_DTYPE

DATASET = 'novelty_recovered_spikes'
MAX_PER_UNIT = 64
MAX_PER_SHARD = 8 * MAX_PER_UNIT
MAX_PER_DAY = 4096 * MAX_PER_UNIT


def encode_observation(waveform, noise):
    """A per-snippet power-of-two scale avoids INT16 clipping, including padding."""
    scale = np.float32(2. ** np.ceil(np.log2(max(
        float(np.max(np.abs(waveform))) / 30000., 2. ** -20))))
    counts = np.rint(waveform / scale).astype(np.int16)
    noise = np.asarray(noise, np.float32)
    if noise.shape != (waveform.shape[1],) or not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError('Invalid retrospective observation noise')
    return counts, scale, noise.copy()


def _fit(template, data, noise):
    # Small selected snippets only. Match the dense center-17 contact mask,
    # full-window amplitude and noise-weighted residual energy decision.
    a, b = template.astype(np.float64) / noise, data.astype(np.float64) / noise
    dot, energy = float(np.sum(a*b)), float(np.sum(a*a))
    amplitude = dot / max(energy, 1.e-12)
    contacts = np.sum(a[len(a)//2-8:len(a)//2+9] ** 2, axis=0)
    mask = contacts >= .05 * max(float(contacts.max()), 1.e-12)
    ac, bc = a[len(a)//2-8:len(a)//2+9, mask], b[len(b)//2-8:len(b)//2+9, mask]
    score = float(np.sum(ac*bc)) / max(float(np.linalg.norm(ac)*np.linalg.norm(bc)), 1.e-12)
    return score, amplitude, 2*amplitude*dot-amplitude*amplitude*energy


def fit_observations(models, unit, channels, observations):
    accepted = []
    competitors = sorted({u for c in channels if c >= 0
        for u in models.by_channel.get(int(c), ()) if u != unit})
    valid = channels >= 0
    for observation in observations:
        counts, scale, noise = observation[4:7]
        data = counts.astype(np.float32) * scale
        score, amplitude, gain = _fit(models.waveforms[unit][:, valid], data[:, valid], noise[valid])
        if score < .76 or not .7 <= amplitude <= 3. or gain < 4.5**2:
            continue
        runner = -1.
        rejected = False
        for other in competitors:
            common, ai, bi = np.intersect1d(models.channels[other], channels, return_indices=True)
            keep = common >= 0
            ai, bi = ai[keep], bi[keep]
            if not len(ai):
                continue
            # A competing template is fitted on observed contacts only. This
            # can reject an ambiguous fit; missing voltage is never fabricated.
            oldscore, oldamp, oldgain = _fit(models.waveforms[other][:, ai], data[:, bi], noise[bi])
            if oldscore >= .76 and .7 <= oldamp <= 3.:
                runner = max(runner, oldscore)
                if oldgain >= gain:
                    rejected = True
                    break
        if not rejected:
            accepted.append((observation[0], int(channels[0]), unit, score, runner,
                             amplitude, gain, -1, -1, models.version))
    return np.asarray(accepted, dtype=SPIKE_DTYPE)


def read_recovered(handle):
    """Validate the fixed payload before allocating or consuming recovery rows."""
    if DATASET not in handle:
        return np.empty(0, SPIKE_DTYPE)
    data = handle[DATASET]
    if data.ndim != 1 or len(data) > MAX_PER_SHARD or data.dtype != SPIKE_DTYPE:
        raise ValueError('Invalid or oversized novelty recovery payload')
    rows = data[:]
    if not len(rows):
        return rows
    if 'novelty/audit_json' not in handle:
        raise ValueError('Novelty recovery is missing enrollment provenance')
    audit = json.loads(handle['novelty/audit_json'][()])
    allowed = {}
    for record in audit:
        unit = record.get('enrolled_unit')
        if unit is not None:
            samples = record.get('backfill_samples', [])
            if len(samples) > MAX_PER_UNIT or not set(samples).issubset(record['supporting_samples']):
                raise ValueError('Invalid novelty recovery provenance')
            if int(unit) in allowed:
                raise ValueError('Duplicate recovery enrollment provenance')
            allowed[int(unit)] = set(samples)
    shape = handle['model_after/waveforms'].shape
    rate = float(handle['novelty'].attrs['sample_rate'])
    first, stop = int(handle.attrs['start_sample']), int(handle.attrs['stop_sample'])
    if (np.any(rows['unit_id'] < 0) or np.any(rows['unit_id'] >= shape[0])
            or np.any(rows['sample_index'] < first - int(1800*rate))
            or np.any(rows['sample_index'] >= stop)
            or np.any(rows['candidate_row'] != -1) or np.any(rows['pass_index'] != -1)):
        raise ValueError('Invalid novelty recovery coordinates or IDs')
    for field in ('score', 'runner_up', 'fitted_amplitude', 'residual_gain'):
        if not np.isfinite(rows[field]).all():
            raise ValueError('Nonfinite novelty recovery fit')
    if (np.any(rows['score'] < .76) or np.any(rows['score'] > 1.00001)
            or np.any(rows['fitted_amplitude'] < .7) or np.any(rows['fitted_amplitude'] > 3.)
            or np.any(rows['residual_gain'] < 4.5**2)):
        raise ValueError('Novelty recovery fit failed acceptance bounds')
    keys = set()
    for row in rows:
        key = (int(row['sample_index']), int(row['unit_id']))
        if key in keys or key[0] not in allowed.get(key[1], ()):
            raise ValueError('Duplicate or unsupported novelty recovery event')
        keys.add(key)
        if int(row['channel_index']) != int(handle['model_after/channels'][key[1], 0]):
            raise ValueError('Invalid novelty recovery physical anchor')
    return rows


def merge_recovered(native, recovered, block):
    """Merge sorted correction rows with streaming native rows in source order."""
    cursor = 0
    for rows in native:
        if not len(rows):
            continue
        stop = int(np.searchsorted(recovered['sample_index'], rows['sample_index'][-1], side='right'))
        extra = recovered[cursor:stop]
        merged = np.concatenate((rows, extra)) if len(extra) else rows
        if len(extra):
            merged = merged[np.argsort(merged['sample_index'], kind='stable')]
        for start in range(0, len(merged), block):
            yield merged[start:start+block]
        cursor = stop
    for start in range(cursor, len(recovered), block):
        yield recovered[start:start+block]
