"""Bounded evidence and temporally held-out promotion of local templates.

This is a conservative adaptation gate, not proof of neuron identity. The
cache retains recent accepted events, not an unbiased sample of all spikes.
"""
from collections import deque
import json

import numpy as np


class TemplateEvidence:
    def extend_models(self, models):
        """Append identity state and shrink queues to preserve the global cap."""
        count = len(models.waveforms)
        old = len(self.last_promotion)
        if count == old:
            return
        if count < old:
            raise ValueError('Unit IDs cannot be removed during adaptation')
        self.capacity = min(self.per_unit, self.budget // max(1, count))
        self.last_promotion = np.pad(self.last_promotion, (0, count-old), constant_values=-1)
        self.rows = {unit: deque(list(rows)[-self.capacity:], maxlen=self.capacity)
                     for unit, rows in self.rows.items()} if self.capacity else {}

    def __init__(self, models, sample_rate, *, budget=32768, per_unit=64,
                 horizon_seconds=1800., quantized=False):
        if not isinstance(quantized, bool):
            raise ValueError('Quantized evidence must be enabled explicitly')
        self.quantized = quantized
        self.shape = models.waveforms.shape[1:]
        self.capacity = min(per_unit, budget // max(1, len(models.waveforms)))
        self.budget = budget
        self.per_unit = per_unit
        self.horizon = round(sample_rate * horizon_seconds)
        self.rows = {}
        self.last_promotion = np.full(len(models.waveforms), -1, np.int64)

    def observe(self, unit, sample, core_start, waveform, *, scales=None):
        if not self.capacity:
            return
        rows = self.rows.setdefault(unit, deque(maxlen=self.capacity))
        # Input is chronological, but duplicate detections must not add support.
        if rows and sample <= rows[-1][0]:
            return
        if self.quantized:
            codes = np.asarray(waveform)
            factors = np.asarray(scales, np.float32)
            if (codes.dtype != np.int16 or codes.shape != self.shape or
                    factors.shape != (self.shape[1],) or
                    not np.all(np.isfinite(factors) & (factors >= 0))):
                raise ValueError('Quantized evidence requires INT16 snippets and channel scales')
            rows.append((int(sample), int(core_start), codes.copy(), factors.copy()))
        else:
            if scales is not None:
                raise ValueError('Floating evidence does not use separate channel scales')
            rows.append((int(sample), int(core_start), waveform.copy()))

    def _waveform(self, row):
        if self.quantized:
            return row[2].astype(np.float32) * row[3][None, :]
        return row[2]

    def expire(self, now):
        for unit in list(self.rows):
            rows = self.rows[unit]
            while rows and rows[0][0] < now - self.horizon:
                rows.popleft()
            if not rows:
                del self.rows[unit]

    def promote(self, models, now):
        self.expire(now)
        audit = []
        for unit, rows in sorted(self.rows.items()):
            record = dict(unit=unit, support=len(rows), promoted=False)
            audit.append(record)
            new = sum(row[0] > self.last_promotion[unit] for row in rows)
            cores = sorted({row[1] for row in rows})
            if len(rows) < 16 or new < 8 or len(cores) < 2:
                record['reason'] = 'insufficient_independent_evidence'
                continue
            # Entire later cores are held out; no event enters both partitions.
            split = cores[max(1, (2 * len(cores)) // 3)]
            training = [self._waveform(row) for row in rows if row[1] < split]
            validation = [self._waveform(row) for row in rows if row[1] >= split]
            if min(len(training), len(validation)) < 4:
                record['reason'] = 'insufficient_temporal_holdout'
                continue
            current = models.waveforms[unit]
            target = np.median(np.stack(training), axis=0)
            proposed = .95 * current + .05 * target
            valid = models.channels[unit] >= 0
            norm = max(float(np.linalg.norm(current[:, valid])), 1e-9)
            change = float(np.linalg.norm((proposed-current)[:, valid]) / norm)
            samples = np.stack(validation)[:, :, valid]
            old_error = np.mean((samples-current[:, valid])**2, axis=(1, 2))
            new_error = np.mean((samples-proposed[:, valid])**2, axis=(1, 2))
            gain = float(1 - np.mean(new_error) / max(float(np.mean(old_error)), 1e-12))
            record.update(validation_gain=gain, relative_change=change,
                          training_count=len(training), validation_count=len(validation))
            # Limit damage to atypical held-out events as well as mean error.
            if (not np.isfinite(proposed).all() or change > .1 or gain < .005
                    or np.quantile(new_error-old_error, .9) > .05 * max(float(np.mean(old_error)), 1e-12)):
                record['reason'] = 'validation_rejected'
                continue
            models.waveforms[unit] = proposed
            self.last_promotion[unit] = max(row[0] for row in rows)
            record.update(promoted=True, reason='heldout_improvement')
        if any(row['promoted'] for row in audit):
            models.version += 1
        return audit

    def save(self, handle, audit):
        group = handle.create_group('adaptation')
        group.attrs.update(version=2 if self.quantized else 1,
                           capacity=self.capacity, horizon=self.horizon)
        group.create_dataset('audit_json', data=json.dumps(audit, sort_keys=True))
        group.create_dataset('last_promotion', data=self.last_promotion)
        for unit, rows in sorted(self.rows.items()):
            child = group.create_group(str(unit))
            child.create_dataset('coordinates', data=np.array([(r[0], r[1]) for r in rows], np.int64))
            child.create_dataset('waveforms', data=np.stack([r[2] for r in rows]), compression='lzf')
            if self.quantized:
                child.create_dataset('scales_uv', data=np.stack([r[3] for r in rows]))

    def restore(self, handle):
        group = handle['adaptation']
        if (group.attrs['version'] != (2 if self.quantized else 1)
                or group.attrs['capacity'] != self.capacity
                or group.attrs['horizon'] != self.horizon):
            raise ValueError('Adaptation checkpoint configuration mismatch')
        self.last_promotion = group['last_promotion'][:]
        self.rows.clear()
        for key in group:
            if key in ('last_promotion', 'audit_json'):
                continue
            child = group[key]
            coordinates = child['coordinates'][:]
            waveforms = child['waveforms'][:]
            if self.quantized:
                scales = child['scales_uv'][:]
                if (waveforms.dtype != np.int16 or
                        waveforms.shape != (len(coordinates), *self.shape) or
                        scales.shape != (len(coordinates), self.shape[1]) or
                        not np.all(np.isfinite(scales) & (scales >= 0))):
                    raise ValueError('Invalid quantized evidence checkpoint')
                self.rows[int(key)] = deque(
                    [(int(t), int(c), w, s) for (t, c), w, s in zip(
                        coordinates, waveforms, scales)], maxlen=self.capacity)
            else:
                self.rows[int(key)] = deque(
                    [(int(t), int(c), w) for (t, c), w in zip(
                        coordinates, waveforms)], maxlen=self.capacity)
