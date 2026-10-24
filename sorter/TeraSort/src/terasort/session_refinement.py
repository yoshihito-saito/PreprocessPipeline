"""Experimental fixed-identity refinement from bounded, disjoint evidence."""
import heapq

import numpy as np


class StratifiedEvidence:
    """Deterministic sample cap for each unit and declared learning window."""

    def __init__(self, units, windows, *, budget=32768, per_window=8):
        self.windows = frozenset(windows)
        if units < 1 or not self.windows or budget < 1 or per_window < 1:
            raise ValueError('Positive evidence bounds required')
        self.units = units
        self.capacity = min(per_window, budget // (units * len(self.windows)))
        self.rows = {}

    def observe(self, unit, sample, core_start, waveform):
        if core_start not in self.windows or not 0 <= unit < self.units:
            raise ValueError('Evidence outside declared strata')
        if not self.capacity:
            return
        # SplitMix64 ranking, independent of arrival order and Python hash seed.
        x = (int(sample) + 0x9E3779B97F4A7C15) & ((1 << 64)-1)
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64)-1)
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & ((1 << 64)-1)
        priority = x ^ (x >> 31)
        rows = self.rows.setdefault((unit, core_start), [])
        if any(-r[1] == sample for r in rows):
            return
        item = (-priority, -int(sample), waveform.copy())
        if len(rows) < self.capacity:
            heapq.heappush(rows, item)
        elif item[:2] > rows[0][:2]:
            heapq.heapreplace(rows, item)

    def samples(self, unit):
        return [row[2] for (u, c), rows in sorted(self.rows.items()) if u == unit
                for row in sorted(rows, key=lambda r: r[:2])]


def refine_templates(models, training, validation, *, blend=.25):
    """Propose only from training; use separate windows to accept or reject.

    Waveform loss is a surrogate, not a guarantee of correct clustering.
    Never consume ground-truth labels here. Keep the original bank for rollback.
    """
    if training.windows & validation.windows:
        raise ValueError('Training and validation windows must be disjoint')
    if not 0 < blend <= 1:
        raise ValueError('Blend must be in (0, 1]')
    audit = []
    for unit, current in enumerate(models.waveforms):
        train, valid = training.samples(unit), validation.samples(unit)
        record = dict(unit=unit, training_count=len(train), validation_count=len(valid),
                      promoted=False, reason='insufficient_evidence')
        audit.append(record)
        if len(train) < 12 or len(valid) < 6:
            continue
        if sum(u == unit for u, c in training.rows) < 2:
            continue
        target = np.median(np.stack(train), axis=0)
        proposed = (1-blend)*current + blend*target
        contacts = models.channels[unit] >= 0
        change = float(np.linalg.norm((proposed-current)[:, contacts]) /
                       max(np.linalg.norm(current[:, contacts]), 1e-12))
        heldout = np.stack(valid)[:, :, contacts]
        old_error = np.mean((heldout-current[:, contacts])**2, axis=(1, 2))
        new_error = np.mean((heldout-proposed[:, contacts])**2, axis=(1, 2))
        mean_old = max(float(np.mean(old_error)), 1e-12)
        gain = float(1 - np.mean(new_error)/mean_old)
        record.update(relative_change=change, validation_gain=gain)
        if (not np.isfinite(proposed).all() or change > .1 or gain < .005
                or np.quantile(new_error-old_error, .9) > .05*mean_old):
            record['reason'] = 'validation_rejected'
            continue
        # Keep the source-sample alignment invariant when reloading a seed bank.
        if (np.argmax(np.max(np.abs(proposed), axis=0)) != 0
                or np.argmax(np.abs(proposed[:, 0])) != models.nt//2):
            record['reason'] = 'alignment_changed'
            continue
        models.waveforms[unit] = proposed
        record.update(promoted=True, reason='heldout_waveform_improvement')
    if any(row['promoted'] for row in audit):
        models.version += 1
    return audit
