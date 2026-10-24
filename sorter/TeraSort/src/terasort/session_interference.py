"""Bounded, amplitude-aware scheduling of weakly interacting template fits."""
from bisect import bisect_left, insort

import numpy as np


class InterferenceScheduler:
    def __init__(self, models, noise, *, budget=.01, cache_limit=32768, max_colors=32):
        self.models = models
        self.budget = budget
        self.cache_limit = cache_limit
        self.max_colors = max_colors
        self.cache = {}
        self.slots = [{int(c):j for j,c in enumerate(row) if c >= 0}
                      for row in models.channels]
        weights = np.zeros(models.channels.shape, np.float32)
        valid = models.channels >= 0
        weights[valid] = 1/np.asarray(noise)[models.channels[valid]]
        self.whitened = models.waveforms * weights[:, None, :]
        self.energy = np.sum(self.whitened**2, axis=(1,2), dtype=np.float64)
        self.reset()

    def reset(self):
        self.fits = []
        self.by_contact = {}

    def coupling(self, unit, other, delta):
        key = (unit, other, delta)
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.cache_limit:
            return float('inf')  # Saturation defers work; never silently loosens the rule.
        common = sorted(self.slots[unit].keys() & self.slots[other].keys())
        ia = [self.slots[unit][c] for c in common]
        ib = [self.slots[other][c] for c in common]
        a, b = self.whitened[unit], self.whitened[other]
        if abs(delta) >= self.models.nt or not common:
            dot = 0.
        elif delta >= 0:
            dot = abs(float(np.sum(a[delta:, ia] * b[:self.models.nt-delta, ib], dtype=np.float64)))
        else:
            dot = abs(float(np.sum(a[:self.models.nt+delta, ia] * b[-delta:, ib], dtype=np.float64)))
        self.cache[key] = dot
        return dot

    def admit(self, t, unit, amplitude, *, score=1., margin=1.):
        neighbors = {}
        for c in self.slots[unit]:
            rows = self.by_contact.get(c, [])
            lo = bisect_left(rows, (t-self.models.nt+1, -1))
            hi = bisect_left(rows, (t+self.models.nt, -1))
            for time, index in rows[lo:hi]:
                neighbors.setdefault(index, c)
        own_charge, charges, colors = 0., [], set()
        for index, contact in sorted(neighbors.items()):
            prior = self.fits[index]
            if score < .8 or margin < .1:
                return None, dict(blocker_time=prior['time'], blocker_unit=prior['unit'],
                                  blocker_contact=contact)
            dot = self.coupling(unit, prior['unit'], prior['time']-t)
            own_charge += dot*prior['amplitude']/max(amplitude*self.energy[unit], 1e-12)
            other_charge = dot*amplitude/max(prior['amplitude']*self.energy[prior['unit']], 1e-12)
            if own_charge > self.budget or prior['charge']+other_charge > self.budget:
                return None, dict(blocker_time=prior['time'], blocker_unit=prior['unit'],
                                  blocker_contact=contact)
            charges.append((index, other_charge))
            colors.add(prior['color'])
        color = 0
        while color in colors:
            color += 1
        if color >= self.max_colors:
            index = min(neighbors)
            prior = self.fits[index]
            return None, dict(blocker_time=prior['time'], blocker_unit=prior['unit'],
                              blocker_contact=neighbors[index])
        for index, charge in charges:
            self.fits[index]['charge'] += charge
        index = len(self.fits)
        self.fits.append(dict(time=t, unit=unit, amplitude=amplitude,
                              charge=own_charge, color=color, score=score, margin=margin))
        for c in self.slots[unit]:
            insort(self.by_contact.setdefault(c, []), (t,index))
        return color, None
