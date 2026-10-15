"""Bounded three-phase telemetry, independent of CUDA and recording length."""
import json
import math
import os
from pathlib import Path
import time

STEPS = (
    ('preprocess', 'Preprocessing', 'Whitening', 2),
    ('drift', 'Preprocessing', 'Drift correction', 19),
    ('detect', 'Sorting', 'Template detection', 18),
    ('cluster1', 'Sorting', 'First clustering', 15),
    ('learned', 'Sorting', 'Learned detection', 18),
    ('cluster2', 'Post-processing', 'Final clustering', 16),
    ('merge', 'Post-processing', 'Merging clusters', 5),
    ('export', 'Post-processing', 'Saving results', 7),
    ('finalize', 'Post-processing', 'Final plots', 1),
)
PHASES = ('Preprocessing', 'Sorting', 'Post-processing')


class JobProgress:
    def __init__(self, output=None, *, skip_drift=False, staging_seconds=0.,
                 lfp=False, clock=time.monotonic, interval=3., on_task_start=None):
        self.clock, self.interval = clock, interval
        self.on_task_start = on_task_start
        self.started = clock() - staging_seconds
        self.path = Path(output) / 'terasort_progress.json' if output else None
        self.steps = [s for s in STEPS if not (skip_drift and s[0] == 'drift')]
        if staging_seconds:
            self.steps.insert(0, ('staging', 'Preprocessing', 'Staging input', 10))
        if lfp:
            self.steps.append(('lfp', 'Post-processing', 'Waiting for LFP export', 5))
        self.fractions = {s[0]: 0. for s in self.steps}
        if staging_seconds: self.fractions['staging'] = 1.
        self.active = None
        self.task = None
        self.last_write = -math.inf
        self.task_started = clock()
        self.task_key = None

    def stage(self, key):
        keys = [s[0] for s in self.steps]
        if key not in keys or key == self.active: return
        index = keys.index(key)
        if self.active is not None and index < keys.index(self.active): return
        for previous in keys[:index]: self.fractions[previous] = 1.
        self.active, self.task, self.task_key = key, None, None
        self.publish(force=True)

    def counter(self, done, total, label, start=0., span=.98, allowed=None):
        if self.active is None or (allowed and self.active not in allowed): return
        done, total = int(done), int(total)
        if total <= 0 or not 0 <= done <= total: return
        key = (self.active, label, total)
        if key != self.task_key:
            self.task_key, self.task_started = key, self.clock()
        elapsed = self.clock() - self.task_started
        eta = round(elapsed * (total-done)/done) if done and elapsed >= 1 else None
        self.task = {'label': label, 'completed': done, 'total': total,
                     'percent': round(100*done/total, 1), 'eta_seconds': eta}
        self.fractions[self.active] = max(self.fractions[self.active], min(.99, start+span*done/total))
        self.publish()

    def track(self, iterable, label, start=0., span=.98, allowed=None):
        total = len(iterable)
        self.counter(0, total, label, start, span, allowed)
        if self.on_task_start is not None:
            self.on_task_start(label)
        for done, value in enumerate(iterable, 1):
            yield value
            self.counter(done, total, label, start, span, allowed)

    def snapshot(self):
        weights = {s[0]: s[3] for s in self.steps}
        complete = sum(weights[k]*v for k,v in self.fractions.items())
        total = sum(weights.values())
        fraction = complete/total
        elapsed = max(0., self.clock()-self.started)
        eta = round(elapsed*(1-fraction)/fraction) if fraction > 0 and elapsed >= 5 else None
        if self.active == 'lfp': eta = None
        phases = []
        for phase in PHASES:
            entries = [s for s in self.steps if s[1] == phase]
            pct = 100*sum(s[3]*self.fractions[s[0]] for s in entries)/sum(s[3] for s in entries)
            remaining = sum(s[3]*(1-self.fractions[s[0]]) for s in entries)
            phase_eta = round(elapsed*remaining/complete) if complete > 0 and elapsed >= 5 else None
            if self.active == 'lfp' and phase == 'Post-processing': phase_eta = None
            phases.append({'name': phase, 'percent': round(pct, 1), 'eta_seconds': 0 if pct >= 100 else phase_eta,
                           'status': 'completed' if pct >= 100 else 'running' if any(s[0] == self.active for s in entries) else 'pending'})
        label = next((s[2] for s in self.steps if s[0] == self.active), 'Starting')
        return {'schema_version': 1, 'updated_at': time.time(), 'pid': os.getpid(),
                'stage': label, 'percent': round(min(99.9, fraction*100), 1),
                'progress_label': 'Overall progress (estimated)',
                'progress_detail': self.task['label'] if self.task else label,
                'eta_label': 'Overall ETA (estimated)', 'eta_seconds': eta,
                'phases': phases, 'task_progress': self.task,
                'estimate_basis': 'Measured task counters; phase percentages and total ETA use stage weights. Later data-dependent work can change the estimate.'}

    def publish(self, force=False):
        now = self.clock()
        if not force and now-self.last_write < self.interval: return
        self.last_write = now
        if self.path is None or not self.path.parent.is_dir(): return
        temporary = self.path.with_suffix('.json.tmp')
        try:
            temporary.write_text(json.dumps(self.snapshot()), encoding='utf-8')
            os.replace(temporary, self.path)
        except OSError:
            # Telemetry must not abort the numerical computation.
            pass


def apply_progress_snapshot(job, path):
    """Read one bounded atomic snapshot; job status remains authoritative."""
    try:
        with Path(path).open('rb') as stream: payload = stream.read(65537)
        if len(payload) > 65536: return
        snapshot = json.loads(payload)
        if snapshot.get('schema_version') != 1 or snapshot.get('pid') != job.get('pid'): return
        if job['status'] == 'running':
            for key in ('stage', 'percent', 'progress_label', 'progress_detail', 'eta_label',
                        'eta_seconds', 'phases', 'task_progress', 'estimate_basis'):
                job[key] = snapshot[key]
        elif job['status'] == 'completed':
            job['phases'] = [{'name': p, 'percent': 100., 'status': 'completed'} for p in PHASES]
        elif job['status'] in ('failed', 'cancelled'):
            job['phases'] = [{**p, 'status': job['status'] if p['status'] == 'running' else p['status']} for p in snapshot['phases']]
    except (OSError, ValueError, KeyError, TypeError):
        return
