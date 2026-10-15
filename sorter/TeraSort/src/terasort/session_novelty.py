"""Experimental bounded unknown-event proposals, isolated from dense sorting."""
from bisect import bisect_left
from collections import deque
import json

import numpy as np


def seed_temporal_basis(models):
    """Three temporal PCs from a bounded seed bank; no event holdout is used."""
    covariance = np.zeros((models.nt, models.nt), np.float64)
    for first in range(0, len(models.waveforms), 16):
        slab = models.waveforms[first:first+16].astype(np.float64)
        covariance += np.einsum('utc,usc->ts', slab, slab)
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, -3:].astype(np.float32)


def similarity(a, ac, b, bc):
    """Cosine on physical contacts, penalizing energy outside the intersection."""
    av, bv = ac >= 0, bc >= 0
    _, ai, bi = np.intersect1d(ac[av], bc[bv], return_indices=True)
    aa, bb = a[:, av], b[:, bv]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-9:
        return 0.
    return max(float(np.sum(aa[2:len(a)-2, ai] * bb[2+s:len(b)-2+s, bi])) / denom
               for s in range(-2, 3))


def raw_anchor_consistency(template, windows):
    """Unprojected, preprocessed observations must support the training shape.

    Cache 21 anchor samples per observation for a center-17 score at +/-2
    shifts. This validation does not alter the PCA training waveform.
    """
    anchor = template[len(template)//2-8:len(template)//2+9, 0]
    norm = float(np.linalg.norm(anchor))
    scores = []
    for window in windows:
        scores.append(max(float(anchor @ window[2+s:19+s]) /
            max(float(norm*np.linalg.norm(window[2+s:19+s])), 1.e-9)
            for s in range(-2, 3)))
    return float(np.quantile(scores, .1))


def validate_raw_holdout_minimum(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Raw holdout minimum must be finite in (0, 1]')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Raw holdout minimum must be finite in (0, 1]') from exc
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError('Raw holdout minimum must be finite in (0, 1]')
    return value


def validate_tentative_retention(seconds):
    if isinstance(seconds, (bool, np.bool_)):
        raise ValueError('Tentative retention must be finite seconds in [0, 1800]')
    try:
        seconds = float(seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Tentative retention must be finite seconds in [0, 1800]') from exc
    if not np.isfinite(seconds) or not 0 <= seconds <= 1800:
        raise ValueError('Tentative retention must be finite seconds in [0, 1800]')
    return seconds


def validate_candidate_floor(snr):
    if isinstance(snr, (bool, np.bool_)):
        raise ValueError('Novelty candidate SNR must be finite and positive')
    try:
        snr = float(snr)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Novelty candidate SNR must be finite and positive') from exc
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError('Novelty candidate SNR must be finite and positive')
    return snr


PEAK_ALIGNMENT_RULE = 'raw_peak_pm2_core_owned_collision_recheck_v1'


def peak_alignment_policy(nt=61):
    return dict(algorithm=PEAK_ALIGNMENT_RULE, maximum_shift_samples=2,
                observation_budget_per_core=256, selection='original_source_coordinate_hash',
                owned_core_required=True, collision_guard_samples=nt-1,
                snr='original_detector_peak', raw_validation='unchanged')


class NoveltyBank:
    """At most 128 proposals, 64 waveforms each, 256 observations per core.

    Enrollment is experimental. Waveform consistency does not prove identity.
    Proposals use fixed reference shapes so held-out samples cannot train them.
    """
    def __init__(self, sample_rate, *, nt=61, max_proposals=128, models=None, tentative_seconds=0.,
                 candidate_floor_snr=6., stratified_admission=False, priority_admission=False,
                 preserve_strong_tentative=False, raw_anchor_holdout=False,
                 raw_holdout_minimum_cosine=.85, backfill=False, peak_alignment=False,
                 residual_duplicates=False):
        from .session_models import validate_nt
        self.nt = validate_nt(models.nt if models is not None else nt)
        if not isinstance(residual_duplicates,bool) or (residual_duplicates and
            (models is None or not (raw_anchor_holdout and backfill))):
            raise ValueError('Novelty residual duplicate validation requires compact raw holdout')
        self.residual_duplicates = residual_duplicates
        if not isinstance(stratified_admission, bool):
            raise ValueError('Stratified admission must be boolean')
        if not isinstance(priority_admission, bool) or (priority_admission and stratified_admission):
            raise ValueError('Priority admission must be boolean and cannot combine with quotas')
        self.stratified_admission = stratified_admission
        self.priority_admission = priority_admission
        if not isinstance(preserve_strong_tentative, bool):
            raise ValueError('Strong tentative retention must be boolean')
        self.preserve_strong_tentative = preserve_strong_tentative
        if not isinstance(raw_anchor_holdout, bool):
            raise ValueError('Raw anchor holdout must be boolean')
        self.raw_anchor_holdout = raw_anchor_holdout
        if not isinstance(backfill, bool) or (backfill and not raw_anchor_holdout):
            raise ValueError('Novelty backfill requires raw anchor validation')
        self.backfill = backfill
        if not isinstance(peak_alignment, bool):
            raise ValueError('Novelty peak alignment must be boolean')
        self.peak_alignment = peak_alignment
        if backfill and (isinstance(max_proposals, bool) or not isinstance(max_proposals, int)
                         or not 1 <= max_proposals <= 128):
            raise ValueError('Novelty backfill proposal budget must be in [1, 128]')
        from .session_store import SPIKE_DTYPE
        self.recovered = np.empty(0, SPIKE_DTYPE)
        self.curation_basis = None
        self.recovered_features = None
        self.raw_holdout_minimum_cosine = validate_raw_holdout_minimum(raw_holdout_minimum_cosine)
        self.retired_raw_holdout = 0
        self.track_snr = (stratified_admission or priority_admission
                          or preserve_strong_tentative or raw_anchor_holdout)
        self.rate = float(sample_rate)
        self.max_proposals = max_proposals
        self.next_id = 0
        self.tentative_seconds = validate_tentative_retention(tentative_seconds)
        self.retired_tentative = 0
        self.candidate_floor_snr = validate_candidate_floor(candidate_floor_snr)
        self.proposals = {}
        self.temporal_basis = None if models is None else seed_temporal_basis(models)

    def expire(self, now):
        for key in list(self.proposals):
            proposal = self.proposals[key]
            idle = now - proposal['rows'][-1][0]
            tentative = (self.tentative_seconds > 0 and len(proposal['rows']) < 3
                         and idle >= self.rate * self.tentative_seconds
                         and (not self.preserve_strong_tentative
                              or proposal['reference_snr'] < 6.))
            if tentative or idle > self.rate * 1800:
                del self.proposals[key]
                self.retired_tentative += int(tentative)
            else:
                rows = proposal['rows']
                while rows and rows[0][0] < now - self.rate * 1800:
                    rows.popleft()

    def observe(self, sample, core_start, channels, waveform, *, snr=6., noise_uv=None):
        if not np.isfinite(waveform).all() or np.linalg.norm(waveform) < 1e-6:
            return
        raw_window = waveform[self.nt//2-10:self.nt//2+11, 0].copy() if self.raw_anchor_holdout else None
        if self.backfill:
            if waveform.shape != (self.nt, len(channels)) or len(channels) > 16:
                raise ValueError('Novelty backfill requires at most 16 local contacts')
            from .session_recovery import encode_observation
            compact = encode_observation(waveform, np.ones(len(channels), np.float32)
                if noise_uv is None else noise_uv)
        if self.temporal_basis is not None:
            # Project only this selected snippet, never the recording core.
            waveform = self.temporal_basis @ (self.temporal_basis.T @ waveform)
        snr = validate_candidate_floor(snr) if self.track_snr else 6.
        anchor = int(channels[0])
        # Incompatible SNR shapes still count toward the four-entry contact
        # cap; they cannot absorb a much stronger sparse neuron.
        nearby = [((similarity(waveform, channels, p['reference'], p['channels'])
                    if not self.raw_anchor_holdout or .5*p['reference_snr'] <= snr <= 2*p['reference_snr']
                    else 0.), key)
                  for key, p in self.proposals.items() if p['channels'][0] == anchor]
        row = (int(sample), int(core_start), waveform.copy())
        if self.raw_anchor_holdout:
            row += (raw_window,)
        if self.backfill:
            row += compact
        best = max(nearby, default=(0., -1))
        if best[0] >= .9:
            rows = self.proposals[best[1]]['rows']
            if sample > rows[-1][0] + max(2, round(.0005*self.rate)):
                rows.append(row)
        else:
            if self.priority_admission:
                full_anchor = len(nearby) >= 4
                if snr >= 6. and (full_anchor or len(self.proposals) >= self.max_proposals):
                    victims = [key for key, p in self.proposals.items() if len(p['rows']) < 3
                               and snr >= 1.25*p['reference_snr']
                               and (not full_anchor or p['channels'][0] == anchor)]
                    if victims:
                        victim = min(victims, key=lambda key: (
                            len(self.proposals[key]['rows']), self.proposals[key]['reference_snr'],
                            self.proposals[key]['rows'][-1][0], key))
                        del self.proposals[victim]
                        self.retired_tentative += 1
                        nearby = [(score, key) for score, key in nearby if key != victim]
            if self.stratified_admission:
                weak = [key for key, p in self.proposals.items() if p['reference_snr'] < 6.]
                if snr < 6. and (len(weak) >= max(1, self.max_proposals//4) or
                        any(self.proposals[key]['channels'][0] == anchor for key in weak)):
                    return
                full_anchor = len(nearby) >= 4
                if snr >= 6. and (full_anchor or len(self.proposals) >= self.max_proposals):
                    victims = [key for key in weak if len(self.proposals[key]['rows']) < 3
                               and (not full_anchor or self.proposals[key]['channels'][0] == anchor)]
                    if victims:
                        victim = min(victims, key=lambda key: (
                            len(self.proposals[key]['rows']), self.proposals[key]['reference_snr'],
                            self.proposals[key]['rows'][-1][0], key))
                        del self.proposals[victim]
                        self.retired_tentative += 1
                        nearby = [(score, key) for score, key in nearby if key != victim]
            if len(self.proposals) < self.max_proposals and len(nearby) < 4:
                self.proposals[self.next_id] = dict(reference=waveform.copy(),
                    channels=channels.copy(), rows=deque(
                        [row], maxlen=64))
                if self.track_snr:
                    self.proposals[self.next_id]['reference_snr'] = snr
                self.next_id += 1

    def collect(self, models, events, matches, signal, quality, channel_map, core):
        self.expire(core.core_stop)
        if quality.interval_bad:
            return
        # Reject candidate neighborhoods containing any assigned waveform.
        occupied = {}
        local_contacts = [tuple(int(c) for c in row if c >= 0)
                          for row in models.channels]
        usable = quality.usable_channels
        for match in matches:
            for c in local_contacts[match.unit]:
                occupied.setdefault(c, []).append(match.source_sample)
        candidates = sorted(set((int(t), int(c)) for t, c, snr in events
            if snr >= self.candidate_floor_snr and core.core_start <= core.data_start+int(t) < core.core_stop))
        # Spend the observation budget on unknowns, not detections already
        # explained by accepted units. Filter metadata before copying snippets.
        eligible = []
        if candidates:
            candidate_array = np.asarray(candidates, np.int64)
            # Group once with a stable contact sort. Within each contact,
            # times keep the same ascending order as the source-sorted set.
            contact_order = np.argsort(candidate_array[:, 1], kind="stable")
            grouped = candidate_array[contact_order]
            contacts, starts, counts = np.unique(grouped[:, 1],
                return_index=True, return_counts=True)
            for c, first, count in zip(contacts, starts, counts):
                channels = channel_map[c]
                valid = channels >= 0
                if not np.all(usable[channels[valid]]):
                    continue
                times = grouped[first:first + count, 0]
                busy = np.array(sorted({t for contact in channels[valid]
                                        for t in occupied.get(int(contact), ())}), np.int64)
                left = np.searchsorted(busy, times-(self.nt-1), side='left')
                right = np.searchsorted(busy, times+(self.nt-1), side='right')
                times = times[left == right]
                if len(times):
                    amplitudes = np.abs(signal[times[:, None], channels[valid]])
                    times = times[np.argmax(amplitudes, axis=1) == 0]
                eligible.extend((int(t), int(c)) for t in times)
        # Deterministic source-coordinate sampling before waveform extraction.
        def key(event):
            t, c = event
            value = ((core.data_start+t)*11400714819323198485 ^ c*7046029254386353131)
            return value & ((1 << 64)-1)
        candidates = sorted(sorted(eligible, key=key)[:256])
        strengths = {}
        if self.track_snr:
            selected = set(candidates)
            for t, c, snr in events:
                event = (int(t), int(c))
                if event in selected:
                    strengths[event] = max(strengths.get(event, 0.), float(snr))
        for t, c in candidates:
            original_t = t
            channels = channel_map[c]
            valid = channels >= 0
            if t < self.nt//2 or t+self.nt//2+1 > len(signal) or not np.all(usable[channels[valid]]):
                continue
            conflict = False
            for contact in channels[valid]:
                times = occupied.get(int(contact), ())
                j = bisect_left(times, t-(self.nt-1))
                if j < len(times) and times[j] <= t+(self.nt-1):
                    conflict = True
                    break
            if conflict:
                continue
            waveform = np.zeros((self.nt, len(channels)), np.float32)
            waveform[:, valid] = signal[t-self.nt//2:t+self.nt//2+1, channels[valid]]
            # One physical anchor and time center per event, avoiding repeated
            # enrollment from multiple detector contacts and rebound phases.
            peak, slot = np.unravel_index(np.argmax(np.abs(waveform)), waveform.shape)
            if slot != 0:
                continue
            if peak != self.nt//2:
                if not self.peak_alignment or abs(int(peak)-self.nt//2) > 2:
                    continue
                # Smoothed detection and raw maxima can differ by two samples.
                # Re-extract the bounded snippet; never roll in samples from a
                # neighboring core or across an accepted waveform's guard.
                aligned = t + int(peak)-self.nt//2
                if (aligned < self.nt//2 or aligned+self.nt//2+1 > len(signal)
                        or not core.core_start <= core.data_start+aligned < core.core_stop):
                    continue
                conflict = False
                for contact in channels[valid]:
                    times = occupied.get(int(contact), ())
                    j = bisect_left(times, aligned-(self.nt-1))
                    if j < len(times) and times[j] <= aligned+(self.nt-1):
                        conflict = True
                        break
                if conflict:
                    continue
                waveform[:, valid] = signal[aligned-self.nt//2:aligned+self.nt//2+1, channels[valid]]
                peak, slot = np.unravel_index(np.argmax(np.abs(waveform)), waveform.shape)
                if slot != 0 or peak != self.nt//2:
                    continue
                t = aligned
            if self.track_snr:
                self.observe(core.data_start+t, core.core_start, channels, waveform,
                             snr=strengths[(original_t, c)], **({'noise_uv': np.maximum(
                                 quality.noise_uv[np.maximum(channels, 0)], .01)} if self.backfill else {}))
            else:
                self.observe(core.data_start+t, core.core_start, channels, waveform)

    def evaluate(self, models, now, *, enroll=False):
        self.expire(now)
        audit, added = [], 0
        self.recovered = self.recovered[:0]
        recovery_batches = []
        feature_batches = []
        self.recovered_features = np.empty((0,3,min(16,models.channels.shape[1])),np.float32)
        for key, p in sorted(self.proposals.items()):
            rows = list(p['rows'])
            cores = sorted({row[1] for row in rows})
            record = dict(proposal_id=key, anchor=int(p['channels'][0]),
                          support=len(rows), enrolled_unit=None,
                          first_sample=rows[0][0], last_sample=rows[-1][0])
            audit.append(record)
            if len(rows) < 24 or len(cores) < 3:
                record['reason'] = 'insufficient_temporal_support'
                continue
            split = cores[(2*len(cores))//3]
            train = [r[2] for r in rows if r[1] < split]
            valid = [r[2] for r in rows if r[1] >= split]
            if len(train) < 12 or len(valid) < 6:
                record['reason'] = 'insufficient_holdout'
                continue
            template = np.median(np.stack(train), axis=0)
            record.update(training_count=len(train), validation_count=len(valid))
            if self.raw_anchor_holdout:
                raw_consistency = raw_anchor_consistency(template, [r[3] for r in rows if r[1] >= split])
                record['raw_anchor_consistency'] = raw_consistency
                if raw_consistency < self.raw_holdout_minimum_cosine:
                    record['reason'] = 'raw_anchor_holdout_rejected'
                    self.retired_raw_holdout += 1
                    continue
            consistency = float(np.quantile([
                similarity(template, p['channels'], w, p['channels']) for w in valid], .1))
            ids = sorted({unit for c in p['channels'] if c >= 0
                          for unit in models.by_channel.get(int(c), ())})
            duplicate = max((similarity(template, p['channels'], models.waveforms[u],
                                        models.channels[u]) for u in ids), default=0.)
            violations = float(np.mean(np.diff([r[0] for r in rows]) < .001*self.rate))
            record.update(consistency=consistency, existing_similarity=duplicate,
                          training_count=len(train), validation_count=len(valid),
                          sampled_refractory_violation_fraction=violations)
            rejected_duplicate = duplicate >= .85
            if self.residual_duplicates and consistency >= .9 and rejected_duplicate and violations <= .01:
                from .session_novelty_residual import distinguish_from_existing
                residual = distinguish_from_existing(template,p['channels'],rows,models,ids)
                record['duplicate_residual_validation'] = residual
                rejected_duplicate = not residual['distinguishable']
            if consistency < .9 or rejected_duplicate or violations > .01:
                record['reason'] = 'consistency_duplicate_or_refractory_rejected'
                continue
            record['reason'] = 'eligible_shadow'
            if not enroll:
                continue
            if (added >= 8 or len(models.waveforms) >= 4096 or
                    any(len(models.by_channel.get(int(c), ())) >= 512
                        for c in p['channels'] if c >= 0)):
                record['reason'] = 'active_model_budget'
                continue
            record.update(reason='experimental_enrollment', enrolled_unit=len(models.waveforms))
            record['supporting_samples'] = [row[0] for row in rows]
            models.waveforms = np.concatenate((models.waveforms, template[None]))
            models.channels = np.concatenate((models.channels, p['channels'][None]))
            models.anchors = np.append(models.anchors, p['channels'][0])
            models.assigned = np.append(models.assigned, np.int64(0))
            models.version += 1
            models.__post_init__()
            if self.backfill:
                from .session_recovery import fit_observations
                recovered = fit_observations(models, record['enrolled_unit'], p['channels'], rows)
                record['backfill_samples'] = recovered['sample_index'].tolist()
                record['backfill_rejected_count'] = len(rows)-len(recovered)
                recovery_batches.append(recovered)
                if self.curation_basis is not None:
                    from .session_phy_capture import project_observations
                    feature_batches.append(project_observations(
                        rows, recovered["sample_index"], self.curation_basis,
                        min(16,models.channels.shape[1])))
                models.assigned[record['enrolled_unit']] += len(recovered)
            added += 1
        for row in audit:
            if row['enrolled_unit'] is not None or row['reason'] == 'raw_anchor_holdout_rejected':
                del self.proposals[row['proposal_id']]
        if recovery_batches:
            self.recovered = np.concatenate(recovery_batches)
            order = np.argsort(self.recovered['sample_index'], kind='stable')
            self.recovered = self.recovered[order]
            if self.curation_basis is not None:
                self.recovered_features = np.concatenate(feature_batches)[order]
        return audit

    def save(self, handle, audit):
        group = handle.create_group('novelty')
        if self.residual_duplicates:
            from .session_novelty_residual import RULE
            group.attrs['residual_duplicate_algorithm'] = RULE
        group.attrs.update(version=4 if self.backfill else (3 if self.raw_anchor_holdout else (1 if self.temporal_basis is None else 2)),
                           next_id=self.next_id, sample_rate=self.rate,
                           max_proposals=self.max_proposals)
        if self.peak_alignment:
            group.attrs['peak_alignment_algorithm'] = PEAK_ALIGNMENT_RULE
        if self.backfill:
            group.attrs['backfill'] = True
        if self.raw_anchor_holdout:
            group.attrs.update(raw_anchor_holdout=True, retired_raw_holdout=self.retired_raw_holdout,
                               raw_holdout_minimum_cosine=self.raw_holdout_minimum_cosine)
        if self.stratified_admission:
            group.attrs['stratified_admission'] = True
        if self.priority_admission:
            group.attrs['priority_admission'] = True
        if self.preserve_strong_tentative:
            group.attrs['preserve_strong_tentative'] = True
        if self.tentative_seconds or self.track_snr:
            group.attrs.update(tentative_seconds=self.tentative_seconds,
                               retired_tentative=self.retired_tentative)
        if self.candidate_floor_snr != 6.:
            group.attrs['candidate_floor_snr'] = self.candidate_floor_snr
        if self.temporal_basis is not None:
            group.create_dataset('temporal_basis', data=self.temporal_basis)
        group.create_dataset('audit_json', data=json.dumps(audit, sort_keys=True))
        for key, p in sorted(self.proposals.items()):
            child = group.create_group(str(key))
            if self.track_snr:
                child.attrs['reference_snr'] = p['reference_snr']
            child.create_dataset('reference', data=p['reference'])
            child.create_dataset('channels', data=p['channels'])
            child.create_dataset('coordinates', data=np.array([(r[0], r[1]) for r in p['rows']], np.int64))
            child.create_dataset('waveforms', data=np.stack([r[2] for r in p['rows']]), compression='lzf')
            if self.raw_anchor_holdout:
                child.create_dataset('raw_anchor_windows', data=np.stack([r[3] for r in p['rows']]), compression='lzf')
            if self.backfill:
                child.create_dataset('raw_counts', data=np.stack([r[4] for r in p['rows']]), compression='lzf')
                child.create_dataset('raw_scales', data=np.array([r[5] for r in p['rows']], np.float32))
                child.create_dataset('raw_noise_uv', data=np.stack([r[6] for r in p['rows']]), compression='lzf')

    def restore(self, handle):
        group = handle['novelty']
        from .session_novelty_residual import RULE
        if group.attrs.get('residual_duplicate_algorithm','') != (RULE if self.residual_duplicates else ''):
            raise ValueError('Novelty checkpoint configuration mismatch')
        if (group.attrs['version'] != (4 if self.backfill else (3 if self.raw_anchor_holdout else (1 if self.temporal_basis is None else 2)))
                or group.attrs['sample_rate'] != self.rate
                or group.attrs['max_proposals'] != self.max_proposals
                or group.attrs.get('tentative_seconds', 0.) != self.tentative_seconds
                or group.attrs.get('candidate_floor_snr', 6.) != self.candidate_floor_snr
                or bool(group.attrs.get('stratified_admission', False)) != self.stratified_admission
                or bool(group.attrs.get('priority_admission', False)) != self.priority_admission
                or bool(group.attrs.get('preserve_strong_tentative', False)) != self.preserve_strong_tentative
                or group.attrs.get('peak_alignment_algorithm', '') != (
                    PEAK_ALIGNMENT_RULE if self.peak_alignment else '')
                or bool(group.attrs.get('backfill', False)) != self.backfill
                or bool(group.attrs.get('raw_anchor_holdout', False)) != self.raw_anchor_holdout
                or (self.raw_anchor_holdout and group.attrs.get('raw_holdout_minimum_cosine', .9)
                    != self.raw_holdout_minimum_cosine)):
            raise ValueError('Novelty checkpoint configuration mismatch')
        if self.temporal_basis is not None:
            basis = group['temporal_basis'][:]
            if (basis.shape != (self.nt, 3) or not np.isfinite(basis).all()
                    or not np.allclose(basis.T @ basis, np.eye(3), atol=1.e-5)):
                raise ValueError('Invalid novelty temporal basis checkpoint')
            self.temporal_basis = basis
        self.retired_tentative = int(group.attrs.get('retired_tentative', 0))
        if self.retired_tentative < 0:
            raise ValueError('Invalid tentative retirement checkpoint')
        self.retired_raw_holdout = int(group.attrs.get('retired_raw_holdout', 0))
        if self.retired_raw_holdout < 0:
            raise ValueError('Invalid raw holdout retirement checkpoint')
        self.next_id = int(group.attrs['next_id'])
        self.proposals.clear()
        if self.backfill and len([key for key in group if key not in ('audit_json', 'temporal_basis')]) > self.max_proposals:
            raise ValueError('Oversized retrospective proposal checkpoint')
        for key in group:
            if key in ('audit_json', 'temporal_basis'):
                continue
            child = group[key]
            if self.backfill:
                width = child['channels'].shape[0]
                count = child['coordinates'].shape[0]
                if (not 1 <= width <= 16 or not 1 <= count <= 64
                        or child['coordinates'].shape != (count, 2)
                        or child['waveforms'].shape != (count, self.nt, width)
                        or child['reference'].shape != (self.nt, width)
                        or child['raw_counts'].shape != (count, self.nt, width)
                        or child['raw_scales'].shape != (count,)
                        or child['raw_noise_uv'].shape != (count, width)
                        or child['raw_anchor_windows'].shape != (count, 21)):
                    raise ValueError('Oversized or invalid retrospective observation checkpoint')
            self.proposals[int(key)] = dict(reference=child['reference'][:],
                channels=child['channels'][:], rows=deque([
                    (int(t), int(c), w) for (t, c), w in zip(
                        child['coordinates'][:], child['waveforms'][:])], maxlen=64))
            if self.track_snr:
                self.proposals[int(key)]['reference_snr'] = validate_candidate_floor(child.attrs['reference_snr'])
            if self.raw_anchor_holdout:
                windows = child['raw_anchor_windows'][:]
                rows = self.proposals[int(key)]['rows']
                if windows.shape != (len(rows), 21) or not np.isfinite(windows).all():
                    raise ValueError('Invalid raw anchor holdout checkpoint')
                self.proposals[int(key)]['rows'] = deque([row+(window,)
                    for row, window in zip(rows, windows)], maxlen=64)
            if self.backfill:
                rows = self.proposals[int(key)]['rows']
                width = len(self.proposals[int(key)]['channels'])
                counts, scales, noise = child['raw_counts'][:], child['raw_scales'][:], child['raw_noise_uv'][:]
                if (width > 16 or counts.shape != (len(rows), self.nt, width) or counts.dtype != np.int16
                        or scales.shape != (len(rows),) or not np.isfinite(scales).all() or np.any(scales <= 0)
                        or noise.shape != (len(rows), width) or not np.isfinite(noise).all() or np.any(noise <= 0)):
                    raise ValueError('Invalid retrospective observation checkpoint')
                self.proposals[int(key)]['rows'] = deque([row+(data, scale, n)
                    for row, data, scale, n in zip(rows, counts, scales, noise)], maxlen=64)
