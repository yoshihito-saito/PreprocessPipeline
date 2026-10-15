"""Experimental heldout evidence for rescuing cosine-rejected novelty.

The sign-tail and BIC-style penalty are screening scores, not calibrated
biological identity probabilities. Independent sorting quality gates remain
mandatory. Work is bounded by existing compact observations and local models.
"""
from math import comb, log
import numpy as np

RULE = 'compact_raw_holdout_gain9_sign01_pca3_equivalence_guard_v2'
MAX_OBSERVATIONS = 64
MAX_CONTACTS = 16
MAX_COMPETITORS = 512
MODEL_SLAB = 16

def policy():
    return dict(algorithm=RULE, minimum_holdout_observations=8,
        maximum_existing_equivalence_cosine=.93,
        minimum_holdout_cores=2, minimum_gain_advantage=9., maximum_sign_tail=.01,
        penalty='3*usable_contacts*log(heldout_scalar_observations)',
        maximum_competitors=MAX_COMPETITORS, model_slab=MODEL_SLAB,
        comparison='best_positive_amplitude_pm2_per_observation_all_local_models')

def _gains(data, noise, templates):
    weights = 1./noise.astype(np.float64)**2
    weighted = data.astype(np.float64)*weights[:,None,:]
    value = templates.astype(np.float64)
    dot = np.einsum('ntc,utc->nu', weighted, value)
    energy = np.einsum('nc,utc->nu', weights, value*value)
    return np.maximum(dot,0.)**2 / np.maximum(energy,1.e-12)

def _project(waveform, contacts, observed, shift):
    projected = np.zeros((len(waveform),len(observed)),np.float32)
    common, src, dst = np.intersect1d(contacts,observed,return_indices=True)
    use = common >= 0; src, dst = src[use], dst[use]
    if shift > 0: projected[shift:,dst] = waveform[:-shift,src]
    elif shift < 0: projected[:shift,dst] = waveform[-shift:,src]
    else: projected[:,dst] = waveform[:,src]
    return projected

def distinguish_from_existing(template, channels, observations, models, competitors):
    """Return affirmative heldout evidence only; uncertainty keeps old rejection.

    Only rows after the existing temporal split are scored. All local models,
    including those with a lower cosine, compete on the same observed contacts
    and receive free nonnegative amplitudes and +/-2-sample alignment. Padding
    is never treated as an observed channel. Raw snippets remain INT16 in
    storage; only these at-most64 selected snippets are decoded for scoring.
    """
    template = np.asarray(template,np.float32)
    channels = np.asarray(channels)
    count = len(observations)
    if (template.ndim != 2 or template.shape[0] != models.nt
        or not 1 <= template.shape[1] <= MAX_CONTACTS
        or channels.shape != (template.shape[1],)
        or not np.issubdtype(channels.dtype,np.integer)
        or np.any(channels < -1) or not np.isfinite(template).all()
        or not 1 <= count <= MAX_OBSERVATIONS):
        raise ValueError('Invalid bounded novelty residual input')
    valid = channels >= 0
    if len(np.unique(channels[valid])) != int(valid.sum()):
        raise ValueError('Duplicate observed novelty contacts')
    if len(models.waveforms) > 4096:
        raise ValueError('Oversized novelty model bank')
    ids = np.asarray(competitors)
    if ids.size == 0: ids = np.empty(0,np.int64)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype,np.integer)
        or np.any(ids < 0) or np.any(ids >= len(models.waveforms))
        or len(np.unique(ids)) != len(ids)):
        raise ValueError('Invalid novelty residual competitor IDs')
    record = dict(algorithm=RULE, distinguishable=False, competitor_count=len(ids))
    if len(ids) > MAX_COMPETITORS:
        return {**record, 'reason':'competitor_budget_uncertain'}
    if valid.sum() < 2 or not len(ids):
        return {**record, 'reason':'insufficient_contact_or_competitor_evidence'}
    # An alternative already inside the selected sorter's template-equivalence
    # threshold cannot be asserted to be a new identity on residual improvement
    # alone. More accurate fitting is also expected for the same drifting cell.
    from .session_novelty import similarity
    equivalent = max(similarity(template,channels,models.waveforms[u],models.channels[u]) for u in ids)
    record['maximum_existing_similarity'] = equivalent
    if equivalent >= .93:
        return {**record,'reason':'existing_equivalent_no_forced_new_identity'}
    coordinates = [(int(r[0]),int(r[1])) for r in observations]
    if any(t < 0 or core < 0 for t,core in coordinates) or any(
        coordinates[i][0] <= coordinates[i-1][0] for i in range(1,count)):
        raise ValueError('Invalid novelty residual source coordinates')
    cores = sorted({c for _,c in coordinates})
    split = cores[(2*len(cores))//3]
    rows = [r for r in observations if int(r[1]) >= split]
    record.update(holdout_observations=len(rows),
        holdout_cores=len({int(r[1]) for r in rows}))
    if count < 24 or len(cores) < 3 or count-len(rows) < 12 or len(rows) < 8 or record['holdout_cores'] < 2:
        return {**record, 'reason':'insufficient_raw_holdout'}
    raw = np.empty((len(rows),models.nt,int(valid.sum())),np.float32)
    noise = np.empty((len(rows),int(valid.sum())),np.float32)
    for index,row in enumerate(rows):
        if len(row) != 7: raise ValueError('Missing compact raw novelty observation')
        counts, scale, sigma = row[4:7]
        counts, sigma = np.asarray(counts), np.asarray(sigma)
        if (counts.dtype != np.int16 or counts.shape != template.shape
            or np.ndim(scale) != 0 or not np.isfinite(scale) or scale <= 0
            or sigma.shape != channels.shape or not np.isfinite(sigma[valid]).all()
            or np.any(sigma[valid] <= 0)):
            raise ValueError('Invalid compact novelty observation payload')
        raw[index] = counts[:,valid].astype(np.float32)*float(scale)
        noise[index] = sigma[valid]
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite decoded novelty payload')
    observed = channels[valid]
    own = np.zeros(len(rows),np.float64)
    for shift in range(-2,3):
        own = np.maximum(own,_gains(raw,noise,
            _project(template,channels,observed,shift)[None])[:,0])
    prior = np.zeros(len(rows),np.float64)
    for first in range(0,len(ids),MODEL_SLAB):
        slab = ids[first:first+MODEL_SLAB]
        for shift in range(-2,3):
            projected = np.stack([_project(models.waveforms[u],models.channels[u],observed,shift)
                for u in slab])
            prior = np.maximum(prior,_gains(raw,noise,projected).max(axis=1))
    delta = own-prior
    wins = int(np.count_nonzero(delta > 9.))
    tail = sum(comb(len(rows), k) for k in range(wins,len(rows)+1)) / 2.**len(rows)
    penalty = 3*len(observed)*log(len(rows)*models.nt*len(observed))
    total = float(delta.sum())
    distinct = bool(tail <= .01 and total > penalty)
    return dict(**{k:v for k,v in record.items() if k!='distinguishable'},
        reason='heldout_residual_advantage' if distinct else 'existing_or_uncertain',
        distinguishable=distinct, wins_above_noise_margin=wins, sign_tail=tail,
        total_gain_advantage=total, complexity_penalty=penalty,
        median_gain_advantage=float(np.median(delta)),
        median_gain_fraction=float(np.median(delta/np.maximum(own,1.e-12))))
