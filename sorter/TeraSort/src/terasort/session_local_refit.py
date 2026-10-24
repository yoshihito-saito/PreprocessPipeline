"""Bounded CPU coordinate refit of ambiguous assignments after subtraction.

This is a slow experimental post-pass, not a joint global optimizer.
"""
import numpy as np

from .session_models import Match, _score_one


def local_refit(signal, noise, models, matches, *, core_start, core_stop,
                rounds=1, budget=64, half_width=8, shift_radius=2,
                score_floor=.65, min_margin=.03, gain_floor=20.25,
                amplitude_min=.3, amplitude_max=3., refractory_samples=16):
    if not 0 <= rounds <= 2 or not 1 <= budget <= 128:
        raise ValueError('Bounded refit requires 0–2 rounds and budget 1–128')
    rows = list(matches)
    for _ in range(rounds):
        rows.sort(key=lambda m:(m.source_sample,m.channel,m.pass_index))
        selected = sorted((i for i,m in enumerate(rows) if m.relative_margin < .2 or m.score < .85),
                          key=lambda i:(rows[i].relative_margin,rows[i].source_sample,i))[:budget]
        changes = 0
        for index in selected:
            old = rows[index]
            t = old.source_sample
            # Halo assignments are not in the published list; avoid their reach.
            if t < core_start+models.nt+shift_radius or t >= core_stop-models.nt-shift_radius:
                continue
            lo, hi = t-models.nt//2-shift_radius, t+models.nt//2+1+shift_radius
            patch = signal[lo:hi].copy()
            nearby = [(j,m) for j,m in enumerate(rows)
                      if j != index and abs(m.source_sample-t) <= models.nt-1+shift_radius]
            for j, neighbor in nearby:
                first, stop = max(lo,neighbor.source_sample-models.nt//2), min(hi,neighbor.source_sample+models.nt//2+1)
                if first >= stop:
                    continue
                channels = models.channels[neighbor.unit]
                valid = channels >= 0
                patch[first-lo:stop-lo, channels[valid]] -= neighbor.amplitude * models.waveforms[
                    neighbor.unit, first-neighbor.source_sample+models.nt//2:stop-neighbor.source_sample+models.nt//2][:,valid]
            center = t-lo
            _,old_optimal_amp,old_optimal_gain = _score_one(
                patch,center,models.waveforms[old.unit],models.channels[old.unit],
                half_width=half_width,noise_uv=noise)
            ratio = old.amplitude/old_optimal_amp if old_optimal_amp > 0 else 0.
            old_gain = old_optimal_gain*(2*ratio-ratio**2)
            fits = []
            ids = models.by_channel.get(old.channel, ())
            if len(ids) > 512:
                raise OverflowError('Refit spatial partition exceeds 512 models')
            for unit in ids:
                best = None
                for shift in range(-shift_radius,shift_radius+1):
                    sample = t+shift
                    if any(m.unit == unit and abs(m.source_sample-sample) <= refractory_samples
                           for j,m in nearby):
                        continue
                    score,amp,gain = _score_one(patch,center+shift,models.waveforms[unit],
                                               models.channels[unit],half_width=half_width,noise_uv=noise)
                    if score < max(.85,score_floor) or not amplitude_min <= amp <= amplitude_max:
                        continue
                    fit = (gain,score,unit,amp,shift)
                    if best is None or (gain,-abs(shift),-shift) > (best[0],-abs(best[4]),-best[4]):
                        best = fit
                if best is not None:
                    fits.append(best)
            fits.sort(key=lambda f:(-f[0],f[2]))
            if not fits:
                continue
            gain,score,unit,amp,shift = fits[0]
            margin = 1-fits[1][0]/gain if len(fits)>1 and gain>0 else 1.
            if gain < max(gain_floor,1.05*old_gain) or margin < max(.1,min_margin):
                continue
            rows[index] = Match(t+shift,old.channel,unit,score, fits[1][1] if len(fits)>1 else -1.,
                                amp,gain,old.pass_index,old.candidate_sample,margin)
            changes += 1
        if not changes:
            break
    return sorted(rows,key=lambda m:(m.source_sample,m.channel,m.pass_index))
