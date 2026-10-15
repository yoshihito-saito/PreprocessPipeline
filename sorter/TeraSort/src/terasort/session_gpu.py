"""Direct-CUDA bounded residual matcher for the experimental session path."""

from __future__ import annotations

from bisect import bisect_left, insort
from pathlib import Path

import numpy as np

from .cuda_environment import configure_cupy_cache
from .session_joint_fit import JointFit
from .session_models import Match, normalize_template_identity_map


def _best_template_per_identity(event_ids, template_ids, gains, n_templates,
                                identity_map):
    positions = np.arange(len(gains), dtype=np.int64)
    if identity_map is None:
        return event_ids, gains, positions
    identities = normalize_template_identity_map(identity_map, n_templates)
    if len(np.unique(identities)) == n_templates:
        return event_ids, gains, positions
    _, template_groups = np.unique(identities, return_inverse=True)
    group_count = int(template_groups.max()) + 1
    keys = (event_ids.astype(np.int64) * group_count
            + template_groups[template_ids])
    _, inverse = np.unique(keys, return_inverse=True)
    group_gain = np.zeros(int(inverse.max()) + 1, np.float32)
    np.maximum.at(group_gain, inverse, gains)
    group_position = np.full(len(group_gain), len(gains), np.int64)
    np.minimum.at(group_position, inverse, np.where(
        (gains > 0) & (gains == group_gain[inverse]), positions, len(gains)))
    selected = group_position[group_position < len(gains)]
    return event_ids[selected], gains[selected], selected


class CudaResidualMatcher:
    """Keep local templates resident; score sparse pairs, subtract independent fits."""

    _kernel_prefix = ""
    _sample_dtype_name = "float32"

    def __init__(self, models, *, pair_batch=65_536, max_pairs=2_000_000,
                 template_identity_map=None, template_contact_scoring=False, recovery=None):
        configure_cupy_cache()
        import cupy as cp
        if not isinstance(template_contact_scoring, bool):
            raise ValueError('Template contact scoring must be boolean')
        if models.waveforms.shape[2] > 32:
            raise ValueError("CUDA local template width must be <= 32")
        if recovery is not None and self._sample_dtype_name != "int16":
            raise ValueError("Spatial recovery requires INT16 residual storage")
        self.recovery = recovery
        self.learning_events = self.learning_matches = None
        self.cp = cp
        self.models = models
        self.template_identity_map = normalize_template_identity_map(
            template_identity_map, len(models.waveforms))
        self.pair_batch = int(pair_batch)
        self.max_pairs = int(max_pairs)
        if self.pair_batch < 1 or self.max_pairs < self.pair_batch:
            raise ValueError("Invalid CUDA pair budget")
        code = Path(__file__).with_name("session_match.cu").read_text()
        score_prefix = ('#define SESSION_CONTACT_SCORE 1\n'
                        if template_contact_scoring else '')
        module = cp.RawModule(code=self._kernel_prefix + f"#define SESSION_NT {models.nt}\n" + score_prefix + code,
                              options=("--std=c++11", "--fmad=false"))
        self.score_kernel = module.get_function("score_residual_pairs")
        self.select_kernel = module.get_function("select_identity_winners")
        self.expand_kernel = module.get_function("expand_template_pairs")
        self.joint_kernel = None
        self.subtract_kernel = module.get_function("subtract_fits")
        self.detect_kernel = module.get_function("detect_residual_peaks")
        self.detect_smooth3_kernel = module.get_function("detect_residual_peaks_smooth3_shared")
        self.detect_buffer = None
        self.detect_count = cp.zeros(1, cp.uint64)
        self.template_version = -1
        self.templates = None
        self.channels = None
        self._proposal_key = None
        self._template_models = None

    def _upload_templates(self):
        return self.cp.asarray(self.models.waveforms)

    def _prepare_residual(self, voltage_uv):
        return self.cp.asarray(voltage_uv, dtype=self.cp.float32, order='C')

    def _channel_values(self, residual, rows, channels=slice(None)):
        return residual[rows, channels]

    def _extra_kernel_args(self):
        return ()

    def _check_subtraction(self):
        pass

    def _refresh(self):
        if self._template_models is not self.models or self.template_version != self.models.version:
            self.templates = self._upload_templates()
            self.channels = self.cp.asarray(self.models.channels)
            identities = (normalize_template_identity_map(self.template_identity_map, len(self.models.waveforms))
                          if self.template_identity_map is not None
                          else np.arange(len(self.models.waveforms), dtype=np.int64))
            self.device_identities = self.cp.asarray(identities, dtype=self.cp.int64)
            self.template_version = self.models.version
            self._template_models = self.models

    def _refresh_proposals(self, n_channels):
        key = (id(self.models), self.models.version, id(self.models.by_channel), n_channels)
        if key == self._proposal_key:
            return
        groups = [self.models.by_channel.get(c, ()) for c in range(n_channels)]
        counts = np.fromiter((len(group) for group in groups), dtype=np.int32,
                             count=n_channels)
        if np.any(counts > 512):
            raise OverflowError("Spatial model partition exceeds 512 units")
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        if offsets[-1] > np.iinfo(np.int32).max:
            raise OverflowError("Channel/template index exceeds INT32 capacity")
        units = np.fromiter((unit for group in groups for unit in group),
                            dtype=np.int32, count=int(offsets[-1]))
        if np.any((units < 0) | (units >= len(self.models.waveforms))):
            raise ValueError("Invalid template proposal index")
        self._proposal_counts = counts
        self._proposal_offsets = self.cp.asarray(offsets, dtype=self.cp.int32)
        self._proposal_units = self.cp.asarray(units)
        self._proposal_key = key

    def _estimate_smooth3_noise(self, residual, noise_uv):
        cp = self.cp
        noise = np.asarray(noise_uv, np.float32)
        if (noise.shape != (residual.shape[1],)
                or np.any(np.isnan(noise) | (noise <= 0))):
            raise ValueError('Positive channel noise required; infinity masks contacts')
        nt = residual.shape[0]
        step = max(1, nt // 4000)
        sample_rows = cp.arange(0, nt, step, dtype=cp.int64)
        sample = self._channel_values(residual, sample_rows).copy()
        interior = (sample_rows > 0) & (sample_rows < nt - 1)
        rows = sample_rows[interior]
        if rows.size:
            sample[interior] = .25 * (
                self._channel_values(residual, rows - 1)
                + 2 * self._channel_values(residual, rows)
                + self._channel_values(residual, rows + 1))
        sample = cp.asnumpy(sample)
        center = np.median(sample, axis=0)
        estimate = np.median(np.abs(sample - center), axis=0) / .67448975
        return np.where(np.isfinite(noise),
                        np.maximum(estimate, .01), np.inf).astype(np.float32)

    def detect(self, residual, noise_uv, floor_snr, *, max_candidates=500_000,
               mode='raw', smooth3_noise_uv=None):
        cp = self.cp
        if mode not in ('raw','smooth3'):
            raise ValueError('Unknown detector mode')
        if (residual.dtype != getattr(cp, self._sample_dtype_name) or not residual.flags.c_contiguous
                or residual.ndim != 2 or not residual.size or max_candidates < 1
                or not np.isfinite(floor_snr) or floor_snr <= 0):
            raise ValueError('Invalid contiguous residual dtype or detection budget')
        noise = np.asarray(noise_uv, np.float32)
        if noise.shape != (residual.shape[1],) or np.any(np.isnan(noise) | (noise <= 0)):
            raise ValueError('Positive channel noise required; infinity masks contacts')
        if mode == 'smooth3':
            if smooth3_noise_uv is None:
                noise = self._estimate_smooth3_noise(residual, noise)
            else:
                noise = np.asarray(smooth3_noise_uv, np.float32)
                if (noise.shape != (residual.shape[1],)
                        or np.any(np.isnan(noise) | (noise <= 0))):
                    raise ValueError('Invalid precomputed smooth3 noise')
        elif smooth3_noise_uv is not None:
            raise ValueError('Precomputed smooth3 noise requires smooth3 mode')
        nt, nc = residual.shape
        device_noise = cp.asarray(noise)
        capacity = min(residual.size, max_candidates)
        if self.detect_buffer is None or len(self.detect_buffer) != capacity:
            self.detect_buffer = cp.empty(capacity, cp.int64)
        self.detect_count.fill(0)
        if mode == 'smooth3':
            self.detect_smooth3_kernel(((nc + 31) // 32, (nt + 7) // 8), (32, 8),
                (residual, device_noise, np.int64(nt), np.int32(nc),
                 np.float32(floor_snr), self.detect_count, self.detect_buffer,
                 np.uint64(capacity), *self._extra_kernel_args()))
        else:
            self.detect_kernel(((residual.size+255)//256,), (256,),
                (residual, device_noise, np.int64(nt), np.int32(nc),
                 np.float32(floor_snr), self.detect_count, self.detect_buffer,
                 np.uint64(capacity), *self._extra_kernel_args()))
        count = int(self.detect_count.get()[0])
        if count > capacity:
            raise OverflowError("Artifact burst exceeds per-core candidate budget")
        flat = cp.sort(self.detect_buffer[:count])
        indices = cp.asnumpy(flat)
        t, c = np.divmod(indices, nc)
        if mode == 'smooth3':
            event_t = flat // nc
            event_c = flat % nc
            center_value = self._channel_values(residual, event_t, event_c).copy()
            interior = (event_t > 0) & (event_t < nt - 1)
            rows = event_t[interior]
            channels = event_c[interior]
            center_value[interior] = .25 * (
                self._channel_values(residual, rows - 1, channels)
                + 2 * self._channel_values(residual, rows, channels)
                + self._channel_values(residual, rows + 1, channels))
            snr = cp.asnumpy(cp.abs(center_value) / device_noise[event_c])
        else:
            snr = cp.asnumpy(
                cp.abs(self._channel_values(residual, flat // nc, flat % nc))
                / device_noise[flat % nc])
        return list(zip(t.astype(np.int32), c.astype(np.int32),
                        snr.astype(np.float32)))

    def _score(self, residual, events, *, half_width=8, noise_uv=None,
               shift_radius=2, score_floor=.65, amplitude_min=.3, amplitude_max=3.):
        cp = self.cp
        self._refresh()
        radius_values = np.asarray(shift_radius)
        if radius_values.ndim == 0:
            if (not np.issubdtype(radius_values.dtype, np.integer)
                    or not 0 <= int(radius_values) <= 8):
                raise ValueError('Timing search radius must be an integer from 0 to 8')
            device_radii = cp.full(len(self.models.waveforms), int(radius_values),
                                   dtype=cp.int32)
        else:
            if (radius_values.shape != (len(self.models.waveforms),)
                    or not np.issubdtype(radius_values.dtype, np.integer)
                    or np.any((radius_values < 0) | (radius_values > 8))):
                raise ValueError('Per-unit timing radii must match model count and be from 0 to 8')
            device_radii = cp.asarray(radius_values, dtype=cp.int32)
        # Keep all pairs for one event together so the identity-aware top-two
        # reduction is exact. Flush at a bounded pair count; a long core or a
        # larger seed bank must not allocate a core-sized pair array.
        choices = [None] * len(events)
        nt = len(residual)
        nc = residual.shape[1]
        self._refresh_proposals(nc)
        times = np.fromiter((event[0] for event in events), dtype=np.int32, count=len(events))
        contacts = np.fromiter((event[1] for event in events), dtype=np.int32, count=len(events))
        valid = (times >= self.models.nt // 2) & (times < nt-self.models.nt // 2) & (contacts >= 0) & (contacts < nc)
        counts = np.zeros(len(events), np.int32)
        counts[valid] = self._proposal_counts[contacts[valid]]
        batch_limit = min(max(self.pair_batch, 512), self.max_pairs)
        if np.any(counts > batch_limit):
            raise OverflowError("Single event exceeds bounded pair budget")
        prefix = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        pair_count = int(prefix[-1])
        proposed_count = int(np.count_nonzero(counts))
        weights = cp.ones(residual.shape[1], cp.float32) if noise_uv is None else cp.asarray(
            1. / np.maximum(np.asarray(noise_uv, np.float32), .01) ** 2)
        events_t = cp.asarray(times)
        event_channels = cp.asarray(contacts)
        width = self.models.waveforms.shape[2]
        batch_first = 0
        while batch_first < len(events):
            batch_stop = int(np.searchsorted(prefix, prefix[batch_first]+batch_limit,
                                              side="right"))-1
            local_count = batch_stop-batch_first
            n = int(prefix[batch_stop]-prefix[batch_first])
            if n == 0:
                break
            offsets = cp.asarray(prefix[batch_first:batch_stop+1]-prefix[batch_first],
                                 dtype=cp.int32)
            pe, pu = cp.empty(n, cp.int32), cp.empty(n, cp.int32)
            self.expand_kernel((local_count,), (64,),
                (event_channels, offsets, self._proposal_offsets, self._proposal_units,
                 np.int32(batch_first), pe, pu))
            ds = cp.empty(n, cp.float32)
            da = cp.empty(n, cp.float32)
            dg = cp.empty(n, cp.float32)
            dt = cp.empty(n, cp.int32)
            self.score_kernel(((n + 3) // 4,), (128,),
                (residual, events_t, self.templates, self.channels,
                 pe, pu, weights, np.int32(n), np.int32(nt),
                 np.int32(residual.shape[1]), np.int32(width),
                 np.int32(half_width), device_radii,
                 np.float32(score_floor), np.float32(amplitude_min),
                 np.float32(amplitude_max), ds, da, dg, dt,
                 *self._extra_kernel_args()))
            # Reduce on-device. Transfer one compact winner/runner-up record
            # per event, rather than four arrays for every template proposal.
            selected = cp.empty((local_count, 8), cp.float32)
            self.select_kernel(((local_count + 127) // 128,), (128,),
                (offsets, pu, self.device_identities, ds, da, dg, dt,
                 np.int32(local_count), selected))
            selected_host = cp.asnumpy(selected)
            integer_fields = selected_host.view(np.int32)
            # Convert only eligible winners; the packed result still retains
            # every event and its original batch index/order.
            for local in np.flatnonzero(selected_host[:, 7]):
                row = selected_host[local]
                choices[batch_first + local] = (
                    float(row[0]), float(row[1]), int(integer_fields[local, 2]),
                    float(row[3]), float(row[4]), int(integer_fields[local, 5]),
                    float(1 - row[6] / row[1]))
            batch_first = batch_stop
        self.last_score_diagnostics = {
            'events_with_template_proposals': proposed_count,
            'event_template_pairs': pair_count,
        }
        return choices

    def score_joint_pairs(self, residual, pairs, noise_uv, *,
                          blocker_amplitudes=None,
                          amplitude_min=.3, amplitude_max=3.,
                          minimum_incremental_gain=20.25,
                          minimum_incremental_fraction=.015,
                          min_center_separation_samples=3,
                          max_template_correlation=.98):
        """Score bounded candidate/blocker pairs without changing assignments.

        Each row is (center_a, unit_a, center_b, unit_b) in the local core.
        Pair selection and conflict resolution remain the caller's job. When
        residual already has every accepted spike subtracted, pass each
        blocker's original fitted amplitude to restore just that blocker in
        the two-template comparison. Omit it for a raw preprocessed signal.
        """
        cp = self.cp
        self._refresh()
        entries = np.asarray(pairs)
        if entries.size == 0:
            entries = np.empty((0, 4), np.int32)
        if (entries.ndim != 2 or entries.shape[1] != 4
                or not np.issubdtype(entries.dtype, np.integer)
                or np.any(entries[:, (0, 2)] < 0)
                or np.any(entries[:, (0, 2)] >= len(residual))
                or np.any(entries[:, (1, 3)] < 0)
                or np.any(entries[:, (1, 3)] >= len(self.models.waveforms))
                or not isinstance(min_center_separation_samples, (int, np.integer))
                or isinstance(min_center_separation_samples, (bool, np.bool_))
                or not 0 <= min_center_separation_samples <= self.models.nt - 1
                or not np.isfinite(amplitude_min) or not np.isfinite(amplitude_max)
                or not 0 < amplitude_min <= amplitude_max
                or not np.isfinite(minimum_incremental_gain)
                or minimum_incremental_gain < 0
                or not np.isfinite(minimum_incremental_fraction)
                or not 0 <= minimum_incremental_fraction < 1
                or not np.isfinite(max_template_correlation)
                or not 0 <= max_template_correlation < 1
                or residual.dtype != getattr(cp, self._sample_dtype_name) or residual.ndim != 2
                or not residual.flags.c_contiguous):
            raise ValueError("Invalid bounded joint-score inputs")
        noise = np.asarray(noise_uv, np.float32)
        if (noise.shape != (residual.shape[1],)
                or np.any(np.isnan(noise) | (noise <= 0))):
            raise ValueError("Positive channel noise required; infinity masks contacts")
        baseline = (np.zeros(len(entries), np.float32) if blocker_amplitudes is None
                    else np.asarray(blocker_amplitudes, np.float32))
        if (baseline.shape != (len(entries),) or not np.isfinite(baseline).all()
                or np.any(baseline < 0)):
            raise ValueError("Nonnegative blocker amplitudes required")
        if len(entries) > self.max_pairs:
            raise OverflowError("Too many joint pairs in one source core")
        if not len(entries):
            return []
        if self.joint_kernel is None:
            code = Path(__file__).with_name("session_joint_match.cu").read_text()
            self.joint_kernel = cp.RawKernel(
                self._kernel_prefix + f"#define SESSION_NT {self.models.nt}\n" + code, "score_joint_template_pairs",
                options=("--std=c++11", "--fmad=false"))
        weights = cp.asarray(1. / noise ** 2, dtype=cp.float32)
        rows = np.asarray(entries, np.int32)
        output = []
        for first in range(0, len(rows), self.pair_batch):
            batch = rows[first:first + self.pair_batch]
            fields = [cp.asarray(batch[:, index].copy()) for index in range(4)]
            blocker_amp = cp.asarray(baseline[first:first + self.pair_batch])
            n = len(batch)
            da, db, dg, di, dc = [cp.empty(n, cp.float32) for _ in range(5)]
            self.joint_kernel((n,), (32,),
                (residual, self.templates, self.channels,
                 *fields, weights, blocker_amp,
                 np.int32(n), np.int32(len(residual)),
                 np.int32(residual.shape[1]),
                 np.int32(self.models.waveforms.shape[2]),
                 np.float32(amplitude_min), np.float32(amplitude_max),
                 np.float32(minimum_incremental_gain),
                 np.float32(minimum_incremental_fraction),
                 np.int32(min_center_separation_samples),
                 np.float32(max_template_correlation), da, db, dg, di, dc,
                 *self._extra_kernel_args()))
            arrays = [cp.asnumpy(values) for values in (da, db, dg, di, dc)]
            output.extend(None if arrays[2][i] <= 0 else JointFit(
                *(float(values[i]) for values in arrays)) for i in range(n))
        return output

    def match(self, voltage_uv, noise_uv, floor_snr, *,
              core_start, core_stop, score_floor=.65, min_margin=.03,
              amplitude_min=.3, amplitude_max=3., max_passes=3,
              max_candidates=500_000, shift_radius=2, refractory_samples=2,
              trace=None, overlap_policy='strict', rescue_floor_snr=None, rescue_passes=1,
              rescue_gain_floor_snr=None, rescue_score_floor=None,
              rescue_last_score_floor=None, rescue_last_amplitude_min=None,
              half_width=8, detector_mode='raw', overlap_window_samples=None):
        """Return all first-pass candidates and fitted spikes in source-core offsets."""
        cp = self.cp
        self.learning_events = self.learning_matches = None
        if overlap_window_samples is None:
            overlap_window_samples = self.models.nt
        recovery = getattr(self, "recovery", None)
        if recovery is not None and (detector_mode != "smooth3"
                or overlap_policy != "strict" or overlap_window_samples != self.models.nt):
            raise ValueError("Spatial recovery requires smooth3 detection and strict full-template overlaps")
        if not isinstance(half_width, int) or not 3 <= half_width <= self.models.nt // 2:
            raise ValueError('Scoring half width must fit the template waveform')
        radius_values = np.asarray(shift_radius)
        valid_radius = (
            np.issubdtype(radius_values.dtype, np.integer) and
            ((radius_values.ndim == 0 and 0 <= int(radius_values) <= 8) or
             (radius_values.shape == (len(self.models.waveforms),) and
              np.all((radius_values >= 0) & (radius_values <= 8))))
        )
        if (not valid_radius or not isinstance(rescue_passes, int)
                or not 1 <= rescue_passes <= 4):
            raise ValueError('Timing radius must be 0–8 and rescue passes 1–4')
        if overlap_policy not in ('strict', 'interference'):
            raise ValueError('Unknown overlap policy')
        if (isinstance(overlap_window_samples, (bool, np.bool_))
                or not isinstance(overlap_window_samples, (int, np.integer))
                or not 0 <= int(overlap_window_samples) <= self.models.nt):
            raise ValueError('Strict overlap window must be an integer from 0 to the template length')
        overlap_window_samples = int(overlap_window_samples)
        if overlap_window_samples != self.models.nt and overlap_policy != 'strict':
            raise ValueError('A shortened overlap window requires strict mode')
        if rescue_floor_snr is not None and (
                not np.isfinite(rescue_floor_snr) or not 0 < rescue_floor_snr <= floor_snr):
            raise ValueError('Rescue threshold must be positive and at most the primary threshold')
        if rescue_gain_floor_snr is not None and (
                rescue_floor_snr is None or not np.isfinite(rescue_gain_floor_snr)
                or not rescue_floor_snr <= rescue_gain_floor_snr <= floor_snr):
            raise ValueError('Rescue gain floor must lie between rescue and primary SNR floors')
        if rescue_score_floor is not None and (
                rescue_floor_snr is None or not np.isfinite(rescue_score_floor)
                or not 0 < rescue_score_floor <= 1):
            raise ValueError('Rescue score floor requires rescue and must lie in (0, 1]')
        if rescue_last_score_floor is not None and (
                rescue_floor_snr is None or (not isinstance(rescue_passes, int) or not 2 <= rescue_passes <= 4)
                or rescue_score_floor is not None
                or not np.isfinite(rescue_last_score_floor)
                or not 0 < rescue_last_score_floor <= 1):
            raise ValueError('Final rescue score floor requires two to four rescue passes and no all-pass override')
        if rescue_last_amplitude_min is not None:
            if isinstance(rescue_last_amplitude_min, (bool, np.bool_)):
                raise ValueError('Final rescue amplitude minimum must be numeric')
            try:
                rescue_last_amplitude_min = float(rescue_last_amplitude_min)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Final rescue amplitude minimum must be numeric') from exc
            if (rescue_floor_snr is None or (not isinstance(rescue_passes, int) or not 2 <= rescue_passes <= 4)
                    or not np.isfinite(rescue_last_amplitude_min)
                    or not 0 < rescue_last_amplitude_min <= amplitude_min):
                raise ValueError('Final rescue amplitude minimum requires two to four rescue passes and must be positive and no higher than the primary minimum')
        scheduler = None
        if overlap_policy == 'interference':
            from .session_interference import InterferenceScheduler
            scheduler = InterferenceScheduler(self.models, noise_uv)
        residual = self._prepare_residual(voltage_uv)
        smooth3_noise = (self._estimate_smooth3_noise(residual, noise_uv)
                         if detector_mode == 'smooth3' else None)
        additional = (recovery.propose(self, residual, noise_uv, floor_snr,
                          smooth3_noise, max_candidates) if recovery is not None else None)
        original = self.detect(
            residual, noise_uv, floor_snr, max_candidates=max_candidates,
            mode=detector_mode, smooth3_noise_uv=smooth3_noise)
        current = original
        all_events = []
        accepted = []
        # Channels remain fixed during this core. Preserve slot order and
        # duplicate contacts while avoiding a mask/allocation per proposal.
        local_contacts = [tuple(int(c) for c in row if c >= 0)
                          for row in self.models.channels]
        identities = (self.template_identity_map if self.template_identity_map is not None
                      else np.arange(len(self.models.waveforms), dtype=np.int64))
        # Linked templates are alternative models of the same neuron. Share
        # temporal occupancy across those models, including between passes.
        occupied_by_identity = {}
        occupied = [occupied_by_identity.setdefault(int(identity), [])
                    for identity in identities]
        total_passes = max_passes + (rescue_passes if rescue_floor_snr is not None else 0)
        strong_done = False
        deferred_events = []
        tiered_rescue = rescue_last_score_floor is not None or rescue_last_amplitude_min is not None
        diagnostics = {
            'candidate_events': 0, 'events_with_template_proposals': 0,
            'event_template_pairs': 0, 'events_with_fit': 0,
            'events_without_eligible_fit': 0, 'gain_rejections': 0,
            'margin_rejections': 0, 'amplitude_rejections': 0,
            'same_unit_refractory_rejections': 0,
            'contact_overlap_rejections': 0,
            'interference_scheduler_rejections': 0,
            'accepted_fits': 0, 'emitted_matches': 0, 'passes': [],
        }
        for pass_index in range(total_passes):
            rescue = pass_index >= max_passes
            if strong_done and not rescue:
                continue
            if tiered_rescue and rescue and pass_index >= max_passes + 2:
                # Extra recovery work is restricted to previously detected,
                # eligible fits blocked by contact overlap. The first two
                # rescue passes and initial candidate pool are unchanged.
                current = deferred_events
            elif pass_index:
                current = self.detect(
                    residual, noise_uv,
                    rescue_floor_snr if rescue else floor_snr,
                    max_candidates=max_candidates, mode=detector_mode,
                    smooth3_noise_uv=smooth3_noise)
            deferred_events = []
            if not current:
                if rescue_floor_snr is not None and not rescue:
                    strong_done = True
                    continue
                break
            all_events.extend(current)
            pass_diagnostics = {
                'pass_index': int(pass_index),
                'rescue': bool(rescue),
                'candidate_events': int(len(current)),
                'events_with_template_proposals': 0, 'event_template_pairs': 0,
                'events_with_fit': 0, 'events_without_eligible_fit': 0,
                'gain_rejections': 0, 'margin_rejections': 0,
                'amplitude_rejections': 0,
                'same_unit_refractory_rejections': 0,
                'contact_overlap_rejections': 0,
                'interference_scheduler_rejections': 0,
                'accepted_fits': 0, 'emitted_matches': 0,
            }
            diagnostics['candidate_events'] += len(current)
            diagnostics['passes'].append(pass_diagnostics)
            pass_score_floor = score_floor
            if pass_index >= 3 or rescue:
                pass_score_floor = max(score_floor, .85)
            if rescue and rescue_score_floor is not None:
                pass_score_floor = max(score_floor, rescue_score_floor)
            # Preserve the original conservative/relaxed pair when adding
            # recovery passes: extra passes continue the relaxed tail instead
            # of inserting conservative fits ahead of the original tail.
            relaxed_rescue = rescue and pass_index >= max_passes + 1
            if relaxed_rescue and rescue_last_score_floor is not None:
                pass_score_floor = max(score_floor, rescue_last_score_floor)
            pass_amplitude_min = (rescue_last_amplitude_min if relaxed_rescue and
                                  rescue_last_amplitude_min is not None else amplitude_min)
            choices = self._score(residual, current, noise_uv=noise_uv,
                                  half_width=half_width,
                                  shift_radius=shift_radius,
                                  score_floor=pass_score_floor,
                                  amplitude_min=pass_amplitude_min, amplitude_max=amplitude_max)
            if trace is not None:
                trace('scores', pass_index=pass_index, residual=residual,
                      events=current, choices=choices)
            proposals = []
            score_diagnostics = self.last_score_diagnostics
            diagnostics['events_with_template_proposals'] += score_diagnostics['events_with_template_proposals']
            diagnostics['event_template_pairs'] += score_diagnostics['event_template_pairs']
            pass_diagnostics.update(score_diagnostics)
            diagnostics['events_with_fit'] += sum(choice is not None for choice in choices)
            diagnostics['events_without_eligible_fit'] += sum(choice is None for choice in choices)
            pass_diagnostics['events_with_fit'] = sum(choice is not None for choice in choices)
            pass_diagnostics['events_without_eligible_fit'] = sum(choice is None for choice in choices)
            required_margin = max(min_margin, .1) if rescue else min_margin
            required_gain_snr = (rescue_gain_floor_snr if rescue and
                                 rescue_gain_floor_snr is not None else floor_snr)
            for i, choice in enumerate(choices):
                if choice is None:
                    continue
                score, gain, unit, amplitude, runner, shift, margin = choice
                if gain < required_gain_snr ** 2:
                    diagnostics['gain_rejections'] += 1
                    pass_diagnostics['gain_rejections'] += 1
                    continue
                if margin < required_margin:
                    diagnostics['margin_rejections'] += 1
                    pass_diagnostics['margin_rejections'] += 1
                    continue
                if not pass_amplitude_min <= amplitude <= amplitude_max:
                    diagnostics['amplitude_rejections'] += 1
                    pass_diagnostics['amplitude_rejections'] += 1
                    continue
                t, channel, _ = current[i]
                proposals.append((gain, int(t) + shift, int(channel), int(unit),
                                  score, runner, amplitude, int(t), margin))
            proposals.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
            collect_deferred = tiered_rescue and rescue and pass_index >= max_passes + 1 and pass_index + 1 < total_passes
            event_lookup = ({(int(row[0]), int(row[1])): row for row in current}
                            if collect_deferred else {})
            accepted_in_pass = []
            subtraction_colors = []
            if scheduler is not None:
                scheduler.reset()
            occupied_contacts = {}
            # In shortened-window mode this stores time/color pairs for
            # deterministic overlap coloring, reusing the conflict index.
            trace_owners = {} if trace is not None else None
            for gain, t, channel, unit, score, runner, amplitude, original_t, margin in proposals:
                prior = occupied[unit]
                pos = bisect_left(prior, t)
                if ((pos and t - prior[pos-1] <= refractory_samples) or
                        (pos < len(prior) and prior[pos] - t <= refractory_samples)):
                    if trace is not None:
                        trace('refractory', pass_index=pass_index, candidate=original_t,
                              channel=channel, unit=unit, fitted_time=t,
                              blocker_time=prior[pos-1] if pos and t-prior[pos-1] <= refractory_samples else prior[pos],
                              blocker_unit=unit)
                    diagnostics['same_unit_refractory_rejections'] += 1
                    pass_diagnostics['same_unit_refractory_rejections'] += 1
                    continue
                contacts = local_contacts[unit]
                conflict = False
                color = 0
                if scheduler is not None:
                    color, blocker = scheduler.admit(t, unit, amplitude, score=score, margin=margin)
                    conflict = blocker is not None
                    if conflict:
                        blocker_time = blocker['blocker_time']
                        blocker_unit = blocker['blocker_unit']
                        blocker_contact = blocker['blocker_contact']
                for contact in (contacts if scheduler is None else ()):
                    contact = int(contact)
                    rows = occupied_contacts.get(contact, ())
                    if overlap_window_samples < self.models.nt:
                        j = bisect_left(rows, (t, -1))
                        previous_time = rows[j-1][0] if j else None
                        next_time = rows[j][0] if j < len(rows) else None
                    else:
                        j = bisect_left(rows, t)
                        previous_time = rows[j-1] if j else None
                        next_time = rows[j] if j < len(rows) else None
                    previous_conflict = (previous_time is not None and
                                         t - previous_time < overlap_window_samples)
                    next_conflict = (next_time is not None and
                                     next_time - t < overlap_window_samples)
                    if previous_conflict or next_conflict:
                        conflict = True
                        if trace is not None:
                            blocker_time = previous_time if previous_conflict else next_time
                            blocker_unit = trace_owners[(contact, blocker_time)]
                            blocker_contact = contact
                        break
                if conflict:
                    if collect_deferred:
                        deferred_events.append(event_lookup[(original_t, channel)])
                    if scheduler is None:
                        diagnostics['contact_overlap_rejections'] += 1
                        pass_diagnostics['contact_overlap_rejections'] += 1
                    else:
                        diagnostics['interference_scheduler_rejections'] += 1
                        pass_diagnostics['interference_scheduler_rejections'] += 1
                    if trace is not None:
                        trace('overlap_deferred', pass_index=pass_index, candidate=original_t,
                              channel=channel, unit=unit, fitted_time=t,
                              blocker_time=blocker_time, blocker_unit=blocker_unit,
                              blocker_contact=blocker_contact)
                    continue
                if scheduler is None and overlap_window_samples < self.models.nt:
                    # Color overlapping template writes without a second
                    # per-contact index of all accepted events.
                    used_colors = set()
                    for contact in contacts:
                        rows = occupied_contacts.get(int(contact), ())
                        lo = bisect_left(rows, (t - self.models.nt + 1, -1))
                        hi = bisect_left(rows, (t + self.models.nt, -1))
                        for position in range(lo, hi):
                            used_colors.add(rows[position][1])
                    for color in range(32):
                        if color not in used_colors:
                            break
                    else:
                        raise OverflowError(
                            'Overlapping subtraction needs more than 32 colors; '
                            'increase the overlap window or shorten the core')
                insort(prior, t)
                for contact in (contacts if scheduler is None else ()):
                    contact = int(contact)
                    rows = occupied_contacts.setdefault(contact, [])
                    if overlap_window_samples < self.models.nt:
                        insort(rows, (t, color))
                    else:
                        insort(rows, t)
                    if trace is not None:
                        trace_owners[(contact, t)] = unit
                accepted_in_pass.append((t, unit, amplitude))
                diagnostics['accepted_fits'] += 1
                pass_diagnostics['accepted_fits'] += 1
                subtraction_colors.append(color)
                if trace is not None:
                    trace('accepted', pass_index=pass_index, candidate=original_t,
                          channel=channel, unit=unit, fitted_time=t)
                if core_start <= t < core_stop:
                    accepted.append(Match(t, channel, unit, score, runner,
                                          amplitude, gain, pass_index, original_t, margin))
                    diagnostics['emitted_matches'] += 1
                    pass_diagnostics['emitted_matches'] += 1
            if not accepted_in_pass:
                if rescue_floor_snr is not None and not rescue:
                    strong_done = True
                    continue
                if (rescue and pass_index == max_passes and
                        (rescue_last_score_floor is not None or
                         rescue_last_amplitude_min is not None)):
                    # A changed final-pass gate can admit candidates even
                    # when the conservative rescue accepted none.
                    continue
                break
            if pass_index + 1 == total_passes:
                break
            # Geometrically disjoint color groups avoid nondeterministic
            # atomic addition order when weakly interacting fits overlap.
            for color in range(max(subtraction_colors)+1):
                batch = [row for row,c in zip(accepted_in_pass, subtraction_colors) if c == color]
                times = cp.asarray([row[0] for row in batch], cp.int32)
                units = cp.asarray([row[1] for row in batch], cp.int32)
                amps = cp.asarray([row[2] for row in batch], cp.float32)
                self.subtract_kernel((len(batch),), (256,),
                    (residual, self.templates, self.channels, times, units, amps,
                     np.int32(len(batch)), np.int32(len(residual)),
                     np.int32(residual.shape[1]),
                     np.int32(self.models.waveforms.shape[2]),
                     *self._extra_kernel_args()))
            self._check_subtraction()
        accepted.sort(key=lambda item: (item.source_sample, item.channel,
                                         item.pass_index))
        if recovery is not None:
            # Learning sees the unchanged baseline; output includes recovery.
            self.learning_events, self.learning_matches = all_events, accepted
            extra, extra_diagnostics = recovery.fit(self, residual, noise_uv,
                additional, accepted, core_start=core_start, core_stop=core_stop,
                last_pass=total_passes-1, score_floor=score_floor,
                amplitude_min=amplitude_min, amplitude_max=amplitude_max,
                min_margin=min_margin, floor_snr=floor_snr, half_width=half_width,
                shift_radius=shift_radius, refractory_samples=refractory_samples)
            for name, value in extra_diagnostics.items():
                diagnostics[name] = diagnostics.get(name, 0) + value
            diagnostics["passes"].append(dict(extra_diagnostics, pass_index=total_passes,
                recovery=True, learning=False))
            all_events = all_events + additional
            accepted = accepted + extra
            accepted.sort(key=lambda item: (item.source_sample, item.channel, item.pass_index))
        self.last_diagnostics = diagnostics
        return all_events, accepted
