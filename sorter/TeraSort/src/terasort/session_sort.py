"""Offline bounded session sorter. Opt-in until ground-truth quality gates pass."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from bisect import bisect_left

import h5py
import numpy as np
import psutil

from .candidates.detectors import numpy_detect
from .candidates.waveforms import geometry_channel_map
from .session_calibration import calibrate_day
from .session_links import build_cross_day_links
from .session_manifest import load_session
from .session_models import (LocalModels, match_residual_cpu,
                             normalize_template_identity_map)
from .session_adaptation import TemplateEvidence
from .scaled_signal import ScaledInt16Signal
from .session_novelty import (NoveltyBank, validate_tentative_retention,
                              validate_candidate_floor, validate_raw_holdout_minimum, peak_alignment_policy)
from .session_signal import (assess_quality, iter_cores, prefetch_cores,
                             preprocess)
from .session_store import RunJournal, ShardWriter
from .session_timing import AdaptiveShiftRadius
from .cuda_environment import configure_cupy_cache
from .session_contact_policy import BAD_CONTACT_POLICY


def _detect_cpu(voltage, noise, floor, *, max_candidates):
    q = np.abs(voltage) / np.asarray(noise, np.float32)[None, :]
    indices = numpy_detect(q, floor=floor)
    if len(indices) > max_candidates:
        raise OverflowError("Artifact burst exceeds per-core candidate budget")
    t, c = np.divmod(indices, voltage.shape[1])
    return list(zip(t.astype(np.int32), c.astype(np.int32),
                    q[t, c].astype(np.float32)))


def _day_models(probe, day_id, root, *, resume, preprocessor=None,
                calibration_estimator="mean", exclude_bad_contacts=False, nt=61):
    calibration = root / probe.probe_id / "calibration"
    path = calibration / f"{day_id}.npz"
    metadata = calibration / f"{day_id}.json"
    partial = path.with_suffix(".npz.partial")
    if path.is_file():
        models = LocalModels.load(path, probe)
        if models.nt != nt:
            raise ValueError("Cached model waveform length differs from nt")
        return models
    if partial.exists():
        if not resume:
            raise FileExistsError(partial)
        partial.unlink()
    seed = probe.seed_for_day(day_id)
    if seed is not None:
        models = LocalModels.load(seed, probe)
        if models.nt != nt:
            raise ValueError("Seed waveform length differs from nt")
        excluded = models.exclude_contacts(probe.bad_channels) if exclude_bad_contacts else 0
        aligned = models.canonicalize()
        description = {"method": "aligned_external_seed",
                       "seed_path": str(seed), "seed_units": len(models.waveforms),
                       "canonicalized_units": aligned}
        if exclude_bad_contacts:
            description.update(bad_contact_neighborhoods=BAD_CONTACT_POLICY, masked_seed_units=excluded)
    else:
        models, description = calibrate_day(
            probe, day_id, preprocessor=preprocessor,
            template_estimator=calibration_estimator, exclude_bad_contacts=exclude_bad_contacts, nt=nt)
    calibration.mkdir(parents=True, exist_ok=True)
    if path.exists() or partial.exists():
        raise FileExistsError(path)
    with partial.open("xb") as handle:
        np.savez_compressed(handle, waveforms=models.waveforms,
                            channels=models.channels, anchors=models.anchors,
                            assigned=models.assigned, version=models.version)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    metadata.write_text(json.dumps(description, indent=2) + "\n",
                        encoding="utf-8")
    return models


def _model_from_completed(path):
    with h5py.File(path, "r") as handle:
        if not handle.attrs.get("complete", False):
            raise ValueError(f"Incomplete published shard: {path}")
        group = handle["model_after"]
        return LocalModels(group["waveforms"][:], group["channels"][:],
                           group["anchors"][:], group["assigned"][:],
                           int(group.attrs["version"]))


def _configure_template_proposals(models, probe, radius_um):
    """Index models by nearby anchor contacts, bounded by probe geometry.

    A detection on a contact only competes against templates anchored on the
    same shank and within ``radius_um``. This keeps event/template pair counts
    proportional to local density instead of total session template count.
    """
    try:
        radius_um = float(radius_um)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Template proposal radius must be finite and positive") from exc
    geometry = np.asarray(probe.geometry, dtype=np.float64)
    shank = np.asarray(probe.shank)
    anchors = np.asarray(models.anchors, dtype=np.int64)
    if (not np.isfinite(radius_um) or radius_um <= 0
            or geometry.shape != (probe.n_channels, 2)
            or shank.shape != (probe.n_channels,)
            or np.any(anchors < 0) or np.any(anchors >= probe.n_channels)):
        raise ValueError("Invalid template proposal radius, probe geometry, or model anchors")
    anchor_geometry = geometry[anchors]
    anchor_shank = shank[anchors]
    by_channel = {}
    for channel in range(probe.n_channels):
        delta = anchor_geometry - geometry[channel]
        distance2 = np.einsum("md,md->m", delta, delta)
        units = np.flatnonzero(
            (anchor_shank == shank[channel])
            & (distance2 <= radius_um * radius_um))
        by_channel[channel] = units.astype(np.int64).tolist()
    models.by_channel = by_channel
    return max((len(units) for units in by_channel.values()), default=0)


def _shard_intervals(probe, *, shard_seconds, start_sample, stop_sample):
    step = max(1, round(probe.sample_rate_hz * shard_seconds))
    for segment in probe.segments:
        first = max(segment.start_sample, start_sample)
        last = min(segment.stop_sample, stop_sample)
        for start in range(first, last, step):
            yield segment, start, min(start + step, last)


def _collect_updates(models, matches, voltage, quality, evidence,
                     *, data_start, core_start):
    if quality.interval_bad:
        return
    usable = quality.usable_channels
    event_times = [match.source_sample for match in matches]
    contact_sets = [set(map(int, contacts[contacts >= 0]))
                    for contacts in models.channels]
    for i, match in enumerate(matches):
        if (match.pass_index != 0 or match.score < .9 or
                match.relative_margin < .1 or
                not .5 <= match.amplitude <= 2.):
            continue
        t = match.source_sample
        if t < models.nt//2 or t + models.nt//2+1 > len(voltage):
            continue
        left = bisect_left(event_times, t - models.nt + 1)
        right = bisect_left(event_times, t + models.nt)
        if any(contact_sets[match.unit] & contact_sets[matches[j].unit]
               for j in range(left, right) if j != i):
            continue
        channels = models.channels[match.unit]
        valid = channels >= 0
        if not np.all(usable[channels[valid]]):
            continue
        if evidence.quantized:
            if not isinstance(voltage, ScaledInt16Signal):
                raise ValueError('Quantized evidence requires an INT16 host signal')
            codes = np.zeros_like(models.waveforms[match.unit], dtype=np.int16)
            codes[:, valid] = voltage.data[t-models.nt//2:t+models.nt//2+1, channels[valid]]
            scales = np.zeros(len(channels), np.float32)
            scales[valid] = (voltage.scales[channels[valid]] /
                             np.float32(match.amplitude))
            evidence.observe(match.unit, data_start + t, core_start, codes,
                             scales=scales)
        else:
            snippet = np.zeros_like(models.waveforms[match.unit])
            snippet[:, valid] = (voltage[t-models.nt//2:t+models.nt//2+1, channels[valid]] /
                                 np.float32(match.amplitude))
            if np.isfinite(snippet).all():
                evidence.observe(match.unit, data_start + t, core_start, snippet)


def run_session(manifest, output_root, *, backend="cuda", resume=False, nt=61,
                core_seconds=2., shard_seconds=300., halo_ms=100.,
                floor_snr=4.5, score_floor=.65, min_margin=.03,
                cache_fraction=.05, max_candidates=500_000,
                waveform_cache_mode="legacy",
                start_sample=0, stop_sample=None, vram_limit_gb=12.,
                template_merge_cosine=None, template_merge_radius_um=32.,
                template_merge_mode="components",
                template_proposal_radius_um=48., export_amplitude_min=None,
                fit_amplitude_min=None,
                progress=None, freeze_templates=False, novelty="off", novelty_denoising=False,
                novelty_tentative_seconds=0., novelty_candidate_floor_snr=6.,
                novelty_stratified_admission=False, novelty_priority_admission=False,
                novelty_preserve_strong_tentative=False, novelty_raw_anchor_holdout=False,
                novelty_raw_minimum_cosine=.85, novelty_backfill=False, novelty_peak_alignment=False,
                novelty_residual_duplicates=False,
                residual_passes=3,
                overlap_policy='strict', overlap_window_samples=None,
                rescue_floor_snr=None, rescue_passes=1,
                rescue_gain_floor_snr=None, rescue_score_floor=None,
                rescue_last_score_floor=None, rescue_last_amplitude_min=None,
                shift_radius=2, half_width=8, refit_rounds=0, detector_mode='raw',
                read_buffer_mb=0, prefetch_depth=2,
                preprocessing_backend="cpu", matching_storage="float32",
                direct_int16_preprocessing=False,
                integer_evidence=False,
                calibration_estimator="mean", template_contact_scoring=False, spatial_recovery=False,
                adaptive_shift_radius=False,
                adaptive_shift_boundary_fraction=AdaptiveShiftRadius.BOUNDARY_FRACTION,
                exclude_bad_contacts=False):
    """Process one session into immutable, per-probe time shards.

    The existing Kilosort-compatible sorter remains the quality baseline.
    This route is opt-in and deliberately records every unknown detection.
    """
    from .session_models import validate_nt
    nt = validate_nt(nt)
    session = load_session(manifest)
    if direct_int16_preprocessing and any(p.input_is_preprocessed for p in session.probes):
        raise ValueError("Direct filter-to-INT16 is a raw-input option; use cuda-int16 for preprocessed input")
    if not isinstance(novelty_residual_duplicates,bool) or (novelty_residual_duplicates and
        (novelty == 'off' or not novelty_denoising or not novelty_raw_anchor_holdout or not novelty_backfill)):
        raise ValueError('Novelty residual duplicate validation requires compact raw holdout')
    if not isinstance(exclude_bad_contacts, bool):
        raise ValueError('Bad-contact exclusion must be boolean')
    novelty_tentative_seconds = validate_tentative_retention(novelty_tentative_seconds)
    novelty_candidate_floor_snr = validate_candidate_floor(novelty_candidate_floor_snr)
    novelty_raw_minimum_cosine = validate_raw_holdout_minimum(novelty_raw_minimum_cosine)
    if not isinstance(novelty_backfill, bool) or (novelty_backfill and not novelty_raw_anchor_holdout):
        raise ValueError('Novelty backfill requires raw anchor validation')
    if not isinstance(novelty_raw_anchor_holdout, bool):
        raise ValueError('Raw anchor holdout must be boolean')
    if not isinstance(novelty_preserve_strong_tentative, bool):
        raise ValueError('Strong tentative retention must be boolean')
    if not isinstance(novelty_priority_admission, bool) or (novelty_priority_admission and novelty_stratified_admission):
        raise ValueError('Priority admission must be boolean and cannot combine with quotas')
    if not isinstance(novelty_stratified_admission, bool):
        raise ValueError('Stratified admission must be boolean')
    if not isinstance(novelty_peak_alignment, bool):
        raise ValueError('Novelty peak alignment must be boolean')
    if not isinstance(novelty_denoising, bool):
        raise ValueError('Novelty denoising must be boolean')
    if (not isinstance(template_contact_scoring, bool) or
            (template_contact_scoring and (backend != 'cuda' or refit_rounds))):
        raise ValueError('Template contact scoring requires CUDA and no CPU refit')
    if (not isinstance(spatial_recovery, bool) or (spatial_recovery and
            (backend != "cuda" or matching_storage != "int16"
             or detector_mode != "smooth3" or refit_rounds
             or overlap_policy != "strict"
             or overlap_window_samples not in (None, nt)))):
        raise ValueError("Spatial recovery requires CUDA INT16, smooth3, strict full-template overlaps and no refit")
    if calibration_estimator not in ("mean", "median_rank3"):
        raise ValueError("Calibration estimator must be mean or median_rank3")
    if (matching_storage not in ("float32", "int16")
            or (matching_storage == "int16" and backend != "cuda")):
        raise ValueError("INT16 matching storage requires the CUDA sorting backend")
    if (preprocessing_backend not in ("cpu", "cuda", "cuda-int16")
            or (preprocessing_backend != "cpu" and backend != "cuda")):
        raise ValueError("CUDA preprocessing requires the CUDA sorting backend")
    integer_filter = preprocessing_backend == "cuda-int16"
    if not isinstance(integer_evidence, bool) or (integer_evidence and not integer_filter):
        raise ValueError("INT16 template evidence requires cuda-int16 preprocessing")
    if integer_filter and (matching_storage != "int16" or refit_rounds):
        raise ValueError("Integer filtering requires INT16 storage and no CPU refit")
    if direct_int16_preprocessing and (preprocessing_backend != "cuda" or matching_storage != "int16"):
        raise ValueError("Direct INT16 preprocessing requires CUDA preprocessing and INT16 matching storage")
    if (isinstance(read_buffer_mb, bool) or not isinstance(read_buffer_mb, int)
            or not 0 <= read_buffer_mb <= 1024 or isinstance(prefetch_depth, bool)
            or not isinstance(prefetch_depth, int) or not 1 <= prefetch_depth <= 64):
        raise ValueError('Read buffer must be 0â€“1024 MiB; prefetch depth must be 1â€“64 cores')
    if waveform_cache_mode not in ("legacy", "balanced", "recall"):
        raise ValueError("Waveform cache mode must be legacy, balanced, or recall")
    if detector_mode not in ('raw','smooth3') or (detector_mode!='raw' and backend!='cuda'):
        raise ValueError('smooth3 detection currently requires CUDA')
    if not isinstance(refit_rounds, int) or not 0 <= refit_rounds <= 2:
        raise ValueError('Refit rounds must be 0â€“2')
    if not isinstance(half_width, int) or not 3 <= half_width <= nt//2:
        raise ValueError('Scoring half width must fit nt')
    def emit(stage, **details):
        if progress is not None:
            progress({"stage": stage, **details})
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be cpu or cuda")
    if (not isinstance(shift_radius, int) or not 0 <= shift_radius <= 8
            or not isinstance(rescue_passes, int) or not 1 <= rescue_passes <= 4):
        raise ValueError('Timing radius must be 0â€“8 and rescue passes 1â€“4')
    if (not isinstance(adaptive_shift_radius, bool)
            or (adaptive_shift_radius and
                (backend != 'cuda' or shift_radius >= 8 or refit_rounds))):
        raise ValueError('Adaptive timing radius requires CUDA, radius below 8, and no local refit')
    if adaptive_shift_radius:
        try:
            adaptive_shift_boundary_fraction = float(adaptive_shift_boundary_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Adaptive timing boundary fraction must be in (0, 1]') from exc
        if (not np.isfinite(adaptive_shift_boundary_fraction)
                or not 0 < adaptive_shift_boundary_fraction <= 1):
            raise ValueError('Adaptive timing boundary fraction must be in (0, 1]')
    if rescue_floor_snr is not None and (backend != 'cuda' or
            not np.isfinite(rescue_floor_snr) or not 0 < rescue_floor_snr <= floor_snr):
        raise ValueError('CUDA rescue threshold must be positive and at most primary floor_snr')
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
    if overlap_policy not in ('strict', 'interference') or (backend == 'cpu' and overlap_policy != 'strict'):
        raise ValueError('Interference scheduling requires CUDA; CPU already fits sequentially')
    if overlap_window_samples is None:
        strict_overlap_window_samples = nt
    else:
        if (isinstance(overlap_window_samples, (bool, np.bool_))
                or not isinstance(overlap_window_samples, (int, np.integer))
                or not 0 <= int(overlap_window_samples) <= nt):
            raise ValueError('Overlap window must be an integer from 0 to nt samples')
        strict_overlap_window_samples = int(overlap_window_samples)
        if strict_overlap_window_samples != nt and (backend != 'cuda' or overlap_policy != 'strict'):
            raise ValueError('A shortened overlap window requires CUDA strict mode')
    if not isinstance(residual_passes, int) or not 1 <= residual_passes <= 12:
        raise ValueError('Residual passes must be an integer from 1 to 12')
    if novelty not in ('off', 'shadow', 'enroll') or (novelty == 'enroll' and freeze_templates):
        raise ValueError('Invalid novelty mode or enrollment with frozen templates')
    if fit_amplitude_min is not None:
        if isinstance(fit_amplitude_min, (bool, np.bool_)):
            raise ValueError("Fit amplitude minimum must be finite and in [0, 3]")
        try:
            fit_amplitude_min = float(fit_amplitude_min)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Fit amplitude minimum must be finite and in [0, 3]") from exc
        if not np.isfinite(fit_amplitude_min) or not 0 <= fit_amplitude_min <= 3:
            raise ValueError("Fit amplitude minimum must be finite and in [0, 3]")
    if rescue_last_amplitude_min is not None:
        if (backend != 'cuda' or rescue_floor_snr is None or (not isinstance(rescue_passes, int) or not 2 <= rescue_passes <= 4)
                or isinstance(rescue_last_amplitude_min, (bool, np.bool_))):
            raise ValueError('Final rescue amplitude minimum requires CUDA and two to four rescue passes')
        try:
            rescue_last_amplitude_min = float(rescue_last_amplitude_min)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Final rescue amplitude minimum must be finite and positive') from exc
        primary_amplitude_min = fit_amplitude_min if fit_amplitude_min is not None else .3
        if (not np.isfinite(rescue_last_amplitude_min) or
                not 0 < rescue_last_amplitude_min <= primary_amplitude_min):
            raise ValueError('Final rescue amplitude minimum must be positive and no higher than the primary minimum')
    if export_amplitude_min is not None:
        if isinstance(export_amplitude_min, (bool, np.bool_)):
            raise ValueError("Export amplitude minimum must be finite and nonnegative")
        try:
            export_amplitude_min = float(export_amplitude_min)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Export amplitude minimum must be finite and nonnegative") from exc
        if not np.isfinite(export_amplitude_min) or export_amplitude_min < 0:
            raise ValueError("Export amplitude minimum must be finite and nonnegative")
    if (not 0 < core_seconds <= shard_seconds or halo_ms < 0 or
            floor_snr <= 0 or not 0 < score_floor <= 1 or
            not 0 <= min_margin <= 1 or not 0 <= cache_fraction <= 1 or
            max_candidates < 1 or start_sample < 0 or
            (stop_sample is not None and stop_sample <= start_sample)):
        raise ValueError("Invalid bounded session configuration")
    if template_merge_cosine is not None:
        template_merge_cosine = float(template_merge_cosine)
        template_merge_radius_um = float(template_merge_radius_um)
        if (not np.isfinite(template_merge_cosine)
                or not 0 < template_merge_cosine <= 1
                or not np.isfinite(template_merge_radius_um)
                or template_merge_radius_um <= 0):
            raise ValueError("Template merge cosine must be in (0, 1] and radius positive")
    if template_merge_mode not in ("components", "reciprocal"):
        raise ValueError("Template merge mode must be components or reciprocal")
    try:
        template_proposal_radius_um = float(template_proposal_radius_um)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Template proposal radius must be finite and positive") from exc
    if not np.isfinite(template_proposal_radius_um) or template_proposal_radius_um <= 0:
        raise ValueError("Template proposal radius must be finite and positive")
    config = {
        "backend": backend, "core_seconds": core_seconds,
        "shard_seconds": shard_seconds, "halo_ms": halo_ms,
        "floor_snr": floor_snr, "score_floor": score_floor,
        "min_margin": min_margin, "cache_fraction": cache_fraction,
        "max_candidates": max_candidates, "start_sample": start_sample,
        "stop_sample": stop_sample, "vram_limit_gb": vram_limit_gb,
        "preprocessing": (
            "upstream preprocessed input; no TeraSort filter/reference"
            if all(p.input_is_preprocessed for p in session.probes) else
            "mixed raw/preprocessed input; see session manifest"
            if any(p.input_is_preprocessed for p in session.probes) else
            "per-shank median; zero-phase 300-6000 Hz Butterworth v1"
        ),
        "matcher": f"center{2*half_width+1} proposal; noise-weighted full{nt} fitting v2; experimental",
        "template_proposal_radius_um": template_proposal_radius_um,
        "adaptation": "bounded_recent_evidence_temporal_holdout_v1",
        "freeze_templates": freeze_templates,
        "residual_passes": residual_passes,
        "overlap_policy": overlap_policy,
        "rescue_floor_snr": rescue_floor_snr,
        "rescue_passes": rescue_passes,
        "shift_radius": shift_radius,
        "half_width": half_width,
        "detector_mode": detector_mode,
        "candidate_snr_frame": 'raw_noise' if detector_mode=='raw' else 'smooth3_core_MAD_v2',
        "refit_rounds": refit_rounds,
        "refit_algorithm": "bounded_neighbor_subtraction_64_v1",
        "rescue_algorithm": "bounded_final_passes_score085_margin010_primary_gain_v2",
        "overlap_scheduler": ("candidate_confidence_bidirectional_1pct_colors32_v1"
                              if overlap_policy == "interference" else "strict_v1"),
        "late_residual_score_floor": .85,
        "novelty": novelty,
        "novelty_algorithm": "bounded_temporal_proposals_v1",
    }
    if nt != 61:
        config['nt'] = nt
    if exclude_bad_contacts:
        config['bad_contact_neighborhoods'] = BAD_CONTACT_POLICY
    if novelty != 'off' and novelty_denoising:
        config['novelty_algorithm'] = 'bounded_temporal_proposals_pca3_v2'
        config['novelty_features'] = 'seed_temporal_pca3_v1'
    if novelty != 'off' and (novelty_tentative_seconds or novelty_preserve_strong_tentative):
        config['novelty_retention'] = {
            'algorithm': ('weak_below3_source_idle_strong30min_snr6_v2'
                          if novelty_preserve_strong_tentative else 'support_below3_source_idle_v1'),
            'seconds': novelty_tentative_seconds,
        }
        if novelty_preserve_strong_tentative:
            config['novelty_retention'].update(strong_minimum_snr=6., strong_horizon_seconds=1800.)
    if novelty != 'off' and novelty_candidate_floor_snr != 6.:
        config['novelty_candidate_floor'] = {
            'algorithm': 'detector_snr_gate_v1',
            'minimum': novelty_candidate_floor_snr,
        }
    if novelty != 'off' and novelty_peak_alignment:
        config['novelty_peak_alignment'] = peak_alignment_policy(nt=nt)
    if novelty != 'off' and novelty_backfill:
        config['novelty_backfill'] = 'compact_int16_frozen_fit_support64_v1'
    if novelty_residual_duplicates:
        from .session_novelty_residual import policy
        config['novelty_duplicate_validation'] = policy()
    if novelty != 'off' and novelty_raw_anchor_holdout:
        config['novelty_raw_validation'] = dict(algorithm='raw_anchor_center17_holdout_cosine_v1',
            minimum_cosine=novelty_raw_minimum_cosine, snr_ratio_min=.5, snr_ratio_max=2.)
    if novelty != 'off' and novelty_stratified_admission:
        config['novelty_admission'] = 'strong_reserve_weak_quarter_cap_v1'
    if novelty != 'off' and novelty_priority_admission:
        config['novelty_admission'] = 'unsupported_priority_replace_snr6_ratio125_v1'
    if template_contact_scoring:
        config['template_contact_scoring'] = {
            'algorithm': 'center_contact_energy_mask_v1',
            'minimum_relative_energy': .05,
            'amplitude_fit': f'all_contacts_full{nt}',
        }
    if waveform_cache_mode == "balanced":
        config["waveform_cache_mode"] = "uncertain40_stratified_assigned40_random20_v1"
    elif waveform_cache_mode == "recall":
        config["waveform_cache_mode"] = "weak_unknown30_high_unknown10_assigned40_random20_v1"
    if preprocessing_backend == "cuda":
        config["preprocessing_execution"] = (
            "cuda_slab_direct_int16_v1" if direct_int16_preprocessing else
            "cuda_median_parallel_sos_fp64_v1")
    elif integer_filter:
        from .session_fixedpoint import FRAME
        config["preprocessing_execution"] = FRAME
    if matching_storage == "int16":
        config["matching_storage"] = (
            "int16_filter_adc_power2_residual_measured_headroom4_templates_uv_power2_v2"
            if integer_filter else "int16_power2_per_contact_headroom4_min_uv_1over256_v1")
    if integer_evidence:
        config["adaptation_storage"] = "int16_source_snippet_channel_scale_v1"
    if calibration_estimator != "mean":
        config["calibration_estimator"] = "median_rank3_local_kmeans_v1"
    if rescue_gain_floor_snr is not None:
        config["rescue_gain_floor_snr"] = rescue_gain_floor_snr
        config["rescue_algorithm"] = "bounded_final_passes_score085_margin010_selective_gain_v1"
    if rescue_score_floor is not None:
        config["rescue_score_floor"] = rescue_score_floor
        config["rescue_algorithm"] = "bounded_final_passes_selective_score_margin010_gain_v1"
    if rescue_last_score_floor is not None:
        config["rescue_last_score_floor"] = rescue_last_score_floor
        config["rescue_algorithm"] = "bounded_conservative_then_relaxed_rescue_v1"
    if rescue_last_amplitude_min is not None:
        config["rescue_last_amplitude_min"] = rescue_last_amplitude_min
        config["rescue_amplitude_algorithm"] = "bounded_final_pass_amplitude_floor_v1"
    if rescue_passes > 2 and (rescue_last_score_floor is not None or rescue_last_amplitude_min is not None):
        config["rescue_algorithm"] = "bounded_relaxed_deferred_collision_tail_v2"
    if adaptive_shift_radius:
        config["adaptive_shift_radius"] = {
            "algorithm": "pilot_fit_boundary_fraction_v1",
            "pilot_seconds": AdaptiveShiftRadius.PILOT_SECONDS,
            "minimum_observations": AdaptiveShiftRadius.MIN_OBSERVATIONS,
            "boundary_fraction": float(adaptive_shift_boundary_fraction),
            "expanded_radius": shift_radius + 1,
        }
    # Preserve existing default-run checkpoint digests. Nondefault IO settings
    # are immutable for a run, just like the scientific configuration.
    if read_buffer_mb or prefetch_depth != 2:
        config.update(read_buffer_mb=read_buffer_mb, prefetch_depth=prefetch_depth)
    if overlap_window_samples is not None and strict_overlap_window_samples != nt:
        config['strict_contact_overlap_samples'] = strict_overlap_window_samples
        config['subtraction_coloring'] = 'contact_time_disjoint_32_v1'
    if export_amplitude_min is not None:
        config["spike_export_filter"] = {
            "algorithm": "matched_amplitude_scale_min_v1",
            "minimum": export_amplitude_min,
            "scope": "output_view_only",
        }
    if fit_amplitude_min is not None:
        config["pre_subtraction_amplitude_floor"] = {
            "algorithm": "matched_amplitude_scale_min_v1",
            "minimum": fit_amplitude_min,
            "scope": "matcher_acceptance_and_residual_subtraction",
        }
    if template_merge_cosine is not None:
        config["template_linking"] = {
            "algorithm": ("local_signed_cosine_components_v1" if
                          template_merge_mode == "components" else
                          "local_signed_cosine_reciprocal_v1"),
            "cosine_threshold": template_merge_cosine,
            "anchor_radius_um": template_merge_radius_um,
            "assignment_margin": "best_fit_per_linked_identity_v1",
        }
    recovery_core_samples = {}
    if spatial_recovery:
        from .session_spatial_recovery import (FrozenSpatialRecovery,
            bounded_core_samples, policy as recovery_policy)
        for probe in session.probes:
            halo = max(nt//2+1, int(round(probe.sample_rate_hz * halo_ms / 1000.)))
            recovery_core_samples[probe.probe_id] = bounded_core_samples(
                probe, core_seconds, halo)
        config["spatial_recovery"] = dict(recovery_policy(nt=nt),
            core_samples_by_probe=recovery_core_samples)
    # Additive output policy: legacy resumes retain their original digest and
    # feature coverage. New runs capture every accepted spike without a raw reread.
    capture_phy_features = True
    if resume:
        prior_config = json.loads((Path(output_root)/"run.json").read_text())["config"]
        capture_phy_features = "phy_features" in prior_config
    if capture_phy_features:
        config["phy_features"] = "resident_preprocessed_temporal_pca3_v1"
    source_record = session.as_source_record()
    journal = RunJournal(output_root, source_record, config, resume=resume)
    if backend == "cuda":
        configure_cupy_cache()
        import cupy as cp
        if vram_limit_gb <= 0:
            raise ValueError("Positive GPU pool limit required")
        cp.get_default_memory_pool().set_limit(
            size=int(vram_limit_gb * 1024 ** 3))
        from .session_gpu import CudaResidualMatcher
        matcher_type = CudaResidualMatcher
        if matching_storage == "int16":
            from .session_int16 import CudaInt16ResidualMatcher
            matcher_type = CudaInt16ResidualMatcher
    started = time.perf_counter()
    process = psutil.Process()
    totals = {"shards": 0, "candidate_rows": 0, "spikes": 0,
              "template_promotions": 0, "peak_evidence_rows": 0,
              "enrolled_units": 0, "recovered_novelty_spikes": 0,
              "source_bytes_read": 0, "cached_waveform_bytes": 0,
              "raw_bytes_covered": 0,
              "derived_bytes": 0, "peak_process_rss_bytes": 0,
              "peak_gpu_pool_bytes": 0,
              "peak_total_vram_used_bytes": 0,
              "matcher_diagnostics": {}}
    for probe in session.probes:
        gpu_preprocessor = None
        if preprocessing_backend == "cuda":
            from .session_preprocess_gpu import CudaPreprocessor
            gpu_preprocessor = CudaPreprocessor(probe)
        elif integer_filter:
            from .session_fixedpoint import FixedInt16Preprocessor
            gpu_preprocessor = FixedInt16Preprocessor(probe)
        day_id = None
        models = None
        template_link_map = None
        template_link_metadata = None
        matcher = None
        adaptive_shift = None
        neighbor_map = geometry_channel_map(
            probe.geometry, probe.shank, 75., 16,
            excluded_channels=probe.bad_channels if exclude_bad_contacts else ())
        halo = max(nt//2+1, int(round(probe.sample_rate_hz * halo_ms / 1000.)))
        probe_core_seconds = (recovery_core_samples[probe.probe_id] / probe.sample_rate_hz
                              if spatial_recovery else core_seconds)
        if probe.channel_map is not None and read_buffer_mb:
            read_samples = read_buffer_mb * 1024**2 // (2 * probe.source_n_channels) - 2 * halo
            if read_samples < 64:
                raise ValueError("Source read buffer cannot fit a selected-channel core and its halos")
            probe_core_seconds = min(probe_core_seconds, read_samples / probe.sample_rate_hz)
        final = probe.stop_sample if stop_sample is None else stop_sample
        for segment, start, stop in _shard_intervals(
                probe, shard_seconds=shard_seconds,
                start_sample=start_sample, stop_sample=final):
            if segment.day_id != day_id:
                day_id = segment.day_id
                emit("calibration_start", probe_id=probe.probe_id, day_id=day_id)
                models = _day_models(probe, day_id, journal.root, resume=resume,
                    preprocessor=gpu_preprocessor.host if gpu_preprocessor is not None else None,
                    calibration_estimator=calibration_estimator, exclude_bad_contacts=exclude_bad_contacts, nt=nt)
                max_local_models = _configure_template_proposals(
                    models, probe, template_proposal_radius_um)
                if backend == "cuda" and max_local_models > 512:
                    raise ValueError(
                        f"Template proposal radius {template_proposal_radius_um:g} um "
                        f"puts {max_local_models} models on one channel; reduce the radius")
                template_link_map = None
                template_link_metadata = None
                if template_merge_cosine is not None:
                    from .session_template_merge import build_template_map
                    template_link_map, template_link_metadata = build_template_map(
                        models.waveforms, models.channels, models.anchors,
                        probe.geometry, probe.shank,
                        cosine_threshold=template_merge_cosine,
                        radius_um=template_merge_radius_um,
                        mode=template_merge_mode,
                    )
                    emit("template_links_complete", probe_id=probe.probe_id,
                         day_id=day_id, **template_link_metadata)
                from .session_phy_capture import curation_basis
                phy_basis = curation_basis(models) if capture_phy_features else None
                evidence = TemplateEvidence(models, probe.sample_rate_hz,
                                            quantized=integer_evidence)
                adaptive_shift = (AdaptiveShiftRadius(
                    len(models.waveforms), probe.sample_rate_hz, shift_radius,
                    boundary_fraction=adaptive_shift_boundary_fraction)
                    if adaptive_shift_radius else None)
                novel = (NoveltyBank(probe.sample_rate_hz, nt=nt,
                                    models=models if novelty_denoising else None,
                                    tentative_seconds=novelty_tentative_seconds,
                                    candidate_floor_snr=novelty_candidate_floor_snr,
                                    stratified_admission=novelty_stratified_admission,
                                    priority_admission=novelty_priority_admission,
                                    preserve_strong_tentative=novelty_preserve_strong_tentative,
                                    raw_anchor_holdout=novelty_raw_anchor_holdout,
                                    raw_holdout_minimum_cosine=novelty_raw_minimum_cosine,
                                    backfill=novelty_backfill,
                                    peak_alignment=novelty_peak_alignment,
                                    residual_duplicates=novelty_residual_duplicates)
                         if novelty != 'off' else None)
                if novel is not None:
                    novel.curation_basis = phy_basis
                if novel is not None and models.waveforms.shape[2] != neighbor_map.shape[1]:
                    raise ValueError('Novelty requires seeds with the session local-channel width')
                emit("calibration_complete", probe_id=probe.probe_id,
                     day_id=day_id, templates=len(models.waveforms))
                recovery = (FrozenSpatialRecovery.for_epoch(probe, day_id,
                    journal.root / probe.probe_id / "calibration", models, resume=resume)
                    if spatial_recovery else None)
                matcher = (matcher_type(
                    models, template_identity_map=template_link_map,
                    template_contact_scoring=template_contact_scoring, recovery=recovery)
                    if backend == "cuda" else None)
            output = journal.shard_path(probe.probe_id, start, stop)
            if journal.completed(output):
                if not resume:
                    raise FileExistsError(output)
                models = _model_from_completed(output)
                _configure_template_proposals(
                    models, probe, template_proposal_radius_um)
                evidence = TemplateEvidence(models, probe.sample_rate_hz,
                                            quantized=integer_evidence)
                with h5py.File(output, 'r') as handle:
                    evidence.restore(handle)
                    if capture_phy_features:
                        if "phy_feature_basis" not in handle:
                            raise ValueError("Completed shard is missing measured Phy feature state")
                        phy_basis = handle["phy_feature_basis"][:]
                    if novel is not None:
                        novel.curation_basis = phy_basis
                        novel.restore(handle)
                    if adaptive_shift is not None:
                        if 'adaptive_shift' not in handle:
                            raise ValueError('Completed shard is missing adaptive timing state')
                        adaptive_shift.restore(handle['adaptive_shift'],
                                               len(models.waveforms))
                if matcher is not None:
                    matcher.models = models
                continue
            template_link_map = normalize_template_identity_map(
                template_link_map, len(models.waveforms))
            if matcher is not None:
                matcher.template_identity_map = template_link_map
            partial = output.with_suffix(".h5.partial")
            if partial.exists():
                if not resume:
                    raise FileExistsError(partial)
                partial.unlink()
            writer = ShardWriter(
                output, probe=probe, day_id=segment.day_id, nt=nt,
                start_sample=start, stop_sample=stop,
                config_sha256=journal.record["config_sha256"],
                template_link_map=template_link_map,
                template_link_metadata=template_link_metadata,
                export_amplitude_min=export_amplitude_min,
                phy_feature_basis=phy_basis,
                phy_feature_width=min(16,models.channels.shape[1]) if capture_phy_features else None)
            shard_started = time.perf_counter()
            shard_bytes = 0
            shard_raw_bytes = 0
            shard_peak_rss = 0
            shard_peak_gpu_pool = 0
            shard_peak_total_vram = 0
            shard_match_diagnostics = {}
            try:
                cores = iter_cores(
                    probe, core_seconds=probe_core_seconds, halo_samples=halo,
                    start_sample=start, stop_sample=stop, read_buffer_mb=read_buffer_mb)
                for core in prefetch_cores(cores, depth=prefetch_depth):
                    device_signal = None
                    if gpu_preprocessor is None:
                        voltage = preprocess(core, probe)
                    elif integer_filter:
                        voltage, device_signal, quality = gpu_preprocessor.process(core)
                    elif direct_int16_preprocessing:
                        voltage, device_signal = gpu_preprocessor.process_quantized(core, matcher)
                    else:
                        voltage, device_signal = gpu_preprocessor.process(core)
                    if not integer_filter:
                        quality = assess_quality(core, voltage, probe)
                    # Preprocessing owns this host core. QC has finished;
                    # masking it in place avoids another whole-core copy.
                    signal = voltage
                    if integer_filter:
                        signal.mask_channels(quality.usable_channels)
                    else:
                        signal[:, ~quality.usable_channels] = 0
                    if device_signal is not None:
                        device_data = (device_signal.data if direct_int16_preprocessing or integer_filter
                                       else device_signal)
                        device_data[:, cp.asarray(~quality.usable_channels)] = 0
                        del device_data
                    noise = np.where(quality.usable_channels,
                                     quality.noise_uv, np.inf).astype(np.float32)
                    local_start = core.core_start - core.data_start
                    local_stop = core.core_stop - core.data_start
                    core_shift_radius = (
                        adaptive_shift.for_core(len(models.waveforms))
                        if adaptive_shift is not None else shift_radius)
                    core_match_diagnostics = None
                    if matcher is not None:
                        if matching_storage == "int16" and not (direct_int16_preprocessing or integer_filter):
                            # Release the FP32 GPU core before fitting. Host
                            # voltage remains available for QC/cache/adaptation.
                            device_signal = matcher.prepare_signal(
                                device_signal if device_signal is not None else signal)
                        events, matches = matcher.match(
                            device_signal if device_signal is not None else signal,
                            noise, floor_snr, core_start=local_start,
                            core_stop=local_stop, score_floor=score_floor,
                            min_margin=min_margin,
                            max_passes=residual_passes,
                            overlap_policy=overlap_policy,
                            overlap_window_samples=strict_overlap_window_samples,
                            rescue_floor_snr=rescue_floor_snr,
                            rescue_gain_floor_snr=rescue_gain_floor_snr,
                            rescue_score_floor=rescue_score_floor,
                            rescue_last_score_floor=rescue_last_score_floor,
                            rescue_last_amplitude_min=rescue_last_amplitude_min,
                            rescue_passes=rescue_passes, shift_radius=core_shift_radius,
                            half_width=half_width,
                            detector_mode=detector_mode,
                            amplitude_min=(fit_amplitude_min if fit_amplitude_min is not None else .3),
                            refractory_samples=max(2, round(probe.sample_rate_hz*.0005)),
                            max_candidates=max_candidates)
                        core_match_diagnostics = matcher.last_diagnostics
                    else:
                        events = _detect_cpu(signal, noise, floor_snr,
                                             max_candidates=max_candidates)
                        all_events = []
                        def redetect(residual, noise_uv, floor):
                            return _detect_cpu(residual, noise_uv, floor,
                                               max_candidates=max_candidates)
                        matches, _ = match_residual_cpu(
                            signal, events, models, score_floor=score_floor,
                            min_margin=min_margin, detector=redetect,
                            max_passes=residual_passes,
                            shift_radius=shift_radius,
                            half_width=half_width,
                            amplitude_min=(fit_amplitude_min if fit_amplitude_min is not None else .3),
                            noise_uv=noise, floor_snr=floor_snr,
                            core_start=local_start, core_stop=local_stop,
                            refractory_samples=max(2, round(probe.sample_rate_hz*.0005)),
                            all_events=all_events,
                            template_identity_map=template_link_map)
                        events = all_events
                    if refit_rounds:
                        from .session_local_refit import local_refit
                        matches = local_refit(signal, noise, models, matches,
                            core_start=local_start, core_stop=local_stop, rounds=refit_rounds,
                            half_width=half_width, shift_radius=shift_radius,
                            score_floor=score_floor, min_margin=min_margin,
                            amplitude_min=(fit_amplitude_min if fit_amplitude_min is not None else .3),
                            gain_floor=floor_snr**2,
                            refractory_samples=max(2,round(probe.sample_rate_hz*.0005)))
                    learning_events, learning_matches = events, matches
                    if spatial_recovery:
                        learning_events, learning_matches = matcher.learning_events, matcher.learning_matches
                        if learning_events is None or learning_matches is None:
                            raise RuntimeError("Recovery did not retain baseline learning observations")
                    if adaptive_shift is not None:
                        completed_pilot = adaptive_shift.observe(
                            learning_matches, core.core_stop-core.core_start,
                            interval_good=not quality.interval_bad)
                        if completed_pilot:
                            emit("adaptive_shift_calibrated", probe_id=probe.probe_id,
                                 day_id=day_id,
                                 expanded_units=int(np.sum(
                                     adaptive_shift.radii > shift_radius)),
                                 units=len(adaptive_shift.radii),
                                 pilot_samples=adaptive_shift.pilot_samples)
                    if core_match_diagnostics is not None:
                        for name, value in core_match_diagnostics.items():
                            if name == 'passes':
                                continue
                            totals['matcher_diagnostics'][name] = (
                                totals['matcher_diagnostics'].get(name, 0) + value)
                            shard_match_diagnostics[name] = (
                                shard_match_diagnostics.get(name, 0) + value)
                    writer.append_core(
                        core=core, voltage_uv=signal, quality=quality,
                        threshold_events=events, matches=matches,
                        channel_map=neighbor_map,
                        cache_fraction=cache_fraction,
                        cache_mode=waveform_cache_mode,
                        model_version=models.version, model_channels=models.channels)
                    if not freeze_templates:
                        _collect_updates(models, learning_matches, signal, quality,
                                         evidence, data_start=core.data_start,
                                         core_start=core.core_start)
                    evidence.expire(core.core_stop)
                    if novel is not None:
                        novel.collect(models, learning_events, learning_matches, signal, quality, neighbor_map, core)
                    if spatial_recovery:
                        matcher.learning_events = matcher.learning_matches = None
                    del learning_events, learning_matches
                    for match in matches:
                        models.assigned[match.unit] += 1
                    totals["source_bytes_read"] += core.source_bytes_read
                    emit("core_complete", probe_id=probe.probe_id,
                         stop_sample=core.core_stop, spikes=len(matches),
                         detections=len(events),
                         matcher_diagnostics=core_match_diagnostics)
                    shard_bytes += core.source_bytes_read
                    raw_bytes = ((core.core_stop - core.core_start) *
                                 probe.n_channels * 2)
                    totals["raw_bytes_covered"] += raw_bytes
                    shard_raw_bytes += raw_bytes
                    shard_peak_rss = max(shard_peak_rss,
                                         process.memory_info().rss)
                    if backend == "cuda":
                        pool = cp.get_default_memory_pool()
                        shard_peak_gpu_pool = max(
                            shard_peak_gpu_pool, pool.total_bytes())
                        free, total = cp.cuda.runtime.memGetInfo()
                        shard_peak_total_vram = max(
                            shard_peak_total_vram, total-free)
                    # Do not retain the previous voltage/events while the
                    # next core's preprocessing allocates its working set.
                    del voltage, signal, device_signal, events, matches
                update_audit = evidence.promote(models, stop)
                if novel is not None:
                    novel_audit = novel.evaluate(models, stop, enroll=novelty == 'enroll')
                    max_local_models = _configure_template_proposals(
                        models, probe, template_proposal_radius_um)
                    if backend == "cuda" and max_local_models > 512:
                        raise ValueError(
                            f"Template enrollment grew a local proposal pool to "
                            f"{max_local_models}; reduce the proposal radius")
                    evidence.extend_models(models)
                    novel.save(writer.handle, novel_audit)
                    if novelty_backfill:
                        writer.handle.create_dataset('novelty_recovered_spikes', data=novel.recovered,
                                                     compression='lzf')
                        if capture_phy_features:
                            writer.handle.create_dataset('novelty_recovered_features',
                                data=novel.recovered_features, compression='lzf')
                    enrolled = sum(row['enrolled_unit'] is not None for row in novel_audit)
                    totals['enrolled_units'] += enrolled
                    totals['recovered_novelty_spikes'] += len(novel.recovered)
                    emit('novelty_complete', probe_id=probe.probe_id, stop_sample=stop,
                         proposals=len(novel_audit), enrolled=enrolled, recovered=len(novel.recovered),
                         retired_tentative=novel.retired_tentative,
                         retired_raw_holdout=novel.retired_raw_holdout)
                if adaptive_shift is not None:
                    adaptive_shift.extend_units(len(models.waveforms))
                evidence.save(writer.handle, update_audit)
                promotions = sum(row['promoted'] for row in update_audit)
                evidence_rows = sum(len(rows) for rows in evidence.rows.values())
                totals['template_promotions'] += promotions
                totals['peak_evidence_rows'] = max(totals['peak_evidence_rows'], evidence_rows)
                emit("adaptation_complete", probe_id=probe.probe_id,
                     stop_sample=stop, proposed=len(update_audit),
                     promoted=promotions, evidence_rows=evidence_rows)
                candidate_count = len(writer.candidates)
                spike_count = len(writer.spikes) + (len(novel.recovered) if novel is not None and novelty_backfill else 0)
                writer.finish(models, telemetry={
                    "wall_seconds": time.perf_counter() - shard_started,
                    "source_bytes_read": shard_bytes,
                    "raw_bytes_covered": shard_raw_bytes,
                    "peak_process_rss_bytes": shard_peak_rss,
                    "peak_gpu_pool_bytes": shard_peak_gpu_pool,
                    "peak_total_vram_used_bytes": shard_peak_total_vram,
                    "matcher_diagnostics": shard_match_diagnostics,
                }, adaptive_shift=adaptive_shift)
            except BaseException:
                writer.abort()
                raise
            totals["shards"] += 1
            totals["candidate_rows"] += candidate_count
            totals["spikes"] += spike_count
            totals["cached_waveform_bytes"] += writer.cached_bytes
            totals["derived_bytes"] += output.stat().st_size
            totals["peak_process_rss_bytes"] = max(
                totals["peak_process_rss_bytes"], shard_peak_rss)
            totals["peak_gpu_pool_bytes"] = max(
                totals["peak_gpu_pool_bytes"], shard_peak_gpu_pool)
            totals["peak_total_vram_used_bytes"] = max(
                totals["peak_total_vram_used_bytes"], shard_peak_total_vram)
    totals["unresolved_cross_day_edges"] = build_cross_day_links(
        session, journal.root)
    totals["wall_seconds"] = time.perf_counter() - started
    totals["derived_bytes_per_raw_byte"] = (
        totals["derived_bytes"] / totals["raw_bytes_covered"]
        if totals["raw_bytes_covered"] else None)
    totals["output_root"] = str(journal.root)
    totals["experimental_quality_gate_passed"] = False
    return totals
