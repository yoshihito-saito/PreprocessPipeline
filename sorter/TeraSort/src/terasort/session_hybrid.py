"""Restartable bounded seed learning before the streaming session sorter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

from .session_manifest import load_session
from .session_seed_learning import learn_seed_models
from .kilosort_settings import validate_seed_settings
from .session_contact_policy import BAD_CONTACT_POLICY, recorded_contact_exclusion


# Chosen dense profile from the full public250 hybrid comparison. Keep this
# separate from the legacy run_session defaults so old checkpoints still resume.
SESSION_SORT_PROFILE = {
    "preprocessing_backend": "cuda-int16",
    "matching_storage": "int16",
    "integer_evidence": True,
    "exclude_bad_contacts": True,
    "template_contact_scoring": True,
    "spatial_recovery": True,  # Selected after quality, learning-isolation and replay checks.
    "novelty": "enroll",
    "novelty_denoising": True,
    "novelty_tentative_seconds": 30.,
    "novelty_candidate_floor_snr": 4.5,
    "novelty_stratified_admission": False,
    "novelty_priority_admission": True,
    "novelty_preserve_strong_tentative": True,
    "novelty_raw_anchor_holdout": True,
    "novelty_raw_minimum_cosine": .85,
    "novelty_backfill": True,
    "novelty_peak_alignment": True,  # Selected after paired recall and exact replay checks.
    "novelty_residual_duplicates": True,  # Bounded raw holdout; no new identity inside equivalence cutoff.
    "detector_mode": "smooth3",
    "score_floor": .76,
    "min_margin": 0.,
    "fit_amplitude_min": .7,
    "template_merge_cosine": .93,
    "rescue_floor_snr": 3.5,
    "rescue_passes": 3,
    "rescue_last_score_floor": .8,
    "rescue_last_amplitude_min": .5,
    "waveform_cache_mode": "balanced",
    "shard_seconds": 60.,
    "read_buffer_mb": 64,
    "prefetch_depth": 4,
}


def session_sort_profile(output_root, *, resume=False):
    """Select defaults while retaining the recorded fitting policy on resume."""
    profile = dict(SESSION_SORT_PROFILE)
    recorded = Path(output_root) / "run.json"
    if resume and recorded.is_file():
        config = json.loads(recorded.read_text(encoding="utf-8"))["config"]
        profile["exclude_bad_contacts"] = recorded_contact_exclusion(config)
        recovery = config.get("spatial_recovery")
        if recovery:
            from .session_spatial_recovery import RULE_VERSION
            if not isinstance(recovery, dict) or recovery.get("algorithm") != RULE_VERSION:
                raise ValueError("Unsupported recorded spatial recovery policy")
        profile["spatial_recovery"] = bool(recovery)
        profile["rescue_passes"] = config["rescue_passes"]
        profile['template_contact_scoring'] = bool(config.get('template_contact_scoring'))
        profile['novelty'] = config.get('novelty', 'off')
        profile['novelty_denoising'] = config.get('novelty_features') == 'seed_temporal_pca3_v1'
        profile['novelty_backfill'] = config.get('novelty_backfill') == 'compact_int16_frozen_fit_support64_v1'
        duplicate_validation = config.get('novelty_duplicate_validation')
        profile['novelty_residual_duplicates'] = duplicate_validation is not None
        if duplicate_validation is not None:
            from .session_novelty_residual import policy
            if duplicate_validation != policy():
                raise ValueError('Unsupported recorded novelty duplicate validation')
            profile['novelty_residual_duplicates'] = True
        alignment = config.get('novelty_peak_alignment')
        if alignment:
            from .session_novelty import peak_alignment_policy
            if alignment != peak_alignment_policy(nt=config.get('nt', 61)):
                raise ValueError('Unsupported recorded novelty peak alignment policy')
        profile['novelty_peak_alignment'] = bool(alignment)
        validation = config.get('novelty_raw_validation', {})
        profile['novelty_raw_anchor_holdout'] = validation.get('algorithm') == 'raw_anchor_center17_holdout_cosine_v1'
        profile['novelty_raw_minimum_cosine'] = validation.get('minimum_cosine', .9 if profile['novelty_raw_anchor_holdout'] else .85)
        retention = config.get('novelty_retention', {})
        profile['novelty_tentative_seconds'] = retention.get('seconds', 0.)
        profile['novelty_preserve_strong_tentative'] = retention.get('algorithm') == 'weak_below3_source_idle_strong30min_snr6_v2'
        profile['novelty_candidate_floor_snr'] = config.get('novelty_candidate_floor', {}).get('minimum', 6.)
        profile['novelty_stratified_admission'] = config.get('novelty_admission') == 'strong_reserve_weak_quarter_cap_v1'
        profile['novelty_priority_admission'] = config.get('novelty_admission') == 'unsupported_priority_replace_snr6_ratio125_v1'
    elif resume:
        seed_plan = Path(output_root).with_name(Path(output_root).name + '.seeds') / 'plan.json'
        if seed_plan.is_file():
            profile['exclude_bad_contacts'] = recorded_contact_exclusion(
                json.loads(seed_plan.read_text(encoding='utf-8')))
    return profile


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _write_json_atomic(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _remove_generated_partial(path, seed_root):
    """Discard only our interrupted seed attempt within the named seed root."""
    root = seed_root.resolve(strict=True)
    target = path.resolve(strict=True)
    if (not target.is_relative_to(root / ".partial")
            or target == root / ".partial"
            or not path.is_dir()):
        raise ValueError("Refusing to remove a path outside the seed scratch area")
    shutil.rmtree(target)


def _verify_completed_seed(folder, *, probe_id, day_id, budget_seconds,
                           window_count, preprocessing_backend,
                           cuda_cluster_updates_enabled, exclude_bad_contacts=False, nt=61,
                           ks4_settings=None, ks4_backend="deep_tiled"):
    report_path = folder / "report.json"
    seed_path = folder / "seeds.npz"
    if not report_path.is_file() or not seed_path.is_file():
        raise ValueError(f"Incomplete published seed directory: {folder}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "probe_id": probe_id, "day_id": day_id,
        "requested_budget_seconds": float(budget_seconds),
        "requested_window_count": window_count,
        "preprocessing_backend": preprocessing_backend,
        "requested_cuda_cluster_updates": cuda_cluster_updates_enabled,
    }
    if (report.get("ks4_settings", {}) != (ks4_settings or {})
            or report.get("ks4_backend", "deep_tiled") != ks4_backend):
        raise ValueError("KS4 seed settings/backend changed on resume")
    if report.get('nt', 61) != nt:
        raise ValueError('Seed waveform length differs from requested nt')
    if exclude_bad_contacts:
        expected['bad_contact_neighborhoods'] = BAD_CONTACT_POLICY
    if (any(report.get(key) != value for key, value in expected.items())
            or recorded_contact_exclusion(report) != exclude_bad_contacts
            or report.get("seeds_sha256") != _digest(seed_path)):
        raise ValueError(f"Seed configuration or checksum differs: {folder}")
    return seed_path


def prepare_hybrid_manifest(source_manifest, output_root, *, resume=False,
                            budget_seconds=60., window_count=3,
                            preprocessing_backend="cpu",
                            cuda_cluster_updates_enabled=False,
                            seed_root=None, exclude_bad_contacts=False, nt=61,
                            ks4_settings=None, ks4_backend="deep_tiled"):
    """Return a generated manifest with immutable per-probe/day preview seeds.

    Seed files live in a sibling directory so ``run_session`` still receives a
    new output root. On resume, the source plan and seed checksums are verified
    before any sorting shard is reopened.
    """
    from .session_models import validate_nt
    nt = validate_nt(nt)
    ks4_settings = validate_seed_settings(ks4_settings)
    if ks4_backend not in ("auto", "standard", "deep_tiled", "cublas"):
        raise ValueError("Unsupported KS4 seed backend")
    if ks4_backend == "cublas" and (nt not in (41, 61) or ks4_settings.get("n_templates", 6) != 6):
        raise ValueError("The cuBLAS seed backend requires nt=41 or 61 and n_templates=6")
    if not 6 <= budget_seconds <= 180:
        raise ValueError("Calibration budget must be 6–180 seconds")
    if (isinstance(window_count, bool) or not isinstance(window_count, int)
            or not 3 <= window_count <= 12):
        raise ValueError("Calibration window count must be 3–12")
    if preprocessing_backend not in ("cpu", "cuda"):
        raise ValueError("Seed preprocessing backend must be cpu or cuda")
    if not isinstance(cuda_cluster_updates_enabled, bool):
        raise ValueError("CUDA seed cluster flag must be boolean")
    if not isinstance(exclude_bad_contacts, bool):
        raise ValueError('Bad-contact exclusion must be boolean')
    source_path = Path(source_manifest).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve()
    seeds = (Path(seed_root).expanduser().resolve() if seed_root is not None
             else output.with_name(output.name + ".seeds"))
    if (seeds == output or seeds.is_relative_to(output)
            or output.is_relative_to(seeds)):
        raise ValueError("Seed root and sorter output root must be separate")
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and output.exists() and not seeds.is_dir():
        raise ValueError("Cannot resume sorter without its seed directory")
    session = load_session(source_path)
    if (ks4_settings or ks4_backend != "deep_tiled") and any(
            probe.seed_templates is not None or probe.seed_templates_by_day for probe in session.probes):
        raise ValueError("KS4 seed overrides cannot be applied to existing external seeds")
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    source_record = session.as_source_record()
    plan = {
        "schema_version": 1,
        "source_manifest": str(source_path),
        "source_manifest_sha256": _digest(source_path),
        "source_record_sha256": hashlib.sha256(
            json.dumps(source_record, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
        "budget_seconds": float(budget_seconds),
        "window_count": window_count,
        "preprocessing_backend": preprocessing_backend,
        "cuda_cluster_updates_enabled": cuda_cluster_updates_enabled,
    }
    if nt != 61:
        plan['nt'] = nt
    if ks4_settings:
        plan["ks4_settings"] = ks4_settings
    if ks4_backend != "deep_tiled":
        plan["ks4_backend"] = ks4_backend
    if exclude_bad_contacts:
        plan['bad_contact_neighborhoods'] = BAD_CONTACT_POLICY
    if not seeds.exists():
        seeds.mkdir(parents=True)
    elif not resume:
        raise FileExistsError(seeds)
    plan_path = seeds / "plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Hybrid seed source or settings changed on resume")
    else:
        contents = list(seeds.iterdir())
        if contents and contents != [seeds / "plan.json.partial"]:
            raise ValueError("Seed directory has no published plan")
        _write_json_atomic(plan_path, plan)

    by_id = {probe.probe_id: probe for probe in session.probes}
    for item in raw["probes"]:
        probe = by_id[item["probe_id"]]
        for segment_json, segment in zip(item["segments"], probe.segments):
            segment_json["path"] = str(segment.path)
        if probe.seed_templates is not None:
            item["seed_templates"] = str(probe.seed_templates)
        mapping = {day: str(path)
                   for day, path in probe.seed_templates_by_day.items()}
        for day_id in dict.fromkeys(segment.day_id for segment in probe.segments):
            if probe.seed_for_day(day_id) is not None:
                continue
            folder = seeds / probe.probe_id / day_id
            if folder.exists():
                seed_path = _verify_completed_seed(
                    folder, probe_id=probe.probe_id, day_id=day_id,
                    budget_seconds=budget_seconds, window_count=window_count,
                    preprocessing_backend=preprocessing_backend,
                    cuda_cluster_updates_enabled=cuda_cluster_updates_enabled,
                    exclude_bad_contacts=exclude_bad_contacts, nt=nt,
                    ks4_settings=ks4_settings, ks4_backend=ks4_backend)
            else:
                partial = seeds / ".partial" / probe.probe_id / day_id
                partial.parent.mkdir(parents=True, exist_ok=True)
                if partial.exists():
                    _remove_generated_partial(partial, seeds)
                learn_seed_models(
                    probe, day_id, partial,
                    budget_seconds=budget_seconds, window_count=window_count,
                    preprocessing_backend=preprocessing_backend,
                    cuda_cluster_updates_enabled=cuda_cluster_updates_enabled,
                    keep_preview=False, exclude_bad_contacts=exclude_bad_contacts, nt=nt,
                    ks4_settings=ks4_settings, ks4_backend=ks4_backend)
                folder.parent.mkdir(parents=True, exist_ok=True)
                root_resolved = seeds.resolve(strict=True)
                if (not partial.resolve(strict=True).is_relative_to(root_resolved)
                        or not folder.resolve().is_relative_to(root_resolved)):
                    raise ValueError("Seed directory move left the named seed root")
                os.replace(partial, folder)
                seed_path = _verify_completed_seed(
                    folder, probe_id=probe.probe_id, day_id=day_id,
                    budget_seconds=budget_seconds, window_count=window_count,
                    preprocessing_backend=preprocessing_backend,
                    cuda_cluster_updates_enabled=cuda_cluster_updates_enabled,
                    exclude_bad_contacts=exclude_bad_contacts, nt=nt,
                    ks4_settings=ks4_settings, ks4_backend=ks4_backend)
            mapping[day_id] = str(seed_path)
            item["seed_preprocessing_id"] = probe.preprocessing_id
        if mapping:
            item["seed_templates_by_day"] = mapping
    generated = seeds / "session_with_seeds.json"
    if generated.is_file():
        if json.loads(generated.read_text(encoding="utf-8")) != raw:
            raise ValueError("Published hybrid manifest differs on resume")
    else:
        _write_json_atomic(generated, raw)
    load_session(generated)
    return generated


def manifest_for_session_run(source_manifest, output_root, *, resume=False, nt=61,
                             ks4_settings=None, ks4_backend="deep_tiled"):
    """Choose the fixed seed-learning path while preserving old checkpoints."""
    ks4_settings = validate_seed_settings(ks4_settings)
    source = Path(source_manifest).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    seed_root = output.with_name(output.name + ".seeds")
    if resume and output.exists() and not seed_root.exists():
        if ks4_settings or ks4_backend != "deep_tiled":
            raise ValueError("Legacy runs without KS4 seed learning cannot use KS4 seed overrides")
        run_path = output / "run.json"
        if not run_path.is_file():
            raise ValueError(f"Missing seed directory for interrupted run: {seed_root}")
        recorded = json.loads(run_path.read_text(encoding="utf-8"))["source"]
        if Path(recorded["manifest_path"]).resolve() != source:
            raise ValueError(f"Missing seed directory for this run: {seed_root}")
        return source, True
    exclusion = True
    # An interrupted pre-policy seed attempt retains its original plan too.
    if resume and (seed_root / 'plan.json').is_file():
        exclusion = recorded_contact_exclusion(json.loads((seed_root / 'plan.json').read_text(encoding='utf-8')))
    generated = prepare_hybrid_manifest(source, output, resume=resume,
        exclude_bad_contacts=exclusion, nt=nt,
        ks4_settings=ks4_settings, ks4_backend=ks4_backend)
    return generated, bool(resume and output.exists())
