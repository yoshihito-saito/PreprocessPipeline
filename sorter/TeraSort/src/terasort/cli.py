"""Command-line entry point for Kilosort-compatible sorting."""

import argparse
import json
from pathlib import Path

import numpy as np

from .api import available_backends, run_kilosort


def main(argv=None):
    parser = argparse.ArgumentParser(prog="terasort", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backends", help="List available sorting backends")
    features = sub.add_parser("phy-features", help="Add measured PCA features to an existing Phy folder without changing curation")
    features.add_argument("--folder", required=True)
    export = sub.add_parser("session-export", help="Export completed session shards to Kilosort/Phy arrays")
    export.add_argument("--source-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True, help="New export directory")
    web = sub.add_parser("web", help="Run the local browser dashboard and job queue")
    web.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost)")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--state-dir", type=Path, help="Persistent job state directory")
    web.add_argument("--token", help="Bearer token required when binding beyond localhost")
    web.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    ks1 = sub.add_parser('ks1-sort', help='Experimental bounded original-KS1 CUDA engine')
    ks1.add_argument('--manifest', type=Path, required=True)
    ks1.add_argument('--output-root', type=Path, required=True)
    ks1.add_argument('--kilosort-path', type=Path, required=True, help='Pinned original KS1 checkout with CUDA sources and calibration MEX files')
    ks1.add_argument('--matlab', type=Path, help='MATLAB executable for bounded calibration; auto-detected if omitted')
    ks1.add_argument('--model', type=Path, help='Optional existing native rez.mat or bounded model.npz, single probe/day only')
    ks1.add_argument('--resume', action='store_true')
    session = sub.add_parser(
        "session-sort", help="Experimental bounded multi-file session sorter")
    # Fixed chosen fitting policy; no additional end-user tuning switch.
    session.set_defaults(template_contact_scoring=False, novelty_denoising=False,
                         novelty_tentative_seconds=0., novelty_candidate_floor_snr=6.,
                         novelty_stratified_admission=False, novelty_priority_admission=False,
                         novelty_preserve_strong_tentative=False, novelty_raw_anchor_holdout=False,
                         novelty_raw_minimum_cosine=.85, novelty_backfill=False, novelty_peak_alignment=False,
                         novelty_residual_duplicates=False,
                         exclude_bad_contacts=False)
    session.add_argument("--manifest", type=Path, required=True,
                         help="Versioned JSON source clock, geometry and gains")
    session.add_argument("--output-root", type=Path, required=True,
                         help="New run directory; use --resume to continue")
    session.add_argument("--phy-flat-output", type=Path,
                         help="Publish one probe/day directly into this new Phy folder")
    session.add_argument("--nt", type=int, default=61,
                         help="Odd bounded waveform length in samples (>=21)")
    session.add_argument("--seed-settings", type=Path,
                         help="JSON KS4 scientific overrides for initial seed learning")
    session.add_argument("--seed-backend", choices=("auto", "standard", "deep_tiled", "cublas"),
                         default="deep_tiled", help="KS4 backend used only for seed learning")
    session.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    session.add_argument("--preprocessing-backend", choices=("cpu", "cuda", "cuda-int16"), default="cpu",
                         help="Opt-in CUDA filtering; experimental cuda-int16 also requires --matching-storage int16")
    session.add_argument("--matching-storage", choices=("float32", "int16"), default="float32",
                         help="Experimental scaled INT16 GPU residual/templates; FP32 fitting; requires CUDA")
    session.add_argument("--direct-int16-preprocessing", action="store_true",
                         help="Experimental slab filtering directly into INT16 GPU residual; requires CUDA preprocessing and INT16 storage")
    session.add_argument("--integer-evidence", action="store_true",
                         help="Store retained template-update snippets as INT16 plus scales; requires cuda-int16 preprocessing")
    session.add_argument("--preset", choices=("default", "fast"), default="default",
                         help="fast: one residual pass, metadata-only cache, 256 MiB block prefetch; explicit flags override")
    session.add_argument("--resume", action="store_true")
    session.add_argument("--freeze-templates", action="store_true",
                         help="Disable evidence collection and template adaptation")
    session.add_argument("--novelty", choices=("off", "shadow", "enroll"), default="off",
                         help="Bounded unknown-event proposals; the chosen CUDA profile uses PCA enrollment")
    session.add_argument("--core-seconds", type=float, default=2.)
    session.add_argument("--shard-seconds", type=float, default=300.)
    session.add_argument("--residual-passes", type=int, default=3,
                         help="Bounded overlap-recovery passes (1–12; default 3)")
    session.add_argument('--overlap-policy', choices=('strict','interference'), default='strict')
    session.add_argument('--overlap-window-samples', type=int,
                         help='Experimental CUDA strict-mode cross-unit exclusion window (0–nt samples; default nt)')
    session.add_argument("--halo-ms", type=float, default=100.)
    session.add_argument("--read-buffer-mb", type=int, default=0,
                         help="Sequential read buffer in MiB, 0 disables grouping (max 1024)")
    session.add_argument("--prefetch-depth", type=int, default=2,
                         help="Bounded queue of raw cores (1–64; default 2)")
    session.add_argument("--floor-snr", type=float, default=4.5)
    session.add_argument('--rescue-floor-snr', type=float,
                         help='Optional lower-threshold final CUDA pass with stricter fitting')
    session.add_argument('--rescue-gain-floor-snr', type=float,
                         help='Experimental rescue-only residual-gain floor; requires --rescue-floor-snr')
    session.add_argument('--rescue-score-floor', type=float,
                         help='Experimental rescue-only waveform score floor; requires --rescue-floor-snr')
    session.add_argument('--rescue-last-score-floor', type=float,
                         help='Score floor for the relaxed rescue tail; requires 2-4 rescue passes')
    session.add_argument('--rescue-last-amplitude-min', type=float,
                         help='Lower fitted-amplitude floor for the relaxed CUDA rescue tail; requires 2-4 rescue passes')
    session.add_argument('--rescue-passes', type=int, default=1)
    session.add_argument('--shift-radius', type=int, default=2,
                         help='Candidate timing search radius in samples (0–8)')
    session.add_argument('--adaptive-shift-radius', action='store_true',
                         help='Pilot one wider timing search, then expand it only for units with boundary fits')
    session.add_argument('--adaptive-shift-boundary-fraction', type=float, default=.05,
                         help='Pilot boundary-fit fraction required to retain the wider radius (default .05)')
    session.add_argument("--score-floor", type=float, default=.65)
    session.add_argument('--template-proposal-radius-um', type=float, default=48.,
                         help='Same-shank anchor radius for local template proposals (default 48 um)')
    session.add_argument("--export-amplitude-min", type=float,
                         help="Optional output-view filter on fitted amplitude; full spike rows remain stored")
    session.add_argument("--fit-amplitude-min", type=float,
                         help="Raise fitted-amplitude minimum before residual subtraction (0–3; default 0.3)")
    session.add_argument("--template-merge-cosine", type=float,
                         help="Optional identity linking for near-duplicate seed templates; also changes matcher margins")
    session.add_argument("--template-merge-radius-um", type=float, default=32.,
                         help="Maximum same-shank anchor spacing for template linking (default 32 um)")
    session.add_argument("--template-merge-mode", choices=("components", "reciprocal"),
                         default="components", help="Template linking rule (default components)")
    session.add_argument('--half-width', type=int, default=8,
                         help='Scoring half width in samples (3–30)')
    session.add_argument('--refit-rounds', type=int, default=0,
                         help='Experimental bounded neighbor-subtracted CPU refits (0–2)')
    session.add_argument('--detector-mode', choices=('raw','smooth3'), default='raw')
    session.add_argument("--min-margin", type=float, default=.03)
    session.add_argument("--cache-fraction", type=float, default=.05)
    session.add_argument("--waveform-cache-mode", choices=("legacy", "balanced", "recall"),
                         default="legacy", help="Experimental cache policy; recall samples weak unknown waveforms")
    session.add_argument("--max-candidates", type=int, default=500_000)
    session.add_argument("--vram-limit-gb", type=float, default=12.)
    session.add_argument("--start-sample", type=int, default=0)
    session.add_argument("--stop-sample", type=int)
    view = sub.add_parser(
        "session-view", help="Build a bounded amplitude-filter view of completed session shards")
    view.add_argument("--source-root", type=Path, required=True)
    view.add_argument("--output-root", type=Path, required=True,
                      help="New directory for compact masks; source shards stay unchanged")
    view.add_argument("--amplitude-min", type=float, required=True)
    view.add_argument("--probe-id", help="Select one probe (default: all probes)")
    view.add_argument("--resume", action="store_true")
    provisional_view = sub.add_parser(
        "session-provisional-view",
        help="Build a bounded support mask for two-tier rescue assignments")
    provisional_view.add_argument("--source-root", type=Path, required=True)
    provisional_view.add_argument("--output-root", type=Path, required=True)
    provisional_view.add_argument("--probe-id", help="Select one probe (default: all probes)")
    provisional_view.add_argument("--resume", action="store_true")
    cofire_view = sub.add_parser(
        "session-cofire-view",
        help="Build an experimental bounded co-firing duplicate view of a frozen-template session")
    cofire_view.add_argument("--source-root", type=Path, required=True)
    cofire_view.add_argument("--output-root", type=Path, required=True)
    cofire_view.add_argument("--probe-id", required=True)
    cofire_view.add_argument("--calibration-root", type=Path,
                             help="Earlier matching frozen-template session; default is source root")
    cofire_view.add_argument("--pilot-seconds", type=float, default=30.)
    cofire_view.add_argument("--amplitude-min", type=float, default=.7)
    cofire_view.add_argument("--resume", action="store_true")
    temporal = sub.add_parser(
        "session-temporal-links",
        help="Build an experimental bounded adjacent-epoch identity view")
    temporal.add_argument("--source-root", type=Path, required=True)
    temporal.add_argument("--output-root", type=Path, required=True,
                          help="New sidecar directory; source shards stay unchanged")
    temporal.add_argument("--probe-id", required=True)
    temporal.add_argument("--first-sample", type=int)
    temporal.add_argument("--stop-sample", type=int)
    temporal.add_argument("--resume", action="store_true")
    motion = sub.add_parser(
        "session-motion-links",
        help="Build experimental bounded motion-aligned adjacent-shard identity view")
    motion.add_argument("--source-root", type=Path, required=True)
    motion.add_argument("--output-root", type=Path, required=True)
    motion.add_argument("--probe-id", required=True)
    motion.add_argument("--temporal-link-root", type=Path,
                        help="Optional matching temporal-link view as the base labels")
    motion.add_argument("--resume", action="store_true")
    quality = sub.add_parser(
        "session-quality", help="Compare completed session shards with ground truth and Kilosort4")
    quality.add_argument("--suite", type=Path, required=True,
                         help="JSON suite with ground truth, session output and Kilosort4 output")
    quality.add_argument("--report", type=Path,
                         help="Optional new JSON report file")
    lfp = sub.add_parser("lfp", help="Stream anti-aliased INT16 voltage to an LFP binary")
    lfp.add_argument("--filename", type=Path, required=True, help="Time-major interleaved INT16 input")
    lfp.add_argument("--output", type=Path, required=True, help="Time-major interleaved INT16 LFP output")
    lfp.add_argument("--sample-rate", type=int, required=True, help="Input sample rate in Hz")
    lfp.add_argument("--n-channels", type=int, required=True)
    lfp.add_argument("--output-rate", type=int, default=1250)
    lfp.add_argument("--passband-hz", type=float, default=500.0)
    lfp.add_argument("--scale-uv-per-count", type=float)
    lfp.add_argument("--chunk-seconds", type=float, default=5.0)
    lfp.add_argument("--workers", type=int, default=8,
                     help="CPU channel-filter workers (default 8)")
    lfp.add_argument("--resume", action="store_true", help="Continue a matching .partial export")
    sort = sub.add_parser("sort", help="Sort one or more ordered binary files into one Kilosort session")
    sort.add_argument("--settings", type=Path, required=True,
                      help="JSON dictionary of Kilosort settings, including n_chan_bin")
    sort.add_argument("--filename", type=Path, required=True, action="append",
                      help="Raw binary file; repeat in acquisition order for one shared session")
    probe = sort.add_mutually_exclusive_group()
    probe.add_argument("--probe-name", help="Bundled Kilosort probe name")
    probe.add_argument("--probe-json", type=Path, help="Probe dictionary saved as JSON")
    sort.add_argument("--results-dir", type=Path, required=True)
    sort.add_argument("--data-dtype", default="int16")
    sort.add_argument("--backend", choices=("auto", "standard", "deep_tiled", "cublas"),
                      default="auto")
    sort.add_argument("--no-fast-int16", action="store_true")
    sort.add_argument("--read-cache-dir", type=Path, help="Bounded SSD cache with background network read-ahead")
    sort.add_argument("--read-cache-mb", type=int, default=4096)
    sort.add_argument("--read-cache-slots", type=int, default=3)
    sort.add_argument("--read-cache-retain", action="store_true",
                      help="Keep the source prefix for later Kilosort passes; use rolling slots for the tail")
    sort.add_argument("--read-cache-retain-gb", type=float, default=0.,
                      help="Maximum retained prefix in GiB (0: automatic with a free-space reserve)")
    sort.add_argument("--read-cache-workers", type=int, default=4,
                      help="Parallel range downloads per prefetched block (1–16; default 4)")
    sort.add_argument("--stage-dir", type=Path,
                      help="New local scratch directory for one-time source copies (retained after sorting)")
    sort.add_argument("--skip-drift-correction", action="store_true",
                      help="Use Kilosort nblocks=0; skips motion estimation and its detection pass")
    sort.add_argument("--invert-sign", action="store_true")
    sort.add_argument("--no-car", action="store_true")
    sort.add_argument("--input-is-preprocessed", action="store_true",
                      help="Bypass high-pass filtering and CAR; retain KS4 whitening")
    sort.add_argument("--verbose", action="store_true")
    sort.add_argument("--lfp-output", type=Path,
                      help="Export 1250 Hz LFP during final clustering, then wait for completion")
    sort.add_argument("--lfp-passband-hz", type=float, default=500.0)
    sort.add_argument("--lfp-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "phy-features":
        from .session_phy_features import generate_phy_features
        result = generate_phy_features(args.folder)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-export":
        from .session_export import export_session
        result = export_session(args.source_root, args.output_root,
            progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-sort":
        from .session_hybrid import session_sort_profile
        seed_root = args.output_root.with_name(args.output_root.name + ".seeds")
        legacy_resume = args.resume and args.output_root.exists() and not seed_root.exists()
        if not legacy_resume:
            session.set_defaults(**session_sort_profile(args.output_root, resume=args.resume))
        if args.preset == "fast":
            # Retain the old preset for exact legacy resumes. Explicit CLI
            # flags still win over any profile setting.
            session.set_defaults(residual_passes=1, cache_fraction=0.,
                                 read_buffer_mb=256, novelty='off')
        if not legacy_resume or args.preset == "fast":
            args = parser.parse_args(argv)
        if args.freeze_templates:
            import sys
            tokens = list(argv) if argv is not None else sys.argv[1:]
            if not any(t == '--novelty' or t.startswith('--novelty=') for t in tokens):
                args.novelty = 'off'
        if args.novelty == 'off':
            args.novelty_residual_duplicates = False
    if args.command == 'ks1-sort':
        from .session_ks1_sort import run_ks1_session, find_matlab
        result = run_ks1_session(args.manifest, args.output_root,
            kilosort_path=args.kilosort_path, matlab=args.matlab or find_matlab(),
            model_path=args.model, resume=args.resume,
            progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps(result))
        return 0
    if args.command == "backends":
        print("\n".join(available_backends()))
        return 0
    if args.command == "web":
        from .web import serve
        import os
        serve(host=args.host, port=args.port, state_dir=args.state_dir,
              token=args.token or os.environ.get("TERASORT_WEB_TOKEN"),
              open_browser=not args.no_browser)
        return 0
    if args.command == "session-sort":
        from .session_hybrid import manifest_for_session_run
        from .session_sort import run_session
        from .session_export import ensure_session_export
        seed_settings = (json.loads(args.seed_settings.read_text(encoding="utf-8"))
                         if args.seed_settings else None)
        manifest, resume_sort = manifest_for_session_run(
            args.manifest, args.output_root, resume=args.resume, nt=args.nt,
            ks4_settings=seed_settings, ks4_backend=args.seed_backend)
        result = run_session(
            manifest, args.output_root, backend=args.backend, nt=args.nt,
            resume=resume_sort, core_seconds=args.core_seconds,
            shard_seconds=args.shard_seconds, halo_ms=args.halo_ms,
            read_buffer_mb=args.read_buffer_mb, prefetch_depth=args.prefetch_depth,
            preprocessing_backend=args.preprocessing_backend,
            matching_storage=args.matching_storage,
            template_contact_scoring=args.template_contact_scoring,
            spatial_recovery=getattr(args, "spatial_recovery", False),
            direct_int16_preprocessing=args.direct_int16_preprocessing,
            integer_evidence=args.integer_evidence,
            exclude_bad_contacts=args.exclude_bad_contacts,
            floor_snr=args.floor_snr, score_floor=args.score_floor,
            half_width=args.half_width,
            refit_rounds=args.refit_rounds,
            detector_mode=args.detector_mode,
            rescue_floor_snr=args.rescue_floor_snr,
            rescue_gain_floor_snr=args.rescue_gain_floor_snr,
            rescue_score_floor=args.rescue_score_floor,
            rescue_last_score_floor=args.rescue_last_score_floor,
            rescue_last_amplitude_min=args.rescue_last_amplitude_min,
            rescue_passes=args.rescue_passes, shift_radius=args.shift_radius,
            adaptive_shift_radius=args.adaptive_shift_radius,
            adaptive_shift_boundary_fraction=args.adaptive_shift_boundary_fraction,
            min_margin=args.min_margin, cache_fraction=args.cache_fraction,
            waveform_cache_mode=args.waveform_cache_mode,
            max_candidates=args.max_candidates,
            start_sample=args.start_sample, stop_sample=args.stop_sample,
            vram_limit_gb=args.vram_limit_gb,
            freeze_templates=args.freeze_templates, novelty=args.novelty,
            novelty_denoising=args.novelty_denoising,
            novelty_tentative_seconds=args.novelty_tentative_seconds,
            novelty_candidate_floor_snr=args.novelty_candidate_floor_snr,
            novelty_stratified_admission=args.novelty_stratified_admission,
            novelty_priority_admission=args.novelty_priority_admission,
            novelty_preserve_strong_tentative=args.novelty_preserve_strong_tentative,
            novelty_raw_anchor_holdout=args.novelty_raw_anchor_holdout,
            novelty_raw_minimum_cosine=args.novelty_raw_minimum_cosine,
            novelty_backfill=args.novelty_backfill,
            novelty_peak_alignment=args.novelty_peak_alignment,
            novelty_residual_duplicates=args.novelty_residual_duplicates,
            residual_passes=args.residual_passes, overlap_policy=args.overlap_policy,
            overlap_window_samples=args.overlap_window_samples,
            template_merge_cosine=args.template_merge_cosine,
            template_merge_radius_um=args.template_merge_radius_um,
            template_merge_mode=args.template_merge_mode,
            template_proposal_radius_um=args.template_proposal_radius_um,
            export_amplitude_min=args.export_amplitude_min,
            fit_amplitude_min=args.fit_amplitude_min)
        result["phy_export"] = ensure_session_export(args.output_root,
            progress=lambda event: print(json.dumps(event), flush=True),
            **({"flat_output": args.phy_flat_output} if args.phy_flat_output is not None else {}))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-quality":
        from .session_quality import evaluate_suite
        result = evaluate_suite(args.suite)
        report = json.dumps(result, indent=2) + "\n"
        if args.report is not None:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(report)
        print(report, end="")
        return 0 if result["quality_gate_passed"] else 2
    if args.command == "session-view":
        from .session_view import build_session_view
        result = build_session_view(
            args.source_root, args.output_root, amplitude_min=args.amplitude_min,
            probe_id=args.probe_id, resume=args.resume)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-provisional-view":
        from .session_provisional_view import build_provisional_support_view
        result = build_provisional_support_view(
            args.source_root, args.output_root,
            probe_id=args.probe_id, resume=args.resume)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-cofire-view":
        from .session_cofire_view import build_cofire_view
        result = build_cofire_view(
            args.source_root, args.output_root, probe_id=args.probe_id,
            calibration_root=args.calibration_root,
            pilot_seconds=args.pilot_seconds,
            amplitude_min=args.amplitude_min, resume=args.resume)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-temporal-links":
        from .session_temporal_links import build_temporal_link_view
        result = build_temporal_link_view(
            args.source_root, args.output_root, probe_id=args.probe_id,
            first_sample=args.first_sample, stop_sample=args.stop_sample,
            resume=args.resume)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "session-motion-links":
        from .session_motion_links import build_motion_link_view
        result = build_motion_link_view(
            args.source_root, args.output_root, probe_id=args.probe_id,
            temporal_link_root=args.temporal_link_root, resume=args.resume)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "lfp":
        from .lfp import export_lfp
        result = export_lfp(args.filename, args.output,
            sample_rate_hz=args.sample_rate, n_channels=args.n_channels,
            output_rate_hz=args.output_rate, passband_hz=args.passband_hz,
            scale_uv_per_count=args.scale_uv_per_count,
            chunk_seconds=args.chunk_seconds, workers=args.workers,
            resume=args.resume)
        print(json.dumps(result, indent=2))
        return 0
    from .kilosort_settings import validate_kilosort_settings
    settings = validate_kilosort_settings(json.loads(args.settings.read_text(encoding="utf-8")))
    if args.input_is_preprocessed and "highpass_cutoff" in settings:
        parser.error("Preprocessed input does not use highpass_cutoff")
    if not isinstance(settings, dict) or not isinstance(settings.get("n_chan_bin"), int):
        parser.error("--settings must be a JSON object with integer n_chan_bin")
    probe_dict = json.loads(args.probe_json.read_text()) if args.probe_json else None
    if probe_dict is not None:
        if not isinstance(probe_dict, dict):
            parser.error("--probe-json must contain a Kilosort probe dictionary")
        for key, dtype in (("chanMap", np.int32), ("xc", np.float32),
                           ("yc", np.float32), ("kcoords", np.float32)):
            if key in probe_dict:
                probe_dict[key] = np.asarray(probe_dict[key], dtype=dtype)
    filename = args.filename[0] if len(args.filename) == 1 else args.filename
    run_kilosort(settings, probe=probe_dict, probe_name=args.probe_name,
                 filename=filename, results_dir=args.results_dir,
                 data_dtype=args.data_dtype, backend=args.backend,
                 fast_int16=not args.no_fast_int16,
                 input_is_preprocessed=args.input_is_preprocessed,
                 skip_drift_correction=args.skip_drift_correction,
                 stage_dir=args.stage_dir,
                 read_cache_dir=args.read_cache_dir, read_cache_mb=args.read_cache_mb,
                 read_cache_slots=args.read_cache_slots, read_cache_workers=args.read_cache_workers,
                 read_cache_retain=args.read_cache_retain,
                 read_cache_retain_gb=args.read_cache_retain_gb,
                 lfp_output=args.lfp_output,
                 lfp_passband_hz=args.lfp_passband_hz,
                 lfp_workers=args.lfp_workers,
                 invert_sign=args.invert_sign, do_CAR=not args.no_car,
                 verbose_console=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
