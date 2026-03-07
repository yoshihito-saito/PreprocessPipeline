from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import spikeinterface.extractors as se

from .artifact_removal import detect_high_amplitude_artifacts, remove_artifacts
from .events import export_analog_digital_events, materialize_intermediate_dat
from .io import (
    build_acquisition_catalog,
    discover_subsessions,
    ensure_rhd,
    ensure_xml,
    load_session_xml_metadata,
    load_xml_metadata,
    print_catalog_summary,
    resolve_basepath_and_basename,
    resolve_local_output_dir,
    save_params_and_manifest,
)
from .intan_rhd import read_intan_rhd_header
from .mergepoints import compute_mergepoints, save_mergepoints_events_mat
from .recording import (
    apply_preprocessing,
    attach_probe_and_remove_bad_channels,
    apply_transform_to_selected_channels_preserve_shape,
    concatenate_recordings_si,
    load_subsession_recordings,
    preprocess_selected_channels_preserve_shape,
    zero_selected_channels_preserve_shape,
    write_concatenated_dat_analogin,
    write_concatenated_dat_digitalin,
    write_concatenated_dat,
    write_lfp,
)
from .session import build_session_struct, save_session_mat
from .metafile import PreprocessConfig, PreprocessResult
from .sorter_runner import execute_sorting_job
from .state_scoring import run_state_scoring


def _sorter_output_prefix(sorter_name: str) -> str:
    sorter_normalized = sorter_name.strip().lower()
    if sorter_normalized == "kilosort":
        return "Kilosort"
    if sorter_normalized == "kilosort4":
        return "Kilosort4"
    return sorter_name[:1].upper() + sorter_name[1:].lower()


def _make_tree_world_rw(root: Path) -> None:
    if not root.exists():
        return
    paths = [root, *root.rglob("*")]
    for p in paths:
        try:
            if p.is_symlink():
                continue
            mode = p.stat().st_mode
            if p.is_dir():
                p.chmod(mode | 0o777)
            elif p.is_file():
                p.chmod(mode | 0o666)
        except Exception as exc:
            print(f"Warning: failed to update permissions for {p}: {exc}")


def _normalize_artifact_ttl_channel(channel: int) -> int:
    ch = int(channel)
    if ch == 0:
        return 0
    if 1 <= ch <= 16:
        return ch - 1
    if 0 <= ch < 16:
        return ch
    raise ValueError(
        "artifact_TTL_channel must be within digital bit range [0, 15] "
        "(or [1, 16] for 1-based input)."
    )


def _save_artifact_events_mat(
    *,
    output_path: Path,
    struct_name: str,
    timestamps_sec: np.ndarray,
    peaks_sec: np.ndarray,
    duration_sec: np.ndarray,
    extra_fields: dict[str, np.ndarray | float | int | str] | None = None,
) -> Path:
    payload: dict[str, np.ndarray | float | int | str] = {
        "timestamps": np.asarray(timestamps_sec, dtype=np.float64).reshape(-1, 2),
        "peaks": np.asarray(peaks_sec, dtype=np.float64).reshape(-1, 1),
        "duration": np.asarray(duration_sec, dtype=np.float64).reshape(-1, 1),
    }
    if extra_fields:
        payload.update(extra_fields)
    savemat(output_path, {struct_name: payload}, do_compression=True)
    return output_path


def _artifact_windows_from_peaks(
    peaks_sec: np.ndarray,
    *,
    ms_before: float,
    ms_after: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peaks = np.asarray(peaks_sec, dtype=np.float64).reshape(-1)
    if peaks.size == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    before_s = float(ms_before) / 1000.0
    after_s = float(ms_after) / 1000.0
    starts = np.maximum(peaks - before_s, 0.0)
    ends = peaks + after_s
    timestamps = np.column_stack((starts, ends))
    duration = ends - starts
    return timestamps, peaks, duration


def run_preprocess_session(config: PreprocessConfig) -> PreprocessResult:
    step_idx = 0
    total_steps = 17

    def _step(label: str) -> None:
        nonlocal step_idx
        step_idx += 1
        print(f"[Step {step_idx}/{total_steps}] {label}")

    _step("Make session metafile context")
    basepath, basename = resolve_basepath_and_basename(config.basepath)
    output_dir = resolve_local_output_dir(basepath, basename, config)

    xml_path = ensure_xml(basepath, output_dir, basename)
    rhd_path = ensure_rhd(basepath, output_dir, basename)
    xml_meta = load_xml_metadata(xml_path)
    session_xml_meta = load_session_xml_metadata(xml_path)
    intan_header = None
    if rhd_path is not None and rhd_path.exists():
        try:
            intan_header = read_intan_rhd_header(rhd_path)
        except Exception as exc:
            print(f"Warning: failed to parse info.rhd ({rhd_path}): {exc}")

    # Neurocode-compatible behavior: use XML-derived amplifier metadata.
    effective_sr = float(xml_meta.sr)
    effective_n_channels = int(xml_meta.n_channels)
    analog_sr = effective_sr
    digital_sr = effective_sr
    if intan_header is not None:
        analog_sr = float(intan_header.board_adc_sample_rate)
        digital_sr = float(intan_header.board_dig_in_sample_rate)

    _step("Discover input recordings")
    amplifier_paths = discover_subsessions(
        basepath=basepath,
        sort_files=config.sort_files,
        alt_sort=config.alt_sort,
        ignore_folders=config.ignore_folders,
    )
    if not amplifier_paths:
        raise FileNotFoundError(f"No subsession dat files found under {basepath}")

    catalog = build_acquisition_catalog(
        amplifier_paths=amplifier_paths,
        n_amplifier_channels=effective_n_channels,
        dtype=config.dtype,
        intan_header=intan_header,
    )
    print_catalog_summary(catalog)

    _step("Build merge points")
    merge_data = compute_mergepoints(
        dat_paths=catalog.amplifier_paths,
        n_channels=effective_n_channels,
        dtype=config.dtype,
        sampling_frequency=effective_sr,
        foldernames=catalog.subsession_names,
    )
    mergepoints_path = output_dir / f"{basename}.MergePoints.events.mat"
    save_mergepoints_events_mat(mergepoints_path, merge_data)

    ttl_channel_0based: int | None = None
    if config.remove_artifact_TTL:
        if config.artifact_TTL_channel is None:
            raise ValueError("remove_artifact_TTL=True requires artifact_TTL_channel to be specified.")
        ttl_channel_0based = _normalize_artifact_ttl_channel(config.artifact_TTL_channel)

    # Build sidecar dat/events early so TTL artifact removal can reuse digitalIn.events.mat
    # instead of re-parsing digitalin binary separately.
    _step("Prepare sidecar dat files")
    intermediate_dat_paths: dict[str, Path] = {}
    need_sidecar_for_events = bool(config.analog_inputs or config.digital_inputs or config.remove_artifact_TTL)
    if config.export_intermediate_dat or need_sidecar_for_events:
        analog_ch = (
            int(catalog.board_adc_channels)
            if int(catalog.board_adc_channels) > 0
            else (len(config.analog_channels) if config.analog_channels else 1)
        )
        digital_word_ch = (
            int(catalog.board_digital_word_channels)
            if int(catalog.board_digital_word_channels) > 0
            else 1
        )

        analog_concat = write_concatenated_dat_analogin(
            dat_paths=catalog.analogin_paths,
            output_dat_path=output_dir / "analogin.dat",
            sampling_frequency=analog_sr,
            num_channels=analog_ch,
            overwrite=config.overwrite,
            job_kwargs=config.job_kwargs,
        )
        if analog_concat is not None:
            intermediate_dat_paths["analogin"] = analog_concat

        digital_concat = write_concatenated_dat_digitalin(
            dat_paths=catalog.digitalin_paths,
            output_dat_path=output_dir / "digitalin.dat",
            sampling_frequency=digital_sr,
            num_channels=digital_word_ch,
            overwrite=config.overwrite,
            job_kwargs=config.job_kwargs,
        )
        if digital_concat is not None:
            intermediate_dat_paths["digitalin"] = digital_concat

    if config.export_intermediate_dat:
        sidecar_paths = materialize_intermediate_dat(
            output_dir=output_dir,
            basename=basename,
            analogin_paths=[],
            digitalin_paths=[],
            auxiliary_paths=catalog.auxiliary_paths,
            supply_paths=catalog.supply_paths,
            time_paths=catalog.time_paths,
            sample_counts=catalog.sample_counts,
            overwrite=config.overwrite,
        )
        intermediate_dat_paths.update(sidecar_paths)

    analog_num_channels = (
        int(catalog.board_adc_channels)
        if int(catalog.board_adc_channels) > 0
        else (len(config.analog_channels) if config.analog_channels else 1)
    )
    digital_word_channels = (
        int(catalog.board_digital_word_channels)
        if int(catalog.board_digital_word_channels) > 0
        else 1
    )
    digital_inputs_for_export = bool(config.digital_inputs or config.remove_artifact_TTL)
    digital_channels_for_export = list(config.digital_channels) if config.digital_channels else None
    if ttl_channel_0based is not None:
        ttl_ch_1based = int(ttl_channel_0based) + 1
        if digital_channels_for_export is None:
            digital_channels_for_export = [ttl_ch_1based]
        elif ttl_ch_1based not in digital_channels_for_export:
            digital_channels_for_export = [*digital_channels_for_export, ttl_ch_1based]

    _step("Export analog/digital events")
    analog_event_paths, digital_event_paths = export_analog_digital_events(
        output_dir=output_dir,
        basename=basename,
        analog_inputs=config.analog_inputs,
        analog_channels=config.analog_channels,
        analog_num_channels=analog_num_channels,
        analog_active_channels_1based=(
            [int(ch) + 1 for ch in catalog.board_adc_native_orders]
            if catalog.board_adc_native_orders
            else None
        ),
        digital_inputs=digital_inputs_for_export,
        digital_channels=digital_channels_for_export,
        digital_word_channels=digital_word_channels,
        digital_active_channels_1based=(
            [int(ch) + 1 for ch in catalog.board_digital_input_native_orders]
            if catalog.board_digital_input_native_orders
            else None
        ),
        sr=effective_sr,
        analog_sr=analog_sr,
        digital_sr=digital_sr,
        analog_dat_path=intermediate_dat_paths.get("analogin"),
        digital_dat_path=intermediate_dat_paths.get("digitalin"),
        merge_timestamps_sec=merge_data.timestamps_sec,
        overwrite=config.overwrite,
    )

    _step("Load and concatenate amplifier dat")
    recordings = load_subsession_recordings(
        dat_paths=catalog.amplifier_paths,
        sampling_frequency=effective_sr,
        num_channels=effective_n_channels,
        dtype=config.dtype,
        gain_to_uV=config.gain_to_uV,
        offset_to_uV=config.offset_to_uV,
    )
    recording_concat = concatenate_recordings_si(recordings)

    _step("Save raw dat (optional)")
    raw_dat_path: Path | None = None
    if config.save_raw:
        raw_dat_path = output_dir / f"{basename}_raw.dat"
        write_concatenated_dat(
            recording=recording_concat,
            output_dat_path=raw_dat_path,
            overwrite=config.overwrite,
            job_kwargs=config.job_kwargs,
    )

    recording_base = recording_concat

    _step("Attach probe and mark bad channels")
    recording_raw, bad_0, bad_1 = attach_probe_and_remove_bad_channels(
        recording=recording_base,
        chanmap_mat_path=config.chanmap_mat_path,
        reject_channels_0based=sorted(set(config.reject_channels + xml_meta.skipped_channels_0based)),
    )

    if hasattr(recording_raw, "get_channel_ids"):
        channel_ids_for_processing = [int(ch) for ch in recording_raw.get_channel_ids()]
    else:
        channel_ids_for_processing = list(range(effective_n_channels))

    good_channels_0based = [int(ch) for ch in channel_ids_for_processing if int(ch) not in set(bad_0)]
    recording_preprocessed = recording_raw
    artifact_ttl_timestamps_sec = np.empty(0, dtype=np.float64)
    artifact_high_timestamps_sec = np.empty(0, dtype=np.float64)
    artifact_high_group_ids = np.empty(0, dtype=np.int64)
    _step("Run CMR preprocess (optional)")
    if config.do_preprocess:
        if hasattr(recording_raw, "get_channel_ids"):
            recording_preprocessed = preprocess_selected_channels_preserve_shape(
                recording_raw=recording_raw,
                selected_channel_ids=good_channels_0based,
                bandpass_min_hz=config.bandpass_min_hz,
                bandpass_max_hz=config.bandpass_max_hz,
                reference=config.reference,
                local_radius_um=config.local_radius_um,
            )
        else:
            recording_preprocessed = apply_preprocessing(
                recording_raw=recording_raw,
                bandpass_min_hz=config.bandpass_min_hz,
                bandpass_max_hz=config.bandpass_max_hz,
                reference=config.reference,
                local_radius_um=config.local_radius_um,
            )
    if (config.remove_artifact_TTL or config.remove_highamp_artifact) and not config.do_preprocess:
        raise ValueError(
            "Artifact removal requires CMR output. Set do_preprocess=True when "
            "remove_artifact_TTL or remove_highamp_artifact is enabled."
        )
    _step("Run TTL artifact removal (optional)")
    if config.remove_artifact_TTL:
        existing_ttl_events_path = output_dir / f"{basename}.artifactTTL.events.mat"
        if existing_ttl_events_path.exists() and not config.overwrite:
            print(f"Skipping TTL artifact removal: existing file found ({existing_ttl_events_path})")
            intermediate_dat_paths["artifact_ttl_events"] = existing_ttl_events_path
        else:
            ttl_events_path = next((p for p in digital_event_paths if "digitalIn.events.mat" in p.name), None)
            if ttl_events_path is None:
                raise ValueError(
                    "remove_artifact_TTL=True but digitalIn.events.mat was not generated. "
                    "Check digitalin availability and TTL channel settings."
                )
            loaded_digital = loadmat(ttl_events_path, simplify_cells=True)
            digital_in = loaded_digital.get("digitalIn")
            if not isinstance(digital_in, dict):
                raise ValueError(f"Invalid digitalIn structure in {ttl_events_path}")

            timestamps_on = digital_in.get("timestampsOn")
            if timestamps_on is None:
                raise ValueError(f"timestampsOn is missing in {ttl_events_path}")
            timestamps_off = digital_in.get("timestampsOff")
            if ttl_channel_0based is None:
                raise ValueError("Internal error: TTL channel was not initialized.")

            ttl_ch_1based = int(ttl_channel_0based) + 1
            try:
                ttl_sec_on = np.asarray(timestamps_on[ttl_ch_1based - 1], dtype=np.float64).reshape(-1)
            except Exception as exc:
                raise ValueError(
                    f"Failed to read timestampsOn for TTL channel {ttl_ch_1based} from {ttl_events_path}"
                ) from exc
            ttl_sec = ttl_sec_on[np.isfinite(ttl_sec_on)]
            source_label = "digitalIn.timestampsOn"
            if config.artifact_TTL_include_offset:
                if timestamps_off is None:
                    raise ValueError(
                        "artifact_TTL_include_offset=True but timestampsOff is missing in "
                        f"{ttl_events_path}."
                    )
                try:
                    ttl_sec_off = np.asarray(timestamps_off[ttl_ch_1based - 1], dtype=np.float64).reshape(-1)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to read timestampsOff for TTL channel {ttl_ch_1based} from {ttl_events_path}"
                    ) from exc
                ttl_sec = np.concatenate((ttl_sec, ttl_sec_off[np.isfinite(ttl_sec_off)])).astype(
                    np.float64, copy=False
                )
                source_label = "digitalIn.timestampsOn+timestampsOff"
            if ttl_sec.size == 0:
                raise ValueError(
                    "remove_artifact_TTL=True but no TTL events were detected for "
                    f"artifact_TTL_channel={ttl_channel_0based} "
                    f"(include_offset={bool(config.artifact_TTL_include_offset)})."
                )
            artifact_ttl_timestamps_sec = np.unique(ttl_sec.astype(np.float64))
            artifact_ttl = np.unique(np.rint(ttl_sec * float(effective_sr)).astype(np.int64)).tolist()

            recording_preprocessed = apply_transform_to_selected_channels_preserve_shape(
                recording_raw=recording_preprocessed,
                selected_channel_ids=good_channels_0based,
                transform_fn=lambda rec_sel: remove_artifacts(
                    recording_in=rec_sel,
                    artifact_per_group=artifact_ttl,
                    by_group=config.artifact_TTL_by_group,
                    ms_before=config.artifact_TTL_ms_before,
                    ms_after=config.artifact_TTL_ms_after,
                    mode=config.artifact_TTL_mode,
                )[0],
            )
            ttl_timestamps, ttl_peaks, ttl_duration_sec = _artifact_windows_from_peaks(
                artifact_ttl_timestamps_sec,
                ms_before=config.artifact_TTL_ms_before,
                ms_after=config.artifact_TTL_ms_after,
            )
            ttl_events_path = _save_artifact_events_mat(
                output_path=output_dir / f"{basename}.artifactTTL.events.mat",
                struct_name="artifactTTL",
                timestamps_sec=ttl_timestamps,
                peaks_sec=ttl_peaks,
                duration_sec=ttl_duration_sec,
                extra_fields={
                    "channel": float(int(ttl_ch_1based)),
                    "source": source_label,
                },
            )
            intermediate_dat_paths["artifact_ttl_events"] = ttl_events_path

    _step("Run high-amplitude artifact removal (optional)")
    if config.remove_highamp_artifact:
        existing_high_events_path = output_dir / f"{basename}.artifactHigh.events.mat"
        if existing_high_events_path.exists() and not config.overwrite:
            print(f"Skipping high-amplitude artifact removal: existing file found ({existing_high_events_path})")
            intermediate_dat_paths["artifact_high_events"] = existing_high_events_path
        else:
            highamp_frames_by_group: dict[int, list[int]] = {}

            def _apply_highamp_artifacts(rec_sel):
                detected = detect_high_amplitude_artifacts(
                    rec_sel,
                    by_group=config.highamp_detect_by_group,
                    estimate_windows=config.highamp_estimate_windows,
                    estimate_window_s=config.highamp_estimate_window_s,
                    threshold_sigma=config.highamp_threshold_sigma,
                    seed=config.highamp_seed,
                    chunk_s=config.highamp_chunk_s,
                    dead_time_ms=config.highamp_dead_time_ms,
                    n_jobs=config.highamp_n_jobs,
                )
                for gid, frames in detected.items():
                    highamp_frames_by_group[int(gid)] = [int(x) for x in frames]
                return remove_artifacts(
                    recording_in=rec_sel,
                    artifact_per_group=detected,
                    by_group=config.highamp_remove_by_group,
                    ms_before=config.highamp_ms_before,
                    ms_after=config.highamp_ms_after,
                    mode=config.highamp_mode,
                )[0]

            recording_preprocessed = apply_transform_to_selected_channels_preserve_shape(
                recording_raw=recording_preprocessed,
                selected_channel_ids=good_channels_0based,
                transform_fn=_apply_highamp_artifacts,
            )
            if highamp_frames_by_group:
                pairs = sorted(
                    (frame, gid)
                    for gid, frames in highamp_frames_by_group.items()
                    for frame in frames
                )
                if pairs:
                    artifact_high_frames = np.asarray([p[0] for p in pairs], dtype=np.int64)
                    artifact_high_group_ids = np.asarray([p[1] for p in pairs], dtype=np.int64)
                    artifact_high_timestamps_sec = artifact_high_frames.astype(np.float64) / float(effective_sr)
            high_timestamps, high_peaks, high_duration_sec = _artifact_windows_from_peaks(
                artifact_high_timestamps_sec,
                ms_before=config.highamp_ms_before,
                ms_after=config.highamp_ms_after,
            )
            high_events_path = _save_artifact_events_mat(
                output_path=output_dir / f"{basename}.artifactHigh.events.mat",
                struct_name="artifactHigh",
                timestamps_sec=high_timestamps,
                peaks_sec=high_peaks,
                duration_sec=high_duration_sec,
                extra_fields={
                    "group_id": artifact_high_group_ids.reshape(-1, 1),
                    "source": "detect_high_amplitude_artifacts",
                },
            )
            intermediate_dat_paths["artifact_high_events"] = high_events_path

    _step("Write final dat output")
    if config.zero_bad and bad_0 and hasattr(recording_preprocessed, "get_channel_ids"):
        recording_preprocessed = zero_selected_channels_preserve_shape(
            recording_raw=recording_preprocessed,
            selected_channel_ids=bad_0,
        )
    dat_path: Path | None = output_dir / f"{basename}.dat"
    write_concatenated_dat(
        recording=recording_preprocessed,
        output_dat_path=dat_path,
        overwrite=config.overwrite,
        job_kwargs=config.job_kwargs,
    )

    _step("Write LFP output (optional)")
    lfp_path: Path | None = None
    if config.make_lfp:
        lfp_path = output_dir / f"{basename}.lfp"
        write_lfp(
            recording_raw=recording_base,
            lfp_path=lfp_path,
            lfp_fs=config.lfp_fs,
            dtype=config.dtype,
            overwrite=config.overwrite,
            job_kwargs=config.job_kwargs,
        )

    session_dat_path = output_dir / f"{basename}.dat"
    if not session_dat_path.exists():
        raise FileNotFoundError(
            "neurocode_strict session generation requires concatenated dat. "
            "Enable save_raw or sorter, or ensure basename.dat exists in output_dir."
        )

    _step("Build session.mat")
    session_struct = build_session_struct(
        source_basepath=basepath,
        local_basepath=output_dir,
        session_basepath_mode=config.session_basepath_mode,
        basename=basename,
        dat_path=session_dat_path,
        dat_dtype=config.dtype,
        sr=effective_sr,
        sr_lfp=(xml_meta.sr_lfp if xml_meta.sr_lfp is not None else config.lfp_fs),
        n_channels=effective_n_channels,
        bad_channels_1based=bad_1,
        merge_data=merge_data,
        xml_meta=session_xml_meta,
    )
    session_mat_path = output_dir / f"{basename}.session.mat"
    save_session_mat(session_mat_path, session_struct)

    def _load_pulses_struct(paths: list[Path], basename_local: str) -> dict | None:
        pulses_candidates = [
            output_dir / f"{basename_local}.pulses.events.mat",
            *paths,
        ]
        for p in pulses_candidates:
            if not p.exists():
                continue
            if "pulses.events.mat" not in p.name:
                continue
            try:
                loaded = loadmat(p, simplify_cells=True)
            except Exception:
                continue
            pulses = loaded.get("pulses")
            if isinstance(pulses, dict):
                return pulses
        return None

    state_score_paths: list[Path] = []
    state_score_figure_paths: list[Path] = []
    _step("Run state scoring (optional)")
    if config.state_score:
        pulses_struct = _load_pulses_struct(analog_event_paths, basename)
        state_score_result = run_state_scoring(
            basepath=output_dir,
            basename=basename,
            session_struct=session_struct,
            pulses=pulses_struct,
            config=config,
        )
        state_score_paths = [
            state_score_result.emg_mat_path,
            state_score_result.sleepscore_lfp_mat_path,
            state_score_result.sleep_state_mat_path,
            state_score_result.sleep_state_episodes_mat_path,
        ]
        state_score_figure_paths = list(state_score_result.figure_paths)

    _step("Run spike sorter (optional)")
    sorter_output_dir: Path | None = None
    if config.sorter:
        sorter_dat_path = output_dir / f"{basename}.dat"
        sorter_preprocess_for_sorting = False
        sorter_input_is_preprocessed = bool(config.do_preprocess)
        if config.bad_channels and bad_0 and not sorter_input_is_preprocessed:
            sorter_exclude_channels_0based = bad_0
        else:
            sorter_exclude_channels_0based = None

        sorter_label = str(config.sorter).strip()
        sorter_name_for_folder = _sorter_output_prefix(sorter_label)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        sorter_output_dir = output_dir / f"{sorter_name_for_folder}_{timestamp}"
        if sorter_output_dir.exists():
            suffix = 1
            while True:
                candidate = output_dir / f"{sorter_name_for_folder}_{timestamp}_{suffix:02d}"
                if not candidate.exists():
                    sorter_output_dir = candidate
                    break
                suffix += 1
        _ = execute_sorting_job(
            sorter=config.sorter,
            dat_path=sorter_dat_path,
            xml_path=xml_path,
            output_folder=sorter_output_dir,
            config_path=config.sorter_config_path,
            kilosort1_path=config.sorter_path,
            kilosort4_path=config.sorter_path,
            matlab_path=config.matlab_path,
            chanmap_mat_path=config.chanmap_mat_path,
            exclude_channels_0based=sorter_exclude_channels_0based,
            job_kwargs=config.job_kwargs,
            remove_existing_folder=config.overwrite,
            preprocess_for_sorting=sorter_preprocess_for_sorting,
            input_is_preprocessed=sorter_input_is_preprocessed,
            bandpass_min_hz=config.bandpass_min_hz,
            bandpass_max_hz=config.bandpass_max_hz,
            reference=config.reference,
            local_radius_um=config.local_radius_um,
            sorter_verbose=bool(config.sorter_verbose),
            cleanup_temp_wh=bool(config.cleanup_temp_wh),
        )

    _step("Save manifest and return")
    result = PreprocessResult(
        basepath=basepath,
        basename=basename,
        local_output_dir=output_dir,
        dat_path=dat_path,
        lfp_path=lfp_path,
        session_mat_path=session_mat_path,
        mergepoints_mat_path=mergepoints_path,
        analog_event_paths=analog_event_paths,
        digital_event_paths=digital_event_paths,
        intermediate_dat_paths=intermediate_dat_paths,
        n_channels=effective_n_channels,
        sr=effective_sr,
        sr_lfp=config.lfp_fs if config.make_lfp else None,
        bad_channels_0based=bad_0,
        bad_channels_1based=bad_1,
        subsession_paths=catalog.amplifier_paths,
        subsession_sample_counts=(
            merge_data.firstlasttimepoints_samples[:, 1].astype(int).tolist()
        ),
        sorter=config.sorter,
        sorter_output_dir=sorter_output_dir,
        state_score_paths=state_score_paths,
        state_score_figure_paths=state_score_figure_paths,
    )

    save_params_and_manifest(
        config=config,
        result=result,
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
    )
    _make_tree_world_rw(output_dir)
    return result
