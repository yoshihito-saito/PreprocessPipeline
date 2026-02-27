from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scipy.io import loadmat

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
    attach_probe_and_remove_bad_channels,
    concatenate_recordings_si,
    load_subsession_recordings,
    preprocess_selected_channels_preserve_shape,
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


def run_preprocess_session(config: PreprocessConfig) -> PreprocessResult:
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

    recordings = load_subsession_recordings(
        dat_paths=catalog.amplifier_paths,
        sampling_frequency=effective_sr,
        num_channels=effective_n_channels,
        dtype=config.dtype,
        gain_to_uV=config.gain_to_uV,
        offset_to_uV=config.offset_to_uV,
    )
    recording_concat = concatenate_recordings_si(recordings)

    raw_dat_path: Path | None = None
    if config.save_raw:
        raw_dat_path = output_dir / f"{basename}_raw.dat"
        write_concatenated_dat(
            recording=recording_concat,
            output_dat_path=raw_dat_path,
            overwrite=config.overwrite,
            job_kwargs=config.job_kwargs,
        )

    merge_data = compute_mergepoints(
        dat_paths=catalog.amplifier_paths,
        n_channels=effective_n_channels,
        dtype=config.dtype,
        sampling_frequency=effective_sr,
        foldernames=catalog.subsession_names,
    )
    mergepoints_path = output_dir / f"{basename}.MergePoints.events.mat"
    save_mergepoints_events_mat(mergepoints_path, merge_data)

    recording_base = recording_concat

    recording_raw, bad_0, bad_1 = attach_probe_and_remove_bad_channels(
        recording=recording_base,
        chanmap_mat_path=config.chanmap_mat_path,
        reject_channels_0based=sorted(set(config.reject_channels + xml_meta.skipped_channels_0based)),
    )

    good_channels_0based = [
        int(ch)
        for ch in recording_raw.get_channel_ids()
        if int(ch) not in set(bad_0)
    ]
    recording_preprocessed = recording_raw
    if config.do_preprocess:
        recording_preprocessed = preprocess_selected_channels_preserve_shape(
            recording_raw=recording_raw,
            selected_channel_ids=good_channels_0based,
            bandpass_min_hz=config.bandpass_min_hz,
            bandpass_max_hz=config.bandpass_max_hz,
            reference=config.reference,
            local_radius_um=config.local_radius_um,
        )
    dat_path: Path | None = output_dir / f"{basename}.dat"
    write_concatenated_dat(
        recording=recording_preprocessed,
        output_dat_path=dat_path,
        overwrite=config.overwrite,
        job_kwargs=config.job_kwargs,
    )

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

    intermediate_dat_paths: dict[str, Path] = {}
    if config.export_intermediate_dat:
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
        digital_inputs=config.digital_inputs,
        digital_channels=config.digital_channels,
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

    session_dat_path = output_dir / f"{basename}.dat"
    if not session_dat_path.exists():
        raise FileNotFoundError(
            "neurocode_strict session generation requires concatenated dat. "
            "Enable save_raw or sorter, or ensure basename.dat exists in output_dir."
        )

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

    sorter_output_dir: Path | None = None
    if config.sorter:
        sorter_dat_path = output_dir / f"{basename}.dat"
        sorter_preprocess_for_sorting = False
        sorter_input_is_preprocessed = bool(config.do_preprocess)
        sorter_exclude_channels_0based = None if sorter_input_is_preprocessed else bad_0

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
        )

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
    return result
