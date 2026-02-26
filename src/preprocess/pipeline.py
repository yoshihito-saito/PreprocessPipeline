from __future__ import annotations

from datetime import datetime
from pathlib import Path

import spikeinterface.extractors as se

from .events import export_analog_digital_events, materialize_intermediate_dat
from .io import (
    build_acquisition_catalog,
    discover_subsessions,
    ensure_xml,
    load_session_xml_metadata,
    load_xml_metadata,
    print_catalog_summary,
    resolve_basepath_and_basename,
    resolve_local_output_dir,
    save_params_and_manifest,
)
from .mergepoints import compute_mergepoints, save_mergepoints_events_mat
from .recording import (
    apply_preprocessing,
    attach_probe_and_remove_bad_channels,
    concatenate_recordings_si,
    load_subsession_recordings,
    write_concatenated_dat_analogin,
    write_concatenated_dat_digitalin,
    write_concatenated_dat,
    write_lfp,
)
from .session import build_session_struct, save_session_mat
from .metafile import PreprocessConfig, PreprocessResult
from .sorter_runner import execute_sorting_job


def run_preprocess_session(config: PreprocessConfig) -> PreprocessResult:
    basepath, basename = resolve_basepath_and_basename(config.basepath)
    output_dir = resolve_local_output_dir(basepath, basename, config)

    xml_path = ensure_xml(basepath, output_dir, basename)
    xml_meta = load_xml_metadata(xml_path)
    session_xml_meta = load_session_xml_metadata(xml_path)

    # Neurocode-compatible behavior: use XML-derived amplifier metadata.
    effective_sr = float(xml_meta.sr)
    effective_n_channels = int(xml_meta.n_channels)
    analog_sr = effective_sr
    digital_sr = effective_sr

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
        intan_header=None,
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

    dat_path: Path | None = None
    if config.save_raw:
        dat_path = output_dir / f"{basename}.dat"
        write_concatenated_dat(
            recording=recording_concat,
            output_dat_path=dat_path,
            overwrite=config.overwrite,
            job_kwargs=config.job_kwargs,
        )

    if config.sorter and dat_path is None:
        dat_path = output_dir / f"{basename}.dat"
        write_concatenated_dat(
            recording=recording_concat,
            output_dat_path=dat_path,
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

    if dat_path is not None:
        recording_base = se.read_binary(
            str(dat_path),
            sampling_frequency=effective_sr,
            dtype=config.dtype,
            num_channels=effective_n_channels,
            gain_to_uV=config.gain_to_uV,
            offset_to_uV=config.offset_to_uV,
        )
    else:
        recording_base = recording_concat

    recording_raw, bad_0, bad_1 = attach_probe_and_remove_bad_channels(
        recording=recording_base,
        chanmap_mat_path=config.chanmap_mat_path,
        reject_channels_0based=sorted(set(config.reject_channels + xml_meta.skipped_channels_0based)),
    )

    if config.do_preprocess:
        _ = apply_preprocessing(
            recording_raw=recording_raw,
            bandpass_min_hz=config.bandpass_min_hz,
            bandpass_max_hz=config.bandpass_max_hz,
            reference=config.reference,
            local_radius_um=config.local_radius_um,
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
        digital_inputs=config.digital_inputs,
        digital_channels=config.digital_channels,
        digital_word_channels=digital_word_channels,
        sr=effective_sr,
        analog_sr=analog_sr,
        digital_sr=digital_sr,
        analog_dat_path=intermediate_dat_paths.get("analogin"),
        digital_dat_path=intermediate_dat_paths.get("digitalin"),
        merge_timestamps_sec=merge_data.timestamps_sec,
        overwrite=config.overwrite,
    )

    session_dat_path = dat_path if dat_path is not None else output_dir / f"{basename}.dat"
    if not session_dat_path.exists():
        raise FileNotFoundError(
            "neurocode_strict session generation requires concatenated dat. "
            "Enable save_raw or sorter, or ensure basename.dat exists in output_dir."
        )
    if dat_path is None:
        dat_path = session_dat_path

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

    sorter_output_dir: Path | None = None
    if config.sorter:
        sorter_label = str(config.sorter).strip()
        sorter_name_for_folder = sorter_label[:1].upper() + sorter_label[1:].lower()
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
            dat_path=dat_path if dat_path is not None else (output_dir / f"{basename}.dat"),
            xml_path=xml_path,
            output_folder=sorter_output_dir,
            config_path=config.sorter_config_path,
            kilosort1_path=config.sorter_path,
            matlab_path=config.matlab_path,
            chanmap_mat_path=config.chanmap_mat_path,
            exclude_channels_0based=bad_0,
            job_kwargs=config.job_kwargs,
            remove_existing_folder=config.overwrite,
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
    )

    save_params_and_manifest(
        config=config,
        result=result,
        output_dir=output_dir,
        script_path=Path(__file__).resolve(),
    )
    return result
