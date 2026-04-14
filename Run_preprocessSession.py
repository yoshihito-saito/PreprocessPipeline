import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Run Preprocess Session

    This notebook runs the preprocessing pipeline from `src.preprocess`.

    Workflow:
    1. Select a remote session folder with a GUI dialog.
    2. Prepare local output and copy `basename.xml`.
    3. Build `chanMap.mat` from XML with probe assignments.
    4. Run `run_preprocess_session(config)` (preprocess + sorting when `sorter` is set).
    5. Sorting output folder is automatically named as `SorterName_YYYY-MM-DD_HHMMSS` (e.g. `Kilosort_2026-02-16_120658`).
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %reload_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    from pathlib import Path

    from src.preprocess import (
        PreprocessConfig,
        prepare_chanmap,
        run_preprocess_session,
        select_paths_with_gui,
        show_chanmap,
    )
    from src.preprocess.io import copy_results_to_basepath

    return (
        Path,
        PreprocessConfig,
        prepare_chanmap,
        run_preprocess_session,
        select_paths_with_gui,
        show_chanmap,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1) Select Basepath And Prepare XML (GUI)
    """)
    return


@app.cell
def _(Path, select_paths_with_gui):
    # Choose one mode:
    # 1) GUI mode: use_gui=True
    # 2) Manual mode: use_gui=False and set manual_basepath
    # use_gui = True
    # manual_basepath = None  # e.g. r'T:\\data\\AutoMaze\\RM010\\RM010_day43_20250925'

    # basepath, basename, local_output_dir, xml_path = select_paths_with_gui(
    #     use_gui=use_gui,
    #     manual_basepath=manual_basepath,
    #     initial_drive=r'S:\\',
    #     local_root=Path.cwd() / 'sorting_temp',
    # )

    # print('basepath      :', basepath)
    # print('basename      :', basename)
    # print('local_output  :', local_output_dir)
    # print('xml_path      :', xml_path)


    use_gui = False
    manual_basepath = "/local/workdir/ys2375/data/ayadataB3/data/AutoMaze/Wallace/260408_Wallace_day025/"
    basepath, basename, local_output_dir, xml_path = select_paths_with_gui(
        use_gui=use_gui,
        manual_basepath=manual_basepath,
        initial_drive=r'S:\\',
        local_root=Path.cwd() / 'sorting_temp',
    )

    print('basepath      :', basepath)
    print('basename      :', basename)
    print('local_output  :', local_output_dir)
    print('xml_path      :', xml_path)
    return basename, basepath, local_output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2) Define Probe Assignments

    `groups` are XML group indices (0-based).
    `type` can be: `double_sided`, `staggered`, `poly3`, `poly5`, `NeuroPixel`, `neurogrid`.
    """)
    return


@app.cell
def _():
    # # Single double-sided probe:
    # probe_assignments = [
    #     {'type': 'double_sided', 'groups': [0, 1, 2, 3, 4, 5, 6, 7], 'x_offset': 0},
    # ]

    # probe_assignments

    ## for multiple double-sided probes example:
    # probe_assignments = [
    #     {'type': 'double_sided', 'groups': [0, 1, 2, 3, 4, 5, 6, 7], 'x_offset': 0},
    #     {'type': 'double_sided', 'groups': [8, 9, 10, 11, 12, 13, 14, 15], 'x_offset': 1200},
    # ]

    ## For flex probe
    probe_assignments = [
        {'type': 'staggered', 'groups': [0, 1, 2, 3, 4, 5, 6, 7], 'x_offset': 0},
    ]

    probe_assignments
    return (probe_assignments,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3) Generate `chanMap.mat`
    """)
    return


@app.cell
def _(
    basename,
    basepath,
    local_output_dir,
    prepare_chanmap,
    probe_assignments,
    show_chanmap,
):
    chanmap_path, bad_ch_ids = prepare_chanmap(
        basepath=basepath,
        basename=basename,
        local_output_dir=local_output_dir,
        probe_assignments=probe_assignments,
        reject_channels=[],
    )

    print('chanMap path           :', chanmap_path)
    print('bad channels (0-based) :', bad_ch_ids)

    # Visualize generated channel map (same style as Spike_sorting_KS1.ipynb)
    bad_ch_ids_plot = show_chanmap(chanmap_path)
    print('bad channels from plot :', bad_ch_ids_plot)
    return (chanmap_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4) Build Preprocess Config
    """)
    return


@app.cell
def _(Path, PreprocessConfig, basepath, chanmap_path, local_output_dir):
    pre_config = PreprocessConfig(
        basepath=basepath,                         # Session source folder
        localpath=local_output_dir.parent,         # Parent directory for local output
        alt_sort=None,                             # Optional custom sort order (0-based)
        ignore_folders=[],                         # Folder names to skip
        save_raw=False,                             # save unprocessed basename_raw.dat

        analog_inputs=False,                       # Export/process analog inputs
        digital_inputs=True,                       # Export/process digital inputs

        do_preprocess=True,                        # Run bandpass + CMR
        bandpass_min_hz=500.0,                     # Bandpass low cutoff (Hz)
        bandpass_max_hz=8000.0,                    # Bandpass high cutoff (Hz)
        reference='local',                         # CMR mode: 'local' or 'global'
        local_radius_um=(20.0, 200.0),             # Local reference neighborhood (um)

        # ----- Artifact removal (after CMR) -----
        # artifact removal from  TTL events (e.g. opto stim)
        artifact_ttl_group_mode='none',            # Artifact mode: none/all/probe/shank
        artifact_TTL_channel=0,                    # TTL channel (0-based)
        artifact_TTL_include_offset=True,          # Include offset of TTL event
        artifact_TTL_ms_before=0.5,                # Window before TTL (ms)
        artifact_TTL_ms_after=2.0,                 # Window after TTL (ms)
        artifact_TTL_mode="linear",                 # Interpolation mode: cubic/linear/0
        # artifact removal after detecting high-amplitude noise events
        artifact_highamp_group_mode='shank',       # Artifact mode: none/all/probe/shank
        highamp_estimate_windows=500,               # Random windows for noise estimation
        highamp_estimate_window_s=1.0,             # Length of each estimation window (s)
        highamp_threshold_sigma=5.0,               # Detection threshold in sigma units
        highamp_seed=0,                            # RNG seed for reproducible detection
        highamp_chunk_s=1.0,                       # Scan chunk duration (s)
        highamp_dead_time_ms=1.0,                  # Refractory merge window between detections (ms)
        highamp_n_jobs=256,                         # Parallel workers for detection
        highamp_ms_before=2.0,                     # Removal window before trigger (ms)
        highamp_ms_after=2.0,                      # Removal window after trigger (ms)
        highamp_mode="linear",                      # Interpolation mode: cubic/linear/0
        # ----------------------------------------

        make_lfp=True,                             # Export LFP binary
        lfp_fs=1250,                               # LFP sampling rate (Hz)

        state_score=True,                          # Run sleep/state scoring
        sw_channels=None,                          # Optional slow-wave channels (0-based)
        theta_channels=None,                       # Optional theta channels (0-based)
        state_block_wake_to_rem=False,             # Block direct wake->REM transitions
        state_min_state_length=6.0,                # Minimum state duration (s)
        emg_th_alpha=1,                            # EMG threshold alpha (higher=stricter)


        chanmap_mat_path=chanmap_path,             # chanMap .mat for geometry + bad channels
        reject_channels=[],                        # Extra manual reject channels (0-based)

        export_intermediate_dat=True,              # Save sidecar dat files (analogin/digitalin/etc)

        matlab_path=Path('/local/workdir/ys2375/MATLAB/R2024b/bin/matlab'),  # MATLAB executable
        matlab_max_workers=256,  # Max parallel workers for MATLAB-based steps (e.g. Kilosort)

        # Sorter settings
        sorter='Kilosort',
        sorter_path=Path('sorter') / 'KiloSort1',
        sorter_config_path=Path('sorter') / 'Kilosort1_config.yaml',

        # sorter='Kilosort2_5',
        # sorter_path=Path('sorter') / 'Kilosort2.5',
        # sorter_config_path=Path('sorter') / 'Kilosort2.5_config.yaml',
    
        # sorter='kilosort4',
        # sorter_path=Path("sorter") / "Kilosort4",                  
        # sorter_config_path=Path("sorter") / "Kilosort4_config.yaml",


        overwrite=False,                           # Overwrite existing outputs
        save_params_json=True,                     # Save preprocess params JSON
        save_manifest_json=True,                   # Save output manifest JSON
        save_log_mat=True,                         # Save MATLAB-style log mat

        job_kwargs={
            "pool_engine": "process",              # Parallel backend
            "n_jobs": 384,                          # Worker count
            "chunk_duration": "1s",
            "progress_bar": True,                  # Show progress bar
            "max_threads_per_worker": 1,           # Limit threads per worker
        },
    )

    pre_config
    return (pre_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5) Run Preprocess
    """)
    return


@app.cell
def _(pre_config, run_preprocess_session):
    from dataclasses import replace
    from src import use_existing_sorting, make_post_recording, run_postprocess_session


    do_sorting = True
    existing_sorting_dir = None

    runtime_config = pre_config if do_sorting else replace(pre_config, sorter=None)
    result = run_preprocess_session(runtime_config)
    if not do_sorting:
        result = use_existing_sorting(
            result,
            sorter=pre_config.sorter,
            existing_sorting_dir=existing_sorting_dir,
        )
        recording_for_post = make_post_recording(result, pre_config)
    else:
        recording_for_post = None

    result
    return recording_for_post, result, run_postprocess_session


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6) Quick Check
    """)
    return


@app.cell
def _(result):
    print('local_output_dir      :', result.local_output_dir)
    print('dat_path              :', result.dat_path)  # preprocessed basename.dat
    print('lfp_path              :', result.lfp_path)
    print('session_mat_path      :', result.session_mat_path)
    print('mergepoints_mat_path  :', result.mergepoints_mat_path)
    print('intermediate_dat_paths:', result.intermediate_dat_paths)
    print('n_channels            :', result.n_channels)
    print('sr                    :', result.sr)
    print('sr_lfp                :', result.sr_lfp)
    print('bad_channels_0based   :', result.bad_channels_0based)
    print('sorter                :', result.sorter)
    print('sorter_output_dir     :', result.sorter_output_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7) Run Postprocess
    Runs postprocessing pipeline and exports into `<KilosortFolder>_spi`.
    """)
    return


@app.cell
def _(pre_config, recording_for_post, result):
    from src import PostprocessConfig

    post_cfg = PostprocessConfig(
        sorting_phy_folder=None,
        sorting_search_root=result.local_output_dir,

        recording=recording_for_post,
        dat_path=result.dat_path,
        sampling_frequency=result.sr,
        num_channels=result.n_channels,
        dtype=pre_config.dtype,
        gain_to_uV=pre_config.gain_to_uV,
        offset_to_uV=pre_config.offset_to_uV,
        chanmap_mat_path=pre_config.chanmap_mat_path,
        reject_channels=result.bad_channels_0based,

        apply_preprocess=False,
        bandpass_min_hz=pre_config.bandpass_min_hz,
        bandpass_max_hz=pre_config.bandpass_max_hz,
        reference=pre_config.reference,
        local_radius_um=pre_config.local_radius_um,

        exclude_cluster_groups=["noise"],
        duplicate_censored_period_ms=0.5,
        duplicate_threshold=0.5,
        remove_strategy="max_spikes",

        analyzer_format="binary_folder",
        analyzer_cache_dir=None,
        delete_analyzer_cache=True,
        skip_curation=False,

        merge_min_spikes=100,
        merge_corr_diff_thresh=0.25,
        merge_template_diff_thresh=0.25,
        merge_sparsity_overlap=0.5,
        merge_censor_ms=0.5,

        split_contamination=0.05,
        split_threshold_mode="adaptive_chi2",
        split_min_clean_frac=0.9,
        split_relax_factor=0.5,
        split_use_waveform_gate=True,
        split_wf_threshold=0.2,
        split_wf_template_max=1000,
        split_wf_n_chans=10,
        split_wf_center="demean",
        split_amp_mad_scale=10.0,
        split_squeeze_all_outlier_to_new=True,
        split_min_spikes=10,
        split_verbose=True,

        metric_names=["firing_rate", "isi_violation", "presence_ratio", "snr", "amplitude_median"],
        template_metric_names=["peak_to_valley", "peak_trough_ratio", "half_width", "repolarization_slope", "recovery_slope"],
        noise_thresholds={
            "isi_violations_ratio_gt": 5.0,
            "isi_violations_count_gt": 50.0,
            "presence_ratio_lt": 0.1,
            "snr_lt": 2.0,
            "amplitude_median_lt": 15.0,
            "amplitude_median_gt": 500.0,
            "peak_to_valley_gt": 0.85,
            "peak_trough_ratio_lt": -0.5,
            "halfwidth_gt": 0.4,
            "slope_lt": 15.0,
            "firing_rate_lt": 0.01,
        },
        skip_pc_metrics=True,

        overwrite=False,
        copy_binary=False,
        use_relative_path=True,
    )
    return (post_cfg,)


@app.cell
def _(post_cfg, run_postprocess_session):
    post_results = run_postprocess_session(post_cfg)
    post_result = post_results[0]  
    len(post_results), [r.output_folder for r in post_results]
    return


@app.cell
def _():
    # copy_results = True
    # delete_local_after_copy = False

    # if copy_results:
    #     copied_to = copy_results_to_basepath(
    #         local_output_dir=result.local_output_dir,
    #         basepath=result.basepath,
    #         delete_local=delete_local_after_copy,
    #     )
    #     print('copied_to             :', copied_to)
    #     print('delete_local_after_copy:', delete_local_after_copy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # For testing diferent threhsolds
    """)
    return


@app.cell
def _():
    # from pathlib import Path
    # import pandas as pd
    # from src.postprocess.unit_classify import mark_noise_clusters_from_metrics

    # phy_dir = Path("/local/workdir/ys2375/PreprocessPipeline/sorting_temp/sake_day24/Kilosort_2026-03-02_093059_spi")
    # metrics_df = pd.read_csv(phy_dir / "quality_metrics.csv", index_col=0)

    # noise_thresholds = {
    #     "isi_violations_ratio_gt": 5.0,
    #     "isi_violations_count_gt": 50.0,
    #     "presence_ratio_lt": 0.1,
    #     "snr_lt": 2.0,
    #     "amplitude_median_lt": 15.0,
    #     "amplitude_median_gt": 500.0,
    #     "peak_to_valley_gt": 0.85,
    #     "peak_trough_ratio_lt": -0.5,
    #     "halfwidth_gt": 0.4,
    #     "slope_lt": 15.0,
    #     "firing_rate_lt": 0.01,
    # }

    # mark_noise_clusters_from_metrics(
    #     phy_dir=phy_dir,
    #     metrics_df=metrics_df,
    #     thresholds=noise_thresholds,
    #     backup=False,
    #     reset_to_unsorted=True,
    #     update_cluster_info=True,
    # )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # For testing Job kwargs tuning
    """)
    return


@app.cell
def _():
    # from time import perf_counter
    # from pathlib import Path
    # import os

    # import numpy as np
    # import spikeinterface as si

    # from src.preprocess.metafile import PreprocessConfig
    # from src.preprocess.io import (
    #     resolve_basepath_and_basename,
    #     resolve_local_output_dir,
    #     ensure_xml,
    #     ensure_rhd,
    #     load_xml_metadata,
    #     discover_subsessions,
    #     build_acquisition_catalog,
    # )
    # from src.preprocess.intan_rhd import read_intan_rhd_header
    # from src.preprocess.recording import (
    #     load_subsession_recordings,
    #     concatenate_recordings_si,
    #     attach_probe_and_remove_bad_channels,
    #     preprocess_selected_channels_preserve_shape,
    # )


    # def build_recording_for_dat_write_benchmark(config: PreprocessConfig):
    #     basepath, basename = resolve_basepath_and_basename(config.basepath)
    #     output_dir = resolve_local_output_dir(basepath, basename, config)

    #     xml_path = ensure_xml(basepath, output_dir, basename)
    #     rhd_path = ensure_rhd(
    #         basepath,
    #         output_dir,
    #         basename,
    #         use_first_child_match=bool(config.rhd_use_first_child_match),
    #     )
    #     xml_meta = load_xml_metadata(xml_path)

    #     intan_header = None
    #     if rhd_path is not None and Path(rhd_path).exists():
    #         intan_header = read_intan_rhd_header(rhd_path)

    #     subsession_paths = discover_subsessions(
    #         basepath=basepath,
    #         sort_files=config.sort_files,
    #         alt_sort=config.alt_sort,
    #         ignore_folders=config.ignore_folders,
    #     )
    #     if not subsession_paths:
    #         raise RuntimeError(f"No subsessions found under {basepath}")

    #     catalog = build_acquisition_catalog(
    #         amplifier_paths=subsession_paths,
    #         n_amplifier_channels=int(xml_meta.n_channels),
    #         dtype=config.dtype,
    #         intan_header=intan_header,
    #     )

    #     recordings = load_subsession_recordings(
    #         dat_paths=catalog.amplifier_paths,
    #         sampling_frequency=float(xml_meta.sr),
    #         num_channels=int(xml_meta.n_channels),
    #         dtype=config.dtype,
    #         gain_to_uV=config.gain_to_uV,
    #         offset_to_uV=config.offset_to_uV,
    #         recording_paths=catalog.recording_paths,
    #         recording_stream_names=catalog.recording_stream_names,
    #     )
    #     recording_concat = concatenate_recordings_si(recordings)

    #     recording_raw, bad_0, _ = attach_probe_and_remove_bad_channels(
    #         recording=recording_concat,
    #         chanmap_mat_path=config.chanmap_mat_path,
    #         reject_channels_0based=sorted(set(config.reject_channels + xml_meta.skipped_channels_0based)),
    #     )

    #     all_channel_ids = [int(ch) for ch in recording_raw.get_channel_ids()]
    #     good_channels = [ch for ch in all_channel_ids if ch not in set(bad_0)]

    #     if config.do_preprocess:
    #         recording_preprocessed = preprocess_selected_channels_preserve_shape(
    #             recording_raw=recording_raw,
    #             selected_channel_ids=good_channels,
    #             bandpass_min_hz=config.bandpass_min_hz,
    #             bandpass_max_hz=config.bandpass_max_hz,
    #             reference=config.reference,
    #             local_radius_um=config.local_radius_um,
    #         )
    #     else:
    #         recording_preprocessed = recording_raw

    #     return recording_preprocessed


    # def benchmark_dat_write(
    #     config: PreprocessConfig,
    #     candidates,
    #     benchmark_seconds=120.0,
    #     cleanup=True,
    # ):
    #     rec = build_recording_for_dat_write_benchmark(config)
    #     sf = rec.get_sampling_frequency()
    #     n_frames = min(rec.get_total_samples(), int(benchmark_seconds * sf))
    #     rec_test = rec.frame_slice(start_frame=0, end_frame=n_frames)

    #     n_channels = rec_test.get_num_channels()
    #     dtype = np.dtype(config.dtype)
    #     bytes_to_write = n_frames * n_channels * dtype.itemsize

    #     bench_root = Path(config.localpath) / "_dat_write_bench"
    #     bench_root.mkdir(parents=True, exist_ok=True)

    #     print(f"Benchmark duration: {n_frames / sf:.1f} s")
    #     print(f"Channels: {n_channels}")
    #     print(f"Estimated bytes: {bytes_to_write / 1024**3:.2f} GiB")
    #     print(f"Benchmark output dir: {bench_root}")

    #     results = []
    #     for idx, job_kwargs in enumerate(candidates, start=1):
    #         print(f"\n=== Candidate {idx} ===")
    #         print(job_kwargs)

    #         out = bench_root / f"bench_{idx}.dat"
    #         if out.exists():
    #             out.unlink()

    #         t0 = perf_counter()
    #         si.write_binary_recording(
    #             rec_test,
    #             file_paths=str(out),
    #             add_file_extension=False,
    #             dtype=config.dtype,
    #             verbose=True,
    #             **job_kwargs,
    #         )
    #         dt = perf_counter() - t0
    #         actual_size = out.stat().st_size if out.exists() else 0
    #         mbps = actual_size / dt / 1024**2 if dt > 0 else float("nan")

    #         row = {
    #             "job_kwargs": job_kwargs,
    #             "elapsed_s": dt,
    #             "size_gib": actual_size / 1024**3,
    #             "throughput_mib_s": mbps,
    #             "sec_per_input_second": dt / (n_frames / sf),
    #             "output_path": str(out),
    #         }
    #         results.append(row)
    #         print(row)

    #         if cleanup and out.exists():
    #             out.unlink()

    #     return results
    return


@app.cell
def _():
    # candidates = [
    #     # {"pool_engine": "process", "n_jobs": 256, "chunk_duration": "0.1s", "progress_bar": True, "max_threads_per_worker": 1},
    #     {"pool_engine": "process", "n_jobs": 384, "chunk_duration": "0.1s", "progress_bar": True, "max_threads_per_worker": 1},
    #     # {"pool_engine": "process", "n_jobs": 256, "chunk_duration": "1s", "progress_bar": True, "max_threads_per_worker": 1},
    #     # {"pool_engine": "process", "n_jobs": 384, "chunk_duration": "1s", "progress_bar": True, "max_threads_per_worker": 1},
    # ]

    # bench = benchmark_dat_write(
    #     pre_config,
    #     candidates=candidates,
    #     benchmark_seconds=3600.0,
    # )
    # bench
    return


if __name__ == "__main__":
    app.run()


