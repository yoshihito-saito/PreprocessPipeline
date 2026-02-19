from __future__ import annotations

from dataclasses import replace
import gc
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import spikeinterface as si
import spikeinterface.curation as scur
import spikeinterface.extractors as se
import spikeinterface.qualitymetrics as sqm
from spikeinterface.exporters import export_to_phy

from utils.unit_classify import mark_noise_clusters_from_metrics
from utils.unit_split import autosplit_outliers_pca

from ..preprocess.metafile import PreprocessConfig, PreprocessResult
from ..preprocess.recording import (
    apply_preprocessing,
    attach_probe_and_remove_bad_channels,
    write_concatenated_dat,
)
from .metafile import PostprocessConfig, PostprocessResult


def _safe_rmtree(path: Path, *, retries: int = 3, delay: float = 1.0) -> None:
    """shutil.rmtree with retry for Windows memory-mapped file locks."""
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt < retries - 1:
                gc.collect()
                time.sleep(delay)
            else:
                raise


def _count_units_and_spikes(sorting) -> tuple[int, int]:
    n_units = len(sorting.unit_ids)
    total_spikes = sum(len(sorting.get_unit_spike_train(u)) for u in sorting.unit_ids)
    return n_units, total_spikes


def _compute_merge_split_features(
    analyzer, *, n_components: int, pc_mode: str, job_kwargs: dict
) -> None:
    """Compute only the features needed for merge + split."""
    analyzer.compute(
        {
            "random_spikes": {"method": "all"},
            "waveforms": {},
            "templates": {},
            "principal_components": {"n_components": n_components, "mode": pc_mode},
            "template_similarity": {},
            "correlograms": {},
        },
        **job_kwargs,
    )


def _compute_final_features(
    analyzer, *, n_components: int, pc_mode: str, job_kwargs: dict
) -> None:
    """Compute all features for quality metrics and Phy export."""
    analyzer.compute(
        {
            "random_spikes": {"method": "all"},
            "waveforms": {},
            "templates": {},
            "noise_levels": {},
            "spike_amplitudes": {},
            "principal_components": {"n_components": n_components, "mode": pc_mode},
            "template_metrics": {},
            "template_similarity": {},
            "correlograms": {},
            "spike_locations": {},
            "unit_locations": {},
        },
        **job_kwargs,
    )



def _resolve_recording_for_postprocess(config: PostprocessConfig):
    if config.recording is None and config.dat_path is None:
        raise ValueError("Either recording or dat_path must be provided")

    if config.recording is not None:
        rec = config.recording
        if config.preprocess_recording_object:
            rec = apply_preprocessing(
                recording_raw=rec,
                bandpass_min_hz=config.bandpass_min_hz,
                bandpass_max_hz=config.bandpass_max_hz,
                reference=config.reference,
                local_radius_um=config.local_radius_um,
            )
        return rec

    if config.sampling_frequency is None or config.num_channels is None:
        raise ValueError("sampling_frequency and num_channels are required when dat_path is used")

    rec_raw = se.read_binary(
        str(Path(config.dat_path)),
        sampling_frequency=float(config.sampling_frequency),
        dtype=config.dtype,
        num_channels=int(config.num_channels),
        gain_to_uV=config.gain_to_uV,
        offset_to_uV=config.offset_to_uV,
    )
    rec_with_probe, _, _ = attach_probe_and_remove_bad_channels(
        recording=rec_raw,
        chanmap_mat_path=config.chanmap_mat_path,
        reject_channels_0based=sorted(set(config.reject_channels)),
    )
    if config.apply_preprocessing_if_dat:
        return apply_preprocessing(
            recording_raw=rec_with_probe,
            bandpass_min_hz=config.bandpass_min_hz,
            bandpass_max_hz=config.bandpass_max_hz,
            reference=config.reference,
            local_radius_um=config.local_radius_um,
        )
    return rec_with_probe


def _fix_phy_params_file(params_file: Path, dat_path: Path, hp_filtered: bool) -> None:
    content = params_file.read_text(encoding="utf-8")
    dat_line = f"dat_path = r'{str(dat_path)}'"
    if re.search(r"(?m)^dat_path\s*=", content):
        content = re.sub(r"(?m)^dat_path\s*=.*$", lambda _: dat_line, content)
    else:
        content += f"\n{dat_line}\n"

    hp_line = f"hp_filtered = {str(bool(hp_filtered))}"
    if re.search(r"(?m)^hp_filtered\s*=", content):
        content = re.sub(r"(?m)^hp_filtered\s*=.*$", lambda _: hp_line, content)
    else:
        content += f"\n{hp_line}\n"
    params_file.write_text(content, encoding="utf-8")


def run_postprocess_session(config: PostprocessConfig) -> PostprocessResult:
    def _log(message: str) -> None:
        if config.verbose:
            print(f"[postprocess] {message}")

    _log("start run_postprocess_session()")
    # Release any stale memory-mapped arrays from a previous call in the same kernel
    gc.collect()

    sorting_phy_folder = Path(config.sorting_phy_folder).resolve()
    if not sorting_phy_folder.exists():
        raise FileNotFoundError(f"sorting_phy_folder not found: {sorting_phy_folder}")
    _log(f"sorting_phy_folder={sorting_phy_folder}")

    output_folder = sorting_phy_folder.parent / config.output_folder_name
    if output_folder.exists() and config.remove_if_exists and not config.skip_curation:
        _safe_rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    _log(f"output_folder={output_folder}")
    analyzer_cache_root: Path | None = None
    if config.analyzer_format == "binary_folder":
        # Place cache OUTSIDE output_folder to avoid PermissionError on Windows
        # when export_to_phy removes output_folder while .npy files are memory-mapped.
        analyzer_cache_root = (
            Path(config.analyzer_cache_dir).resolve()
            if config.analyzer_cache_dir is not None
            else (sorting_phy_folder.parent / (config.output_folder_name + "_analyzer_cache")).resolve()
        )
        if analyzer_cache_root.exists() and config.remove_if_exists and not config.skip_curation:
            _safe_rmtree(analyzer_cache_root)
        analyzer_cache_root.mkdir(parents=True, exist_ok=True)
        _log(f"analyzer_cache_dir={analyzer_cache_root}")

    # Clean up stale old-location cache inside output_folder (from runs before cache was moved outside)
    old_cache_inside_output = output_folder / "analyzer_cache"
    if old_cache_inside_output.exists():
        _log("removing stale analyzer_cache inside output_folder")
        _safe_rmtree(old_cache_inside_output)

    # Container to track intermediate analyzers for cleanup before export
    _intermediate_analyzers: list = []

    def _create_stage_analyzer(stage_name: str, sorting_obj):
        if config.analyzer_format == "binary_folder":
            assert analyzer_cache_root is not None
            stage_folder = analyzer_cache_root / stage_name
            if stage_folder.exists():
                _safe_rmtree(stage_folder)
            return si.create_sorting_analyzer(
                sorting=sorting_obj,
                recording=recording_for_post,
                format="binary_folder",
                folder=stage_folder,
                overwrite=True,
                **config.job_kwargs,
            )
        return si.create_sorting_analyzer(
            sorting=sorting_obj,
            recording=recording_for_post,
            format="memory",
            **config.job_kwargs,
        )

    recording_for_post = _resolve_recording_for_postprocess(config)
    _log("recording resolved")
    preprocessed_dat_path: Path | None = None
    params_dat_path_override: Path | None = None
    if config.write_preprocessed_dat_for_phy:
        _log("writing preprocessed DAT for Phy")
        preprocessed_dat_path = output_folder / "preprocessed_for_phy.dat"
        write_concatenated_dat(
            recording=recording_for_post,
            output_dat_path=preprocessed_dat_path,
            overwrite=True,
            job_kwargs=config.job_kwargs,
        )
        params_dat_path_override = preprocessed_dat_path

    _log("loading Phy sorting")
    sorting = se.read_phy(str(sorting_phy_folder), exclude_cluster_groups=config.exclude_cluster_groups)
    n_units_initial, total_spikes_initial = _count_units_and_spikes(sorting)
    _log(f"loaded sorting: n_units={n_units_initial}, total_spikes={total_spikes_initial}")

    # ---- skip_curation=True: try to load existing final analyzer -----------------
    _skipped_curation = False

    if config.skip_curation and config.analyzer_format == "binary_folder" and analyzer_cache_root is not None:
        split_folder = analyzer_cache_root / "split"
        if split_folder.exists():
            _log("[skip_curation=True] loading existing 'split' analyzer – skipping dedup/merge/split")
            analyzer_split = si.load_sorting_analyzer(folder=split_folder)
            sorting_split = analyzer_split.sorting
            _skipped_curation = True
        else:
            _log(
                "[skip_curation=True] WARNING: no existing 'split' analyzer found – "
                "falling back to full pipeline"
            )

    # ---- Full curation pipeline (when skip_curation=False or no cache found) ------
    if not _skipped_curation:
        _log("removing duplicated spikes")
        sorting_removed_duplicates = scur.remove_duplicated_spikes(
            sorting, censored_period_ms=config.duplicate_censored_period_ms
        )
        n_units_dedup, total_spikes_dedup = _count_units_and_spikes(sorting_removed_duplicates)
        _log(
            "after dedup: "
            f"n_units={n_units_dedup} (delta={n_units_dedup - n_units_initial}), "
            f"total_spikes={total_spikes_dedup} (removed={total_spikes_initial - total_spikes_dedup})"
        )
        _log("removing redundant units")
        analyzer_tmp = si.create_sorting_analyzer(
            sorting_removed_duplicates,
            recording_for_post,
            format="memory",
            sparse=False,
        )
        sorting_clean, _ = scur.remove_redundant_units(
            analyzer_tmp,
            align=False,
            duplicate_threshold=config.duplicate_threshold,
            remove_strategy=config.remove_strategy,
            extra_outputs=True,
        )
        del analyzer_tmp
        n_units_clean, total_spikes_clean = _count_units_and_spikes(sorting_clean)
        _log(
            "after remove_redundant_units: "
            f"n_units={n_units_clean} (removed={n_units_dedup - n_units_clean}), "
            f"total_spikes={total_spikes_clean}"
        )

        _log("computing merge/split features")
        analyzer = _create_stage_analyzer("main", sorting_clean)
        _intermediate_analyzers.append(analyzer)
        _compute_merge_split_features(
            analyzer,
            n_components=config.n_components,
            pc_mode=config.pc_mode,
            job_kwargs=config.job_kwargs,
        )

        # -- Merge --
        steps_params = {
            "num_spikes": {"min_spikes": config.merge_min_spikes},
            "correlogram": {"corr_diff_thresh": config.merge_corr_diff_thresh},
            "template_similarity": {"template_diff_thresh": config.merge_template_diff_thresh},
        }
        _log("computing merge candidates")
        merge_groups = scur.compute_merge_unit_groups(
            analyzer,
            preset="similarity_correlograms",
            resolve_graph=True,
            steps_params=steps_params,
            **config.job_kwargs,
        )
        _log(f"merge candidates: {len(merge_groups)} groups")
        mergeable = analyzer.are_units_mergeable(
            merge_unit_groups=merge_groups,
            merging_mode="soft",
            sparsity_overlap=config.merge_sparsity_overlap,
        )
        merge_groups = [g for g, ok in mergeable.items() if ok]
        _log(f"mergeable groups: {len(merge_groups)}")
        if merge_groups:
            _log("merging units")
            merge_kwargs = {
                "merge_unit_groups": merge_groups,
                "merging_mode": "soft",
                "censor_ms": config.merge_censor_ms,
                "sparsity_overlap": config.merge_sparsity_overlap,
                "return_new_unit_ids": False,
                "format": config.analyzer_format,
                "overwrite": True,
                **config.job_kwargs,
            }
            if config.analyzer_format == "binary_folder":
                assert analyzer_cache_root is not None
                merge_folder = analyzer_cache_root / "merged"
                if merge_folder.exists():
                    _safe_rmtree(merge_folder)
                merge_kwargs["folder"] = merge_folder
            analyzer_merged = analyzer.merge_units(
                **merge_kwargs,
            )
            _intermediate_analyzers.append(analyzer_merged)
        else:
            _log("skip merge (no mergeable groups)")
            analyzer_merged = analyzer

        _log(f"running auto split (verbose={config.split_verbose})")
        sorting_split = autosplit_outliers_pca(
            analyzer_merged,
            contamination=config.split_contamination,
            threshold_mode=config.split_threshold_mode,
            min_clean_frac=config.split_min_clean_frac,
            relax_factor=config.split_relax_factor,
            use_waveform_gate=config.split_use_waveform_gate,
            wf_threshold=config.split_wf_threshold,
            wf_template_max=config.split_wf_template_max,
            wf_n_chans=config.split_wf_n_chans,
            wf_center=config.split_wf_center,
            squeeze_all_outlier_to_new=config.split_squeeze_all_outlier_to_new,
            min_spikes=config.split_min_spikes,
            return_details=False,
            verbose=config.split_verbose,
            n_jobs=config.job_kwargs.get("n_jobs", -1),
        )

        _log("computing final features for metrics/export")
        # Release intermediate analyzers to free memory-mapped .npy handles (Windows)
        _intermediate_analyzers.clear()
        gc.collect()

        analyzer_split = _create_stage_analyzer("split", sorting_split)
        _compute_final_features(
            analyzer_split,
            n_components=config.n_components,
            pc_mode=config.pc_mode,
            job_kwargs=config.job_kwargs,
        )

    _log("computing quality metrics")
    qm_params = sqm.get_default_qm_params()
    metrics_df = sqm.compute_quality_metrics(
        analyzer_split,
        metric_names=config.metric_names,
        metric_params=qm_params,
        skip_pc_metrics=config.skip_pc_metrics,
        **config.job_kwargs,
    )
    metrics_csv_path = output_folder / config.metrics_csv_name
    metrics_out_df = pd.DataFrame(metrics_df)
    metrics_out_df.to_csv(metrics_csv_path, index=True)

    _log("exporting to Phy")
    export_to_phy(
        sorting_analyzer=analyzer_split,
        output_folder=output_folder,
        compute_pc_features=True,
        compute_amplitudes=True,
        copy_binary=config.copy_binary,
        remove_if_exists=config.remove_if_exists,
        template_mode="average",
        add_quality_metrics=True,
        add_template_metrics=True,
        dtype=None,
        use_relative_path=config.use_relative_path,
        verbose=True,
        **config.job_kwargs,
    )
    params_file = output_folder / "params.py"
    if config.force_params_dat_path is not None:
        params_dat_path_override = Path(config.force_params_dat_path)
    # Default: use the input dat_path (concatenated recording) when nothing else is set
    if params_dat_path_override is None and config.dat_path is not None:
        params_dat_path_override = Path(config.dat_path)
    if params_dat_path_override is not None:
        _log("fixing params.py dat_path/hp_filtered")
        _fix_phy_params_file(
            params_file=params_file,
            dat_path=params_dat_path_override,
            hp_filtered=config.phy_hp_filtered,
        )

    # export_to_phy can recreate output_folder when remove_if_exists=True.
    # Persist metrics again after export so quality_metrics.csv is always present.
    metrics_out_df.to_csv(metrics_csv_path, index=True)

    _log("marking noise clusters from metrics")
    updated = mark_noise_clusters_from_metrics(
        phy_dir=output_folder,
        metrics_df=metrics_df,
        thresholds=config.noise_thresholds,
        backup=config.noise_backup,
        reset_to_unsorted=True,
        update_cluster_info=True,
    )
    n_noise_clusters = int((updated["group"] == "noise").sum())

    n_units_final, total_spikes_final = _count_units_and_spikes(sorting_split)
    if config.delete_analyzer_cache and analyzer_cache_root is not None and analyzer_cache_root.exists():
        _safe_rmtree(analyzer_cache_root)
        analyzer_cache_for_result = None
    else:
        analyzer_cache_for_result = analyzer_cache_root

    _log(
        f"done: n_units_final={n_units_final}, total_spikes_final={total_spikes_final}, "
        f"n_noise_clusters={n_noise_clusters}"
    )
    return PostprocessResult(
        sorting_phy_folder=sorting_phy_folder,
        output_folder=output_folder,
        preprocessed_dat_path=preprocessed_dat_path,
        metrics_csv_path=metrics_csv_path,
        analyzer_cache_dir=analyzer_cache_for_result,
        n_units_initial=n_units_initial,
        n_units_final=n_units_final,
        total_spikes_initial=total_spikes_initial,
        total_spikes_final=total_spikes_final,
        n_noise_clusters=n_noise_clusters,
    )


def attach_existing_sorting_result(
    result: PreprocessResult,
    *,
    sorter: str | None,
    existing_sorting_dir: str | Path | None = None,
    sorting_temp_root: str | Path | None = None,
) -> PreprocessResult:
    if result.sorter_output_dir is not None and Path(result.sorter_output_dir).exists():
        return result

    if existing_sorting_dir is not None:
        sorting_dir = Path(existing_sorting_dir)
    else:
        root = Path(sorting_temp_root) if sorting_temp_root is not None else (Path("sorting_temp") / result.basename)
        candidates = sorted(
            [p for p in root.glob("Kilosort_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No sorting result found under {root}.")
        sorting_dir = candidates[0]

    if not sorting_dir.exists():
        raise FileNotFoundError(f"Existing sorting folder not found: {sorting_dir}")

    return replace(
        result,
        sorter=sorter,
        sorter_output_dir=sorting_dir.resolve(),
    )


def build_preprocessed_recording_from_result(
    result: PreprocessResult,
    preprocess_config: PreprocessConfig,
):
    if result.dat_path is None:
        raise ValueError("result.dat_path is None. Run preprocess with save_raw=True.")

    rec_raw = se.read_binary(
        str(result.dat_path),
        sampling_frequency=result.sr,
        dtype=preprocess_config.dtype,
        num_channels=result.n_channels,
        gain_to_uV=preprocess_config.gain_to_uV,
        offset_to_uV=preprocess_config.offset_to_uV,
    )
    rec_with_probe, _, _ = attach_probe_and_remove_bad_channels(
        recording=rec_raw,
        chanmap_mat_path=preprocess_config.chanmap_mat_path,
        reject_channels_0based=preprocess_config.reject_channels,
    )
    return apply_preprocessing(
        recording_raw=rec_with_probe,
        bandpass_min_hz=preprocess_config.bandpass_min_hz,
        bandpass_max_hz=preprocess_config.bandpass_max_hz,
        reference=preprocess_config.reference,
        local_radius_um=preprocess_config.local_radius_um,
    )


def run_postprocess_from_preprocess(
    result: PreprocessResult,
    preprocess_config: PreprocessConfig,
    *,
    recording=None,
    output_folder_name: str = "sorter_output_postprocessed",
    analyzer_format: str = "binary_folder",
    analyzer_cache_dir: str | Path | None = None,
    delete_analyzer_cache: bool = False,
    phy_hp_filtered: bool = True,
) -> PostprocessResult:
    if result.sorter_output_dir is None:
        raise RuntimeError("No sorter output found in result.sorter_output_dir")

    sorting_phy_folder = Path(result.sorter_output_dir) / "sorter_output"
    use_recording_object = recording is not None

    post_cfg = PostprocessConfig(
        sorting_phy_folder=sorting_phy_folder,
        recording=recording if use_recording_object else None,
        dat_path=None if use_recording_object else result.dat_path,
        sampling_frequency=result.sr,
        num_channels=result.n_channels,
        dtype=preprocess_config.dtype,
        gain_to_uV=preprocess_config.gain_to_uV,
        offset_to_uV=preprocess_config.offset_to_uV,
        chanmap_mat_path=preprocess_config.chanmap_mat_path,
        reject_channels=preprocess_config.reject_channels,
        apply_preprocessing_if_dat=True,
        output_folder_name=output_folder_name,
        analyzer_format=analyzer_format,
        analyzer_cache_dir=Path(analyzer_cache_dir) if analyzer_cache_dir is not None else None,
        delete_analyzer_cache=delete_analyzer_cache,
        phy_hp_filtered=phy_hp_filtered,
    )
    return run_postprocess_session(post_cfg)


# Short aliases for notebook use
def use_existing_sorting(
    result: PreprocessResult,
    *,
    sorter: str | None,
    existing_sorting_dir: str | Path | None = None,
    sorting_temp_root: str | Path | None = None,
) -> PreprocessResult:
    return attach_existing_sorting_result(
        result,
        sorter=sorter,
        existing_sorting_dir=existing_sorting_dir,
        sorting_temp_root=sorting_temp_root,
    )


def make_post_recording(result: PreprocessResult, preprocess_config: PreprocessConfig):
    return build_preprocessed_recording_from_result(result, preprocess_config)


def run_postprocess(
    result: PreprocessResult,
    preprocess_config: PreprocessConfig,
    *,
    recording=None,
    output_folder_name: str = "sorter_output_postprocessed",
    analyzer_format: str = "binary_folder",
    analyzer_cache_dir: str | Path | None = None,
    delete_analyzer_cache: bool = False,
    phy_hp_filtered: bool = True,
) -> PostprocessResult:
    return run_postprocess_from_preprocess(
        result,
        preprocess_config,
        recording=recording,
        output_folder_name=output_folder_name,
        analyzer_format=analyzer_format,
        analyzer_cache_dir=analyzer_cache_dir,
        delete_analyzer_cache=delete_analyzer_cache,
        phy_hp_filtered=phy_hp_filtered,
    )
