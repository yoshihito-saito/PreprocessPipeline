from __future__ import annotations

from dataclasses import replace
import gc
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.curation as scur
import spikeinterface.extractors as se
import spikeinterface.qualitymetrics as sqm
from spikeinterface.exporters import export_to_phy

from .unit_classify import mark_noise_clusters_from_metrics
from .unit_split import autosplit_outliers_pca

from ..preprocess.metafile import PreprocessConfig, PreprocessResult
from ..preprocess.recording import (
    apply_preprocessing,
    attach_probe_and_remove_bad_channels,
    preprocess_selected_channels_preserve_shape,
)
from .metafile import PostprocessConfig, PostprocessResult


def _find_sorting_output_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            p
            for pattern in ("Kilosort_*", "Kilosort4_*")
            for p in root.glob(pattern)
            if p.is_dir()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def search_sorter_paths(local_output_dir: str | Path) -> list[Path]:
    root = Path(local_output_dir)
    if not root.exists():
        raise FileNotFoundError(f"local output dir not found: {root}")
    return [p.resolve() for p in _find_sorting_output_dirs(root)]


def _resolve_sorting_run_root(sorting_phy_folder: Path) -> Path:
    # Legacy layouts may point to <Kilosort_xxx>/sorter_output.
    # Current layouts use <Kilosort_xxx> directly.
    return sorting_phy_folder.parent if sorting_phy_folder.name == "sorter_output" else sorting_phy_folder


def _resolve_postprocess_output_folder(sorting_phy_folder: Path) -> Path:
    sorting_run_root = _resolve_sorting_run_root(sorting_phy_folder)
    spi_root = (sorting_run_root.parent / f"{sorting_run_root.name}_spi").resolve()
    return spi_root


def _resolve_sorting_root_from_result(
    result: PreprocessResult,
    *,
    kilosort_path: str | None = None,
) -> Path:
    root = Path(result.local_output_dir)
    if kilosort_path is not None:
        name = str(kilosort_path).strip()
        folder = Path(name)
        if (
            not name
            or folder.is_absolute()
            or len(folder.parts) != 1
            or folder.name in {".", ".."}
        ):
            raise ValueError(
                "kilosort_path must be a single folder name under result.local_output_dir "
                f"(got: {kilosort_path!r})"
            )
        resolved = (root / folder.name).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Kilosort folder not found: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Kilosort path is not a directory: {resolved}")
        return resolved

    candidates = _find_sorting_output_dirs(root)
    if not candidates:
        raise FileNotFoundError(f"No Kilosort result found under {root}.")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            "Multiple Kilosort folders found under "
            f"{root}. Specify kilosort_path explicitly. Candidates: {names}"
        )
    return candidates[0]


def _resolve_bad_channels_for_postprocess(
    result: PreprocessResult, preprocess_config: PreprocessConfig
) -> list[int]:
    if result.bad_channels_0based:
        return sorted(set(int(ch) for ch in result.bad_channels_0based))
    return sorted(set(int(ch) for ch in preprocess_config.reject_channels))


def _resolve_phy_sample_rate(phy_dir: Path, fallback_sample_rate: float | None = None) -> float:
    params_path = phy_dir / "params.py"
    if params_path.exists():
        text = params_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^\s*sample_rate\s*=\s*([0-9eE+\-.]+)", text)
        if match is not None:
            return float(match.group(1))
    if fallback_sample_rate is not None:
        return float(fallback_sample_rate)
    raise ValueError(
        f"Could not resolve sample_rate from {params_path} and no fallback sampling_frequency was provided."
    )


def _mark_low_firing_rate_clusters_as_noise(
    *,
    phy_dir: Path,
    threshold_hz: float,
    fallback_sample_rate: float | None,
) -> tuple[int, int]:
    spike_times_path = phy_dir / "spike_times.npy"
    spike_clusters_path = phy_dir / "spike_clusters.npy"
    if not spike_times_path.exists() or not spike_clusters_path.exists():
        raise FileNotFoundError(
            "Missing Phy spike files. Expected both spike_times.npy and spike_clusters.npy under "
            f"{phy_dir}."
        )

    spike_times = np.asarray(np.load(spike_times_path, mmap_mode="r")).reshape(-1)
    spike_clusters = np.asarray(np.load(spike_clusters_path, mmap_mode="r")).reshape(-1)
    if spike_times.size != spike_clusters.size:
        raise ValueError(
            "spike_times.npy and spike_clusters.npy have mismatched lengths: "
            f"{spike_times.size} vs {spike_clusters.size}"
        )

    sample_rate = _resolve_phy_sample_rate(phy_dir, fallback_sample_rate=fallback_sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample_rate resolved for {phy_dir}: {sample_rate}")

    if spike_clusters.size == 0:
        unique_clusters = np.asarray([], dtype=np.int64)
        low_rate_clusters = np.asarray([], dtype=np.int64)
    else:
        unique_clusters, counts = np.unique(spike_clusters.astype(np.int64), return_counts=True)
        duration_sec = float(np.max(spike_times.astype(np.int64)) + 1) / float(sample_rate)
        if duration_sec <= 0:
            low_rate_clusters = np.asarray([], dtype=np.int64)
        else:
            firing_rates = counts.astype(np.float64) / duration_sec
            low_rate_clusters = unique_clusters[firing_rates < float(threshold_hz)]

    cg_path = phy_dir / "cluster_group.tsv"
    if cg_path.exists():
        cg = pd.read_csv(cg_path, sep="\t")
        cols_lut = {str(col).strip().lower(): col for col in cg.columns}
        cluster_col = cols_lut.get("cluster_id", cg.columns[0] if cg.shape[1] >= 1 else None)
        group_col = cols_lut.get("group") or cols_lut.get("kslabel") or cols_lut.get("label")
        if group_col is None and cg.shape[1] >= 2:
            group_col = cg.columns[1]
        if cluster_col is None or group_col is None:
            raise ValueError(f"Invalid cluster_group.tsv format: {cg_path}")
        cg = cg[[cluster_col, group_col]].copy()
        cg.columns = ["cluster_id", "group"]
        cg["cluster_id"] = pd.to_numeric(cg["cluster_id"], errors="coerce")
        cg = cg.dropna(subset=["cluster_id"]).copy()
        cg["cluster_id"] = cg["cluster_id"].astype(np.int64)
        cg["group"] = cg["group"].astype(str)
    else:
        cg = pd.DataFrame(columns=["cluster_id", "group"])

    existing_ids = cg["cluster_id"].to_numpy(dtype=np.int64) if not cg.empty else np.asarray([], dtype=np.int64)
    all_ids = np.unique(np.concatenate((existing_ids, unique_clusters))).astype(np.int64, copy=False)
    if all_ids.size == 0:
        out_df = pd.DataFrame(columns=["cluster_id", "group"])
    else:
        out_df = pd.DataFrame({"cluster_id": all_ids, "group": "unsorted"})
        if not cg.empty:
            group_map = dict(zip(cg["cluster_id"].tolist(), cg["group"].tolist()))
            out_df["group"] = out_df["cluster_id"].map(group_map).fillna("unsorted")
        if low_rate_clusters.size > 0:
            low_set = {int(x) for x in low_rate_clusters.tolist()}
            out_df.loc[out_df["cluster_id"].isin(low_set), "group"] = "noise"
        out_df = out_df.sort_values("cluster_id").reset_index(drop=True)

    out_df.to_csv(cg_path, sep="\t", index=False)
    return int(out_df.shape[0]), int(low_rate_clusters.size)


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


def _clear_folder_contents(folder: Path, *, keep_names: set[str] | None = None) -> None:
    keep = keep_names or set()
    if not folder.exists():
        return
    for child in folder.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            _safe_rmtree(child)
        else:
            child.unlink(missing_ok=True)


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
        return config.recording

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
    rec_with_probe, bad_0, _ = attach_probe_and_remove_bad_channels(
        recording=rec_raw,
        chanmap_mat_path=config.chanmap_mat_path,
        reject_channels_0based=sorted(set(config.reject_channels)),
    )
    if config.apply_preprocess:
        if hasattr(rec_with_probe, "get_channel_ids"):
            bad_set = {int(ch) for ch in bad_0}
            all_channels = [int(ch) for ch in rec_with_probe.get_channel_ids()]
            good_channels = [ch for ch in all_channels if ch not in bad_set]
            if not good_channels:
                return rec_with_probe
            return preprocess_selected_channels_preserve_shape(
                recording_raw=rec_with_probe,
                selected_channel_ids=good_channels,
                bandpass_min_hz=config.bandpass_min_hz,
                bandpass_max_hz=config.bandpass_max_hz,
                reference=config.reference,
                local_radius_um=config.local_radius_um,
            )
        return apply_preprocessing(
            recording_raw=rec_with_probe,
            bandpass_min_hz=config.bandpass_min_hz,
            bandpass_max_hz=config.bandpass_max_hz,
            reference=config.reference,
            local_radius_um=config.local_radius_um,
        )
    return rec_with_probe


def _fix_phy_params_file(
    params_file: Path,
    dat_path: Path,
    hp_filtered: bool,
    *,
    use_relative_path: bool,
) -> None:
    content = params_file.read_text(encoding="utf-8")
    dat_path_obj = Path(dat_path)
    dat_value = str(dat_path_obj)
    if use_relative_path:
        try:
            dat_value = Path(
                os.path.relpath(dat_path_obj.resolve(), start=params_file.parent.resolve())
            ).as_posix()
        except Exception:
            dat_value = str(dat_path_obj)
    dat_line = f"dat_path = r'{dat_value}'"
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

    output_folder = _resolve_postprocess_output_folder(sorting_phy_folder)
    if output_folder.exists() and config.remove_if_exists and not config.skip_curation:
        _safe_rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    _log(f"output_folder={output_folder}")
    analyzer_cache_root: Path | None = None
    if config.analyzer_format == "binary_folder":
        analyzer_cache_root = (
            Path(config.analyzer_cache_dir).resolve()
            if config.analyzer_cache_dir is not None
            else (output_folder / "analyzer_cache").resolve()
        )
        if analyzer_cache_root.exists() and config.remove_if_exists and not config.skip_curation:
            _safe_rmtree(analyzer_cache_root)
        analyzer_cache_root.mkdir(parents=True, exist_ok=True)
        _log(f"analyzer_cache_dir={analyzer_cache_root}")

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

    low_rate_threshold_hz = float(config.noise_thresholds.get("firing_rate_lt", 0.01))
    _log(
        "marking low firing-rate clusters as noise before Phy load "
        f"(threshold={low_rate_threshold_hz:g} Hz)"
    )
    n_clusters_total, n_clusters_low_rate = _mark_low_firing_rate_clusters_as_noise(
        phy_dir=sorting_phy_folder,
        threshold_hz=low_rate_threshold_hz,
        fallback_sample_rate=config.sampling_frequency,
    )
    _log(
        "updated cluster_group.tsv before Phy load: "
        f"total_clusters={n_clusters_total}, low_rate_noise={n_clusters_low_rate}"
    )

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
    keep_names: set[str] = set()
    if analyzer_cache_root is not None and analyzer_cache_root.parent == output_folder:
        keep_names.add(analyzer_cache_root.name)
    if config.remove_if_exists:
        _clear_folder_contents(output_folder, keep_names=keep_names)

    # export_to_phy requires output_folder not to exist. Since we keep output_folder
    # (and optionally analyzer_cache) around, export into a temporary child folder and
    # then move exported files to output_folder root.
    phy_export_tmp = output_folder / "__phy_export_tmp__"
    if analyzer_cache_root is not None and phy_export_tmp.resolve() == analyzer_cache_root.resolve():
        raise ValueError("analyzer_cache_dir cannot be '__phy_export_tmp__' under output_folder.")
    if phy_export_tmp.exists():
        _safe_rmtree(phy_export_tmp)

    export_to_phy(
        sorting_analyzer=analyzer_split,
        output_folder=phy_export_tmp,
        compute_pc_features=True,
        compute_amplitudes=True,
        copy_binary=config.copy_binary,
        remove_if_exists=False,
        template_mode="average",
        add_quality_metrics=True,
        add_template_metrics=True,
        dtype=None,
        use_relative_path=config.use_relative_path,
        verbose=True,
        **config.job_kwargs,
    )
    for child in phy_export_tmp.iterdir():
        destination = output_folder / child.name
        if destination.exists():
            if destination.is_dir():
                _safe_rmtree(destination)
            else:
                destination.unlink(missing_ok=True)
        shutil.move(str(child), str(destination))
    _safe_rmtree(phy_export_tmp)

    params_file = output_folder / "params.py"
    if config.dat_path is not None:
        _log("fixing params.py dat_path/hp_filtered")
        _fix_phy_params_file(
            params_file=params_file,
            dat_path=Path(config.dat_path),
            hp_filtered=(not bool(config.apply_preprocess)),
            use_relative_path=bool(config.use_relative_path),
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
        root = Path(sorting_temp_root) if sorting_temp_root is not None else Path(result.local_output_dir)
        candidates = _find_sorting_output_dirs(root)
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
        reject_channels_0based=_resolve_bad_channels_for_postprocess(result, preprocess_config),
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
    kilosort_path: str | None = None,
    apply_preprocess: bool | None = None,
    analyzer_format: str = "binary_folder",
    analyzer_cache_dir: str | Path | None = None,
    delete_analyzer_cache: bool = False,
) -> PostprocessResult:
    sorting_root = _resolve_sorting_root_from_result(result, kilosort_path=kilosort_path)
    sorting_phy_folder = sorting_root
    use_recording_object = recording is not None
    apply_preprocess_effective = (
        (not bool(preprocess_config.do_preprocess))
        if apply_preprocess is None
        else bool(apply_preprocess)
    )
    reject_channels = _resolve_bad_channels_for_postprocess(result, preprocess_config)

    post_cfg = PostprocessConfig(
        sorting_phy_folder=sorting_phy_folder,
        recording=recording if use_recording_object else None,
        dat_path=result.dat_path,
        sampling_frequency=result.sr,
        num_channels=result.n_channels,
        dtype=preprocess_config.dtype,
        gain_to_uV=preprocess_config.gain_to_uV,
        offset_to_uV=preprocess_config.offset_to_uV,
        chanmap_mat_path=preprocess_config.chanmap_mat_path,
        reject_channels=reject_channels,
        apply_preprocess=apply_preprocess_effective,
        analyzer_format=analyzer_format,
        analyzer_cache_dir=Path(analyzer_cache_dir) if analyzer_cache_dir is not None else None,
        delete_analyzer_cache=delete_analyzer_cache,
    )
    return run_postprocess_session(post_cfg)


def run_postprocess_many_from_preprocess(
    result: PreprocessResult,
    preprocess_config: PreprocessConfig,
    *,
    kilosort_paths: list[str],
    recording=None,
    apply_preprocess: bool | None = None,
    analyzer_format: str = "binary_folder",
    analyzer_cache_dir: str | Path | None = None,
    delete_analyzer_cache: bool = False,
) -> list[PostprocessResult]:
    if not kilosort_paths:
        raise ValueError("kilosort_paths must contain at least one folder name.")

    results: list[PostprocessResult] = []
    for folder_name in kilosort_paths:
        results.append(
            run_postprocess_from_preprocess(
                result=result,
                preprocess_config=preprocess_config,
                recording=recording,
                kilosort_path=folder_name,
                apply_preprocess=apply_preprocess,
                analyzer_format=analyzer_format,
                analyzer_cache_dir=analyzer_cache_dir,
                delete_analyzer_cache=delete_analyzer_cache,
            )
        )
    return results


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
    kilosort_path: str | None = None,
    apply_preprocess: bool | None = None,
    analyzer_format: str = "binary_folder",
    analyzer_cache_dir: str | Path | None = None,
    delete_analyzer_cache: bool = False,
) -> PostprocessResult:
    return run_postprocess_from_preprocess(
        result,
        preprocess_config,
        recording=recording,
        kilosort_path=kilosort_path,
        apply_preprocess=apply_preprocess,
        analyzer_format=analyzer_format,
        analyzer_cache_dir=analyzer_cache_dir,
        delete_analyzer_cache=delete_analyzer_cache,
    )


def run_postprocess_many(
    result: PreprocessResult,
    preprocess_config: PreprocessConfig,
    *,
    kilosort_paths: list[str],
    recording=None,
    apply_preprocess: bool | None = None,
    analyzer_format: str = "binary_folder",
    analyzer_cache_dir: str | Path | None = None,
    delete_analyzer_cache: bool = False,
) -> list[PostprocessResult]:
    return run_postprocess_many_from_preprocess(
        result=result,
        preprocess_config=preprocess_config,
        kilosort_paths=kilosort_paths,
        recording=recording,
        apply_preprocess=apply_preprocess,
        analyzer_format=analyzer_format,
        analyzer_cache_dir=analyzer_cache_dir,
        delete_analyzer_cache=delete_analyzer_cache,
    )
