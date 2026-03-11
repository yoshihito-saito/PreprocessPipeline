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

_SORTING_OUTPUT_PATTERNS = (
    "Kilosort_*",
    "Kilosort2_5_*",
    "Kilosort2.5_*",
    "Kilosort4_*",
)


def _find_sorting_output_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            p
            for pattern in _SORTING_OUTPUT_PATTERNS
            for p in root.glob(pattern)
            if p.is_dir() and not p.name.endswith("_spi")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

def _resolve_sorting_run_root(sorting_phy_folder: Path) -> Path:
    # Legacy layouts may point to <Kilosort_xxx>/sorter_output.
    # Current layouts use <Kilosort_xxx> directly.
    return sorting_phy_folder.parent if sorting_phy_folder.name == "sorter_output" else sorting_phy_folder


def _resolve_postprocess_output_folder(sorting_phy_folder: Path) -> Path:
    sorting_run_root = _resolve_sorting_run_root(sorting_phy_folder)
    spi_root = (sorting_run_root.parent / f"{sorting_run_root.name}_spi").resolve()
    return spi_root


def _resolve_analyzer_cache_root(config: PostprocessConfig, output_folder: Path) -> Path | None:
    if config.analyzer_format != "binary_folder":
        return None
    return (
        Path(config.analyzer_cache_dir).resolve()
        if config.analyzer_cache_dir is not None
        else (output_folder / "analyzer_cache").resolve()
    )


def _should_skip_postprocess_target(
    config: PostprocessConfig,
    *,
    output_folder: Path,
    metrics_csv_path: Path,
) -> bool:
    overwrite = _postprocess_overwrite_enabled(config)
    return (not overwrite) and metrics_csv_path.exists() and _phy_export_outputs_exist(output_folder)


def _resolve_postprocess_search_root(config: PostprocessConfig) -> Path:
    if config.sorting_search_root is not None:
        root = Path(config.sorting_search_root).resolve()
    elif config.dat_path is not None:
        root = Path(config.dat_path).resolve().parent
    else:
        raise ValueError(
            "sorting_phy_folder is None, but no sorting_search_root was provided and dat_path is unavailable."
        )
    if not root.exists():
        raise FileNotFoundError(f"sorting search root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"sorting search root is not a directory: {root}")
    return root


def _resolve_postprocess_targets(config: PostprocessConfig) -> list[Path]:
    if config.sorting_phy_folder is not None:
        sorting_phy_folder = Path(config.sorting_phy_folder).resolve()
        if not sorting_phy_folder.exists():
            raise FileNotFoundError(f"sorting_phy_folder not found: {sorting_phy_folder}")
        if not sorting_phy_folder.is_dir():
            raise NotADirectoryError(f"sorting_phy_folder is not a directory: {sorting_phy_folder}")
        return [sorting_phy_folder]

    root = _resolve_postprocess_search_root(config)
    candidates = [p.resolve() for p in _find_sorting_output_dirs(root)]
    if not candidates:
        raise FileNotFoundError(f"No Kilosort result found under {root}.")
    return candidates


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
            low_rate_clusters = unique_clusters[firing_rates <= float(threshold_hz)]

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


def _postprocess_overwrite_enabled(config: PostprocessConfig) -> bool:
    if config.remove_if_exists is not None:
        return bool(config.remove_if_exists)
    return bool(config.overwrite)


def _phy_export_outputs_exist(output_folder: Path) -> bool:
    required = (
        "params.py",
        "spike_times.npy",
        "spike_clusters.npy",
        "cluster_group.tsv",
        "cluster_info.tsv",
    )
    return all((output_folder / name).exists() for name in required)


def _count_noise_clusters_in_phy_dir(phy_dir: Path) -> int:
    for filename in ("cluster_info.tsv", "cluster_group.tsv"):
        path = phy_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        cols_lut = {str(col).strip().lower(): col for col in df.columns}
        group_col = cols_lut.get("group") or cols_lut.get("kslabel") or cols_lut.get("label")
        if group_col is None and df.shape[1] >= 2:
            group_col = df.columns[1]
        if group_col is None:
            continue
        groups = df[group_col].astype(str).str.lower()
        return int((groups == "noise").sum())
    return 0


def _load_postprocess_result_from_existing_outputs(
    *,
    config: PostprocessConfig,
    sorting_phy_folder: Path,
    output_folder: Path,
    analyzer_cache_root: Path | None,
    metrics_csv_path: Path,
    preprocessed_dat_path: Path | None,
) -> PostprocessResult:
    sorting_initial = se.read_phy(str(sorting_phy_folder), exclude_cluster_groups=config.exclude_cluster_groups)
    sorting_final = se.read_phy(str(output_folder), exclude_cluster_groups=config.exclude_cluster_groups)
    n_units_initial, total_spikes_initial = _count_units_and_spikes(sorting_initial)
    n_units_final, total_spikes_final = _count_units_and_spikes(sorting_final)
    analyzer_cache_for_result = (
        analyzer_cache_root
        if analyzer_cache_root is not None and analyzer_cache_root.exists() and not config.delete_analyzer_cache
        else None
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
        n_noise_clusters=_count_noise_clusters_in_phy_dir(output_folder),
    )


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


def _merge_template_metrics_into_metrics_df(
    metrics_df: pd.DataFrame,
    analyzer,
    *,
    template_metric_names: list[str],
) -> pd.DataFrame:
    if not template_metric_names:
        return pd.DataFrame(metrics_df)

    template_metrics = analyzer.compute(
        "template_metrics",
        metric_names=template_metric_names,
        peak_sign="neg",
    )
    template_df = pd.DataFrame(template_metrics)
    if template_df.empty:
        return pd.DataFrame(metrics_df)

    merged = pd.DataFrame(metrics_df).join(template_df, how="left")
    repol = pd.to_numeric(merged.get("repolarization_slope"), errors="coerce")
    recovery = pd.to_numeric(merged.get("recovery_slope"), errors="coerce")
    if repol is not None and recovery is not None:
        merged["slope"] = pd.concat((repol.abs(), recovery.abs()), axis=1).min(axis=1)
    return merged


def _sorting_analyzer_sparsity_kwargs(config: PostprocessConfig) -> dict[str, Any]:
    if not config.analyzer_sparse:
        return {"sparse": False}

    kwargs: dict[str, Any] = {
        "sparse": True,
        "method": config.sparsity_method,
    }
    if config.sparsity_method in {"best_channels", "closest_channels"}:
        kwargs["num_channels"] = int(config.sparsity_num_channels)
    return kwargs


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


def _export_phy_to_output_folder(
    *,
    sorting_analyzer,
    output_folder: Path,
    analyzer_cache_root: Path | None,
    dat_path: Path | None,
    hp_filtered: bool,
    copy_binary: bool,
    use_relative_path: bool,
    job_kwargs: dict[str, Any],
) -> None:
    # export_to_phy requires output_folder not to exist. Since we keep output_folder
    # (and optionally analyzer_cache) around, export into a temporary child folder and
    # then move exported files to output_folder root. Because the temp folder is a child
    # of the final output folder, export_to_phy cannot safely compute a relative dat_path
    # against it when the binary lives alongside the session output. Export absolute paths
    # first, then rewrite params.py against the final folder.
    phy_export_tmp = output_folder / "__phy_export_tmp__"
    if analyzer_cache_root is not None and phy_export_tmp.resolve() == analyzer_cache_root.resolve():
        raise ValueError("analyzer_cache_dir cannot be '__phy_export_tmp__' under output_folder.")
    if phy_export_tmp.exists():
        _safe_rmtree(phy_export_tmp)

    export_to_phy(
        sorting_analyzer=sorting_analyzer,
        output_folder=phy_export_tmp,
        compute_pc_features=True,
        compute_amplitudes=True,
        copy_binary=copy_binary,
        remove_if_exists=False,
        template_mode="average",
        add_quality_metrics=True,
        add_template_metrics=True,
        dtype=None,
        use_relative_path=False,
        verbose=True,
        **job_kwargs,
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
    if dat_path is not None:
        _fix_phy_params_file(
            params_file=params_file,
            dat_path=Path(dat_path),
            hp_filtered=hp_filtered,
            use_relative_path=bool(use_relative_path),
        )


def _config_for_sorting_target(
    config: PostprocessConfig,
    *,
    sorting_phy_folder: Path,
    multiple_targets: bool,
) -> PostprocessConfig:
    if not multiple_targets or config.analyzer_cache_dir is None:
        return config
    return replace(
        config,
        analyzer_cache_dir=(Path(config.analyzer_cache_dir).resolve() / _resolve_sorting_run_root(sorting_phy_folder).name),
    )


def _run_postprocess_single_session(
    config: PostprocessConfig,
    *,
    sorting_phy_folder: Path,
) -> PostprocessResult:
    def _log(message: str) -> None:
        if config.verbose:
            print(f"[postprocess] {message}")

    # Release any stale memory-mapped arrays from a previous call in the same kernel
    gc.collect()

    overwrite = _postprocess_overwrite_enabled(config)
    output_folder = _resolve_postprocess_output_folder(sorting_phy_folder)
    preprocessed_dat_path: Path | None = None
    metrics_csv_path = output_folder / config.metrics_csv_name
    analyzer_cache_root = _resolve_analyzer_cache_root(config, output_folder)
    if _should_skip_postprocess_target(
        config,
        output_folder=output_folder,
        metrics_csv_path=metrics_csv_path,
    ):
        return _load_postprocess_result_from_existing_outputs(
            config=config,
            sorting_phy_folder=sorting_phy_folder,
            output_folder=output_folder,
            analyzer_cache_root=analyzer_cache_root,
            metrics_csv_path=metrics_csv_path,
            preprocessed_dat_path=preprocessed_dat_path,
        )

    _log(f"sorting_phy_folder={sorting_phy_folder}")
    if output_folder.exists() and overwrite and not config.skip_curation:
        _safe_rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    _log(f"output_folder={output_folder}")
    if analyzer_cache_root is not None:
        if analyzer_cache_root.exists() and overwrite and not config.skip_curation:
            _safe_rmtree(analyzer_cache_root)
        analyzer_cache_root.mkdir(parents=True, exist_ok=True)
        _log(f"analyzer_cache_dir={analyzer_cache_root}")

    # Container to track intermediate analyzers for cleanup before export
    _intermediate_analyzers: list = []

    recording_for_post = None

    def _ensure_recording_for_post():
        nonlocal recording_for_post
        if recording_for_post is None:
            recording_for_post = _resolve_recording_for_postprocess(config)
            _log("recording resolved")
        return recording_for_post

    def _create_stage_analyzer(stage_name: str, sorting_obj):
        analyzer_kwargs = {
            "sorting": sorting_obj,
            "recording": _ensure_recording_for_post(),
            **_sorting_analyzer_sparsity_kwargs(config),
        }
        if config.analyzer_format == "binary_folder":
            assert analyzer_cache_root is not None
            stage_folder = analyzer_cache_root / stage_name
            if stage_folder.exists():
                _safe_rmtree(stage_folder)
            return si.create_sorting_analyzer(
                **analyzer_kwargs,
                format="binary_folder",
                folder=stage_folder,
                overwrite=True,
                **config.job_kwargs,
            )
        return si.create_sorting_analyzer(
            **analyzer_kwargs,
            format="memory",
            **config.job_kwargs,
        )

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

    if (config.skip_curation or not overwrite) and config.analyzer_format == "binary_folder" and analyzer_cache_root is not None:
        split_folder = analyzer_cache_root / "split"
        if split_folder.exists():
            _log("loading existing 'split' analyzer – skipping dedup/merge/split")
            analyzer_split = si.load_sorting_analyzer(folder=split_folder)
            sorting_split = analyzer_split.sorting
            _skipped_curation = True
        else:
            if config.skip_curation:
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
            _ensure_recording_for_post(),
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

    if not overwrite and metrics_csv_path.exists():
        _log("overwrite=False and quality metrics already exist; skipping metric recomputation")
        metrics_out_df = pd.read_csv(metrics_csv_path, index_col=0)
        metrics_df = metrics_out_df
    else:
        _log("computing quality metrics")
        qm_params = sqm.get_default_qm_params()
        metrics_df = sqm.compute_quality_metrics(
            analyzer_split,
            metric_names=config.metric_names,
            metric_params=qm_params,
            skip_pc_metrics=config.skip_pc_metrics,
            **config.job_kwargs,
        )
        metrics_df = _merge_template_metrics_into_metrics_df(
            metrics_df,
            analyzer_split,
            template_metric_names=config.template_metric_names,
        )
        metrics_out_df = pd.DataFrame(metrics_df)
        metrics_out_df.to_csv(metrics_csv_path, index=True)

    if not overwrite and _phy_export_outputs_exist(output_folder):
        _log("overwrite=False and Phy export already exists; skipping export")
    else:
        _log("exporting to Phy")
        keep_names: set[str] = set()
        if analyzer_cache_root is not None and analyzer_cache_root.parent == output_folder:
            keep_names.add(analyzer_cache_root.name)
        if overwrite:
            _clear_folder_contents(output_folder, keep_names=keep_names)

        _export_phy_to_output_folder(
            sorting_analyzer=analyzer_split,
            output_folder=output_folder,
            analyzer_cache_root=analyzer_cache_root,
            dat_path=Path(config.dat_path) if config.dat_path is not None else None,
            hp_filtered=(not bool(config.apply_preprocess)),
            copy_binary=config.copy_binary,
            use_relative_path=bool(config.use_relative_path),
            job_kwargs=config.job_kwargs,
        )

    # export_to_phy can recreate output_folder when remove_if_exists=True.
    # Persist metrics again after export so quality_metrics.csv is always present.
    metrics_out_df.to_csv(metrics_csv_path, index=True)

    if not overwrite and (output_folder / "cluster_info.tsv").exists():
        _log("overwrite=False and cluster_info.tsv already exists; skipping noise relabel")
        n_noise_clusters = _count_noise_clusters_in_phy_dir(output_folder)
    else:
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


def run_postprocess_session(config: PostprocessConfig) -> list[PostprocessResult]:
    if config.verbose:
        print("[postprocess] start run_postprocess_session()")

    targets = _resolve_postprocess_targets(config)
    if config.verbose and config.sorting_phy_folder is None:
        search_root = _resolve_postprocess_search_root(config)
        print(
            f"[postprocess] auto-discovered {len(targets)} Kilosort folder(s) under {search_root}"
        )

    results: list[PostprocessResult] = []
    multiple_targets = len(targets) > 1
    skipped_count = 0
    for sorting_phy_folder in targets:
        target_config = _config_for_sorting_target(
            config,
            sorting_phy_folder=sorting_phy_folder,
            multiple_targets=multiple_targets,
        )
        output_folder = _resolve_postprocess_output_folder(sorting_phy_folder)
        metrics_csv_path = output_folder / target_config.metrics_csv_name
        should_skip = _should_skip_postprocess_target(
            target_config,
            output_folder=output_folder,
            metrics_csv_path=metrics_csv_path,
        )
        if config.verbose:
            status = "skip" if should_skip else "run "
            print(
                f"[postprocess] [{status} {len(results) + 1}/{len(targets)}] "
                f"{sorting_phy_folder.name} -> {output_folder.name}"
            )
        results.append(
            _run_postprocess_single_session(
                target_config,
                sorting_phy_folder=sorting_phy_folder,
            )
        )
        if should_skip:
            skipped_count += 1
    if config.verbose and len(targets) > 1:
        print(
            f"[postprocess] summary: total={len(targets)}, ran={len(targets) - skipped_count}, skipped={skipped_count}"
        )
    return results


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
