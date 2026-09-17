from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

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
    select_recording_channels,
)
from .metafile import PostprocessConfig, PostprocessResult
from .phy_export import write_centered_native_templates
from ..phy_metadata import read_phy_params, resolve_phy_dat_path
from ..preprocess.channel_layout import NoActiveChannels
from ..sorting_manifest import all_partitions_skipped

_SORTING_OUTPUT_PATTERNS = (
    "Kilosort_*",
    "Kilosort2_5_*",
    "Kilosort2.5_*",
    "Kilosort4_*",
)
_CANONICAL_CHANMAP_NAME = "chanMap.mat"


def _is_windows() -> bool:
    return os.name == "nt"


def _find_sorting_output_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            p
            for pattern in _SORTING_OUTPUT_PATTERNS
            for p in root.glob(pattern)
            if p.is_dir() and "_spi" not in p.name and ".preserved-" not in p.name
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _find_sorting_output_dirs_from_manifest(root: Path) -> list[Path]:
    manifest_path = root / "sorter_partition_manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[Path] = []
    seen: set[Path] = set()
    for partition in payload.get("partitions", []):
        if not isinstance(partition, dict) or partition.get("status") not in {None, "completed"}:
            continue
        folder_text = str(partition.get("output_folder") or "").strip()
        if not folder_text:
            continue
        folder = Path(folder_text).expanduser().resolve()
        try:
            relative_folder = folder.relative_to(root.resolve())
        except ValueError:
            # A manifest is an input, not authority to read arbitrary folders.
            continue
        # Partition manifests are allowed to select only direct Kilosort run
        # directories below the session root.  Do not let a malformed manifest
        # turn the session root itself (or a nested arbitrary folder) into a
        # Phy input.
        if len(relative_folder.parts) != 1 or not any(
            folder.match(pattern) for pattern in _SORTING_OUTPUT_PATTERNS
        ):
            continue
        if (
            not folder.exists()
            or not folder.is_dir()
            or "_spi" in folder.name
            or ".preserved-" in folder.name
        ):
            continue
        if folder in seen:
            continue
        candidates.append(folder)
        seen.add(folder)
    return candidates


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


def _create_postprocess_staging_folder(output_folder: Path) -> Path:
    """Allocate a same-filesystem attempt directory beside the canonical output."""
    output_folder.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{output_folder.name}.attempt-", dir=output_folder.parent)
    )


def _validate_postprocess_output(output_folder: Path, metrics_csv_name: str) -> None:
    """Reject incomplete or unreadable exports before they can become canonical."""
    required = (
        metrics_csv_name,
        "params.py",
        "spike_times.npy",
        "spike_clusters.npy",
        "cluster_group.tsv",
        "cluster_info.tsv",
    )
    missing = [name for name in required if not (output_folder / name).is_file()]
    if missing:
        raise RuntimeError(
            "Postprocess attempt did not produce a complete Phy output: "
            f"{output_folder} (missing: {', '.join(missing)})"
        )
    try:
        params_text = (output_folder / "params.py").read_text(encoding="utf-8")
        if not params_text.strip():
            raise ValueError("params.py is empty")

        spike_times = np.load(output_folder / "spike_times.npy", allow_pickle=False)
        spike_clusters = np.load(output_folder / "spike_clusters.npy", allow_pickle=False)
        if spike_times.reshape(-1).shape[0] != spike_clusters.reshape(-1).shape[0]:
            raise ValueError(
                "spike_times.npy and spike_clusters.npy have mismatched lengths: "
                f"{spike_times.reshape(-1).shape[0]} vs {spike_clusters.reshape(-1).shape[0]}"
            )

        for name, separator in (
            ("cluster_group.tsv", "\t"),
            ("cluster_info.tsv", "\t"),
            (metrics_csv_name, ","),
        ):
            table = pd.read_csv(output_folder / name, sep=separator)
            if table.empty and len(table.columns) == 0:
                raise ValueError(f"{name} has no columns")
    except Exception as exc:
        raise RuntimeError(
            "Postprocess attempt validation failed; canonical output was not modified. "
            f"Staging directory: {output_folder}. Cause: {exc}"
        ) from exc


def _publish_postprocess_attempt(
    *, output_folder: Path, staging_folder: Path, metrics_csv_name: str
) -> Path | None:
    """Atomically replace a canonical output only after validating the attempt.

    The previous canonical directory is first renamed to a versioned sibling.
    If publication of the attempt fails, that immediate prior directory is
    restored before the exception is propagated.
    """
    _validate_postprocess_output(staging_folder, metrics_csv_name)
    params = read_phy_params(staging_folder)
    dat = resolve_phy_dat_path(staging_folder)
    if dat is not None and dat.parent == staging_folder.resolve() and Path(str(params.get("dat_path", ""))).is_absolute():
        _fix_phy_params_file(
            staging_folder / "params.py", output_folder / dat.name,
            bool(params.get("hp_filtered", False)), n_channels_dat=params.get("n_channels_dat"),
            use_relative_path=False,
        )
    preserved: Path | None = None
    if output_folder.exists():
        preserved = _preserve_or_remove_postprocess_output(output_folder)
    try:
        staging_folder.rename(output_folder)
    except Exception:
        if preserved is not None and not output_folder.exists():
            preserved.rename(output_folder)
        raise
    return preserved


def _assert_postprocess_output_is_writable(
    config: PostprocessConfig,
    *,
    output_folder: Path,
    metrics_csv_path: Path,
) -> None:
    """Fail closed rather than merging a partial export into an immutable output."""
    if _postprocess_overwrite_enabled(config) or not output_folder.exists():
        return
    if _should_skip_postprocess_target(
        config, output_folder=output_folder, metrics_csv_path=metrics_csv_path
    ):
        return
    missing = [
        name
        for name in (
            config.metrics_csv_name,
            "params.py",
            "spike_times.npy",
            "spike_clusters.npy",
            "cluster_group.tsv",
            "cluster_info.tsv",
        )
        if not (output_folder / name).exists()
    ]
    detail = ", ".join(missing) if missing else "incompatible postprocess contents"
    raise FileExistsError(
        "Refusing to modify existing postprocess output with overwrite=False: "
        f"{output_folder} (missing/invalid: {detail}). Enable overwrite to rebuild it."
    )


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
        if config.sorting_search_root is not None:
            manifest_targets = _find_sorting_output_dirs_from_manifest(_resolve_postprocess_search_root(config))
            if len(manifest_targets) > 1 and sorting_phy_folder in manifest_targets:
                return manifest_targets
        return [sorting_phy_folder]

    root = _resolve_postprocess_search_root(config)
    if all_partitions_skipped(root / "sorter_partition_manifest.json"):
        return []
    manifest_candidates = _find_sorting_output_dirs_from_manifest(root)
    if manifest_candidates:
        return manifest_candidates
    if (root / "sorter_partition_manifest.json").exists():
        raise ValueError(f"No completed sorting targets in manifest under {root}; select an input explicitly")
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


def _resolve_phy_n_channels_dat(phy_dir: Path, fallback_num_channels: int | None = None) -> int | None:
    params_path = phy_dir / "params.py"
    if params_path.exists():
        text = params_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^\s*n_channels_dat\s*=\s*([0-9]+)", text)
        if match is not None:
            return int(match.group(1))
    if fallback_num_channels is not None:
        return int(fallback_num_channels)
    return None


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

    # This is the one deliberate mutation of the input Kilosort result. Retain
    # the immediate pre-mutation table on every call, so recovery never points
    # at an old or already-relabelled version.
    backup_path = cg_path.with_name(f"{cg_path.name}.pre-low-rate-backup")
    backup_sha256: str | None = None
    if cg_path.exists():
        source_bytes = cg_path.read_bytes()
        backup_sha256 = hashlib.sha256(source_bytes).hexdigest()
        fd, backup_tmp_name = tempfile.mkstemp(
            prefix=f".{backup_path.name}.", suffix=".tmp", dir=backup_path.parent
        )
        backup_tmp = Path(backup_tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(source_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            if backup_tmp.read_bytes() != source_bytes:
                raise RuntimeError(f"Failed to validate low-rate recovery backup: {backup_path}")
            os.replace(backup_tmp, backup_path)
        finally:
            backup_tmp.unlink(missing_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{cg_path.name}.", suffix=".tmp", dir=cg_path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            out_df.to_csv(handle, sep="\t", index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, cg_path)
        provenance_path = cg_path.with_name(f"{cg_path.name}.pre-low-rate-provenance.json")
        fd, provenance_tmp_name = tempfile.mkstemp(
            prefix=f".{provenance_path.name}.", suffix=".tmp", dir=provenance_path.parent
        )
        provenance_tmp = Path(provenance_tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "operation": "mark_low_firing_rate_clusters_as_noise",
                        "backup": str(backup_path) if backup_path.exists() else None,
                        "backup_sha256": backup_sha256,
                        "threshold_hz": float(threshold_hz),
                        "updated_at_unix": time.time(),
                    },
                    handle,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(provenance_tmp, provenance_path)
        finally:
            provenance_tmp.unlink(missing_ok=True)
    finally:
        tmp_path.unlink(missing_ok=True)
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


def _delete_final_analyzer_cache(path: Path, log: Callable[[str], None]) -> bool:
    """Delete postprocess cache after outputs are complete; tolerate Windows locks."""
    retries = 8 if _is_windows() else 3
    try:
        gc.collect()
        _safe_rmtree(path, retries=retries, delay=1.0)
        return True
    except PermissionError as exc:
        if not _is_windows():
            raise
        log(
            "[WARN] analyzer_cache is still locked by Windows; leaving it in place: "
            f"{path}. Close Python/Phy handles and delete it manually if disk space matters. "
            f"Original error: {exc}"
        )
        return False


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


def _preserve_or_remove_postprocess_output(folder: Path) -> Path | None:
    """Version prior output so a failed replacement cannot destroy valid results."""

    if not folder.exists():
        return None
    suffix = time.strftime("%Y-%m-%d_%H%M%S")
    preserved = folder.with_name(f"{folder.name}.preserved-{suffix}")
    counter = 1
    while preserved.exists():
        preserved = folder.with_name(f"{folder.name}.preserved-{suffix}-{counter}")
        counter += 1
    folder.rename(preserved)
    return preserved


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


def _count_units_and_spikes_from_phy_files(phy_dir: Path) -> tuple[int, int]:
    cluster_ids: set[int] = set()
    for filename in ("cluster_group.tsv", "cluster_info.tsv"):
        path = phy_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        cols_lut = {str(col).strip().lower(): col for col in df.columns}
        cluster_col = cols_lut.get("cluster_id") or cols_lut.get("id")
        if cluster_col is not None:
            cluster_ids.update(
                int(x)
                for x in pd.to_numeric(df[cluster_col], errors="coerce").dropna().astype(int).tolist()
            )
            break

    spike_clusters_path = phy_dir / "spike_clusters.npy"
    total_spikes = 0
    if spike_clusters_path.exists():
        spike_clusters = np.load(spike_clusters_path, mmap_mode="r")
        total_spikes = int(spike_clusters.shape[0])
        if not cluster_ids:
            cluster_ids.update(int(x) for x in np.unique(spike_clusters).tolist())
    return len(cluster_ids), total_spikes


def _run_noise_label_only(
    *,
    config: PostprocessConfig,
    sorting_phy_folder: Path,
    output_folder: Path,
    metrics_csv_path: Path,
    analyzer_cache_root: Path | None,
) -> PostprocessResult:
    if not output_folder.exists():
        raise FileNotFoundError(
            f"Noise labeling only requires an existing postprocess output folder: {output_folder}"
        )
    if not metrics_csv_path.exists():
        raise FileNotFoundError(
            f"Noise labeling only requires existing metrics: {metrics_csv_path}"
        )
    if not (output_folder / "cluster_si_unit_ids.tsv").exists():
        raise FileNotFoundError(
            f"Noise labeling only requires cluster_si_unit_ids.tsv in: {output_folder}"
        )

    metrics_df = pd.read_csv(metrics_csv_path, index_col=0)
    updated = mark_noise_clusters_from_metrics(
        phy_dir=output_folder,
        metrics_df=metrics_df,
        thresholds=config.noise_thresholds,
        backup=config.noise_backup,
        reset_to_unsorted=True,
        update_cluster_info=True,
    )
    n_units_initial, total_spikes_initial = _count_units_and_spikes_from_phy_files(sorting_phy_folder)
    n_units_final, total_spikes_final = _count_units_and_spikes_from_phy_files(output_folder)
    return PostprocessResult(
        sorting_phy_folder=sorting_phy_folder,
        output_folder=output_folder,
        preprocessed_dat_path=None,
        metrics_csv_path=metrics_csv_path,
        analyzer_cache_dir=(
            analyzer_cache_root
            if analyzer_cache_root is not None and analyzer_cache_root.exists() and not config.delete_analyzer_cache
            else None
        ),
        n_units_initial=n_units_initial,
        n_units_final=n_units_final,
        total_spikes_initial=total_spikes_initial,
        total_spikes_final=total_spikes_final,
        n_noise_clusters=int((updated["group"] == "noise").sum()),
    )


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
    if isinstance(template_metrics, pd.DataFrame):
        template_df = template_metrics
    elif hasattr(template_metrics, "get_data"):
        template_df = pd.DataFrame(template_metrics.get_data())
    else:
        template_extension = analyzer.get_extension("template_metrics")
        if template_extension is None:
            return pd.DataFrame(metrics_df)
        template_df = pd.DataFrame(template_extension.get_data())
    if template_df.empty:
        return pd.DataFrame(metrics_df)

    merged = pd.DataFrame(metrics_df).join(template_df, how="left")
    for column_name in ("peak_to_valley", "half_width"):
        if column_name in merged.columns:
            merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce") * 1000.0
    repol = pd.to_numeric(merged.get("repolarization_slope"), errors="coerce")
    recovery = pd.to_numeric(merged.get("recovery_slope"), errors="coerce")
    if repol is not None and recovery is not None:
        merged["slope"] = pd.concat((repol.abs(), recovery.abs()), axis=1).min(axis=1) / 1000.0
    return merged


def _sorting_analyzer_sparsity_kwargs(config: PostprocessConfig) -> dict[str, Any]:
    if not config.analyzer_sparse:
        return {"sparse": False}

    kwargs: dict[str, Any] = {
        "sparse": True,
        "method": config.sparsity_method,
    }
    if config.sparsity_method == "radius":
        kwargs["radius_um"] = float(config.sparsity_radius_um)
    elif config.sparsity_method in {"best_channels", "closest_channels"}:
        kwargs["num_channels"] = int(config.sparsity_num_channels)
    return kwargs


def _resolve_effective_chanmap_for_postprocess(
    chanmap_mat_path: Path | None,
    *,
    local_output_dir: Path | None = None,
    dat_path: Path | None = None,
) -> Path | None:
    if chanmap_mat_path is not None:
        path = Path(chanmap_mat_path)
        if not path.is_file():
            raise FileNotFoundError(f"Selected chanMap does not exist: {path}")
        return path
    candidates: list[Path] = []
    if local_output_dir is not None:
        candidates.append(Path(local_output_dir) / _CANONICAL_CHANMAP_NAME)
    if dat_path is not None:
        candidates.append(Path(dat_path).parent / _CANONICAL_CHANMAP_NAME)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return chanmap_mat_path


def _resolve_recording_for_postprocess(config: PostprocessConfig):
    if config.recording is None and config.dat_path is None:
        raise ValueError("Either recording or dat_path must be provided")

    if config.recording is None and (config.sampling_frequency is None or config.num_channels is None):
        raise ValueError("sampling_frequency and num_channels are required when dat_path is used")

    if config.recording is not None:
        rec_raw = config.recording
    else:
        width_bytes = int(config.num_channels) * np.dtype(config.dtype).itemsize
        size = Path(config.dat_path).stat().st_size - config.binary_offset
        if width_bytes <= 0 or size <= 0 or size % width_bytes:
            raise ValueError("Binary size/offset is incompatible with the configured channel count and dtype")
        rec_raw = se.read_binary(
            str(Path(config.dat_path)),
            sampling_frequency=float(config.sampling_frequency),
            dtype=config.dtype,
            num_channels=int(config.num_channels),
            gain_to_uV=config.gain_to_uV,
            offset_to_uV=config.offset_to_uV,
            file_offset=config.binary_offset,
        )
    if config.num_channels is not None and hasattr(rec_raw, "annotate"):
        rec_raw.annotate(binary_num_channels=int(config.num_channels))
    rejects = set(config.reject_channels)
    if config.xml_path is not None:
        from ..preprocess.io import load_session_xml_metadata
        xml = load_session_xml_metadata(Path(config.xml_path))
        rejects.update(xml.skipped_channels_0based)
        if xml.spike_groups_0based:
            included = {ch for group in xml.spike_groups_0based for ch in group}
            rejects.update(int(ch) for ch in rec_raw.get_channel_ids() if int(ch) not in included)
    rec_with_probe, bad_0, _ = attach_probe_and_remove_bad_channels(
        recording=rec_raw,
        chanmap_mat_path=_resolve_effective_chanmap_for_postprocess(
            config.chanmap_mat_path,
            dat_path=Path(config.dat_path) if config.dat_path is not None else None,
        ),
        reject_channels_0based=sorted(rejects),
        original_num_channels=config.num_channels,
    )
    if hasattr(rec_with_probe, "get_channel_ids"):
        bad_set = {int(ch) for ch in bad_0}
        all_channels = [int(ch) for ch in rec_with_probe.get_channel_ids()]
        good_channels = [ch for ch in all_channels if ch not in bad_set]
        if config.sorting_phy_folder is not None:
            folder = _resolve_sorting_run_root(Path(config.sorting_phy_folder).resolve())
            manifest = folder.parent / "sorter_partition_manifest.json"
            if manifest.exists():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                for item in payload.get("partitions", []):
                    if item.get("output_folder") and Path(item["output_folder"]).resolve() == folder:
                        scope = {int(ch) for ch in item["channels_0based"]}
                        good_channels = [ch for ch in good_channels if ch in scope]
                        break
    else:
        good_channels = []

    if not good_channels:
        raise NoActiveChannels("All recording channels are excluded; postprocess skipped")

    if config.apply_preprocess:
        if good_channels:
            rec_processed = preprocess_selected_channels_preserve_shape(
                recording_raw=rec_with_probe,
                selected_channel_ids=good_channels,
                bandpass_min_hz=config.bandpass_min_hz,
                bandpass_max_hz=config.bandpass_max_hz,
                reference=config.reference,
                local_radius_um=config.local_radius_um,
            )
        else:
            rec_processed = apply_preprocessing(
                recording_raw=rec_with_probe,
                bandpass_min_hz=config.bandpass_min_hz,
                bandpass_max_hz=config.bandpass_max_hz,
                reference=config.reference,
                local_radius_um=config.local_radius_um,
            )
    else:
        rec_processed = rec_with_probe

    if good_channels:
        return select_recording_channels(rec_processed, good_channels)
    return rec_processed


def _fix_phy_params_file(
    params_file: Path,
    dat_path: Path,
    hp_filtered: bool,
    *,
    n_channels_dat: int | None,
    use_relative_path: bool,
    dtype: str | None = None,
    sample_rate: float | None = None,
    offset: int | None = None,
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
    dat_line = f"dat_path = {dat_value!r}"
    if re.search(r"(?m)^dat_path\s*=", content):
        content = re.sub(r"(?m)^dat_path\s*=.*$", lambda _: dat_line, content)
    else:
        content += f"\n{dat_line}\n"

    hp_line = f"hp_filtered = {str(bool(hp_filtered))}"
    if re.search(r"(?m)^hp_filtered\s*=", content):
        content = re.sub(r"(?m)^hp_filtered\s*=.*$", lambda _: hp_line, content)
    else:
        content += f"\n{hp_line}\n"

    if n_channels_dat is not None:
        n_channels_line = f"n_channels_dat = {int(n_channels_dat)}"
        if re.search(r"(?m)^n_channels_dat\s*=", content):
            content = re.sub(r"(?m)^n_channels_dat\s*=.*$", lambda _: n_channels_line, content)
        else:
            content += f"\n{n_channels_line}\n"
    for key, value in (("dtype", dtype), ("sample_rate", sample_rate), ("offset", offset)):
        if value is not None:
            line = f"{key} = {value!r}"
            if re.search(rf"(?m)^{key}\s*=", content):
                content = re.sub(rf"(?m)^{key}\s*=.*$", lambda _: line, content)
            else:
                content += f"\n{line}\n"
    params_file.write_text(content, encoding="utf-8")


def _fix_phy_channel_map_file(output_folder: Path) -> None:
    channel_map_si_file = output_folder / "channel_map_si.npy"
    channel_map_file = output_folder / "channel_map.npy"
    if not channel_map_si_file.exists() or not channel_map_file.exists():
        return

    channel_map_si = np.load(channel_map_si_file)
    np.save(channel_map_file, np.asarray(channel_map_si, dtype="int32"))


def _export_phy_to_output_folder(
    *,
    sorting_analyzer,
    output_folder: Path,
    analyzer_cache_root: Path | None,
    dat_path: Path | None,
    hp_filtered: bool,
    raw_num_channels: int | None,
    copy_binary: bool,
    use_relative_path: bool,
    job_kwargs: dict[str, Any],
    binary_dtype: str | None = None,
    binary_offset: int = 0,
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
    write_centered_native_templates(sorting_analyzer, phy_export_tmp, job_kwargs)
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
    if copy_binary:
        _fix_phy_params_file(
            params_file, output_folder / "recording.dat", sorting_analyzer.is_filtered(),
            n_channels_dat=sorting_analyzer.get_num_channels(),
            use_relative_path=bool(use_relative_path),
            dtype=np.dtype(sorting_analyzer.recording.get_dtype()).str,
            sample_rate=float(sorting_analyzer.sampling_frequency), offset=0,
        )
    elif dat_path is not None:
        _fix_phy_params_file(
            params_file=params_file,
            dat_path=Path(dat_path),
            hp_filtered=hp_filtered,
            n_channels_dat=raw_num_channels,
            use_relative_path=bool(use_relative_path),
            dtype=binary_dtype,
            sample_rate=float(sorting_analyzer.sampling_frequency) if hasattr(sorting_analyzer, "sampling_frequency") else None,
            offset=binary_offset,
        )
        _fix_phy_channel_map_file(output_folder)
        channel_map = np.load(output_folder / "channel_map.npy")
        if raw_num_channels is None or np.any(channel_map < 0) or np.any(channel_map >= raw_num_channels):
            raise ValueError("Phy channel IDs do not address columns of the referenced binary")


def _config_for_sorting_target(
    config: PostprocessConfig,
    *,
    sorting_phy_folder: Path,
    multiple_targets: bool,
) -> PostprocessConfig:
    config = replace(config, sorting_phy_folder=sorting_phy_folder)
    if not multiple_targets or config.analyzer_cache_dir is None:
        return config
    return replace(
        config,
        analyzer_cache_dir=(Path(config.analyzer_cache_dir).resolve() / _resolve_sorting_run_root(sorting_phy_folder).name),
    )


def _analyzer_input_identity(recording, sorting_folder: Path, config: PostprocessConfig) -> str:
    """Bind reusable features to their lazy signal graph and original sorting."""
    def normalize(value):
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return normalize(value.tolist())
            return {"shape": value.shape, "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            value = str(value.resolve())
        if isinstance(value, str) and Path(value).is_absolute() and Path(value).is_file():
            stat = Path(value).stat()
            return {"path": str(Path(value).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return value

    source_files = {}
    # Phy extractors also accept CSV labels and arbitrary CSV/TSV properties.
    names = {"spike_times.npy", "spike_clusters.npy", "spike_templates.npy"}
    names.update(path.name for suffix in ("*.csv", "*.tsv") for path in sorting_folder.glob(suffix))
    for name in sorted(names):
        path = sorting_folder / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            source_files[name] = digest.hexdigest()
    payload = normalize({
        "recording": recording.to_dict(recursive=True, include_properties=True),
        "source_sorting": str(sorting_folder.resolve()),
        "source_files": source_files,
        "exclude_cluster_groups": config.exclude_cluster_groups,
        "low_rate_threshold_hz": config.noise_thresholds.get("firing_rate_lt", 0.01),
        "n_components": config.n_components,
        "pc_mode": config.pc_mode,
    })
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_analyzer_cache_input(folder: Path, identity: str) -> None:
    path = folder / "pipeline_input_identity.json"
    if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("identity") != identity:
        raise ValueError(
            "Cached analyzer input does not match or has no verified provenance. "
            "Rebuild with skip_curation=False and overwrite=True for the selected recording/sorting."
        )


def _run_postprocess_single_session_impl(
    config: PostprocessConfig,
    *,
    sorting_phy_folder: Path,
    attempt_state: dict[str, Path],
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
    if config.noise_label_only:
        _log(f"noise labeling only: sorting_phy_folder={sorting_phy_folder}")
        return _run_noise_label_only(
            config=config,
            sorting_phy_folder=sorting_phy_folder,
            output_folder=output_folder,
            metrics_csv_path=metrics_csv_path,
            analyzer_cache_root=analyzer_cache_root,
        )
    recording_for_post = (_resolve_recording_for_postprocess(config)
                          if config.dat_path is not None or config.recording is not None else None)
    source_params = read_phy_params(sorting_phy_folder)
    rate = source_params.get("sample_rate")
    if recording_for_post is not None and rate is not None and not np.isclose(
        float(rate), recording_for_post.get_sampling_frequency(), rtol=1e-5, atol=1e-2
    ):
        raise ValueError("Sorting and recording sampling rates disagree")
    spike_file = sorting_phy_folder / "spike_times.npy"
    if recording_for_post is not None and spike_file.exists():
        samples = np.load(spike_file, mmap_mode="r", allow_pickle=False)
        if samples.size and (samples.min() < 0 or samples.max() >= recording_for_post.get_num_samples()):
            raise ValueError("Sorting spike samples fall outside the selected recording timeline")
    if config.dat_path is not None and resolve_phy_dat_path(sorting_phy_folder) == Path(config.dat_path).resolve():
        expected = {"n_channels_dat": config.num_channels, "dtype": np.dtype(config.dtype),
                    "offset": config.binary_offset}
        for key, value in expected.items():
            actual = source_params.get(key)
            if key == "dtype" and actual is not None:
                actual = np.dtype(actual)
            if key in source_params and value is not None and actual != value:
                raise ValueError(f"Conflicting {key} metadata for the same binary")
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

    _assert_postprocess_output_is_writable(
        config, output_folder=output_folder, metrics_csv_path=metrics_csv_path
    )
    if (config.skip_curation or not overwrite) and config.analyzer_format == "binary_folder" and config.analyzer_cache_dir is not None:
        split_folder = analyzer_cache_root / "split"
        if split_folder.exists():
            _validate_analyzer_cache_input(split_folder, _analyzer_input_identity(recording_for_post, sorting_phy_folder, config))

    _log(f"sorting_phy_folder={sorting_phy_folder}")
    canonical_output_folder = output_folder
    output_folder = _create_postprocess_staging_folder(canonical_output_folder)
    attempt_state["staging_folder"] = output_folder
    metrics_csv_path = output_folder / config.metrics_csv_name
    # All derived Phy files, metrics, and classification tables are written
    # inside this attempt directory. The canonical *_spi directory remains
    # untouched until a complete attempt has been validated and published.
    _log(f"staging_output_folder={output_folder}")
    if analyzer_cache_root is not None:
        if config.analyzer_cache_dir is None:
            analyzer_cache_root = output_folder / "analyzer_cache"
        elif analyzer_cache_root.exists() and overwrite and not config.skip_curation:
            _safe_rmtree(analyzer_cache_root)
        analyzer_cache_root.mkdir(parents=True, exist_ok=True)
        _log(f"analyzer_cache_dir={analyzer_cache_root}")

    # Container to track intermediate analyzers for cleanup before export
    _intermediate_analyzers: list = []

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
    # Validate before changing source cluster labels or loading a cached analyzer.
    recording = _ensure_recording_for_post()
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
            amp_mad_scale=config.split_amp_mad_scale,
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
        if config.analyzer_format == "binary_folder" and analyzer_cache_root is not None:
            identity_path = analyzer_cache_root / "split" / "pipeline_input_identity.json"
            identity_path.write_text(json.dumps({"identity": _analyzer_input_identity(recording, sorting_phy_folder, config)}), encoding="utf-8")

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
            raw_num_channels=config.num_channels,
            copy_binary=config.copy_binary,
            use_relative_path=bool(config.use_relative_path),
            job_kwargs=config.job_kwargs,
            binary_dtype=config.dtype,
            binary_offset=config.binary_offset,
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
        _intermediate_analyzers.clear()
        try:
            del analyzer_split
        except NameError:
            pass
        try:
            del analyzer_merged
        except NameError:
            pass
        try:
            del analyzer
        except NameError:
            pass
        gc.collect()
        analyzer_cache_deleted = _delete_final_analyzer_cache(analyzer_cache_root, _log)
        analyzer_cache_for_result = None if analyzer_cache_deleted else analyzer_cache_root
    else:
        analyzer_cache_for_result = analyzer_cache_root

    _log(
        f"done: n_units_final={n_units_final}, total_spikes_final={total_spikes_final}, "
        f"n_noise_clusters={n_noise_clusters}"
    )
    preserved = _publish_postprocess_attempt(
        output_folder=canonical_output_folder,
        staging_folder=output_folder,
        metrics_csv_name=config.metrics_csv_name,
    )
    attempt_state.pop("staging_folder", None)
    if preserved is not None:
        _log(f"versioned prior output after successful replacement: {preserved}")
    if config.analyzer_cache_dir is None and analyzer_cache_for_result is not None:
        analyzer_cache_for_result = canonical_output_folder / analyzer_cache_for_result.name
    return PostprocessResult(
        sorting_phy_folder=sorting_phy_folder,
        output_folder=canonical_output_folder,
        preprocessed_dat_path=preprocessed_dat_path,
        metrics_csv_path=canonical_output_folder / config.metrics_csv_name,
        analyzer_cache_dir=analyzer_cache_for_result,
        n_units_initial=n_units_initial,
        n_units_final=n_units_final,
        total_spikes_initial=total_spikes_initial,
        total_spikes_final=total_spikes_final,
        n_noise_clusters=n_noise_clusters,
    )


def _run_postprocess_single_session(
    config: PostprocessConfig,
    *,
    sorting_phy_folder: Path,
) -> PostprocessResult:
    """Run one target, removing failed attempt artifacts without touching canonical output."""
    attempt_state: dict[str, Path] = {}
    try:
        return _run_postprocess_single_session_impl(
            config,
            sorting_phy_folder=sorting_phy_folder,
            attempt_state=attempt_state,
        )
    except Exception:
        staging_folder = attempt_state.get("staging_folder")
        if staging_folder is not None and staging_folder.exists():
            _safe_rmtree(staging_folder)
        raise


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
        try:
            result = _run_postprocess_single_session(
                target_config,
                sorting_phy_folder=sorting_phy_folder,
            )
        except NoActiveChannels as exc:
            skipped_count += 1
            if config.verbose:
                print(f"[postprocess] skip {sorting_phy_folder.name}: {exc}")
            continue
        results.append(result)
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

    return _resolve_recording_for_postprocess(PostprocessConfig(
        dat_path=Path(result.dat_path),
        sampling_frequency=result.sr,
        dtype=preprocess_config.dtype,
        num_channels=result.n_channels,
        gain_to_uV=preprocess_config.gain_to_uV,
        offset_to_uV=preprocess_config.offset_to_uV,
        chanmap_mat_path=_resolve_effective_chanmap_for_postprocess(
            (Path(result.local_output_dir) / _CANONICAL_CHANMAP_NAME
             if (Path(result.local_output_dir) / _CANONICAL_CHANMAP_NAME).exists()
             else preprocess_config.chanmap_mat_path),
            local_output_dir=result.local_output_dir,
            dat_path=result.dat_path,
        ),
        reject_channels=_resolve_bad_channels_for_postprocess(result, preprocess_config),
        # The main preprocessing pipeline already wrote its final signal here.
        apply_preprocess=not preprocess_config.do_preprocess,
        bandpass_min_hz=preprocess_config.bandpass_min_hz,
        bandpass_max_hz=preprocess_config.bandpass_max_hz,
        reference=preprocess_config.reference,
        local_radius_um=preprocess_config.local_radius_um,
    ))


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
