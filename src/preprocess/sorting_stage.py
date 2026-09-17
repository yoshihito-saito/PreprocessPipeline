from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import uuid
from pathlib import Path
from typing import Any, Callable

from .metafile import PreprocessConfig, PreprocessResult
from .sorter_runner import (
    build_sorter_partitions,
    execute_sorting_job,
    write_sorter_partition_manifest,
)


def sorter_output_prefix(sorter_name: str) -> str:
    name = str(sorter_name).strip()
    normalized = name.lower()
    if normalized == "kilosort":
        return "Kilosort"
    if normalized == "kilosort4":
        return "Kilosort4"
    return name[:1].upper() + name[1:].lower()


def _resolved_xml_path(config: PreprocessConfig, result: PreprocessResult) -> Path:
    candidates = [
        Path(result.local_output_dir) / f"{result.basename}.xml",
        Path(config.xml_path) if config.xml_path is not None else None,
        Path(result.basepath) / f"{result.basename}.xml",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Sorting requires the session XML for {result.basename}")


def _resolved_chanmap_path(config: PreprocessConfig, result: PreprocessResult) -> Path | None:
    candidates = [
        Path(result.local_output_dir) / "chanMap.mat",
        Path(config.chanmap_mat_path) if config.chanmap_mat_path is not None else None,
        Path(result.basepath) / "chanMap.mat",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    if config.chanmap_mat_path is not None:
        raise FileNotFoundError(f"Selected chanMap does not exist: {config.chanmap_mat_path}")
    return None


def run_sorting_stage(
    config: PreprocessConfig,
    preprocess_result: PreprocessResult,
    *,
    output_timestamp: str | None = None,
    execute_job: Callable[..., Path] = execute_sorting_job,
) -> PreprocessResult:
    """Run only the sorter portion of the existing preprocess workflow."""
    if not config.sorter:
        return replace(
            preprocess_result,
            sorter=None,
            sorter_output_dir=None,
            sorter_output_dirs=[],
            sorter_partition_manifest_path=None,
        )
    if preprocess_result.dat_path is None or not Path(preprocess_result.dat_path).exists():
        raise FileNotFoundError("Sorting requires the completed preprocess .dat output")

    output_dir = Path(preprocess_result.local_output_dir).resolve()
    sorter_dat_path = Path(preprocess_result.dat_path).resolve()
    xml_path = _resolved_xml_path(config, preprocess_result)
    chanmap_path = _resolved_chanmap_path(config, preprocess_result)
    bad_0 = sorted(set(int(value) for value in preprocess_result.bad_channels_0based))
    sorter_input_is_preprocessed = bool(config.do_preprocess)
    sorter_exclude_channels_0based = (
        bad_0 if config.bad_channels and bad_0 else None
    )
    partition_excluded = bad_0 if config.bad_channels else []

    sorter_label = str(config.sorter).strip()
    partition_mode = str(config.sorter_partition_mode).strip().lower()
    if partition_mode not in {"all", "probe", "shank"}:
        raise ValueError(
            "sorter_partition_mode must be one of ['all', 'probe', 'shank']. "
            f"Got: {config.sorter_partition_mode}"
        )
    partitions = build_sorter_partitions(
        mode=partition_mode,
        chanmap_mat_path=chanmap_path,
        num_channels=int(preprocess_result.n_channels),
        excluded_channels_0based=partition_excluded,
    )
    timestamp = output_timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    manifest_partitions = []
    sorter_output_dirs: list[Path] = []
    manifest_path: Path | None = None
    attempt_manifest_name = f"sorter_partition_manifest.attempt-{timestamp}-{uuid.uuid4().hex[:8]}.json"
    for partition in partitions:
        if partition.status == "skipped":
            print(f"Skipping sorter partition {partition.name}: {partition.skip_reason}")
            manifest_partitions.append(partition)
            continue
        suffix = "" if partition.mode == "all" else f"_{partition.name}"
        output_folder = output_dir / f"{sorter_output_prefix(sorter_label)}_{timestamp}{suffix}"
        if output_folder.exists():
            counter = 1
            while (candidate := output_dir / f"{sorter_output_prefix(sorter_label)}_{timestamp}{suffix}_{counter:02d}").exists():
                counter += 1
            output_folder = candidate
        print(
            f"Sorter partition {partition.name}: "
            f"{len(partition.channels_0based)} channels -> {output_folder}"
        )
        current = replace(partition, output_folder=str(output_folder), status="running")
        manifest_partitions.append(current)
        write_sorter_partition_manifest(
            output_dir=output_dir,
            mode=partition_mode,
            sorter=sorter_label,
            partitions=manifest_partitions,
            filename=attempt_manifest_name,
        )
        execute_job(
            sorter=sorter_label,
            dat_path=sorter_dat_path,
            xml_path=xml_path,
            output_folder=output_folder,
            config_path=config.sorter_config_path,
            kilosort1_path=config.sorter_path,
            kilosort25_path=config.sorter_path,
            kilosort4_path=config.sorter_path,
            matlab_path=config.matlab_path,
            matlab_max_workers=config.matlab_max_workers,
            chanmap_mat_path=chanmap_path,
            dtype=config.dtype,
            gain_to_uV=config.gain_to_uV,
            offset_to_uV=config.offset_to_uV,
            sampling_frequency=float(preprocess_result.sr),
            num_channels=int(preprocess_result.n_channels),
            active_channels_0based=(
                partition.channels_0based if partition.mode != "all" else None
            ),
            exclude_channels_0based=(
                partition_excluded
                if partition.mode != "all"
                else sorter_exclude_channels_0based
            ),
            job_kwargs=config.job_kwargs,
            remove_existing_folder=config.overwrite,
            preprocess_for_sorting=False,
            input_is_preprocessed=sorter_input_is_preprocessed,
            bandpass_min_hz=config.bandpass_min_hz,
            bandpass_max_hz=config.bandpass_max_hz,
            reference=config.reference,
            local_radius_um=config.local_radius_um,
            sorter_verbose=bool(config.sorter_verbose),
            cleanup_temp_wh=bool(config.cleanup_temp_wh),
        )
        sorter_output_dirs.append(output_folder)
        manifest_partitions[-1] = replace(current, status="completed")
        write_sorter_partition_manifest(
            output_dir=output_dir,
            mode=partition_mode,
            sorter=sorter_label,
            partitions=manifest_partitions,
            filename=attempt_manifest_name,
        )

    if manifest_partitions:
        manifest_path = write_sorter_partition_manifest(
            output_dir=output_dir,
            mode=partition_mode,
            sorter=sorter_label,
            partitions=manifest_partitions,
        )

    sorter_output_dir = sorter_output_dirs[0] if len(sorter_output_dirs) == 1 else None
    if sorter_output_dirs:
        print("Sorter output folders: " + ", ".join(str(path) for path in sorter_output_dirs))
    if manifest_path is not None:
        print(f"Sorter partition manifest: {manifest_path}")
    return replace(
        preprocess_result,
        sorter=sorter_label,
        sorter_output_dir=sorter_output_dir,
        sorter_output_dirs=sorter_output_dirs,
        sorter_partition_manifest_path=manifest_path,
    )
