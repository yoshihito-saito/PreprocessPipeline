"""Run TeraSort's session sorter against the existing preprocessing output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .channel_layout import load_channel_layout, validate_channel_ids
from .io import atomic_write_json


_OPTIONS = {
    "core_seconds": "--core-seconds",
    "shard_seconds": "--shard-seconds",
    "read_buffer_mb": "--read-buffer-mb",
    "prefetch_depth": "--prefetch-depth",
    "vram_limit_gb": "--vram-limit-gb",
}


def _python_executable(path: Path | None) -> Path:
    if path is None:
        raise ValueError("TeraSort requires sorter_path/--terasort-path pointing to its checkout or Python executable")
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_file():
        raise FileNotFoundError(f"TeraSort Python environment not found: {path}")
    return path


def build_terasort_manifest(
    *, dat_path: Path, chanmap_path: Path, num_channels: int,
    sampling_frequency: float, gain_to_uV: float,
    active_channels_0based: list[int], input_is_preprocessed: bool,
) -> dict:
    """Pair every geometry row with its original binary column, without rewriting data."""
    dat_path = Path(dat_path).expanduser().resolve(strict=True)
    frame_bytes = int(num_channels) * 2
    source_bytes = dat_path.stat().st_size
    if frame_bytes <= 0 or source_bytes <= 0 or source_bytes % frame_bytes:
        raise ValueError("TeraSort input must contain whole, nonempty INT16 sample frames")
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("TeraSort requires a positive sampling frequency")
    if not np.isfinite(gain_to_uV) or gain_to_uV <= 0:
        raise ValueError("TeraSort requires a positive gain_to_uV in microvolts per ADC count")
    active = validate_channel_ids(active_channels_0based, num_channels, name="active_channels_0based")
    if not active or len(set(active)) != len(active):
        raise ValueError("TeraSort requires unique active binary columns")
    layout = load_channel_layout(Path(chanmap_path), num_channels)
    for key in ("xcoords", "ycoords", "kcoords"):
        if key not in layout:
            raise ValueError(f"TeraSort requires {key} in the effective chanMap")
    channel_ids = np.asarray(layout["chanMap0ind"]).reshape(-1)
    positions = {int(channel): row for row, channel in enumerate(channel_ids)}
    if any(channel not in positions for channel in active):
        raise ValueError("TeraSort geometry is missing an active binary column")
    rows = np.asarray([positions[channel] for channel in active], dtype=np.int64)
    x = np.asarray(layout["xcoords"], dtype=float).reshape(-1)[rows]
    y = np.asarray(layout["ycoords"], dtype=float).reshape(-1)[rows]
    shanks = np.asarray(layout["kcoords"], dtype=float).reshape(-1)[rows]
    probes = np.asarray(layout.get("probe_ids", np.zeros(len(channel_ids))), dtype=float).reshape(-1)[rows]
    if not np.isfinite(np.column_stack((x, y, shanks, probes))).all():
        raise ValueError("TeraSort requires finite geometry and probe/shank identifiers")
    if np.any(shanks != np.floor(shanks)) or np.any(probes != np.floor(probes)):
        raise ValueError("Probe and shank identifiers must be integers")
    # Local shank numbers may repeat across probes. Keep those groups separate
    # for median reference, seed learning, and spatial neighborhoods.
    groups: dict[tuple[int, int], int] = {}
    group_ids = []
    for probe, shank in zip(probes, shanks, strict=True):
        group_ids.append(groups.setdefault((int(probe), int(shank)), len(groups) + 1))
    return {
        "schema_version": 1,
        "session_id": "preprocess",
        "probes": [{
            "probe_id": "recording",
            "sample_rate_hz": float(sampling_frequency),
            "gain_uv_per_count": float(gain_to_uV),
            "n_channels_dat": int(num_channels),
            "channel_map": active,
            "input_is_preprocessed": bool(input_is_preprocessed),
            "geometry": {"x_um": x.tolist(), "y_um": y.tolist(), "shank": group_ids},
            "bad_channels": [],
            "segments": [{"path": str(dat_path), "start_sample": 0,
                          "n_samples": source_bytes // frame_bytes, "day_id": "session"}],
            "gaps": [],
        }],
    }


def _pin_json(path: Path, value: dict, *, resume: bool) -> None:
    if path.exists():
        if not resume or json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"TeraSort input/settings changed or already exist; choose a new output: {path}")
    else:
        atomic_write_json(path, value)


def execute_terasort_job(
    *, dat_path: Path, chanmap_path: Path | None, output_folder: Path,
    terasort_path: Path | None, sampling_frequency: float, num_channels: int,
    gain_to_uV: float, offset_to_uV: float, dtype: str,
    active_channels_0based: list[int], input_is_preprocessed: bool,
    params: dict, resume: bool = False,
) -> Path:
    """Retain native checkpoints separately and publish one flat Phy folder."""
    if np.dtype(dtype) != np.dtype("<i2"):
        raise ValueError("TeraSort session input currently requires little-endian INT16")
    if offset_to_uV != 0:
        raise ValueError("TeraSort input currently requires offset_to_uV=0")
    if chanmap_path is None:
        raise ValueError("TeraSort requires the effective chanMap.mat")
    unknown = set(params) - (set(_OPTIONS) | {"nt", "seed_backend"})
    if unknown:
        raise ValueError(f"Unsupported TeraSort options: {sorted(unknown)}")
    option_arguments = []
    for key, value in params.items():
        if key == "nt":
            if type(value) is not int or value < 21 or value % 2 != 1:
                raise ValueError("TeraSort nt must be an odd integer >=21 samples")
            option_arguments.extend(["--nt", str(value)])
            continue
        if key == "seed_backend":
            if value not in ("auto", "standard", "deep_tiled", "cublas"):
                raise ValueError("TeraSort seed_backend must be auto, standard, deep_tiled, or cublas")
            option_arguments.extend(["--seed-backend", value])
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"TeraSort {key} must be a finite number")
        if key in ("read_buffer_mb", "prefetch_depth") and not isinstance(value, int):
            raise ValueError(f"TeraSort {key} must be an integer")
        if key == "read_buffer_mb":
            if not 0 <= value <= 1024:
                raise ValueError("TeraSort read_buffer_mb must be between 0 and 1024 MiB")
        elif value <= 0 or (key == "prefetch_depth" and value > 64):
            raise ValueError(f"TeraSort {key} is outside its supported positive range")
        option_arguments.extend([_OPTIONS[key], str(value)])
    if params.get("seed_backend") == "cublas" and params.get("nt", 61) not in (41, 61):
        raise ValueError("TeraSort cuBLAS seed backend requires nt=41 or 61")
    executable = _python_executable(terasort_path)
    manifest = build_terasort_manifest(
        dat_path=dat_path, chanmap_path=chanmap_path, num_channels=num_channels,
        sampling_frequency=sampling_frequency, gain_to_uV=gain_to_uV,
        active_channels_0based=active_channels_0based,
        input_is_preprocessed=input_is_preprocessed,
    )
    output = Path(output_folder).expanduser().resolve()
    native_parent = output.parent / ".terasort"
    native_root = native_parent / output.name
    seed_root = native_root.with_name(native_root.name + ".seeds")
    native_exists = native_root.exists() or seed_root.exists()
    if (output.exists() or native_exists) and not resume:
        raise FileExistsError(f"TeraSort output already exists; use an explicit resume or new output: {output}")
    if output.exists() and not (native_root / "run.json").is_file():
        raise ValueError("A TeraSort Phy folder cannot be resumed without its native run.json")
    native_parent.mkdir(parents=True, exist_ok=True)
    manifest_path = native_parent / (output.name + ".manifest.json")
    job_path = native_parent / (output.name + ".job.json")
    _pin_json(manifest_path, manifest, resume=resume)
    _pin_json(job_path, {"python": str(executable), "params": params,
                        "phy_output": str(output)}, resume=resume)
    command = [
        str(executable), "-u", "-m", "terasort.cli", "session-sort",
        "--manifest", str(manifest_path), "--output-root", str(native_root),
        "--phy-flat-output", str(output),
    ]
    command.extend(option_arguments)
    if resume and native_exists:
        command.append("--resume")
    print(f"Running TeraSort session sorter -> {output}", flush=True)
    # Inherit the controller's CUDA visibility, process tree and streamed logs.
    subprocess.run(command, check=True)
    required = ("params.py", "spike_times.npy", "spike_clusters.npy", "templates.npy", "channel_map.npy")
    if not all((output / name).is_file() for name in required):
        raise RuntimeError(f"TeraSort did not publish a complete Phy folder: {output}")
    exported = np.load(output / "channel_map.npy", allow_pickle=False).reshape(-1)
    if not np.array_equal(exported, np.asarray(active_channels_0based)):
        raise ValueError("TeraSort export channel_map disagrees with input binary columns")
    pointer = output / "terasort_run.json"
    if not pointer.exists():
        atomic_write_json(pointer, {"native_run": str(native_root), "manifest": str(manifest_path),
                                   "job": str(job_path)})
    return output
