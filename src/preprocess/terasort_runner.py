"""Run TeraSort sorting modes against the existing preprocessing output."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .channel_layout import load_channel_layout, validate_channel_ids
from .io import atomic_write_json


_OPTIONS = {
    "nt": "--nt",
    "half_width": "--half-width",
    "shift_radius": "--shift-radius",
    "overlap_window_samples": "--overlap-window-samples",
    "halo_ms": "--halo-ms",
    "core_seconds": "--core-seconds",
    "shard_seconds": "--shard-seconds",
    "read_buffer_mb": "--read-buffer-mb",
    "prefetch_depth": "--prefetch-depth",
    "vram_limit_gb": "--vram-limit-gb",
    "floor_snr": "--floor-snr",
    "score_floor": "--score-floor",
    "residual_passes": "--residual-passes",
    "max_candidates": "--max-candidates",
}


_BOOL_OPTIONS = {"freeze_templates": "--freeze-templates"}
_ENUM_OPTIONS = {"preset": ("default", "fast"), "novelty": ("off", "shadow", "enroll")}
_MANAGED_KS4_SETTINGS = {"n_chan_bin", "fs", "scale", "shift", "nt", "tmin", "tmax",
                         "highpass_cutoff", "filename", "data_dir", "results_dir",
                         "probe_name", "probe_path"}


def terasort_mode(params: dict) -> str:
    mode = params.get("mode", "bounded")
    if mode not in ("bounded", "kilosort4"):
        raise ValueError("TeraSort mode must be bounded or kilosort4; original KS1 does not accept this input contract")
    return mode


def normalize_terasort_config(params: dict) -> dict:
    """Validate mode-specific controls without changing scientific defaults."""
    if not isinstance(params, dict) or any(not isinstance(key, str) for key in params):
        raise ValueError("TeraSort config must be a mapping with string keys")
    mode = terasort_mode(params)
    unknown = set(params) - (set(_OPTIONS) | set(_BOOL_OPTIONS) | set(_ENUM_OPTIONS)
                              | {"mode", "ks4_backend", "ks4_settings"})
    if unknown:
        raise ValueError(f"Unsupported TeraSort options: {sorted(unknown)}")
    backend = params.get("ks4_backend", "deep_tiled")
    if backend not in ("auto", "standard", "deep_tiled", "cublas"):
        raise ValueError("ks4_backend must be auto, standard, deep_tiled, or cublas")
    ks4 = params.get("ks4_settings", {})
    if not isinstance(ks4, dict) or any(not isinstance(key, str) for key in ks4):
        raise ValueError("ks4_settings must be a mapping with native KS4 parameter names")
    managed = set(ks4) & _MANAGED_KS4_SETTINGS
    if managed:
        raise ValueError(f"KS4 input/preprocessing settings are managed by PP/common nt: {sorted(managed)}")
    ks4 = dict(ks4)
    if mode == "bounded":
        if "nblocks" in ks4 and (type(ks4["nblocks"]) is not int or ks4["nblocks"] != 0):
            raise ValueError("Bounded KS4 seed learning requires nblocks=0")
        ks4.pop("nblocks", None)
    elif set(params) & ((set(_OPTIONS) - {"nt"}) | set(_BOOL_OPTIONS) | set(_ENUM_OPTIONS)):
        raise ValueError("Bounded controls cannot be used with mode=kilosort4")
    numeric = {key: value for key, value in params.items() if key in _OPTIONS}
    for key, value in numeric.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"TeraSort {key} must be a finite number")
        if key in ("nt", "half_width", "shift_radius", "overlap_window_samples", "read_buffer_mb",
                   "prefetch_depth", "residual_passes", "max_candidates") and not isinstance(value, int):
            raise ValueError(f"TeraSort {key} must be an integer")
        if key == "nt":
            minimum = 21 if mode == "bounded" else 1
            if value < minimum or value % 2 != 1:
                raise ValueError(f"TeraSort {mode} nt must be an odd integer >= {minimum}")
        elif key == "half_width":
            if not 3 <= value <= numeric.get("nt", 61) // 2:
                raise ValueError("TeraSort half_width must fit nt")
        elif key == "shift_radius":
            if not 0 <= value <= 8:
                raise ValueError("TeraSort shift_radius must be in [0, 8]")
        elif key == "overlap_window_samples":
            if not 0 <= value <= numeric.get("nt", 61):
                raise ValueError("TeraSort overlap_window_samples must be in [0, nt]")
        elif key == "halo_ms":
            if value < 0:
                raise ValueError("TeraSort halo_ms must be nonnegative")
        elif key == "read_buffer_mb":
            if not 0 <= value <= 1024:
                raise ValueError("TeraSort read_buffer_mb must be between 0 and 1024 MiB")
        elif key == "residual_passes":
            if not 1 <= value <= 12:
                raise ValueError("TeraSort residual_passes must be in [1, 12]")
        elif key == "score_floor":
            if not 0 < value <= 1:
                raise ValueError("TeraSort score_floor must be in (0, 1]")
        elif value <= 0 or (key == "prefetch_depth" and value > 64):
            raise ValueError(f"TeraSort {key} is outside its supported positive range")
    for key in _BOOL_OPTIONS:
        if key in params and not isinstance(params[key], bool):
            raise ValueError(f"TeraSort {key} must be boolean")
    for key, choices in _ENUM_OPTIONS.items():
        if key in params and params[key] not in choices:
            raise ValueError(f"TeraSort {key} must be one of {choices}")
    if backend == "cublas":
        if numeric.get("nt", 61) not in (41, 61) or ks4.get("n_templates", 6) != 6:
            raise ValueError("The dedicated cuBLAS backend requires nt=41 or 61 and n_templates=6")
    return {**params, "mode": mode, "ks4_backend": backend, "ks4_settings": ks4}


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
    if "zcoords" in layout:
        z = np.asarray(layout["zcoords"], dtype=float).reshape(-1)[rows]
        if not np.isfinite(z).all() or np.any(z != 0):
            raise ValueError("TeraSort integration currently requires planar XY geometry (active zcoords=0)")
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
    params = normalize_terasort_config(params)
    mode = params["mode"]
    ks4_backend = params["ks4_backend"]
    ks4_settings = params["ks4_settings"]
    if mode == "kilosort4" and resume:
        raise ValueError("TeraSort kilosort4 mode cannot resume checkpoints; choose a new output")
    option_arguments = []
    for key, value in params.items():
        if key in _OPTIONS:
            option_arguments.extend([_OPTIONS[key], str(value)])
        elif key in _BOOL_OPTIONS and value:
            option_arguments.append(_BOOL_OPTIONS[key])
        elif key in _ENUM_OPTIONS:
            option_arguments.extend(["--" + key.replace("_", "-"), value])
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
    if mode == "bounded" and output.exists() and not (native_root / "run.json").is_file():
        raise ValueError("A TeraSort Phy folder cannot be resumed without its native run.json")
    native_parent.mkdir(parents=True, exist_ok=True)
    manifest_path = native_parent / (output.name + ".manifest.json")
    job_path = native_parent / (output.name + ".job.json")
    _pin_json(manifest_path, manifest, resume=resume)
    pinned_params = dict(params)
    if mode == "bounded":
        pinned_params.pop("mode")
        if ks4_backend == "deep_tiled":
            pinned_params.pop("ks4_backend")
        if not ks4_settings:
            pinned_params.pop("ks4_settings")
    _pin_json(job_path, {"python": str(executable), "params": pinned_params,
                        "phy_output": str(output)}, resume=resume)
    command = [str(executable), "-u", "-m", "terasort.cli"]
    if mode == "bounded":
        command.extend(["session-sort", "--manifest", str(manifest_path),
                        "--output-root", str(native_root), "--phy-flat-output", str(output)])
        command.extend(option_arguments)
        if ks4_settings:
            settings_path = native_parent / (output.name + ".seed-settings.json")
            _pin_json(settings_path, ks4_settings, resume=resume)
            command.extend(["--seed-settings", str(settings_path)])
        if ks4_backend != "deep_tiled":
            command.extend(["--seed-backend", ks4_backend])
        if resume and native_exists:
            command.append("--resume")
    else:
        probe = manifest["probes"][0]
        geometry = probe["geometry"]
        settings_path = native_parent / (output.name + ".ks4-settings.json")
        probe_path = native_parent / (output.name + ".ks4-probe.json")
        settings = {**ks4_settings, "nt": params.get("nt", 61),
                    "fs": float(sampling_frequency), "n_chan_bin": int(num_channels),
                    "scale": float(gain_to_uV)}
        _pin_json(settings_path, settings, resume=False)
        _pin_json(probe_path, {"chanMap": active_channels_0based,
                              "xc": geometry["x_um"], "yc": geometry["y_um"],
                              "kcoords": geometry["shank"], "n_chan": len(active_channels_0based)},
                  resume=False)
        command.extend(["sort", "--settings", str(settings_path), "--probe-json", str(probe_path),
                        "--filename", str(Path(dat_path).resolve()), "--results-dir", str(output),
                        "--backend", ks4_backend])
        if input_is_preprocessed:
            command.append("--input-is-preprocessed")
    print(f"Running TeraSort mode={mode}, ks4_backend={ks4_backend} -> {output}", flush=True)
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
        atomic_write_json(pointer, {"mode": mode, "manifest": str(manifest_path),
                                   "job": str(job_path),
                                   **({"native_run": str(native_root)} if mode == "bounded" else {})})
    return output
