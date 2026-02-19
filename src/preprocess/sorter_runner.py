from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any
import ast

import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import yaml
from scipy.io import loadmat

from .io import load_xml_metadata
from .recording import attach_probe_from_chanmap


_SORTER_ALIASES = {
    "kilosort": "kilosort",
    "kilosort4": "kilosort4",
}

_INSTALLED_SORTERS = tuple(_SORTER_ALIASES.keys())

_KILOSORT1_WRAPPER_ALIASES = {
    "GPU": "useGPU",
    "fshigh": "freq_min",
    "fslow": "freq_max",
    "spkTh": "detect_threshold",
}

_KILOSORT1_BOOL_PARAMS = {
    "car",
    "useGPU",
    "delete_recording_dat",
    "parfor",
    "shuffle_clusters",
    "progress_bar",
}

_KILOSORT1_FLOAT_VECTOR_PARAMS = {
    "loc_range",
    "long_range",
    "Th",
    "lam",
    "momentum",
}

_KILOSORT1_ALLOWED_PARAM_KEYS: set[str] | None = None


def _prepend_to_path(path_to_add: str) -> None:
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = path_to_add + os.pathsep + existing_path


def _prepend_to_windows_path(path_to_add: str) -> None:
    _prepend_to_path(path_to_add)
    existing_path_alt = os.environ.get("Path", "")
    os.environ["Path"] = path_to_add + os.pathsep + existing_path_alt


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sorter_config_path(sorter: str) -> Path:
    s = sorter.lower()
    if s == "kilosort":
        return _repo_root() / "sorter" / "Kilosort1_config.yaml"
    if s == "kilosort4":
        return _repo_root() / "sorter" / "Kilosort4_config.yaml"
    raise ValueError(f"Unsupported sorter: {sorter}")


def _default_kilosort1_path() -> Path:
    return _repo_root() / "sorter" / "Kilosort1"


def _resolve_matlab_cmd(matlab_path: Path | None) -> str | None:
    if matlab_path is not None:
        mp = Path(matlab_path).expanduser().resolve()
        if mp.is_file() and mp.name.lower() in {"matlab", "matlab.exe"}:
            return str(mp)
        if mp.is_dir():
            candidates = (
                mp / "matlab",
                mp / "matlab.exe",
                mp / "bin" / "matlab",
                mp / "bin" / "matlab.exe",
            )
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
            raise FileNotFoundError(
                "MATLAB executable was not found under directory: "
                f"{mp}. Checked: " + ", ".join(str(c) for c in candidates)
            )
        raise FileNotFoundError(
            f"Invalid matlab_path: {mp}. Provide MATLAB executable or MATLAB/bin directory."
        )

    matlab_cmd = shutil.which("matlab")
    if matlab_cmd is not None:
        return matlab_cmd

    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        root = Path(program_files) / "MATLAB"
        if root.exists():
            versions = sorted([d for d in root.iterdir() if d.is_dir()], reverse=True)
            for vdir in versions:
                candidate = vdir / "bin" / "matlab.exe"
                if candidate.exists():
                    return str(candidate)
    return None


def _inject_matlab_shim(matlab_cmd: str, matlab_log: Path) -> Path:
    shim_dir = _repo_root() / ".matlab_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    startup_m = shim_dir / "startup.m"
    startup_m.write_text(
        (
            "try\n"
            "    parallel.gpu.enableCUDAForwardCompatibility(true);\n"
            "catch\n"
            "end\n"
        ),
        encoding="utf-8",
    )
    existing_matlabpath = os.environ.get("MATLABPATH", "")
    os.environ["MATLABPATH"] = (
        str(shim_dir) if not existing_matlabpath else str(shim_dir) + os.pathsep + existing_matlabpath
    )

    if os.name == "nt":
        shim_path = shim_dir / "matlab.bat"
        shim_path.write_text(
            f'@echo off\r\n"{matlab_cmd}" -logfile "{matlab_log}" %*\r\n',
            encoding="utf-8",
        )
        _prepend_to_windows_path(str(shim_dir))
        return shim_path

    shim_path = shim_dir / "matlab"
    shim_path.write_text(
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "args=(\"$@\")\n"
            "for ((i=0; i<${#args[@]}; i++)); do\n"
            "  if [[ \"${args[$i]}\" == \"-r\" ]] && (( i + 1 < ${#args[@]} )); then\n"
            "    args[$((i + 1))]=\"${args[$((i + 1))]}; try, delete(gcp('nocreate')); catch, end; exit\"\n"
            "    break\n"
            "  fi\n"
            "done\n"
            f'exec "{matlab_cmd}" -logfile "{matlab_log}" "${{args[@]}}"\n'
        ),
        encoding="utf-8",
    )
    shim_path.chmod(0o755)
    _prepend_to_path(str(shim_dir))
    return shim_path


def _load_params(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {path}. Use .yaml/.yml/.json")
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    _validate_no_inf_strings(params, path)
    return params


def _validate_no_inf_strings(obj: Any, cfg_path: Path, parent_key: str = "") -> None:
    inf_tokens = {"inf", "+inf", "infinity", "+infinity", ".inf", "-inf", "-infinity", "-.inf"}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{parent_key}.{k}" if parent_key else str(k)
            _validate_no_inf_strings(v, cfg_path, key)
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
            _validate_no_inf_strings(v, cfg_path, key)
        return
    if isinstance(obj, str) and obj.strip().lower() in inf_tokens:
        raise ValueError(
            f"Invalid string infinity token at '{parent_key}' in {cfg_path}. "
            "Use YAML numeric infinity literal (.inf) instead of quoted string."
        )


def _resolve_sr_nch(
    sampling_frequency: float | None,
    num_channels: int | None,
    xml_path: Path | None,
) -> tuple[float, int]:
    if sampling_frequency is not None and num_channels is not None:
        return float(sampling_frequency), int(num_channels)

    if xml_path is None:
        raise ValueError("sampling_frequency/num_channels or xml_path is required")

    meta = load_xml_metadata(xml_path)
    sr = float(sampling_frequency) if sampling_frequency is not None else float(meta.sr)
    nch = int(num_channels) if num_channels is not None else int(meta.n_channels)
    return sr, nch


def _safe_eval_int_expr(expr: str, variables: dict[str, int]) -> int:
    node = ast.parse(expr, mode="eval")

    def _eval(n: ast.AST) -> int:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return int(n.value)
        if isinstance(n, ast.Name):
            if n.id not in variables:
                raise ValueError(f"Unknown variable in expression: {n.id}")
            return int(variables[n.id])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return +val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod)):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, (ast.Div, ast.FloorDiv)):
                return left // right
            if isinstance(n.op, ast.Mod):
                return left % right
        raise ValueError(f"Unsupported expression: {expr}")

    return int(_eval(node))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _get_kilosort1_allowed_param_keys() -> set[str]:
    global _KILOSORT1_ALLOWED_PARAM_KEYS
    if _KILOSORT1_ALLOWED_PARAM_KEYS is not None:
        return _KILOSORT1_ALLOWED_PARAM_KEYS
    try:
        keys = set(ss.KilosortSorter.default_params().keys())
    except Exception:
        keys = set()
    _KILOSORT1_ALLOWED_PARAM_KEYS = keys
    return keys


def _get_active_channel_count(chanmap_mat_path: Path | None, num_channels: int) -> int:
    if chanmap_mat_path is None or not chanmap_mat_path.exists():
        return int(num_channels)
    try:
        mat = loadmat(str(chanmap_mat_path), simplify_cells=True)
        connected = mat.get("connected")
        if connected is None:
            return int(num_channels)
        connected_arr = (connected > 0).astype(int)
        active = int(connected_arr.sum())
        return active if active > 0 else int(num_channels)
    except Exception:
        return int(num_channels)


def _normalize_kilosort_params(
    params: dict[str, Any],
    *,
    num_channels: int,
    chanmap_mat_path: Path | None,
) -> dict[str, Any]:
    normalized = {
        _KILOSORT1_WRAPPER_ALIASES.get(key, key): value
        for key, value in params.items()
    }

    if "detect_threshold" in normalized:
        normalized["detect_threshold"] = abs(float(normalized["detect_threshold"]))

    for key in _KILOSORT1_BOOL_PARAMS:
        if key in normalized:
            normalized[key] = _coerce_bool(normalized[key])

    for key in _KILOSORT1_FLOAT_VECTOR_PARAMS:
        if key not in normalized:
            continue
        value = normalized[key]
        if isinstance(value, (list, tuple)):
            normalized[key] = [float(v) for v in value]

    ntbuff_raw = normalized.get("ntbuff", 64)
    ntbuff = int(float(ntbuff_raw))
    normalized["ntbuff"] = ntbuff

    nt_raw = normalized.get("NT", None)
    if isinstance(nt_raw, str):
        normalized["NT"] = _safe_eval_int_expr(nt_raw.replace(" ", ""), {"ntbuff": ntbuff})
    elif nt_raw is None:
        normalized["NT"] = int(32 * 1028 + ntbuff)
    else:
        normalized["NT"] = int(float(nt_raw))

    nfilt_raw = normalized.get("Nfilt", None)
    if nfilt_raw is None:
        n_active = _get_active_channel_count(chanmap_mat_path, num_channels)
        templatemultiplier = 8
        nfilt = n_active * templatemultiplier
        normalized["Nfilt"] = int(nfilt - (nfilt % 32))
    else:
        normalized["Nfilt"] = int(float(nfilt_raw))

    allowed_keys = _get_kilosort1_allowed_param_keys()
    if allowed_keys:
        dropped = sorted(k for k in normalized.keys() if k not in allowed_keys)
        if dropped:
            print("Dropped unsupported Kilosort1 params: " + ", ".join(dropped))
        normalized = {k: v for k, v in normalized.items() if k in allowed_keys}

    return normalized


def execute_sorting_job(
    *,
    sorter: str,
    dat_path: Path,
    xml_path: Path,
    output_folder: Path,
    config_path: Path | None = None,
    kilosort1_path: Path | None = None,
    matlab_path: Path | None = None,
    chanmap_mat_path: Path | None = None,
    dtype: str = "int16",
    gain_to_uV: float = 0.195,
    offset_to_uV: float = 0.0,
    sampling_frequency: float | None = None,
    num_channels: int | None = None,
    remove_existing_folder: bool = True,
    docker_image: str | None = None,
) -> Path:
    sorter_input = sorter.lower()
    if sorter_input not in _SORTER_ALIASES:
        installed = ", ".join(_INSTALLED_SORTERS)
        raise ValueError(
            f"Unsupported sorter: {sorter}. installed sorter: {installed}"
        )
    sorter_name = _SORTER_ALIASES[sorter_input]
    matlab_cmd: str | None = None
    output_folder = Path(output_folder).resolve()

    if sorter_input == "kilosort":
        ks1_path = (kilosort1_path or _default_kilosort1_path()).resolve()
        if not ks1_path.exists():
            raise FileNotFoundError(f"KiloSort1 path not found: {ks1_path}")

        matlab_cmd = _resolve_matlab_cmd(matlab_path)

        if matlab_cmd is None:
            raise RuntimeError(
                "MATLAB executable was not found. "
                "Set PreprocessConfig(matlab_path=Path('/local/workdir/ys2375/MATLAB/R2024b/bin/matlab')) "
                "or PreprocessConfig(matlab_path=Path('C:/.../MATLAB/.../bin/matlab.exe'))."
            )

        matlab_bin = str(Path(matlab_cmd).parent)
        if os.name == "nt":
            _prepend_to_windows_path(matlab_bin)
        else:
            _prepend_to_path(matlab_bin)
        print(f"Prepended MATLAB bin to PATH for this run: {matlab_bin}")

        matlab_log = _repo_root() / ".matlab_shim" / "matlab_run.log"
        shim_path = _inject_matlab_shim(matlab_cmd, matlab_log)
        print(f"Injected MATLAB shim: {shim_path}")
        print(f"MATLAB output will be logged to: {matlab_log}")

        matlab_resolved = shutil.which("matlab")
        if matlab_resolved is None:
            raise RuntimeError(
                "Failed to resolve 'matlab' command even after PATH injection. "
                f"matlab_path={matlab_cmd}, shim={shim_path}"
            )
        print(f"Resolved matlab command: {matlab_resolved}")

        ss.KilosortSorter.set_kilosort_path(str(ks1_path))
        print(f"Kilosort path set to: {ks1_path}")

    cfg_path = (config_path or _default_sorter_config_path(sorter_input)).resolve()
    params = _load_params(cfg_path)
    print(f"Loaded sorter params: {cfg_path}")

    sr, nch = _resolve_sr_nch(sampling_frequency, num_channels, xml_path)

    if sorter_input == "kilosort":
        params = _normalize_kilosort_params(
            params,
            num_channels=nch,
            chanmap_mat_path=Path(chanmap_mat_path) if chanmap_mat_path is not None else None,
        )
        print(f"Resolved Kilosort params: NT={params.get('NT')} Nfilt={params.get('Nfilt')} ntbuff={params.get('ntbuff')}")

    dat_path = Path(dat_path).resolve()
    recording = se.read_binary(
        str(dat_path),
        sampling_frequency=sr,
        dtype=dtype,
        num_channels=nch,
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
    )

    if chanmap_mat_path is not None and Path(chanmap_mat_path).exists():
        recording = attach_probe_from_chanmap(recording, Path(chanmap_mat_path))

    run_kwargs: dict[str, Any] = {
        "sorter_name": sorter_name,
        "recording": recording,
        "folder": output_folder,
        "verbose": True,
        "with_output": True,
        "remove_existing_folder": remove_existing_folder,
        **params,
    }
    if docker_image:
        run_kwargs["docker_image"] = docker_image

    print(f"Running sorter={sorter_name} -> {output_folder}")
    try:
        _ = ss.run_sorter(**run_kwargs)
    except Exception as err:
        # GPU initialization can fail transiently on first MATLAB call.
        # Check the MATLAB log for details, then re-run the cell.
        matlab_log = _repo_root() / ".matlab_shim" / "matlab_run.log"
        hint = ""
        if matlab_log.exists():
            log_tail = matlab_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            print(f"\n[MATLAB log: {matlab_log}]")
            print(log_tail)
            if "higher compute capability" in log_tail:
                hint = (
                    "\nDetected MATLAB CUDA compatibility error. "
                    "Try updating NVIDIA driver/CUDA runtime visible to MATLAB, "
                    "or use a MATLAB release with support for your GPU architecture."
                )
        raise RuntimeError(
            f"Kilosort failed (possible GPU initialization error). "
            f"Check the MATLAB log above, then re-run the cell.{hint}\nOriginal error: {err}"
        ) from err
    print("Sorter finished")
    return output_folder


def run_sorter_cli(args: argparse.Namespace) -> None:
    xml_path = Path(args.xml_path) if args.xml_path else None
    if xml_path is None:
        raise ValueError("--xml-path is required")

    _ = execute_sorting_job(
        sorter=args.sorter,
        dat_path=Path(args.dat_path),
        xml_path=xml_path,
        output_folder=Path(args.output_folder),
        config_path=Path(args.config) if args.config else None,
        kilosort1_path=Path(args.kilosort1_path) if args.kilosort1_path else None,
        matlab_path=Path(args.matlab_path) if args.matlab_path else None,
        chanmap_mat_path=Path(args.chanmap) if args.chanmap else None,
        dtype=args.dtype,
        gain_to_uV=args.gain_to_uV,
        offset_to_uV=args.offset_to_uV,
        sampling_frequency=args.sampling_frequency,
        num_channels=args.num_channels,
        remove_existing_folder=args.remove_existing_folder,
        docker_image=args.docker_image,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run spike sorting from config file (YAML/JSON)")
    p.add_argument("--sorter", default="kilosort", help="Sorter name (kilosort or kilosort4)")
    p.add_argument("--config", default="sorter/Kilosort1_config.yaml", help="Config path (.yaml/.yml/.json)")
    p.add_argument("--kilosort1-path", default="sorter/Kilosort1", help="Kilosort1 folder path")
    p.add_argument(
        "--matlab-path",
        default=None,
        help="Optional MATLAB executable or MATLAB bin directory path",
    )

    p.add_argument("--dat-path", required=True, help="Path to input dat file")
    p.add_argument("--xml-path", default=None, help="Optional XML path for sr/nChannels")
    p.add_argument("--chanmap", default=None, help="Optional chanMap.mat path to attach probe geometry")
    p.add_argument("--sampling-frequency", type=float, default=None, help="Sampling frequency (Hz)")
    p.add_argument("--num-channels", type=int, default=None, help="Number of channels")

    p.add_argument("--dtype", default="int16", help="Binary dtype")
    p.add_argument("--gain-to-uV", type=float, default=0.195, help="gain_to_uV")
    p.add_argument("--offset-to-uV", type=float, default=0.0, help="offset_to_uV")

    p.add_argument("--output-folder", required=True, help="Sorter output folder")
    p.add_argument("--remove-existing-folder", action="store_true", help="Delete existing output folder")
    p.add_argument("--docker-image", default=None, help="Optional docker image")
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_sorter_cli(parser.parse_args())
