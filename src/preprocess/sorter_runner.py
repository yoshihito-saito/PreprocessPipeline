from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import ast

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import yaml
from scipy.io import loadmat, savemat
from spikeinterface.core.job_tools import job_keys as _SI_JOB_KEYS

from .io import load_xml_metadata
from .recording import apply_preprocessing, attach_probe_from_chanmap, select_recording_channels


def _default_parallel_n_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 128:
        return max(1, int(cpu_count) - 4)
    return 128


_SORTER_ALIASES = {
    "kilosort": "kilosort",
    "kilosort2.5": "kilosort2_5",
    "kilosort2_5": "kilosort2_5",
    "kilosort25": "kilosort2_5",
    "kilosort4": "kilosort4",
}

_INSTALLED_SORTERS = tuple(_SORTER_ALIASES.keys())
_MATLAB_SORTER_INPUTS = {"kilosort", "kilosort2.5", "kilosort2_5", "kilosort25"}

_KILOSORT1_WRAPPER_ALIASES = {
    "GPU": "useGPU",
    "fshigh": "freq_min",
    "fslow": "freq_max",
    "spkTh": "detect_threshold",
}
_KILOSORT25_WRAPPER_ALIASES = {
    "fshigh": "freq_min",
    "Th": "projection_threshold",
    "spkTh": "detect_threshold",
    "ThPre": "preclust_threshold",
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
_KILOSORT25_BOOL_PARAMS = {
    "car",
    "delete_recording_dat",
    "do_correction",
    "keep_good_only",
    "progress_bar",
    "save_rez_to_mat",
    "skip_kilosort_preprocessing",
}
_KILOSORT25_FLOAT_VECTOR_PARAMS = {
    "projection_threshold",
    "momentum",
}
_KILOSORT25_OPS_OVERRIDE_KEYS = {
    "GPU",
    "mergeShapeEnable",
    "mergeShapeMinCorr",
    "mergeShapeExcludeMs",
    "mergeShapeWindowMs",
    "mergeTemplateSimThr",
    "nSkipCov",
    "nskip",
    "reorder",
    "useRAM",
}

_KILOSORT1_ALLOWED_PARAM_KEYS: set[str] | None = None
_KILOSORT25_ALLOWED_PARAM_KEYS: set[str] | None = None
_KILOSORT4_ALLOWED_PARAM_KEYS: set[str] | None = None
_KILOSORT4_BOOL_PARAMS = {
    "do_CAR",
    "clear_cache",
    "delete_recording_dat",
    "progress_bar",
}
_JOB_KWARG_KEYS = set(_SI_JOB_KEYS)


def _prepend_to_path(path_to_add: str) -> None:
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = path_to_add + os.pathsep + existing_path


def _prepend_to_windows_path(path_to_add: str) -> None:
    _prepend_to_path(path_to_add)
    existing_path_alt = os.environ.get("Path", "")
    os.environ["Path"] = path_to_add + os.pathsep + existing_path_alt


def _ensure_matlab_shell_env() -> None:
    if os.name == "nt":
        return
    current = os.environ.get("MATLAB_SHELL", "").strip()
    if current:
        return
    bash_path = shutil.which("bash") or "/bin/bash"
    os.environ["MATLAB_SHELL"] = bash_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sorter_config_path(sorter: str) -> Path:
    s = sorter.lower()
    if s == "kilosort":
        return _repo_root() / "sorter" / "Kilosort1_config.yaml"
    if s in {"kilosort2.5", "kilosort2_5", "kilosort25"}:
        return _repo_root() / "sorter" / "Kilosort2.5_config.yaml"
    if s == "kilosort4":
        return _repo_root() / "sorter" / "Kilosort4_config.yaml"
    raise ValueError(f"Unsupported sorter: {sorter}")


def _default_kilosort1_path() -> Path:
    return _repo_root() / "sorter" / "Kilosort1"


def _default_kilosort25_path() -> Path:
    return _repo_root() / "sorter" / "Kilosort2.5"


def _default_kilosort4_path() -> Path:
    return _repo_root() / "sorter" / "Kilosort4"


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


def _inject_matlab_shim(
    matlab_cmd: str,
    matlab_log: Path,
    *,
    matlab_max_workers: int | None = None,
) -> Path:
    shim_dir = _repo_root() / ".matlab_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    requested_workers = None if matlab_max_workers is None else int(matlab_max_workers)
    if requested_workers is not None and requested_workers < 1:
        requested_workers = 1
    startup_lines = [
        "try",
        "    parallel.gpu.enableCUDAForwardCompatibility(true);",
        "catch",
        "end",
    ]
    if requested_workers is not None:
        startup_lines.extend(
            [
                "try",
                "    if isempty(getCurrentTask())",
                f"        requested_workers = {requested_workers};",
                "        c = parcluster('Processes');",
                "        detected_max_workers = c.NumWorkers;",
                "        pool_workers = min(requested_workers, detected_max_workers);",
                "        if pool_workers >= 1",
                "            c.NumWorkers = pool_workers;",
                "            try",
                "                c.PreferredPoolNumWorkers = pool_workers;",
                "            catch",
                "            end",
                "            try",
                "                c.NumThreads = 1;",
                "            catch",
                "            end",
                "            pool = gcp('nocreate');",
                "            if ~isempty(pool) && pool.NumWorkers ~= pool_workers",
                "                delete(pool);",
                "                pool = [];",
                "            end",
                "            if isempty(pool)",
                "                parpool(c, pool_workers);",
                "            end",
                "        end",
                "    end",
                "catch ME",
                "    fprintf(2, 'Warning: failed to configure MATLAB process pool: %s\\n', ME.message);",
                "end",
            ]
        )
    startup_m = shim_dir / "startup.m"
    startup_m.write_text("\n".join(startup_lines) + "\n", encoding="utf-8")
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
            f"matlab_log={repr(str(matlab_log))}\n"
            f"\"{matlab_cmd}\" -logfile \"$matlab_log\" \"${{args[@]}}\" &\n"
            "matlab_pid=$!\n"
            "monitor_pid=\"\"\n"
            "(\n"
            "  shutdown_seen_at=0\n"
            "  final_pass_seen=0\n"
            "  all_batches_done=0\n"
            "  while kill -0 \"$matlab_pid\" 2>/dev/null; do\n"
            "    if IFS= read -r -t 1 line; then\n"
            "      if [[ \"$line\" =~ batch[[:space:]]([0-9]+)/([0-9]+) ]]; then\n"
            "        if [[ \"${BASH_REMATCH[1]}\" == \"${BASH_REMATCH[2]}\" ]]; then\n"
            "          all_batches_done=1\n"
            "        fi\n"
            "      fi\n"
            "      if [[ \"$line\" == *\"Running the final template matching pass\"* ]]; then\n"
            "        final_pass_seen=1\n"
            "        shutdown_seen_at=0\n"
            "      fi\n"
            "      if [[ \"$line\" == *\"batch \"* ]]; then\n"
            "        shutdown_seen_at=0\n"
            "      fi\n"
            "      if (( final_pass_seen == 1 || all_batches_done == 1 )) && [[ \"$line\" == *\"Parallel pool using the 'Processes' profile is shutting down.\"* ]]; then\n"
            "        shutdown_seen_at=$(date +%s)\n"
            "      fi\n"
            "    fi\n"
            "    if (( shutdown_seen_at > 0 )); then\n"
            "      now=$(date +%s)\n"
            "      if (( now - shutdown_seen_at >= 60 )); then\n"
            "        if kill -0 \"$matlab_pid\" 2>/dev/null; then\n"
            "          echo \"[matlab-shim] MATLAB stuck after parallel pool shutdown; sending SIGTERM\" >&2\n"
            "          kill -TERM \"$matlab_pid\" 2>/dev/null || true\n"
            "          sleep 15\n"
            "        fi\n"
            "        if kill -0 \"$matlab_pid\" 2>/dev/null; then\n"
            "          echo \"[matlab-shim] MATLAB still alive; sending SIGKILL\" >&2\n"
            "          kill -KILL \"$matlab_pid\" 2>/dev/null || true\n"
            "        fi\n"
            "        break\n"
            "      fi\n"
            "    fi\n"
            "  done < <(tail -n0 -F \"$matlab_log\" 2>/dev/null)\n"
            ") &\n"
            "monitor_pid=$!\n"
            "if wait \"$matlab_pid\"; then\n"
            "  exit_code=0\n"
            "else\n"
            "  exit_code=$?\n"
            "fi\n"
            "if [[ -n \"$monitor_pid\" ]]; then\n"
            "  kill \"$monitor_pid\" 2>/dev/null || true\n"
            "  wait \"$monitor_pid\" 2>/dev/null || true\n"
            "fi\n"
            "exit \"$exit_code\"\n"
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


def _get_kilosort4_allowed_param_keys() -> set[str]:
    global _KILOSORT4_ALLOWED_PARAM_KEYS
    if _KILOSORT4_ALLOWED_PARAM_KEYS is not None:
        return _KILOSORT4_ALLOWED_PARAM_KEYS
    try:
        keys = set(ss.Kilosort4Sorter.default_params().keys()).union(_JOB_KWARG_KEYS)
    except Exception:
        keys = set()
    _KILOSORT4_ALLOWED_PARAM_KEYS = keys
    return keys


def _get_kilosort25_allowed_param_keys() -> set[str]:
    global _KILOSORT25_ALLOWED_PARAM_KEYS
    if _KILOSORT25_ALLOWED_PARAM_KEYS is not None:
        return _KILOSORT25_ALLOWED_PARAM_KEYS
    try:
        keys = set(ss.Kilosort2_5Sorter.default_params().keys()).union(_JOB_KWARG_KEYS)
    except Exception:
        keys = set()
    _KILOSORT25_ALLOWED_PARAM_KEYS = keys
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
    active_channel_count: int | None = None,
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
        n_active = (
            int(active_channel_count)
            if active_channel_count is not None and int(active_channel_count) > 0
            else _get_active_channel_count(chanmap_mat_path, num_channels)
        )
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


def _normalize_kilosort4_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)

    for key in _KILOSORT4_BOOL_PARAMS:
        if key in normalized:
            normalized[key] = _coerce_bool(normalized[key])

    allowed_keys = _get_kilosort4_allowed_param_keys()
    if allowed_keys:
        dropped = sorted(k for k in normalized.keys() if k not in allowed_keys)
        if dropped:
            print("Dropped unsupported Kilosort4 params: " + ", ".join(dropped))
        normalized = {k: v for k, v in normalized.items() if k in allowed_keys}
    return normalized


def _normalize_kilosort25_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = {
        _KILOSORT25_WRAPPER_ALIASES.get(key, key): value
        for key, value in params.items()
        if key not in _KILOSORT25_OPS_OVERRIDE_KEYS
    }
    ops_overrides = {k: params[k] for k in _KILOSORT25_OPS_OVERRIDE_KEYS if k in params}

    if "detect_threshold" in normalized:
        normalized["detect_threshold"] = abs(float(normalized["detect_threshold"]))

    for key in _KILOSORT25_BOOL_PARAMS:
        if key in normalized:
            normalized[key] = _coerce_bool(normalized[key])

    for key in _KILOSORT25_FLOAT_VECTOR_PARAMS:
        if key not in normalized:
            continue
        value = normalized[key]
        if isinstance(value, (list, tuple)):
            normalized[key] = [float(v) for v in value]

    normalized["ntbuff"] = int(float(normalized.get("ntbuff", 64)))

    if "NT" in normalized:
        nt_raw = normalized["NT"]
        if isinstance(nt_raw, str):
            normalized["NT"] = _safe_eval_int_expr(nt_raw.replace(" ", ""), {"ntbuff": int(normalized["ntbuff"])})
        elif nt_raw is not None:
            normalized["NT"] = int(float(nt_raw))

    for key in {"nblocks", "nPCs", "nfilt_factor", "preclust_threshold", "wave_length"}:
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(float(normalized[key]))

    # SpikeInterface defaults do_correction=True for Kilosort2.5. When nblocks=0,
    # drift correction is effectively disabled and the generated master script can
    # still try to save rez.dshift at the end, which crashes because that field was
    # never created. Keep these settings consistent in the runner.
    if int(normalized.get("nblocks", 5)) == 0:
        normalized["do_correction"] = False

    for key in {"AUCsplit", "freq_min", "lam", "minFR", "minfr_goodchannels", "scaleproc", "sig", "sigmaMask", "whiteningRange"}:
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])

    for key in {"GPU", "nSkipCov", "nskip", "reorder", "useRAM"}:
        if key in ops_overrides and ops_overrides[key] is not None:
            value = ops_overrides[key]
            if isinstance(value, bool):
                ops_overrides[key] = float(value)
            elif isinstance(value, (int, float)):
                ops_overrides[key] = float(value)

    if "mergeShapeEnable" in ops_overrides:
        ops_overrides["mergeShapeEnable"] = _coerce_bool(ops_overrides["mergeShapeEnable"])

    for key in {"mergeShapeMinCorr", "mergeShapeExcludeMs", "mergeShapeWindowMs", "mergeTemplateSimThr"}:
        if key in ops_overrides and ops_overrides[key] is not None:
            ops_overrides[key] = float(ops_overrides[key])

    allowed_keys = _get_kilosort25_allowed_param_keys()
    if allowed_keys:
        dropped = sorted(k for k in normalized.keys() if k not in allowed_keys)
        if dropped:
            print("Dropped unsupported Kilosort2.5 params: " + ", ".join(dropped))
        normalized = {k: v for k, v in normalized.items() if k in allowed_keys}

    return normalized, ops_overrides


def _pop_kilosort4_auto_geom_options(
    params: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    cleaned = dict(params)
    enabled = _coerce_bool(cleaned.pop("auto_geom_from_probe", False))
    options = {
        "max_channel_distance_factor": float(cleaned.pop("auto_geom_max_channel_distance_factor", 1.25)),
        "gap_factor": float(cleaned.pop("auto_geom_gap_factor", 2.5)),
        "max_xcenters_cap": int(float(cleaned.pop("auto_geom_max_xcenters_cap", 12))),
    }
    return cleaned, enabled, options


def _collect_probe_contact_positions(probe_like: Any) -> np.ndarray:
    positions: list[np.ndarray] = []
    if hasattr(probe_like, "probes"):
        probes = getattr(probe_like, "probes")
        for probe in probes:
            pos = np.asarray(getattr(probe, "contact_positions", []), dtype=float)
            if pos.ndim == 2 and pos.shape[0] > 0 and pos.shape[1] >= 2:
                positions.append(pos[:, :2])
    else:
        pos = np.asarray(getattr(probe_like, "contact_positions", []), dtype=float)
        if pos.ndim == 2 and pos.shape[0] > 0 and pos.shape[1] >= 2:
            positions.append(pos[:, :2])
    if not positions:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(positions)

def _pairwise_channel_distances(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(D, np.inf)
    return D


def _estimate_contact_pitch(values: np.ndarray, *, fallback: float) -> float:
    unique_values = np.unique(np.round(values.astype(float), 6))
    if unique_values.size <= 1:
        return float(fallback)
    diffs = np.diff(np.sort(unique_values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float(fallback)
    return float(np.median(diffs))


def _estimate_shank_components(pos: np.ndarray, spacing_scale: float) -> list[np.ndarray]:
    n_ch = int(pos.shape[0])
    radius = max(1.6 * float(spacing_scale), 1.0)
    D = _pairwise_channel_distances(pos)
    neighbors = [list(np.where(D[i] <= radius)[0]) for i in range(n_ch)]

    visited = np.zeros(n_ch, dtype=bool)
    comps: list[np.ndarray] = []
    for i in range(n_ch):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            u = int(stack.pop())
            comp.append(u)
            for v in neighbors[u]:
                vv = int(v)
                if not visited[vv]:
                    visited[vv] = True
                    stack.append(vv)
        comps.append(np.asarray(comp, dtype=int))
    return comps


def auto_geom_params_from_probe(
    probe: Any,
    *,
    max_channel_distance_factor: float = 1.25,
    gap_factor: float = 2.5,
    max_xcenters_cap: int = 12,
) -> dict[str, Any]:
    """
    Derive Kilosort4 geometry parameters from probe contact positions.
    Works for Probe and ProbeGroup-like objects exposing contact positions.
    """
    pos = _collect_probe_contact_positions(probe)
    if pos.shape[0] == 0:
        return {}

    n_ch = int(pos.shape[0])
    if n_ch == 1:
        return {
            "dmin": 20.0,
            "dminx": 32.0,
            "max_channel_distance": 20.0,
            "x_centers": None,
        }
    D = _pairwise_channel_distances(pos)
    nn = np.asarray(D.min(axis=1), dtype=float)

    finite_nn = nn[np.isfinite(nn) & (nn > 0)]
    nn_med_raw = float(np.median(finite_nn)) if finite_nn.size > 0 else 20.0

    if nn_med_raw < 1e-3:
        scale = 1e6
    elif nn_med_raw < 1.0:
        scale = 1e3
    else:
        scale = 1.0

    pos = pos * scale
    x = pos[:, 0]
    y = pos[:, 1]
    nn_med = nn_med_raw * scale
    D = _pairwise_channel_distances(pos)
    dmin = max(1.0, _estimate_contact_pitch(y, fallback=nn_med))
    dminx = max(16.0, _estimate_contact_pitch(x, fallback=32.0))
    comps = _estimate_shank_components(pos, max(nn_med, dmin, dminx))

    min_inter = np.inf
    if len(comps) > 1:
        centroids = np.asarray([[float(x[c].mean()), float(y[c].mean())] for c in comps], dtype=float)
        Dc = np.sqrt(np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(Dc, np.inf)
        min_inter = float(np.min(Dc))

    local_radius = max(1.75 * min(dmin, dminx), 1.75 * nn_med)
    local_counts = np.sum(D <= local_radius, axis=1) + 1
    neighborhood_order = int(np.median(local_counts)) if local_counts.size > 0 else 2
    kth_index = max(1, min(n_ch - 1, max(2, neighborhood_order // 2))) - 1
    sorted_distances = np.sort(D, axis=1)
    kth_distances = sorted_distances[:, kth_index]
    kth_distances = kth_distances[np.isfinite(kth_distances) & (kth_distances > 0)]
    if kth_distances.size == 0:
        base_distance = nn_med
    else:
        base_distance = float(np.median(kth_distances))
    max_channel_distance = max(20.0, float(max_channel_distance_factor) * base_distance)
    if np.isfinite(min_inter):
        max_channel_distance = min(max_channel_distance, 0.45 * min_inter)

    gap_thr = max(50.0, float(gap_factor) * max(nn_med, dminx))
    x_centers_total = 0
    for comp in comps:
        x_comp = np.unique(np.round(x[comp], 6))
        if x_comp.size == 0:
            continue
        if x_comp.size == 1:
            x_centers_total += 1
            continue
        x_sorted = np.sort(x_comp)
        groups: list[list[float]] = [[float(x_sorted[0])]]
        for xv in x_sorted[1:]:
            xvf = float(xv)
            if (xvf - groups[-1][-1]) <= gap_thr:
                groups[-1].append(xvf)
            else:
                groups.append([xvf])

        for group in groups:
            n_unique_x = len(group)
            x_centers_total += max(1, int(np.ceil(n_unique_x / 2.0)))

    if x_centers_total <= 1:
        x_centers = None
    else:
        x_centers = int(min(x_centers_total, int(max_xcenters_cap)))

    return {
        "dmin": float(dmin),
        "dminx": float(dminx),
        "max_channel_distance": float(max_channel_distance),
        "x_centers": x_centers,
    }


def _auto_geom_params_from_recording(recording: Any, **kwargs: Any) -> dict[str, Any]:
    probe_like: Any | None = None
    if hasattr(recording, "get_probegroup"):
        try:
            probe_like = recording.get_probegroup()
        except Exception:
            probe_like = None
    if probe_like is None and hasattr(recording, "get_probe"):
        try:
            probe_like = recording.get_probe()
        except Exception:
            probe_like = None
    if probe_like is None:
        return {}
    try:
        return auto_geom_params_from_probe(probe_like, **kwargs)
    except Exception as exc:
        print(f"Warning: failed to derive Kilosort4 geometry params from probe: {exc}")
        return {}


def _merge_sorter_job_kwargs(
    params: dict[str, Any],
    job_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(params)
    if not job_kwargs:
        return merged
    for k, v in job_kwargs.items():
        if k in _JOB_KWARG_KEYS and v is not None:
            merged[k] = v
    n_jobs = merged.get("n_jobs", None)
    try:
        n_jobs_val = None if n_jobs is None else float(n_jobs)
    except (TypeError, ValueError):
        n_jobs_val = None
    has_chunk_setting = any(
        merged.get(k, None) is not None
        for k in ("chunk_size", "chunk_memory", "total_memory", "chunk_duration")
    )
    if n_jobs_val is not None and n_jobs_val != 1.0 and not has_chunk_setting:
        merged["chunk_duration"] = "1s"
        print(
            "No chunk_* job kwargs provided for n_jobs!=1. "
            "Setting chunk_duration='1s' for sorter binary export."
        )
    return merged


def _resolve_active_channels_0based(
    *,
    num_channels: int,
    chanmap_mat_path: Path | None,
    exclude_channels_0based: list[int] | None,
) -> list[int]:
    all_channels = list(range(int(num_channels)))
    active_from_chanmap: list[int] | None = None

    if chanmap_mat_path is not None and chanmap_mat_path.exists():
        try:
            mat = loadmat(str(chanmap_mat_path), simplify_cells=True)
            chanmap_raw = mat.get("chanMap", None)
            connected_raw = mat.get("connected", None)

            if chanmap_raw is not None:
                chanmap = np.asarray(chanmap_raw).reshape(-1).astype(np.int64) - 1
                chanmap = chanmap[(chanmap >= 0) & (chanmap < int(num_channels))]
            else:
                chanmap = np.arange(int(num_channels), dtype=np.int64)

            if connected_raw is not None:
                connected = np.asarray(connected_raw).reshape(-1)
                if connected.size == chanmap.size:
                    conn_mask = connected.astype(float) > 0
                    chanmap = chanmap[conn_mask]
                elif connected.size == int(num_channels) and chanmap_raw is None:
                    chanmap = np.flatnonzero(connected.astype(float) > 0).astype(np.int64)

            # Preserve channel order from chanMap while removing duplicates.
            active_from_chanmap = []
            seen: set[int] = set()
            for ch in chanmap.tolist():
                ch_i = int(ch)
                if ch_i not in seen:
                    active_from_chanmap.append(ch_i)
                    seen.add(ch_i)
        except Exception:
            active_from_chanmap = None

    base = active_from_chanmap if active_from_chanmap is not None else all_channels
    excluded_set = {
        int(ch)
        for ch in (exclude_channels_0based or [])
        if 0 <= int(ch) < int(num_channels)
    }
    active = [ch for ch in base if ch not in excluded_set]
    if not active:
        raise ValueError(
            "No active channels remain for sorting after applying chanMap/reject channels."
        )
    return active


def _patch_params_py(
    params_path: Path,
    *,
    dat_path: Path,
    n_channels_dat: int,
    hp_filtered: bool | None = None,
) -> None:
    if not params_path.exists():
        return
    dat_relative = os.path.relpath(str(dat_path), start=str(params_path.parent))
    dat_literal = repr(str(dat_relative))
    lines = params_path.read_text(encoding="utf-8").splitlines()

    out_lines: list[str] = []
    dat_set = False
    nch_set = False
    hp_set = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("dat_path"):
            out_lines.append(f"dat_path = {dat_literal}")
            dat_set = True
        elif stripped.startswith("n_channels_dat"):
            out_lines.append(f"n_channels_dat = {int(n_channels_dat)}")
            nch_set = True
        elif stripped.startswith("hp_filtered") and hp_filtered is not None:
            out_lines.append(f"hp_filtered = {bool(hp_filtered)}")
            hp_set = True
        else:
            out_lines.append(line)

    if not dat_set:
        out_lines.append(f"dat_path = {dat_literal}")
    if not nch_set:
        out_lines.append(f"n_channels_dat = {int(n_channels_dat)}")
    if hp_filtered is not None and not hp_set:
        out_lines.append(f"hp_filtered = {bool(hp_filtered)}")

    params_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _patch_phy_outputs_for_raw_dat(
    *,
    output_folder: Path,
    raw_dat_path: Path,
    n_channels_dat: int,
    hp_filtered: bool | None = None,
) -> None:
    for params_path in output_folder.rglob("params.py"):
        _patch_params_py(
            params_path,
            dat_path=raw_dat_path,
            n_channels_dat=n_channels_dat,
            hp_filtered=hp_filtered,
        )

def _flatten_sorter_output_folder(output_folder: Path) -> None:
    sorter_output = output_folder / "sorter_output"
    if not sorter_output.exists() or not sorter_output.is_dir():
        return
    entries = list(sorter_output.iterdir())
    conflicts = [output_folder / e.name for e in entries if (output_folder / e.name).exists()]
    if conflicts:
        conflict_list = ", ".join(str(p.name) for p in conflicts[:5])
        raise RuntimeError(
            f"Cannot flatten sorter_output due to existing files in {output_folder}: {conflict_list}"
        )
    for e in entries:
        shutil.move(str(e), str(output_folder / e.name))
    sorter_output.rmdir()


def _rewrite_kilosort_ops_nchan_from_chanmap(
    *,
    sorter_output_folder: Path,
    n_channels_total: int | None = None,
    criterion_noise_channels_default: float = 0.2,
) -> None:
    chanmap_path = Path(sorter_output_folder) / "chanMap.mat"
    ops_path = Path(sorter_output_folder) / "ops.mat"
    if not chanmap_path.exists() or not ops_path.exists():
        return
    try:
        chanmap = loadmat(str(chanmap_path), simplify_cells=True)
        ops = loadmat(str(ops_path), simplify_cells=True).get("ops")
        if not isinstance(ops, dict):
            return
        connected = chanmap.get("connected", None)
        if connected is not None:
            connected_arr = np.asarray(connected).reshape(-1)
            n_active = int(np.count_nonzero(connected_arr > 0))
        else:
            chan_map = np.asarray(chanmap.get("chanMap", [])).reshape(-1)
            n_active = int(chan_map.size)
        if n_active <= 0:
            return
        ops["Nchan"] = float(n_active)
        if n_channels_total is not None and int(n_channels_total) > 0:
            ops["NchanTOT"] = float(int(n_channels_total))
        # Some Kilosort1 forks require this field in ops.
        if "criterionNoiseChannels" not in ops or ops.get("criterionNoiseChannels") is None:
            ops["criterionNoiseChannels"] = float(criterion_noise_channels_default)
        savemat(str(ops_path), {"ops": ops})
    except Exception as exc:
        print(f"Warning: failed to rewrite Kilosort ops Nchan from chanMap: {exc}")


def _to_matlab_ops_value(value: Any) -> Any:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    return value


def _apply_kilosort_ops_overrides(*, sorter_output_folder: Path, ops_overrides: dict[str, Any]) -> None:
    if not ops_overrides:
        return
    ops_path = Path(sorter_output_folder) / "ops.mat"
    if not ops_path.exists():
        return
    try:
        payload = loadmat(str(ops_path), simplify_cells=True)
        ops = payload.get("ops")
        if not isinstance(ops, dict):
            return
        for key, value in ops_overrides.items():
            ops[key] = _to_matlab_ops_value(value)
        savemat(str(ops_path), {"ops": ops})
    except Exception as exc:
        print(f"Warning: failed to apply Kilosort ops overrides: {exc}")


@contextmanager
def _kilosort_chanmap_override(
    sorter_cls: type,
    *,
    chanmap_mat_path: Path | None,
    ops_overrides: dict[str, Any] | None = None,
):
    if (chanmap_mat_path is None or not Path(chanmap_mat_path).exists()) and not ops_overrides:
        yield
        return

    original = getattr(sorter_cls, "_generate_channel_map_file", None)
    original_ops = getattr(sorter_cls, "_generate_ops_file", None)

    def _copy_channel_map(_recording, sorter_output_folder):
        dst = Path(sorter_output_folder) / "chanMap.mat"
        shutil.copy2(str(Path(chanmap_mat_path).resolve()), str(dst))

    def _patched_generate_ops_file(_cls, recording, params, sorter_output_folder, binary_file_path):
        original_ops(recording, params, sorter_output_folder, binary_file_path)
        if chanmap_mat_path is not None and Path(chanmap_mat_path).exists():
            _rewrite_kilosort_ops_nchan_from_chanmap(
                sorter_output_folder=Path(sorter_output_folder),
                n_channels_total=int(recording.get_num_channels()),
            )
        if ops_overrides:
            _apply_kilosort_ops_overrides(
                sorter_output_folder=Path(sorter_output_folder),
                ops_overrides=ops_overrides,
            )

    if chanmap_mat_path is not None and Path(chanmap_mat_path).exists():
        sorter_cls._generate_channel_map_file = staticmethod(_copy_channel_map)
    sorter_cls._generate_ops_file = classmethod(_patched_generate_ops_file)
    try:
        yield
    finally:
        if original is not None:
            sorter_cls._generate_channel_map_file = original
        if original_ops is not None:
            sorter_cls._generate_ops_file = original_ops


@contextmanager
def _kilosort1_chanmap_override(chanmap_mat_path: Path | None):
    with _kilosort_chanmap_override(
        ss.KilosortSorter,
        chanmap_mat_path=chanmap_mat_path,
        ops_overrides=None,
    ):
        yield


@contextmanager
def _kilosort4_package_override(kilosort4_path: Path | None):
    target_path = Path(kilosort4_path) if kilosort4_path is not None else _default_kilosort4_path()
    if target_path is None:
        yield
        return

    import_root = Path(target_path).resolve()
    if not import_root.exists():
        raise FileNotFoundError(f"Kilosort4 path not found: {import_root}")

    if (import_root / "kilosort").is_dir():
        package_root = import_root
    elif import_root.name == "kilosort" and import_root.is_dir():
        package_root = import_root.parent
    else:
        raise FileNotFoundError(
            "Kilosort4 path must point to a repository root containing a "
            f"'kilosort' package directory, or to the package directory itself: {import_root}"
        )

    package_root_str = str(package_root)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "kilosort" or name.startswith("kilosort.")
    }
    original_sys_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")
    original_is_installed = ss.Kilosort4Sorter.is_installed
    original_get_sorter_version = ss.Kilosort4Sorter.get_sorter_version
    original_check_sorter_version = ss.Kilosort4Sorter.check_sorter_version

    for name in list(sys.modules):
        if name == "kilosort" or name.startswith("kilosort."):
            del sys.modules[name]

    sys.path.insert(0, package_root_str)
    if original_pythonpath:
        os.environ["PYTHONPATH"] = package_root_str + os.pathsep + original_pythonpath
    else:
        os.environ["PYTHONPATH"] = package_root_str

    ss.Kilosort4Sorter.is_installed = classmethod(lambda cls: True)
    ss.Kilosort4Sorter.get_sorter_version = classmethod(lambda cls: "4.1.2")
    ss.Kilosort4Sorter.check_sorter_version = classmethod(lambda cls: None)

    try:
        print(f"Kilosort4 package path set to: {package_root}")
        yield
    finally:
        ss.Kilosort4Sorter.is_installed = original_is_installed
        ss.Kilosort4Sorter.get_sorter_version = original_get_sorter_version
        ss.Kilosort4Sorter.check_sorter_version = original_check_sorter_version
        for name in list(sys.modules):
            if name == "kilosort" or name.startswith("kilosort."):
                del sys.modules[name]
        sys.path[:] = original_sys_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath
        sys.modules.update(original_modules)


def _cleanup_temp_wh_dat(output_folder: Path) -> None:
    candidates = [
        output_folder / "temp_wh.dat",
        output_folder / "sorter_output" / "temp_wh.dat",
    ]
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        try:
            size_bytes = p.stat().st_size
            p.unlink()
            print(f"Removed Kilosort temporary file: {p} ({size_bytes} bytes)")
        except Exception as exc:
            print(f"Warning: failed to remove Kilosort temporary file {p}: {exc}")


def execute_sorting_job(
    *,
    sorter: str,
    dat_path: Path,
    xml_path: Path,
    output_folder: Path,
    config_path: Path | None = None,
    kilosort1_path: Path | None = None,
    kilosort25_path: Path | None = None,
    kilosort4_path: Path | None = None,
    matlab_path: Path | None = None,
    matlab_max_workers: int = _default_parallel_n_jobs(),
    chanmap_mat_path: Path | None = None,
    dtype: str = "int16",
    gain_to_uV: float = 0.195,
    offset_to_uV: float = 0.0,
    sampling_frequency: float | None = None,
    num_channels: int | None = None,
    exclude_channels_0based: list[int] | None = None,
    job_kwargs: dict[str, Any] | None = None,
    remove_existing_folder: bool = True,
    docker_image: str | None = None,
    preprocess_for_sorting: bool = False,
    input_is_preprocessed: bool = False,
    bandpass_min_hz: float = 500.0,
    bandpass_max_hz: float = 8000.0,
    reference: str = "local",
    local_radius_um: tuple[float, float] = (50.0, 200.0),
    sorter_verbose: bool = False,
    cleanup_temp_wh: bool = True,
) -> Path:
    sorter_input = sorter.lower()
    if sorter_input not in _SORTER_ALIASES:
        installed = ", ".join(_INSTALLED_SORTERS)
        raise ValueError(
            f"Unsupported sorter: {sorter}. installed sorter: {installed}"
        )
    sorter_name = _SORTER_ALIASES[sorter_input]
    matlab_cmd: str | None = None
    ks4_auto_geom_enabled = False
    ks4_auto_geom_options: dict[str, Any] = {}
    ks25_ops_overrides: dict[str, Any] = {}
    kilosort4_import_root: Path | None = None
    output_folder = Path(output_folder).resolve()

    if sorter_input in _MATLAB_SORTER_INPUTS:
        if sorter_input == "kilosort":
            ks_path = (kilosort1_path or _default_kilosort1_path()).resolve()
            if not ks_path.exists():
                raise FileNotFoundError(f"KiloSort1 path not found: {ks_path}")
        else:
            ks_path = (kilosort25_path or _default_kilosort25_path()).resolve()
            if not ks_path.exists():
                raise FileNotFoundError(f"Kilosort2.5 path not found: {ks_path}")

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
            _ensure_matlab_shell_env()
        print(f"Prepended MATLAB bin to PATH for this run: {matlab_bin}")

        matlab_log = _repo_root() / ".matlab_shim" / "matlab_run.log"
        shim_path = _inject_matlab_shim(
            matlab_cmd,
            matlab_log,
            matlab_max_workers=matlab_max_workers,
        )
        print(f"Injected MATLAB shim: {shim_path}")
        print(f"MATLAB output will be logged to: {matlab_log}")

        matlab_resolved = shutil.which("matlab")
        if matlab_resolved is None:
            raise RuntimeError(
                "Failed to resolve 'matlab' command even after PATH injection. "
                f"matlab_path={matlab_cmd}, shim={shim_path}"
            )
        print(f"Resolved matlab command: {matlab_resolved}")

        if sorter_input == "kilosort":
            ss.KilosortSorter.set_kilosort_path(str(ks_path))
            print(f"Kilosort path set to: {ks_path}")
        else:
            ss.Kilosort2_5Sorter.set_kilosort2_5_path(str(ks_path))
            print(f"Kilosort2.5 path set to: {ks_path}")
    elif sorter_input == "kilosort4":
        kilosort4_import_root = (kilosort4_path or _default_kilosort4_path()).resolve()
        if not kilosort4_import_root.exists():
            raise FileNotFoundError(f"Kilosort4 path not found: {kilosort4_import_root}")

    cfg_path = (config_path or _default_sorter_config_path(sorter_input)).resolve()
    params = _load_params(cfg_path)
    if sorter_input == "kilosort4":
        params, ks4_auto_geom_enabled, ks4_auto_geom_options = _pop_kilosort4_auto_geom_options(params)
    print(f"Loaded sorter params: {cfg_path}")

    sr, nch = _resolve_sr_nch(sampling_frequency, num_channels, xml_path)
    resolved_active_channels_0based = _resolve_active_channels_0based(
        num_channels=nch,
        chanmap_mat_path=Path(chanmap_mat_path) if chanmap_mat_path is not None else None,
        exclude_channels_0based=exclude_channels_0based,
    )
    ignored_channels_0based = sorted(set(range(int(nch))) - set(resolved_active_channels_0based))
    # Keep channel count unchanged when the sorter input is already preprocessed.
    active_channels_0based = (
        list(range(int(nch))) if input_is_preprocessed else resolved_active_channels_0based
    )

    if sorter_input == "kilosort":
        params = _normalize_kilosort_params(
            params,
            num_channels=nch,
            chanmap_mat_path=Path(chanmap_mat_path) if chanmap_mat_path is not None else None,
            active_channel_count=len(resolved_active_channels_0based),
        )
        print(f"Resolved Kilosort params: NT={params.get('NT')} Nfilt={params.get('Nfilt')} ntbuff={params.get('ntbuff')}")
    elif sorter_input in {"kilosort2.5", "kilosort2_5", "kilosort25"}:
        params, ks25_ops_overrides = _normalize_kilosort25_params(params)
        print("Resolved Kilosort2.5 params")
    elif sorter_input == "kilosort4":
        global _KILOSORT4_ALLOWED_PARAM_KEYS
        with _kilosort4_package_override(kilosort4_import_root):
            _KILOSORT4_ALLOWED_PARAM_KEYS = None
            params = _normalize_kilosort4_params(params)
        if ignored_channels_0based:
            params["bad_channels"] = ignored_channels_0based
        print("Resolved Kilosort4 params")
    params = _merge_sorter_job_kwargs(params, job_kwargs)

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

    if len(active_channels_0based) != int(nch):
        recording = select_recording_channels(recording, active_channels_0based)
        print(
            "Sorting with active channels only: "
            f"{len(active_channels_0based)}/{int(nch)}"
        )
    effective_preprocess_for_sorting = bool(preprocess_for_sorting) and not bool(input_is_preprocessed)
    if effective_preprocess_for_sorting:
        recording = apply_preprocessing(
            recording_raw=recording,
            bandpass_min_hz=bandpass_min_hz,
            bandpass_max_hz=bandpass_max_hz,
            reference=reference,
            local_radius_um=local_radius_um,
        )
        print("Applied preprocessing to sorter input recording")
    elif input_is_preprocessed:
        print("Skipping sorter-side preprocessing/channel exclusion: using preprocessed binary input as-is")
    if sorter_input == "kilosort4" and ks4_auto_geom_enabled:
        auto_geom_params = _auto_geom_params_from_recording(recording, **ks4_auto_geom_options)
        if auto_geom_params:
            params.update(auto_geom_params)
            print(
                "Applied Kilosort4 probe-geometry params: "
                + ", ".join(f"{k}={auto_geom_params[k]}" for k in sorted(auto_geom_params.keys()))
            )
        else:
            print("Kilosort4 auto geometry params requested, but no probe geometry was available.")

    run_kwargs: dict[str, Any] = {
        "sorter_name": sorter_name,
        "recording": recording,
        "folder": output_folder,
        "verbose": bool(sorter_verbose),
        "with_output": True,
        "remove_existing_folder": remove_existing_folder,
        **params,
    }
    if docker_image:
        run_kwargs["docker_image"] = docker_image

    print(f"Running sorter={sorter_name} -> {output_folder}")
    try:
        chanmap_override_path = Path(chanmap_mat_path) if chanmap_mat_path is not None else None
        if sorter_input == "kilosort":
            override_ctx = _kilosort_chanmap_override(
                ss.KilosortSorter,
                chanmap_mat_path=chanmap_override_path,
                ops_overrides=None,
            )
        elif sorter_input in {"kilosort2.5", "kilosort2_5", "kilosort25"}:
            override_ctx = _kilosort_chanmap_override(
                ss.Kilosort2_5Sorter,
                chanmap_mat_path=chanmap_override_path,
                ops_overrides=ks25_ops_overrides,
            )
        elif sorter_input == "kilosort4":
            override_ctx = _kilosort4_package_override(
                Path(kilosort4_path) if kilosort4_path is not None else None
            )
        else:
            override_ctx = nullcontext()
        with override_ctx:
            _ = ss.run_sorter(**run_kwargs)
    except Exception as err:
        if sorter_input not in _MATLAB_SORTER_INPUTS:
            raise RuntimeError(
                f"Sorting failed for sorter={sorter_name}. Original error: {err}"
            ) from err
        # Surface MATLAB-side failure reasons (GPU, license/service, etc).
        matlab_log = _repo_root() / ".matlab_shim" / "matlab_run.log"
        hint = ""
        cause = "MATLAB runtime error"
        if matlab_log.exists():
            log_tail = matlab_log.read_text(encoding="utf-8", errors="replace")[-3000:]
            if bool(sorter_verbose):
                print(f"\n[MATLAB log: {matlab_log}]")
                print(log_tail)
            if "higher compute capability" in log_tail:
                cause = "possible GPU initialization error"
                hint = (
                    "\nDetected MATLAB CUDA compatibility error. "
                    "Try updating NVIDIA driver/CUDA runtime visible to MATLAB, "
                    "or use a MATLAB release with support for your GPU architecture."
                )
            elif (
                "Unable to communicate with required MathWorks services (error 5001)" in log_tail
                or "support/lme/5001" in log_tail
            ):
                cause = "MATLAB license/service communication error (5001)"
                hint = (
                    "\nDetected MATLAB licensing/service error 5001. "
                    "Check MathWorks login/license activation for this host and outbound connectivity, "
                    "or configure a valid offline/network license (e.g., MLM_LICENSE_FILE)."
                )
        raise RuntimeError(
            f"Kilosort failed ({cause}). "
            f"Check MATLAB log at: {matlab_log}, then re-run the cell.{hint}\nOriginal error: {err}"
        ) from err
    _flatten_sorter_output_folder(output_folder)
    _patch_phy_outputs_for_raw_dat(
        output_folder=output_folder,
        raw_dat_path=dat_path,
        n_channels_dat=int(nch),
        hp_filtered=bool(input_is_preprocessed),
    )
    if sorter_input in _MATLAB_SORTER_INPUTS and cleanup_temp_wh:
        _cleanup_temp_wh_dat(output_folder)
    print(
        "Patched output metadata for raw.dat compatibility: "
        f"n_channels_dat={int(nch)}, active={len(resolved_active_channels_0based)}"
    )
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
        kilosort25_path=Path(args.kilosort25_path) if args.kilosort25_path else None,
        kilosort4_path=Path(args.kilosort4_path) if args.kilosort4_path else None,
        matlab_path=Path(args.matlab_path) if args.matlab_path else None,
        matlab_max_workers=int(args.matlab_max_workers),
        chanmap_mat_path=Path(args.chanmap) if args.chanmap else None,
        dtype=args.dtype,
        gain_to_uV=args.gain_to_uV,
        offset_to_uV=args.offset_to_uV,
        sampling_frequency=args.sampling_frequency,
        num_channels=args.num_channels,
        exclude_channels_0based=None,
        job_kwargs=None,
        remove_existing_folder=args.remove_existing_folder,
        docker_image=args.docker_image,
        preprocess_for_sorting=False,
        sorter_verbose=bool(args.sorter_verbose),
        cleanup_temp_wh=not bool(args.keep_temp_wh_dat),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run spike sorting from config file (YAML/JSON)")
    p.add_argument("--sorter", default="kilosort", help="Sorter name (kilosort, kilosort2.5, or kilosort4)")
    p.add_argument(
        "--config",
        default=None,
        help="Config path (.yaml/.yml/.json). If omitted, uses sorter-specific default config.",
    )
    p.add_argument(
        "--kilosort1-path",
        default="sorter/Kilosort1",
        help="Kilosort1 folder path (used only when --sorter kilosort).",
    )
    p.add_argument(
        "--kilosort2-5-path",
        dest="kilosort25_path",
        default="sorter/Kilosort2.5",
        help="Kilosort2.5 folder path (used only when --sorter kilosort2.5).",
    )
    p.add_argument(
        "--kilosort4-path",
        default="sorter/Kilosort4",
        help="Kilosort4 repository root path containing the kilosort package (used only when --sorter kilosort4).",
    )
    p.add_argument(
        "--matlab-path",
        default=None,
        help="Optional MATLAB executable or MATLAB bin directory path",
    )
    p.add_argument(
        "--matlab-max-workers",
        type=int,
        default=_default_parallel_n_jobs(),
        help="Cap MATLAB process-pool workers; uses min(requested, profile/device max).",
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
    p.add_argument("--sorter-verbose", action="store_true", help="Enable verbose sorter logs")
    p.add_argument(
        "--keep-temp-wh-dat",
        action="store_true",
        help="Keep Kilosort1 temp_wh.dat (default behavior removes it after sorting).",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    run_sorter_cli(parser.parse_args())
