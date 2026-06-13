from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import argparse
import base64
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import shutil
import threading
import traceback
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
import webbrowser

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from scipy.io import loadmat

from src.postprocess import PostprocessConfig, run_postprocess_session
from src.preprocess import prepare_chanmap, run_preprocess_session, select_paths_with_gui
from src.preprocess.io import build_channel_map_data, set_tree_world_rw

from .config_model import PipelineGuiSettings, RunMode
from .preflight import run_preflight


DEFAULT_PORT = 8765
LOG_LIMIT_CHARS = 300_000
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(
    os.environ.get(
        "PREPROCESS_GUI_DEFAULT_CONFIG",
        str(REPO_ROOT / "config" / "preprocess_gui_default_config.json"),
    )
).expanduser()


class LogBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text = ""

    def _replace_current_line(self) -> None:
        last_newline = self._text.rfind("\n")
        if last_newline < 0:
            self._text = ""
        else:
            self._text = self._text[: last_newline + 1]

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            current = ""
            for char in text:
                if char == "\r":
                    self._replace_current_line()
                    current = ""
                    continue
                current += char
                if char == "\n":
                    self._text += current
                    current = ""
            if current:
                self._text += current
            if len(self._text) > LOG_LIMIT_CHARS:
                self._text = self._text[-LOG_LIMIT_CHARS:]

    def clear(self) -> None:
        with self._lock:
            self._text = ""

    def read(self) -> str:
        with self._lock:
            return self._text

    def write(self, text: str) -> int:
        self.append(text)
        return len(text)

    def flush(self) -> None:
        return None


class AppState:
    def __init__(self) -> None:
        self.log = LogBuffer()
        self.lock = threading.Lock()
        self.running = False
        self.last_error: str | None = None
        self.last_result: dict[str, Any] | None = None


STATE = AppState()


def _make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def _bind_server_with_fallback(
    host: str, port: int, *, max_tries: int = 50
) -> ThreadingHTTPServer:
    if port == 0:
        return _make_server(host, 0)
    last_error: OSError | None = None
    for candidate in range(port, port + max_tries):
        try:
            return _make_server(host, candidate)
        except OSError as exc:
            last_error = exc
            if getattr(exc, "errno", None) not in {98, 48, 10048}:
                raise
    raise RuntimeError(
        f"Could not bind a GUI port in range {port}-{port + max_tries - 1}."
    ) from last_error


def _settings_from_payload(payload: dict[str, Any]) -> PipelineGuiSettings:
    return PipelineGuiSettings.from_json(json.dumps(payload))


def _load_default_settings() -> PipelineGuiSettings:
    if DEFAULT_CONFIG_PATH.exists():
        return PipelineGuiSettings.load(DEFAULT_CONFIG_PATH)
    return PipelineGuiSettings()


def _settings_as_parameter_defaults(settings: PipelineGuiSettings) -> PipelineGuiSettings:
    default_settings = PipelineGuiSettings.from_json(settings.to_json())
    default_settings.basepath = ""
    default_settings.chanmap_path = ""
    default_settings.preprocess.reject_channels = []
    default_settings.postprocess.sorting_phy_folder = ""
    default_settings.postprocess.sorting_search_root = ""
    return default_settings


def _save_default_settings(settings: PipelineGuiSettings) -> Path:
    defaults = _settings_as_parameter_defaults(settings)
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    defaults.save(DEFAULT_CONFIG_PATH)
    return DEFAULT_CONFIG_PATH


def _resolve_config_path(path_text: str | None) -> Path:
    config_dir = (REPO_ROOT / "config").resolve()
    if not path_text:
        return DEFAULT_CONFIG_PATH.resolve()
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == "config":
            candidate = REPO_ROOT / candidate
        else:
            candidate = config_dir / candidate
    candidate = candidate.resolve()
    if config_dir not in (candidate, *candidate.parents):
        raise ValueError(f"Default config must be under {config_dir}")
    if candidate.suffix.lower() != ".json":
        raise ValueError("Default config path must end with .json")
    return candidate


def _load_default_settings_from_path(path: Path) -> PipelineGuiSettings:
    return PipelineGuiSettings.load(path)


def _save_default_settings_to_path(settings: PipelineGuiSettings, path: Path) -> Path:
    defaults = _settings_as_parameter_defaults(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults.save(path)
    return path


def _list_config_files() -> dict[str, Any]:
    config_dir = (REPO_ROOT / "config").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p.resolve() for p in config_dir.glob("*.json") if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    return {
        "config_dir": str(config_dir),
        "default_path": str(DEFAULT_CONFIG_PATH.resolve()),
        "files": [str(p) for p in files],
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {k: _json_default(v) for k, v in value.__dict__.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    return str(value)


def _chanmap_png_bytes(path: Path) -> bytes:
    data = loadmat(path)
    return _chanmap_png_bytes_from_data(data, title=path.name)


def _chanmap_png_bytes_from_data(data: dict[str, Any], *, title: str) -> bytes:
    x = np.asarray(data["xcoords"]).reshape(-1)
    y = np.asarray(data["ycoords"]).reshape(-1)
    kcoords = np.asarray(data.get("kcoords", np.ones_like(x))).reshape(-1)
    probe_ids = np.asarray(data.get("probe_ids", np.ones_like(x))).reshape(-1)
    connected = np.asarray(data["connected"]).reshape(-1).astype(bool)
    device_ch = np.asarray(
        data.get("chanMap0ind", np.asarray(data["chanMap"]).reshape(-1) - 1)
    ).reshape(-1).astype(int)

    n = min(len(x), len(y), len(kcoords), len(probe_ids), len(connected), len(device_ch))
    x = x[:n]
    y = y[:n]
    kcoords = kcoords[:n]
    probe_ids = probe_ids[:n]
    connected = connected[:n]
    device_ch = device_ch[:n]

    fig = Figure(figsize=(10.0, 7.2), tight_layout=True, facecolor="#111827")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0b1220")
    palette = np.asarray(
        [
            "#38bdf8",
            "#22c55e",
            "#a78bfa",
            "#67e8f9",
            "#f0abfc",
            "#60a5fa",
            "#5eead4",
            "#c4b5fd",
            "#86efac",
            "#93c5fd",
            "#d8b4fe",
            "#99f6e4",
        ],
        dtype=object,
    )
    probe_keys = probe_ids.astype(int)
    unique_probes = {int(key): idx for idx, key in enumerate(sorted(set(probe_keys.tolist())))}
    colors = np.asarray([palette[unique_probes[int(key)] % len(palette)] for key in probe_keys])
    ax.scatter(
        x[connected],
        y[connected],
        c=colors[connected],
        s=9,
        edgecolor="#06101f",
        linewidth=0.08,
    )
    if np.any(~connected):
        ax.scatter(
            x[~connected],
            y[~connected],
            c="none",
            edgecolor="red",
            marker="x",
            s=24,
            linewidth=0.9,
        )
    if n <= 256:
        for xi, yi, ch, is_connected in zip(x, y, device_ch, connected):
            ax.text(
                float(xi),
                float(yi),
                str(int(ch)),
                fontsize=3.8,
                ha="center",
                va="bottom",
                color="#d7e1ee" if is_connected else "#ff6b6b",
            )
    ax.set_title(title, color="#e6edf7")
    ax.set_xlabel("x (um)", color="#c8d3df")
    ax.set_ylabel("y (um)", color="#c8d3df")
    ax.tick_params(colors="#aeb9c6")
    for spine in ax.spines.values():
        spine.set_color("#607089")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.18, color="#9aa8bd")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor())
    return buf.getvalue()


def _chanmap_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    data = loadmat(path)
    return _chanmap_summary_from_data(data, path=str(path), exists=True)


def _chanmap_summary_from_data(
    data: dict[str, Any], *, path: str | None, exists: bool
) -> dict[str, Any]:
    connected = np.asarray(data["connected"]).reshape(-1).astype(bool)
    device_ch = np.asarray(
        data.get("chanMap0ind", np.asarray(data["chanMap"]).reshape(-1) - 1)
    ).reshape(-1).astype(int)
    n = min(len(connected), len(device_ch))
    connected = connected[:n]
    device_ch = device_ch[:n]
    bad = device_ch[~connected].astype(int).tolist()
    probe_assignments = _probe_assignments_from_chanmap_data(data)
    return {
        "exists": exists,
        "path": path,
        "channels": int(n),
        "connected": int(np.sum(connected)),
        "bad_count": int(len(bad)),
        "bad_channels": bad,
        "probe_assignments": probe_assignments,
    }


def _preview_chanmap(settings: PipelineGuiSettings) -> tuple[dict[str, Any] | None, str | None]:
    basepath = settings.basepath_path
    if basepath is None:
        return None, None
    data = build_channel_map_data(
        basepath=basepath,
        basename=settings.basename,
        probe_assignments=settings.preprocess.probe_assignments,
        reject_channels=settings.preprocess.reject_channels,
    )
    if data is None:
        return None, None
    return data, f"{settings.basename or basepath.name} current settings"


def _decode_matlab_string(value: Any) -> str | None:
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    if arr.dtype.kind == "O":
        parts: list[str] = []
        for item in arr.reshape(-1):
            text = _decode_matlab_string(item)
            if text:
                parts.append(text)
        return "".join(parts).strip() or None
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(x) for x in arr.reshape(-1)).strip()
    if arr.dtype.kind in {"i", "u"}:
        chars: list[str] = []
        for item in arr.reshape(-1):
            code = int(item)
            if code:
                chars.append(chr(code))
        return "".join(chars).strip() or None
    return None


def _probe_assignments_from_chanmap_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_json = data.get("probe_assignments_json")
    if raw_json is not None:
        text = _decode_matlab_string(raw_json)
        if text:
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return loaded
            except Exception:
                pass

    kcoords = np.asarray(data.get("kcoords", [])).reshape(-1).astype(int)
    xcoords = np.asarray(data.get("xcoords", [])).reshape(-1).astype(float)
    probe_ids = np.asarray(data.get("probe_ids", np.ones_like(kcoords))).reshape(-1).astype(int)
    n = min(kcoords.size, probe_ids.size)
    if xcoords.size:
        n = min(n, xcoords.size)
    if n == 0:
        return []
    kcoords = kcoords[:n]
    probe_ids = probe_ids[:n]
    xcoords = xcoords[:n] if xcoords.size else np.asarray([], dtype=float)

    valid_probe_ids = sorted(set(int(p) for p in probe_ids.tolist() if int(p) > 0))
    if len(valid_probe_ids) > 1:
        assignments: list[dict[str, Any]] = []
        first_probe_x: float | None = None
        for probe_id in valid_probe_ids:
            mask = probe_ids == probe_id
            groups = sorted(set(int(k) - 1 for k in kcoords[mask].tolist() if int(k) > 0))
            if not groups:
                continue
            x_offset = 0
            if xcoords.size:
                probe_x = float(np.nanmin(xcoords[mask]))
                if first_probe_x is None:
                    first_probe_x = probe_x
                x_offset = int(round(probe_x - first_probe_x))
            assignments.append({"type": "staggered", "groups": groups, "x_offset": x_offset})
        return assignments

    if xcoords.size:
        inferred = _infer_probe_assignments_from_geometry(kcoords, xcoords)
        if inferred:
            return inferred

    assignments: list[dict[str, Any]] = []
    groups = sorted(set(int(k) - 1 for k in kcoords.tolist() if int(k) > 0))
    if groups:
        assignments.append({"type": "staggered", "groups": groups, "x_offset": 0})
    return assignments


def _normalize_probe_assignments_for_compare(
    assignments: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for probe in assignments or []:
        groups = [int(group) for group in probe.get("groups", [])]
        normalized.append(
            {
                "type": str(probe.get("type", "staggered")),
                "groups": groups,
                "x_offset": int(round(float(probe.get("x_offset", 0) or 0))),
            }
        )
    return normalized


def _probe_assignments_json_from_chanmap(path: Path) -> list[dict[str, Any]] | None:
    data = loadmat(path)
    raw_json = data.get("probe_assignments_json")
    if raw_json is None:
        return None
    text = _decode_matlab_string(raw_json)
    if not text:
        return None
    loaded = json.loads(text)
    if not isinstance(loaded, list):
        return None
    return loaded


def _chanmap_has_current_gui_settings(
    path: Path,
    settings: PipelineGuiSettings,
) -> bool:
    try:
        existing = loadmat(path)
        stored = _probe_assignments_json_from_chanmap(path)
    except Exception:
        return False
    if _normalize_probe_assignments_for_compare(
        stored
    ) != _normalize_probe_assignments_for_compare(settings.preprocess.probe_assignments):
        return False

    basepath = settings.basepath_path
    if basepath is None:
        return False
    expected = build_channel_map_data(
        basepath=basepath,
        basename=settings.basename,
        probe_assignments=settings.preprocess.probe_assignments,
        reject_channels=settings.preprocess.reject_channels,
    )
    if expected is None:
        return False

    for key in ("chanMap0ind", "connected", "xcoords", "ycoords", "kcoords", "probe_ids"):
        if key not in existing or key not in expected:
            return False
        left = np.asarray(existing[key]).reshape(-1)
        right = np.asarray(expected[key]).reshape(-1)
        if left.shape != right.shape or not np.allclose(left, right, equal_nan=True):
            return False
    return True



def _infer_probe_assignments_from_geometry(
    kcoords: np.ndarray, xcoords: np.ndarray
) -> list[dict[str, Any]]:
    group_centers: list[tuple[int, float]] = []
    for kcoord in sorted(set(int(k) for k in kcoords.tolist() if int(k) > 0)):
        mask = kcoords == kcoord
        if not np.any(mask):
            continue
        xs = xcoords[mask]
        xs = xs[np.isfinite(xs)]
        if xs.size:
            group_centers.append((kcoord - 1, float(np.nanmedian(xs))))

    if not group_centers:
        return []

    group_centers.sort(key=lambda item: item[1])
    if len(group_centers) == 1:
        return [{"type": "staggered", "groups": [group_centers[0][0]], "x_offset": 0}]

    gaps = np.diff([x for _group, x in group_centers])
    positive_gaps = gaps[gaps > 1e-6]
    if positive_gaps.size == 0:
        groups = sorted(group for group, _x in group_centers)
        return [{"type": "staggered", "groups": groups, "x_offset": 0}]

    typical_gap = float(np.nanmedian(positive_gaps))
    split_threshold = max(300.0, typical_gap * 2.5)

    blocks: list[list[tuple[int, float]]] = [[]]
    for idx, item in enumerate(group_centers):
        if idx > 0 and (item[1] - group_centers[idx - 1][1]) > split_threshold:
            blocks.append([])
        blocks[-1].append(item)

    first_x = min(x for block in blocks for _group, x in block)
    assignments: list[dict[str, Any]] = []
    for block in blocks:
        if not block:
            continue
        groups = sorted(group for group, _x in block)
        block_x = min(x for _group, x in block)
        assignments.append(
            {
                "type": "staggered",
                "groups": groups,
                "x_offset": int(round(block_x - first_x)),
            }
        )
    return assignments


def _preprocess_result_summary(result: Any) -> dict[str, Any]:
    return {
        "local_output_dir": str(result.local_output_dir),
        "dat_path": str(result.dat_path) if result.dat_path is not None else None,
        "lfp_path": str(result.lfp_path) if result.lfp_path is not None else None,
        "session_mat_path": str(result.session_mat_path),
        "mergepoints_mat_path": str(result.mergepoints_mat_path),
        "sorter": result.sorter,
        "sorter_output_dir": str(result.sorter_output_dir) if result.sorter_output_dir else None,
    }


def _postprocess_result_summary(result: Any) -> dict[str, Any]:
    return {
        "sorting_phy_folder": str(result.sorting_phy_folder),
        "output_folder": str(result.output_folder),
        "metrics_csv_path": str(result.metrics_csv_path),
        "n_units_initial": result.n_units_initial,
        "n_units_final": result.n_units_final,
        "n_noise_clusters": result.n_noise_clusters,
    }


def _postprocess_config_from_preprocess_result(
    settings: PipelineGuiSettings, pre_result: Any
) -> PostprocessConfig:
    post_config = settings.to_postprocess_config()
    post_config.sorting_phy_folder = pre_result.sorter_output_dir or post_config.sorting_phy_folder
    post_config.sorting_search_root = pre_result.local_output_dir
    post_config.dat_path = pre_result.dat_path
    post_config.sampling_frequency = pre_result.sr
    post_config.num_channels = pre_result.n_channels
    post_config.chanmap_mat_path = settings.resolved_chanmap_path()
    post_config.reject_channels = list(pre_result.bad_channels_0based)
    return post_config


def _ensure_gui_chanmap_for_preprocess(settings: PipelineGuiSettings) -> Path:
    if settings.basepath_path is None:
        raise ValueError("basepath is required.")

    chanmap_path = settings.resolved_chanmap_path()
    if (
        chanmap_path is not None
        and chanmap_path.exists()
        and _chanmap_has_current_gui_settings(chanmap_path, settings)
    ):
        settings.chanmap_path = str(chanmap_path)
        print(f"Using chanMap: {chanmap_path}")
        return chanmap_path

    if chanmap_path is None or not chanmap_path.exists():
        print("chanMap is missing; generating before preprocess.")
    else:
        print(
            "Existing chanMap has no GUI probe assignment metadata or differs from "
            "the current GUI settings; regenerating local chanMap before preprocess."
        )

    basepath, basename, local_output_dir, _xml_path = select_paths_with_gui(
        use_gui=False,
        manual_basepath=settings.basepath_path,
        local_root=settings.local_root_path,
    )
    generated_path, bad_channels = prepare_chanmap(
        basepath=basepath,
        basename=basename,
        local_output_dir=local_output_dir,
        probe_assignments=settings.preprocess.probe_assignments,
        reject_channels=settings.preprocess.reject_channels,
    )
    settings.chanmap_path = str(generated_path)
    print(f"Generated chanMap: {generated_path}")
    print(f"Bad channels: {bad_channels}")
    return generated_path


def _run_pipeline(settings: PipelineGuiSettings, mode: RunMode) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": mode}
    pre_result = None
    if mode in ("all", "preprocess"):
        _ensure_gui_chanmap_for_preprocess(settings)
        pre_result = run_preprocess_session(settings.to_preprocess_config())
        payload["preprocess_result"] = _preprocess_result_summary(pre_result)

    if mode in ("all", "postprocess", "noise_label"):
        post_config = (
            _postprocess_config_from_preprocess_result(settings, pre_result)
            if mode == "all" and pre_result is not None
            else settings.to_postprocess_config()
        )
        if mode == "noise_label":
            post_config.noise_label_only = True
        post_results = run_postprocess_session(post_config)
        payload["postprocess_results"] = [
            _postprocess_result_summary(r) for r in post_results
        ]
    return payload


def _start_job(settings: PipelineGuiSettings, mode: RunMode) -> None:
    with STATE.lock:
        if STATE.running:
            raise RuntimeError("A job is already running.")
        STATE.running = True
        STATE.last_error = None
        STATE.last_result = None
        STATE.log.clear()

    def target() -> None:
        try:
            STATE.log.append(f"=== Running {mode} ===\n")
            with redirect_stdout(STATE.log), redirect_stderr(STATE.log):
                result = _run_pipeline(settings, mode)
            STATE.log.append("\n=== Run finished ===\n")
            with STATE.lock:
                STATE.last_result = result
        except Exception:
            error = traceback.format_exc()
            STATE.log.append("\n=== Run failed ===\n")
            STATE.log.append(error)
            with STATE.lock:
                STATE.last_error = error
        finally:
            with STATE.lock:
                STATE.running = False

    threading.Thread(target=target, daemon=True).start()


def _list_dirs(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser() if path_text else Path.cwd()
    if path.is_file():
        path = path.parent
    if not path.exists():
        path = Path.cwd()
    dirs = []
    try:
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.is_dir() and not child.name.startswith("."):
                dirs.append(str(child))
    except PermissionError:
        pass
    return {
        "path": str(path),
        "parent": str(path.parent),
        "dirs": dirs[:300],
    }


def _move_local_output_to_basepath(
    settings: PipelineGuiSettings, *, move_dat: bool, overwrite: bool
) -> dict[str, Any]:
    basepath = settings.basepath_path
    local_output_dir = settings.local_output_dir
    basename = settings.basename
    if basepath is None or not basename:
        raise ValueError("basepath is required.")
    if local_output_dir is None:
        raise ValueError("local output directory cannot be resolved.")

    src_root = local_output_dir.resolve()
    dst_root = basepath.resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Local output directory does not exist: {src_root}")
    if not dst_root.exists() or not dst_root.is_dir():
        raise NotADirectoryError(f"Basepath does not exist or is not a directory: {dst_root}")
    if src_root == dst_root:
        raise ValueError(f"Local output and basepath are identical: {src_root}")

    excluded: dict[str, str] = {
        f"{basename}.xml": "input metadata already belongs in basepath",
        f"{basename}.rhd": "input metadata already belongs in basepath",
        "@eaDir": "system metadata folder",
    }
    if not move_dat:
        excluded[f"{basename}.dat"] = "move basename.dat is off"

    move_items: list[tuple[Path, Path]] = []
    skipped: list[dict[str, str]] = []
    for child in sorted(src_root.iterdir(), key=lambda p: p.name.lower()):
        reason = excluded.get(child.name)
        if reason is None and child.suffix.lower() in {".rhd", ".xml"}:
            reason = "input metadata/raw source file stays local"
        if reason is not None:
            skipped.append({"name": child.name, "reason": reason})
            continue
        move_items.append((child, dst_root / child.name))

    conflicts = [
        dst
        for _src, dst in move_items
        if dst.exists() or dst.is_symlink()
    ]
    if conflicts and not overwrite:
        shown = "\n".join(str(path) for path in conflicts[:20])
        suffix = "\n..." if len(conflicts) > 20 else ""
        raise FileExistsError(
            "Destination already exists. Turn on overwrite existing files to replace it:\n"
            f"{shown}{suffix}"
        )

    moved: list[dict[str, str]] = []
    for src, dst in move_items:
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        set_tree_world_rw(dst)
        moved.append({"name": dst.name, "path": str(dst)})

    return {
        "basepath": str(dst_root),
        "local_output_dir": str(src_root),
        "moved": moved,
        "skipped": skipped,
        "overwrite": overwrite,
        "move_dat": move_dat,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "PreprocessWebGUI/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/defaults":
            query = parse_qs(parsed.query)
            requested_path = query.get("path", [None])[0]
            config_path = _resolve_config_path(requested_path)
            if requested_path and not config_path.exists():
                raise FileNotFoundError(f"Default config not found: {config_path}")
            settings = (
                _load_default_settings_from_path(config_path)
                if config_path.exists()
                else PipelineGuiSettings()
            )
            self._send_json(
                {
                    "settings": asdict(settings),
                    "path": str(config_path),
                    "exists": config_path.exists(),
                }
            )
            return
        if parsed.path == "/api/config_files":
            self._send_json(_list_config_files())
            return
        if parsed.path == "/api/log":
            with STATE.lock:
                payload = {
                    "running": STATE.running,
                    "log": STATE.log.read(),
                    "last_error": STATE.last_error,
                    "last_result": STATE.last_result,
                }
            self._send_json(payload)
            return
        if parsed.path == "/api/list_dirs":
            query = parse_qs(parsed.query)
            self._send_json(_list_dirs(query.get("path", [""])[0]))
            return
        if parsed.path == "/api/chanmap.png":
            query = parse_qs(parsed.query)
            path_text = query.get("path", [""])[0]
            path = Path(unquote(path_text)).expanduser()
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "chanMap not found")
                return
            self._send_bytes(_chanmap_png_bytes(path), "image/png")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/defaults":
                settings = _settings_from_payload(payload["settings"])
                path = _save_default_settings_to_path(
                    settings,
                    _resolve_config_path(payload.get("path")),
                )
                self._send_json({"saved": True, "path": str(path)})
                return
            if parsed.path == "/api/preflight":
                settings = _settings_from_payload(payload["settings"])
                mode = payload.get("mode", "all")
                checks = run_preflight(settings, mode)
                existing_chanmap = settings.resolved_chanmap_path()
                preview_data, preview_title = _preview_chanmap(settings)
                if preview_data is not None and preview_title is not None:
                    preview_summary = _chanmap_summary_from_data(
                        preview_data,
                        path=str(existing_chanmap) if existing_chanmap is not None else None,
                        exists=True,
                    )
                    preview_png = base64.b64encode(
                        _chanmap_png_bytes_from_data(preview_data, title=preview_title)
                    ).decode("ascii")
                else:
                    preview_summary = {
                        "exists": False,
                        "path": str(existing_chanmap) if existing_chanmap is not None else None,
                    }
                    preview_png = None
                self._send_json(
                    {
                        "checks": [asdict(c) for c in checks],
                        "basename": settings.basename,
                        "local_output_dir": str(settings.local_output_dir)
                        if settings.local_output_dir
                        else None,
                        "chanmap": preview_summary,
                        "chanmap_preview_png": preview_png,
                        "existing_chanmap": _chanmap_summary(existing_chanmap)
                        if existing_chanmap is not None and existing_chanmap.exists()
                        else {"exists": False, "path": None},
                    }
                )
                return
            if parsed.path == "/api/generate_chanmap":
                settings = _settings_from_payload(payload["settings"])
                if settings.basepath_path is None:
                    raise ValueError("basepath is required.")
                basepath, basename, local_output_dir, _xml_path = select_paths_with_gui(
                    use_gui=False,
                    manual_basepath=settings.basepath_path,
                    local_root=settings.local_root_path,
                )
                chanmap_path, bad_channels = prepare_chanmap(
                    basepath=basepath,
                    basename=basename,
                    local_output_dir=local_output_dir,
                    probe_assignments=settings.preprocess.probe_assignments,
                    reject_channels=settings.preprocess.reject_channels,
                )
                self._send_json(
                    {
                        "chanmap_path": str(chanmap_path),
                        "bad_channels": bad_channels,
                        "summary": _chanmap_summary(chanmap_path),
                    }
                )
                return
            if parsed.path == "/api/load_chanmap":
                settings = _settings_from_payload(payload["settings"])
                chanmap_path = settings.resolved_chanmap_path()
                if chanmap_path is None:
                    raise ValueError("basepath is required to resolve chanMap.mat.")
                if not chanmap_path.exists():
                    raise FileNotFoundError(f"chanMap not found: {chanmap_path}")
                self._send_json({"summary": _chanmap_summary(chanmap_path)})
                return
            if parsed.path == "/api/move_to_basepath":
                with STATE.lock:
                    if STATE.running:
                        raise RuntimeError("Cannot move outputs while a pipeline job is running.")
                settings = _settings_from_payload(payload["settings"])
                result = _move_local_output_to_basepath(
                    settings,
                    move_dat=bool(payload.get("move_dat", False)),
                    overwrite=bool(payload.get("overwrite", False)),
                )
                self._send_json(result)
                return
            if parsed.path == "/api/run":
                settings = _settings_from_payload(payload["settings"])
                mode = payload.get("mode", "all")
                _start_job(settings, mode)
                self._send_json({"started": True})
                return
            if parsed.path == "/api/force_stop":
                STATE.log.append("\n=== Force stop requested; exiting GUI process ===\n")
                self._send_json({"stopping": True})
                threading.Timer(0.2, lambda: os._exit(130)).start()
                return
        except Exception as exc:
            self._send_json(
                {"error": str(exc), "traceback": traceback.format_exc()},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, raw: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PreprocessPipeline Web GUI</title>
  <style>
    :root { --bg: #070b12; --panel: #111827; --panel2: #0d1422; --line: #263449; --line2: #354761; --text: #e6edf7; --muted: #98a7ba; --accent: #4da3ff; --accent2: #2563eb; --danger: #ff6b6b; --warn: #f7c948; --ok: #42d392; }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { margin: 0; font: 14px/1.4 system-ui, -apple-system, Segoe UI, sans-serif; color: var(--text); background: radial-gradient(circle at top left, #132033 0, var(--bg) 38%, #05070b 100%); display: grid; grid-template-rows: auto minmax(0, 1fr) auto; overflow: hidden; }
    button, input, select, textarea { font: inherit; }
    .top { display: grid; grid-template-columns: auto minmax(180px, 1fr) auto minmax(180px, 1fr) auto auto; gap: 8px; padding: 10px; border-bottom: 1px solid var(--line); background: rgba(17,24,39,.96); align-items: center; box-shadow: 0 8px 28px rgba(0,0,0,.35); min-width: 0; }
    input, select, textarea { width: 100%; min-width: 0; border: 1px solid var(--line2); border-radius: 5px; padding: 7px 9px; background: #0a1020; color: var(--text); outline: none; accent-color: var(--accent2); }
    input[type="checkbox"] { width: 16px; min-width: 16px; height: 16px; padding: 0; margin: 0; vertical-align: middle; }
    input:focus, select:focus, textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(77,163,255,.18); }
    input:disabled, select:disabled, textarea:disabled { opacity: .52; cursor: not-allowed; }
    textarea { min-height: 120px; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; resize: vertical; }
    button { border: 1px solid var(--line2); border-radius: 5px; padding: 7px 11px; background: linear-gradient(#1b2638, #121b2b); color: var(--text); cursor: pointer; white-space: nowrap; }
    button:hover { border-color: var(--accent); background: linear-gradient(#24344d, #162237); }
    button.primary { background: linear-gradient(135deg, #2386ff, #1d4ed8); color: #fff; border-color: #3b82f6; }
    button.danger { color: var(--danger); border-color: #7f2d36; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .main { display: grid; grid-template-columns: minmax(410px, 520px) minmax(420px, 1fr); gap: 12px; padding: 12px; min-height: 0; overflow: hidden; }
    .panel { background: rgba(17,24,39,.94); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 10px 35px rgba(0,0,0,.24); min-height: 0; }
    .tabs { display: flex; border-bottom: 1px solid var(--line); background: #0a1020; }
    .tab { border: 0; border-right: 1px solid var(--line); border-radius: 0; background: transparent; color: var(--muted); padding: 9px 13px; }
    .tab.active { background: var(--panel); color: var(--text); font-weight: 650; box-shadow: inset 0 2px 0 var(--accent); }
    .tabbody { display: none; padding: 10px; overflow: auto; height: calc(100% - 38px); }
    .tabbody.active { display: block; }
    .grid { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 9px 11px; align-items: center; margin-bottom: 14px; }
    .grid .wide { grid-column: 1 / -1; }
    .form-one { display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; margin-bottom: 14px; }
    .field { display: grid; grid-template-columns: 165px minmax(0, 1fr); gap: 10px; align-items: center; }
    .field > label:first-child { color: #d4deeb; }
    .field-inline { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; min-width: 0; }
    .hint { margin: -2px 0 4px; color: var(--muted); font-size: 12px; }
    .probe-editor { display: grid; gap: 8px; }
    .probe-head { display: grid; grid-template-columns: minmax(110px, 1fr) minmax(130px, 1fr) 86px 34px; gap: 8px; color: var(--muted); font-size: 12px; }
    .probe-row { display: grid; grid-template-columns: minmax(110px, 1fr) minmax(130px, 1fr) 86px 34px; gap: 8px; align-items: center; }
    .icon-btn { width: 34px; padding: 7px 0; }
    label { color: #d4deeb; }
    label.check { display: inline-flex; align-items: center; gap: 8px; min-height: 34px; line-height: 1.2; }
    .section { font-weight: 750; letter-spacing: .02em; color: #f4f8ff; margin: 16px 0 8px; padding-top: 11px; border-top: 1px solid var(--line); }
    .center { display: grid; grid-template-columns: minmax(340px, 1fr) minmax(300px, 420px); gap: 12px; min-height: 0; overflow: hidden; }
    .monitor-column { display: grid; grid-template-rows: minmax(240px, 0.58fr) minmax(210px, 0.42fr); gap: 12px; min-height: 0; overflow: hidden; }
    .chanmap { padding: 10px; display: flex; flex-direction: column; min-height: 0; }
    .chanmap-head { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .chanmap-head .spacer { flex: 1; }
    .zoom-btn { padding: 3px 8px; min-width: 30px; }
    .badge { border: 1px solid var(--line2); border-radius: 999px; padding: 2px 8px; font-size: 12px; color: var(--muted); }
    .badge.fresh { color: var(--ok); border-color: rgba(66,211,146,.45); }
    .badge.stale { color: var(--warn); border-color: rgba(247,201,72,.45); }
    .chanmap-view { flex: 1; min-height: 0; overflow: auto; border: 1px solid var(--line); border-radius: 6px; background: #0b1220; cursor: grab; }
    .chanmap-view.dragging { cursor: grabbing; }
    .chanmap img { display: block; width: 100%; height: auto; max-width: none; min-height: 0; background: #0b1220; }
    .side-output { display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }
    .mini-panel { display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; overflow: hidden; border: 1px solid var(--line); border-radius: 6px; background: #070c15; }
    .mini-head { padding: 7px 10px; border-bottom: 1px solid var(--line); color: #f4f8ff; font-weight: 700; background: #0a1020; }
    .preview { padding: 9px 11px; overflow-y: auto; overflow-x: hidden; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; color: #cbd7e6; min-height: 0; background: rgba(10,16,32,.88); }
    .runbar { display: flex; gap: 8px; justify-content: flex-end; align-items: center; flex-wrap: wrap; padding: 9px 12px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: rgba(10,16,32,.96); box-shadow: 0 -6px 20px rgba(0,0,0,.18); }
    .runbar .spacer { flex: 1 1 auto; }
    .runbar .move-options { display: inline-flex; gap: 14px; align-items: center; flex-wrap: wrap; padding-right: 4px; }
    .log { height: 100%; min-height: 0; padding: 11px 12px; margin: 0; overflow: auto; background: #05080e; color: #dce5ee; border: 0; font: 12px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .status-ok { color: var(--ok); } .status-warn { color: var(--warn); } .status-error { color: var(--danger); }
    .browser { position: fixed; inset: 7% 12%; background: var(--panel); border: 1px solid var(--line2); border-radius: 8px; box-shadow: 0 18px 60px rgba(0,0,0,.48); display: none; flex-direction: column; z-index: 10; }
    .browser.open { display: flex; }
    .browser-head { display: flex; gap: 8px; padding: 10px; border-bottom: 1px solid var(--line); align-items: center; background: #0a1020; }
    .dirlist { overflow: auto; padding: 8px; }
    .dirrow { display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid var(--line); border-radius: 0; padding: 7px; background: transparent; color: var(--text); }
    .dirrow:hover { background: #152238; }
  </style>
</head>
<body>
  <div class="top">
    <button onclick="openBrowser('basepath')">Browse basepath</button>
    <input id="basepath" placeholder="/path/to/session">
    <button onclick="openBrowser('local_root')">Browse local root</button>
    <input id="local_root">
    <button onclick="openConfigBrowser('load')">Load default</button>
    <button onclick="openConfigBrowser('save')">Save default</button>
  </div>

  <div class="main">
    <div class="panel">
      <div class="tabs">
        <button class="tab active" data-tab="pre" onclick="showTab('pre')">Preprocess setting</button>
        <button class="tab" data-tab="post" onclick="showTab('post')">Postprocess setting</button>
      </div>
      <div id="tab-pre" class="tabbody active">
        <div class="form-one">
          <div class="section">Session and channel map</div>
          <div class="hint">Set output overwrite behavior, channel exclusions, and channel-map geometry. Existing local/basepath chanMap is used only to initialize these fields; the preview follows the current settings.</div>
          <label class="check" title="Overwrite reusable preprocess outputs in the local output folder. Sorting still creates a new timestamped folder for each run."><input id="pre_overwrite" type="checkbox"> overwrite preprocess outputs</label>
          <div class="field"><label title="0-based channels to mark as disconnected in chanMap.">bad channels</label><input id="bad_channels" placeholder="0, 3, 17"></div>
          <div class="probe-editor">
            <div class="hint">Probe assignment rows define geometry per XML group set. Groups are 0-based XML group indices.</div>
            <div class="probe-head"><span>geometry</span><span>XML groups</span><span>x offset</span><span></span></div>
            <div id="probe_rows"></div>
            <div><button onclick="addProbeRow()">Add probe assignment</button></div>
          </div>

          <div class="section">Inputs</div>
          <div class="hint">Choose auxiliary streams and whether to preserve raw amplifier data.</div>
          <label class="check" title="Export digital input events such as TTL timestamps."><input id="digital_inputs" type="checkbox"> digital inputs</label>
          <label class="check" title="Export analog input channels when available."><input id="analog_inputs" type="checkbox"> analog inputs</label>

          <div class="section">Signal processing</div>
          <div class="hint">Configure filtering and referencing before writing the cleaned .dat.</div>
          <label class="check" title="Apply bandpass filtering and common reference before writing the cleaned .dat."><input id="do_preprocess" type="checkbox"> do preprocess</label>
          <div class="field"><label title="Parallel worker count for preprocessing writes and artifact detection. Large values are faster but can overload shared machines.">workers for preprocess</label><input id="preprocess_worker_count" type="number" min="1"></div>
          <div class="field"><label title="Low cutoff for spike-band filtering.">bandpass min Hz</label><input id="bandpass_min_hz" type="number" step="0.1"></div>
          <div class="field"><label title="High cutoff for spike-band filtering.">bandpass max Hz</label><input id="bandpass_max_hz" type="number" step="0.1"></div>
          <div class="field"><label title="Common median reference mode. none skips CMR after bandpass; local uses nearby channels; global uses all selected channels.">common median reference</label><select id="reference"><option>none</option><option>local</option><option>global</option></select></div>
          <div class="field"><label title="Inner radius for local common median reference, in micrometers.">CMR radius min (um)</label><input id="local_radius_min_um" type="number" step="1"></div>
          <div class="field"><label title="Outer radius for local common median reference, in micrometers.">CMR radius max (um)</label><input id="local_radius_max_um" type="number" step="1"></div>

          <div class="section">LFP and state scoring</div>
          <div class="hint">Write downsampled LFP and optionally run sleep/wake state scoring.</div>
          <label class="check" title="Write a downsampled LFP file."><input id="make_lfp" type="checkbox"> make LFP</label>
          <div class="field"><label title="Sampling rate for the generated LFP file.">LFP fs</label><input id="lfp_fs" type="number" step="1"></div>
          <label class="check" title="Run sleep/wake state scoring from the generated LFP output."><input id="state_score" type="checkbox"> state score</label>
          <div class="field"><label title="Optional slow-wave channels, 0-based and comma separated. Leave blank for automatic selection.">slow-wave channels</label><input id="sw_channels" placeholder="auto"></div>
          <div class="field"><label title="Optional theta channels, 0-based and comma separated. Leave blank for automatic selection.">theta channels</label><input id="theta_channels" placeholder="auto"></div>
          <div class="field"><label title="Sleep scoring window size in seconds. Notebook default is 2.">state window sec</label><input id="state_window_sec" type="number" step="0.1"></div>
          <div class="field"><label title="Sleep scoring smoothing factor. Notebook default is 15.">state smooth factor</label><input id="state_smoothfact" type="number" step="0.1"></div>
          <div class="field"><label title="Minimum state duration in seconds after post-processing.">min state length sec</label><input id="state_min_state_length" type="number" step="0.1"></div>
          <div class="field"><label title="Wake bouts up to this duration are treated as microarousals/interruption.">microarousal sec</label><input id="state_microarousal_sec" type="number" step="0.1"></div>
          <div class="field"><label title="EMG threshold multiplier. Higher values make high-EMG wake detection stricter.">EMG threshold alpha</label><input id="emg_th_alpha" type="number" step="0.1"></div>
          <label class="check" title="Require low EMG for NREM; high-EMG slow-wave periods stay awake."><input id="useEMG_NREM" type="checkbox"> use EMG for NREM</label>
          <label class="check" title="Block direct Wake to REM transitions during state post-processing."><input id="state_block_wake_to_rem" type="checkbox"> block wake to REM</label>

          <div class="section">TTL artifact removal</div>
          <div class="hint">Optionally remove stimulation-locked artifacts around digital input edges.</div>
          <label class="check" title="Enable TTL-locked artifact removal. Default is off."><input id="remove_ttl_artifacts" type="checkbox"> remove TTL artifacts</label>
          <div class="field"><label title="How TTL artifact interpolation is grouped: all, probe, or shank.">TTL group mode</label><select id="artifact_ttl_group_mode"><option>all</option><option>probe</option><option>shank</option></select></div>
          <div class="field"><label title="0-based digital input bit used for TTL artifact timing.">TTL channel</label><input id="artifact_ttl_channel" type="number" min="0" max="15"></div>
          <label class="check" title="Also remove windows around TTL falling edges."><input id="artifact_ttl_include_offset" type="checkbox"> include TTL offset</label>
          <div class="field"><label title="Artifact interpolation window before each TTL event, in milliseconds.">TTL ms before</label><input id="artifact_ttl_ms_before" type="number" step="0.1"></div>
          <div class="field"><label title="Artifact interpolation window after each TTL event, in milliseconds.">TTL ms after</label><input id="artifact_ttl_ms_after" type="number" step="0.1"></div>
          <div class="field"><label title="Interpolation mode used inside removed TTL windows.">TTL interpolation mode</label><select id="artifact_ttl_mode"><option>linear</option><option>cubic</option><option>0</option></select></div>

          <div class="section">High-amplitude artifact removal</div>
          <div class="hint">Optionally detect large noise transients after preprocessing and interpolate around them.</div>
          <label class="check" title="Enable high-amplitude artifact detection and removal. Default is off."><input id="remove_highamp_artifacts" type="checkbox"> remove high-amplitude artifacts</label>
          <div class="field"><label title="How high-amplitude artifact interpolation is grouped: all, probe, or shank.">High amp group mode</label><select id="artifact_highamp_group_mode"><option>all</option><option>probe</option><option>shank</option></select></div>
          <div class="field"><label title="Detection threshold in robust noise sigma units.">High amp sigma</label><input id="highamp_threshold_sigma" type="number" step="0.1"></div>
          <div class="field"><label title="Removal window before each high-amplitude event, in milliseconds.">High amp ms before</label><input id="highamp_ms_before" type="number" step="0.1"></div>
          <div class="field"><label title="Removal window after each high-amplitude event, in milliseconds.">High amp ms after</label><input id="highamp_ms_after" type="number" step="0.1"></div>
          <div class="field"><label title="Interpolation mode used inside removed high-amplitude windows.">High amp interpolation mode</label><select id="highamp_mode"><option>linear</option><option>cubic</option><option>0</option></select></div>

          <div class="section">Sorter and runtime</div>
          <div class="hint">Choose whether to run sorting after preprocessing and set runtime paths.</div>
          <label class="check" title="Run spike sorting after preprocessing. Turn off to write preprocess outputs only."><input id="run_sorter" type="checkbox"> run sorter</label>
          <div class="field"><label title="Sorter to run after preprocessing. Disabled writes preprocess outputs only.">sorter</label><select id="sorter" onchange="sorterChanged()"><option>Kilosort</option><option>Kilosort2_5</option><option>kilosort4</option><option>disabled</option></select></div>
          <div class="field"><label title="Path to the selected sorter implementation.">sorter path</label><input id="sorter_path"></div>
          <div class="field"><label title="YAML config file for the selected sorter.">sorter config</label><input id="sorter_config_path"></div>
          <div class="field"><label title="MATLAB executable used for MATLAB-based sorters.">MATLAB path</label><input id="matlab_path"></div>
          <div class="field"><label title="Parallel worker count passed to sorter-specific runtime settings such as MATLAB max workers.">workers for sorter</label><input id="sorter_worker_count" type="number" min="1"></div>
        </div>
      </div>
      <div id="tab-post" class="tabbody">
        <div class="grid">
          <div class="wide section">Postprocess target</div>
          <div class="wide hint">Default: use the newest Kilosort folder for this basepath under local output.</div>
          <label>sorting folder</label><div class="field-inline"><input id="sorting_phy_folder"><button onclick="openBrowser('sorting_phy_folder')">Browse</button></div>
          <input id="sorting_search_root" type="hidden">
          <div class="wide section">Recording data</div>
          <label class="check wide" title="Enable only when basename.dat is a legacy raw concatenated binary. Leave off for basename.dat generated by this pipeline."><input id="post_apply_preprocess" type="checkbox"> basename.dat is raw; apply preprocessing before postprocess</label>
          <div class="wide hint">Leave off when basename.dat was generated by this pipeline. Turn on only for legacy raw concatenated basename.dat.</div>
          <div class="wide section">Curation and metrics</div>
          <label>exclude groups</label><input id="exclude_cluster_groups" placeholder="noise, mua">
          <label>duplicate censor ms</label><input id="duplicate_censored_period_ms" type="number" step="0.1">
          <label>duplicate threshold</label><input id="duplicate_threshold" type="number" step="0.01">
          <label>merge min spikes</label><input id="merge_min_spikes" type="number">
          <label>merge corr diff</label><input id="merge_corr_diff_thresh" type="number" step="0.01">
          <label>merge template diff</label><input id="merge_template_diff_thresh" type="number" step="0.01">
          <label>split contamination</label><input id="split_contamination" type="number" step="0.01">
          <label>split threshold mode</label><select id="split_threshold_mode"><option>adaptive_chi2</option><option>chi2</option><option>quantile</option></select>
          <label>split wf threshold</label><input id="split_wf_threshold" type="number" step="0.01">
          <label>split wf n chans</label><input id="split_wf_n_chans" type="number">
          <label>split amp MAD scale</label><input id="split_amp_mad_scale" type="number" step="0.1">
          <label>workers</label><input id="post_worker_count" type="number" min="1">
          <label class="check"><input id="post_overwrite" type="checkbox"> overwrite curation/metrics outputs</label><span></span>
          <div class="wide section">Noise labeling thresholds</div>
          <div class="wide hint">Clusters are marked noise when any enabled metric crosses its threshold. Waveform thresholds use template metrics computed after curation.</div>
          <label title="Mark noise when firing rate is less than or equal to this Hz value.">firing rate <= Hz</label><input id="noise_firing_rate_lt" type="number" step="0.001">
          <label title="Mark noise when ISI violations ratio exceeds this value.">ISI violation ratio ></label><input id="noise_isi_violations_ratio_gt" type="number" step="0.1">
          <label title="Mark noise when ISI violation count exceeds this value.">ISI violation count ></label><input id="noise_isi_violations_count_gt" type="number" step="1">
          <label title="Mark noise when presence ratio is below this value.">presence ratio <</label><input id="noise_presence_ratio_lt" type="number" step="0.01">
          <label title="Mark noise when SNR is below this value.">SNR <</label><input id="noise_snr_lt" type="number" step="0.1">
          <label title="Mark noise when absolute median amplitude is below this microvolt value.">amplitude median abs < uV</label><input id="noise_amplitude_median_lt" type="number" step="1">
          <label title="Mark noise when absolute median amplitude is above this microvolt value.">amplitude median abs > uV</label><input id="noise_amplitude_median_gt" type="number" step="1">
          <label title="Mark noise when peak-to-valley waveform duration exceeds this milliseconds value.">peak-to-valley > ms</label><input id="noise_peak_to_valley_gt" type="number" step="0.01">
          <label title="Mark noise when peak/trough ratio is below this value.">peak/trough ratio <</label><input id="noise_peak_trough_ratio_lt" type="number" step="0.01">
          <label title="Mark noise when waveform half-width exceeds this milliseconds value.">half-width > ms</label><input id="noise_halfwidth_gt" type="number" step="0.01">
          <label title="Mark noise when repolarization slope is below this value.">repolarization slope <</label><input id="noise_slope_lt" type="number" step="1">
          <div class="wide"><button onclick="startRun('noise_label')" title="Only update noise labels from existing quality_metrics.csv. Requires full postprocess output to already exist.">Run noise labeling only</button></div>
        </div>
      </div>
    </div>

    <div class="center">
      <div class="monitor-column">
        <div class="panel chanmap">
          <div class="chanmap-head">
            <strong>chanMap Preview</strong>
            <span id="chanmap_status" class="badge">file preview</span>
            <span class="spacer"></span>
            <button class="zoom-btn" title="Zoom out" onclick="setChanmapZoom(chanmapZoom / 1.2)">-</button>
            <button class="zoom-btn" title="Reset zoom" onclick="setChanmapZoom(1)">Reset</button>
            <button class="zoom-btn" title="Zoom in" onclick="setChanmapZoom(chanmapZoom * 1.2)">+</button>
            <span id="chanmap_zoom_label" class="badge">100%</span>
          </div>
          <div id="chanmap_view" class="chanmap-view" title="Scroll to zoom. Drag scrollbars to pan.">
            <img id="chanmap_img" alt="chanMap preview">
          </div>
        </div>
        <div class="panel side-output">
          <div class="mini-head">Log</div>
          <pre id="log" class="log"></pre>
        </div>
      </div>
      <div class="panel side-output">
        <div class="mini-head">Setting config</div>
        <div id="preview" class="preview"></div>
      </div>
    </div>
  </div>

  <div class="runbar">
    <button class="primary" onclick="startRun('all')">Run all process</button>
    <button onclick="startRun('preprocess')">Run preprocess only</button>
    <button onclick="startRun('postprocess')">Run postprocess only</button>
    <span class="spacer"></span>
    <span class="move-options">
      <label class="check" title="Also move the large preprocessed basename.dat file from local output to basepath."><input id="move_basename_dat" type="checkbox"> move basename.dat</label>
      <label class="check" title="Replace existing destination files or folders in basepath. Leave off to stop before moving if any destination already exists."><input id="move_overwrite" type="checkbox"> overwrite existing files</label>
    </span>
    <button onclick="moveToBasepath()" title="Move local output files and folders for this session back to basepath after local curation work.">Move to basepath</button>
    <button class="danger" onclick="forceStopRun()" title="Force quit the GUI Python process. Use only when a run must be stopped immediately.">Force stop</button>
    <button onclick="clearLog()">Clear log</button>
  </div>

  <div id="browser" class="browser">
    <div class="browser-head">
      <button onclick="closeBrowser()">Close</button>
      <button onclick="chooseCurrentDir()">Use this folder</button>
      <button onclick="browseUp()">Up</button>
      <input id="browser_path" onkeydown="if(event.key==='Enter') loadDirs(this.value)">
    </div>
    <div id="dirlist" class="dirlist"></div>
  </div>

  <div id="config_browser" class="browser">
    <div class="browser-head">
      <button onclick="closeConfigBrowser()">Close</button>
      <button id="config_confirm" onclick="confirmConfigBrowser()">Load</button>
      <input id="config_path" placeholder="config/preprocess_gui_default_config.json">
    </div>
    <div class="hint" style="padding: 8px 12px 0;">Default config files are stored under the repository config folder. Enter a new .json filename to save another default.</div>
    <div id="config_filelist" class="dirlist"></div>
  </div>

  <script>
    let settings = null;
    let browseTarget = null;
    let currentBrowsePath = "";
    let configBrowserMode = "load";
    let lastAppliedChanmapPath = null;
    let chanmapZoom = 1;
    let runWasRunning = false;

    async function api(path, body=null) {
      const opts = body ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)} : {};
      const res = await fetch(path, opts);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    }
    function id(x) { return document.getElementById(x); }
    function val(x) { return id(x).value; }
    function checked(x) { return id(x).checked; }
    function num(x) { return Number(id(x).value); }
    function intList(text) { return text.split(',').map(s => s.trim()).filter(Boolean).map(Number); }
    function jsString(value) { return String(value).replaceAll("\\", "\\\\").replaceAll("'", "\\'"); }
    function cloneSettings(value) { return JSON.parse(JSON.stringify(value)); }
    function resetChanmapAutoload() { lastAppliedChanmapPath = null; }
    function isChanmapContextField(target) {
      return target && ['basepath', 'local_root', 'sorting_phy_folder', 'sorting_search_root'].includes(target.id);
    }
    function mergeDefaultSettingsWithSession(defaultSettings, currentSettings) {
      const merged = cloneSettings(defaultSettings);
      merged.basepath = currentSettings.basepath || '';
      merged.local_root = currentSettings.local_root || '';
      merged.chanmap_path = currentSettings.chanmap_path || '';
      merged.postprocess = merged.postprocess || {};
      const currentPost = currentSettings.postprocess || {};
      merged.postprocess.sorting_phy_folder = currentPost.sorting_phy_folder || '';
      merged.postprocess.sorting_search_root = currentPost.sorting_search_root || '';
      return merged;
    }
    const probeTypes = ['middle_finger', 'staggered', 'poly2', 'poly3', 'poly5', 'linear', 'neurogrid', 'double_sided', 'NeuroPixel'];
    const noiseThresholdKeys = [
      'isi_violations_ratio_gt',
      'isi_violations_count_gt',
      'presence_ratio_lt',
      'snr_lt',
      'amplitude_median_lt',
      'amplitude_median_gt',
      'peak_to_valley_gt',
      'peak_trough_ratio_lt',
      'halfwidth_gt',
      'slope_lt',
      'firing_rate_lt'
    ];

    function probeRowsToAssignments() {
      return Array.from(document.querySelectorAll('.probe-row')).map(row => ({
        type: row.querySelector('.probe-type').value,
        groups: intList(row.querySelector('.probe-groups').value),
        x_offset: Number(row.querySelector('.probe-x-offset').value || 0)
      })).filter(p => p.groups.length > 0);
    }
    function collectNoiseThresholds() {
      const values = {};
      for (const key of noiseThresholdKeys) {
        const el = id('noise_' + key);
        if (el && el.value !== '') values[key] = Number(el.value);
      }
      return values;
    }
    function applyNoiseThresholds(values = {}) {
      for (const key of noiseThresholdKeys) {
        const el = id('noise_' + key);
        if (el) el.value = values[key] ?? '';
      }
    }
    function setChanmapStatus(text, cls = '') {
      const el = id('chanmap_status');
      el.textContent = text;
      el.className = 'badge' + (cls ? ' ' + cls : '');
    }
    function setChanmapZoom(value, anchor = null) {
      const view = id('chanmap_view');
      const img = id('chanmap_img');
      const oldZoom = chanmapZoom;
      chanmapZoom = Math.max(0.35, Math.min(8, value));
      img.style.width = `${chanmapZoom * 100}%`;
      id('chanmap_zoom_label').textContent = `${Math.round(chanmapZoom * 100)}%`;
      if (anchor && oldZoom > 0) {
        const scale = chanmapZoom / oldZoom;
        view.scrollLeft = (view.scrollLeft + anchor.x) * scale - anchor.x;
        view.scrollTop = (view.scrollTop + anchor.y) * scale - anchor.y;
      }
    }
    function installChanmapWheelZoom() {
      const view = id('chanmap_view');
      view.addEventListener('wheel', (event) => {
        if (!id('chanmap_img').getAttribute('src')) return;
        event.preventDefault();
        const rect = view.getBoundingClientRect();
        const anchor = {x: event.clientX - rect.left, y: event.clientY - rect.top};
        const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
        setChanmapZoom(chanmapZoom * factor, anchor);
      }, {passive: false});
    }
    function installChanmapDragPan() {
      const view = id('chanmap_view');
      let dragging = false;
      let startX = 0;
      let startY = 0;
      let startLeft = 0;
      let startTop = 0;
      view.addEventListener('mousedown', (event) => {
        if (!id('chanmap_img').getAttribute('src')) return;
        dragging = true;
        startX = event.clientX;
        startY = event.clientY;
        startLeft = view.scrollLeft;
        startTop = view.scrollTop;
        view.classList.add('dragging');
        event.preventDefault();
      });
      window.addEventListener('mousemove', (event) => {
        if (!dragging) return;
        view.scrollLeft = startLeft - (event.clientX - startX);
        view.scrollTop = startTop - (event.clientY - startY);
      });
      window.addEventListener('mouseup', () => {
        dragging = false;
        view.classList.remove('dragging');
      });
    }
    function addProbeRow(assignment = null, refresh = true) {
      const root = id('probe_rows');
      const p = assignment || {type: 'staggered', groups: [], x_offset: 0};
      const row = document.createElement('div');
      row.className = 'probe-row';
      row.innerHTML = `
        <select class="probe-type" title="Probe geometry type used for these XML groups.">${probeTypes.map(t => `<option${t === p.type ? ' selected' : ''}>${t}</option>`).join('')}</select>
        <input class="probe-groups" title="0-based XML group indices assigned to this probe, comma separated." placeholder="0,1,2,3" value="${(p.groups || []).join(', ')}">
        <input class="probe-x-offset" title="Horizontal offset for this probe in micrometers." type="number" step="1" value="${p.x_offset || 0}">
        <button class="icon-btn" title="Remove this probe assignment" onclick="this.closest('.probe-row').remove(); refreshPreview();">-</button>
      `;
      row.querySelectorAll('input,select').forEach(el => el.addEventListener('input', refreshPreview));
      row.querySelectorAll('input,select').forEach(el => el.addEventListener('change', refreshPreview));
      root.appendChild(row);
      if (refresh) refreshPreview();
    }
    function renderProbeRows(assignments, refresh = true) {
      const root = id('probe_rows');
      root.innerHTML = '';
      const rows = assignments && assignments.length ? assignments : [{type: 'staggered', groups: [0,1,2,3,4,5,6,7], x_offset: 0}];
      rows.forEach(row => addProbeRow(row, false));
      if (refresh) refreshPreview();
    }
    function collectSettings() {
      const probeAssignments = probeRowsToAssignments();
      return {
        basepath: val('basepath'),
        local_root: val('local_root'),
        chanmap_path: '',
        preprocess: {
          analog_inputs: checked('analog_inputs'),
          digital_inputs: checked('digital_inputs'),
          save_raw: false,
          do_preprocess: checked('do_preprocess'),
          bandpass_min_hz: num('bandpass_min_hz'),
          bandpass_max_hz: num('bandpass_max_hz'),
          reference: val('reference'),
          local_radius_um: [num('local_radius_min_um'), num('local_radius_max_um')],
          make_lfp: checked('make_lfp'),
          lfp_fs: num('lfp_fs'),
          state_score: checked('state_score'),
          sw_channels: intList(val('sw_channels')),
          theta_channels: intList(val('theta_channels')),
          state_winparms: [num('state_window_sec'), num('state_smoothfact')],
          emg_th_alpha: num('emg_th_alpha'),
          useEMG_NREM: checked('useEMG_NREM'),
          state_min_state_length: num('state_min_state_length'),
          state_microarousal_sec: num('state_microarousal_sec'),
          state_block_wake_to_rem: checked('state_block_wake_to_rem'),
          remove_ttl_artifacts: checked('remove_ttl_artifacts'),
          artifact_ttl_group_mode: val('artifact_ttl_group_mode'),
          artifact_ttl_channel: num('artifact_ttl_channel'),
          artifact_ttl_include_offset: checked('artifact_ttl_include_offset'),
          artifact_ttl_ms_before: num('artifact_ttl_ms_before'),
          artifact_ttl_ms_after: num('artifact_ttl_ms_after'),
          artifact_ttl_mode: val('artifact_ttl_mode'),
          remove_highamp_artifacts: checked('remove_highamp_artifacts'),
          artifact_highamp_group_mode: val('artifact_highamp_group_mode'),
          highamp_threshold_sigma: num('highamp_threshold_sigma'),
          highamp_ms_before: num('highamp_ms_before'),
          highamp_ms_after: num('highamp_ms_after'),
          highamp_mode: val('highamp_mode'),
          reject_channels: intList(val('bad_channels')),
          probe_assignments: probeAssignments,
          run_sorter: checked('run_sorter'),
          sorter: val('sorter'),
          sorter_path: val('sorter_path'),
          sorter_config_path: val('sorter_config_path'),
          matlab_path: val('matlab_path'),
          preprocess_worker_count: num('preprocess_worker_count'),
          sorter_worker_count: num('sorter_worker_count'),
          overwrite: checked('pre_overwrite')
        },
        postprocess: {
          sorting_phy_folder: val('sorting_phy_folder'),
          sorting_search_root: val('sorting_search_root'),
          apply_preprocess: checked('post_apply_preprocess'),
          exclude_cluster_groups: val('exclude_cluster_groups').split(',').map(s => s.trim()).filter(Boolean),
          duplicate_censored_period_ms: num('duplicate_censored_period_ms'),
          duplicate_threshold: num('duplicate_threshold'),
          merge_min_spikes: num('merge_min_spikes'),
          merge_corr_diff_thresh: num('merge_corr_diff_thresh'),
          merge_template_diff_thresh: num('merge_template_diff_thresh'),
          split_contamination: num('split_contamination'),
          split_threshold_mode: val('split_threshold_mode'),
          split_wf_threshold: num('split_wf_threshold'),
          split_wf_n_chans: num('split_wf_n_chans'),
          split_amp_mad_scale: num('split_amp_mad_scale'),
          skip_pc_metrics: true,
          noise_label_only: false,
          noise_thresholds: collectNoiseThresholds(),
          overwrite: checked('post_overwrite'),
          worker_count: num('post_worker_count')
        }
      };
    }
    function applySettings(s, refresh = true) {
      settings = s;
      id('basepath').value = s.basepath || '';
      id('local_root').value = s.local_root || '';
      const p = s.preprocess, pp = s.postprocess;
      for (const k of ['analog_inputs','digital_inputs','do_preprocess','make_lfp','state_score']) id(k).checked = !!p[k];
      id('remove_ttl_artifacts').checked = !!p.remove_ttl_artifacts && p.artifact_ttl_group_mode !== 'none';
      id('remove_highamp_artifacts').checked = !!p.remove_highamp_artifacts && p.artifact_highamp_group_mode !== 'none';
      id('run_sorter').checked = p.run_sorter !== false && (p.sorter || 'disabled') !== 'disabled';
      for (const k of ['useEMG_NREM','state_block_wake_to_rem']) id(k).checked = !!p[k];
      id('pre_overwrite').checked = !!p.overwrite;
      for (const k of ['bandpass_min_hz','bandpass_max_hz','lfp_fs','artifact_ttl_channel','artifact_ttl_ms_before','artifact_ttl_ms_after','highamp_threshold_sigma','highamp_ms_before','highamp_ms_after','emg_th_alpha','state_min_state_length','state_microarousal_sec']) id(k).value = p[k];
      id('sw_channels').value = (p.sw_channels || []).join(', ');
      id('theta_channels').value = (p.theta_channels || []).join(', ');
      id('state_window_sec').value = (p.state_winparms || [2, 15])[0];
      id('state_smoothfact').value = (p.state_winparms || [2, 15])[1];
      id('reference').value = p.reference || 'local';
      id('local_radius_min_um').value = (p.local_radius_um || [20, 200])[0];
      id('local_radius_max_um').value = (p.local_radius_um || [20, 200])[1];
      id('artifact_ttl_group_mode').value = p.artifact_ttl_group_mode === 'none' ? 'all' : p.artifact_ttl_group_mode; id('artifact_ttl_include_offset').checked = !!p.artifact_ttl_include_offset;
      id('artifact_ttl_mode').value = p.artifact_ttl_mode; id('artifact_highamp_group_mode').value = p.artifact_highamp_group_mode === 'none' ? 'shank' : p.artifact_highamp_group_mode; id('highamp_mode').value = p.highamp_mode;
      id('bad_channels').value = p.reject_channels.join(', ');
      renderProbeRows(p.probe_assignments, false);
      id('sorter').value = p.sorter || 'disabled'; id('sorter_path').value = p.sorter_path; id('sorter_config_path').value = p.sorter_config_path; id('matlab_path').value = p.matlab_path;
      id('preprocess_worker_count').value = p.preprocess_worker_count || p.worker_count || 1;
      id('sorter_worker_count').value = p.sorter_worker_count || p.worker_count || 1;
      updateSorterEnabled();
      id('sorting_phy_folder').value = pp.sorting_phy_folder; id('sorting_search_root').value = pp.sorting_search_root; id('post_apply_preprocess').checked = !!pp.apply_preprocess;
      id('exclude_cluster_groups').value = pp.exclude_cluster_groups.join(', ');
      for (const k of ['duplicate_censored_period_ms','duplicate_threshold','merge_min_spikes','merge_corr_diff_thresh','merge_template_diff_thresh','split_contamination','split_wf_threshold','split_wf_n_chans','split_amp_mad_scale']) id(k).value = pp[k];
      applyNoiseThresholds(pp.noise_thresholds || {});
      id('split_threshold_mode').value = pp.split_threshold_mode; id('post_overwrite').checked = !!pp.overwrite; id('post_worker_count').value = pp.worker_count;
      if (refresh) refreshPreview();
    }
    function showTab(name) {
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
      document.querySelectorAll('.tabbody').forEach(t => t.classList.toggle('active', t.id === 'tab-' + name));
      refreshPreview();
    }
    function currentPreviewMode() {
      return document.querySelector('.tab.active')?.dataset.tab === 'post' ? 'postprocess' : 'all';
    }
    async function refreshPreview() {
      try {
        let s = collectSettings();
        const data = await api('/api/preflight', {settings: s, mode: currentPreviewMode()});
        const existing = data.existing_chanmap || {};
        if (existing.exists && existing.path && existing.path !== lastAppliedChanmapPath) {
          applyChanmapSummaryToGui(existing, false);
          lastAppliedChanmapPath = existing.path;
          await refreshPreview();
          return;
        }
        const lines = [`Basepath: ${s.basepath || '-'}`, `Basename: ${data.basename || '-'}`, `Local output: ${data.local_output_dir || '-'}`, `chanMap target: ${data.chanmap.path || '-'}`];
        if (existing.exists && existing.path) lines.push(`Existing chanMap loaded from: ${existing.path}`);
        lines.push('', 'Preflight:');
        for (const c of data.checks) lines.push(`[${c.status.toUpperCase()}] ${c.label}: ${c.detail}`);
        if (data.chanmap.exists) {
          lines.push('', `chanMap channels=${data.chanmap.channels}, connected=${data.chanmap.connected}, bad=${data.chanmap.bad_count}`);
          id('chanmap_img').src = 'data:image/png;base64,' + data.chanmap_preview_png;
          setChanmapZoom(chanmapZoom);
          setChanmapStatus('settings preview', 'fresh');
        } else {
          id('chanmap_img').removeAttribute('src');
          setChanmapZoom(1);
          setChanmapStatus('no preview', 'stale');
        }
        id('preview').textContent = lines.join('\n');
      } catch (e) {
        id('preview').textContent = 'Config error:\n' + e.message;
      }
    }
    function applyChanmapSummaryToGui(summary, refresh = true) {
      if (!summary || !summary.exists) return;
      id('bad_channels').value = (summary.bad_channels || []).join(', ');
      if (summary.probe_assignments && summary.probe_assignments.length) {
        renderProbeRows(summary.probe_assignments, refresh);
      }
    }
    async function startRun(mode) {
      try {
        const current = collectSettings();
        const preflight = await api('/api/preflight', {settings: current, mode});
        const errors = preflight.checks.filter(c => c.status === 'error');
        if (errors.length) {
          alert(errors.map(c => `${c.label}: ${c.detail}`).join('\n'));
          await refreshPreview();
          return;
        }
        await api('/api/run', {settings: current, mode});
        runWasRunning = true;
      } catch (e) { alert(e.message); }
    }
    async function moveToBasepath() {
      try {
        const current = collectSettings();
        const moveDat = checked('move_basename_dat');
        const overwrite = checked('move_overwrite');
        const message = [
          'Move local output files to basepath?',
          '',
          `Basepath: ${current.basepath || '-'}`,
          `Move basename.dat: ${moveDat ? 'yes' : 'no'}`,
          `Overwrite existing files: ${overwrite ? 'yes' : 'no'}`
        ].join('\n');
        if (!confirm(message)) return;
        const data = await api('/api/move_to_basepath', {
          settings: current,
          move_dat: moveDat,
          overwrite
        });
        const lines = [
          'Move to basepath finished',
          `Basepath: ${data.basepath}`,
          `Local output: ${data.local_output_dir}`,
          '',
          `Moved (${data.moved.length}):`
        ];
        lines.push(...(data.moved.length ? data.moved.map(item => `- ${item.name}`) : ['- none']));
        lines.push('', `Skipped (${data.skipped.length}):`);
        lines.push(...(data.skipped.length ? data.skipped.map(item => `- ${item.name}: ${item.reason}`) : ['- none']));
        id('preview').textContent = lines.join('\n');
      } catch (e) { alert(e.message); }
    }
    async function forceStopRun() {
      const ok = confirm([
        'Force stop the running GUI process?',
        '',
        'This immediately exits Python and can leave partially written output files.',
        'Restart the GUI after it stops.'
      ].join('\n'));
      if (!ok) return;
      try {
        await api('/api/force_stop', {});
        id('preview').textContent = 'Force stop requested. The GUI process will exit.';
      } catch (e) {
        id('preview').textContent = 'Force stop requested. The GUI process may already have exited.';
      }
    }
    async function pollLog() {
      try {
        const data = await api('/api/log');
        id('log').textContent = data.log || '';
        id('log').scrollTop = id('log').scrollHeight;
        if (data.running) {
          runWasRunning = true;
        } else if (runWasRunning) {
          runWasRunning = false;
          resetChanmapAutoload();
          refreshPreview();
        }
      } catch (e) {}
      setTimeout(pollLog, 1500);
    }
    function clearLog() { id('log').textContent = ''; }
    async function saveDefaultConfig(path = null) {
      try {
        const data = await api('/api/defaults', {settings: collectSettings(), path});
        id('preview').textContent = `Saved default config:\n${data.path}`;
      } catch (e) { alert(e.message); }
    }
    async function loadDefaultConfig(path = null) {
      try {
        const url = path ? '/api/defaults?path=' + encodeURIComponent(path) : '/api/defaults';
        const data = await api(url);
        const current = settings ? collectSettings() : null;
        const nextSettings = current ? mergeDefaultSettingsWithSession(data.settings, current) : data.settings;
        resetChanmapAutoload();
        applySettings(nextSettings, false);
        id('preview').textContent = `${data.exists ? 'Loaded' : 'Using built-in'} default config:\n${data.path}`;
        await refreshPreview();
      } catch (e) { alert(e.message); }
    }
    function sorterChanged() {
      const map = {Kilosort: ['sorter/KiloSort1','sorter/Kilosort1_config.yaml'], Kilosort2_5: ['sorter/Kilosort2.5','sorter/Kilosort2.5_config.yaml'], kilosort4: ['sorter/Kilosort4','sorter/Kilosort4_config.yaml'], disabled: ['','']};
      const pair = map[val('sorter')] || ['','']; id('sorter_path').value = pair[0]; id('sorter_config_path').value = pair[1];
      if (val('sorter') === 'disabled') id('run_sorter').checked = false;
      updateSorterEnabled();
      refreshPreview();
    }
    function updateSorterEnabled() {
      const enabled = checked('run_sorter');
      for (const k of ['sorter','sorter_path','sorter_config_path','matlab_path']) id(k).disabled = !enabled;
    }
    function openBrowser(target) { browseTarget = target; id('browser').classList.add('open'); loadDirs(val(target) || val('basepath') || '/workdir/ys2375'); }
    function closeBrowser() { id('browser').classList.remove('open'); }
    async function loadDirs(path) {
      const data = await api('/api/list_dirs?path=' + encodeURIComponent(path));
      currentBrowsePath = data.path; id('browser_path').value = data.path;
      id('dirlist').innerHTML = data.dirs.map(d => `<button class="dirrow" onclick="loadDirs('${jsString(d)}')">${d}</button>`).join('');
    }
    function browseUp() { loadDirs(currentBrowsePath.split('/').slice(0, -1).join('/') || '/'); }
    function chooseCurrentDir() {
      if (browseTarget) {
        id(browseTarget).value = currentBrowsePath;
        if (['basepath', 'local_root', 'sorting_phy_folder', 'sorting_search_root'].includes(browseTarget)) resetChanmapAutoload();
      }
      closeBrowser();
      refreshPreview();
    }
    async function openConfigBrowser(mode) {
      configBrowserMode = mode;
      id('config_confirm').textContent = mode === 'save' ? 'Save' : 'Load';
      id('config_browser').classList.add('open');
      await loadConfigFiles();
    }
    function closeConfigBrowser() { id('config_browser').classList.remove('open'); }
    async function loadConfigFiles() {
      const data = await api('/api/config_files');
      id('config_path').value = data.default_path;
      id('config_filelist').innerHTML = data.files.map(p => {
        const label = p.replace(data.config_dir + '/', 'config/');
        return `<button class="dirrow" onclick="selectConfigPath('${jsString(p)}')">${label}</button>`;
      }).join('') || '<div class="hint">No config JSON files found.</div>';
    }
    function selectConfigPath(path) { id('config_path').value = path; }
    async function confirmConfigBrowser() {
      const path = val('config_path');
      closeConfigBrowser();
      if (configBrowserMode === 'save') {
        await saveDefaultConfig(path);
      } else {
        await loadDefaultConfig(path);
      }
    }
    document.addEventListener('input', (e) => {
      if (e.target.id === 'run_sorter') updateSorterEnabled();
      if (isChanmapContextField(e.target)) resetChanmapAutoload();
      if (e.target.matches('input,select,textarea')) refreshPreview();
    });
    document.addEventListener('change', (e) => {
      if (e.target.id === 'run_sorter') updateSorterEnabled();
      if (isChanmapContextField(e.target)) resetChanmapAutoload();
      if (e.target.matches('input,select,textarea')) refreshPreview();
    });
    installChanmapWheelZoom();
    installChanmapDragPan();
    api('/api/defaults').then(data => applySettings(data.settings)); pollLog();
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    global DEFAULT_CONFIG_PATH

    parser = argparse.ArgumentParser(description="Run the PreprocessPipeline browser GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred port. Use 0 for an OS-assigned free port.")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Default GUI config JSON to load/save. Relative paths are resolved "
            "from the repository config folder."
        ),
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to open the GUI in a browser automatically.",
    )
    args = parser.parse_args(argv)

    if args.config:
        DEFAULT_CONFIG_PATH = _resolve_config_path(args.config)

    server = _bind_server_with_fallback(args.host, args.port)
    actual_port = int(server.server_address[1])
    open_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{open_host}:{actual_port}"
    print(f"PreprocessPipeline web GUI running at {url}")
    if actual_port != args.port and args.port != 0:
        print(f"Preferred port {args.port} was busy; using {actual_port}.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        print("Opening browser automatically. If it does not open, use the URL above.")
    else:
        print("Browser auto-open disabled. Use the URL above.")
    print("In VS Code Remote, forward this port if prompted and open it in Simple Browser.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web GUI")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
