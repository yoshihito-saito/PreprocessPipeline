from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

import numpy as np
from scipy.io import loadmat

from PySide6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.preprocess.behavior import (
    dlc_point_names,
    discover_dlc_files,
    inspect_dlc_ttl_sync,
    load_representative_frame,
    load_dlc_tracking,
    process_dlc_behavior,
)
from src.preprocess import prepare_chanmap, select_paths_with_gui
from src.preprocess.io import (
    A5X12_16_BUZ_LIN_PROBE_TYPE,
    _normalize_chanmap_layout,
    build_channel_map_data,
    derive_probe_assignments_from_xml,
    discover_subsessions,
    save_cell_explorer_chan_coords,
    set_tree_world_rw,
)
from src.preprocess.multiday import MULTI_DAY_MANIFEST, discover_multi_day_subepochs
from src.preprocess.paths import find_project_root, resolve_project_path
from src.worker_defaults import default_worker_count, normalize_worker_count
from src.execution.backends import SlurmCapabilities, detect_slurm_capabilities
from src.execution.controller import create_run, resolve_backend
from src.execution.models import BackendName, RequestedBackend, StageName, StageStatus
from src.execution.store import RunStore, read_json

from .config_model import (
    BehaviorGuiSettings,
    PipelineGuiSettings,
    PostprocessGuiSettings,
    PreprocessGuiSettings,
    ExecutionGuiSettings,
    StageResourceGuiSettings,
    RunMode,
    parse_float_pair,
    parse_int_list,
    postprocess_output_folder_for_sorting,
    resolve_existing_session_settings,
    load_local_session_resume,
)
from .anatomical_map import (
    AnatomicalChannelGroup,
    AnatomicalMapError,
    channel_groups_from_chanmap_data,
    load_anatomical_map_csv,
    save_anatomical_map_csv,
)
from .preflight import CheckResult, run_preflight
from .run_pipeline import ERROR_PREFIX, RESULT_PREFIX


REPO_ROOT = find_project_root()
CONFIG_DIR = REPO_ROOT / "config"
PUBLIC_DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocess_gui_default_config.json"
LOCAL_DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocess_gui_default_config.local.json"
VENDORED_CELLEXPLORER_ROOT = REPO_ROOT / "external" / "CellExplorer"
MATLAB_SUPPORT_ROOT = REPO_ROOT / "external" / "matlab"
ANATOMICAL_MAP_FILENAME = "anatomical_map.csv"


def _resolve_cell_explorer_source_basepath(
    settings: PipelineGuiSettings,
    local_output_dir: Path,
) -> Path:
    """Resolve the raw waveform root, preferring authoritative multi-day staging."""
    local_output_dir = Path(local_output_dir).expanduser().resolve()
    manifest_path = local_output_dir / MULTI_DAY_MANIFEST
    if not manifest_path.exists():
        source = settings.preprocess_source_path or settings.basepath_path
        if source is None:
            raise ValueError("Cannot resolve the CellExplorer source basepath.")
        return Path(source).expanduser().resolve()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read the multi-day manifest for CellExplorer: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid multi-day manifest object: {manifest_path}")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported multi-day manifest schema for CellExplorer: "
            f"{manifest.get('schema_version')!r} ({manifest_path})"
        )

    manifest_name = str(manifest.get("name") or "").strip()
    if manifest_name != local_output_dir.name:
        raise ValueError(
            "Multi-day manifest name does not match the selected local session: "
            f"manifest={manifest_name or '<missing>'}, session={local_output_dir.name}"
        )

    server_text = str(manifest.get("server_basepath") or "").strip()
    if not server_text:
        raise ValueError(
            f"Multi-day manifest is missing server_basepath: {manifest_path}"
        )
    server_basepath = Path(server_text).expanduser()
    if not server_basepath.is_dir():
        raise FileNotFoundError(
            f"Multi-day staging root is missing: {server_basepath}"
        )

    subepochs = manifest.get("subepochs")
    if not isinstance(subepochs, list) or not subepochs:
        raise ValueError(
            f"Multi-day manifest has no subepochs: {manifest_path}"
        )
    missing: list[Path] = []
    for index, entry in enumerate(subepochs, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid multi-day subepoch entry {index}: {manifest_path}"
            )
        staged_text = str(entry.get("staged_subepoch_path") or "").strip()
        staged_name = Path(staged_text).name if staged_text else ""
        if not staged_name:
            raise ValueError(
                f"Multi-day subepoch entry {index} is missing staged_subepoch_path: "
                f"{manifest_path}"
            )
        staged_path = server_basepath / staged_name
        if not staged_path.is_dir():
            missing.append(staged_path)
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = "" if len(missing) <= 3 else f", ... ({len(missing)} missing)"
        raise FileNotFoundError(
            "Multi-day staged subepochs required by CellExplorer are missing: "
            f"{preview}{suffix}"
        )
    return server_basepath.resolve()


def _default_config_path() -> Path:
    override = os.environ.get("PREPROCESS_GUI_DEFAULT_CONFIG")
    if override:
        return Path(override).expanduser()
    if LOCAL_DEFAULT_CONFIG_PATH.exists():
        return LOCAL_DEFAULT_CONFIG_PATH
    return PUBLIC_DEFAULT_CONFIG_PATH


DEFAULT_CONFIG_PATH = _default_config_path()


def _default_config_has_backend_choice(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    execution = payload.get("execution") if isinstance(payload, dict) else None
    return isinstance(execution, dict) and "requested_backend" in execution


def _has_slurm_server_commands(
    capabilities: SlurmCapabilities, *, require_sacct: bool
) -> bool:
    required = (
        capabilities.sbatch,
        capabilities.squeue,
        capabilities.scancel,
        capabilities.sacct if require_sacct else True,
    )
    return all(required)

PROBE_TYPES = (
    A5X12_16_BUZ_LIN_PROBE_TYPE,
    "Buzsaki 5x12",
    "flex-G5",
    "staggered",
    "poly2",
    "poly3",
    "poly5",
    "linear",
    "neurogrid",
    "double_sided",
    "NeuroPixel",
)

SORTER_DEFAULTS: dict[str, tuple[str, str]] = {
    "Kilosort": ("sorter/KiloSort1", "sorter/Kilosort1_config.yaml"),
    "Kilosort2_5": ("sorter/Kilosort2.5", "sorter/Kilosort2.5_config.yaml"),
    "kilosort4": ("sorter/Kilosort4", "sorter/Kilosort4_config.yaml"),
    "disabled": ("", ""),
}

NOISE_THRESHOLD_FIELDS = (
    ("firing_rate_lt", "firing rate <= Hz"),
    ("isi_violations_ratio_gt", "ISI violation ratio >"),
    ("isi_violations_count_gt", "ISI violation count >"),
    ("presence_ratio_lt", "presence ratio <"),
    ("snr_lt", "SNR <"),
    ("amplitude_median_lt", "amplitude median abs < uV"),
    ("amplitude_median_gt", "amplitude median abs > uV"),
)


def _settings_as_parameter_defaults(settings: PipelineGuiSettings) -> PipelineGuiSettings:
    defaults = PipelineGuiSettings.from_json(settings.to_json())
    defaults.basepath = ""
    defaults.local_root = ""
    defaults.source_basepath = ""
    defaults.existing_session_dir = ""
    defaults.chanmap_path = ""
    defaults.preprocess.reject_channels = []
    defaults.preprocess.matlab_path = ""
    defaults.execution.matlab_path = ""
    defaults.execution.workspace = ""
    defaults.execution.shared_workspace_acknowledged = False
    defaults.postprocess.sorting_phy_folder = ""
    defaults.postprocess.sorting_search_root = ""
    return defaults


def _load_default_settings(path: Path = DEFAULT_CONFIG_PATH) -> PipelineGuiSettings:
    return PipelineGuiSettings.load(path) if path.exists() else PipelineGuiSettings()


def _save_default_settings(settings: PipelineGuiSettings, path: Path = DEFAULT_CONFIG_PATH) -> Path:
    defaults = _settings_as_parameter_defaults(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults.save(path)
    return path


def _default_output_storage_dir(settings: PipelineGuiSettings) -> Path | None:
    basepath = settings.basepath_path
    if basepath is None:
        return None
    if settings.multi_day_enabled and settings.multi_day_name.strip():
        return (basepath.resolve().parent / settings.multi_day_name.strip()).resolve()
    return basepath.resolve()


def _move_local_output_to_storage(
    settings: PipelineGuiSettings,
    *,
    move_dat: bool,
    overwrite: bool,
    clean_after_move: bool,
    destination_dir: Path | None = None,
    source_dir: Path | None = None,
    source_basename: str | None = None,
) -> dict[str, Any]:
    local_output_dir = source_dir or settings.local_output_dir
    basename = source_basename or settings.basename
    storage_dir = destination_dir or _default_output_storage_dir(settings)
    if storage_dir is None or not basename:
        raise ValueError("A storage destination and basename are required.")
    if local_output_dir is None:
        raise ValueError("local output directory cannot be resolved.")

    src_root = local_output_dir.resolve()
    dst_root = storage_dir.expanduser().resolve()
    if not src_root.exists() or not src_root.is_dir():
        raise FileNotFoundError(f"Local output directory does not exist: {src_root}")
    if destination_dir is None and settings.multi_day_enabled:
        dst_root.mkdir(parents=False, exist_ok=True)
    if not dst_root.exists() or not dst_root.is_dir():
        raise NotADirectoryError(
            f"Storage destination does not exist or is not a directory: {dst_root}"
        )
    if src_root == dst_root:
        raise ValueError(f"Local output and storage destination are identical: {src_root}")
    try:
        src_root.relative_to(dst_root)
    except ValueError:
        try:
            dst_root.relative_to(src_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"Storage destination cannot be inside local output: {dst_root}")
    else:
        raise ValueError(f"Local output cannot be inside storage destination: {src_root}")

    # A persistent Run marker associates this output folder with immutable Run
    # records, which can live in an external workspace.  Relocating only the
    # session tree would leave those records inconsistent, so reject before
    # staging or changing either tree rather than mutating the Run in place.
    claim_path = src_root / ".pipeline-active-run.json"
    if claim_path.exists():
        raise ValueError(
            "Cannot move an output folder with .pipeline-active-run.json. "
            "Resume or resolve the persistent Run before moving its outputs."
        )

    session_xml_name = f"{basename}.xml"
    destination_xml = dst_root / session_xml_name
    destination_has_xml = destination_xml.exists() or destination_xml.is_symlink()
    excluded: dict[str, str] = {
        f"{basename}.rhd": "input metadata already belongs in basepath",
        "@eaDir": "system metadata folder",
    }
    if not move_dat:
        excluded[f"{basename}.dat"] = "move basename.dat is off"

    move_items: list[tuple[Path, Path]] = []
    skipped: list[dict[str, str]] = []
    for child in sorted(src_root.iterdir(), key=lambda p: p.name.lower()):
        reason = excluded.get(child.name)
        if reason is None and child.name == session_xml_name and destination_has_xml:
            reason = "destination session XML already exists"
        elif (
            reason is None
            and child.suffix.lower() == ".xml"
            and child.name != session_xml_name
        ):
            reason = "input metadata XML stays local"
        elif reason is None and child.suffix.lower() == ".rhd":
            reason = "input metadata/raw source file stays local"
        if reason is not None:
            skipped.append({"name": child.name, "reason": reason})
            continue
        move_items.append((child, dst_root / child.name))

    conflicts = [dst for _src, dst in move_items if dst.exists() or dst.is_symlink()]
    if conflicts and not overwrite:
        shown = "\n".join(str(path) for path in conflicts[:20])
        suffix = "\n..." if len(conflicts) > 20 else ""
        raise FileExistsError(
            "Destination already exists. Enable overwrite to replace it:\n"
            f"{shown}{suffix}"
        )

    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _content_signature(path: Path) -> tuple[tuple[str, str, str], ...]:
        """Return a content-aware, symlink-safe inventory rooted at *path*."""
        entries: list[tuple[str, str, str]] = []

        def digest(file_path: Path) -> str:
            hasher = hashlib.sha256()
            with file_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            return hasher.hexdigest()

        def visit(item: Path, relative: Path) -> None:
            if item.is_symlink():
                entries.append((relative.as_posix(), "symlink", os.readlink(item)))
            elif item.is_dir():
                entries.append((relative.as_posix(), "directory", ""))
                for child in sorted(item.iterdir(), key=lambda candidate: candidate.name):
                    visit(child, relative / child.name)
            elif item.is_file():
                entries.append((relative.as_posix(), "file", digest(item)))
            else:
                raise ValueError(f"Unsupported output item for transactional move: {item}")

        visit(path, Path("."))
        return tuple(entries)

    def _rewrite_relocated_paths(path: Path) -> None:
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json", ".py"}:
            return
        if path.name == "preprocess_run.yaml":
            # The completed-record execution workspace may be a valid external
            # Run workspace.  It is not relocated with this session tree and
            # must stay usable after recovery; update only session-referencing
            # values in the completed record.
            import yaml

            payload = yaml.safe_load(path.read_text(encoding="utf-8"))

            def rewrite(value: Any, *, key: str | None = None) -> Any:
                if isinstance(value, dict):
                    return {name: rewrite(item, key=str(name)) for name, item in value.items()}
                if isinstance(value, list):
                    return [rewrite(item) for item in value]
                if isinstance(value, str) and key != "workspace":
                    return value.replace(str(src_root), str(dst_root))
                return value

            rewritten = rewrite(payload)
            if rewritten != payload:
                tmp = path.with_name(f".{path.name}.move-rewrite-{uuid.uuid4().hex}")
                tmp.write_text(yaml.safe_dump(rewritten, sort_keys=False), encoding="utf-8")
                os.replace(tmp, path)
            # Validate that the rewritten file is parseable before publication.
            if not isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), (dict, list, type(None))):
                raise ValueError(f"Rewritten YAML metadata has an invalid root: {path}")
            return
        if path.suffix.lower() == ".json":
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
            rewritten = text.replace(str(src_root), str(dst_root))
            if rewritten != text:
                tmp = path.with_name(f".{path.name}.move-rewrite-{uuid.uuid4().hex}")
                tmp.write_text(rewritten, encoding="utf-8", errors="surrogateescape")
                os.replace(tmp, path)
            json.loads(path.read_text(encoding="utf-8"))
            return
        text = path.read_text(encoding="utf-8", errors="surrogateescape")
        rewritten = text.replace(str(src_root), str(dst_root))
        if rewritten != text:
            tmp = path.with_name(f".{path.name}.move-rewrite-{uuid.uuid4().hex}")
            tmp.write_text(rewritten, encoding="utf-8", errors="surrogateescape")
            os.replace(tmp, path)

    inventory_move = [{"name": src.name, "bytes": _path_size(src)} for src, _dst in move_items]

    # A custom storage location must retain the minimal raw metadata required to
    # reopen a session after local cleanup. The session XML is already either at
    # the destination or in move_items; RHD metadata retains the established
    # copy-without-counting semantics.
    metadata_copies: list[tuple[Path, Path]] = []
    if destination_dir is not None and clean_after_move:
        xml_src = src_root / session_xml_name
        if not destination_has_xml and not xml_src.exists():
            raise FileNotFoundError(
                f"Custom storage needs {session_xml_name} before local cleanup, but it is absent "
                f"from both the source and destination: {dst_root}"
            )
        for name in (f"{basename}.rhd",):
            src = src_root / name
            dst = dst_root / name
            if not dst.exists():
                if not src.exists():
                    raise FileNotFoundError(
                        f"Custom storage needs {name} before local cleanup, but it is absent from both "
                        f"the source and destination: {dst_root}"
                    )
                metadata_copies.append((src, dst))

    staging: Path | None = None
    for _attempt in range(10):
        candidate = dst_root / f".{basename}.move-staging-{uuid.uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        staging = candidate
        break
    if staging is None:
        raise FileExistsError(f"Could not allocate a unique move staging directory under {dst_root}")
    backups = staging / "backups"
    published: list[tuple[Path, Path | None]] = []
    moved: list[dict[str, str]] = []
    staged_signatures: dict[Path, tuple[tuple[str, str, str], ...]] = {}
    publication_skipped_sources: set[Path] = set()
    try:
        # Stage a byte-for-byte copy before modifying either canonical tree.
        for src, dst in move_items + metadata_copies:
            staged = staging / src.name
            if src.is_dir() and not src.is_symlink():
                shutil.copytree(src, staged, symlinks=True)
            else:
                shutil.copy2(src, staged, follow_symlinks=False)
            # First prove that the raw copy is exact.  Metadata rewriting below
            # is intentional and therefore validated separately.
            if _content_signature(src) != _content_signature(staged):
                raise IOError(f"Staged transfer validation failed for {src}")
            for candidate in [staged, *staged.rglob("*")] if staged.is_dir() else [staged]:
                _rewrite_relocated_paths(candidate)
            staged_signatures[dst] = _content_signature(staged)

        # Publish only after every item has been copied and validated.  Existing
        # destinations are held in the staging tree so a later failure restores
        # the exact prior destination rather than leaving a partial move behind.
        for src, dst in move_items + metadata_copies:
            staged = staging / src.name
            backup: Path | None = None
            if src.name == session_xml_name:
                try:
                    if staged.is_symlink():
                        os.symlink(os.readlink(staged), dst)
                    elif staged.is_file():
                        # Staging is inside dst_root, so this same-filesystem
                        # hard link publishes atomically without replacing an
                        # XML another process may have created concurrently.
                        os.link(staged, dst)
                    else:
                        raise ValueError(
                            f"Session XML must be a file or symlink: {src}"
                        )
                except FileExistsError:
                    publication_skipped_sources.add(src)
                    skipped.append(
                        {
                            "name": src.name,
                            "reason": "destination session XML appeared during transfer",
                        }
                    )
                    continue
                except OSError:
                    if staged.is_symlink():
                        raise
                    try:
                        descriptor = os.open(
                            dst,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            staged.stat().st_mode & 0o777,
                        )
                    except FileExistsError:
                        publication_skipped_sources.add(src)
                        skipped.append(
                            {
                                "name": src.name,
                                "reason": "destination session XML appeared during transfer",
                            }
                        )
                        continue
                    published.append((dst, None))
                    with os.fdopen(descriptor, "wb") as destination_handle:
                        with staged.open("rb") as source_handle:
                            shutil.copyfileobj(source_handle, destination_handle)
                        destination_handle.flush()
                        os.fsync(destination_handle.fileno())
                    shutil.copystat(staged, dst, follow_symlinks=False)
                else:
                    published.append((dst, None))
                staged.unlink()
                moved.append(
                    {"name": dst.name, "path": str(dst), "bytes": _path_size(dst)}
                )
                continue
            if dst.exists() or dst.is_symlink():
                if not overwrite:
                    # Metadata can coexist only when it already existed; output
                    # conflicts were rejected above.
                    if (src, dst) in metadata_copies:
                        continue
                    raise FileExistsError(f"Destination appeared during move: {dst}")
                backups.mkdir(exist_ok=True)
                backup = backups / dst.name
                os.replace(dst, backup)
            os.replace(staged, dst)
            published.append((dst, backup))
            if (src, dst) in move_items:
                moved.append({"name": dst.name, "path": str(dst), "bytes": _path_size(dst)})

        # Verify the canonical destination after publication before deleting the
        # source.  This makes injected copy/publish failures recoverable.
        for src, dst in move_items + metadata_copies:
            if src in publication_skipped_sources:
                continue
            if _content_signature(dst) != staged_signatures[dst]:
                raise IOError(f"Published transfer validation failed for {dst}")

        # Make the storage root and all published or pre-existing content
        # collaborative. Directories need execute permission for traversal;
        # regular files need read/write permission for every user.
        set_tree_world_rw(dst_root)
    except Exception:
        for dst, backup in reversed(published):
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            if backup is not None and backup.exists():
                os.replace(backup, dst)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    cleaned = False
    if clean_after_move:
        try:
            if src_root.exists():
                shutil.rmtree(src_root)
            cleaned = True
        except Exception as exc:
            raise OSError(
                "Destination publication succeeded, but local source cleanup failed; "
                f"the destination remains valid at {dst_root}: {exc}"
            ) from exc

    return {
        "basepath": str(dst_root),
        "storage_dir": str(dst_root),
        "local_output_dir": str(src_root),
        "moved": moved,
        "skipped": skipped,
        "inventory": {
            "move": [
                item
                for item in inventory_move
                if (src_root / item["name"]) not in publication_skipped_sources
            ],
            "retain": skipped,
            "delete_local": clean_after_move,
        },
        "overwrite": overwrite,
        "move_dat": move_dat,
        "clean_after_move": clean_after_move,
        "cleaned": cleaned,
    }


def _move_local_output_to_basepath(
    settings: PipelineGuiSettings, *, move_dat: bool, overwrite: bool, clean_after_move: bool
) -> dict[str, Any]:
    """Backward-compatible wrapper for the former fixed-basepath operation."""
    return _move_local_output_to_storage(
        settings,
        move_dat=move_dat,
        overwrite=overwrite,
        clean_after_move=clean_after_move,
    )


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class HorizontalOnlyScrollArea(QScrollArea):
    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        content = self.widget()
        if content is None:
            return
        content.resize(
            max(content.minimumSizeHint().width(), self.viewport().width()),
            self.viewport().height(),
        )


class ChanMapCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10.0, 4.8), facecolor="#2f2f2f")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(280, 180)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.summary = QLabel("No chanMap loaded")
        self.summary.setWordWrap(True)
        self._ax: Any | None = None
        self._full_xlim: tuple[float, float] | None = None
        self._full_ylim: tuple[float, float] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._selection_patch: Rectangle | None = None
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.summary)
        self.show_empty()

    def show_empty(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.90)
        self._ax = ax
        self._full_xlim = None
        self._full_ylim = None
        self._drag_start = None
        self._selection_patch = None
        ax.set_facecolor("#252525")
        ax.text(
            0.5,
            0.5,
            "Generate or load chanMap.mat",
            ha="center",
            va="center",
            color="#d4d4d4",
        )
        ax.set_axis_off()
        self.canvas.draw_idle()
        self.summary.setText("No chanMap loaded")

    def load_chanmap(self, path: Path) -> None:
        if not path.exists():
            self.show_empty()
            self.summary.setText(f"chanMap not found: {path}")
            return
        self.render_chanmap(loadmat(path), source=path)

    def render_chanmap(self, data: dict[str, Any], *, source: Path | str) -> None:
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

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.07, right=0.995, bottom=0.16, top=0.90)
        self._ax = ax
        self._drag_start = None
        self._selection_patch = None
        ax.set_facecolor("#252525")
        probe_keys = [int(p) for p in probe_ids.tolist()]
        unique_probes = sorted(set(probe_keys))
        palette = [
            "#80deea",
            "#b39ddb",
            "#a5d6a7",
            "#ffcc80",
            "#90caf9",
            "#ce93d8",
            "#c5e1a5",
            "#f48fb1",
            "#bcaaa4",
            "#fff59d",
            "#9fa8da",
            "#ffab91",
            "#b0bec5",
            "#81c784",
            "#64b5f6",
            "#e6ee9c",
            "#f8bbd0",
            "#d7ccc8",
            "#b2dfdb",
            "#d1c4e9",
        ]
        color_by_probe = {
            probe: palette[idx % len(palette)] for idx, probe in enumerate(unique_probes)
        }
        point_colors = np.array([color_by_probe[probe] for probe in probe_keys], dtype=object)
        ax.scatter(
            x[connected],
            y[connected],
            c=point_colors[connected],
            s=42,
            edgecolor="#252525",
            linewidth=0.3,
            clip_on=True,
            zorder=2,
        )
        if np.any(~connected):
            ax.scatter(
                x[~connected],
                y[~connected],
                c="red",
                marker="x",
                s=70,
                linewidth=1.8,
                clip_on=True,
                zorder=3,
            )

        if n <= 256:
            for xi, yi, ch, is_connected in zip(x, y, device_ch, connected):
                color = "#e5e7eb" if is_connected else "#f87171"
                ax.text(
                    float(xi),
                    float(yi),
                    str(int(ch)),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color=color,
                    clip_on=False,
                    zorder=4,
                )

        source_text = str(source)
        title = Path(source_text).name if source_text and source_text != "current settings" else source_text
        ax.set_title(title, color="#f5f5f5")
        ax.set_xlabel("x (um)", color="#d4d4d4")
        ax.set_ylabel("y (um)", color="#d4d4d4")
        ax.tick_params(colors="#a3a3a3")
        for spine in ax.spines.values():
            spine.set_color("#404040")
        ax.set_aspect("equal", adjustable="box", anchor="C")
        ax.grid(True, alpha=0.25, color="#737373")
        finite_x = x[np.isfinite(x)]
        finite_y = y[np.isfinite(y)]
        if finite_x.size > 0 and finite_y.size > 0:
            x_min = float(np.min(finite_x))
            x_max = float(np.max(finite_x))
            y_min = float(np.min(finite_y))
            y_max = float(np.max(finite_y))
            x_range = max(1.0, x_max - x_min)
            y_range = max(1.0, y_max - y_min)
            x_pad = max(180.0, x_range * 0.20)
            y_pad = max(180.0, y_range * 0.20)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        self._full_xlim = tuple(float(v) for v in ax.get_xlim())
        self._full_ylim = tuple(float(v) for v in ax.get_ylim())
        self.canvas.draw_idle()

        bad = device_ch[~connected].astype(int).tolist()
        probes = sorted(set(int(v) for v in probe_ids.tolist()))
        shanks = sorted(set(int(v) for v in kcoords.tolist()))
        self.summary.setText(
            f"{source_text}\n"
            f"channels={n}, connected={int(np.sum(connected))}, bad={len(bad)}, "
            f"probes={len(probes)}, groups/shanks={len(shanks)}\n"
            f"bad channels: {bad[:40]}{' ...' if len(bad) > 40 else ''}"
        )

    def _on_scroll(self, event: Any) -> None:
        ax = self._ax
        if ax is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        step = float(getattr(event, "step", 0.0) or 0.0)
        zoom_in = event.button == "up" or step > 0
        scale = 0.8 if zoom_in else 1.25
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_center = float(event.xdata)
        y_center = float(event.ydata)
        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale
        x_rel = (x_center - xlim[0]) / (xlim[1] - xlim[0])
        y_rel = (y_center - ylim[0]) / (ylim[1] - ylim[0])
        if new_width <= 1e-12 or new_height <= 1e-12:
            return
        ax.set_xlim(x_center - new_width * x_rel, x_center + new_width * (1.0 - x_rel))
        ax.set_ylim(y_center - new_height * y_rel, y_center + new_height * (1.0 - y_rel))
        self.canvas.draw_idle()

    def _on_button_press(self, event: Any) -> None:
        ax = self._ax
        if ax is None or event.inaxes is not ax:
            return
        if event.dblclick:
            self.reset_view()
            return
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        self._drag_start = (float(event.xdata), float(event.ydata))
        if self._selection_patch is not None:
            self._selection_patch.remove()
        self._selection_patch = Rectangle(
            self._drag_start,
            0,
            0,
            facecolor="#ef4b2d22",
            edgecolor="#ef4b2d",
            linewidth=1.2,
            linestyle="-",
        )
        ax.add_patch(self._selection_patch)
        self.canvas.draw_idle()

    def _on_motion(self, event: Any) -> None:
        if self._drag_start is None or self._selection_patch is None:
            return
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._drag_start
        x1 = float(event.xdata)
        y1 = float(event.ydata)
        self._selection_patch.set_x(min(x0, x1))
        self._selection_patch.set_y(min(y0, y1))
        self._selection_patch.set_width(abs(x1 - x0))
        self._selection_patch.set_height(abs(y1 - y0))
        self.canvas.draw_idle()

    def _on_button_release(self, event: Any) -> None:
        ax = self._ax
        if ax is None or self._drag_start is None or self._selection_patch is None:
            return
        x0, y0 = self._drag_start
        x1 = event.xdata
        y1 = event.ydata
        patch = self._selection_patch
        patch.remove()
        self._selection_patch = None
        self._drag_start = None
        if event.inaxes is ax and x1 is not None and y1 is not None:
            x1 = float(x1)
            y1 = float(y1)
            if abs(x1 - x0) > 1e-9 and abs(y1 - y0) > 1e-9:
                ax.set_xlim(min(x0, x1), max(x0, x1))
                ax.set_ylim(min(y0, y1), max(y0, y1))
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        ax = self._ax
        if ax is None or self._full_xlim is None or self._full_ylim is None:
            return
        if self._selection_patch is not None:
            self._selection_patch.remove()
            self._selection_patch = None
        self._drag_start = None
        ax.set_xlim(*self._full_xlim)
        ax.set_ylim(*self._full_ylim)
        self.canvas.draw_idle()


class BrainRegionProbeWidget(QWidget):
    def __init__(
        self,
        groups: list[AnatomicalChannelGroup],
        channel_regions: dict[int, str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._groups = list(groups)
        self._channel_regions = dict(channel_regions)
        self._selected_channels: set[int] = set()
        self._drag_preview_channels: set[int] = set()
        self._dot_hits: dict[int, tuple[float, float, float]] = {}
        self._drag_start: QPointF | None = None
        self._drag_current: QPointF | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.setMinimumWidth(980)
        self.setMinimumHeight(520)
        self.setMouseTracking(True)

    @property
    def selected_channels(self) -> set[int]:
        return set(self._selected_channels)

    @property
    def channel_regions(self) -> dict[int, str]:
        return dict(self._channel_regions)

    def assign_region(self, name: str) -> None:
        label = name.strip()
        if not label:
            return
        for channel in self._selected_channels:
            self._channel_regions[channel] = label
        self.update()

    def clear_region(self) -> None:
        for channel in self._selected_channels:
            self._channel_regions.pop(channel, None)
        self.update()

    def region_names(self) -> list[str]:
        return sorted({label for label in self._channel_regions.values() if label.strip()})

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101216"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._dot_hits = {}

        channels = self._geometry_channels()
        if not channels:
            painter.setPen(QPen(QColor("#8a9099")))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No channel groups")
            return

        title_height = 28
        view = self._geometry_view(channels)
        radius = max(3.5, min(7.0, view["scale"] * 20.0))

        painter.setPen(QPen(QColor("#d6dde8")))
        painter.drawText(QRectF(0, 0, self.width(), title_height), Qt.AlignmentFlag.AlignCenter, "Brain Regions")

        painter.setPen(QPen(QColor("#252b34")))
        painter.drawRect(QRectF(view["x_origin"], view["y_origin"], view["draw_width"], view["draw_height"]))

        group_centers: dict[int, list[float]] = {}
        for group_id, _channel, x, _y in channels:
            group_centers.setdefault(group_id, []).append(self._to_canvas(x, view["y_min"], view).x())
        small_font = painter.font()
        small_font.setPointSizeF(7.0)
        painter.setFont(small_font)
        for group_id, positions in group_centers.items():
            x_center = sum(positions) / max(1, len(positions))
            painter.setPen(QPen(QColor("#303743")))
            painter.drawLine(
                int(x_center),
                int(view["y_origin"]),
                int(x_center),
                int(view["y_origin"] + view["draw_height"]),
            )
            painter.setPen(QPen(QColor("#9aa4b2")))
            painter.drawText(
                QRectF(x_center - 16, view["y_origin"] - 16, 32, 12),
                Qt.AlignmentFlag.AlignCenter,
                f"G{group_id}",
            )

        label_font = painter.font()
        label_font.setPointSizeF(6.5)
        painter.setFont(label_font)
        for _group_id, channel, x, y in channels:
            point = self._to_canvas(x, y, view)
            label = self._channel_regions.get(channel, "")
            selected = channel in self._selected_channels or channel in self._drag_preview_channels
            painter.setBrush(QColor(self._region_color(label)))
            pen = QPen(QColor("#facc15") if selected else QColor("#c3ccd8"))
            pen.setWidth(3 if selected else 1)
            painter.setPen(pen)
            painter.drawEllipse(point, radius, radius)
            self._dot_hits[channel] = (point.x(), point.y(), radius + 7)

            painter.setPen(QPen(QColor("#b8c7da")))
            painter.drawText(
                QRectF(point.x() + radius + 1, point.y() - 6, 28, 11),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                str(channel - 1),
            )
            if label:
                painter.setPen(QPen(QColor("#9bd1ff")))
                painter.drawText(
                    QRectF(point.x() + radius + 22, point.y() - 6, 56, 11),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )

        if self._drag_start is not None and self._drag_current is not None:
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            painter.setPen(QPen(QColor("#facc15"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(250, 204, 21, 40))
            painter.drawRect(rect)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        channels = self._geometry_channels()
        if not channels:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return

        old_view = self._geometry_view(channels)
        old_zoom = self._zoom
        zoom_factor = 1.15 ** (delta / 120.0)
        new_zoom = min(24.0, max(0.5, old_zoom * zoom_factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        cursor = event.position()
        scale_ratio = new_zoom / old_zoom
        new_origin_x = cursor.x() - (cursor.x() - old_view["x_origin"]) * scale_ratio
        new_origin_y = cursor.y() - (cursor.y() - old_view["y_origin"]) * scale_ratio
        centered_view = self._geometry_view(channels, zoom=new_zoom, pan=QPointF(0.0, 0.0))
        self._zoom = new_zoom
        self._pan = QPointF(
            new_origin_x - centered_view["x_origin"],
            new_origin_y - centered_view["y_origin"],
        )
        self.update()
        event.accept()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        channel = self._channel_at(event.position().x(), event.position().y())
        modifiers = event.modifiers()
        if channel is not None:
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                if channel in self._selected_channels:
                    self._selected_channels.remove(channel)
                else:
                    self._selected_channels.add(channel)
            else:
                self._selected_channels = {channel}
            self._drag_preview_channels.clear()
            self._drag_start = None
            self._drag_current = None
            self.update()
            return
        self._drag_start = event.position()
        self._drag_current = event.position()
        self._drag_preview_channels.clear()
        if not (modifiers & Qt.KeyboardModifier.ControlModifier):
            self._selected_channels.clear()
        self.update()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._drag_start is None:
            return
        self._drag_current = event.position()
        self._drag_preview_channels = self._channels_in_rect(QRectF(self._drag_start, self._drag_current).normalized())
        self.update()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_start is not None and self._drag_current is not None:
            rect = QRectF(self._drag_start, self._drag_current).normalized()
            if rect.width() > 3 or rect.height() > 3:
                self._selected_channels |= self._channels_in_rect(rect)
        self._drag_start = None
        self._drag_current = None
        self._drag_preview_channels.clear()
        self.update()

    def _channels_in_rect(self, rect: QRectF) -> set[int]:
        selected: set[int] = set()
        for channel, (cx, cy, radius) in self._dot_hits.items():
            hit_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
            if rect.intersects(hit_rect) or rect.contains(QPointF(cx, cy)):
                selected.add(channel)
        return selected

    def _channel_at(self, x: float, y: float) -> int | None:
        best_channel: int | None = None
        best_distance = float("inf")
        for channel, (cx, cy, radius) in self._dot_hits.items():
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if distance <= radius and distance < best_distance:
                best_channel = channel
                best_distance = distance
        return best_channel

    def _geometry_channels(self) -> list[tuple[int, int, float, float]]:
        return [
            (group.group_id, channel_info.channel, channel_info.x, channel_info.y)
            for group in self._groups
            for channel_info in group.channels
        ]

    def _geometry_view(
        self,
        channels: list[tuple[int, int, float, float]],
        *,
        zoom: float | None = None,
        pan: QPointF | None = None,
    ) -> dict[str, float]:
        zoom_value = self._zoom if zoom is None else zoom
        pan_value = self._pan if pan is None else pan
        margin = 22
        title_height = 28
        plot_rect = QRectF(
            margin,
            title_height + 8,
            max(1, self.width() - 2 * margin),
            max(1, self.height() - title_height - margin - 8),
        )
        xs = [x for _group_id, _channel, x, _y in channels]
        ys = [y for _group_id, _channel, _x, y in channels]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_range = max(1.0, x_max - x_min)
        y_range = max(1.0, y_max - y_min)
        x_pad = max(20.0, x_range * 0.04)
        y_pad = max(20.0, y_range * 0.04)
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
        x_range = max(1.0, x_max - x_min)
        y_range = max(1.0, y_max - y_min)
        scale = min(plot_rect.width() / x_range, plot_rect.height() / y_range) * zoom_value
        draw_width = x_range * scale
        draw_height = y_range * scale
        x_origin = plot_rect.left() + (plot_rect.width() - draw_width) / 2 + pan_value.x()
        y_origin = plot_rect.top() + (plot_rect.height() - draw_height) / 2 + pan_value.y()
        return {
            "x_min": x_min,
            "y_min": y_min,
            "scale": scale,
            "draw_width": draw_width,
            "draw_height": draw_height,
            "x_origin": x_origin,
            "y_origin": y_origin,
        }

    def _to_canvas(self, x: float, y: float, view: dict[str, float]) -> QPointF:
        px = view["x_origin"] + (x - view["x_min"]) * view["scale"]
        py = view["y_origin"] + view["draw_height"] - (y - view["y_min"]) * view["scale"]
        return QPointF(px, py)

    def _region_color(self, label: str) -> str:
        if not label:
            return "#808080"
        palette = [
            "#80deea",
            "#ffcc80",
            "#a5d6a7",
            "#f48fb1",
            "#90caf9",
            "#ce93d8",
            "#fff59d",
            "#bcaaa4",
            "#b0bec5",
            "#d1c4e9",
        ]
        labels = sorted({value for value in self._channel_regions.values() if value.strip()})
        return palette[labels.index(label) % len(palette)] if label in labels else "#808080"


class BrainRegionEditorDialog(QDialog):
    def __init__(
        self,
        groups: list[AnatomicalChannelGroup],
        channel_regions: dict[int, str],
        default_save_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit anatomical map")
        self.resize(1280, 740)
        self.setMinimumSize(1120, 680)
        self.setStyleSheet(
            """
            QDialog { background: #101216; color: #d6dde8; }
            QLabel { color: #d6dde8; }
            QListWidget {
                background: #151922;
                color: #d6dde8;
                border: 1px solid #303743;
                selection-background-color: #31425f;
            }
            QPushButton {
                background: #34373d;
                color: #f0f3f7;
                border: 1px solid #4b515b;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton:hover { background: #404651; }
            QPushButton:pressed { background: #2b3038; }
            """
        )
        self.groups = groups
        self.default_save_path = default_save_path
        self.viewer = BrainRegionProbeWidget(groups, channel_regions)
        self.region_list = QListWidget()
        self.status = QLabel(f"Default save: {default_save_path}")
        self.status.setWordWrap(True)

        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(self.viewer, 1)

        controls = QHBoxLayout()
        assign_button = QPushButton("Assign Region")
        assign_button.clicked.connect(self._assign_region)
        clear_button = QPushButton("Clear Region")
        clear_button.clicked.connect(self._clear_region)
        help_button = QPushButton("Help")
        help_button.clicked.connect(self._show_help)
        controls.addWidget(assign_button)
        controls.addWidget(clear_button)
        controls.addWidget(help_button)
        controls.addStretch(1)
        left.addLayout(controls)
        layout.addLayout(left, 1)

        side = QVBoxLayout()
        side.addWidget(QLabel("Assigned regions"))
        side.addWidget(self.region_list, 1)
        save_as = QPushButton("Save anatomical map as CSV")
        save_as.clicked.connect(self._save_as_csv)
        side.addWidget(save_as)
        side.addWidget(self.status)
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        side.addWidget(dialog_buttons)
        layout.addLayout(side)
        self._refresh_region_list()

    @property
    def channel_regions(self) -> dict[int, str]:
        return self.viewer.channel_regions

    def _assign_region(self) -> None:
        selected = self.viewer.selected_channels
        if not selected:
            return
        value, ok = QInputDialog.getText(
            self,
            "Assign Region",
            f"Region name for {len(selected)} selected channel(s)",
        )
        if not ok:
            return
        self.viewer.assign_region(value)
        self._refresh_region_list()

    def _clear_region(self) -> None:
        if not self.viewer.selected_channels:
            return
        self.viewer.clear_region()
        self._refresh_region_list()

    def _refresh_region_list(self) -> None:
        self.region_list.clear()
        self.region_list.addItems(self.viewer.region_names())

    def _save_as_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save anatomical_map.csv",
            str(self.default_save_path),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        saved = save_anatomical_map_csv(path, self.groups, self.viewer.channel_regions)
        self.status.setText(f"Saved anatomical map: {saved}")

    def _show_help(self) -> None:
        QMessageBox.information(
            self,
            "How to Use Brain Regions",
            "\n".join(
                [
                    "Click a channel to select one channel.",
                    "Ctrl+click to add or remove individual channels.",
                    "Drag an empty area to select a block of channels.",
                    "Assign Region writes the label to all selected channels.",
                    "Save anatomical map as CSV writes the pyNeuroscope-compatible anatomical_map.csv.",
                    "Displayed channel numbers are 0-based; saved CellExplorer channels are 1-based.",
                ]
            ),
        )


class CalibrationFrameCanvas(QWidget):
    def __init__(self, on_distance: Any) -> None:
        super().__init__()
        self._on_distance = on_distance
        self._drag_start: tuple[float, float] | None = None
        self._line: Any | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5.0, 3.2), facecolor="#2f2f2f")
        self.canvas = FigureCanvas(self.figure)
        self._ax = self.figure.add_subplot(111)
        self._ax.set_axis_off()
        self._ax.text(0.5, 0.5, "Load a video frame", ha="center", va="center", color="#d4d4d4")
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        layout.addWidget(self.canvas)

    def show_frame(self, frame: np.ndarray, *, title: str) -> None:
        self.figure.clear()
        self._ax = self.figure.add_subplot(111)
        self._ax.imshow(frame)
        self._ax.set_title(title, color="#f5f5f5")
        self._ax.set_axis_off()
        self._drag_start = None
        self._line = None
        self.canvas.draw_idle()

    def _on_button_press(self, event: Any) -> None:
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        self._drag_start = (float(event.xdata), float(event.ydata))
        if self._line is not None:
            self._line.remove()
        self._line = self._ax.plot([event.xdata, event.xdata], [event.ydata, event.ydata], color="#ef4b2d", linewidth=2)[0]
        self.canvas.draw_idle()

    def _on_motion(self, event: Any) -> None:
        if self._drag_start is None or self._line is None:
            return
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._drag_start
        self._line.set_data([x0, float(event.xdata)], [y0, float(event.ydata)])
        self.canvas.draw_idle()

    def _on_button_release(self, event: Any) -> None:
        if self._drag_start is None or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = float(event.xdata), float(event.ydata)
        distance = float(np.hypot(x1 - x0, y1 - y0))
        self._drag_start = None
        if distance > 0:
            self._on_distance(distance)
        self.canvas.draw_idle()


class BehaviorFrameCanvas(QWidget):
    def __init__(self, epoch_name: str, on_line_changed: Any) -> None:
        super().__init__()
        self.epoch_name = epoch_name
        self._on_line_changed = on_line_changed
        self.pixel_distance: float | None = None
        self._pending_start: tuple[float, float] | None = None
        self._line: Any | None = None
        self._start_marker: Any | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(7.0, 5.0), facecolor="#2f2f2f")
        self.canvas = FigureCanvas(self.figure)
        self._ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        layout.addWidget(self.canvas, 1)
        self.show_message("No frame loaded")

    def show_message(self, message: str) -> None:
        self.figure.clear()
        self._ax = self.figure.add_subplot(111)
        self._ax.set_facecolor("#252525")
        self._ax.text(0.5, 0.5, message, ha="center", va="center", color="#d4d4d4", wrap=True)
        self._ax.set_axis_off()
        self._pending_start = None
        self._line = None
        self._start_marker = None
        self.pixel_distance = None
        self.canvas.draw_idle()
        self._on_line_changed()

    def show_frame(self, frame: np.ndarray, *, title: str) -> None:
        self.figure.clear()
        self._ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        self._ax.imshow(frame)
        self._ax.set_axis_off()
        self._pending_start = None
        self._line = None
        self._start_marker = None
        self.pixel_distance = None
        self.canvas.draw_idle()
        self._on_line_changed()

    def reset_line(self) -> None:
        if self._line is not None:
            self._line.remove()
            self._line = None
        if self._start_marker is not None:
            self._start_marker.remove()
            self._start_marker = None
        self._pending_start = None
        self.pixel_distance = None
        self.canvas.draw_idle()
        self._on_line_changed()

    def _on_button_press(self, event: Any) -> None:
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        x1, y1 = float(event.xdata), float(event.ydata)
        if self._pending_start is None:
            self.reset_line()
            self._pending_start = (x1, y1)
            self._start_marker = self._ax.plot(
                [x1],
                [y1],
                marker="o",
                markersize=6,
                color="#ef4b2d",
                markeredgecolor="#ffffff",
                markeredgewidth=0.8,
            )[0]
            self.canvas.draw_idle()
            return

        x0, y0 = self._pending_start
        if self._line is not None:
            self._line.remove()
        self._line = self._ax.plot([x0, x1], [y0, y1], color="#ef4b2d", linewidth=2.2)[0]
        if self._start_marker is not None:
            self._start_marker.remove()
            self._start_marker = None
        self._pending_start = None
        distance = float(np.hypot(x1 - x0, y1 - y0))
        self.pixel_distance = distance if distance > 0 else None
        self.canvas.draw_idle()
        self._on_line_changed()


def _points_near_closed_polyline(points: np.ndarray, polygon: np.ndarray, margin: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if points.size == 0 or polygon.shape[0] < 2 or margin <= 0:
        return np.zeros(points.shape[0], dtype=bool)
    closed = np.vstack([polygon, polygon[0]])
    margin_sq = float(margin) ** 2
    near = np.zeros(points.shape[0], dtype=bool)
    for start, stop in zip(closed[:-1], closed[1:]):
        segment = stop - start
        denom = float(np.dot(segment, segment))
        if denom <= 0:
            closest = np.broadcast_to(start, points.shape)
        else:
            t = np.clip(((points - start) @ segment) / denom, 0.0, 1.0)
            closest = start + t[:, None] * segment
        distances_sq = np.sum((points - closest) ** 2, axis=1)
        near |= distances_sq <= margin_sq
    return near


def _points_in_closed_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)
    if points.size == 0 or polygon.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    inside = np.zeros(points.shape[0], dtype=bool)
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        yi = poly_y[i]
        yj = poly_y[j]
        crosses_y = finite & ((yi > y) != (yj > y))
        if np.any(crosses_y):
            xi = poly_x[i]
            xj = poly_x[j]
            x_intersection = (xj - xi) * (y[crosses_y] - yi) / (yj - yi) + xi
            selected = np.flatnonzero(crosses_y)
            inside[selected] ^= x[selected] < x_intersection
        j = i
    return inside


class BehaviorTrackCanvas(QWidget):
    def __init__(self, on_mask_changed: Any) -> None:
        super().__init__()
        self._on_mask_changed = on_mask_changed
        self.timestamps = np.empty((0,), dtype=np.float64)
        self.x = np.empty((0,), dtype=np.float64)
        self.y = np.empty((0,), dtype=np.float64)
        self.good_mask = np.empty((0,), dtype=bool)
        self._keep_polygons: list[np.ndarray] = []
        self._current_polygon: list[tuple[float, float]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(7.0, 5.0), facecolor="#2f2f2f")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.summary)
        self.show_empty()

    def show_empty(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#252525")
        self.ax.text(
            0.5,
            0.5,
            "Load outlier cleanup preview",
            ha="center",
            va="center",
            color="#d4d4d4",
        )
        self.ax.set_axis_off()
        self.summary.setText("No behavior track loaded")
        self.canvas.draw_idle()

    def show_message(self, message: str) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#252525")
        self.ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color="#d4d4d4",
            wrap=True,
        )
        self.ax.set_axis_off()
        self.summary.setText("")
        self.canvas.draw_idle()

    def set_track(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        self.timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        self.x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = min(self.timestamps.size, self.x.size, self.y.size)
        self.timestamps = self.timestamps[:n]
        self.x = self.x[:n]
        self.y = self.y[:n]
        self.good_mask = np.isfinite(self.x) & np.isfinite(self.y)
        self._keep_polygons = []
        self._current_polygon = []
        self._render()
        self._on_mask_changed(None)

    def set_processed_track(self, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        self.timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        self.x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = min(self.timestamps.size, self.x.size, self.y.size)
        self.timestamps = self.timestamps[:n]
        self.x = self.x[:n]
        self.y = self.y[:n]
        self.good_mask = np.isfinite(self.x) & np.isfinite(self.y)
        self._keep_polygons = []
        self._current_polygon = []
        self._render()

    def reset_keep_ranges(self) -> None:
        self._keep_polygons = []
        self._current_polygon = []
        self.good_mask = np.isfinite(self.x) & np.isfinite(self.y)
        self._render()
        self._on_mask_changed(None)

    def _render(self) -> None:
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.94)
        self.ax.set_facecolor("#252525")
        finite = np.isfinite(self.x) & np.isfinite(self.y)
        rejected = finite & ~self.good_mask
        accepted = finite & self.good_mask
        self.ax.scatter(self.x[accepted], self.y[accepted], s=5, c="#80deea", alpha=0.55, linewidth=0)
        if np.any(rejected):
            self.ax.scatter(self.x[rejected], self.y[rejected], s=8, c="#ef4b2d", alpha=0.70, linewidth=0)
        for polygon in self._keep_polygons:
            closed = np.vstack([polygon, polygon[0]])
            self.ax.fill(closed[:, 0], closed[:, 1], facecolor="#80deea18", edgecolor="#80deea", linewidth=1.2)
        if self._current_polygon:
            current = np.asarray(self._current_polygon, dtype=np.float64)
            self.ax.plot(current[:, 0], current[:, 1], color="#f5d76e", linewidth=1.4, marker="o", markersize=4)
            if current.shape[0] >= 3:
                closed = np.vstack([current, current[0]])
                self.ax.plot(closed[:, 0], closed[:, 1], color="#f5d76e", linewidth=0.9, linestyle="--")
        self.ax.set_xlabel("x (cm)", color="#d4d4d4")
        self.ax.set_ylabel("y (cm)", color="#d4d4d4")
        self.ax.tick_params(colors="#a3a3a3")
        for spine in self.ax.spines.values():
            spine.set_color("#404040")
        self._set_padded_limits()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.20, color="#737373")
        self.summary.setText(
            f"keep polygons={len(self._keep_polygons)}, current vertices={len(self._current_polygon)}, "
            f"accepted={int(np.sum(accepted))}, outlier/NaN={int(np.sum(~accepted))}"
        )
        self.canvas.draw_idle()

    def _set_padded_limits(self) -> None:
        arrays_x: list[np.ndarray] = []
        arrays_y: list[np.ndarray] = []
        finite = np.isfinite(self.x) & np.isfinite(self.y)
        if np.any(finite):
            arrays_x.append(self.x[finite])
            arrays_y.append(self.y[finite])
        for polygon in self._keep_polygons:
            arrays_x.append(polygon[:, 0])
            arrays_y.append(polygon[:, 1])
        if self._current_polygon:
            current = np.asarray(self._current_polygon, dtype=np.float64)
            arrays_x.append(current[:, 0])
            arrays_y.append(current[:, 1])
        if not arrays_x:
            return
        x_values = np.concatenate(arrays_x)
        y_values = np.concatenate(arrays_y)
        x_values = x_values[np.isfinite(x_values)]
        y_values = y_values[np.isfinite(y_values)]
        if x_values.size == 0 or y_values.size == 0:
            return
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        x_span = max(x_max - x_min, 1.0)
        y_span = max(y_max - y_min, 1.0)
        self.ax.set_xlim(x_min - x_span * 0.10, x_max + x_span * 0.10)
        self.ax.set_ylim(y_min - y_span * 0.10, y_max + y_span * 0.10)

    def _apply_keep_polygons(self) -> None:
        finite = np.isfinite(self.x) & np.isfinite(self.y)
        if not self._keep_polygons:
            self.good_mask = finite
            self._on_mask_changed(None)
            return
        keep = np.zeros(self.x.shape, dtype=bool)
        points = np.column_stack([self.x, self.y])
        for polygon in self._keep_polygons:
            inside = _points_in_closed_polygon(points, polygon)
            near_boundary = _points_near_closed_polyline(points, polygon, 1e-9)
            keep |= finite & (inside | near_boundary)
        self.good_mask = keep
        self._on_mask_changed(self.good_mask.copy())

    def _on_button_press(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        self._current_polygon.append((float(event.xdata), float(event.ydata)))
        if len(self._current_polygon) >= 3:
            polygon = np.asarray(self._current_polygon, dtype=np.float64)
            if np.linalg.matrix_rank(polygon - polygon[0]) >= 2:
                self._keep_polygons = [polygon]
                self._apply_keep_polygons()
        self._render()


class TrackerJumpDialog(QDialog):
    def __init__(self, parent: QWidget, timestamps: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__(parent)
        self.setWindowTitle("Clean Tracker Jumps")
        self.resize(900, 700)
        self.timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        self.x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = min(self.timestamps.size, self.x.size, self.y.size)
        self.timestamps = self.timestamps[:n]
        self.x = self.x[:n]
        self.y = self.y[:n]
        self.good_mask = np.isfinite(self.x) & np.isfinite(self.y)
        self._drag_start: tuple[float, float] | None = None
        self._selection_patch: Rectangle | None = None

        layout = QVBoxLayout(self)
        hint = QLabel("Drag a rectangle around outlier points to reject them. Rejected points are saved as NaN.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.figure = Figure(figsize=(7.0, 5.0), facecolor="#2f2f2f")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_button_release)
        layout.addWidget(self.canvas, 1)
        self.summary = QLabel("")
        layout.addWidget(self.summary)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        reset = QPushButton("Reset mask")
        reset.clicked.connect(self._reset_mask)
        buttons_layout.addWidget(reset)
        buttons_layout.addStretch(1)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        buttons_layout.addWidget(button_box)
        layout.addWidget(buttons_row)
        self._render()

    def _render(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#252525")
        finite = np.isfinite(self.x) & np.isfinite(self.y)
        rejected = finite & ~self.good_mask
        accepted = finite & self.good_mask
        self.ax.scatter(self.x[accepted], self.y[accepted], s=6, c="#80deea", alpha=0.65, linewidth=0)
        if np.any(rejected):
            self.ax.scatter(self.x[rejected], self.y[rejected], s=12, c="#ef4b2d", alpha=0.9, linewidth=0)
        self.ax.set_xlabel("x (cm)", color="#d4d4d4")
        self.ax.set_ylabel("y (cm)", color="#d4d4d4")
        self.ax.tick_params(colors="#a3a3a3")
        for spine in self.ax.spines.values():
            spine.set_color("#404040")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.summary.setText(f"accepted={int(np.sum(accepted))}, rejected={int(np.sum(rejected))}")
        self.canvas.draw_idle()

    def _on_button_press(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        self._drag_start = (float(event.xdata), float(event.ydata))
        if self._selection_patch is not None:
            self._selection_patch.remove()
        self._selection_patch = Rectangle(
            self._drag_start,
            0,
            0,
            facecolor="#ef4b2d22",
            edgecolor="#ef4b2d",
            linewidth=1.2,
        )
        self.ax.add_patch(self._selection_patch)
        self.canvas.draw_idle()

    def _on_motion(self, event: Any) -> None:
        if self._drag_start is None or self._selection_patch is None:
            return
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = float(event.xdata), float(event.ydata)
        self._selection_patch.set_x(min(x0, x1))
        self._selection_patch.set_y(min(y0, y1))
        self._selection_patch.set_width(abs(x1 - x0))
        self._selection_patch.set_height(abs(y1 - y0))
        self.canvas.draw_idle()

    def _on_button_release(self, event: Any) -> None:
        if self._drag_start is None or event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = float(event.xdata), float(event.ydata)
        xmin, xmax = sorted((x0, x1))
        ymin, ymax = sorted((y0, y1))
        selected = (
            np.isfinite(self.x)
            & np.isfinite(self.y)
            & (self.x >= xmin)
            & (self.x <= xmax)
            & (self.y >= ymin)
            & (self.y <= ymax)
        )
        self.good_mask[selected] = False
        if self._selection_patch is not None:
            self._selection_patch.remove()
            self._selection_patch = None
        self._drag_start = None
        self._render()

    def _reset_mask(self) -> None:
        self.good_mask = np.isfinite(self.x) & np.isfinite(self.y)
        self._render()


class SubsessionOrderTable(QTableWidget):
    def __init__(
        self,
        basepath: Path,
        paths: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._basepath = Path(basepath).expanduser().resolve()
        self._paths = [Path(path).expanduser().resolve() for path in paths]
        self._populating = False
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Order", "Subsession", "Discovered path"])
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self.setDropIndicatorShown(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(0, 70)
        self.setColumnWidth(1, 330)
        self.itemChanged.connect(self._handle_item_changed)
        self.refresh()

    def paths(self) -> list[Path]:
        return list(self._paths)

    def relative_paths(self) -> list[str]:
        values: list[str] = []
        for path in self._paths:
            try:
                values.append(path.relative_to(self._basepath).as_posix())
            except ValueError:
                values.append(str(path))
        return values

    def _display_name(self, path: Path) -> str:
        try:
            relative = path.relative_to(self._basepath)
        except ValueError:
            return path.parent.name if path.is_file() else path.name
        if not relative.parts:
            return path.name
        return relative.parts[0]

    def refresh(self, select_row: int | None = None) -> None:
        self._populating = True
        try:
            self.setRowCount(len(self._paths))
            for row, path in enumerate(self._paths):
                order_item = QTableWidgetItem(str(row + 1))
                name_item = QTableWidgetItem(self._display_name(path))
                try:
                    path_text = path.relative_to(self._basepath).as_posix()
                except ValueError:
                    path_text = str(path)
                path_item = QTableWidgetItem(path_text)
                order_item.setFlags(order_item.flags() | Qt.ItemFlag.ItemIsEditable)
                for item in (name_item, path_item):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, 0, order_item)
                self.setItem(row, 1, name_item)
                self.setItem(row, 2, path_item)
            if select_row is not None and 0 <= select_row < self.rowCount():
                self.selectRow(select_row)
        finally:
            self._populating = False

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != 0:
            return
        source_index = item.row()
        try:
            target_index = int(item.text().strip()) - 1
        except ValueError:
            self.refresh(source_index)
            return
        target_index = max(0, min(target_index, len(self._paths) - 1))
        if target_index == source_index:
            self.refresh(source_index)
            return
        path = self._paths.pop(source_index)
        self._paths.insert(target_index, path)
        self.refresh(target_index)


class MultiDaySessionOrderTable(QTableWidget):
    def __init__(
        self,
        paths: list[str],
        selected_subepoch_paths: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._paths = list(paths)
        self._selected_subepoch_paths = {
            self._path_key(path) for path in (selected_subepoch_paths or []) if str(path).strip()
        }
        self._has_explicit_subepoch_selection = bool(self._selected_subepoch_paths)
        self._populating = False
        self._rows: list[dict[str, str]] = []
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Use", "Order", "Session", "Subepoch"])
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self.setDropIndicatorShown(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(0, 58)
        self.setColumnWidth(1, 70)
        self.setColumnWidth(2, 240)
        self.itemChanged.connect(self._handle_item_changed)
        self.refresh()

    @staticmethod
    def _path_key(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    def paths(self) -> list[str]:
        return list(self._paths)

    def discovered_subepoch_count(self) -> int:
        return len(self._all_subepoch_keys_in_order())

    def selected_subepoch_paths(self) -> list[str]:
        keys = self._all_subepoch_keys_in_order()
        if not self._has_explicit_subepoch_selection:
            return keys
        return [key for key in keys if key in self._selected_subepoch_paths]

    def _all_subepoch_keys_in_order(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for row in self._rows:
            key = row.get("subepoch_key", "")
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def _discover_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for order, path_text in enumerate(self._paths, start=1):
            session_path = Path(path_text).expanduser()
            base_row = {
                "order": str(order),
                "session": session_path.name,
                "session_path": path_text,
            }
            try:
                subepochs = discover_multi_day_subepochs([session_path], require_subepochs=False)
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "subepoch": f"Discovery error: {exc}",
                        "path": "",
                        "subepoch_key": "",
                    }
                )
                continue
            if not subepochs:
                rows.append(
                    {
                        **base_row,
                        "subepoch": "No subepochs found",
                        "path": "",
                        "subepoch_key": "",
                    }
                )
                continue
            for subepoch in subepochs:
                source_path = self._path_key(subepoch.source_subepoch_path)
                rows.append(
                    {
                        **base_row,
                        "subepoch": f"{subepoch.subepoch_index}: {Path(source_path).name}",
                        "path": source_path,
                        "subepoch_key": source_path,
                    }
                )
        return rows

    def refresh(self, select_row: int | None = None) -> None:
        self._populating = True
        try:
            self._rows = self._discover_rows()
            self.setRowCount(len(self._rows))
            for row, entry in enumerate(self._rows):
                subepoch_key = entry.get("subepoch_key", "")
                use_item = QTableWidgetItem("")
                if subepoch_key:
                    use_item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    checked = (
                        not self._has_explicit_subepoch_selection
                        or subepoch_key in self._selected_subepoch_paths
                    )
                    use_item.setCheckState(
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                    )
                else:
                    use_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                order_item = QTableWidgetItem(str(row + 1))
                order_item.setText(entry["order"])
                session_item = QTableWidgetItem(entry["session"])
                subepoch_item = QTableWidgetItem(entry["subepoch"])
                if entry.get("path"):
                    subepoch_item.setToolTip(entry["path"])
                self.setItem(row, 0, use_item)
                self.setItem(row, 1, order_item)
                self.setItem(row, 2, session_item)
                self.setItem(row, 3, subepoch_item)
                order_item.setFlags(order_item.flags() | Qt.ItemFlag.ItemIsEditable)
                for item in (session_item, subepoch_item):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if select_row is not None and 0 <= select_row < self.rowCount():
                self.selectRow(select_row)
        finally:
            self._populating = False

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating:
            return
        if item.column() == 0:
            self._handle_subepoch_check_changed(item)
            return
        if item.column() != 1:
            return
        row = item.row()
        if row < 0 or row >= len(self._rows):
            return
        session_path = self._rows[row]["session_path"]
        try:
            target = int(item.text().strip()) - 1
        except ValueError:
            self.refresh(row)
            return
        if not self._paths:
            return
        try:
            source_index = self._paths.index(session_path)
        except ValueError:
            self.refresh(row)
            return
        target = max(0, min(target, len(self._paths) - 1))
        if target == source_index:
            self.refresh(row)
            return
        path = self._paths.pop(source_index)
        self._paths.insert(target, path)
        self.refresh()

    def _handle_subepoch_check_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if row < 0 or row >= len(self._rows):
            return
        key = self._rows[row].get("subepoch_key", "")
        if not key:
            return
        if not self._has_explicit_subepoch_selection:
            self._selected_subepoch_paths = set(self._all_subepoch_keys_in_order())
            self._has_explicit_subepoch_selection = True
        if item.checkState() == Qt.CheckState.Checked:
            self._selected_subepoch_paths.add(key)
        else:
            self._selected_subepoch_paths.discard(key)


class CellExploreFolderTable(QTableWidget):
    def __init__(
        self,
        rows: list[dict[str, str]],
        selected_paths: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._rows = list(rows)
        self._selected = {str(Path(path).expanduser().resolve()) for path in selected_paths}
        self._populating = False
        self._prune_selected_conflicts()
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Use", "Folder", "Status", "Path"])
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self.setDropIndicatorShown(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(0, 58)
        self.setColumnWidth(1, 270)
        self.setColumnWidth(2, 170)
        self.itemChanged.connect(self._handle_item_changed)
        self.refresh()

    def refresh(self) -> None:
        self._populating = True
        try:
            self.setRowCount(len(self._rows))
            for row, entry in enumerate(self._rows):
                path_text = entry["path"]
                key = str(Path(path_text).expanduser().resolve())
                use_item = QTableWidgetItem("")
                use_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                use_item.setCheckState(
                    Qt.CheckState.Checked if key in self._selected else Qt.CheckState.Unchecked
                )
                folder_item = QTableWidgetItem(Path(path_text).name)
                status_item = QTableWidgetItem(entry.get("status", ""))
                path_item = QTableWidgetItem(path_text)
                for item in (folder_item, status_item, path_item):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.setItem(row, 0, use_item)
                self.setItem(row, 1, folder_item)
                self.setItem(row, 2, status_item)
                self.setItem(row, 3, path_item)
        finally:
            self._populating = False

    def add_folder(self, path: Path, *, checked: bool = True, source: str = "manual") -> None:
        resolved = str(path.expanduser().resolve())
        if resolved not in {str(Path(row["path"]).expanduser().resolve()) for row in self._rows}:
            status = "ok" if (path / "params.py").exists() else "missing params.py"
            self._rows.append(
                {
                    "path": resolved,
                    "source": source,
                    "status": f"{source}; {status}",
                    "group": self._default_group_key(Path(resolved)),
                }
            )
        if checked:
            self._select_exclusive(resolved, self._default_group_key(Path(resolved)))
        self.refresh()

    @staticmethod
    def _default_group_key(path: Path) -> str:
        resolved = path.expanduser().resolve()
        name = resolved.name
        if name.endswith("_spi"):
            return str(resolved.with_name(name[: -len("_spi")]))
        return str(resolved)

    def _select_exclusive(self, key: str, group: str) -> None:
        for entry in self._rows:
            other_group = entry.get("group") or self._default_group_key(Path(entry["path"]))
            if other_group == group:
                self._selected.discard(str(Path(entry["path"]).expanduser().resolve()))
        self._selected.add(key)

    def _prune_selected_conflicts(self) -> None:
        selected_by_group: set[str] = set()
        pruned: set[str] = set()
        for entry in self._rows:
            key = str(Path(entry["path"]).expanduser().resolve())
            if key not in self._selected:
                continue
            group = entry.get("group") or self._default_group_key(Path(entry["path"]))
            if group in selected_by_group:
                continue
            selected_by_group.add(group)
            pruned.add(key)
        self._selected = pruned

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self._rows):
            return
        path_item = self.item(row, 3)
        if path_item is None:
            return
        key = str(Path(path_item.text()).expanduser().resolve())
        if item.checkState() == Qt.CheckState.Checked:
            group = self._rows[row].get("group") or self._default_group_key(Path(key))
            self._select_exclusive(key, group)
            for other_row, entry in enumerate(self._rows):
                if other_row == row:
                    continue
                other_group = entry.get("group") or self._default_group_key(Path(entry["path"]))
                if other_group != group:
                    continue
                other_key = str(Path(entry["path"]).expanduser().resolve())
                self._selected.discard(other_key)
                other_item = self.item(other_row, 0)
                if other_item is not None:
                    self._populating = True
                    try:
                        other_item.setCheckState(Qt.CheckState.Unchecked)
                    finally:
                        self._populating = False
        else:
            self._selected.discard(key)

    def selected_paths(self) -> list[str]:
        selected: list[str] = []
        for row in range(self.rowCount()):
            use_item = self.item(row, 0)
            path_item = self.item(row, 3)
            if use_item is None or path_item is None:
                continue
            if use_item.checkState() == Qt.CheckState.Checked:
                selected.append(path_item.text())
        return selected


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PreprocessPipeline GUI")
        self.resize(1500, 900)
        self._process: QProcess | None = None
        self._process_config_path: Path | None = None
        self._process_result: dict[str, Any] | None = None
        self._process_error: dict[str, Any] | None = None
        self._process_tail = ""
        self._process_stop_escalated = False
        self._legacy_process_session_claim: tuple[Path, str] | None = None
        # The QProcess can report that its direct child exited while a legacy
        # noise-label worker it spawned is still alive.  Keep the process-group
        # leader separately because QProcess.processId() becomes zero on exit.
        self._legacy_process_group_pid: int | None = None
        self._move_storage_override: Path | None = None
        self._move_source_snapshot: tuple[Path, str] | None = None
        self._active_run_dir: Path | None = None
        self._active_run_session_dir: Path | None = None
        self._slurm_capabilities: SlurmCapabilities | None = None
        self._persistent_log_offsets: dict[Path, int] = {}
        self._persistent_log_announced: set[Path] = set()
        self._last_persistent_status_signature = ""
        self._persistent_monitor_ticks = 0
        self._phy_process: QProcess | None = None
        self._phy_session_claim: tuple[Path, str] | None = None
        self._phy_working_dir: Path | None = None
        self._phy_last_counts: dict[str, int] | None = None
        self._cell_explorer_process: QProcess | None = None
        self._cell_explorer_session_claims: list[tuple[Path, str]] = []
        self._cell_explorer_working_dir: Path | None = None
        self._phy_status_timer = QTimer(self)
        self._phy_status_timer.setInterval(30000)
        self._phy_status_timer.timeout.connect(self._append_phy_curation_status)
        self._log_buffer = ""
        self._settings_preview_text = ""
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.setInterval(80)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._force_stop_requested = False
        self._probe_rows: list[dict[str, QWidget]] = []
        self.noise_threshold_fields: dict[str, QLineEdit] = {}
        self._refresh_suspended = False
        self._last_chanmap_preview_key: tuple[Any, ...] | None = None
        self._last_xml_warning_key: tuple[Any, ...] | None = None
        self._chanmap_controls_dirty = False
        self._subsession_order: list[str] = []
        self._multi_day_session_paths: list[str] = []
        self._multi_day_selected_subepoch_paths: list[str] = []
        self._cell_explorer_sorting_folders: list[str] = []
        self._channel_regions: dict[int, str] = {}
        self._behavior_dlc_files: list[Any] = []
        self._behavior_frame_canvases: dict[str, BehaviorFrameCanvas] = {}
        self._behavior_pixel_distances_by_folder: dict[str, float] = {}
        self._behavior_pixel_to_cm_ratios_by_folder: dict[str, float] = {}
        self._behavior_clean_mask: np.ndarray | None = None
        self._behavior_outlier_canvases: dict[str, tuple[BehaviorTrackCanvas, np.ndarray]] = {}
        self._reported_behavior_warnings: set[str] = set()
        self._behavior_outlier_processed_preview = False
        self._environment_backend_default_pending = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh_preview)
        self._run_monitor_timer = QTimer(self)
        self._run_monitor_timer.setInterval(2000)
        self._run_monitor_timer.timeout.connect(self._refresh_persistent_run_monitor)
        self._apply_dark_theme()
        self._build_ui()
        self._set_running(False)
        self._apply_config_settings(_load_default_settings(), DEFAULT_CONFIG_PATH)
        self._refresh_preview()
        self._run_monitor_timer.start()
        QTimer.singleShot(0, self._refresh_slurm_capabilities)

    def _apply_dark_theme(self) -> None:
        check_icon = (Path(__file__).resolve().parent / "assets" / "check-orange.svg").as_posix()
        stylesheet = """
            QMainWindow {
                background: #252525;
            }
            QWidget {
                color: #e5e5e5;
                font-size: 11px;
            }
            QWidget#rootWidget {
                background: #252525;
            }
            QWidget#settingsPage {
                background: #252525;
            }
            QGroupBox {
                background: #2f2f2f;
                border: 0;
                border-radius: 5px;
                margin-top: 16px;
                padding: 10px;
                font-weight: 650;
                color: #f5f5f5;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #f5f5f5;
            }
            QLabel {
                color: #d4d4d4;
                background: transparent;
            }
            QLabel#hintLabel {
                color: #a8a8a8;
                font-size: 10px;
                font-weight: 400;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #383838;
                color: #f0f0f0;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 4px 6px;
                selection-background-color: #606060;
            }
            QPlainTextEdit {
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                font-size: 11px;
            }
            QComboBox::drop-down {
                border: 0;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background: #303030;
                color: #e5e5e5;
                border: 1px solid #555555;
                selection-background-color: #555555;
                selection-color: #ffffff;
                outline: 0;
            }
            QComboBox QAbstractItemView::item {
                min-height: 22px;
                padding: 4px 8px;
            }
            QPushButton {
                background: #3a3a3a;
                color: #f0f0f0;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 6px 9px;
            }
            QPushButton:hover {
                background: #464646;
                border-color: #707070;
            }
            QPushButton:disabled {
                color: #737373;
                border-color: #2a2a2a;
                background: #101010;
            }
            QPushButton#primaryButton {
                background: #3f3f3f;
                border-color: #707070;
                color: #ffffff;
            }
            QPushButton#dangerButton {
                color: #ffb088;
                border-color: #8a3a16;
            }
            QCheckBox {
                color: #d4d4d4;
                spacing: 8px;
                background: transparent;
                min-height: 22px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #7a7a7a;
                background: #2f2f2f;
            }
            QCheckBox::indicator:hover {
                border-color: #9a9a9a;
                background: #363636;
            }
            QCheckBox::indicator:checked {
                border: 0;
                background: #ef4b2d;
                image: url(__CHECK_ICON__);
            }
            QCheckBox::indicator:checked:hover {
                border: 0;
                background: #f05a3d;
            }
            QCheckBox::indicator:checked:disabled,
            QCheckBox::indicator:unchecked:disabled {
                border-color: #666666;
                background: #303030;
            }
            QTabWidget::pane {
                border: 0;
                background: #252525;
            }
            QTabBar::tab {
                background: #1f1f1f;
                color: #a3a3a3;
                border: 1px solid #444444;
                padding: 7px 10px;
            }
            QTabBar::tab:selected {
                background: #111111;
                color: #f5f5f5;
                border-top: 2px solid #ef4b2d;
            }
            QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget, QSplitter {
                background: #252525;
                border: 0;
            }
            QTableView, QTableWidget, QTreeView {
                background: #2f2f2f;
                color: #e5e5e5;
                alternate-background-color: #292929;
                selection-background-color: #555555;
                selection-color: #ffffff;
                gridline-color: #505050;
                border: 1px solid #505050;
            }
            QHeaderView::section {
                background: #3a3a3a;
                color: #f0f0f0;
                border: 0;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                padding: 4px 6px;
            }
            QTableCornerButton::section {
                background: #3a3a3a;
                border: 1px solid #555555;
            }
            QFrame#miniPanel {
                background: #2f2f2f;
                border: 0;
                border-radius: 6px;
            }
            QLabel#miniHead {
                background: #1f1f1f;
                color: #f5f5f5;
                font-weight: 700;
                padding: 7px 10px;
                border-bottom: 1px solid #444444;
            }
            QFileDialog {
                background: #2f2f2f;
                color: #e5e5e5;
            }
            QFileDialog QWidget {
                background: #2f2f2f;
                color: #e5e5e5;
            }
            QFileDialog QLabel {
                color: #e5e5e5;
                background: transparent;
            }
            QFileDialog QTreeView,
            QFileDialog QListView,
            QFileDialog QTableView,
            QFileDialog QAbstractItemView {
                background: #303030;
                color: #e5e5e5;
                alternate-background-color: #2a2a2a;
                selection-background-color: #555555;
                selection-color: #ffffff;
                border: 1px solid #555555;
            }
            QFileDialog QHeaderView::section {
                background: #3a3a3a;
                color: #e5e5e5;
                border: 1px solid #555555;
                padding: 3px 6px;
            }
            QFileDialog QToolButton {
                background: #3a3a3a;
                color: #e5e5e5;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 3px;
            }
            QFileDialog QLineEdit,
            QFileDialog QComboBox {
                background: #3a3a3a;
                color: #f0f0f0;
                border: 1px solid #5a5a5a;
            }
            QMessageBox {
                background: #2f2f2f;
                color: #e8e8e8;
            }
            QMessageBox QLabel {
                color: #e8e8e8;
                background: transparent;
                font-size: 11px;
            }
            QMessageBox QPushButton {
                background: #3a3a3a;
                color: #f5f5f5;
                border: 1px solid #5f5f5f;
                border-radius: 5px;
                min-width: 72px;
                padding: 6px 10px;
            }
            QMessageBox QPushButton:hover {
                background: #464646;
                border-color: #777777;
            }
            """
        self.setStyleSheet(stylesheet.replace("__CHECK_ICON__", check_icon))

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("rootWidget")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_top_bar())

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self._build_settings_tabs())
        self.center_scroll_area = HorizontalOnlyScrollArea()
        self.center_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.center_scroll_area.setWidget(self._build_center_panel())
        self.center_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.main_splitter.addWidget(self.center_scroll_area)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([360, 1140])
        layout.addWidget(self.main_splitter, 1)

        layout.addWidget(self._build_run_bar())

        self.setCentralWidget(root)

    def _fit_to_available_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        margin = 16
        target_width = min(self.width(), max(1, available.width() - margin))
        target_height = min(self.height(), max(1, available.height() - margin))
        self.resize(target_width, target_height)
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.basepath = QLineEdit()
        self.basepath.setPlaceholderText("Select session/basepath")
        browse_basepath = QPushButton("Browse basepath")
        browse_basepath.setMaximumWidth(190)
        browse_basepath.clicked.connect(self._browse_basepath)
        check_basepath_order = QPushButton("Check order")
        check_basepath_order.setMinimumWidth(100)
        check_basepath_order.setMaximumWidth(145)
        check_basepath_order.clicked.connect(self._view_basepath_order)
        browse_multiday = QPushButton("Browse for multi-days")
        browse_multiday.setMaximumWidth(230)
        browse_multiday.clicked.connect(self._browse_multi_days)
        self.multi_day_sessions = QLineEdit()
        self.multi_day_sessions.setReadOnly(True)
        self.multi_day_sessions.setPlaceholderText("No multi-day sessions selected")
        view_multiday = QPushButton("Check order")
        view_multiday.setMinimumWidth(110)
        view_multiday.setMaximumWidth(145)
        view_multiday.clicked.connect(self._view_multi_day_sessions)
        self.multi_day_name = QLineEdit()
        self.multi_day_name.setPlaceholderText("multi-day name")

        self.local_root = QLineEdit()
        self.local_root.setPlaceholderText("Local working directory")
        browse_local = QPushButton("Browse local")
        browse_local.setMaximumWidth(155)
        browse_local.clicked.connect(self._browse_local_root)

        load_config = QPushButton("Load config")
        load_config.setMaximumWidth(145)
        load_config.clicked.connect(self._load_config)
        save_config = QPushButton("Save config")
        save_config.setMaximumWidth(145)
        save_config.clicked.connect(self._save_config)

        self.browse_local_session_resume = QPushButton("Browse local session to resume")
        self.browse_local_session_resume.clicked.connect(
            self._browse_local_session_to_resume
        )
        resume_hint = QLabel(
            "Select an existing local single- or multi-day output folder."
        )
        resume_hint.setObjectName("hintLabel")
        resume_hint.setWordWrap(True)

        local_row = QWidget()
        local_layout = QHBoxLayout(local_row)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(6)
        local_layout.addWidget(browse_local)
        local_layout.addWidget(self.browse_local_session_resume)
        local_layout.addWidget(QLabel("Local working dir"))
        local_layout.addWidget(self.local_root, 1)

        config_row = QWidget()
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(6)
        config_layout.addWidget(resume_hint, 1)
        config_layout.addWidget(load_config)
        config_layout.addWidget(save_config)

        layout.addWidget(browse_basepath, 0, 0)
        layout.addWidget(check_basepath_order, 0, 1)
        layout.addWidget(self.basepath, 0, 2, 1, 5)
        layout.addWidget(browse_multiday, 1, 0)
        layout.addWidget(view_multiday, 1, 1)
        layout.addWidget(self.multi_day_sessions, 1, 2, 1, 2)
        layout.addWidget(QLabel("Multi-day name"), 1, 4)
        layout.addWidget(self.multi_day_name, 1, 5, 1, 2)
        layout.addWidget(local_row, 2, 0, 1, 7)
        layout.addWidget(config_row, 3, 0, 1, 7)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 3)
        layout.setColumnStretch(5, 1)
        layout.setColumnStretch(6, 1)

        self.basepath.textChanged.connect(self._schedule_refresh)
        self.local_root.textChanged.connect(self._schedule_refresh)
        self.multi_day_name.textChanged.connect(self._schedule_refresh)
        self.basepath.textChanged.connect(self._reset_behavior_discovery_state)
        self.local_root.textChanged.connect(self._reset_behavior_discovery_state)
        self.basepath.textEdited.connect(self._clear_move_storage_override)
        self.basepath.textEdited.connect(lambda _text: self._set_subsession_order([]))
        self.multi_day_name.textEdited.connect(self._clear_move_storage_override)
        return panel

    def _build_settings_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_ephys_tab(), "Ephys")
        self.tabs.addTab(self._scroll_area(self._build_behavior_tab()), "Behavior")
        self.tabs.setMinimumWidth(300)
        self.tabs.currentChanged.connect(lambda _index: self._schedule_refresh())
        return self.tabs

    def _build_execution_tab(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settingsPage")
        panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        run_box = QGroupBox("Run scope")
        run_layout = QGridLayout(run_box)
        self.run_all = QPushButton("Run all")
        self.run_pre = QPushButton("Preprocess only")
        self.run_post = QPushButton("Postprocess only")
        self.run_all.clicked.connect(lambda: self._start_run("all"))
        self.run_pre.clicked.connect(lambda: self._start_run("preprocess"))
        self.run_post.clicked.connect(lambda: self._start_run("postprocess"))
        run_layout.addWidget(self.run_all, 0, 0)
        run_layout.addWidget(self.run_pre, 0, 1)
        run_layout.addWidget(self.run_post, 0, 2)
        for column in range(3):
            run_layout.setColumnStretch(column, 1)

        backend_box = QGroupBox("Execution backend")
        backend_form = self._form_layout(backend_box)
        self.execution_backend = NoWheelComboBox()
        self.execution_backend.addItem("Auto", RequestedBackend.AUTO.value)
        self.execution_backend.addItem("Local", RequestedBackend.LOCAL.value)
        self.execution_backend.addItem("Slurm", RequestedBackend.SLURM.value)
        self.execution_backend.setToolTip(
            "Slurm submits jobs to the server scheduler. Auto uses Slurm when "
            "available and otherwise runs locally."
        )
        self.execution_resolved = QLabel("Not resolved")
        self.execution_resolved.setWordWrap(True)
        self.execution_resolved.setMinimumHeight(38)
        backend_form.addRow("Run on", self.execution_backend)
        backend_form.addRow("Availability", self.execution_resolved)

        resources_title = QLabel("Stage resources")
        resources_title_font = resources_title.font()
        resources_title_font.setBold(True)
        resources_title.setFont(resources_title_font)
        resources_hint = self._hint_label(
            "CPU cores and Memory are Slurm reservations. Sorting requests one "
            "GPU; Slurm chooses its device ID when the job starts."
        )
        self._resource_widgets: dict[str, dict[str, QWidget]] = {}
        resource_groups = [
            self._build_stage_resource_group(StageName.PREPROCESS, gpu_count=0),
            self._build_stage_resource_group(StageName.SORTING, gpu_count=1),
            self._build_stage_resource_group(StageName.POSTPROCESS, gpu_count=0),
        ]

        monitor_box = QGroupBox("Run progress")
        monitor_layout = QVBoxLayout(monitor_box)
        self.execution_run_status = QLabel("No active Run")
        self.execution_run_status.setWordWrap(True)
        stage_grid = QGridLayout()
        stage_grid.setContentsMargins(0, 0, 0, 0)
        stage_grid.setHorizontalSpacing(12)
        stage_grid.addWidget(QLabel("Stage"), 0, 0)
        stage_grid.addWidget(QLabel("Status"), 0, 1)
        stage_grid.addWidget(QLabel("Job"), 0, 2)
        self.execution_stage_status_labels: dict[str, QLabel] = {}
        self.execution_stage_job_labels: dict[str, QLabel] = {}
        for row, stage in enumerate(StageName, start=1):
            stage_name = QLabel(stage.value.capitalize())
            stage_name_font = stage_name.font()
            stage_name_font.setBold(True)
            stage_name.setFont(stage_name_font)
            status = QLabel("Not started")
            job = QLabel("—")
            job.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.execution_stage_status_labels[stage.value] = status
            self.execution_stage_job_labels[stage.value] = job
            stage_grid.addWidget(stage_name, row, 0)
            stage_grid.addWidget(status, row, 1)
            stage_grid.addWidget(job, row, 2)
        stage_grid.setColumnStretch(1, 1)
        stage_grid.setColumnStretch(2, 1)
        progress_hint = self._hint_label(
            "Detailed progress and warnings are shown in the main Log. "
            "Slurm status is refreshed automatically."
        )
        monitor_layout.addWidget(self.execution_run_status)
        monitor_layout.addLayout(stage_grid)
        monitor_layout.addWidget(progress_hint)

        layout.addWidget(run_box)
        layout.addWidget(backend_box)
        layout.addWidget(resources_title)
        layout.addWidget(resources_hint)
        for resource_group in resource_groups:
            layout.addWidget(resource_group)
        layout.addWidget(monitor_box)
        layout.addStretch(1)
        return panel

    def _build_stage_resource_group(self, stage: StageName, *, gpu_count: int) -> QGroupBox:
        box = QGroupBox(stage.value.capitalize())
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 8, 6, 6)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(6)
        cpus = self._spin(1, 4096, default_worker_count())
        memory_gib = self._spin(1, 65536, 512 if stage == StageName.SORTING else 256)
        cpus.setToolTip(
            "Exact CPU cores requested from Slurm for this Stage. "
            "This also limits the Stage worker count."
        )
        memory_gib.setToolTip(
            "Host memory requested from Slurm in GiB. This is not GPU memory."
        )
        gpu = QLabel("1 (auto)" if gpu_count else "0")
        gpu.setToolTip(
            "Sorting requests one GPU. Slurm selects the device ID from the GPUs "
            "available when the job starts."
            if gpu_count
            else "This Stage does not request a GPU."
        )
        stage_label = stage.value.capitalize()
        cpu_label = QLabel("CPU cores")
        cpu_label.setBuddy(cpus)
        cpus.setAccessibleName(f"{stage_label} CPU cores")
        memory_label = QLabel("Memory (GiB)")
        memory_label.setBuddy(memory_gib)
        memory_gib.setAccessibleName(f"{stage_label} host memory in GiB")
        gpu.setAccessibleName(f"{stage_label} GPU request")
        widgets: dict[str, QWidget] = {
            "cpus": cpus,
            "memory_gib": memory_gib,
            "gpu_count": gpu,
        }
        self._resource_widgets[stage.value] = widgets
        grid.addWidget(cpu_label, 0, 0)
        grid.addWidget(cpus, 0, 1)
        grid.addWidget(memory_label, 0, 2)
        grid.addWidget(memory_gib, 0, 3)
        grid.addWidget(QLabel("GPUs"), 0, 4)
        grid.addWidget(gpu, 0, 5)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)
        for widget in widgets.values():
            self._connect_refresh(widget)
        return box

    def _build_ephys_tab(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settingsPage")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.manual_sorting_folder = QLineEdit()
        self.manual_sorting_folder.setPlaceholderText("selected sorting folder")

        self.ephys_tabs = QTabWidget()
        self.ephys_tabs.addTab(self._scroll_area(self._build_preprocess_tab()), "Preprocess")
        self.ephys_tabs.addTab(self._scroll_area(self._build_postprocess_tab()), "Postprocess")
        self.execution_scroll_area = self._scroll_area(self._build_execution_tab())
        self.execution_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._ephys_run_tab_index = self.ephys_tabs.addTab(
            self.execution_scroll_area, "Run"
        )
        self.ephys_tabs.currentChanged.connect(lambda _index: self._schedule_refresh())

        manual_box = QGroupBox("Manual Curation")
        manual_layout = QGridLayout(manual_box)
        self.run_phy = QPushButton("Launch Phy")
        self.run_phy.clicked.connect(self._run_phy)
        browse_manual_sorting = QPushButton("Browse folder")
        browse_manual_sorting.clicked.connect(self._browse_manual_sorting_folder)
        self.launch_cell_explorer = QPushButton("Run CellExplore")
        self.launch_cell_explorer.clicked.connect(self._launch_cell_explorer)
        self.cell_explorer_folders_summary = QLineEdit()
        self.cell_explorer_folders_summary.setReadOnly(True)
        self.cell_explorer_folders_summary.setPlaceholderText("No CellExplore folders selected")
        select_cell_explorer_folders = QPushButton("Select folders")
        select_cell_explorer_folders.clicked.connect(self._select_cell_explorer_folders)
        cell_explorer_hint = QLabel("CellExplore uses only checked and registered folders.")
        cell_explorer_hint.setWordWrap(True)
        manual_layout.addWidget(self.run_phy, 0, 0)
        manual_layout.addWidget(self.manual_sorting_folder, 0, 1)
        manual_layout.addWidget(browse_manual_sorting, 0, 2)
        manual_layout.addWidget(self.launch_cell_explorer, 1, 0)
        manual_layout.addWidget(self.cell_explorer_folders_summary, 1, 1)
        manual_layout.addWidget(select_cell_explorer_folders, 1, 2)
        manual_layout.addWidget(cell_explorer_hint, 2, 0, 1, 3)
        manual_layout.setColumnStretch(1, 1)

        layout.addWidget(self.ephys_tabs, 1)
        layout.addWidget(manual_box)
        return panel

    def _scroll_area(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _form_layout(self, parent: QWidget) -> QFormLayout:
        layout = QFormLayout(parent)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(9)
        return layout

    def _hint_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hintLabel")
        label.setWordWrap(True)
        return label

    def _build_preprocess_tab(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settingsPage")
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        options = QGroupBox("Preprocess options")
        options_form = self._form_layout(options)
        self.pre_overwrite = QCheckBox("overwrite preprocess outputs")
        options_form.addRow(self.pre_overwrite)

        session = QGroupBox("Session and channel map")
        session_form = self._form_layout(session)
        self.xml_path = QLineEdit()
        self.xml_path.setPlaceholderText("Auto: basepath/basename.xml, or Load XML")
        self.xml_path.textChanged.connect(lambda _text: setattr(self, "_last_xml_warning_key", None))
        self.xml_path.textChanged.connect(self._schedule_refresh)
        load_xml = QPushButton("Load XML")
        load_xml.clicked.connect(self._load_xml)
        xml_row = QHBoxLayout()
        xml_row.addWidget(load_xml)
        xml_row.addWidget(self.xml_path, 1)
        self.reject_channels = QLineEdit()
        self.reject_channels.setPlaceholderText("0, 3, 17")
        self.reject_channels.textChanged.connect(self._mark_chanmap_controls_dirty)
        self.chanmap_path = QLineEdit()
        self.chanmap_path.setPlaceholderText("optional explicit chanMap.mat path")
        load_chanmap = QPushButton("Load chanMap")
        load_chanmap.clicked.connect(self._browse_chanmap)
        chanmap_buttons = QHBoxLayout()
        chanmap_buttons.addWidget(load_chanmap)
        session_form.addRow(QLabel("XML"), xml_row)
        session_form.addRow("bad channels", self.reject_channels)
        session_form.addRow(QLabel("probe assignments"))
        session_form.addRow(self._build_probe_assignment_editor())
        session_form.addRow(chanmap_buttons)

        inputs = QGroupBox("Inputs")
        input_form = self._form_layout(inputs)
        self.digital_inputs = QCheckBox("digital inputs")
        self.analog_inputs = QCheckBox("analog inputs")
        self.save_raw = QCheckBox("save raw dat")
        self.save_raw.setVisible(False)
        input_form.addRow(self.digital_inputs)
        input_form.addRow(self.analog_inputs)

        signal = QGroupBox("Signal processing")
        sig = self._form_layout(signal)
        self.do_preprocess = QCheckBox("do preprocess")
        self.preprocess_worker_count = self._worker_spin()
        self.bandpass_min = self._double_spin(0.1, 100000.0, 500.0)
        self.bandpass_max = self._double_spin(0.1, 100000.0, 8000.0)
        self.reference = NoWheelComboBox()
        self.reference.addItems(["none", "local", "global"])
        self.local_radius = QLineEdit("20, 200")
        sig.addRow(self.do_preprocess)
        sig.addRow("workers for preprocess", self.preprocess_worker_count)
        sig.addRow("bandpass min Hz", self.bandpass_min)
        sig.addRow("bandpass max Hz", self.bandpass_max)
        sig.addRow("common median reference", self.reference)
        sig.addRow("CMR radius min/max (um)", self.local_radius)
        self.preprocess_worker_count.setVisible(False)
        if sig.labelForField(self.preprocess_worker_count) is not None:
            sig.labelForField(self.preprocess_worker_count).setVisible(False)

        state = QGroupBox("LFP and state scoring")
        st = self._form_layout(state)
        self.make_lfp = QCheckBox("Make LFP")
        self.lfp_fs = self._double_spin(1.0, 100000.0, 1250.0)
        self.state_score = QCheckBox("Run state scoring")
        self.sw_channels = QLineEdit()
        self.sw_channels.setPlaceholderText("auto")
        self.theta_channels = QLineEdit()
        self.theta_channels.setPlaceholderText("auto")
        self.state_ignore_manual = QCheckBox("Ignore manual scoring")
        self.state_save_lfp_mat = QCheckBox("Save LFP MAT")
        self.state_sticky_trigger = QCheckBox("Sticky trigger")
        self.state_window_sec = self._double_spin(0.1, 3600.0, 2.0)
        self.state_smoothfact = self._double_spin(0.1, 1000.0, 15.0)
        self.emg_th_alpha = self._double_spin(0.0, 100.0, 1.0)
        self.useEMG_NREM = QCheckBox("Use EMG for NREM")
        self.state_min_state_length = self._double_spin(0.0, 10000.0, 6.0)
        self.state_microarousal_sec = self._double_spin(0.0, 10000.0, 100.0)
        self.state_block_wake_to_rem = QCheckBox("Block Wake to REM")
        self.state_ignore_manual.setVisible(False)
        self.state_save_lfp_mat.setVisible(False)
        self.state_sticky_trigger.setVisible(False)
        st.addRow(self.make_lfp)
        st.addRow("LFP fs", self.lfp_fs)
        st.addRow(self.state_score)
        st.addRow("slow-wave channels", self.sw_channels)
        st.addRow("theta channels", self.theta_channels)
        st.addRow("state window sec", self.state_window_sec)
        st.addRow("state smooth factor", self.state_smoothfact)
        st.addRow("min state length sec", self.state_min_state_length)
        st.addRow("microarousal sec", self.state_microarousal_sec)
        st.addRow("EMG threshold alpha", self.emg_th_alpha)
        st.addRow(self.useEMG_NREM)
        st.addRow(self.state_block_wake_to_rem)

        ttl = QGroupBox("TTL artifact removal")
        ttl_form = self._form_layout(ttl)
        self.remove_ttl_artifacts = QCheckBox("Remove TTL artifacts")
        self.ttl_group = NoWheelComboBox()
        self.ttl_group.addItems(["all", "probe", "shank"])
        self.ttl_channel = self._spin(0, 15, 0)
        self.ttl_include_offset = QCheckBox("Include TTL offset")
        self.ttl_before = self._double_spin(0.0, 1000.0, 0.5)
        self.ttl_after = self._double_spin(0.0, 1000.0, 2.0)
        self.ttl_mode = NoWheelComboBox()
        self.ttl_mode.addItems(["linear", "cubic", "0"])
        ttl_form.addRow(self.remove_ttl_artifacts)
        ttl_form.addRow("TTL group mode", self.ttl_group)
        ttl_form.addRow("TTL channel", self.ttl_channel)
        ttl_form.addRow(self.ttl_include_offset)
        ttl_form.addRow("TTL ms before", self.ttl_before)
        ttl_form.addRow("TTL ms after", self.ttl_after)
        ttl_form.addRow("TTL interpolation mode", self.ttl_mode)

        highamp = QGroupBox("High-amplitude artifact removal")
        high_form = self._form_layout(highamp)
        self.remove_highamp_artifacts = QCheckBox("Remove high-amplitude artifacts")
        self.highamp_group = NoWheelComboBox()
        self.highamp_group.addItems(["all", "probe", "shank"])
        self.highamp_sigma = self._double_spin(0.1, 1000.0, 5.0)
        self.highamp_before = self._double_spin(0.0, 1000.0, 2.0)
        self.highamp_after = self._double_spin(0.0, 1000.0, 2.0)
        self.highamp_mode = NoWheelComboBox()
        self.highamp_mode.addItems(["linear", "cubic", "0"])
        high_form.addRow(self.remove_highamp_artifacts)
        high_form.addRow("High amp group mode", self.highamp_group)
        high_form.addRow("High amp sigma", self.highamp_sigma)
        high_form.addRow("High amp ms before", self.highamp_before)
        high_form.addRow("High amp ms after", self.highamp_after)
        high_form.addRow("High amp interpolation mode", self.highamp_mode)

        sorter = QGroupBox("Sorter")
        sf = self._form_layout(sorter)
        self.run_sorter = QCheckBox("Run sorter")
        self.sorter = NoWheelComboBox()
        self.sorter.addItems(["Kilosort", "Kilosort2_5", "kilosort4", "disabled"])
        self.sorter.currentTextChanged.connect(self._sorter_changed)
        self.run_sorter.toggled.connect(self._update_sorter_enabled)
        self.sorter_partition_mode = NoWheelComboBox()
        self.sorter_partition_mode.addItem("All channels", "all")
        self.sorter_partition_mode.addItem("Each probe", "probe")
        self.sorter_partition_mode.addItem("Each shank", "shank")
        self.sorter_path = QLineEdit()
        self.sorter_config_path = QLineEdit()
        self.matlab_path = QLineEdit()
        self.matlab_path.setPlaceholderText("auto-detect from PATH")
        self.open_sorter_config = QPushButton("Open config")
        self.open_sorter_config.clicked.connect(self._open_sorter_config)
        sorter_config_row = QWidget()
        sorter_config_layout = QHBoxLayout(sorter_config_row)
        sorter_config_layout.setContentsMargins(0, 0, 0, 0)
        sorter_config_layout.setSpacing(6)
        sorter_config_layout.addWidget(self.sorter_config_path, 1)
        sorter_config_layout.addWidget(self.open_sorter_config)
        self.sorter_worker_count = self._worker_spin()
        sf.addRow(self.run_sorter)
        sf.addRow("Sorter", self.sorter)
        sf.addRow("Run sorter on", self.sorter_partition_mode)
        sf.addRow("Sorter path", self.sorter_path)
        sf.addRow("Sorter config", sorter_config_row)
        sf.addRow("MATLAB path", self.matlab_path)
        sf.addRow("Workers for sorter", self.sorter_worker_count)
        self.sorter_worker_count.setVisible(False)
        if sf.labelForField(self.sorter_worker_count) is not None:
            sf.labelForField(self.sorter_worker_count).setVisible(False)
        self.matlab_path.setVisible(False)
        if sf.labelForField(self.matlab_path) is not None:
            sf.labelForField(self.matlab_path).setVisible(False)

        for widget in [
            self.pre_overwrite,
            self.reject_channels,
            self.chanmap_path,
            self.analog_inputs,
            self.digital_inputs,
            self.save_raw,
            self.do_preprocess,
            self.bandpass_min,
            self.bandpass_max,
            self.reference,
            self.local_radius,
            self.make_lfp,
            self.lfp_fs,
            self.state_score,
            self.sw_channels,
            self.theta_channels,
            self.state_ignore_manual,
            self.state_save_lfp_mat,
            self.state_sticky_trigger,
            self.state_window_sec,
            self.state_smoothfact,
            self.emg_th_alpha,
            self.useEMG_NREM,
            self.state_min_state_length,
            self.state_microarousal_sec,
            self.state_block_wake_to_rem,
            self.remove_ttl_artifacts,
            self.ttl_group,
            self.ttl_channel,
            self.ttl_include_offset,
            self.ttl_before,
            self.ttl_after,
            self.ttl_mode,
            self.remove_highamp_artifacts,
            self.highamp_group,
            self.highamp_sigma,
            self.highamp_before,
            self.highamp_after,
            self.highamp_mode,
            self.run_sorter,
            self.sorter,
            self.sorter_path,
            self.sorter_config_path,
            self.matlab_path,
            self.preprocess_worker_count,
            self.sorter_worker_count,
        ]:
            self._connect_refresh(widget)

        layout.addWidget(options)
        layout.addWidget(session)
        layout.addWidget(inputs)
        layout.addWidget(signal)
        layout.addWidget(state)
        layout.addWidget(ttl)
        layout.addWidget(highamp)
        layout.addWidget(sorter)
        layout.addStretch(1)
        return panel

    def _build_behavior_tab(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settingsPage")
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        options = QGroupBox("Behavior export")
        options_form = self._form_layout(options)
        self.behavior_enabled = QCheckBox("enable behavior export")
        self.behavior_enabled.setChecked(True)
        self.behavior_enabled.setVisible(False)
        self.behavior_overwrite = QCheckBox("overwrite behavior output")
        self.behavior_clean_jumps = QCheckBox("clean tracker jumps")
        self.behavior_clean_jumps.setChecked(True)
        self.behavior_clean_jumps.setVisible(False)
        self.behavior_dlc_batch_path = QLineEdit()
        self.behavior_dlc_batch_path.setPlaceholderText("optional DLC batch/script path")
        browse_batch = QPushButton("Browse")
        browse_batch.clicked.connect(self._browse_behavior_dlc_batch)
        options_form.addRow(self.behavior_overwrite)
        options_form.addRow("DLC batch/script", self._field_with_button(self.behavior_dlc_batch_path, browse_batch))

        dlc = QGroupBox("DLC detection")
        dlc_form = self._form_layout(dlc)
        self.behavior_primary_coords = self._spin(1, 128, 2)
        self.behavior_primary_coords.setVisible(False)
        self.behavior_primary_point = NoWheelComboBox()
        self.behavior_primary_point.addItem("Discover DLC files first", "")
        self.behavior_primary_point.currentIndexChanged.connect(self._on_behavior_primary_point_changed)
        self.behavior_likelihood = self._double_spin(0.0, 1.0, 0.6)
        self.behavior_ttl_tolerance = self._double_spin(0.0, 1.0, 0.01)
        self.behavior_ttl_tolerance.setToolTip(
            "Fractional tolerance for removing camera TTL intervals shorter than one video frame. "
            "0.010 means pulses closer than 99% of the expected frame interval are treated as extra pulses."
        )
        self.behavior_fallback_fps = self._double_spin(0.001, 1000.0, 40.0)
        discover = QPushButton("Discover DLC files")
        discover.clicked.connect(self._discover_behavior_dlc)
        self.behavior_dlc_summary = QPlainTextEdit()
        self.behavior_dlc_summary.setReadOnly(True)
        self.behavior_dlc_summary.setMinimumHeight(110)
        self.behavior_dlc_summary.setVisible(False)
        dlc_form.addRow("tracking point", self.behavior_primary_point)
        dlc_form.addRow("likelihood threshold", self.behavior_likelihood)
        dlc_form.addRow("TTL duplicate tolerance", self.behavior_ttl_tolerance)
        dlc_form.addRow("fallback video FPS (Hz)", self.behavior_fallback_fps)
        dlc_form.addRow(discover)

        self.behavior_distance_cm = self._double_spin(0.001, 1_000_000.0, 100.0)
        self.behavior_pixel_distance = self._double_spin(0.0, 1_000_000.0, 0.0)
        self.behavior_pixel_distance.setVisible(False)
        self.behavior_gap_sec = self._double_spin(0.0, 3600.0, 1.0)

        run = QGroupBox("Run")
        run_form = self._form_layout(run)
        run_behavior = QPushButton("Export behavior to local")
        run_behavior.setObjectName("primaryButton")
        run_behavior.clicked.connect(self._run_behavior_export)
        self.run_behavior = run_behavior
        run_form.addRow(run_behavior)

        for widget in [
            self.behavior_enabled,
            self.behavior_overwrite,
            self.behavior_clean_jumps,
            self.behavior_dlc_batch_path,
            self.behavior_primary_point,
            self.behavior_primary_coords,
            self.behavior_likelihood,
            self.behavior_ttl_tolerance,
            self.behavior_fallback_fps,
            self.behavior_distance_cm,
            self.behavior_pixel_distance,
            self.behavior_gap_sec,
        ]:
            self._connect_refresh(widget)
        self.behavior_distance_cm.valueChanged.connect(self._invalidate_behavior_calibration)

        layout.addWidget(options)
        layout.addWidget(dlc)
        layout.addWidget(run)
        layout.addStretch(1)
        return panel

    def _build_postprocess_tab(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settingsPage")
        layout = QVBoxLayout(panel)

        options = QGroupBox("Postprocess options")
        options_form = self._form_layout(options)
        self.post_overwrite = QCheckBox("overwrite postprocess outputs")
        options_form.addRow(self.post_overwrite)

        target = QGroupBox("Postprocess target")
        target_form = self._form_layout(target)
        self.sorting_phy_folder = QLineEdit()
        self.sorting_search_root = QLineEdit()
        self.sorting_search_root.setVisible(False)
        browse_phy = QPushButton("Browse sorting folder")
        browse_phy.clicked.connect(self._browse_sorting_folder)
        target_form.addRow(
            self._hint_label(
                "Leave sorting folder blank to use the newest sorter output under the local output directory. "
                "Choose a folder only when postprocessing an existing Phy/Kilosort result."
            )
        )
        target_form.addRow("sorting folder", self._field_with_button(self.sorting_phy_folder, browse_phy))

        recording = QGroupBox("Recording data")
        recording_form = self._form_layout(recording)
        self.post_apply_preprocess = QCheckBox("apply preprocess filter")
        recording_form.addRow(self.post_apply_preprocess)
        recording_form.addRow(
            self._hint_label(
                "Enable only when basename.dat is legacy raw data. Leave off for basename.dat generated by this pipeline."
            )
        )

        curation = QGroupBox("Curation and metrics")
        form = self._form_layout(curation)
        self.exclude_groups = QLineEdit("noise")
        self.duplicate_censored = self._double_spin(0.0, 1000.0, 0.5)
        self.duplicate_threshold = self._double_spin(0.0, 1.0, 0.5)
        self.merge_min_spikes = self._spin(0, 1000000, 100)
        self.merge_corr = self._double_spin(0.0, 10.0, 0.25)
        self.merge_template = self._double_spin(0.0, 10.0, 0.25)
        self.split_contamination = self._double_spin(0.0, 1.0, 0.05)
        self.split_threshold_mode = NoWheelComboBox()
        self.split_threshold_mode.addItems(["adaptive_chi2", "chi2", "quantile"])
        self.split_wf_threshold = self._double_spin(0.0, 10.0, 0.2)
        self.split_wf_n_chans = self._spin(1, 4096, 10)
        self.split_amp_mad_scale = self._double_spin(0.1, 1000.0, 10.0)
        self.skip_pc_metrics = QCheckBox("Skip PC metrics")
        self.noise_label_only = QCheckBox("Noise label only")
        self.noise_label_only.setVisible(False)
        self.post_worker_count = self._worker_spin()

        form.addRow("Exclude groups", self.exclude_groups)
        form.addRow("Duplicate censor ms", self.duplicate_censored)
        form.addRow("Duplicate threshold", self.duplicate_threshold)
        form.addRow("Merge min spikes", self.merge_min_spikes)
        form.addRow("Merge corr diff", self.merge_corr)
        form.addRow("Merge template diff", self.merge_template)
        form.addRow("Split contamination", self.split_contamination)
        form.addRow("Split threshold mode", self.split_threshold_mode)
        form.addRow("Split waveform threshold", self.split_wf_threshold)
        form.addRow("Split waveform n chans", self.split_wf_n_chans)
        form.addRow("Split amp MAD scale", self.split_amp_mad_scale)
        form.addRow("Workers", self.post_worker_count)
        self.post_worker_count.setVisible(False)
        if form.labelForField(self.post_worker_count) is not None:
            form.labelForField(self.post_worker_count).setVisible(False)
        form.addRow(self.skip_pc_metrics)

        noise = QGroupBox("Noise labeling thresholds")
        noise_form = self._form_layout(noise)
        for key, label in NOISE_THRESHOLD_FIELDS:
            field = QLineEdit()
            field.setPlaceholderText("blank disables")
            self.noise_threshold_fields[key] = field
            noise_form.addRow(label, field)
        run_noise_label = QPushButton("Run noise labeling only")
        run_noise_label.clicked.connect(lambda: self._start_run("noise_label"))
        noise_form.addRow(run_noise_label)
        self.run_noise_label = run_noise_label

        for widget in [
            self.sorting_phy_folder,
            self.sorting_search_root,
            self.post_apply_preprocess,
            self.exclude_groups,
            self.duplicate_censored,
            self.duplicate_threshold,
            self.merge_min_spikes,
            self.merge_corr,
            self.merge_template,
            self.split_contamination,
            self.split_threshold_mode,
            self.split_wf_threshold,
            self.split_wf_n_chans,
            self.split_amp_mad_scale,
            self.skip_pc_metrics,
            self.noise_label_only,
            self.post_overwrite,
            self.post_worker_count,
        ]:
            self._connect_refresh(widget)
        for widget in self.noise_threshold_fields.values():
            self._connect_refresh(widget)

        layout.addWidget(options)
        layout.addWidget(target)
        layout.addWidget(recording)
        layout.addWidget(curation)
        layout.addWidget(noise)
        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        monitor = QWidget()
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        monitor_layout.setSpacing(8)

        chanmap_panel = QFrame()
        chanmap_panel.setObjectName("miniPanel")
        chanmap_layout = QVBoxLayout(chanmap_panel)
        chanmap_layout.setContentsMargins(0, 0, 0, 0)
        chanmap_layout.setSpacing(0)
        chanmap_head = QLabel("chanMap Preview")
        chanmap_head.setObjectName("miniHead")
        self.chanmap_canvas = ChanMapCanvas()
        anatomical_controls = QWidget()
        anatomical_layout = QHBoxLayout(anatomical_controls)
        anatomical_layout.setContentsMargins(6, 6, 6, 6)
        anatomical_layout.setSpacing(6)
        load_anatomical = QPushButton("Load anatomical map")
        load_anatomical.clicked.connect(self._load_anatomical_map)
        edit_anatomical = QPushButton("Edit anatomical map")
        edit_anatomical.clicked.connect(self._edit_anatomical_map)
        anatomical_layout.addWidget(load_anatomical)
        anatomical_layout.addWidget(edit_anatomical)
        chanmap_layout.addWidget(chanmap_head)
        chanmap_layout.addWidget(self.chanmap_canvas, 1)
        chanmap_layout.addWidget(anatomical_controls)

        behavior_panel = QFrame()
        behavior_panel.setObjectName("miniPanel")
        behavior_layout = QVBoxLayout(behavior_panel)
        behavior_layout.setContentsMargins(0, 0, 0, 0)
        behavior_layout.setSpacing(0)
        behavior_head = QLabel("Behavior track preview")
        behavior_head.setObjectName("miniHead")
        self.behavior_mode_tabs = QTabWidget()

        calibration_page = QWidget()
        calibration_layout = QVBoxLayout(calibration_page)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_layout.setSpacing(0)
        self.behavior_frame_tabs = QTabWidget()
        self.behavior_frame_tabs.setObjectName("behaviorFrameTabs")
        calibration_controls = QWidget()
        calibration_controls_layout = QHBoxLayout(calibration_controls)
        calibration_controls_layout.setContentsMargins(8, 6, 8, 6)
        calibration_controls_layout.setSpacing(8)
        known_distance_label = QLabel("known distance cm")
        self.behavior_reset_calibration = QPushButton("Reset")
        self.behavior_reset_calibration.clicked.connect(self._reset_behavior_calibration_viewer)
        self.behavior_run_calibration = QPushButton("Run calibration")
        self.behavior_run_calibration.setObjectName("primaryButton")
        self.behavior_run_calibration.clicked.connect(self._run_behavior_calibration)
        calibration_controls_layout.addStretch(1)
        calibration_controls_layout.addWidget(known_distance_label)
        calibration_controls_layout.addWidget(self.behavior_distance_cm)
        calibration_controls_layout.addWidget(self.behavior_reset_calibration)
        calibration_controls_layout.addWidget(self.behavior_run_calibration)
        calibration_layout.addWidget(self.behavior_frame_tabs, 1)
        calibration_layout.addWidget(calibration_controls)

        outlier_page = QWidget()
        outlier_layout = QVBoxLayout(outlier_page)
        outlier_layout.setContentsMargins(0, 0, 0, 0)
        outlier_layout.setSpacing(0)
        self.behavior_outlier_tabs = QTabWidget()
        self.behavior_track_canvas = BehaviorTrackCanvas(lambda _mask: None)
        self.behavior_outlier_tabs.addTab(self.behavior_track_canvas, "Track")
        outlier_controls = QWidget()
        outlier_controls_layout = QHBoxLayout(outlier_controls)
        outlier_controls_layout.setContentsMargins(8, 6, 8, 6)
        outlier_controls_layout.setSpacing(8)
        interpolate_label = QLabel("interpolate gaps <= sec")
        self.behavior_reset_outlier = QPushButton("Reset")
        self.behavior_reset_outlier.clicked.connect(self._reset_behavior_keep_ranges)
        self.run_behavior_cleanup = QPushButton("Apply outlier cleanup + interpolate")
        self.run_behavior_cleanup.clicked.connect(self._run_behavior_outlier_cleanup)
        outlier_controls_layout.addStretch(1)
        outlier_controls_layout.addWidget(interpolate_label)
        outlier_controls_layout.addWidget(self.behavior_gap_sec)
        outlier_controls_layout.addWidget(self.behavior_reset_outlier)
        outlier_controls_layout.addWidget(self.run_behavior_cleanup)
        outlier_layout.addWidget(self.behavior_outlier_tabs, 1)
        outlier_layout.addWidget(outlier_controls)

        self.behavior_mode_tabs.addTab(calibration_page, "Calibration")
        self.behavior_mode_tabs.addTab(outlier_page, "Outlier cleanup & interpolation")
        self.behavior_mode_tabs.currentChanged.connect(self._on_behavior_mode_tab_changed)
        self.behavior_preview_status = QLabel("Discover DLC files to load epoch frames")
        self.behavior_preview_status.setWordWrap(True)
        self.behavior_preview_status.setVisible(False)
        behavior_layout.addWidget(behavior_head)
        behavior_layout.addWidget(self.behavior_mode_tabs, 1)

        self.monitor_stack = QStackedWidget()
        self.monitor_stack.setMinimumHeight(200)
        self.monitor_stack.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.monitor_stack.addWidget(chanmap_panel)
        self.monitor_stack.addWidget(behavior_panel)

        log_panel = QFrame()
        log_panel.setObjectName("miniPanel")
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)
        log_head = QLabel("Log")
        log_head.setObjectName("miniHead")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(80)
        self.log.document().setMaximumBlockCount(5000)
        log_layout.addWidget(log_head)
        log_layout.addWidget(self.log, 1)

        monitor_layout.addWidget(self.monitor_stack, 9)
        monitor_layout.addWidget(log_panel, 1)

        layout.addWidget(monitor, 1)
        return panel

    def _build_run_bar(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.move_dat_to_basepath = QCheckBox("Copy basename.dat")
        self.move_overwrite = QCheckBox("Overwrite copied files")
        self.move_clean_local = QCheckBox("Delete local after verified copy")
        self.move_clean_local.setChecked(False)
        self.move_storage_dir = QLineEdit()
        self.move_storage_dir.setReadOnly(True)
        self.move_storage_dir.setPlaceholderText("Default: Basepath")
        self.browse_move_storage = QPushButton("Browse save dir")
        self.move_outputs = QPushButton("Copy outputs to storage")
        self.force_stop = QPushButton("Force stop")
        self.force_stop.setObjectName("dangerButton")
        self.clear_log = QPushButton("Clear log")
        self.browse_move_storage.clicked.connect(self._browse_move_storage_dir)
        self.move_outputs.clicked.connect(self._move_outputs_to_storage)
        self.basepath.textChanged.connect(self._update_move_storage_destination)
        self.multi_day_name.textChanged.connect(self._update_move_storage_destination)
        self.force_stop.clicked.connect(self._force_stop_process)
        self.clear_log.clicked.connect(self.log.clear)
        layout.addWidget(self.move_dat_to_basepath, 0, 0)
        layout.addWidget(self.move_overwrite, 0, 1)
        layout.addWidget(self.move_clean_local, 0, 2)
        layout.addWidget(QLabel("Storage destination"), 1, 0)
        layout.addWidget(self.move_storage_dir, 1, 1)
        layout.addWidget(self.browse_move_storage, 1, 2)
        layout.addWidget(self.move_outputs, 1, 3)
        layout.addWidget(self.force_stop, 2, 2)
        layout.addWidget(self.clear_log, 2, 3)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self._update_move_storage_destination()
        return panel

    def _double_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        box = NoWheelDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setValue(value)
        return box

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        box = NoWheelSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    def _worker_spin(self) -> QSpinBox:
        worker_count = default_worker_count()
        return self._spin(1, worker_count, worker_count)

    def _field_with_button(self, field: QLineEdit, button: QPushButton) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        layout.addWidget(button)
        return panel

    def _build_probe_assignment_editor(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        for text, stretch in [("geometry", 2), ("XML groups", 4), ("x offset", 1), ("", 0)]:
            label = QLabel(text)
            header_layout.addWidget(label, stretch)
        layout.addWidget(header)

        self.probe_rows_container = QWidget()
        self.probe_rows_layout = QVBoxLayout(self.probe_rows_container)
        self.probe_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.probe_rows_layout.setSpacing(6)
        layout.addWidget(self.probe_rows_container)

        add_row = QPushButton("Add probe assignment")
        add_row.clicked.connect(lambda: self._add_probe_assignment_row())
        layout.addWidget(add_row)
        return panel

    def _add_probe_assignment_row(self, assignment: dict[str, Any] | None = None) -> None:
        assignment = assignment or {"type": "staggered", "groups": [], "x_offset": 0}
        row_panel = QWidget()
        row_layout = QHBoxLayout(row_panel)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        geometry = NoWheelComboBox()
        geometry.addItems(list(PROBE_TYPES))
        geometry.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        geometry.setMinimumContentsLength(11)
        geometry.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        geometry.setCurrentText(
            _normalize_chanmap_layout(str(assignment.get("type") or "staggered"))
        )

        groups = QLineEdit()
        groups.setMinimumWidth(100)
        groups.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        raw_groups = assignment.get("groups") or []
        groups.setPlaceholderText("0, 1, 2, 3")
        groups.setToolTip("Editable comma-separated 0-based XML group numbers")
        groups.setText(", ".join(str(int(v)) for v in raw_groups))

        x_offset = self._spin(-1000000, 1000000, int(assignment.get("x_offset") or 0))
        x_offset.setMinimumWidth(78)
        x_offset.setMaximumWidth(96)
        x_offset.setToolTip("Horizontal center position of this probe assignment (um)")

        remove = QPushButton("-")
        remove.setFixedWidth(34)
        remove.clicked.connect(lambda: self._remove_probe_assignment_row(row_panel))

        row_layout.addWidget(geometry, 2)
        row_layout.addWidget(groups, 4)
        row_layout.addWidget(x_offset, 1)
        row_layout.addWidget(remove)
        self.probe_rows_layout.addWidget(row_panel)

        row: dict[str, QWidget] = {
            "panel": row_panel,
            "type": geometry,
            "groups": groups,
            "x_offset": x_offset,
        }
        self._probe_rows.append(row)
        for widget in [geometry, groups, x_offset]:
            self._connect_refresh(widget)
        geometry.currentTextChanged.connect(self._mark_chanmap_controls_dirty)
        groups.textChanged.connect(self._mark_chanmap_controls_dirty)
        x_offset.valueChanged.connect(self._mark_chanmap_controls_dirty)
        self._schedule_refresh()

    def _remove_probe_assignment_row(self, row_panel: QWidget) -> None:
        if len(self._probe_rows) <= 1:
            return
        self._probe_rows = [row for row in self._probe_rows if row["panel"] is not row_panel]
        row_panel.setParent(None)
        row_panel.deleteLater()
        self._mark_chanmap_controls_dirty()
        self._schedule_refresh()

    def _render_probe_assignments(self, assignments: list[dict[str, Any]]) -> None:
        for row in self._probe_rows:
            row["panel"].setParent(None)
            row["panel"].deleteLater()
        self._probe_rows = []
        for assignment in assignments or PreprocessGuiSettings().probe_assignments:
            self._add_probe_assignment_row(assignment)

    def _probe_rows_to_assignments(self) -> list[dict[str, Any]]:
        assignments: list[dict[str, Any]] = []
        for row in self._probe_rows:
            geometry = row["type"].currentText()  # type: ignore[attr-defined]
            groups = parse_int_list(row["groups"].text())  # type: ignore[attr-defined]
            x_offset = row["x_offset"].value()  # type: ignore[attr-defined]
            assignments.append({"type": geometry, "groups": groups, "x_offset": int(x_offset)})
        return assignments

    @staticmethod
    def _mat_text(value: Any) -> str:
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        if arr.dtype.kind in {"U", "S"} and arr.ndim > 1:
            return "".join(str(v) for v in arr.reshape(-1)).strip()
        item = arr.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)

    def _assignments_from_chanmap_data(self, data: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw = data.get("probe_assignments_json")
        if raw is None:
            return None
        text = self._mat_text(raw).strip()
        if not text:
            return None
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return None
        return parsed

    def _mark_chanmap_controls_dirty(self, *_args: Any) -> None:
        if self._refresh_suspended:
            return
        self._chanmap_controls_dirty = True

    @staticmethod
    def _bad_channels_from_chanmap_data(data: dict[str, Any]) -> list[int]:
        connected = np.asarray(data.get("connected", []), dtype=bool).reshape(-1)
        if connected.size == 0:
            return []
        device_ch = np.asarray(
            data.get("chanMap0ind", np.asarray(data["chanMap"]).reshape(-1) - 1)
        ).reshape(-1).astype(int)
        n = min(len(connected), len(device_ch))
        return device_ch[:n][~connected[:n]].astype(int).tolist()

    def _connect_refresh(self, widget: QWidget) -> None:
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._schedule_refresh)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._schedule_refresh)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self._schedule_refresh)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._schedule_refresh)

    def _schedule_refresh(self) -> None:
        if self._refresh_suspended:
            return
        self._refresh_timer.start()

    def _load_chanmap_preview(self, path: Path) -> None:
        if not path.exists():
            key = (path, -1, -1)
        else:
            stat = path.stat()
            key = ("file", path.resolve(), int(stat.st_mtime_ns), int(stat.st_size))
        if key == self._last_chanmap_preview_key:
            return
        self.chanmap_canvas.load_chanmap(path)
        self._last_chanmap_preview_key = key

    def _candidate_existing_chanmaps(self, *, include_explicit: bool = True) -> list[Path]:
        settings = self._collect_settings()
        candidates: list[Path] = []

        explicit = Path(settings.chanmap_path).expanduser() if settings.chanmap_path.strip() else None
        if include_explicit and explicit is not None:
            candidates.append(explicit)

        output_dir = settings.local_output_dir
        if output_dir is not None:
            candidates.append(output_dir / "chanMap.mat")
            sorter_candidates: list[Path] = []
            for pattern in ("Kilosort_*", "Kilosort2_5_*", "Kilosort2.5_*", "Kilosort4_*"):
                for run_dir in output_dir.glob(pattern):
                    if not run_dir.is_dir() or run_dir.name.endswith("_spi"):
                        continue
                    sorter_candidates.extend(
                        [run_dir / "chanMap.mat", run_dir / "sorter_output" / "chanMap.mat"]
                    )
            sorter_candidates = [p for p in sorter_candidates if p.exists()]
            sorter_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(sorter_candidates)

        basepath = settings.basepath_path
        if basepath is not None:
            candidates.append(basepath / "chanMap.mat")

        local_root_text = self.local_root.text().strip()
        if local_root_text:
            candidates.append(Path(local_root_text).expanduser() / "chanMap.mat")

        seen: set[Path] = set()
        unique: list[Path] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(candidate)
        return unique

    def _apply_chanmap_file_to_controls(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = loadmat(path)
        assignments = self._assignments_from_chanmap_data(data)
        self._refresh_suspended = True
        try:
            self.chanmap_path.setText(str(path))
            self.reject_channels.setText(", ".join(str(v) for v in self._bad_channels_from_chanmap_data(data)))
            if assignments:
                self._render_probe_assignments(assignments)
            self._chanmap_controls_dirty = False
        finally:
            self._refresh_suspended = False
        self._last_chanmap_preview_key = None
        self.chanmap_canvas.render_chanmap(data, source=path)
        self._append_log(f"Loaded chanMap: {path}\n")
        return True

    def _auto_load_existing_chanmap(self) -> bool:
        for candidate in self._candidate_existing_chanmaps(include_explicit=False):
            if candidate.exists():
                return self._apply_chanmap_file_to_controls(candidate)
        return False

    def _warn_missing_xml_for_settings(self, settings: PipelineGuiSettings, *, context: str) -> None:
        explicit = self.xml_path.text().strip()
        if not explicit:
            return
        detail = str(Path(explicit).expanduser())
        message = f"Warning: selected XML not found: {detail}. Set XML first with Load XML.\n"
        key = (context, "explicit", detail)
        if key == self._last_xml_warning_key:
            return
        self._last_xml_warning_key = key
        self._append_warning_log(message)

    def _load_settings_chanmap_preview(self, settings: PipelineGuiSettings) -> bool:
        basepath = settings.basepath_path
        if basepath is None:
            return False
        xml_path = settings.resolved_xml_path()
        if xml_path is None or not xml_path.exists():
            self._warn_missing_xml_for_settings(settings, context="chanmap-preview")
            return False
        try:
            assignments_json = json.dumps(settings.preprocess.probe_assignments, sort_keys=True)
            stat = xml_path.stat()
            key = (
                "settings",
                xml_path.resolve(),
                int(stat.st_mtime_ns),
                int(stat.st_size),
                tuple(settings.preprocess.reject_channels),
                assignments_json,
            )
            if key == self._last_chanmap_preview_key:
                return True
            data = build_channel_map_data(
                basepath=basepath,
                basename=settings.basename,
                reject_channels=settings.preprocess.reject_channels,
                probe_assignments=settings.preprocess.probe_assignments,
                xml_path=xml_path,
                emit_warnings=False,
            )
            if data is None:
                return False
            self.chanmap_canvas.render_chanmap(data, source="current GUI settings")
            self._last_chanmap_preview_key = key
            return True
        except Exception:
            return False

    def _current_chanmap_data(self) -> dict[str, Any] | None:
        settings = self._collect_settings()
        chanmap_path = settings.resolved_chanmap_path()
        if chanmap_path is not None and chanmap_path.exists():
            return loadmat(chanmap_path)
        basepath = settings.basepath_path
        if basepath is None:
            return None
        xml_path = settings.resolved_xml_path()
        if xml_path is None or not xml_path.exists():
            return None
        return build_channel_map_data(
            basepath=basepath,
            basename=settings.basename,
            reject_channels=settings.preprocess.reject_channels,
            probe_assignments=settings.preprocess.probe_assignments,
            xml_path=xml_path,
            emit_warnings=False,
        )

    def _current_anatomical_channel_groups(self) -> list[AnatomicalChannelGroup]:
        data = self._current_chanmap_data()
        if data is None:
            return []
        return channel_groups_from_chanmap_data(data)

    def _default_anatomical_map_path(self) -> Path:
        settings = self._collect_settings()
        if settings.local_output_dir is not None:
            return settings.local_output_dir / ANATOMICAL_MAP_FILENAME
        basepath = settings.basepath_path
        if basepath is not None:
            return basepath / ANATOMICAL_MAP_FILENAME
        return REPO_ROOT / ANATOMICAL_MAP_FILENAME

    def _persist_anatomical_map_to_default(self) -> Path | None:
        groups = self._current_anatomical_channel_groups()
        if not groups:
            return None
        return save_anatomical_map_csv(self._default_anatomical_map_path(), groups, self._channel_regions)

    def _persist_cell_explorer_chan_coords(self) -> Path | None:
        settings = self._collect_settings()
        output_dir = settings.local_output_dir
        if output_dir is None or not settings.basename:
            return None
        data = self._current_chanmap_data()
        if data is None:
            return None
        return save_cell_explorer_chan_coords(
            chanmap_data=data,
            output_dir=output_dir,
            basename=settings.basename,
        )

    def _load_anatomical_map(self) -> None:
        groups = self._current_anatomical_channel_groups()
        if not groups:
            QMessageBox.information(
                self,
                "Anatomical map",
                "Generate or load a chanMap before loading an anatomical map.",
            )
            return
        default = self._default_anatomical_map_path()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load anatomical_map.csv",
            str(default),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            self._channel_regions = load_anatomical_map_csv(path, groups)
            saved = save_anatomical_map_csv(default, groups, self._channel_regions)
        except AnatomicalMapError as exc:
            QMessageBox.critical(self, "Anatomical map", str(exc))
            return
        self._append_log(f"Loaded anatomical map: {path}\n")
        self._append_log(f"Saved anatomical map for CellExplorer: {saved}\n")

    def _edit_anatomical_map(self) -> None:
        groups = self._current_anatomical_channel_groups()
        if not groups:
            QMessageBox.information(
                self,
                "Anatomical map",
                "Generate or load a chanMap before editing an anatomical map.",
            )
            return
        default = self._default_anatomical_map_path()
        if not self._channel_regions and default.exists():
            try:
                self._channel_regions = load_anatomical_map_csv(default, groups)
            except AnatomicalMapError:
                self._channel_regions = {}
        dialog = BrainRegionEditorDialog(groups, self._channel_regions, default, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._channel_regions = dialog.channel_regions
        saved = save_anatomical_map_csv(default, groups, self._channel_regions)
        self._append_log(f"Saved anatomical map for CellExplorer: {saved}\n")

    def _select_directory(self, title: str, start: str) -> str:
        dialog = QFileDialog(self, title, start or str(Path.cwd()))
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        return dialog.selectedFiles()[0] if dialog.exec() else ""

    def _select_directories(self, title: str, start: str) -> list[str]:
        dialog = QFileDialog(self, title, start or str(Path.cwd()))
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        for view in dialog.findChildren(QAbstractItemView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        return dialog.selectedFiles() if dialog.exec() else []

    def _select_open_file(self, title: str, start: str, file_filter: str) -> str:
        dialog = QFileDialog(self, title, start or str(Path.cwd()), file_filter)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        return dialog.selectedFiles()[0] if dialog.exec() else ""

    def _select_save_file(self, title: str, start: str, file_filter: str) -> str:
        dialog = QFileDialog(self, title, start or str(Path.cwd()), file_filter)
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        return dialog.selectedFiles()[0] if dialog.exec() else ""

    def _browse_basepath(self) -> None:
        path = self._select_directory("Select basepath", self.basepath.text() or str(Path.cwd()))
        if path:
            self._clear_move_storage_override()
            self._set_subsession_order([])
            self.basepath.setText(path)
            self._auto_load_existing_xml()
            self._auto_load_existing_chanmap()
            self._schedule_refresh()

    def _auto_load_existing_xml(self) -> None:
        basepath_text = self.basepath.text().strip()
        if not basepath_text:
            self.xml_path.clear()
            return
        basepath = Path(basepath_text).expanduser()
        candidate = basepath / f"{basepath.name}.xml"
        self.xml_path.setText(str(candidate) if candidate.exists() else "")

    def _load_xml(self) -> None:
        path = self._select_open_file(
            "Select XML metadata",
            self.xml_path.text() or self.basepath.text() or str(Path.cwd()),
            "XML files (*.xml);;All files (*)",
        )
        if path:
            xml_path = Path(path).expanduser()
            try:
                assignments, skipped_channels = derive_probe_assignments_from_xml(xml_path)
            except Exception as exc:
                self.xml_path.setText(path)
                self._chanmap_controls_dirty = True
                self._last_chanmap_preview_key = None
                self.chanmap_canvas.show_empty()
                self.chanmap_canvas.summary.setText(
                    f"XML channel groups unavailable: {xml_path}"
                )
                self._append_warning_log(f"Could not load XML channel groups: {xml_path}\n{exc}\n")
                self._schedule_refresh()
                return

            self._refresh_suspended = True
            try:
                self.xml_path.setText(str(xml_path))
                self._render_probe_assignments(assignments)
                xml_bad = sorted(set(int(channel) for channel in skipped_channels))
                self.reject_channels.setText(", ".join(str(v) for v in xml_bad))
                # An explicit XML selection invalidates a previously loaded
                # chanMap file as the preview source until the map is regenerated.
                self._chanmap_controls_dirty = True
                self._last_chanmap_preview_key = None
            finally:
                self._refresh_suspended = False
            self.chanmap_canvas.show_empty()
            self._append_log(f"Loaded XML: {xml_path}\n")
            self._refresh_preview()

    def _set_multi_day_session_paths(
        self,
        paths: list[str],
        *,
        selected_subepoch_paths: list[str] | None = None,
    ) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in paths:
            text = str(item).strip()
            if not text:
                continue
            key = str(Path(text).expanduser())
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        previous = list(self._multi_day_session_paths)
        self._multi_day_session_paths = cleaned
        if cleaned != previous and hasattr(self, "move_storage_dir"):
            self._clear_move_storage_override()
        if selected_subepoch_paths is not None:
            selected: list[str] = []
            selected_seen: set[str] = set()
            for item in selected_subepoch_paths:
                text = str(item).strip()
                if not text:
                    continue
                key = str(Path(text).expanduser().resolve())
                if key in selected_seen:
                    continue
                selected_seen.add(key)
                selected.append(key)
            self._multi_day_selected_subepoch_paths = selected
        elif cleaned != previous:
            self._multi_day_selected_subepoch_paths = []
        if not cleaned:
            self._multi_day_selected_subepoch_paths = []
        selection_label = (
            f"{len(self._multi_day_selected_subepoch_paths)} subepochs"
            if self._multi_day_selected_subepoch_paths
            else "all subepochs"
        )
        if not cleaned:
            self.multi_day_sessions.clear()
        elif len(cleaned) == 1:
            self.multi_day_sessions.setText(f"1 session: {Path(cleaned[0]).name}; {selection_label}")
        else:
            self.multi_day_sessions.setText(
                f"{len(cleaned)} sessions: {Path(cleaned[0]).name} -> {Path(cleaned[-1]).name}; "
                f"{selection_label}"
            )
        if hasattr(self, "move_storage_dir"):
            self._update_move_storage_destination()

    def _set_subsession_order(self, paths: list[str]) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in paths:
            text = str(item).strip()
            if not text:
                continue
            path = Path(text).expanduser()
            key = str(path) if path.is_absolute() else path.as_posix()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        if cleaned != self._subsession_order and hasattr(self, "move_storage_dir"):
            self._clear_move_storage_override()
        self._subsession_order = cleaned

    def _view_basepath_order(self) -> None:
        if self._multi_day_session_paths:
            QMessageBox.information(
                self,
                "Subsession order",
                "Multi-day mode is active. Use the Check order button on the multi-day row.",
            )
            return
        basepath_text = self.basepath.text().strip()
        if not basepath_text:
            QMessageBox.information(
                self,
                "Subsession order",
                "Select a basepath first.",
            )
            return
        basepath = Path(basepath_text).expanduser()
        if not basepath.is_dir():
            QMessageBox.warning(
                self,
                "Subsession order",
                f"Basepath does not exist:\n{basepath}",
            )
            return
        try:
            paths = discover_subsessions(
                basepath=basepath,
                sort_files=True,
                alt_sort=None,
                ignore_folders=[],
                subsession_order=self._subsession_order or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Cannot check subsession order", str(exc))
            return
        if not paths:
            QMessageBox.information(
                self,
                "Subsession order",
                "No input recordings were discovered under this basepath.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Subsession concat order")
        dialog.resize(1050, 500)
        dialog.setStyleSheet(
            """
            QDialog {
                background: #252525;
                color: #e5e5e5;
            }
            QTableWidget {
                background: #1f1f1f;
                alternate-background-color: #272727;
                color: #e5e5e5;
                gridline-color: #565656;
                selection-background-color: #2d6f9f;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #303030;
                color: #f0f0f0;
                border: 1px solid #555555;
                padding: 5px;
            }
            """
        )
        layout = QVBoxLayout(dialog)
        explanation = QLabel(
            "Edit the Order value to define the exact concat order. "
            "The saved order is validated against all discovered recordings before preprocessing."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        table = SubsessionOrderTable(basepath, paths, dialog)
        table.setAlternatingRowColors(True)
        layout.addWidget(table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        update_button = buttons.addButton(
            "Update order", QDialogButtonBox.ButtonRole.AcceptRole
        )

        def apply_order() -> None:
            ordered_paths = table.paths()
            if len(ordered_paths) != len(paths) or len(set(ordered_paths)) != len(paths):
                QMessageBox.warning(dialog, "Update order", "Subsession table is incomplete.")
                return
            relative_order = table.relative_paths()
            self._set_subsession_order(relative_order)
            self._schedule_refresh()
            self._append_log(
                "Updated single-session concat order:\n"
                + "".join(
                    f"  [{index}] {path}\n"
                    for index, path in enumerate(relative_order, start=1)
                )
            )
            dialog.accept()

        update_button.clicked.connect(apply_order)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _set_cell_explorer_sorting_folders(self, paths: list[str]) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in paths:
            text = str(item).strip()
            if not text:
                continue
            key = str(Path(text).expanduser().resolve())
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        self._cell_explorer_sorting_folders = cleaned
        if not hasattr(self, "cell_explorer_folders_summary"):
            return
        if not cleaned:
            self.cell_explorer_folders_summary.clear()
        elif len(cleaned) == 1:
            self.cell_explorer_folders_summary.setText(f"1 folder: {Path(cleaned[0]).name}")
        else:
            self.cell_explorer_folders_summary.setText(f"{len(cleaned)} folders selected")

    def _cell_explorer_candidate_rows(self, settings: PipelineGuiSettings) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen: set[str] = set()

        def _add(path: Path, source: str, status_note: str = "", group: str | None = None) -> None:
            resolved = path.expanduser().resolve()
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            if status_note:
                status = f"{source}; {status_note}"
            else:
                status = f"{source}; {'ok' if (resolved / 'params.py').exists() else 'missing params.py'}"
            rows.append(
                {
                    "path": key,
                    "source": source,
                    "status": status,
                    "group": group or CellExploreFolderTable._default_group_key(resolved),
                }
            )

        for path_text in self._cell_explorer_sorting_folders:
            _add(Path(path_text), "registered")

        roots = [
            _path
            for _path in (
                settings.local_output_dir,
                Path(settings.postprocess.sorting_search_root).expanduser()
                if settings.postprocess.sorting_search_root.strip()
                else None,
            )
            if _path is not None and _path.exists()
        ]
        for root in roots:
            manifest_path = root / "sorter_partition_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._append_warning_log(f"[WARN] Failed to read sorter partition manifest {manifest_path}: {exc}\n")
                continue
            for partition in payload.get("partitions", []):
                folder_text = str(partition.get("output_folder") or "").strip()
                if not folder_text:
                    continue
                raw_folder = Path(folder_text).expanduser()
                raw_group = str(raw_folder.expanduser().resolve())
                spi_folder = postprocess_output_folder_for_sorting(raw_folder)
                if raw_folder.exists():
                    raw_status = "_spi available" if spi_folder.exists() else "raw folder"
                    _add(raw_folder, "manifest", raw_status, raw_group)
                if spi_folder.exists():
                    _add(spi_folder, "manifest", "postprocess _spi", raw_group)
        return rows

    def _select_cell_explorer_folders(self) -> None:
        settings = self._collect_settings()
        rows = self._cell_explorer_candidate_rows(settings)
        if not rows:
            QMessageBox.information(
                self,
                "CellExplore folders",
                "No candidate folders found. Run postprocess first or add folders manually.",
            )
        dialog = QDialog(self)
        dialog.setWindowTitle("CellExplore folders")
        dialog.resize(1100, 520)
        dialog.setStyleSheet(
            """
            QDialog {
                background: #252525;
                color: #e5e5e5;
            }
            QTableWidget {
                background: #1f1f1f;
                color: #eeeeee;
                gridline-color: #555555;
                selection-background-color: #327aa8;
            }
            QHeaderView::section {
                background: #333333;
                color: #eeeeee;
                border: 1px solid #555555;
                padding: 4px;
            }
            """
        )
        layout = QVBoxLayout(dialog)
        table = CellExploreFolderTable(rows, self._cell_explorer_sorting_folders, dialog)
        layout.addWidget(table, 1)
        buttons = QHBoxLayout()
        add_button = QPushButton("Add folder")
        register_button = QPushButton("Register checked")
        close_button = QPushButton("Close")
        buttons.addStretch(1)
        buttons.addWidget(add_button)
        buttons.addWidget(register_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        def _add_folder() -> None:
            path = self._select_directory(
                "Add CellExplore sorting folder",
                self.local_root.text() or str(Path.cwd()),
            )
            if path:
                table.add_folder(Path(path), checked=True)

        def _register() -> None:
            self._set_cell_explorer_sorting_folders(table.selected_paths())
            self._append_log(
                "Registered CellExplore folders:\n"
                + "".join(f"  - {path}\n" for path in self._cell_explorer_sorting_folders)
            )
            dialog.accept()

        add_button.clicked.connect(_add_folder)
        register_button.clicked.connect(_register)
        close_button.clicked.connect(dialog.reject)
        dialog.exec()

    def _view_multi_day_sessions(self) -> None:
        if not self._multi_day_session_paths:
            QMessageBox.information(
                self,
                "Multi-day sessions",
                "No multi-day sessions selected. Use Browse for multi-days first.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Multi-day session order and subepochs")
        dialog.resize(1200, 540)
        dialog.setStyleSheet(
            """
            QDialog {
                background: #252525;
                color: #e5e5e5;
            }
            QTableWidget {
                background: #1f1f1f;
                alternate-background-color: #272727;
                color: #e5e5e5;
                gridline-color: #565656;
                selection-background-color: #2d6f9f;
                selection-color: #ffffff;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background: #303030;
                color: #f0f0f0;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton {
                background: #3a3a3a;
                color: #f0f0f0;
                border: 1px solid #5f5f5f;
                border-radius: 5px;
                padding: 7px 14px;
            }
            QPushButton:hover {
                background: #464646;
                border-color: #777777;
            }
            """
        )
        layout = QVBoxLayout(dialog)
        table = MultiDaySessionOrderTable(
            self._multi_day_session_paths,
            self._multi_day_selected_subepoch_paths,
            dialog,
        )
        table.setAlternatingRowColors(True)
        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        update_button = buttons.addButton("Update selection", QDialogButtonBox.ButtonRole.AcceptRole)

        def apply_order() -> None:
            paths = table.paths()
            if len(paths) != len(self._multi_day_session_paths):
                QMessageBox.warning(dialog, "Update order", "Session table is incomplete.")
                return
            if table.discovered_subepoch_count() == 0:
                QMessageBox.warning(dialog, "Update selection", "No subepochs were discovered.")
                return
            selected_subepochs = table.selected_subepoch_paths()
            if not selected_subepochs:
                QMessageBox.warning(dialog, "Update selection", "Select at least one subepoch.")
                return
            self._set_multi_day_session_paths(
                paths,
                selected_subepoch_paths=selected_subepochs,
            )
            if len(paths) >= 2:
                self.basepath.setText(paths[0])
            self._auto_load_existing_chanmap()
            self._schedule_refresh()
            self._append_log(
                f"Updated multi-day session order and selected {len(selected_subepochs)} subepochs.\n"
            )
            dialog.accept()

        update_button.clicked.connect(apply_order)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _browse_multi_days(self) -> None:
        paths = self._select_directories(
            "Select multi-day session folders",
            self.basepath.text() or str(Path.cwd()),
        )
        if not paths:
            return
        self._clear_move_storage_override()
        cleaned = [str(Path(path).expanduser()) for path in paths]
        self._set_multi_day_session_paths(cleaned, selected_subepoch_paths=[])
        if not self.multi_day_name.text().strip() and len(cleaned) >= 2:
            first = Path(cleaned[0]).name
            last = Path(cleaned[-1]).name
            self.multi_day_name.setText(f"multiday_{first}_to_{last}")
        self.basepath.setText(cleaned[0])
        existing_multiday_xml = Path(cleaned[0]).parent / self.multi_day_name.text().strip() / f"{self.multi_day_name.text().strip()}.xml"
        if existing_multiday_xml.exists():
            self.xml_path.setText(str(existing_multiday_xml))
        else:
            self._auto_load_existing_xml()
        self._auto_load_existing_chanmap()
        self._schedule_refresh()

    def _browse_local_session_to_resume(self) -> None:
        start = self.local_root.text().strip() or str(Path.cwd())
        path = self._select_directory(
            "Select local session output to resume",
            start,
        )
        if not path:
            return
        session_dir = Path(path).expanduser().resolve()
        try:
            recovered = load_local_session_resume(session_dir)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Cannot resume local session",
                str(exc),
            )
            return

        self._apply_settings(recovered.settings, preserve_missing_paths=True)
        self._clear_move_storage_override()
        if recovered.run_dir is not None:
            self._set_active_run(recovered.run_dir, session_dir=session_dir)
            self._request_run_reconcile()
            detail = f" and reconnected to Run {recovered.run_dir.name}"
        else:
            self._active_run_dir = None
            self._active_run_session_dir = None
            self._persistent_log_offsets.clear()
            self._persistent_log_announced.clear()
            self._last_persistent_status_signature = ""
            self.execution_run_status.setText("No active Run")
            self.execution_run_status.setToolTip("")
            for stage in StageName:
                self.execution_stage_status_labels[stage.value].setText("—")
                self.execution_stage_job_labels[stage.value].setText("—")
            self.force_stop.setEnabled(False)
            self._refresh_persistent_run_monitor()
            detail = " from its completed Run record"
        self._auto_load_existing_chanmap()
        self._append_log(
            f"Loaded local session {session_dir}{detail}. No job was started or retried.\n"
        )
        self._schedule_refresh()

    def _browse_local_root(self) -> None:
        path = self._select_directory("Select local output root", self.local_root.text() or str(Path.cwd()))
        if path:
            self.local_root.setText(path)
            self._auto_load_existing_chanmap()
            self._schedule_refresh()

    def _apply_environment_backend_default(
        self, capabilities: SlurmCapabilities
    ) -> None:
        if not self._environment_backend_default_pending:
            return
        self._environment_backend_default_pending = False
        requested = (
            RequestedBackend.SLURM
            if _has_slurm_server_commands(
                capabilities, require_sacct=True
            )
            else RequestedBackend.AUTO
        )
        index = self.execution_backend.findData(requested.value)
        self.execution_backend.setCurrentIndex(index if index >= 0 else 0)

    def _refresh_slurm_capabilities(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._slurm_capabilities = detect_slurm_capabilities(timeout=3.0)
        finally:
            QApplication.restoreOverrideCursor()
        capabilities = self._slurm_capabilities
        assert capabilities is not None
        self._apply_environment_backend_default(capabilities)
        requested = RequestedBackend(
            str(self.execution_backend.currentData() or RequestedBackend.AUTO.value)
        )
        require_sacct = True
        if requested == RequestedBackend.LOCAL:
            detail = "Jobs will run on this computer."
        elif capabilities.usable(require_sacct=require_sacct):
            detail = "Slurm is available."
        elif requested == RequestedBackend.AUTO:
            detail = "Slurm is unavailable; Auto will run locally."
        else:
            detail = "Slurm is unavailable; submission will stop with an error."
        errors = "; ".join(capabilities.errors)
        self.execution_resolved.setText(detail)
        self.execution_resolved.setToolTip(errors)

    def _browse_persistent_run(self) -> None:
        start = self.local_root.text().strip() or str(Path.cwd())
        path = self._select_directory("Reopen persistent Run directory", start)
        if not path:
            return
        run_dir = Path(path).expanduser().resolve()
        if not (run_dir / "run.json").exists():
            QMessageBox.critical(self, "Invalid Run", f"run.json was not found in {run_dir}")
            return
        self._set_active_run(run_dir)
        self._request_run_reconcile()

    @staticmethod
    def _session_dir_for_run(run_dir: Path) -> Path | None:
        try:
            run = read_json(Path(run_dir) / "run.json")
            claim_text = str(run.get("session_claim_path") or "").strip()
            if claim_text:
                return Path(claim_text).expanduser().resolve().parent
            session_text = str(run.get("session_output_dir") or "").strip()
            return Path(session_text).expanduser().resolve() if session_text else None
        except Exception:
            return None

    def _current_persistent_session_dir(self) -> Path | None:
        try:
            settings = self._collect_settings()
            return (
                Path(settings.local_output_dir).resolve()
                if settings.local_output_dir is not None
                else None
            )
        except Exception:
            return None

    def _set_active_run(
        self, run_dir: Path, *, session_dir: Path | None = None
    ) -> None:
        self._active_run_dir = Path(run_dir).resolve()
        self._active_run_session_dir = (
            Path(session_dir).resolve()
            if session_dir is not None
            else self._session_dir_for_run(self._active_run_dir)
        )
        self._persistent_monitor_ticks = 0
        self._persistent_log_offsets.clear()
        self._persistent_log_announced.clear()
        self._last_persistent_status_signature = ""
        self._run_monitor_timer.start()
        self._refresh_persistent_run_monitor()

    def _launch_persistent_controller(
        self, arguments: list[str], *, report_errors: bool = True
    ) -> bool:
        if self._active_run_dir is None:
            if report_errors:
                QMessageBox.information(self, "Persistent Run", "No persistent Run is selected.")
            return False
        command_args = [
            "-m",
            "src.execution.controller",
            "--run-dir",
            str(self._active_run_dir),
            *arguments,
        ]
        result = QProcess.startDetached(sys.executable, command_args, str(REPO_ROOT))
        started = bool(result[0]) if isinstance(result, tuple) else bool(result)
        if not started and report_errors:
            QMessageBox.critical(
                self,
                "Persistent Run",
                "Failed to start the detached Run controller.",
            )
        return started

    def _request_run_reconcile(self) -> None:
        if self._launch_persistent_controller(["--reconcile"]):
            QTimer.singleShot(750, self._refresh_persistent_run_monitor)

    def _request_run_resume(self) -> None:
        if self._launch_persistent_controller([]):
            self._append_log(f"Resume/submit requested for Run: {self._active_run_dir}\n")
            QTimer.singleShot(750, self._refresh_persistent_run_monitor)

    def _request_persistent_run_cancel(self) -> None:
        if self._active_run_dir is None:
            return
        answer = QMessageBox.question(
            self,
            "Force stop",
            "Stop the current Run and cancel all active Local/Slurm jobs? "
            "Logs will be kept. Incomplete preprocessing binary files will be "
            "removed after worker termination is confirmed.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._launch_persistent_controller(["--cancel"]):
            self.force_stop.setEnabled(False)
            self._update_pipeline_run_buttons(persistent_active=True)
            self._append_log(f"Force stop requested for Run: {self._active_run_dir}\n")
            QTimer.singleShot(750, self._refresh_persistent_run_monitor)

    def _resource_settings_from_widgets(self, stage: StageName) -> StageResourceGuiSettings:
        widgets = self._resource_widgets[stage.value]
        return StageResourceGuiSettings(
            cpus=int(widgets["cpus"].value()),  # type: ignore[attr-defined]
            memory_mb=int(widgets["memory_gib"].value()) * 1024,  # type: ignore[attr-defined]
            walltime_minutes=None,
            gpu_count=1 if stage == StageName.SORTING else 0,
        )

    def _request_persistent_stage_cancel(self) -> None:
        if self._active_run_dir is None:
            return
        try:
            state = RunStore(self._active_run_dir).derive_state()
        except Exception as exc:
            QMessageBox.critical(self, "Cancel Stage failed", str(exc))
            return
        candidates = [
            stage
            for stage in StageName
            if state["stages"][stage.value]["enabled"]
            and state["stages"][stage.value]["status"]
            not in {
                StageStatus.COMPLETED.value,
                StageStatus.FAILED.value,
                StageStatus.CANCELLED.value,
                StageStatus.SUPERSEDED.value,
            }
        ]
        if not candidates:
            QMessageBox.information(self, "Cancel Stage", "No active Stage can be cancelled.")
            return
        selected, accepted = QInputDialog.getItem(
            self,
            "Cancel Stage and downstream",
            "Stage",
            [stage.value for stage in candidates],
            0,
            False,
        )
        if not accepted:
            return
        answer = QMessageBox.question(
            self,
            "Cancel Stage",
            f"Cancel {selected} and all downstream Attempts? Logs and outputs will be kept.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self._launch_persistent_controller(["--cancel-stage", str(selected)]):
            self._append_log(f"Cancellation requested from Stage {selected}.\n")
            QTimer.singleShot(750, self._refresh_persistent_run_monitor)

    def _request_persistent_stage_retry(self) -> None:
        if self._active_run_dir is None:
            return
        try:
            state_path = self._active_run_dir / "state.json"
            state = read_json(state_path) if state_path.exists() else RunStore(self._active_run_dir).derive_state()
        except Exception as exc:
            QMessageBox.critical(self, "Retry failed", str(exc))
            return
        retryable = {
            StageStatus.FAILED.value,
            StageStatus.CANCELLED.value,
            StageStatus.BLOCKED.value,
        }
        candidates = [
            stage
            for stage in StageName
            if state["stages"][stage.value]["enabled"]
            and state["stages"][stage.value]["status"] in retryable
        ]
        if not candidates:
            QMessageBox.information(
                self,
                "Retry",
                "No failed, cancelled, or blocked Stage is retryable. Lost/unknown Slurm "
                "Attempts require scheduler investigation to avoid duplicate execution.",
            )
            return
        labels = [stage.value for stage in candidates]
        selected, accepted = QInputDialog.getItem(
            self,
            "Retry Stage",
            "Stage",
            labels,
            0,
            False,
        )
        if not accepted:
            return
        stage = StageName(selected)
        resource = self._resource_settings_from_widgets(stage).to_resource_spec()
        arguments = [
            "--retry",
            stage.value,
            "--cpus",
            str(resource.cpus),
            "--memory-mb",
            str(resource.memory_mb),
        ]
        if resource.walltime_minutes is None:
            arguments.append("--unlimited-walltime")
        else:
            arguments.extend(["--walltime-minutes", str(resource.walltime_minutes)])
        if self._launch_persistent_controller(arguments):
            walltime_text = (
                "partition default"
                if resource.walltime_minutes is None
                else f"{resource.walltime_minutes} min"
            )
            self._append_log(
                f"Retry requested for {stage.value} with CPU={resource.cpus}, "
                f"RAM={resource.memory_mb} MiB, walltime={walltime_text}.\n"
            )
            QTimer.singleShot(750, self._refresh_persistent_run_monitor)

    def _discover_active_persistent_run(self) -> Path | None:
        try:
            settings = self._collect_settings()
            session_dir = settings.local_output_dir
            if session_dir is None:
                return None
            claim_path = Path(session_dir) / ".pipeline-active-run.json"
            if not claim_path.exists():
                return None
            claim = read_json(claim_path)
            if claim.get("kind", "run") != "run":
                return None
            run_dir = Path(str(claim.get("run_dir") or "")).expanduser().resolve()
            return run_dir if (run_dir / "run.json").exists() else None
        except Exception:
            return None

    @staticmethod
    def _selected_attempt_view(stage_view: dict[str, Any]) -> dict[str, Any] | None:
        attempts = stage_view.get("attempts") or []
        selected = stage_view.get("selected_attempt")
        for attempt in attempts:
            if attempt.get("attempt") == selected:
                return attempt
        return attempts[-1] if attempts else None

    @staticmethod
    def _human_stage_status(stage_view: dict[str, Any]) -> str:
        if not stage_view.get("enabled", False):
            return "Not requested"
        attempt = MainWindow._selected_attempt_view(stage_view) or {}
        status = str(stage_view.get("status") or "unknown").lower()
        if attempt.get("cancel_requested") and status in {
            StageStatus.PENDING.value,
            StageStatus.SUBMITTED.value,
            StageStatus.RUNNING.value,
        }:
            return "Stopping"
        observation = ((attempt.get("latest_observation") or {}).get("status") or {})
        reason = str(observation.get("reason") or "")
        if status == StageStatus.SUBMITTED.value and "dependency" in reason.lower():
            return "Waiting for previous Stage"
        labels = {
            StageStatus.PENDING.value: "Not started",
            StageStatus.SUBMITTED.value: "Queued",
            StageStatus.RUNNING.value: "Running",
            StageStatus.COMPLETED.value: "Completed",
            StageStatus.FAILED.value: "Failed",
            StageStatus.CANCELLED.value: "Cancelled",
            StageStatus.BLOCKED.value: "Blocked",
            StageStatus.LOST.value: "Status unknown",
            StageStatus.SUPERSEDED.value: "Replaced",
        }
        return labels.get(status, status.capitalize() or "Unknown")

    def _persistent_progress_log_paths(self, state: dict[str, Any]) -> list[Path]:
        if self._active_run_dir is None:
            return []
        candidates: list[Path] = []
        stages = state.get("stages") or {}
        for stage in StageName:
            stage_view = stages.get(stage.value) or {}
            attempt = self._selected_attempt_view(stage_view)
            if attempt is None:
                continue
            attempt_number = int(attempt.get("attempt") or 0)
            if attempt_number <= 0:
                continue
            attempt_dir = (
                self._active_run_dir
                / "stages"
                / stage.value
                / f"attempt-{attempt_number:03d}"
            )
            candidates.extend([attempt_dir / "stdout.log", attempt_dir / "stderr.log"])

        preprocess_view = stages.get(StageName.PREPROCESS.value) or {}
        preprocess_attempt = self._selected_attempt_view(preprocess_view) or {}
        preprocess_result = preprocess_attempt.get("result") or {}
        preprocess_outputs = preprocess_result.get("outputs") or {}
        preprocess_data = preprocess_outputs.get("preprocess_result") or {}
        session_text = str(preprocess_data.get("local_output_dir") or "").strip()
        if session_text:
            manifest_path = Path(session_text) / "sorter_partition_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = read_json(manifest_path)
                    for partition in manifest.get("partitions") or []:
                        output_text = str(partition.get("output_folder") or "").strip()
                        if not output_text:
                            continue
                        output = Path(output_text)
                        candidates.extend(
                            [
                                output / "sorter_output" / "kilosort.log",
                                output / "kilosort.log",
                            ]
                        )
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

        unique: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.expanduser().resolve()
            if resolved not in seen and resolved.exists():
                seen.add(resolved)
                unique.append(resolved)
        return unique

    def _tail_persistent_progress_logs(self, state: dict[str, Any]) -> None:
        max_initial_bytes = 128 * 1024
        for path in self._persistent_progress_log_paths(state):
            try:
                size = path.stat().st_size
                offset = self._persistent_log_offsets.get(path)
                if offset is None:
                    offset = max(0, size - max_initial_bytes)
                elif offset > size:
                    offset = 0
                with path.open("rb") as handle:
                    handle.seek(offset)
                    payload = handle.read()
                    self._persistent_log_offsets[path] = handle.tell()
            except OSError:
                continue
            if not payload:
                continue
            if path not in self._persistent_log_announced:
                self._persistent_log_announced.add(path)
                try:
                    label = str(path.relative_to(self._active_run_dir))
                except ValueError:
                    label = path.name
                self._queue_log(f"\n--- {label} ---\n")
            text = payload.decode(errors="replace").replace("\b", "").replace("\r", "")
            self._queue_log(text)

    @staticmethod
    def _persistent_state_has_active_work(state: dict[str, Any]) -> bool:
        active = {
            StageStatus.PENDING.value,
            StageStatus.SUBMITTED.value,
            StageStatus.RUNNING.value,
        }
        for view in (state.get("stages") or {}).values():
            if bool(view.get("enabled")) and str(view.get("status")) in active:
                return True
            for attempt in view.get("attempts") or []:
                if not attempt.get("jobs"):
                    continue
                # These statuses require an immutable worker terminal fact or a
                # conclusive backend/cancellation fact. A stale non-terminal
                # observation must not keep the GUI locked after the worker has
                # already written failure.json. Superseded Attempts intentionally
                # do not get this shortcut because an older job may still be alive.
                if str(attempt.get("status") or "") in {
                    StageStatus.COMPLETED.value,
                    StageStatus.FAILED.value,
                    StageStatus.CANCELLED.value,
                }:
                    continue
                status = ((attempt.get("latest_observation") or {}).get("status") or {})
                terminal = bool(status.get("terminal", False))
                conclusive = terminal and (
                    status.get("successful") is not None
                    or str(status.get("state") or "").lower()
                    in {"cancelled", "canceled"}
                )
                if not conclusive:
                    return True
        return False

    def _update_pipeline_run_buttons(self, *, persistent_active: bool) -> None:
        legacy_running = (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        )
        enabled = (
            not persistent_active
            and not legacy_running
            and not self._phy_is_running()
            and not self._cell_explorer_is_running()
        )
        for button in (self.run_all, self.run_pre, self.run_post):
            button.setEnabled(enabled)

    def _refresh_persistent_run_monitor(self) -> None:
        current_session = self._current_persistent_session_dir()
        if (
            self._active_run_dir is not None
            and current_session is not None
            and self._active_run_session_dir is not None
            and current_session != self._active_run_session_dir
        ):
            discovered = self._discover_active_persistent_run()
            if discovered is not None:
                self._set_active_run(discovered, session_dir=current_session)
                self._append_log(
                    f"\n=== Reconnected to active Run: {discovered.name} ===\n"
                )
                return
            self._active_run_dir = None
            self._active_run_session_dir = None
            self._persistent_log_offsets.clear()
            self._persistent_log_announced.clear()
            self._last_persistent_status_signature = ""
            self.execution_run_status.setText("No active Run for this session")
            self.execution_run_status.setToolTip("")
            for stage in StageName:
                self.execution_stage_status_labels[stage.value].setText("—")
                self.execution_stage_job_labels[stage.value].setText("—")
            self.force_stop.setEnabled(False)
            legacy_running = (
                self._process is not None
                and self._process.state() != QProcess.ProcessState.NotRunning
            )
            self.move_outputs.setEnabled(not legacy_running)
            self.browse_move_storage.setEnabled(not legacy_running)
            self._update_pipeline_run_buttons(persistent_active=False)
            return
        if self._active_run_dir is None:
            discovered = self._discover_active_persistent_run()
            if discovered is None:
                self.execution_run_status.setText("No active Run")
                legacy_running = (
                    self._process is not None
                    and self._process.state() != QProcess.ProcessState.NotRunning
                )
                self.move_outputs.setEnabled(not legacy_running)
                self.browse_move_storage.setEnabled(not legacy_running)
                self._update_pipeline_run_buttons(persistent_active=False)
                return
            self._set_active_run(discovered, session_dir=current_session)
            self._append_log(f"\n=== Reconnected to active Run: {discovered.name} ===\n")
            return
        try:
            state_path = self._active_run_dir / "state.json"
            state = (
                read_json(state_path)
                if state_path.exists()
                else RunStore(self._active_run_dir).derive_state()
            )
        except Exception as exc:
            self.execution_run_status.setText(f"Run state error: {exc}")
            self._update_pipeline_run_buttons(persistent_active=True)
            return

        run_status = str(state.get("status") or "unknown")
        backend = str(state.get("resolved_backend") or "unknown")
        run_id = str(state.get("run_id") or self._active_run_dir.name)
        run_status_label = {
            "pending": "Not started",
            "submitted": "Queued",
            "running": "Running",
            "completed": "Completed",
            "failed": "Failed",
            "cancelled": "Cancelled",
            "lost": "Status unknown",
        }.get(run_status, run_status.capitalize())
        self.execution_run_status.setText(
            f"{run_status_label} on {backend.capitalize()} · {run_id}"
        )
        self.execution_run_status.setToolTip(str(self._active_run_dir))
        self._persistent_monitor_ticks += 1
        needs_backend_reconcile = any(
            attempt.get("jobs")
            and not (
                bool(
                    ((attempt.get("latest_observation") or {}).get("status") or {}).get(
                        "terminal", False
                    )
                )
                and (
                    ((attempt.get("latest_observation") or {}).get("status") or {}).get(
                        "successful"
                    )
                    is not None
                    or str(
                        ((attempt.get("latest_observation") or {}).get("status") or {}).get(
                            "state", ""
                        )
                    ).lower()
                    in {"cancelled", "canceled"}
                )
            )
            for stage_view in state.get("stages", {}).values()
            for attempt in stage_view.get("attempts", [])
        )
        if (
            backend == BackendName.SLURM.value
            and needs_backend_reconcile
            and self._persistent_monitor_ticks % 8 == 0
        ):
            self._launch_persistent_controller(["--reconcile"], report_errors=False)

        status_parts: list[str] = []
        for stage in StageName:
            stage_view = (state.get("stages") or {}).get(stage.value) or {}
            human_status = self._human_stage_status(stage_view)
            attempt = self._selected_attempt_view(stage_view) or {}
            jobs = attempt.get("jobs") or []
            job_ids = ", ".join(str(job.get("job_id") or "") for job in jobs)
            self.execution_stage_status_labels[stage.value].setText(human_status)
            self.execution_stage_job_labels[stage.value].setText(job_ids or "—")
            status_parts.append(f"{stage.value}={human_status}")

        signature = f"{run_status}|{'|'.join(status_parts)}"
        if signature != self._last_persistent_status_signature:
            self._last_persistent_status_signature = signature
            self._append_log(
                f"[Run progress] {run_status}: " + ", ".join(status_parts) + "\n"
            )
        self._tail_persistent_progress_logs(state)
        legacy_running = (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        )
        persistent_active = self._persistent_state_has_active_work(state)
        self.force_stop.setEnabled(legacy_running or persistent_active)
        self._update_pipeline_run_buttons(persistent_active=persistent_active)
        self.move_outputs.setEnabled(not legacy_running and not persistent_active)
        self.browse_move_storage.setEnabled(not legacy_running and not persistent_active)

    def _browse_chanmap(self) -> None:
        path = self._select_open_file(
            "Select chanMap.mat",
            self.local_root.text() or str(Path.cwd()),
            "MAT files (*.mat);;All files (*)",
        )
        if path:
            self._apply_chanmap_file_to_controls(Path(path))
            self._schedule_refresh()

    def _browse_sorting_folder(self) -> None:
        path = self._select_directory(
            "Select sorting folder",
            self.sorting_phy_folder.text() or self.local_root.text() or str(Path.cwd()),
        )
        if path:
            self.sorting_phy_folder.setText(path)

    def _browse_manual_sorting_folder(self) -> None:
        path = self._select_directory(
            "Select sorting folder",
            self.manual_sorting_folder.text() or self.local_root.text() or str(Path.cwd()),
        )
        if path:
            self.manual_sorting_folder.setText(path)

    def _browse_sorting_search_root(self) -> None:
        path = self._select_directory(
            "Select sorting search root",
            self.sorting_search_root.text() or self.local_root.text() or str(Path.cwd()),
        )
        if path:
            self.sorting_search_root.setText(path)

    def _browse_behavior_dlc_batch(self) -> None:
        path = self._select_open_file(
            "Select DLC batch/script",
            self.behavior_dlc_batch_path.text() or self.basepath.text() or str(Path.cwd()),
            "Scripts (*.bat *.cmd *.sh *.py);;All files (*)",
        )
        if path:
            self.behavior_dlc_batch_path.setText(path)

    def _behavior_paths(self) -> tuple[PipelineGuiSettings, Path, str, Path]:
        settings = self._collect_settings()
        basepath = settings.basepath_path
        if basepath is None:
            raise ValueError("basepath is required.")
        basename = settings.basename
        if not basename:
            raise ValueError("basename cannot be resolved.")
        local_output = settings.local_output_dir
        if local_output is None:
            raise ValueError("local output directory cannot be resolved.")
        return settings, basepath, basename, local_output

    def _discover_behavior_dlc(self) -> None:
        try:
            settings, basepath, basename, local_output = self._behavior_paths()
            files = discover_dlc_files(basepath, output_dir=local_output, basename=basename)
            if not files:
                text = "No DLC files found."
            else:
                lines = [f"Found {len(files)} DLC file(s):"]
                for item in files:
                    video = item.video_path.name if item.video_path is not None else "no video"
                    lines.append(f"- {item.folder_name}: {item.path.name} ({video})")
                text = "\n".join(lines)
            self.behavior_dlc_summary.setPlainText(text)
            self._populate_behavior_primary_points(files)
            self._populate_behavior_frame_tabs(files)
            self._append_log(text + "\n")
            if files:
                try:
                    sync_warnings = inspect_dlc_ttl_sync(
                        basepath,
                        output_dir=local_output,
                        basename=basename,
                        dlc_files=files,
                        pulses_delta_range=settings.behavior.pulses_delta_range,
                        fallback_video_fps=settings.behavior.fallback_video_fps,
                    )
                    self._append_behavior_warnings("Behavior sync warnings", sync_warnings)
                except Exception as sync_exc:
                    self._append_behavior_warnings("Behavior sync warnings", [f"Sync check skipped: {sync_exc}"])
        except Exception as exc:
            QMessageBox.critical(self, "DLC discovery failed", str(exc))

    def _reset_behavior_discovery_state(self, *_args: Any) -> None:
        if self._refresh_suspended or not hasattr(self, "behavior_frame_tabs"):
            return
        self._behavior_dlc_files = []
        self._behavior_frame_canvases = {}
        self._behavior_pixel_distances_by_folder = {}
        self._behavior_pixel_to_cm_ratios_by_folder = {}
        self._behavior_clean_mask = None
        self._reported_behavior_warnings = set()
        while self.behavior_frame_tabs.count():
            widget = self.behavior_frame_tabs.widget(0)
            self.behavior_frame_tabs.removeTab(0)
            widget.deleteLater()
        self._clear_behavior_outlier_tabs()
        self.behavior_mode_tabs.setCurrentIndex(0)
        self.behavior_preview_status.setText("Discover DLC files to load epoch frames")
        self.behavior_pixel_distance.setValue(0.0)
        self.behavior_primary_point.blockSignals(True)
        try:
            self.behavior_primary_point.clear()
            self.behavior_primary_point.addItem("Discover DLC files first", "")
        finally:
            self.behavior_primary_point.blockSignals(False)

    def _load_behavior_calibration_frame(self) -> None:
        self._discover_behavior_dlc()

    @staticmethod
    def _short_dlc_point_label(name: str) -> str:
        return str(name).strip()

    def _clear_behavior_outlier_tabs(
        self,
        message: str = "Load outlier cleanup preview",
        *,
        add_placeholder: bool = True,
        preserve_clean_mask: bool = False,
    ) -> None:
        if not hasattr(self, "behavior_outlier_tabs"):
            return
        while self.behavior_outlier_tabs.count():
            widget = self.behavior_outlier_tabs.widget(0)
            self.behavior_outlier_tabs.removeTab(0)
            widget.deleteLater()
        self._behavior_outlier_canvases = {}
        self.behavior_track_canvas = BehaviorTrackCanvas(lambda _mask: None)
        if add_placeholder:
            self.behavior_track_canvas.show_message(message)
            self.behavior_outlier_tabs.addTab(self.behavior_track_canvas, "Track")
        if not preserve_clean_mask:
            self._behavior_clean_mask = None
        self._behavior_outlier_processed_preview = False

    def _populate_behavior_track_tabs_from_result(self, result: Any, *, processed: bool) -> None:
        behavior = result.behavior
        timestamps = np.asarray(behavior["timestamps"], dtype=np.float64).reshape(-1)
        x = np.asarray(behavior["position"]["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(behavior["position"]["y"], dtype=np.float64).reshape(-1)
        sub_mask = result.sub_session_mask
        self._clear_behavior_outlier_tabs(
            message="No behavior track loaded",
            add_placeholder=False,
            preserve_clean_mask=processed,
        )
        if sub_mask is None or np.asarray(sub_mask).size != timestamps.size:
            indices = np.arange(timestamps.size, dtype=np.int64)
            canvas = BehaviorTrackCanvas(
                (lambda _mask: None)
                if processed
                else (lambda _mask: self._update_behavior_clean_mask_from_outlier_tabs())
            )
            if processed:
                canvas.set_processed_track(timestamps, x, y)
            else:
                canvas.set_track(timestamps, x, y)
                self._behavior_outlier_canvases = {"Track": (canvas, indices)}
            self.behavior_outlier_tabs.addTab(canvas, "Track")
        else:
            sub_mask = np.asarray(sub_mask, dtype=np.int32).reshape(-1)
            for idx, item in enumerate(result.dlc_files, start=1):
                indices = np.flatnonzero(sub_mask == idx).astype(np.int64)
                if indices.size == 0:
                    continue
                canvas = BehaviorTrackCanvas(
                    (lambda _mask: None)
                    if processed
                    else (lambda _mask: self._update_behavior_clean_mask_from_outlier_tabs())
                )
                if processed:
                    canvas.set_processed_track(timestamps[indices], x[indices], y[indices])
                else:
                    canvas.set_track(timestamps[indices], x[indices], y[indices])
                    self._behavior_outlier_canvases[item.folder_name] = (canvas, indices)
                self.behavior_outlier_tabs.addTab(canvas, item.folder_name)
        if not processed:
            self._update_behavior_clean_mask_from_outlier_tabs()
        self._behavior_outlier_processed_preview = processed

    def _populate_behavior_primary_points(self, files: list[Any]) -> None:
        previous = str(self.behavior_primary_point.currentData() or "")
        point_sets: list[list[str]] = []
        errors: list[str] = []
        for item in files:
            try:
                point_sets.append(dlc_point_names(load_dlc_tracking(item.path)))
            except Exception as exc:
                errors.append(f"{item.folder_name}: {exc}")
        all_points: list[str] = []
        if point_sets:
            common = set(point_sets[0])
            for names in point_sets[1:]:
                common &= set(names)
            source_names = [name for name in point_sets[0] if name in common] if common else []
            if not source_names:
                seen: set[str] = set()
                source_names = []
                for names in point_sets:
                    for name in names:
                        if name not in seen:
                            seen.add(name)
                            source_names.append(name)
            all_points = source_names

        self.behavior_primary_point.blockSignals(True)
        try:
            self.behavior_primary_point.clear()
            self.behavior_primary_point.addItem("Select tracking point", "")
            if all_points:
                label_counts: dict[str, int] = {}
                for name in all_points:
                    label = self._short_dlc_point_label(name)
                    label_counts[label] = label_counts.get(label, 0) + 1
                    display = label if label_counts[label] == 1 else f"{label} ({label_counts[label]})"
                    self.behavior_primary_point.addItem(display, name)
                idx = self.behavior_primary_point.findData(previous)
                if idx < 0:
                    idx = 0
            else:
                idx = 0
            self.behavior_primary_point.setCurrentIndex(max(0, idx))
        finally:
            self.behavior_primary_point.blockSignals(False)
        if errors:
            self._append_warning_log("DLC point discovery warnings:\n" + "\n".join(f"- {item}" for item in errors) + "\n")

    def _populate_behavior_frame_tabs(self, files: list[Any]) -> None:
        while self.behavior_frame_tabs.count():
            widget = self.behavior_frame_tabs.widget(0)
            self.behavior_frame_tabs.removeTab(0)
            widget.deleteLater()
        self._behavior_dlc_files = list(files)
        self._behavior_frame_canvases = {}
        self._behavior_pixel_distances_by_folder = {}
        self._behavior_pixel_to_cm_ratios_by_folder = {}
        self._behavior_clean_mask = None
        self._clear_behavior_outlier_tabs()
        self.behavior_mode_tabs.setCurrentIndex(0)
        if not files:
            self.behavior_preview_status.setText("No DLC files found")
            return

        frame_errors: list[str] = []
        for item in files:
            canvas = BehaviorFrameCanvas(item.folder_name, self._on_behavior_calibration_line_changed)
            self._behavior_frame_canvases[item.folder_name] = canvas
            if item.video_path is None:
                canvas.show_message("No video found beside DLC file")
            else:
                try:
                    frame = load_representative_frame(item.video_path)
                    canvas.show_frame(frame, title=f"{item.folder_name}: {item.video_path.name}")
                except Exception as exc:
                    frame_errors.append(f"{item.folder_name}: {exc}")
                    canvas.show_message(f"Could not load first frame:\n{exc}")
            self.behavior_frame_tabs.addTab(canvas, item.folder_name)
        self._on_behavior_calibration_line_changed()
        if frame_errors:
            self._append_warning_log("Behavior frame preview warnings:\n" + "\n".join(f"- {item}" for item in frame_errors) + "\n")

    def _ensure_behavior_dlc_files_loaded(self) -> list[Any]:
        if self._behavior_dlc_files:
            return self._behavior_dlc_files
        settings, basepath, basename, local_output = self._behavior_paths()
        files = discover_dlc_files(basepath, output_dir=local_output, basename=basename)
        self._populate_behavior_frame_tabs(files)
        return files

    def _on_behavior_calibration_line_changed(self) -> None:
        distances: dict[str, float] = {}
        for folder, canvas in getattr(self, "_behavior_frame_canvases", {}).items():
            if canvas.pixel_distance is not None and canvas.pixel_distance > 0:
                distances[folder] = float(canvas.pixel_distance)
        self._behavior_pixel_distances_by_folder = distances
        self._behavior_pixel_to_cm_ratios_by_folder = {
            folder: ratio
            for folder, ratio in self._behavior_pixel_to_cm_ratios_by_folder.items()
            if folder in distances
        }
        if len(distances) == 1:
            self.behavior_pixel_distance.setValue(next(iter(distances.values())))
        elif not distances:
            self.behavior_pixel_distance.setValue(0.0)
        total = len(getattr(self, "_behavior_dlc_files", []))
        self.behavior_preview_status.setText(
            f"Calibration lines: {len(distances)}/{total}. "
            "Click two endpoints in each epoch tab, then run calibration."
        )
        self._schedule_refresh()

    def _invalidate_behavior_calibration(self) -> None:
        if self._refresh_suspended:
            return
        if self._behavior_pixel_to_cm_ratios_by_folder:
            self._behavior_pixel_to_cm_ratios_by_folder = {}
            self.behavior_preview_status.setText("Known distance changed. Run calibration again.")
        self._schedule_refresh()

    def _reset_behavior_calibration_viewer(self) -> None:
        widget = self.behavior_frame_tabs.currentWidget()
        if isinstance(widget, BehaviorFrameCanvas):
            self._behavior_pixel_distances_by_folder.pop(widget.epoch_name, None)
            self._behavior_pixel_to_cm_ratios_by_folder.pop(widget.epoch_name, None)
            widget.reset_line()
        self._on_behavior_calibration_line_changed()

    def _run_behavior_calibration(self) -> None:
        try:
            files = self._ensure_behavior_dlc_files_loaded()
            if not files:
                raise FileNotFoundError("No DLC files found.")
            current = self.behavior_frame_tabs.currentWidget()
            if not isinstance(current, BehaviorFrameCanvas):
                raise ValueError("Select a calibration epoch tab first.")
            folder = current.epoch_name
            if current.pixel_distance is None or current.pixel_distance <= 0:
                raise ValueError("Click two calibration endpoints in the current epoch tab first.")
            known_cm = float(self.behavior_distance_cm.value())
            if known_cm <= 0:
                raise ValueError("known distance cm must be positive.")
            self._behavior_pixel_distances_by_folder[folder] = float(current.pixel_distance)
            self._behavior_pixel_to_cm_ratios_by_folder[folder] = float(current.pixel_distance) / known_cm
            ratios = self._behavior_pixel_to_cm_ratios_by_folder
            self.behavior_pixel_distance.setValue(float(current.pixel_distance))
            missing = [item.folder_name for item in files if item.folder_name not in ratios]
            lines = [f"Behavior calibration updated: {folder}: {ratios[folder]:.6g} pixel/cm"]
            if missing:
                lines.append("Remaining epochs: " + ", ".join(missing))
            else:
                lines.append("All epochs calibrated.")
            text = "\n".join(lines)
            self.behavior_preview_status.setText(text)
            self._append_log(text + "\n")
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Behavior calibration failed", str(exc))

    def _require_behavior_calibration(self) -> dict[str, float]:
        files = self._ensure_behavior_dlc_files_loaded()
        if not files:
            raise FileNotFoundError("No DLC files found.")
        if not self._behavior_pixel_to_cm_ratios_by_folder:
            raise ValueError("Click two calibration endpoints and press Run calibration before this step.")
        missing = [
            item.folder_name
            for item in files
            if item.folder_name not in self._behavior_pixel_to_cm_ratios_by_folder
        ]
        if missing:
            raise ValueError("Calibration is missing for DLC epoch(s): " + ", ".join(missing))
        return dict(self._behavior_pixel_to_cm_ratios_by_folder)

    def _require_behavior_primary_point(self) -> str:
        point = str(self.behavior_primary_point.currentData() or "").strip()
        if not point:
            raise ValueError("Select a DLC tracking point before running behavior processing.")
        return point

    def _set_behavior_clean_mask(self, mask: np.ndarray | None) -> None:
        self._behavior_clean_mask = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)

    def _reset_behavior_keep_ranges(self) -> None:
        if getattr(self, "_behavior_outlier_processed_preview", False):
            self._behavior_clean_mask = None
            self._load_behavior_outlier_preview(show_errors=False)
            self._append_log("Reset outlier cleanup preview to raw track.\n")
            return
        widget = self.behavior_outlier_tabs.currentWidget()
        if isinstance(widget, BehaviorTrackCanvas):
            widget.reset_keep_ranges()

    def _update_behavior_clean_mask_from_outlier_tabs(self) -> None:
        if not self._behavior_outlier_canvases:
            self._behavior_clean_mask = None
            return
        nonempty = [
            indices
            for _canvas, indices in self._behavior_outlier_canvases.values()
            if indices.size
        ]
        if not nonempty:
            self._behavior_clean_mask = None
            return
        max_index = max(
            int(indices.max())
            for indices in nonempty
        )
        mask = np.ones(max_index + 1, dtype=bool)
        any_rejected = False
        for canvas, indices in self._behavior_outlier_canvases.values():
            if indices.size != canvas.good_mask.size:
                continue
            mask[indices] = canvas.good_mask
            any_rejected = any_rejected or bool(np.any(~canvas.good_mask))
        self._behavior_clean_mask = mask if any_rejected else None

    def _on_behavior_mode_tab_changed(self, index: int) -> None:
        if index == 1:
            self._load_behavior_outlier_preview(show_errors=False)

    def _on_behavior_primary_point_changed(self, *_args: Any) -> None:
        if self._refresh_suspended or not hasattr(self, "behavior_mode_tabs"):
            return
        self._behavior_clean_mask = None
        if self.behavior_mode_tabs.currentIndex() != 1:
            return
        if not str(self.behavior_primary_point.currentData() or "").strip():
            self._clear_behavior_outlier_tabs("Select a tracking point to load the behavior track.")
            return
        if not self._behavior_dlc_files:
            return
        calibrated = {
            item.folder_name
            for item in self._behavior_dlc_files
            if item.folder_name in self._behavior_pixel_to_cm_ratios_by_folder
        }
        if len(calibrated) != len(self._behavior_dlc_files):
            return
        self._load_behavior_outlier_preview(show_errors=False)

    def _load_behavior_outlier_preview(self, *, show_errors: bool) -> bool:
        try:
            settings, basepath, basename, local_output = self._behavior_paths()
            b = settings.behavior
            ratios = self._require_behavior_calibration()
            primary_point = self._require_behavior_primary_point()
            preview = process_dlc_behavior(
                basepath=basepath,
                output_dir=local_output,
                basename=basename,
                primary_coords=b.primary_coords,
                primary_point=primary_point,
                likelihood=b.likelihood,
                pulses_delta_range=b.pulses_delta_range,
                calibration_distance_cm=b.calibration_distance_cm,
                pixel_to_cm_ratios_by_folder=ratios,
                interpolate_gap_sec=0.0,
                fallback_video_fps=b.fallback_video_fps,
                overwrite=True,
                save_mat=False,
            )
            self._populate_behavior_track_tabs_from_result(preview, processed=False)
            self.behavior_mode_tabs.setCurrentIndex(1)
            self._append_log(f"Loaded behavior 2D track map: {primary_point}\n")
            return True
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self, "Behavior outlier cleanup failed", str(exc))
            else:
                self._clear_behavior_outlier_tabs(str(exc))
            return False

    def _run_behavior_outlier_cleanup(self) -> None:
        try:
            if self._behavior_outlier_canvases:
                self._update_behavior_clean_mask_from_outlier_tabs()
            clean_mask = None if self._behavior_clean_mask is None else self._behavior_clean_mask.copy()
            settings, basepath, basename, local_output = self._behavior_paths()
            b = settings.behavior
            ratios = self._require_behavior_calibration()
            primary_point = self._require_behavior_primary_point()
            result = process_dlc_behavior(
                basepath=basepath,
                output_dir=local_output,
                basename=basename,
                primary_coords=b.primary_coords,
                primary_point=primary_point,
                likelihood=b.likelihood,
                pulses_delta_range=b.pulses_delta_range,
                calibration_distance_cm=b.calibration_distance_cm,
                pixel_to_cm_ratios_by_folder=ratios,
                interpolate_gap_sec=b.interpolate_gap_sec,
                clean_mask=clean_mask,
                fallback_video_fps=b.fallback_video_fps,
                overwrite=True,
                save_mat=False,
            )
            self._behavior_clean_mask = clean_mask
            self._populate_behavior_track_tabs_from_result(result, processed=True)
            self.behavior_mode_tabs.setCurrentIndex(1)
            rejected_frames = int(np.sum(~clean_mask)) if clean_mask is not None else 0
            total_frames = int(clean_mask.size) if clean_mask is not None else 0
            mask_text = (
                f"applied, rejected frames={rejected_frames}/{total_frames}"
                if clean_mask is not None
                else "not applied"
            )
            note_items = result.behavior.get("notes", [])
            if isinstance(note_items, np.ndarray):
                note_lines = [str(item) for item in note_items.reshape(-1).tolist()]
            elif isinstance(note_items, (list, tuple)):
                note_lines = [str(item) for item in note_items]
            else:
                note_lines = [str(note_items)] if note_items else []
            interpolation_note = next(
                (item for item in reversed(note_lines) if item.startswith("interpolated_short_gaps:")),
                "",
            )
            self._append_log(
                f"Applied outlier cleanup & interpolation: {primary_point}, mask={mask_text}, "
                f"interpolate gaps <= {b.interpolate_gap_sec:g} sec"
                + (f", {interpolation_note}\n" if interpolation_note else "\n")
            )
        except Exception as exc:
            QMessageBox.critical(self, "Behavior outlier cleanup failed", str(exc))

    def _run_behavior_export(self) -> None:
        try:
            settings, basepath, basename, local_output = self._behavior_paths()
            b = settings.behavior
            output_path = local_output / f"{basename}.animal.behavior.mat"
            if output_path.exists() and not b.overwrite:
                raise FileExistsError(f"Behavior output already exists. Enable overwrite to replace it:\n{output_path}")
            ratios = self._require_behavior_calibration()
            primary_point = self._require_behavior_primary_point()

            batch_path = Path(b.dlc_batch_path).expanduser() if b.dlc_batch_path.strip() else None
            if batch_path is not None and batch_path.exists():
                self._append_log(f"Running DLC batch/script: {batch_path}\n")
                subprocess.run([str(batch_path)], cwd=str(basepath), check=True)

            result = process_dlc_behavior(
                basepath=basepath,
                output_dir=local_output,
                basename=basename,
                primary_coords=b.primary_coords,
                primary_point=primary_point,
                likelihood=b.likelihood,
                pulses_delta_range=b.pulses_delta_range,
                calibration_distance_cm=b.calibration_distance_cm,
                pixel_to_cm_ratios_by_folder=ratios,
                interpolate_gap_sec=b.interpolate_gap_sec,
                clean_mask=self._behavior_clean_mask,
                fallback_video_fps=b.fallback_video_fps,
                overwrite=True,
                save_mat=True,
            )
            lines = [
                "Behavior export finished",
                f"Output: {result.output_path}",
                f"DLC files: {len(result.dlc_files)}",
                f"pixel_to_cm_ratio: {result.pixel_to_cm_ratio}",
                f"outlier mask: {'applied' if self._behavior_clean_mask is not None else 'not applied'}",
            ]
            text = "\n".join(lines)
            self.behavior_dlc_summary.setPlainText(text)
            self._append_log(text + "\n")
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Behavior export failed", str(exc))

    def _load_config(self) -> None:
        path = self._select_open_file(
            "Load GUI config",
            str(CONFIG_DIR),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            config_path = Path(path)
            self._apply_config_settings(PipelineGuiSettings.load(config_path), config_path)
            self._append_log(f"Loaded config: {path}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Load config failed", str(exc))

    def _save_config(self) -> None:
        path = self._select_save_file(
            "Save GUI config",
            str(Path.cwd() / "preprocess_gui_config.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self._collect_settings().save(Path(path))
            self._append_log(f"Saved config: {path}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Save config failed", str(exc))

    def _load_default_config(self) -> None:
        try:
            current = self._collect_settings()
            loaded = _load_default_settings()
            loaded.basepath = current.basepath
            loaded.local_root = current.local_root
            loaded.xml_path = current.xml_path
            loaded.multi_day_enabled = current.multi_day_enabled
            loaded.multi_day_session_paths = list(current.multi_day_session_paths)
            loaded.multi_day_selected_subepoch_paths = list(current.multi_day_selected_subepoch_paths)
            loaded.multi_day_name = current.multi_day_name
            loaded.subsession_order = list(current.subsession_order)
            loaded.chanmap_path = current.chanmap_path
            loaded.postprocess.sorting_phy_folder = current.postprocess.sorting_phy_folder
            loaded.postprocess.sorting_search_root = current.postprocess.sorting_search_root
            loaded.postprocess.cell_explorer_sorting_folders = list(
                current.postprocess.cell_explorer_sorting_folders
            )
            self._apply_config_settings(loaded, DEFAULT_CONFIG_PATH)
            self._append_log(f"Loaded default config: {DEFAULT_CONFIG_PATH}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Load default config failed", str(exc))

    def _save_default_config(self) -> None:
        try:
            saved = _save_default_settings(self._collect_settings())
            self._append_log(f"Saved default config: {saved}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Save default config failed", str(exc))

    def _sorter_changed(self, sorter: str) -> None:
        path, config = SORTER_DEFAULTS.get(sorter, ("", ""))
        self.sorter_path.setText(path)
        self.sorter_config_path.setText(config)
        if sorter == "disabled":
            self.run_sorter.setChecked(False)
        elif not self.run_sorter.isChecked():
            self.run_sorter.setChecked(True)
        self._update_sorter_enabled()
        self._schedule_refresh()

    @staticmethod
    def _resolve_repo_path(text: str) -> Path:
        path = Path(text).expanduser()
        if not path.is_absolute():
            return resolve_project_path(path, root=REPO_ROOT)
        return path.resolve()

    def _current_sorter_config_path(self) -> Path | None:
        text = self.sorter_config_path.text().strip()
        if not text:
            _path, config = SORTER_DEFAULTS.get(self.sorter.currentText(), ("", ""))
            text = config
        return self._resolve_repo_path(text) if text else None

    def _current_sorter_defaults(self) -> tuple[str, str]:
        return SORTER_DEFAULTS.get(self.sorter.currentText(), ("", ""))

    def _open_sorter_config(self) -> None:
        path = self._current_sorter_config_path()
        if path is None:
            QMessageBox.information(self, "Open config", "No sorter config is selected.")
            return
        if not path.exists():
            QMessageBox.warning(self, "Open config", f"Sorter config not found:\n{path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.information(self, "Open config", f"Open this file manually:\n{path}")

    def _update_sorter_enabled(self) -> None:
        enabled = self.run_sorter.isChecked()
        for widget in [
            self.sorter,
            self.sorter_partition_mode,
            self.sorter_path,
            self.sorter_config_path,
            self.open_sorter_config,
            self.matlab_path,
        ]:
            widget.setEnabled(enabled)

    def _collect_settings(self) -> PipelineGuiSettings:
        self._normalize_worker_fields()
        noise_thresholds: dict[str, float] = {}
        for key, field in self.noise_threshold_fields.items():
            text = field.text().strip()
            if text:
                noise_thresholds[key] = float(text)
        preprocess = PreprocessGuiSettings(
            sorter_path=self.sorter_path.text().strip() or self._current_sorter_defaults()[0],
            sorter_config_path=self.sorter_config_path.text().strip() or self._current_sorter_defaults()[1],
            analog_inputs=self.analog_inputs.isChecked(),
            digital_inputs=self.digital_inputs.isChecked(),
            save_raw=self.save_raw.isChecked(),
            do_preprocess=self.do_preprocess.isChecked(),
            bandpass_min_hz=self.bandpass_min.value(),
            bandpass_max_hz=self.bandpass_max.value(),
            reference=self.reference.currentText(),
            local_radius_um=parse_float_pair(self.local_radius.text(), default=(20.0, 200.0)),
            make_lfp=self.make_lfp.isChecked(),
            lfp_fs=self.lfp_fs.value(),
            state_score=self.state_score.isChecked(),
            sw_channels=parse_int_list(self.sw_channels.text()),
            theta_channels=parse_int_list(self.theta_channels.text()),
            state_ignore_manual=self.state_ignore_manual.isChecked(),
            state_save_lfp_mat=self.state_save_lfp_mat.isChecked(),
            state_sticky_trigger=self.state_sticky_trigger.isChecked(),
            state_winparms=(self.state_window_sec.value(), self.state_smoothfact.value()),
            emg_th_alpha=self.emg_th_alpha.value(),
            useEMG_NREM=self.useEMG_NREM.isChecked(),
            state_min_state_length=self.state_min_state_length.value(),
            state_microarousal_sec=self.state_microarousal_sec.value(),
            state_block_wake_to_rem=self.state_block_wake_to_rem.isChecked(),
            remove_ttl_artifacts=self.remove_ttl_artifacts.isChecked(),
            artifact_ttl_group_mode=self.ttl_group.currentText(),
            artifact_ttl_channel=self.ttl_channel.value(),
            artifact_ttl_include_offset=self.ttl_include_offset.isChecked(),
            artifact_ttl_ms_before=self.ttl_before.value(),
            artifact_ttl_ms_after=self.ttl_after.value(),
            artifact_ttl_mode=self.ttl_mode.currentText(),
            remove_highamp_artifacts=self.remove_highamp_artifacts.isChecked(),
            artifact_highamp_group_mode=self.highamp_group.currentText(),
            highamp_threshold_sigma=self.highamp_sigma.value(),
            highamp_ms_before=self.highamp_before.value(),
            highamp_ms_after=self.highamp_after.value(),
            highamp_mode=self.highamp_mode.currentText(),
            reject_channels=parse_int_list(self.reject_channels.text()),
            probe_assignments=self._probe_rows_to_assignments(),
            run_sorter=self.run_sorter.isChecked(),
            sorter=self.sorter.currentText(),
            sorter_partition_mode=str(self.sorter_partition_mode.currentData() or "all"),
            matlab_path="",
            preprocess_worker_count=normalize_worker_count(self.preprocess_worker_count.value()),
            sorter_worker_count=normalize_worker_count(self.sorter_worker_count.value()),
            overwrite=self.pre_overwrite.isChecked(),
        )
        behavior = BehaviorGuiSettings(
            enabled=self.behavior_enabled.isChecked(),
            primary_coords=self.behavior_primary_coords.value(),
            primary_point=str(self.behavior_primary_point.currentData() or ""),
            likelihood=self.behavior_likelihood.value(),
            pulses_delta_range=self.behavior_ttl_tolerance.value(),
            calibration_distance_cm=self.behavior_distance_cm.value(),
            calibration_pixel_distance=self.behavior_pixel_distance.value(),
            interpolate_gap_sec=self.behavior_gap_sec.value(),
            fallback_video_fps=self.behavior_fallback_fps.value(),
            clean_tracker_jumps=self.behavior_clean_jumps.isChecked(),
            dlc_batch_path=self.behavior_dlc_batch_path.text().strip(),
            overwrite=self.behavior_overwrite.isChecked(),
        )
        postprocess = PostprocessGuiSettings(
            sorting_phy_folder=self.sorting_phy_folder.text().strip(),
            sorting_search_root=self.sorting_search_root.text().strip(),
            cell_explorer_sorting_folders=list(self._cell_explorer_sorting_folders),
            apply_preprocess=self.post_apply_preprocess.isChecked(),
            exclude_cluster_groups=[g.strip() for g in self.exclude_groups.text().split(",") if g.strip()],
            duplicate_censored_period_ms=self.duplicate_censored.value(),
            duplicate_threshold=self.duplicate_threshold.value(),
            merge_min_spikes=self.merge_min_spikes.value(),
            merge_corr_diff_thresh=self.merge_corr.value(),
            merge_template_diff_thresh=self.merge_template.value(),
            split_contamination=self.split_contamination.value(),
            split_threshold_mode=self.split_threshold_mode.currentText(),
            split_wf_threshold=self.split_wf_threshold.value(),
            split_wf_n_chans=self.split_wf_n_chans.value(),
            split_amp_mad_scale=self.split_amp_mad_scale.value(),
            skip_pc_metrics=self.skip_pc_metrics.isChecked(),
            noise_label_only=self.noise_label_only.isChecked(),
            noise_thresholds=noise_thresholds,
            overwrite=self.post_overwrite.isChecked(),
            worker_count=normalize_worker_count(self.post_worker_count.value()),
        )
        basepath_text = self.basepath.text().strip()
        basename = (
            self.multi_day_name.text().strip()
            if self._multi_day_session_paths and self.multi_day_name.text().strip()
            else (Path(basepath_text).name if basepath_text else "")
        )
        local_root_text = self.local_root.text().strip() or str(REPO_ROOT / "preprocess_tmp")
        execution = ExecutionGuiSettings(
            requested_backend=str(
                self.execution_backend.currentData() or RequestedBackend.AUTO.value
            ),
            workspace=str(Path(local_root_text).expanduser().resolve()),
            matlab_path="",
            shared_workspace_acknowledged=False,
            require_sacct=True,
            preprocess=self._resource_settings_from_widgets(StageName.PREPROCESS),
            sorting=self._resource_settings_from_widgets(StageName.SORTING),
            postprocess=self._resource_settings_from_widgets(StageName.POSTPROCESS),
        )
        settings = PipelineGuiSettings(
            basepath=self.basepath.text().strip(),
            local_root=self.local_root.text().strip(),
            xml_path=self.xml_path.text().strip(),
            chanmap_path=self.chanmap_path.text().strip(),
            multi_day_enabled=bool(self._multi_day_session_paths),
            multi_day_session_paths=list(self._multi_day_session_paths),
            multi_day_selected_subepoch_paths=list(self._multi_day_selected_subepoch_paths),
            multi_day_name=self.multi_day_name.text().strip(),
            subsession_order=(
                [] if self._multi_day_session_paths else list(self._subsession_order)
            ),
            preprocess=preprocess,
            behavior=behavior,
            postprocess=postprocess,
            execution=execution,
        )
        resolve_existing_session_settings(settings)
        return settings

    def _apply_config_settings(
        self, settings: PipelineGuiSettings, config_path: Path
    ) -> None:
        self._environment_backend_default_pending = not _default_config_has_backend_choice(
            config_path
        )
        self._apply_settings(settings)
        if self._slurm_capabilities is not None:
            self._apply_environment_backend_default(self._slurm_capabilities)

    def _apply_settings(
        self,
        settings: PipelineGuiSettings,
        *,
        preserve_missing_paths: bool = False,
    ) -> None:
        self._move_storage_override = None
        self._move_source_snapshot = None
        self._refresh_suspended = True
        try:
            self.basepath.setText(settings.basepath)
            self.local_root.setText(settings.local_root or str(settings.local_root_path))
            xml_text = settings.xml_path.strip()
            if (
                xml_text
                and not preserve_missing_paths
                and not Path(xml_text).expanduser().exists()
            ):
                xml_text = ""
            self.xml_path.setText(xml_text)
            self._set_multi_day_session_paths(
                list(settings.multi_day_session_paths),
                selected_subepoch_paths=list(settings.multi_day_selected_subepoch_paths),
            )
            self.multi_day_name.setText(settings.multi_day_name)
            self._set_subsession_order(list(settings.subsession_order))
            self.chanmap_path.setText(settings.chanmap_path)
            p = settings.preprocess
            self.analog_inputs.setChecked(p.analog_inputs)
            self.digital_inputs.setChecked(p.digital_inputs)
            self.save_raw.setChecked(p.save_raw)
            self.do_preprocess.setChecked(p.do_preprocess)
            self.bandpass_min.setValue(p.bandpass_min_hz)
            self.bandpass_max.setValue(p.bandpass_max_hz)
            self.reference.setCurrentText(p.reference)
            self.local_radius.setText(f"{p.local_radius_um[0]}, {p.local_radius_um[1]}")
            self.make_lfp.setChecked(p.make_lfp)
            self.lfp_fs.setValue(p.lfp_fs)
            self.state_score.setChecked(p.state_score)
            self.sw_channels.setText(", ".join(str(v) for v in p.sw_channels))
            self.theta_channels.setText(", ".join(str(v) for v in p.theta_channels))
            self.state_ignore_manual.setChecked(p.state_ignore_manual)
            self.state_save_lfp_mat.setChecked(p.state_save_lfp_mat)
            self.state_sticky_trigger.setChecked(p.state_sticky_trigger)
            self.state_window_sec.setValue(p.state_winparms[0])
            self.state_smoothfact.setValue(p.state_winparms[1])
            self.emg_th_alpha.setValue(p.emg_th_alpha)
            self.useEMG_NREM.setChecked(p.useEMG_NREM)
            self.state_min_state_length.setValue(p.state_min_state_length)
            self.state_microarousal_sec.setValue(p.state_microarousal_sec)
            self.state_block_wake_to_rem.setChecked(p.state_block_wake_to_rem)
            self.remove_ttl_artifacts.setChecked(p.remove_ttl_artifacts and p.artifact_ttl_group_mode != "none")
            self.ttl_group.setCurrentText("all" if p.artifact_ttl_group_mode == "none" else p.artifact_ttl_group_mode)
            self.ttl_channel.setValue(p.artifact_ttl_channel)
            self.ttl_include_offset.setChecked(p.artifact_ttl_include_offset)
            self.ttl_before.setValue(p.artifact_ttl_ms_before)
            self.ttl_after.setValue(p.artifact_ttl_ms_after)
            self.ttl_mode.setCurrentText(p.artifact_ttl_mode)
            self.remove_highamp_artifacts.setChecked(
                p.remove_highamp_artifacts and p.artifact_highamp_group_mode != "none"
            )
            self.highamp_group.setCurrentText(
                "shank" if p.artifact_highamp_group_mode == "none" else p.artifact_highamp_group_mode
            )
            self.highamp_sigma.setValue(p.highamp_threshold_sigma)
            self.highamp_before.setValue(p.highamp_ms_before)
            self.highamp_after.setValue(p.highamp_ms_after)
            self.highamp_mode.setCurrentText(p.highamp_mode)
            self.reject_channels.setText(", ".join(str(v) for v in p.reject_channels))
            self._render_probe_assignments(p.probe_assignments)
            self._chanmap_controls_dirty = False
            self.run_sorter.setChecked(p.run_sorter and (p.sorter or "disabled") != "disabled")
            self.sorter.setCurrentText(p.sorter or "disabled")
            partition_index = self.sorter_partition_mode.findData(getattr(p, "sorter_partition_mode", "all"))
            self.sorter_partition_mode.setCurrentIndex(partition_index if partition_index >= 0 else 0)
            default_sorter_path, default_sorter_config = self._current_sorter_defaults()
            self.sorter_path.setText(p.sorter_path or default_sorter_path)
            self.sorter_config_path.setText(p.sorter_config_path or default_sorter_config)
            self.matlab_path.setText("")
            self.preprocess_worker_count.setValue(normalize_worker_count(p.preprocess_worker_count))
            self.sorter_worker_count.setValue(normalize_worker_count(p.sorter_worker_count))
            self.pre_overwrite.setChecked(p.overwrite)
            self._update_sorter_enabled()
            b = settings.behavior
            self.behavior_enabled.setChecked(b.enabled)
            self.behavior_overwrite.setChecked(b.overwrite)
            self.behavior_clean_jumps.setChecked(b.clean_tracker_jumps)
            self.behavior_dlc_batch_path.setText(b.dlc_batch_path)
            if b.primary_point:
                idx = self.behavior_primary_point.findData(b.primary_point)
                if idx < 0:
                    self.behavior_primary_point.addItem(b.primary_point, b.primary_point)
                    idx = self.behavior_primary_point.findData(b.primary_point)
                self.behavior_primary_point.setCurrentIndex(idx)
            else:
                self.behavior_primary_point.setCurrentIndex(0)
            self.behavior_primary_coords.setValue(b.primary_coords)
            self.behavior_likelihood.setValue(b.likelihood)
            self.behavior_ttl_tolerance.setValue(b.pulses_delta_range)
            self.behavior_fallback_fps.setValue(b.fallback_video_fps)
            self.behavior_distance_cm.setValue(b.calibration_distance_cm)
            self.behavior_pixel_distance.setValue(b.calibration_pixel_distance)
            self.behavior_gap_sec.setValue(b.interpolate_gap_sec)
            pp = settings.postprocess
            self.sorting_phy_folder.setText(pp.sorting_phy_folder)
            self.sorting_search_root.setText(pp.sorting_search_root)
            self._set_cell_explorer_sorting_folders(list(pp.cell_explorer_sorting_folders))
            self.post_apply_preprocess.setChecked(pp.apply_preprocess)
            self.exclude_groups.setText(", ".join(pp.exclude_cluster_groups))
            self.duplicate_censored.setValue(pp.duplicate_censored_period_ms)
            self.duplicate_threshold.setValue(pp.duplicate_threshold)
            self.merge_min_spikes.setValue(pp.merge_min_spikes)
            self.merge_corr.setValue(pp.merge_corr_diff_thresh)
            self.merge_template.setValue(pp.merge_template_diff_thresh)
            self.split_contamination.setValue(pp.split_contamination)
            self.split_threshold_mode.setCurrentText(pp.split_threshold_mode)
            self.split_wf_threshold.setValue(pp.split_wf_threshold)
            self.split_wf_n_chans.setValue(pp.split_wf_n_chans)
            self.split_amp_mad_scale.setValue(pp.split_amp_mad_scale)
            self.skip_pc_metrics.setChecked(pp.skip_pc_metrics)
            self.noise_label_only.setChecked(pp.noise_label_only)
            for key, field in self.noise_threshold_fields.items():
                value = pp.noise_thresholds.get(key)
                field.setText("" if value is None else str(value))
            self.post_overwrite.setChecked(pp.overwrite)
            self.post_worker_count.setValue(normalize_worker_count(pp.worker_count))
            execution = settings.execution
            backend_index = self.execution_backend.findData(execution.requested_backend)
            self.execution_backend.setCurrentIndex(backend_index if backend_index >= 0 else 0)
            self._apply_stage_resource_settings(StageName.PREPROCESS, execution.preprocess)
            self._apply_stage_resource_settings(StageName.SORTING, execution.sorting)
            self._apply_stage_resource_settings(StageName.POSTPROCESS, execution.postprocess)
            chanmap = settings.resolved_chanmap_path()
            if not self._load_settings_chanmap_preview(settings) and chanmap is not None and chanmap.exists():
                self._load_chanmap_preview(chanmap)
        finally:
            self._normalize_worker_fields()
            self._refresh_suspended = False

    def _apply_stage_resource_settings(
        self, stage: StageName, resource: StageResourceGuiSettings
    ) -> None:
        widgets = self._resource_widgets[stage.value]
        widgets["cpus"].setValue(max(1, int(resource.cpus)))  # type: ignore[attr-defined]
        widgets["memory_gib"].setValue(max(1, int(resource.memory_mb) // 1024))  # type: ignore[attr-defined]

    def _normalize_worker_fields(self) -> None:
        for field in (self.preprocess_worker_count, self.sorter_worker_count, self.post_worker_count):
            field.setMaximum(default_worker_count())
            normalized = normalize_worker_count(field.value())
            if field.value() != normalized:
                field.setValue(normalized)

    def _refresh_preview(self) -> None:
        try:
            settings = self._collect_settings()
            current_tab = self.tabs.currentIndex()
            if hasattr(self, "monitor_stack"):
                self.monitor_stack.setCurrentIndex(1 if current_tab == 1 else 0)
            ephys_tab = self.ephys_tabs.currentIndex() if hasattr(self, "ephys_tabs") else 0
            mode: RunMode = "postprocess" if current_tab == 0 and ephys_tab == 1 else "preprocess"
            try:
                checks = [] if current_tab == 1 else run_preflight(settings, mode)
            except Exception as exc:
                checks = [CheckResult("Preflight", "warn", str(exc))]
            behavior_output = (
                settings.local_output_dir / (settings.basename + ".animal.behavior.mat")
                if settings.local_output_dir and settings.basename
                else None
            )
            behavior_ready = (
                bool(self._behavior_pixel_to_cm_ratios_by_folder)
                and len(self._behavior_pixel_to_cm_ratios_by_folder) == len(self._behavior_dlc_files)
            )
            lines = [
                f"Basepath: {settings.basepath or '-'}",
                f"Basename: {settings.basename or '-'}",
                f"Local output: {settings.local_output_dir or '-'}",
                f"chanMap: {settings.resolved_chanmap_path() or '-'}",
                f"Behavior output: {behavior_output or '-'}",
                f"Active workflow: {'Behavior' if current_tab == 1 else 'Ephys ' + mode}",
            ]
            if settings.multi_day_enabled:
                selected_count = len(settings.multi_day_selected_subepoch_paths)
                lines.insert(
                    2,
                    f"Multi-day subepochs: {selected_count} selected"
                    if selected_count
                    else "Multi-day subepochs: all discovered",
                )
            if current_tab == 1:
                lines.extend(
                    [
                        "",
                        "Behavior:",
                        f"[{'OK' if settings.basepath_path else 'ERROR'}] Basepath: {'set' if settings.basepath_path else 'not set'}",
                        f"[{'OK' if settings.local_output_dir else 'ERROR'}] Local output: {settings.local_output_dir or 'not set'}",
                        f"[{'OK' if behavior_ready else 'ERROR'}] Calibration: {'ready' if behavior_ready else 'click endpoints for all epochs and run calibration'}",
                        f"[OK] Export target: local output only",
                    ]
                )
            else:
                lines.extend(["", "Ephys preflight:"])
                lines.extend(self._format_checks(checks))
            self._settings_preview_text = "\n".join(lines)
            chanmap = settings.resolved_chanmap_path()
            explicit_chanmap = Path(settings.chanmap_path).expanduser() if settings.chanmap_path.strip() else None
            if (
                explicit_chanmap is not None
                and explicit_chanmap.exists()
                and not self._chanmap_controls_dirty
            ):
                self._load_chanmap_preview(explicit_chanmap)
            elif not self._load_settings_chanmap_preview(settings):
                if not self._chanmap_controls_dirty and chanmap is not None and chanmap.exists():
                    self._load_chanmap_preview(chanmap)
        except Exception as exc:
            self._settings_preview_text = f"Config error:\n{exc}"

    def _format_checks(self, checks: list[CheckResult]) -> list[str]:
        prefix = {"ok": "[OK]", "warn": "[WARN]", "error": "[ERROR]"}
        return [f"{prefix.get(c.status, '[?]')} {c.label}: {c.detail}" for c in checks]

    def _generate_chanmap(self) -> None:
        try:
            settings = self._collect_settings()
            if settings.basepath_path is None:
                raise ValueError("basepath is required.")
            xml_path = settings.resolved_xml_path()
            if xml_path is None or not xml_path.exists():
                self._warn_missing_xml_for_settings(settings, context="generate-chanmap")
                return
            basepath, basename, local_output_dir, _xml_path = select_paths_with_gui(
                use_gui=False,
                manual_basepath=settings.basepath_path,
                local_root=settings.local_root_path,
                manual_xml_path=xml_path,
            )
            chanmap_path, bad_channels = prepare_chanmap(
                basepath=basepath,
                basename=basename,
                local_output_dir=local_output_dir,
                probe_assignments=settings.preprocess.probe_assignments,
                reject_channels=settings.preprocess.reject_channels,
                xml_path=_xml_path,
            )
            self.chanmap_path.setText(str(chanmap_path))
            self._refresh_suspended = True
            try:
                self.reject_channels.setText(", ".join(str(v) for v in bad_channels))
            finally:
                self._refresh_suspended = False
            self._load_chanmap_preview(chanmap_path)
            self._append_log(f"Generated chanMap: {chanmap_path}\nBad channels: {bad_channels}\n")
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Generate chanMap failed", str(exc))

    def _ensure_current_chanmap_for_run(self, settings: PipelineGuiSettings) -> PipelineGuiSettings:
        if settings.basepath_path is None:
            return settings
        xml_path = settings.resolved_xml_path()
        if xml_path is None or not xml_path.exists():
            return settings
        basepath = settings.preprocess_source_path or settings.basepath_path
        basename = settings.basename
        local_output_dir = settings.local_output_dir
        if basepath is None or local_output_dir is None:
            raise ValueError("Cannot resolve raw source and processed-session output paths")
        chanmap_path, bad_channels = prepare_chanmap(
            basepath=basepath,
            basename=basename,
            local_output_dir=local_output_dir,
            probe_assignments=settings.preprocess.probe_assignments,
            reject_channels=settings.preprocess.reject_channels,
            xml_path=xml_path,
        )
        settings.chanmap_path = str(chanmap_path)
        settings.preprocess.reject_channels = list(bad_channels)
        self._refresh_suspended = True
        try:
            self.reject_channels.setText(", ".join(str(v) for v in bad_channels))
        finally:
            self._refresh_suspended = False
        self.chanmap_path.setText(str(chanmap_path))
        self._chanmap_controls_dirty = False
        self._last_chanmap_preview_key = None
        self._load_chanmap_preview(chanmap_path)
        self._append_log(f"Prepared chanMap for run: {chanmap_path}\nBad channels: {bad_channels}\n")
        return settings

    def _default_move_storage_dir_from_widgets(self) -> Path | None:
        basepath_text = self.basepath.text().strip()
        if not basepath_text:
            return None
        basepath = Path(basepath_text).expanduser()
        name = self.multi_day_name.text().strip()
        if self._multi_day_session_paths and name:
            return (basepath.parent / name).resolve()
        return basepath.resolve()

    def _update_move_storage_destination(self) -> None:
        if not hasattr(self, "move_storage_dir"):
            return
        destination = self._move_storage_override or self._default_move_storage_dir_from_widgets()
        self.move_storage_dir.setText(str(destination) if destination is not None else "")

    def _clear_move_storage_override(self, *_args: Any) -> None:
        self._move_storage_override = None
        self._move_source_snapshot = None
        self._update_move_storage_destination()

    def _browse_move_storage_dir(self) -> None:
        try:
            settings = self._collect_settings()
            if settings.local_output_dir is None or not settings.basename:
                raise ValueError("Select Basepath before choosing a storage destination.")
            start = self.move_storage_dir.text().strip() or settings.basepath or str(Path.cwd())
            path = self._select_directory("Select output storage directory", start)
            if not path:
                return
            destination = Path(path).expanduser().resolve()
            self._move_source_snapshot = (settings.local_output_dir.resolve(), settings.basename)
            self._move_storage_override = destination
            self.move_storage_dir.setText(str(destination))
            # The selected storage location becomes the displayed Basepath as
            # requested, while the pre-change Local source is retained above.
            self.basepath.setText(str(destination))
            self._schedule_refresh()
        except Exception as exc:
            QMessageBox.critical(self, "Select storage failed", str(exc))

    def _move_outputs_to_storage(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Run active", "Cannot move outputs while a pipeline job is running.")
            return
        if self._active_run_dir is not None:
            try:
                state_path = self._active_run_dir / "state.json"
                active_state = (
                    read_json(state_path)
                    if state_path.exists()
                    else RunStore(self._active_run_dir).derive_state()
                )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Run state unavailable",
                    f"Cannot safely move outputs until Run state can be checked: {exc}",
                )
                return
            if self._persistent_state_has_active_work(active_state):
                QMessageBox.warning(
                    self,
                    "Run active",
                    "Cannot move outputs while a Local/Slurm pipeline job is active.",
                )
                return
        try:
            settings = self._collect_settings()
            destination = self._move_storage_override or _default_output_storage_dir(settings)
            source_dir = self._move_source_snapshot[0] if self._move_source_snapshot else None
            source_basename = self._move_source_snapshot[1] if self._move_source_snapshot else None
            inventory_text = "Inventory will be resolved during staged transfer."
            preview_root = source_dir or settings.local_output_dir
            preview_basename = source_basename or settings.basename
            if preview_root is not None and Path(preview_root).is_dir():
                destination_path = Path(destination).expanduser().resolve() if destination else None
                session_xml_name = f"{preview_basename}.xml"
                destination_has_xml = bool(
                    destination_path
                    and (
                        (destination_path / session_xml_name).exists()
                        or (destination_path / session_xml_name).is_symlink()
                    )
                )
                selected: list[Path] = []
                retained: list[Path] = []
                for item in Path(preview_root).iterdir():
                    keep_local = (
                        item.name == "@eaDir"
                        or item.suffix.lower() == ".rhd"
                        or (
                            item.suffix.lower() == ".xml"
                            and (item.name != session_xml_name or destination_has_xml)
                        )
                        or (
                            item.name == f"{preview_basename}.dat"
                            and not self.move_dat_to_basepath.isChecked()
                        )
                    )
                    (retained if keep_local else selected).append(item)
                total_bytes = sum(
                    p.stat().st_size if p.is_file() else sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
                    for p in selected
                )
                inventory_text = (
                    f"Inventory: move {len(selected)} item(s), {total_bytes:,} bytes; "
                    f"retain {len(retained)} metadata/unchecked item(s); "
                    f"delete local source: {'yes' if self.move_clean_local.isChecked() else 'no'}."
                )
            message = (
                "Copy local output files to storage?\n\n"
                f"Storage destination: {destination or '-'}\n"
                f"Local output: {source_dir or settings.local_output_dir or '-'}\n"
                f"Copy basename.dat: {'yes' if self.move_dat_to_basepath.isChecked() else 'no'}\n"
                f"Overwrite existing files: {'yes' if self.move_overwrite.isChecked() else 'no'}\n"
                "Delete local after verified copy: "
                f"{'yes' if self.move_clean_local.isChecked() else 'no'}\n\n"
                f"{inventory_text}"
            )
            answer = QMessageBox.question(self, "Copy outputs to storage", message)
            if answer != QMessageBox.StandardButton.Yes:
                return
            result = _move_local_output_to_storage(
                settings,
                move_dat=self.move_dat_to_basepath.isChecked(),
                overwrite=self.move_overwrite.isChecked(),
                clean_after_move=self.move_clean_local.isChecked(),
                destination_dir=self._move_storage_override,
                source_dir=source_dir,
                source_basename=source_basename,
            )
            lines = [
                "Copy to storage finished",
                f"Storage destination: {result['storage_dir']}",
                f"Local output: {result['local_output_dir']}",
                "",
                f"Copied ({len(result['moved'])}):",
            ]
            lines.extend(
                [f"- {item['name']}" for item in result["moved"]]
                if result["moved"]
                else ["- none"]
            )
            lines.append("")
            lines.append(f"Skipped ({len(result['skipped'])}):")
            lines.extend(
                [f"- {item['name']}: {item['reason']}" for item in result["skipped"]]
                if result["skipped"]
                else ["- none"]
            )
            lines.append("")
            lines.append(f"Cleaned local output: {'yes' if result['cleaned'] else 'no'}")
            lines.append("")
            lines.append("Copy outputs to storage complete!")
            text = "\n".join(lines)
            self._settings_preview_text = text
            self._append_log(text + "\n")
        except Exception as exc:
            QMessageBox.critical(self, "Copy outputs failed", str(exc))

    def _move_outputs_to_basepath(self) -> None:
        """Compatibility alias for older direct callers."""
        self._move_outputs_to_storage()

    def _resolve_phy_params_path(self, sorting_folder: Path, *, prefer_postprocessed: bool = False) -> Path:
        sorting_folder = sorting_folder.expanduser().resolve()
        run_root = sorting_folder.parent if sorting_folder.name == "sorter_output" else sorting_folder
        candidate_folders: list[Path] = []

        def _add_candidate(folder: Path) -> None:
            folder = folder.resolve()
            if folder not in candidate_folders:
                candidate_folders.append(folder)

        if sorting_folder.name.endswith("_spi"):
            _add_candidate(sorting_folder)
        elif prefer_postprocessed:
            _add_candidate(postprocess_output_folder_for_sorting(sorting_folder))
            _add_candidate(sorting_folder)
            _add_candidate(run_root)
            _add_candidate(run_root / "sorter_output")
        else:
            _add_candidate(sorting_folder)
            _add_candidate(run_root)
            _add_candidate(run_root / "sorter_output")
            _add_candidate(postprocess_output_folder_for_sorting(sorting_folder))

        candidates = [folder / "params.py" for folder in candidate_folders]
        for params_path in candidates:
            if params_path.exists() and params_path.is_file():
                return params_path
        raise FileNotFoundError(
            "Could not find params.py for Phy. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )

    def _resolve_phy_program(self) -> str:
        executable_dir = Path(sys.executable).resolve().parent
        candidate_names = ["phy.exe", "phy"] if os.name == "nt" else ["phy"]
        for name in candidate_names:
            candidate = executable_dir / name
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        phy_program = shutil.which("phy")
        if phy_program is not None:
            return phy_program
        raise FileNotFoundError(
            "Could not find 'phy'. Start this GUI from the phy2 environment or install Phy there."
        )

    def _phy_is_running(self) -> bool:
        return self._phy_process is not None and self._phy_process.state() != QProcess.ProcessState.NotRunning

    def _cell_explorer_is_running(self) -> bool:
        return (
            self._cell_explorer_process is not None
            and self._cell_explorer_process.state() != QProcess.ProcessState.NotRunning
        )

    def _persistent_session_is_active(self, settings: PipelineGuiSettings) -> bool:
        try:
            session_dir = settings.local_output_dir
            if session_dir is None:
                return False
            workspace_text = settings.execution.workspace.strip()
            workspace = (
                Path(workspace_text).expanduser().resolve()
                if workspace_text
                else settings.local_root_path
            )
            pipeline_root = workspace if workspace.name == ".pipeline" else workspace / ".pipeline"
            from src.execution.session import active_session_claim

            return (
                active_session_claim(
                    pipeline_root=pipeline_root,
                    session_dir=Path(session_dir).resolve(),
                )
                is not None
            )
        except (OSError, ValueError, json.JSONDecodeError):
            # An unreadable persistent claim is unsafe to ignore.
            return True

    def _resolve_matlab_program(self, settings: PipelineGuiSettings) -> str:
        from src.preprocess.sorter_runner import _resolve_matlab_cmd

        matlab_text = (settings.execution.matlab_path or settings.preprocess.matlab_path).strip()
        matlab_cmd = _resolve_matlab_cmd(Path(matlab_text).expanduser() if matlab_text else None)
        if matlab_cmd is None:
            raise FileNotFoundError(
                "Could not find MATLAB. Set Preprocess > MATLAB path or add matlab to PATH."
            )
        return matlab_cmd

    def _resolve_cell_explorer_sorting_dir(self, settings: PipelineGuiSettings) -> Path:
        manual_text = self.manual_sorting_folder.text().strip()
        sorting_folder = Path(manual_text).expanduser() if manual_text else settings.postprocess_sorting_folder()
        if sorting_folder is None:
            raise FileNotFoundError(
                "No sorting folder could be resolved. Run sorting/postprocess first or set Postprocess target > sorting folder."
            )
        return self._resolve_phy_params_path(sorting_folder, prefer_postprocessed=True).parent

    def _resolve_cell_explorer_sorting_dirs(self, settings: PipelineGuiSettings) -> list[Path]:
        if not self._cell_explorer_sorting_folders:
            raise ValueError(
                "No CellExplore folders registered. Use Select folders and check at least one folder first."
            )
        folders: list[Path] = []
        seen: set[str] = set()
        for folder_text in self._cell_explorer_sorting_folders:
            folder = Path(folder_text).expanduser()
            resolved = self._resolve_phy_params_path(folder, prefer_postprocessed=True).parent
            key = str(resolved.resolve())
            if key in seen:
                continue
            folders.append(resolved)
            seen.add(key)
        return folders

    @staticmethod
    def _matlab_string(value: Path | str) -> str:
        return "'" + str(value).replace("'", "''") + "'"

    @classmethod
    def _matlab_cellstr(cls, values: list[Path]) -> str:
        return "{" + ", ".join(cls._matlab_string(value) for value in values) + "}"

    def _read_cell_explorer_stdout(self) -> None:
        process = self._cell_explorer_process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode(errors="replace")
        self._queue_log(text)

    def _read_cell_explorer_stderr(self) -> None:
        process = self._cell_explorer_process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode(errors="replace")
        self._queue_log(text)

    def _release_cell_explorer_session_claim(self) -> None:
        claims = self._cell_explorer_session_claims
        self._cell_explorer_session_claims = []
        from src.execution.session import release_manual_session_claim

        for session_dir, token in claims:
            release_manual_session_claim(session_dir=session_dir, token=token)

    def _release_phy_session_claim(self) -> None:
        claim = self._phy_session_claim
        self._phy_session_claim = None
        if claim is None:
            return
        from src.execution.session import release_manual_session_claim

        release_manual_session_claim(session_dir=claim[0], token=claim[1])

    def _cell_explorer_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._flush_log_buffer()
        self._append_log(f"\n=== CellExplorer MATLAB process closed (exit code {exit_code}) ===\n")
        self._cell_explorer_process = None
        self._cell_explorer_working_dir = None
        self._release_cell_explorer_session_claim()
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            self.launch_cell_explorer.setEnabled(True)
            if not self._phy_is_running():
                self.run_phy.setEnabled(True)

    def _cell_explorer_error_occurred(self, error: QProcess.ProcessError) -> None:
        self._cell_explorer_process = None
        self._cell_explorer_working_dir = None
        self._release_cell_explorer_session_claim()
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            self.launch_cell_explorer.setEnabled(True)
        QMessageBox.critical(
            self,
            "Run CellExplore postprocess failed",
            f"MATLAB process failed to start: {error.name}",
        )

    def _terminate_cell_explorer_process(self) -> None:
        process = self._cell_explorer_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            self._cell_explorer_process = None
            self._cell_explorer_working_dir = None
            self._release_cell_explorer_session_claim()
            return
        pid = int(process.processId())
        if os.name == "nt" and pid > 0:
            QProcess.startDetached("taskkill", ["/PID", str(pid), "/T", "/F"])
        elif pid > 0:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                process.terminate()
        else:
            process.terminate()
        if not process.waitForFinished(3000):
            process.kill()
            process.waitForFinished(2000)
        self._cell_explorer_process = None
        self._cell_explorer_working_dir = None
        self._release_cell_explorer_session_claim()

    def _launch_cell_explorer(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Run active", "Cannot run CellExplore postprocess while a pipeline job is running.")
            return
        if self._phy_is_running():
            QMessageBox.warning(self, "Phy active", "Close Phy before launching CellExplorer.")
            return
        if self._cell_explorer_is_running():
            QMessageBox.warning(self, "CellExplorer active", "A CellExplorer MATLAB process is already running.")
            return
        try:
            settings = self._collect_settings()
            if self._persistent_session_is_active(settings):
                raise RuntimeError(
                    "A persistent Run still owns this session. Wait for it to finish or cancel it before CellExplorer."
                )
            basepath = settings.local_output_dir
            if basepath is None or not settings.basename:
                raise ValueError("Local output directory cannot be resolved.")
            if not basepath.exists():
                raise FileNotFoundError(f"Local output directory does not exist: {basepath}")
            source_basepath = _resolve_cell_explorer_source_basepath(settings, basepath)
            if not VENDORED_CELLEXPLORER_ROOT.exists():
                raise FileNotFoundError(f"Vendored CellExplorer not found: {VENDORED_CELLEXPLORER_ROOT}")
            wrapper_path = MATLAB_SUPPORT_ROOT / "run_cell_explorer_processing.m"
            if not wrapper_path.exists():
                raise FileNotFoundError(f"CellExplorer MATLAB wrapper not found: {wrapper_path}")
            sorting_dirs = self._resolve_cell_explorer_sorting_dirs(settings)
            matlab_program = self._resolve_matlab_program(settings)
            from src.execution.session import acquire_manual_session_claim

            claim_dirs = {Path(basepath).resolve()}
            for sorting_dir in sorting_dirs:
                run_root = (
                    sorting_dir.parent if sorting_dir.name == "sorter_output" else sorting_dir
                )
                claim_dirs.add(run_root.parent.resolve())
            for claim_dir in sorted(claim_dirs, key=str):
                claim_token = acquire_manual_session_claim(
                    session_dir=claim_dir, owner="CellExplorer"
                )
                self._cell_explorer_session_claims.append((claim_dir, claim_token))
            if self._channel_regions:
                saved = self._persist_anatomical_map_to_default()
                if saved is not None:
                    self._append_log(f"Saved anatomical map for CellExplorer: {saved}\n")
            chan_coords_path = self._persist_cell_explorer_chan_coords()
            if chan_coords_path is not None:
                self._append_log(f"Saved chanCoords for CellExplorer: {chan_coords_path}\n")

            addpath_arg = self._matlab_string(MATLAB_SUPPORT_ROOT)
            args = [
                self._matlab_string(basepath),
                self._matlab_string(source_basepath),
                self._matlab_cellstr(sorting_dirs),
                self._matlab_string(VENDORED_CELLEXPLORER_ROOT),
            ]
            matlab_command = (
                "try, "
                f"addpath({addpath_arg}); "
                f"run_cell_explorer_processing({', '.join(args)}, "
                "'preferMergePointsDat', true, 'prePhy', false); "
                "exit(0); "
                "catch ME, disp(getReport(ME, 'extended', 'hyperlinks', 'off')); exit(1); end"
            )
            self._append_log("\n=== Running CellExplore postprocess ===\n")
            self._append_log(f"MATLAB: {matlab_program}\n")
            self._append_log(f"Working directory: {basepath}\n")
            self._append_log(f"Waveform source basepath: {source_basepath}\n")
            self._append_log(f"CellExplorer root: {VENDORED_CELLEXPLORER_ROOT}\n")
            self._append_log(
                f"CellExplore sorter mode: {'multi-sorter' if len(sorting_dirs) > 1 else 'single-sorter'}\n"
            )
            self._append_log(
                "Sorting folders:\n" + "".join(f"  - {path}\n" for path in sorting_dirs)
            )

            process = QProcess(self)
            process_environment = QProcessEnvironment.systemEnvironment()
            process_environment.insert("CUDA_VISIBLE_DEVICES", "")
            process.setProcessEnvironment(process_environment)
            process.setProgram(matlab_program)
            process.setArguments(["-nosplash", "-r", matlab_command])
            process.setWorkingDirectory(str(basepath))
            process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            if os.name != "nt" and hasattr(process, "setChildProcessModifier"):
                process.setChildProcessModifier(os.setsid)
            process.readyReadStandardOutput.connect(self._read_cell_explorer_stdout)
            process.readyReadStandardError.connect(self._read_cell_explorer_stderr)
            process.finished.connect(self._cell_explorer_finished)
            process.errorOccurred.connect(self._cell_explorer_error_occurred)
            self._cell_explorer_process = process
            self._cell_explorer_working_dir = basepath
            self.launch_cell_explorer.setEnabled(False)
            self.run_phy.setEnabled(False)
            process.start()
            if not process.waitForStarted(5000):
                self._cell_explorer_error_occurred(process.error())
            else:
                from src.execution.session import update_manual_session_claim_pid

                child_pid = int(process.processId())
                try:
                    for claim_dir, claim_token in self._cell_explorer_session_claims:
                        update_manual_session_claim_pid(
                            session_dir=claim_dir, token=claim_token, child_pid=child_pid
                        )
                except Exception:
                    self._terminate_cell_explorer_process()
                    raise
        except Exception as exc:
            self._release_cell_explorer_session_claim()
            QMessageBox.critical(self, "Run CellExplore postprocess failed", str(exc))

    def _format_seconds(self, total_seconds: float) -> str:
        total_seconds = max(0.0, float(total_seconds))
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds - hours * 3600 - minutes * 60
        if hours:
            return f"{hours} h {minutes:02d} min {seconds:06.3f} s"
        if minutes:
            return f"{minutes} min {seconds:06.3f} s"
        return f"{seconds:.3f} s"

    def _format_phy_duration(self, duration: Any) -> str:
        if duration is None:
            return "-"
        if isinstance(duration, (int, float, np.integer, np.floating)):
            return self._format_seconds(float(duration))
        text = str(duration).strip()
        match = re.fullmatch(r"(\d+):(\d{2}):(\d{2}):(\d{3})\s+hours", text)
        if match:
            hours, minutes, seconds, millis = (int(part) for part in match.groups())
            return f"{hours} h {minutes:02d} min {seconds:02d}.{millis:03d} s"
        return text

    def _read_phy_cluster_counts(self, phy_dir: Path) -> dict[str, int]:
        cluster_info_path = phy_dir / "cluster_info.tsv"
        if not cluster_info_path.exists():
            raise FileNotFoundError(f"cluster_info.tsv not found: {cluster_info_path}")

        counts = {
            "total": 0,
            "good": 0,
            "mua": 0,
            "noise": 0,
            "unclassified": 0,
            "other": 0,
        }
        unclassified_labels = {"", "nan", "none", "unsorted", "unclassified"}
        with cluster_info_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            if reader.fieldnames is None or "group" not in reader.fieldnames:
                raise ValueError(f"cluster_info.tsv does not contain a group column: {cluster_info_path}")
            for row in reader:
                counts["total"] += 1
                group = str(row.get("group") or "").strip().lower()
                if group in unclassified_labels:
                    counts["unclassified"] += 1
                elif group in ("good", "mua", "noise"):
                    counts[group] += 1
                else:
                    counts["other"] += 1
        return counts

    def _format_phy_cluster_counts(self, counts: dict[str, int]) -> str:
        return (
            f"total={counts['total']}, good={counts['good']}, mua={counts['mua']}, "
            f"noise={counts['noise']}, unclassified={counts['unclassified']}, other={counts['other']}"
        )

    def _append_phy_curation_status(self, *, force: bool = False, final: bool = False) -> None:
        if self._phy_working_dir is None:
            return
        try:
            counts = self._read_phy_cluster_counts(self._phy_working_dir)
        except Exception as exc:
            if force or self._phy_last_counts is None:
                self._append_warning_log(f"[WARN] Could not read Phy curation status: {exc}\n")
            return

        if not force and counts == self._phy_last_counts:
            return
        self._phy_last_counts = dict(counts)
        prefix = "Final Phy curation status" if final else "Phy curation status"
        self._append_log(f"{prefix}: {self._format_phy_cluster_counts(counts)}\n")

    def _append_phy_log_summary(self, phy_log_path: Path) -> None:
        try:
            phy_log_path = phy_log_path.resolve()
            if not phy_log_path.exists():
                self._append_warning_log(f"[WARN] Phy log not found yet: {phy_log_path}\n")
                return
            os.environ.setdefault(
                "MPLCONFIGDIR",
                str((Path(tempfile.gettempdir()) / "matplotlib-preprocess-gui").resolve()),
            )
            os.environ.setdefault(
                "NUMBA_CACHE_DIR",
                str((Path(tempfile.gettempdir()) / "numba-preprocess-gui").resolve()),
            )
            from neuro_py.raw.spike_sorting import phy_log_to_epocharray

            epochs = phy_log_to_epocharray(str(phy_log_path))
            n_epochs = getattr(epochs, "n_epochs", None)
            duration = getattr(epochs, "duration", None)
            self._append_log(
                "Phy log summary:\n"
                f"Log file: {phy_log_path}\n"
                f"Curation epochs: {n_epochs if n_epochs is not None else '-'}\n"
                f"Estimated curation time: {self._format_phy_duration(duration)}\n"
            )
        except Exception as exc:
            self._append_warning_log(f"[WARN] Could not summarize Phy log: {exc}\n")

    def _phy_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._phy_status_timer.stop()
        self._append_log(f"\n=== Phy closed (exit code {exit_code}) ===\n")
        self._append_phy_curation_status(force=True, final=True)
        if self._phy_working_dir is not None:
            self._append_phy_log_summary(self._phy_working_dir / "phy.log")
        self._phy_process = None
        self._phy_working_dir = None
        self._phy_last_counts = None
        self._release_phy_session_claim()
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            self.run_phy.setEnabled(True)
            if not self._cell_explorer_is_running():
                self.launch_cell_explorer.setEnabled(True)

    def _phy_error_occurred(self, error: QProcess.ProcessError) -> None:
        self._phy_status_timer.stop()
        self._phy_process = None
        self._phy_working_dir = None
        self._phy_last_counts = None
        self._release_phy_session_claim()
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            self.run_phy.setEnabled(True)
            if not self._cell_explorer_is_running():
                self.launch_cell_explorer.setEnabled(True)
        QMessageBox.critical(self, "Run phy failed", f"Phy process failed to start: {error.name}")

    def _run_phy(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Run active", "Cannot launch Phy while a pipeline job is running.")
            return
        if self._phy_is_running():
            QMessageBox.warning(self, "Phy already active", "A Phy process is already running.")
            return
        try:
            settings = self._collect_settings()
            if self._persistent_session_is_active(settings):
                raise RuntimeError(
                    "A persistent Run still owns this session. Wait for it to finish or cancel it before Phy."
                )
            manual_text = self.manual_sorting_folder.text().strip()
            sorting_folder = Path(manual_text).expanduser() if manual_text else settings.postprocess_sorting_folder()
            if sorting_folder is None:
                raise FileNotFoundError(
                    "No sorting folder could be resolved. Run postprocess first or choose a Manual Curation folder."
                )
            params_path = self._resolve_phy_params_path(
                sorting_folder,
                prefer_postprocessed=not bool(manual_text),
            )
            phy_program = self._resolve_phy_program()
            working_dir = params_path.parent
            run_root = sorting_folder.parent if sorting_folder.name == "sorter_output" else sorting_folder
            session_dir = run_root.parent.resolve()
            from src.execution.session import acquire_manual_session_claim

            claim_token = acquire_manual_session_claim(session_dir=session_dir, owner="Phy")
            self._phy_session_claim = (session_dir, claim_token)
            self._append_log(f"\n=== Launching Phy ===\n{phy_program} template-gui {params_path.name}\n")
            self._append_log(f"Working directory: {working_dir}\n")
            process = QProcess(self)
            process_environment = QProcessEnvironment.systemEnvironment()
            process_environment.insert("CUDA_VISIBLE_DEVICES", "")
            process.setProcessEnvironment(process_environment)
            process.setProgram(phy_program)
            process.setArguments(["template-gui", params_path.name])
            process.setWorkingDirectory(str(working_dir))
            process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            process.finished.connect(self._phy_finished)
            process.errorOccurred.connect(self._phy_error_occurred)
            self._phy_process = process
            self._phy_working_dir = working_dir
            self._phy_last_counts = None
            self.run_phy.setEnabled(False)
            process.start()
            if not process.waitForStarted(3000):
                self._phy_error_occurred(process.error())
                return
            from src.execution.session import update_manual_session_claim_pid

            try:
                update_manual_session_claim_pid(
                    session_dir=session_dir,
                    token=claim_token,
                    child_pid=int(process.processId()),
                )
            except Exception:
                process.kill()
                process.waitForFinished(3000)
                raise
            QTimer.singleShot(3000, lambda: self._append_phy_curation_status(force=True))
            QTimer.singleShot(3000, lambda path=working_dir / "phy.log": self._append_phy_log_summary(path))
            self._phy_status_timer.start()
        except Exception as exc:
            self._release_phy_session_claim()
            QMessageBox.critical(self, "Run phy failed", str(exc))

    def _force_stop_process(self) -> None:
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
            # Basepath/Local working dir may have changed since the last monitor
            # tick. Rebind synchronously so this click cannot cancel the Run from
            # the previously selected session.
            self._refresh_persistent_run_monitor()
            if self._active_run_dir is not None:
                try:
                    state_path = self._active_run_dir / "state.json"
                    state = (
                        read_json(state_path)
                        if state_path.exists()
                        else RunStore(self._active_run_dir).derive_state()
                    )
                except Exception as exc:
                    QMessageBox.critical(self, "Force stop failed", str(exc))
                    return
                if self._persistent_state_has_active_work(state):
                    self._request_persistent_run_cancel()
                    return
            self._append_log("\n=== Force stop requested, but no pipeline job is running ===\n")
            return
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Force stop")
        dialog.setText("Stop the current run?")
        stop_button = dialog.addButton("Stop", QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(stop_button)
        dialog.exec()
        if dialog.clickedButton() is not stop_button:
            return
        self._force_stop_requested = True
        self._process_stop_escalated = False
        self._append_log("\n=== Force stop requested ===\n")
        self._set_running(False)
        self._kill_process_tree()
        QTimer.singleShot(2500, self._escalate_force_stop)

    def _legacy_mutating_session_dir(
        self, settings: PipelineGuiSettings, mode: RunMode
    ) -> Path | None:
        """Resolve the canonical output session touched by a legacy GUI process."""
        if mode != "noise_label":
            return None
        sorting_folder = settings.postprocess_sorting_folder()
        if sorting_folder is not None:
            run_root = sorting_folder.parent if sorting_folder.name == "sorter_output" else sorting_folder
            return run_root.parent.resolve()
        local_output_dir = settings.local_output_dir
        return local_output_dir.resolve() if local_output_dir is not None else None

    def _legacy_process_group_is_stopped(self) -> bool:
        pid = self._legacy_process_group_pid
        if pid is None or pid <= 0:
            return True
        if os.name == "nt":
            # Windows taskkill / QProcess do not expose an equivalent safe
            # process-group probe.  Do not infer worker termination from a
            # generic error; the finished callback is the conclusive signal.
            return self._process is None
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False

    def _release_legacy_process_session_claim(self) -> bool:
        claim = self._legacy_process_session_claim
        if claim is None:
            return True
        if not self._legacy_process_group_is_stopped():
            QTimer.singleShot(250, self._release_legacy_process_session_claim)
            return False
        self._legacy_process_session_claim = None
        self._legacy_process_group_pid = None
        from src.execution.session import release_manual_session_claim

        release_manual_session_claim(session_dir=claim[0], token=claim[1])
        return True

    def _start_run(self, mode: RunMode) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Run already active", "A pipeline job is already running.")
            return
        try:
            settings = self._collect_settings()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        checks = run_preflight(settings, mode)
        blocking = [c for c in checks if c.status == "error"]
        if blocking:
            QMessageBox.critical(self, "Preflight failed", "\n".join(self._format_checks(blocking)))
            return
        warnings = [c for c in checks if c.status == "warn"]
        if warnings:
            answer = QMessageBox.question(
                self,
                "Warnings",
                "\n".join(self._format_checks(warnings)) + "\n\nContinue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._force_stop_requested = False
        self._process_stop_escalated = False
        if mode in {"all", "preprocess", "postprocess"}:
            self._start_persistent_run(settings, mode)
            return
        session_dir = self._legacy_mutating_session_dir(settings, mode)
        if session_dir is not None:
            try:
                from src.execution.session import acquire_manual_session_claim

                claim_token = acquire_manual_session_claim(
                    session_dir=session_dir, owner="Legacy noise labeling"
                )
                self._legacy_process_session_claim = (session_dir, claim_token)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Session is active",
                    f"Cannot start noise labeling because this session is active: {exc}",
                )
                return
        self._set_running(True)
        self._append_log(f"\n=== Running {mode} ===\n")
        try:
            fd, config_name = tempfile.mkstemp(prefix="preprocess_gui_", suffix=".json")
            os.close(fd)
            config_path = Path(config_name)
            settings.save(config_path)
        except Exception as exc:
            self._set_running(False)
            self._release_legacy_process_session_claim()
            QMessageBox.critical(self, "Run failed", f"Could not prepare legacy run: {exc}")
            return
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments([
            "-m",
            "src.preprocess.gui.run_pipeline",
            "--config",
            str(config_path),
            "--mode",
            mode,
        ])
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        if os.name != "nt" and hasattr(process, "setChildProcessModifier"):
            process.setChildProcessModifier(os.setsid)
        process.readyReadStandardOutput.connect(self._read_process_stdout)
        process.readyReadStandardError.connect(self._read_process_stderr)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error_occurred)
        self._process = process
        self._process_config_path = config_path
        self._process_result = None
        self._process_error = None
        self._process_tail = ""
        process.start()
        if not process.waitForStarted(3000):
            self._process_error_occurred(process.error())
            return
        claim = self._legacy_process_session_claim
        if claim is not None:
            try:
                from src.execution.session import update_manual_session_claim_pid

                update_manual_session_claim_pid(
                    session_dir=claim[0], token=claim[1], child_pid=int(process.processId())
                )
                self._legacy_process_group_pid = int(process.processId())
            except Exception:
                process.kill()
                process.waitForFinished(3000)
                self._release_legacy_process_session_claim()
                if self._process is process:
                    self._process = None
                if self._process_config_path == config_path:
                    config_path.unlink(missing_ok=True)
                    self._process_config_path = None
                self._set_running(False)
                QMessageBox.critical(
                    self,
                    "Run failed",
                    "Noise labeling started, but its session claim could not be updated. "
                    "The process was stopped before continuing.",
                )
                return

    def _start_persistent_run(self, settings: PipelineGuiSettings, mode: RunMode) -> None:
        try:
            requested = RequestedBackend(settings.execution.requested_backend.lower())
            capabilities = (
                None
                if requested == RequestedBackend.LOCAL
                else self._slurm_capabilities or detect_slurm_capabilities(timeout=3.0)
            )
            resolved, capabilities = resolve_backend(
                requested,
                require_sacct=settings.execution.require_sacct,
                capabilities=capabilities,
            )
            if requested != RequestedBackend.LOCAL:
                self._slurm_capabilities = capabilities
            execution = settings.execution.to_execution_config(resolved_backend=resolved)
            run_dir = create_run(
                settings=settings,
                execution=execution,
                mode=mode,
                capabilities=capabilities,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Persistent Run submission failed", str(exc))
            return
        self._set_active_run(run_dir, session_dir=settings.local_output_dir)
        self.execution_resolved.setText(f"Using {resolved.value.capitalize()} for this Run.")
        if not self._launch_persistent_controller([]):
            self._append_warning_log(
                f"[WARN] Run was created but its controller did not start: {run_dir}\n"
            )
            return
        self._append_log(
            f"\n=== Persistent {mode} Run submitted ===\n"
            f"Run directory: {run_dir}\n"
            f"Requested backend: {requested.value}\n"
            f"Resolved backend: {resolved.value}\n"
            "The GUI may be closed without cancelling this Run.\n"
        )
        self.tabs.setCurrentIndex(0)
        self.ephys_tabs.setCurrentIndex(self._ephys_run_tab_index)
        QTimer.singleShot(750, self._refresh_persistent_run_monitor)

    def _read_process_stdout(self) -> None:
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardOutput()).decode(errors="replace")
        self._handle_process_output(text)

    def _read_process_stderr(self) -> None:
        process = self._process
        if process is None:
            return
        text = bytes(process.readAllStandardError()).decode(errors="replace")
        self._queue_log(text)

    def _handle_process_output(self, text: str) -> None:
        combined = self._process_tail + text
        lines = combined.splitlines(keepends=True)
        self._process_tail = ""
        for line in lines:
            if not line.endswith(("\n", "\r")):
                self._process_tail = line
                continue
            stripped = line.strip()
            if stripped.startswith(RESULT_PREFIX):
                try:
                    self._process_result = json.loads(stripped.removeprefix(RESULT_PREFIX))
                except json.JSONDecodeError:
                    self._queue_log(line)
            elif stripped.startswith(ERROR_PREFIX):
                try:
                    self._process_error = json.loads(stripped.removeprefix(ERROR_PREFIX))
                except json.JSONDecodeError:
                    self._queue_log(line)
            else:
                self._queue_log(line)

    def _process_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._process_tail:
            self._handle_process_output("\n")
        self._flush_log_buffer()
        stopped = self._force_stop_requested
        self._force_stop_requested = False
        self._process_stop_escalated = False
        self._set_running(False)
        if self._process_config_path is not None:
            self._process_config_path.unlink(missing_ok=True)
            self._process_config_path = None
        self._process = None
        self._release_legacy_process_session_claim()
        if stopped:
            self._append_log("=== Force stop complete ===\n")
            return
        if exit_code == 0:
            self._append_log("\n=== Run finished ===\n")
            if self._process_result:
                multi_day_result = self._process_result.get("multi_day_result") or {}
                server_basepath = multi_day_result.get("server_basepath")
                if server_basepath:
                    self.basepath.setText(server_basepath)
                    self.multi_day_name.setText(str(multi_day_result.get("name") or self.multi_day_name.text()))
                pre_result = self._process_result.get("preprocess_result") or {}
                sorter_output = pre_result.get("sorter_output_dir")
                sorter_outputs = pre_result.get("sorter_output_dirs") or []
                if sorter_output:
                    self.sorting_phy_folder.setText(sorter_output)
                elif len(sorter_outputs) > 1:
                    self.sorting_phy_folder.clear()
                    local_output_dir = pre_result.get("local_output_dir")
                    if local_output_dir:
                        self.sorting_search_root.setText(local_output_dir)
                    self._append_log(
                        "Multiple sorter folders were created; postprocess will use the sorting search root.\n"
                    )
                self._cleanup_postprocess_caches_from_result(self._process_result)
        else:
            self._append_log("\n=== Run failed ===\n")
            message = f"Pipeline process exited with code {exit_code}."
            if self._process_error:
                err_type = self._process_error.get("type", "Error")
                err_message = self._process_error.get("message", "")
                message = f"{err_type}: {err_message}" if err_message else str(err_type)
            QMessageBox.critical(self, "Run failed", message)
        self._refresh_preview()

    def _cleanup_postprocess_caches_from_result(self, result: dict[str, Any]) -> None:
        # The worker is the cache owner: a returned directory means the saved
        # postprocess configuration intentionally retained it.  Deleting it here
        # used to override delete_analyzer_cache=False after an otherwise
        # successful run.
        del result

    def _remove_tree_with_retry(self, path: Path, *, retries: int = 8, delay: float = 1.0) -> None:
        for attempt in range(retries):
            try:
                shutil.rmtree(path)
                return
            except PermissionError:
                if attempt >= retries - 1:
                    raise
                time.sleep(delay)

    def _descendant_pids(self, root_pid: int) -> list[int]:
        proc_root = Path("/proc")
        if os.name == "nt" or not proc_root.exists() or root_pid <= 0:
            return []
        children: dict[int, list[int]] = {}
        for stat_path in proc_root.glob("[0-9]*/stat"):
            try:
                text = stat_path.read_text(encoding="utf-8", errors="replace")
                rparen = text.rfind(")")
                if rparen < 0:
                    continue
                pid = int(stat_path.parent.name)
                fields = text[rparen + 2 :].split()
                if len(fields) < 2:
                    continue
                ppid = int(fields[1])
            except Exception:
                continue
            children.setdefault(ppid, []).append(pid)

        descendants: list[int] = []
        stack = list(children.get(root_pid, []))
        while stack:
            pid = stack.pop()
            descendants.append(pid)
            stack.extend(children.get(pid, []))
        return descendants

    def _send_signal_to_process_tree(self, sig: signal.Signals) -> None:
        process = self._process
        if process is None:
            return
        pid = int(process.processId())
        if os.name == "nt" and pid > 0:
            QProcess.startDetached("taskkill", ["/PID", str(pid), "/T", "/F"])
            return
        if pid > 0:
            descendants = self._descendant_pids(pid)
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            for child_pid in reversed(descendants):
                try:
                    os.kill(child_pid, sig)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        else:
            if sig == signal.SIGKILL:
                process.kill()
            else:
                process.terminate()

    def _kill_process_tree(self) -> None:
        self._send_signal_to_process_tree(signal.SIGTERM)

    def _escalate_force_stop(self) -> None:
        process = self._process
        if not self._force_stop_requested or process is None:
            return
        if process.state() == QProcess.ProcessState.NotRunning:
            return
        if self._process_stop_escalated:
            return
        self._process_stop_escalated = True
        self._append_log("=== Force stop escalation: killing remaining worker processes ===\n")
        self._send_signal_to_process_tree(signal.SIGKILL)
        process.kill()
        QTimer.singleShot(1500, self._detach_stuck_process_after_force_stop)

    def _detach_stuck_process_after_force_stop(self) -> None:
        process = self._process
        if not self._force_stop_requested or process is None:
            return
        if process.state() == QProcess.ProcessState.NotRunning:
            return
        self._append_warning_log(
            "[WARN] Pipeline process did not report exit after SIGKILL; detaching GUI state.\n"
        )
        self._set_running(False)

    def _process_error_occurred(self, error: QProcess.ProcessError) -> None:
        self._force_stop_requested = False
        self._process_stop_escalated = False
        self._set_running(False)
        if self._process_config_path is not None:
            self._process_config_path.unlink(missing_ok=True)
            self._process_config_path = None
        self._process = None
        # A generic QProcess error is not evidence that descendants are gone.
        # Retain the manual claim until the process group probe proves it.
        self._release_legacy_process_session_claim()
        QMessageBox.critical(self, "Run failed", f"Pipeline process failed to start: {error.name}")

    def closeEvent(self, event: Any) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            if self._force_stop_requested:
                self._send_signal_to_process_tree(signal.SIGKILL)
                self._process.kill()
                self._append_log("\n=== Closing GUI after force stop request ===\n")
                super().closeEvent(event)
                return
            QMessageBox.warning(
                self,
                "Run active",
                "A pipeline job is still running. Wait for it to finish or use Force stop to terminate the run.",
            )
            event.ignore()
            return
        if self._phy_is_running():
            QMessageBox.warning(
                self,
                "Phy active",
                "A Phy process is still running. Close Phy first so the GUI can write the final curation summary.",
            )
            event.ignore()
            return
        if self._cell_explorer_is_running():
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setWindowTitle("CellExplorer active")
            dialog.setText("A CellExplorer MATLAB process is still running.")
            terminate_button = dialog.addButton("Terminate and close", QMessageBox.ButtonRole.DestructiveRole)
            dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            dialog.setDefaultButton(terminate_button)
            dialog.exec()
            if dialog.clickedButton() is not terminate_button:
                event.ignore()
                return
            self._terminate_cell_explorer_process()
            self._append_log("\n=== Terminated CellExplorer MATLAB process while closing GUI ===\n")
        super().closeEvent(event)

    def _set_running(self, running: bool) -> None:
        for button in [
            self.run_all,
            self.run_pre,
            self.run_post,
            self.run_phy,
            self.launch_cell_explorer,
            self.run_noise_label,
            self.run_behavior_cleanup,
            self.run_behavior,
            self.browse_move_storage,
            self.move_outputs,
        ]:
            button.setEnabled(not running)
        if self._phy_is_running():
            self.run_phy.setEnabled(False)
        if self._cell_explorer_is_running():
            self.launch_cell_explorer.setEnabled(False)
            self.run_phy.setEnabled(False)
        self.force_stop.setEnabled(running)

    def _append_log(self, text: str, *, warning: bool = False) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        cursor = self.log.textCursor()
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#ff8a80" if warning else "#d4d4d4"))
        cursor.insertText(text, fmt)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _append_warning_log(self, text: str) -> None:
        self._append_log(text, warning=True)

    def _append_behavior_warnings(self, header: str, warnings_list: list[str]) -> None:
        new_warnings: list[str] = []
        for item in warnings_list:
            if item in self._reported_behavior_warnings:
                continue
            self._reported_behavior_warnings.add(item)
            new_warnings.append(item)
        if new_warnings:
            self._append_warning_log(header + ":\n" + "\n".join(f"- {item}" for item in new_warnings) + "\n")

    def _queue_log(self, text: str) -> None:
        if not text:
            return
        self._log_buffer += text
        if len(self._log_buffer) > 8192:
            self._flush_log_buffer()
        elif not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_log_buffer(self) -> None:
        if not self._log_buffer:
            return
        text = self._log_buffer
        self._log_buffer = ""
        self._append_log(text)

def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._fit_to_available_screen()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
