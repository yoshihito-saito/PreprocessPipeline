from __future__ import annotations

import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
from scipy.io import loadmat

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
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
from src.preprocess.io import build_channel_map_data, set_tree_world_rw
from src.preprocess.paths import find_project_root, resolve_project_path
from src.worker_defaults import default_worker_count, normalize_worker_count

from .config_model import (
    BehaviorGuiSettings,
    PipelineGuiSettings,
    PostprocessGuiSettings,
    PreprocessGuiSettings,
    RunMode,
    parse_float_pair,
    parse_int_list,
)
from .preflight import CheckResult, run_preflight
from .run_pipeline import ERROR_PREFIX, RESULT_PREFIX


REPO_ROOT = find_project_root()
PUBLIC_DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocess_gui_default_config.json"
LOCAL_DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "preprocess_gui_default_config.local.json"


def _default_config_path() -> Path:
    override = os.environ.get("PREPROCESS_GUI_DEFAULT_CONFIG")
    if override:
        return Path(override).expanduser()
    if LOCAL_DEFAULT_CONFIG_PATH.exists():
        return LOCAL_DEFAULT_CONFIG_PATH
    return PUBLIC_DEFAULT_CONFIG_PATH


DEFAULT_CONFIG_PATH = _default_config_path()

PROBE_TYPES = (
    "middle_finger",
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
    defaults.chanmap_path = ""
    defaults.preprocess.reject_channels = []
    defaults.preprocess.matlab_path = ""
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


def _move_local_output_to_basepath(
    settings: PipelineGuiSettings, *, move_dat: bool, overwrite: bool, clean_after_move: bool
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

    conflicts = [dst for _src, dst in move_items if dst.exists() or dst.is_symlink()]
    if conflicts and not overwrite:
        shown = "\n".join(str(path) for path in conflicts[:20])
        suffix = "\n..." if len(conflicts) > 20 else ""
        raise FileExistsError(
            "Destination already exists. Enable overwrite to replace it:\n"
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

    cleaned = False
    if clean_after_move and src_root.exists():
        shutil.rmtree(src_root)
        cleaned = True

    return {
        "basepath": str(dst_root),
        "local_output_dir": str(src_root),
        "moved": moved,
        "skipped": skipped,
        "overwrite": overwrite,
        "move_dat": move_dat,
        "clean_after_move": clean_after_move,
        "cleaned": cleaned,
    }


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class ChanMapCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5.5, 4.0), facecolor="#2f2f2f")
        self.canvas = FigureCanvas(self.figure)
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
        self.figure.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.90)
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
        self.figure.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.90)
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
                    clip_on=True,
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PreprocessPipeline GUI")
        self.resize(1320, 860)
        self._process: QProcess | None = None
        self._process_config_path: Path | None = None
        self._process_result: dict[str, Any] | None = None
        self._process_error: dict[str, Any] | None = None
        self._process_tail = ""
        self._log_buffer = ""
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.setInterval(80)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._force_stop_requested = False
        self._probe_rows: list[dict[str, QWidget]] = []
        self.noise_threshold_fields: dict[str, QLineEdit] = {}
        self._refresh_suspended = False
        self._last_chanmap_preview_key: tuple[Any, ...] | None = None
        self._chanmap_controls_dirty = False
        self._behavior_dlc_files: list[Any] = []
        self._behavior_frame_canvases: dict[str, BehaviorFrameCanvas] = {}
        self._behavior_pixel_distances_by_folder: dict[str, float] = {}
        self._behavior_pixel_to_cm_ratios_by_folder: dict[str, float] = {}
        self._behavior_clean_mask: np.ndarray | None = None
        self._behavior_outlier_canvases: dict[str, tuple[BehaviorTrackCanvas, np.ndarray]] = {}
        self._reported_behavior_warnings: set[str] = set()
        self._behavior_outlier_processed_preview = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh_preview)
        self._apply_dark_theme()
        self._build_ui()
        self._set_running(False)
        self._apply_settings(_load_default_settings())
        self._refresh_preview()

    def _apply_dark_theme(self) -> None:
        check_icon = (Path(__file__).resolve().parent / "assets" / "check-orange.svg").as_posix()
        stylesheet = """
            QMainWindow {
                background: #252525;
            }
            QWidget {
                color: #e5e5e5;
                font-size: 12px;
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
                margin-top: 18px;
                padding: 12px;
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
                font-size: 11px;
                font-weight: 400;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #383838;
                color: #f0f0f0;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px 7px;
                selection-background-color: #606060;
            }
            QPlainTextEdit {
                font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
                font-size: 12px;
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
                padding: 7px 11px;
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
                padding: 8px 12px;
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
                font-size: 12px;
            }
            QMessageBox QPushButton {
                background: #3a3a3a;
                color: #f5f5f5;
                border: 1px solid #5f5f5f;
                border-radius: 5px;
                min-width: 72px;
                padding: 7px 12px;
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

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_settings_tabs())
        splitter.addWidget(self._build_center_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 780])
        layout.addWidget(splitter, 1)

        layout.addWidget(self._build_run_bar())

        self.setCentralWidget(root)

    def _build_top_bar(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.basepath = QLineEdit()
        self.basepath.setPlaceholderText("Select session/basepath")
        browse_basepath = QPushButton("Browse basepath")
        browse_basepath.clicked.connect(self._browse_basepath)

        self.local_root = QLineEdit()
        self.local_root.setPlaceholderText("Local working directory")
        browse_local = QPushButton("Browse local")
        browse_local.clicked.connect(self._browse_local_root)

        load_config = QPushButton("Load config")
        load_config.clicked.connect(self._load_config)
        save_config = QPushButton("Save config")
        save_config.clicked.connect(self._save_config)

        layout.addWidget(browse_basepath, 0, 0)
        layout.addWidget(self.basepath, 0, 1)
        layout.addWidget(browse_local, 0, 2)
        layout.addWidget(QLabel("Local working dir"), 0, 3)
        layout.addWidget(self.local_root, 0, 4)
        layout.addWidget(load_config, 0, 5)
        layout.addWidget(save_config, 0, 6)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(4, 2)

        self.basepath.textChanged.connect(self._schedule_refresh)
        self.local_root.textChanged.connect(self._schedule_refresh)
        self.basepath.textChanged.connect(self._reset_behavior_discovery_state)
        self.local_root.textChanged.connect(self._reset_behavior_discovery_state)
        return panel

    def _build_settings_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_ephys_tab(), "Ephys")
        self.tabs.addTab(self._scroll_area(self._build_behavior_tab()), "Behavior")
        self.tabs.setMinimumWidth(500)
        self.tabs.currentChanged.connect(lambda _index: self._schedule_refresh())
        return self.tabs

    def _build_ephys_tab(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settingsPage")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.ephys_tabs = QTabWidget()
        self.ephys_tabs.addTab(self._scroll_area(self._build_preprocess_tab()), "Preprocess")
        self.ephys_tabs.addTab(self._scroll_area(self._build_postprocess_tab()), "Postprocess")
        self.ephys_tabs.currentChanged.connect(lambda _index: self._schedule_refresh())

        run_box = QGroupBox("Ephys run")
        run_layout = QHBoxLayout(run_box)
        self.run_all = QPushButton("Run all")
        self.run_all.setObjectName("primaryButton")
        self.run_pre = QPushButton("Run preprocess")
        self.run_post = QPushButton("Run postprocess")
        self.run_all.clicked.connect(lambda: self._start_run("all"))
        self.run_pre.clicked.connect(lambda: self._start_run("preprocess"))
        self.run_post.clicked.connect(lambda: self._start_run("postprocess"))
        run_layout.addWidget(self.run_all)
        run_layout.addWidget(self.run_pre)
        run_layout.addWidget(self.run_post)

        layout.addWidget(self.ephys_tabs, 1)
        layout.addWidget(run_box)
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
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
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
        self.reject_channels = QLineEdit()
        self.reject_channels.setPlaceholderText("0, 3, 17")
        self.reject_channels.textChanged.connect(self._mark_chanmap_controls_dirty)
        self.chanmap_path = QLineEdit()
        self.chanmap_path.setPlaceholderText("optional explicit chanMap.mat path")
        load_chanmap = QPushButton("Load chanMap")
        load_chanmap.clicked.connect(self._browse_chanmap)
        chanmap_buttons = QHBoxLayout()
        chanmap_buttons.addWidget(load_chanmap)
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

        sorter = QGroupBox("Sorter and runtime")
        sf = self._form_layout(sorter)
        self.run_sorter = QCheckBox("Run sorter")
        self.sorter = NoWheelComboBox()
        self.sorter.addItems(["Kilosort", "Kilosort2_5", "kilosort4", "disabled"])
        self.sorter.currentTextChanged.connect(self._sorter_changed)
        self.run_sorter.toggled.connect(self._update_sorter_enabled)
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
        sf.addRow("Sorter path", self.sorter_path)
        sf.addRow("Sorter config", sorter_config_row)
        sf.addRow("MATLAB path", self.matlab_path)
        sf.addRow("Workers for sorter", self.sorter_worker_count)

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
        chanmap_layout.addWidget(chanmap_head)
        chanmap_layout.addWidget(self.chanmap_canvas, 1)

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
        self.log.setMinimumHeight(160)
        log_layout.addWidget(log_head)
        log_layout.addWidget(self.log, 1)

        monitor_layout.addWidget(self.monitor_stack, 3)
        monitor_layout.addWidget(log_panel, 2)

        preview_panel = QFrame()
        preview_panel.setObjectName("miniPanel")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        preview_head = QLabel("Setting config")
        preview_head.setObjectName("miniHead")
        self.run_preview = QPlainTextEdit()
        self.run_preview.setReadOnly(True)
        self.run_preview.setMinimumHeight(180)
        preview_layout.addWidget(preview_head)
        preview_layout.addWidget(self.run_preview, 1)

        layout.addWidget(monitor, 3)
        layout.addWidget(preview_panel, 2)
        return panel

    def _build_run_bar(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.move_dat_to_basepath = QCheckBox("Move basename.dat")
        self.move_overwrite = QCheckBox("Overwrite moved files")
        self.move_clean_local = QCheckBox("Clean local after move")
        self.move_clean_local.setChecked(True)
        self.move_outputs = QPushButton("Move outputs to basepath")
        self.force_stop = QPushButton("Force stop")
        self.force_stop.setObjectName("dangerButton")
        self.clear_log = QPushButton("Clear log")
        self.move_outputs.clicked.connect(self._move_outputs_to_basepath)
        self.force_stop.clicked.connect(self._force_stop_process)
        self.clear_log.clicked.connect(self.log.clear)
        layout.addStretch(1)
        layout.addWidget(self.move_dat_to_basepath)
        layout.addWidget(self.move_overwrite)
        layout.addWidget(self.move_clean_local)
        layout.addWidget(self.move_outputs)
        layout.addWidget(self.force_stop)
        layout.addWidget(self.clear_log)
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
        geometry.setCurrentText(str(assignment.get("type") or "staggered"))

        groups = QLineEdit()
        raw_groups = assignment.get("groups") or []
        groups.setPlaceholderText("0, 1, 2, 3")
        groups.setText(", ".join(str(int(v)) for v in raw_groups))

        x_offset = self._spin(-1000000, 1000000, int(assignment.get("x_offset") or 0))

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

    def _load_settings_chanmap_preview(self, settings: PipelineGuiSettings) -> bool:
        basepath = settings.basepath_path
        if basepath is None:
            return False
        xml_path = basepath / f"{settings.basename}.xml"
        if not xml_path.exists():
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
            )
            if data is None:
                return False
            self.chanmap_canvas.render_chanmap(data, source="current GUI settings")
            self._last_chanmap_preview_key = key
            return True
        except Exception:
            return False

    def _select_directory(self, title: str, start: str) -> str:
        dialog = QFileDialog(self, title, start or str(Path.cwd()))
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        return dialog.selectedFiles()[0] if dialog.exec() else ""

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
            self.basepath.setText(path)
            self._auto_load_existing_chanmap()
            self._schedule_refresh()

    def _browse_local_root(self) -> None:
        path = self._select_directory("Select local output root", self.local_root.text() or str(Path.cwd()))
        if path:
            self.local_root.setText(path)
            self._auto_load_existing_chanmap()
            self._schedule_refresh()

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
        path = self._select_open_file("Load GUI config", str(Path.cwd()), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            self._apply_settings(PipelineGuiSettings.load(Path(path)))
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
            loaded.chanmap_path = current.chanmap_path
            loaded.postprocess.sorting_phy_folder = current.postprocess.sorting_phy_folder
            loaded.postprocess.sorting_search_root = current.postprocess.sorting_search_root
            self._apply_settings(loaded)
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
            matlab_path=self.matlab_path.text().strip(),
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
        return PipelineGuiSettings(
            basepath=self.basepath.text().strip(),
            local_root=self.local_root.text().strip(),
            chanmap_path=self.chanmap_path.text().strip(),
            preprocess=preprocess,
            behavior=behavior,
            postprocess=postprocess,
        )

    def _apply_settings(self, settings: PipelineGuiSettings) -> None:
        self._refresh_suspended = True
        try:
            self.basepath.setText(settings.basepath)
            self.local_root.setText(settings.local_root or str(settings.local_root_path))
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
            default_sorter_path, default_sorter_config = self._current_sorter_defaults()
            self.sorter_path.setText(p.sorter_path or default_sorter_path)
            self.sorter_config_path.setText(p.sorter_config_path or default_sorter_config)
            self.matlab_path.setText(p.matlab_path)
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
            chanmap = settings.resolved_chanmap_path()
            if not self._load_settings_chanmap_preview(settings) and chanmap is not None and chanmap.exists():
                self._load_chanmap_preview(chanmap)
        finally:
            self._normalize_worker_fields()
            self._refresh_suspended = False

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
            self.run_preview.setPlainText("\n".join(lines))
            chanmap = settings.resolved_chanmap_path()
            explicit_chanmap = Path(settings.chanmap_path).expanduser() if settings.chanmap_path.strip() else None
            if (
                explicit_chanmap is not None
                and explicit_chanmap.exists()
                and not self._chanmap_controls_dirty
            ):
                self._load_chanmap_preview(explicit_chanmap)
            elif not self._load_settings_chanmap_preview(settings) and chanmap is not None and chanmap.exists():
                self._load_chanmap_preview(chanmap)
        except Exception as exc:
            self.run_preview.setPlainText(f"Config error:\n{exc}")

    def _format_checks(self, checks: list[CheckResult]) -> list[str]:
        prefix = {"ok": "[OK]", "warn": "[WARN]", "error": "[ERROR]"}
        return [f"{prefix.get(c.status, '[?]')} {c.label}: {c.detail}" for c in checks]

    def _generate_chanmap(self) -> None:
        try:
            settings = self._collect_settings()
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
            self.chanmap_path.setText(str(chanmap_path))
            self._load_chanmap_preview(chanmap_path)
            self._append_log(f"Generated chanMap: {chanmap_path}\nBad channels: {bad_channels}\n")
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Generate chanMap failed", str(exc))

    def _move_outputs_to_basepath(self) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Run active", "Cannot move outputs while a pipeline job is running.")
            return
        try:
            settings = self._collect_settings()
            message = (
                "Move local output files to basepath?\n\n"
                f"Basepath: {settings.basepath or '-'}\n"
                f"Local output: {settings.local_output_dir or '-'}\n"
                f"Move basename.dat: {'yes' if self.move_dat_to_basepath.isChecked() else 'no'}\n"
                f"Overwrite existing files: {'yes' if self.move_overwrite.isChecked() else 'no'}\n"
                f"Clean local after move: {'yes' if self.move_clean_local.isChecked() else 'no'}"
            )
            answer = QMessageBox.question(self, "Move outputs to basepath", message)
            if answer != QMessageBox.StandardButton.Yes:
                return
            result = _move_local_output_to_basepath(
                settings,
                move_dat=self.move_dat_to_basepath.isChecked(),
                overwrite=self.move_overwrite.isChecked(),
                clean_after_move=self.move_clean_local.isChecked(),
            )
            lines = [
                "Move to basepath finished",
                f"Basepath: {result['basepath']}",
                f"Local output: {result['local_output_dir']}",
                "",
                f"Moved ({len(result['moved'])}):",
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
            lines.append("Move outputs to basepath complete!")
            text = "\n".join(lines)
            self.run_preview.setPlainText(text)
            self._append_log(text + "\n")
        except Exception as exc:
            QMessageBox.critical(self, "Move outputs failed", str(exc))

    def _force_stop_process(self) -> None:
        if self._process is None or self._process.state() == QProcess.ProcessState.NotRunning:
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
        self._append_log("\n=== Force stop requested ===\n")
        self._kill_process_tree()

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
        self._set_running(True)
        self._append_log(f"\n=== Running {mode} ===\n")
        fd, config_name = tempfile.mkstemp(prefix="preprocess_gui_", suffix=".json")
        os.close(fd)
        config_path = Path(config_name)
        settings.save(config_path)
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
        self._set_running(False)
        if self._process_config_path is not None:
            self._process_config_path.unlink(missing_ok=True)
            self._process_config_path = None
        self._process = None
        if stopped:
            self._append_log("=== Force stop complete ===\n")
            return
        if exit_code == 0:
            self._append_log("\n=== Run finished ===\n")
            if self._process_result:
                pre_result = self._process_result.get("preprocess_result") or {}
                sorter_output = pre_result.get("sorter_output_dir")
                if sorter_output:
                    self.sorting_phy_folder.setText(sorter_output)
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
        post_result = result.get("postprocess_results") or {}
        cache_dirs = post_result.get("analyzer_cache_dirs") or []
        if not cache_dirs:
            return
        for cache_dir in cache_dirs:
            path = Path(str(cache_dir))
            if not path.exists():
                continue
            try:
                self._append_log(f"Cleaning analyzer cache after process exit: {path}\n")
                self._remove_tree_with_retry(path)
                self._append_log(f"Analyzer cache removed: {path}\n")
            except Exception as exc:
                self._append_log(
                    "[WARN] Analyzer cache could not be removed after process exit: "
                    f"{path}. Close Python/Phy/MATLAB handles and delete it manually. "
                    f"Original error: {exc}\n"
                )

    def _remove_tree_with_retry(self, path: Path, *, retries: int = 8, delay: float = 1.0) -> None:
        for attempt in range(retries):
            try:
                shutil.rmtree(path)
                return
            except PermissionError:
                if attempt >= retries - 1:
                    raise
                time.sleep(delay)

    def _kill_process_tree(self) -> None:
        process = self._process
        if process is None:
            return
        pid = int(process.processId())
        if os.name == "nt" and pid > 0:
            QProcess.startDetached("taskkill", ["/PID", str(pid), "/T", "/F"])
        elif pid > 0:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except OSError:
                process.kill()
        else:
            process.kill()

    def _process_error_occurred(self, error: QProcess.ProcessError) -> None:
        self._force_stop_requested = False
        self._set_running(False)
        if self._process_config_path is not None:
            self._process_config_path.unlink(missing_ok=True)
            self._process_config_path = None
        self._process = None
        QMessageBox.critical(self, "Run failed", f"Pipeline process failed to start: {error.name}")

    def closeEvent(self, event: Any) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(
                self,
                "Run active",
                "A pipeline job is still running. Wait for it to finish or use Force stop to terminate the run.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _set_running(self, running: bool) -> None:
        for button in [
            self.run_all,
            self.run_pre,
            self.run_post,
            self.run_noise_label,
            self.run_behavior_cleanup,
            self.run_behavior,
            self.move_outputs,
        ]:
            button.setEnabled(not running)
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
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
