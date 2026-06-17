from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Callable

import numpy as np
from scipy.io import loadmat

from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from src.postprocess import PostprocessConfig, run_postprocess_session
from src.preprocess import prepare_chanmap, run_preprocess_session, select_paths_with_gui
from src.preprocess.io import build_channel_map_data, set_tree_world_rw
from src.preprocess.paths import find_project_root, resolve_project_path

from .config_model import (
    PipelineGuiSettings,
    PostprocessGuiSettings,
    PreprocessGuiSettings,
    RunMode,
    parse_float_pair,
    parse_int_list,
)
from .preflight import CheckResult, run_preflight


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

    return {
        "basepath": str(dst_root),
        "local_output_dir": str(src_root),
        "moved": moved,
        "skipped": skipped,
        "overwrite": overwrite,
        "move_dat": move_dat,
    }


class SignalWriter:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit

    def write(self, text: str) -> int:
        if text:
            self._emit(text)
        return len(text)

    def flush(self) -> None:
        return None


class PipelineWorker(QObject):
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        writer = SignalWriter(self.log.emit)
        try:
            with redirect_stdout(writer), redirect_stderr(writer):
                result = self._fn()
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


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
        group_keys = [(int(p), int(k)) for p, k in zip(probe_ids.tolist(), kcoords.tolist())]
        unique_groups = sorted(set(group_keys))
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
        color_by_group = {
            group: palette[idx % len(palette)] for idx, group in enumerate(unique_groups)
        }
        point_colors = np.array([color_by_group[group] for group in group_keys], dtype=object)
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PreprocessPipeline GUI")
        self.resize(1320, 860)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_preprocess_result: Any | None = None
        self._probe_rows: list[dict[str, QWidget]] = []
        self.noise_threshold_fields: dict[str, QLineEdit] = {}
        self._refresh_suspended = False
        self._last_chanmap_preview_key: tuple[Any, ...] | None = None
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
        return panel

    def _build_settings_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll_area(self._build_preprocess_tab()), "Preprocess setting")
        self.tabs.addTab(self._scroll_area(self._build_postprocess_tab()), "Postprocess setting")
        self.tabs.setMinimumWidth(500)
        self.tabs.currentChanged.connect(lambda _index: self._schedule_refresh())
        return self.tabs

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
        self.preprocess_worker_count = self._spin(1, 4096, 1)
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
        self.sorter_worker_count = self._spin(1, 4096, 1)
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
        self.post_worker_count = self._spin(1, 4096, 1)

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

        monitor_layout.addWidget(chanmap_panel, 3)
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
        self.run_all = QPushButton("Run all process")
        self.run_all.setObjectName("primaryButton")
        self.run_pre = QPushButton("Run preprocess only")
        self.run_post = QPushButton("Run postprocess only")
        self.move_dat_to_basepath = QCheckBox("Move basename.dat")
        self.move_overwrite = QCheckBox("Overwrite moved files")
        self.move_outputs = QPushButton("Move outputs to basepath")
        self.force_stop = QPushButton("Force stop")
        self.force_stop.setObjectName("dangerButton")
        self.clear_log = QPushButton("Clear log")
        self.run_all.clicked.connect(lambda: self._start_run("all"))
        self.run_pre.clicked.connect(lambda: self._start_run("preprocess"))
        self.run_post.clicked.connect(lambda: self._start_run("postprocess"))
        self.move_outputs.clicked.connect(self._move_outputs_to_basepath)
        self.force_stop.clicked.connect(self._force_stop_process)
        self.clear_log.clicked.connect(self.log.clear)
        layout.addWidget(self.run_all)
        layout.addWidget(self.run_pre)
        layout.addWidget(self.run_post)
        layout.addStretch(1)
        layout.addWidget(self.move_dat_to_basepath)
        layout.addWidget(self.move_overwrite)
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
        self._schedule_refresh()

    def _remove_probe_assignment_row(self, row_panel: QWidget) -> None:
        if len(self._probe_rows) <= 1:
            return
        self._probe_rows = [row for row in self._probe_rows if row["panel"] is not row_panel]
        row_panel.setParent(None)
        row_panel.deleteLater()
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

    def _browse_local_root(self) -> None:
        path = self._select_directory("Select local output root", self.local_root.text() or str(Path.cwd()))
        if path:
            self.local_root.setText(path)

    def _browse_chanmap(self) -> None:
        path = self._select_open_file(
            "Select chanMap.mat",
            self.local_root.text() or str(Path.cwd()),
            "MAT files (*.mat);;All files (*)",
        )
        if path:
            chanmap_path = Path(path)
            data = loadmat(chanmap_path)
            self._refresh_suspended = True
            try:
                self.chanmap_path.setText(path)
                self.reject_channels.setText(", ".join(str(v) for v in self._bad_channels_from_chanmap_data(data)))
                assignments = self._assignments_from_chanmap_data(data)
                if assignments:
                    self._render_probe_assignments(assignments)
            finally:
                self._refresh_suspended = False
            self._last_chanmap_preview_key = None
            self.chanmap_canvas.render_chanmap(data, source=chanmap_path)
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
            preprocess_worker_count=self.preprocess_worker_count.value(),
            sorter_worker_count=self.sorter_worker_count.value(),
            overwrite=self.pre_overwrite.isChecked(),
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
            worker_count=self.post_worker_count.value(),
        )
        return PipelineGuiSettings(
            basepath=self.basepath.text().strip(),
            local_root=self.local_root.text().strip(),
            chanmap_path=self.chanmap_path.text().strip(),
            preprocess=preprocess,
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
            self.run_sorter.setChecked(p.run_sorter and (p.sorter or "disabled") != "disabled")
            self.sorter.setCurrentText(p.sorter or "disabled")
            default_sorter_path, default_sorter_config = self._current_sorter_defaults()
            self.sorter_path.setText(p.sorter_path or default_sorter_path)
            self.sorter_config_path.setText(p.sorter_config_path or default_sorter_config)
            self.matlab_path.setText(p.matlab_path)
            self.preprocess_worker_count.setValue(p.preprocess_worker_count)
            self.sorter_worker_count.setValue(p.sorter_worker_count)
            self.pre_overwrite.setChecked(p.overwrite)
            self._update_sorter_enabled()
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
            self.post_worker_count.setValue(pp.worker_count)
            chanmap = settings.resolved_chanmap_path()
            if not self._load_settings_chanmap_preview(settings) and chanmap is not None and chanmap.exists():
                self._load_chanmap_preview(chanmap)
        finally:
            self._refresh_suspended = False

    def _refresh_preview(self) -> None:
        try:
            settings = self._collect_settings()
            mode: RunMode = "postprocess" if self.tabs.currentIndex() == 1 else "all"
            try:
                checks = run_preflight(settings, mode)
            except Exception as exc:
                checks = [CheckResult("Preflight", "warn", str(exc))]
            lines = [
                f"Basepath: {settings.basepath or '-'}",
                f"Basename: {settings.basename or '-'}",
                f"Local output: {settings.local_output_dir or '-'}",
                f"chanMap: {settings.resolved_chanmap_path() or '-'}",
                f"Preview mode: {mode}",
                "",
                "Preflight:",
            ]
            lines.extend(self._format_checks(checks))
            self.run_preview.setPlainText("\n".join(lines))
            chanmap = settings.resolved_chanmap_path()
            if not self._load_settings_chanmap_preview(settings) and chanmap is not None and chanmap.exists():
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
        if self._thread is not None:
            QMessageBox.warning(self, "Run active", "Cannot move outputs while a pipeline job is running.")
            return
        try:
            settings = self._collect_settings()
            message = (
                "Move local output files to basepath?\n\n"
                f"Basepath: {settings.basepath or '-'}\n"
                f"Local output: {settings.local_output_dir or '-'}\n"
                f"Move basename.dat: {'yes' if self.move_dat_to_basepath.isChecked() else 'no'}\n"
                f"Overwrite existing files: {'yes' if self.move_overwrite.isChecked() else 'no'}"
            )
            answer = QMessageBox.question(self, "Move outputs to basepath", message)
            if answer != QMessageBox.StandardButton.Yes:
                return
            result = _move_local_output_to_basepath(
                settings,
                move_dat=self.move_dat_to_basepath.isChecked(),
                overwrite=self.move_overwrite.isChecked(),
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
            lines.append("Move outputs to basepath complete!")
            text = "\n".join(lines)
            self.run_preview.setPlainText(text)
            self._append_log(text + "\n")
        except Exception as exc:
            QMessageBox.critical(self, "Move outputs failed", str(exc))

    def _force_stop_process(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            self._append_log("\n=== Force stop requested, but no pipeline job is running ===\n")
            return
        QMessageBox.warning(
            self,
            "Force stop",
            "The GUI will stay open.\n\n"
            "The current QThread runner cannot be safely force-killed without risking a GUI crash. "
            "Wait for the current step to finish, or close the terminal process if an emergency kill is required.",
        )
        self._append_log(
            "\n=== Force stop requested; GUI remains open. "
            "Safe in-GUI force kill requires subprocess-based execution. ===\n"
        )

    def _start_run(self, mode: RunMode) -> None:
        if self._thread is not None:
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

        self._set_running(True)
        self._append_log(f"\n=== Running {mode} ===\n")
        worker = PipelineWorker(lambda: self._run_pipeline(settings, mode))
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._run_finished, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._run_failed, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished, Qt.ConnectionType.QueuedConnection)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _run_pipeline(self, settings: PipelineGuiSettings, mode: RunMode) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": mode}
        pre_result = None
        if mode in ("all", "preprocess"):
            if settings.basepath_path is None:
                raise ValueError("basepath is required.")
            if settings.resolved_chanmap_path() is None or not settings.resolved_chanmap_path().exists():
                print("chanMap is missing; generating before preprocess.")
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
                print(f"Generated chanMap: {chanmap_path}")
                print(f"Bad channels: {bad_channels}")
                settings.chanmap_path = str(chanmap_path)
            pre_config = settings.to_preprocess_config()
            pre_result = run_preprocess_session(pre_config)
            payload["preprocess_result"] = pre_result

        if mode in ("all", "postprocess", "noise_label"):
            if mode == "all" and pre_result is not None:
                post_config = self._postprocess_config_from_preprocess_result(settings, pre_result)
            else:
                post_config = settings.to_postprocess_config()
            if mode == "noise_label":
                post_config.noise_label_only = True
            payload["postprocess_results"] = run_postprocess_session(post_config)

        return payload

    def _postprocess_config_from_preprocess_result(self, settings: PipelineGuiSettings, pre_result: Any) -> PostprocessConfig:
        post_config = settings.to_postprocess_config()
        post_config.sorting_phy_folder = pre_result.sorter_output_dir or post_config.sorting_phy_folder
        post_config.sorting_search_root = pre_result.local_output_dir
        post_config.dat_path = pre_result.dat_path
        post_config.sampling_frequency = pre_result.sr
        post_config.num_channels = pre_result.n_channels
        post_config.chanmap_mat_path = settings.resolved_chanmap_path()
        post_config.reject_channels = list(pre_result.bad_channels_0based)
        return post_config

    def _run_finished(self, result: object) -> None:
        self._set_running(False)
        if isinstance(result, dict):
            pre_result = result.get("preprocess_result")
            if pre_result is not None:
                self._last_preprocess_result = pre_result
                if getattr(pre_result, "sorter_output_dir", None):
                    self.sorting_phy_folder.setText(str(pre_result.sorter_output_dir))
            self._append_log("\n=== Run finished ===\n")
            self._append_result_summary(result)
        self._refresh_preview()

    def _run_failed(self, text: str) -> None:
        self._set_running(False)
        self._append_log("\n=== Run failed ===\n")
        self._append_log(text)
        QMessageBox.critical(self, "Run failed", text.splitlines()[-1] if text.splitlines() else text)

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    def closeEvent(self, event: Any) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(
                self,
                "Run active",
                "A pipeline job is still running. Wait for it to finish or use Force stop to terminate the run.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def _set_running(self, running: bool) -> None:
        for button in [self.run_all, self.run_pre, self.run_post, self.run_noise_label, self.move_outputs]:
            button.setEnabled(not running)
        self.force_stop.setEnabled(running)

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _append_result_summary(self, result: dict[str, Any]) -> None:
        pre = result.get("preprocess_result")
        if pre is not None:
            self._append_log(f"local_output_dir: {pre.local_output_dir}\n")
            self._append_log(f"dat_path: {pre.dat_path}\n")
            self._append_log(f"lfp_path: {pre.lfp_path}\n")
            self._append_log(f"sorter_output_dir: {pre.sorter_output_dir}\n")
        post_results = result.get("postprocess_results") or []
        for idx, post in enumerate(post_results, start=1):
            self._append_log(f"postprocess[{idx}] output_folder: {post.output_folder}\n")
            self._append_log(f"postprocess[{idx}] metrics_csv_path: {post.metrics_csv_path}\n")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
