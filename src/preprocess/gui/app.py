from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Callable

import numpy as np
from scipy.io import loadmat

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
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
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.postprocess import PostprocessConfig, run_postprocess_session
from src.preprocess import PreprocessConfig, prepare_chanmap, run_preprocess_session, select_paths_with_gui

from .config_model import (
    PipelineGuiSettings,
    PostprocessGuiSettings,
    PreprocessGuiSettings,
    RunMode,
    parse_float_pair,
    parse_int_list,
)
from .preflight import CheckResult, run_preflight


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


class ChanMapCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(5.5, 4.0), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.summary = QLabel("No chanMap loaded")
        self.summary.setWordWrap(True)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.summary)
        self.show_empty()

    def show_empty(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, "Generate or load chanMap.mat", ha="center", va="center")
        ax.set_axis_off()
        self.canvas.draw_idle()
        self.summary.setText("No chanMap loaded")

    def load_chanmap(self, path: Path) -> None:
        if not path.exists():
            self.show_empty()
            self.summary.setText(f"chanMap not found: {path}")
            return
        data = loadmat(path)
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
        colors = probe_ids * 1000 + kcoords
        ax.scatter(x[connected], y[connected], c=colors[connected], cmap="tab20", s=42, edgecolor="black", linewidth=0.3)
        if np.any(~connected):
            ax.scatter(x[~connected], y[~connected], c="none", edgecolor="red", marker="x", s=70, linewidth=1.8)

        if n <= 256:
            for xi, yi, ch, is_connected in zip(x, y, device_ch, connected):
                color = "#222222" if is_connected else "#b00020"
                ax.text(float(xi), float(yi), str(int(ch)), fontsize=7, ha="center", va="bottom", color=color)

        ax.set_title(path.name)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()

        bad = device_ch[~connected].astype(int).tolist()
        probes = sorted(set(int(v) for v in probe_ids.tolist()))
        shanks = sorted(set(int(v) for v in kcoords.tolist()))
        self.summary.setText(
            f"{path}\n"
            f"channels={n}, connected={int(np.sum(connected))}, bad={len(bad)}, "
            f"probes={len(probes)}, groups/shanks={len(shanks)}\n"
            f"bad channels: {bad[:40]}{' ...' if len(bad) > 40 else ''}"
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PreprocessPipeline GUI")
        self.resize(1320, 860)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._last_preprocess_result: Any | None = None
        self._build_ui()
        self._apply_settings(PipelineGuiSettings())
        self._refresh_preview()

    def _build_ui(self) -> None:
        root = QWidget()
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
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        layout.addWidget(splitter, 1)

        layout.addWidget(self._build_run_bar())
        layout.addWidget(self.log)

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
        self.local_root.setPlaceholderText("Local output root")
        browse_local = QPushButton("Browse local")
        browse_local.clicked.connect(self._browse_local_root)

        load_config = QPushButton("Load config")
        load_config.clicked.connect(self._load_config)
        save_config = QPushButton("Save config")
        save_config.clicked.connect(self._save_config)

        layout.addWidget(browse_basepath, 0, 0)
        layout.addWidget(self.basepath, 0, 1)
        layout.addWidget(QLabel("Local root"), 0, 2)
        layout.addWidget(self.local_root, 0, 3)
        layout.addWidget(browse_local, 0, 4)
        layout.addWidget(load_config, 0, 5)
        layout.addWidget(save_config, 0, 6)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(3, 2)

        self.basepath.textChanged.connect(self._refresh_preview)
        self.local_root.textChanged.connect(self._refresh_preview)
        return panel

    def _build_settings_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_preprocess_tab(), "Preprocess setting")
        self.tabs.addTab(self._build_postprocess_tab(), "Postprocess setting")
        self.tabs.setMinimumWidth(500)
        return self.tabs

    def _build_preprocess_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        basic = QGroupBox("Core")
        form = QFormLayout(basic)
        self.analog_inputs = QCheckBox("Export analog inputs")
        self.digital_inputs = QCheckBox("Export digital inputs")
        self.save_raw = QCheckBox("Save raw dat")
        self.do_preprocess = QCheckBox("Bandpass + reference")
        self.bandpass_min = self._double_spin(0.1, 100000.0, 500.0)
        self.bandpass_max = self._double_spin(0.1, 100000.0, 8000.0)
        self.reference = QComboBox()
        self.reference.addItems(["local", "global"])
        self.local_radius = QLineEdit("20, 200")
        self.make_lfp = QCheckBox("Make LFP")
        self.lfp_fs = self._double_spin(1.0, 100000.0, 1250.0)
        self.state_score = QCheckBox("State scoring")
        form.addRow(self.analog_inputs)
        form.addRow(self.digital_inputs)
        form.addRow(self.save_raw)
        form.addRow(self.do_preprocess)
        form.addRow("Bandpass min Hz", self.bandpass_min)
        form.addRow("Bandpass max Hz", self.bandpass_max)
        form.addRow("Reference", self.reference)
        form.addRow("Local radius um", self.local_radius)
        form.addRow(self.make_lfp)
        form.addRow("LFP fs", self.lfp_fs)
        form.addRow(self.state_score)

        artifacts = QGroupBox("Artifacts")
        af = QFormLayout(artifacts)
        self.ttl_group = QComboBox()
        self.ttl_group.addItems(["none", "all", "probe", "shank"])
        self.ttl_channel = self._spin(0, 15, 0)
        self.ttl_include_offset = QCheckBox("Include TTL offset")
        self.ttl_before = self._double_spin(0.0, 1000.0, 0.5)
        self.ttl_after = self._double_spin(0.0, 1000.0, 2.0)
        self.ttl_mode = QComboBox()
        self.ttl_mode.addItems(["linear", "cubic", "0"])
        self.highamp_group = QComboBox()
        self.highamp_group.addItems(["none", "all", "probe", "shank"])
        self.highamp_sigma = self._double_spin(0.1, 1000.0, 5.0)
        self.highamp_before = self._double_spin(0.0, 1000.0, 2.0)
        self.highamp_after = self._double_spin(0.0, 1000.0, 2.0)
        self.highamp_mode = QComboBox()
        self.highamp_mode.addItems(["linear", "cubic", "0"])
        af.addRow("TTL group mode", self.ttl_group)
        af.addRow("TTL channel", self.ttl_channel)
        af.addRow(self.ttl_include_offset)
        af.addRow("TTL ms before", self.ttl_before)
        af.addRow("TTL ms after", self.ttl_after)
        af.addRow("TTL mode", self.ttl_mode)
        af.addRow("High amp group mode", self.highamp_group)
        af.addRow("High amp sigma", self.highamp_sigma)
        af.addRow("High amp ms before", self.highamp_before)
        af.addRow("High amp ms after", self.highamp_after)
        af.addRow("High amp mode", self.highamp_mode)

        chanmap = QGroupBox("chanMap")
        cf = QFormLayout(chanmap)
        self.reject_channels = QLineEdit()
        self.reject_channels.setPlaceholderText("0-based, comma separated")
        self.probe_assignments = QTextEdit()
        self.probe_assignments.setMinimumHeight(110)
        self.chanmap_path = QLineEdit()
        load_chanmap = QPushButton("Load chanMap")
        load_chanmap.clicked.connect(self._browse_chanmap)
        generate_chanmap = QPushButton("Generate chanMap")
        generate_chanmap.clicked.connect(self._generate_chanmap)
        chanmap_buttons = QHBoxLayout()
        chanmap_buttons.addWidget(load_chanmap)
        chanmap_buttons.addWidget(generate_chanmap)
        cf.addRow("Reject channels", self.reject_channels)
        cf.addRow("Probe assignments JSON", self.probe_assignments)
        cf.addRow("chanMap path", self.chanmap_path)
        cf.addRow(chanmap_buttons)

        sorter = QGroupBox("Sorter")
        sf = QFormLayout(sorter)
        self.sorter = QComboBox()
        self.sorter.addItems(["Kilosort", "Kilosort2_5", "kilosort4", "disabled"])
        self.sorter.currentTextChanged.connect(self._sorter_changed)
        self.sorter_path = QLineEdit()
        self.sorter_config_path = QLineEdit()
        self.matlab_path = QLineEdit()
        self.preprocess_worker_count = self._spin(1, 4096, 1)
        self.sorter_worker_count = self._spin(1, 4096, 1)
        self.pre_overwrite = QCheckBox("Overwrite preprocess outputs")
        sf.addRow("Sorter", self.sorter)
        sf.addRow("Sorter path", self.sorter_path)
        sf.addRow("Sorter config", self.sorter_config_path)
        sf.addRow("MATLAB path", self.matlab_path)
        sf.addRow("Workers for preprocess", self.preprocess_worker_count)
        sf.addRow("Workers for sorter", self.sorter_worker_count)
        sf.addRow(self.pre_overwrite)

        for widget in [
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
            self.ttl_group,
            self.ttl_channel,
            self.ttl_include_offset,
            self.ttl_before,
            self.ttl_after,
            self.ttl_mode,
            self.highamp_group,
            self.highamp_sigma,
            self.highamp_before,
            self.highamp_after,
            self.highamp_mode,
            self.reject_channels,
            self.probe_assignments,
            self.chanmap_path,
            self.sorter_path,
            self.sorter_config_path,
            self.matlab_path,
            self.preprocess_worker_count,
            self.sorter_worker_count,
            self.pre_overwrite,
        ]:
            self._connect_refresh(widget)

        layout.addWidget(basic)
        layout.addWidget(artifacts)
        layout.addWidget(chanmap)
        layout.addWidget(sorter)
        layout.addStretch(1)
        return panel

    def _build_postprocess_tab(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        settings = QGroupBox("Postprocess")
        form = QFormLayout(settings)
        self.sorting_phy_folder = QLineEdit()
        self.sorting_search_root = QLineEdit()
        browse_phy = QPushButton("Browse sorting folder")
        browse_phy.clicked.connect(self._browse_sorting_folder)
        browse_search = QPushButton("Browse search root")
        browse_search.clicked.connect(self._browse_sorting_search_root)
        self.post_apply_preprocess = QCheckBox("Apply preprocess before metrics")
        self.exclude_groups = QLineEdit("noise")
        self.duplicate_censored = self._double_spin(0.0, 1000.0, 0.5)
        self.duplicate_threshold = self._double_spin(0.0, 1.0, 0.5)
        self.merge_min_spikes = self._spin(0, 1000000, 100)
        self.merge_corr = self._double_spin(0.0, 10.0, 0.25)
        self.merge_template = self._double_spin(0.0, 10.0, 0.25)
        self.split_contamination = self._double_spin(0.0, 1.0, 0.05)
        self.split_threshold_mode = QComboBox()
        self.split_threshold_mode.addItems(["adaptive_chi2", "chi2", "quantile"])
        self.split_wf_threshold = self._double_spin(0.0, 10.0, 0.2)
        self.split_wf_n_chans = self._spin(1, 4096, 10)
        self.split_amp_mad_scale = self._double_spin(0.1, 1000.0, 10.0)
        self.skip_pc_metrics = QCheckBox("Skip PC metrics")
        self.post_overwrite = QCheckBox("Overwrite postprocess outputs")
        self.post_worker_count = self._spin(1, 4096, 1)

        form.addRow("Sorting folder", self._field_with_button(self.sorting_phy_folder, browse_phy))
        form.addRow("Search root", self._field_with_button(self.sorting_search_root, browse_search))
        form.addRow(self.post_apply_preprocess)
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
        form.addRow(self.skip_pc_metrics)
        form.addRow(self.post_overwrite)
        form.addRow("Workers", self.post_worker_count)

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
            self.post_overwrite,
            self.post_worker_count,
        ]:
            self._connect_refresh(widget)

        layout.addWidget(settings)
        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.chanmap_canvas = ChanMapCanvas()
        self.run_preview = QPlainTextEdit()
        self.run_preview.setReadOnly(True)
        self.run_preview.setMinimumHeight(180)
        self.run_preview.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.chanmap_canvas, 2)
        layout.addWidget(QLabel("Run preview / preflight"))
        layout.addWidget(self.run_preview, 1)
        return panel

    def _build_run_bar(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.run_all = QPushButton("Run all process")
        self.run_pre = QPushButton("Run preprocess only")
        self.run_post = QPushButton("Run postprocess only")
        self.clear_log = QPushButton("Clear log")
        self.run_all.clicked.connect(lambda: self._start_run("all"))
        self.run_pre.clicked.connect(lambda: self._start_run("preprocess"))
        self.run_post.clicked.connect(lambda: self._start_run("postprocess"))
        self.clear_log.clicked.connect(self.log.clear)
        layout.addStretch(1)
        layout.addWidget(self.run_all)
        layout.addWidget(self.run_pre)
        layout.addWidget(self.run_post)
        layout.addWidget(self.clear_log)
        return panel

    def _double_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setValue(value)
        return box

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        box = QSpinBox()
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

    def _connect_refresh(self, widget: QWidget) -> None:
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._refresh_preview)
        elif isinstance(widget, QTextEdit):
            widget.textChanged.connect(self._refresh_preview)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._refresh_preview)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self._refresh_preview)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._refresh_preview)

    def _browse_basepath(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select basepath", self.basepath.text() or str(Path.cwd()))
        if path:
            self.basepath.setText(path)

    def _browse_local_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select local output root", self.local_root.text() or str(Path.cwd()))
        if path:
            self.local_root.setText(path)

    def _browse_chanmap(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select chanMap.mat", self.local_root.text() or str(Path.cwd()), "MAT files (*.mat);;All files (*)")
        if path:
            self.chanmap_path.setText(path)
            self.chanmap_canvas.load_chanmap(Path(path))

    def _browse_sorting_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select sorting folder", self.sorting_phy_folder.text() or self.local_root.text() or str(Path.cwd()))
        if path:
            self.sorting_phy_folder.setText(path)

    def _browse_sorting_search_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select sorting search root", self.sorting_search_root.text() or self.local_root.text() or str(Path.cwd()))
        if path:
            self.sorting_search_root.setText(path)

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load GUI config", str(Path.cwd()), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            self._apply_settings(PipelineGuiSettings.load(Path(path)))
            self._append_log(f"Loaded config: {path}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Load config failed", str(exc))

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save GUI config", str(Path.cwd() / "preprocess_gui_config.json"), "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            self._collect_settings().save(Path(path))
            self._append_log(f"Saved config: {path}\n")
        except Exception as exc:
            QMessageBox.critical(self, "Save config failed", str(exc))

    def _sorter_changed(self, sorter: str) -> None:
        defaults = {
            "Kilosort": ("sorter/KiloSort1", "sorter/Kilosort1_config.yaml"),
            "Kilosort2_5": ("sorter/Kilosort2.5", "sorter/Kilosort2.5_config.yaml"),
            "kilosort4": ("sorter/Kilosort4", "sorter/Kilosort4_config.yaml"),
            "disabled": ("", ""),
        }
        path, config = defaults.get(sorter, ("", ""))
        self.sorter_path.setText(path)
        self.sorter_config_path.setText(config)
        self._refresh_preview()

    def _collect_settings(self) -> PipelineGuiSettings:
        preprocess = PreprocessGuiSettings(
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
            artifact_ttl_group_mode=self.ttl_group.currentText(),
            artifact_ttl_channel=self.ttl_channel.value(),
            artifact_ttl_include_offset=self.ttl_include_offset.isChecked(),
            artifact_ttl_ms_before=self.ttl_before.value(),
            artifact_ttl_ms_after=self.ttl_after.value(),
            artifact_ttl_mode=self.ttl_mode.currentText(),
            artifact_highamp_group_mode=self.highamp_group.currentText(),
            highamp_threshold_sigma=self.highamp_sigma.value(),
            highamp_ms_before=self.highamp_before.value(),
            highamp_ms_after=self.highamp_after.value(),
            highamp_mode=self.highamp_mode.currentText(),
            reject_channels=parse_int_list(self.reject_channels.text()),
            probe_assignments=json.loads(self.probe_assignments.toPlainText() or "[]"),
            sorter=self.sorter.currentText(),
            sorter_path=self.sorter_path.text().strip(),
            sorter_config_path=self.sorter_config_path.text().strip(),
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
        self.basepath.setText(settings.basepath)
        self.local_root.setText(settings.local_root)
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
        self.ttl_group.setCurrentText(p.artifact_ttl_group_mode)
        self.ttl_channel.setValue(p.artifact_ttl_channel)
        self.ttl_include_offset.setChecked(p.artifact_ttl_include_offset)
        self.ttl_before.setValue(p.artifact_ttl_ms_before)
        self.ttl_after.setValue(p.artifact_ttl_ms_after)
        self.ttl_mode.setCurrentText(p.artifact_ttl_mode)
        self.highamp_group.setCurrentText(p.artifact_highamp_group_mode)
        self.highamp_sigma.setValue(p.highamp_threshold_sigma)
        self.highamp_before.setValue(p.highamp_ms_before)
        self.highamp_after.setValue(p.highamp_ms_after)
        self.highamp_mode.setCurrentText(p.highamp_mode)
        self.reject_channels.setText(", ".join(str(v) for v in p.reject_channels))
        self.probe_assignments.setPlainText(json.dumps(p.probe_assignments, indent=2))
        self.sorter.setCurrentText(p.sorter or "disabled")
        self.sorter_path.setText(p.sorter_path)
        self.sorter_config_path.setText(p.sorter_config_path)
        self.matlab_path.setText(p.matlab_path)
        self.preprocess_worker_count.setValue(p.preprocess_worker_count)
        self.sorter_worker_count.setValue(p.sorter_worker_count)
        self.pre_overwrite.setChecked(p.overwrite)
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
        self.post_overwrite.setChecked(pp.overwrite)
        self.post_worker_count.setValue(pp.worker_count)
        chanmap = settings.resolved_chanmap_path()
        if chanmap is not None and chanmap.exists():
            self.chanmap_canvas.load_chanmap(chanmap)

    def _refresh_preview(self) -> None:
        try:
            settings = self._collect_settings()
            checks = run_preflight(settings, "all")
            lines = [
                f"Basepath: {settings.basepath or '-'}",
                f"Basename: {settings.basename or '-'}",
                f"Local output: {settings.local_output_dir or '-'}",
                f"chanMap: {settings.resolved_chanmap_path() or '-'}",
                "",
                "Preflight:",
            ]
            lines.extend(self._format_checks(checks))
            self.run_preview.setPlainText("\n".join(lines))
            chanmap = settings.resolved_chanmap_path()
            if chanmap is not None and chanmap.exists():
                self.chanmap_canvas.load_chanmap(chanmap)
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
            self.chanmap_canvas.load_chanmap(chanmap_path)
            self._append_log(f"Generated chanMap: {chanmap_path}\nBad channels: {bad_channels}\n")
            self._refresh_preview()
        except Exception as exc:
            QMessageBox.critical(self, "Generate chanMap failed", str(exc))

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
                "Preflight warnings",
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
        worker.log.connect(self._append_log)
        worker.finished.connect(lambda result: self._run_finished(result, thread, worker))
        worker.failed.connect(lambda text: self._run_failed(text, thread, worker))
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

        if mode in ("all", "postprocess"):
            if mode == "all" and pre_result is not None:
                post_config = self._postprocess_config_from_preprocess_result(settings, pre_result)
            else:
                post_config = settings.to_postprocess_config()
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

    def _run_finished(self, result: object, thread: QThread, worker: PipelineWorker) -> None:
        self._cleanup_thread(thread, worker)
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

    def _run_failed(self, text: str, thread: QThread, worker: PipelineWorker) -> None:
        self._cleanup_thread(thread, worker)
        self._set_running(False)
        self._append_log("\n=== Run failed ===\n")
        self._append_log(text)
        QMessageBox.critical(self, "Run failed", text.splitlines()[-1] if text.splitlines() else text)

    def _cleanup_thread(self, thread: QThread, worker: PipelineWorker) -> None:
        worker.deleteLater()
        thread.quit()
        thread.wait()
        thread.deleteLater()
        self._thread = None
        self._worker = None

    def _set_running(self, running: bool) -> None:
        for button in [self.run_all, self.run_pre, self.run_post]:
            button.setEnabled(not running)

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
