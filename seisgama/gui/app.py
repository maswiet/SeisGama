"""SEISGAMA desktop application (PySide6 / Qt 6 + matplotlib).

Runs on Windows, macOS and Linux.  Heavy work (opening files, QC, brute stack,
SEG-Y export) runs in a background thread so the window stays responsive.
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFileSystemModel, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QTreeView, QVBoxLayout, QWidget,
)

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import CITATION, __version__  # noqa: E402
from .. import processing as P  # noqa: E402
from .. import qc as QC  # noqa: E402
from ..plotting import (SEISMIC_CMAPS, plot_fk, plot_fx, plot_qc_overview, plot_section,  # noqa: E402
                        plot_semblance, plot_spectrum)
from ..segy import (TRACE_HEADER_FIELDS, SegyFile, binary_header_table, empty_trace_headers,  # noqa: E402
                    open_segy, write_segy)
from ..workflow import (brute_stack, load_flow, process_file, read_velocity_picks, save_flow,  # noqa: E402
                        write_stack, write_velocity_picks)

SEGY_FILTER = "SEG-Y files (*.sgy *.segy *.SGY *.SEGY);;All files (*)"
GATHER_KEYS = ("fldr", "cdp", "ep", "tracf", "iline", "xline", "all")
MAX_DISPLAY_SAMPLES = 40_000_000  # traces x samples shown at once before decimating


# --------------------------------------------------------------------------
# Background tasks
# --------------------------------------------------------------------------

class _Worker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int)

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.cancelled = False

    def run(self):
        try:
            if self.kwargs.pop("_with_progress", False):
                self.kwargs["progress"] = self._report
            self.finished.emit(self.fn(*self.args, **self.kwargs))
        except InterruptedError:
            self.failed.emit("Cancelled.")
        except Exception as exc:  # report to the user instead of crashing the app
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=4)}")

    def _report(self, pct):
        self.progress.emit(int(pct))
        return self.cancelled


class FigurePanel(QWidget):
    """A matplotlib figure with the standard zoom/pan toolbar."""

    def __init__(self, parent=None, figsize=(8, 6)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()


class TraceHeaderModel(QAbstractTableModel):
    """Virtual table over the structured header array (fast for millions of traces)."""

    def __init__(self, headers: np.ndarray | None = None):
        super().__init__()
        self.headers = headers if headers is not None else empty_trace_headers(0)
        self.names = list(self.headers.dtype.names)

    def set_headers(self, headers):
        self.beginResetModel()
        self.headers = headers
        self.names = list(headers.dtype.names)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.names)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return str(self.headers[self.names[index.column()]][index.row()])
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self.names[section] if orientation == Qt.Horizontal else str(section + 1)
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal:
            name, byte, typ, desc = TRACE_HEADER_FIELDS[section]
            return f"{desc} (bytes {byte}-{byte + np.dtype(typ).itemsize - 1})"
        return None


def _table(rows, headers) -> QTableWidget:
    t = QTableWidget(len(rows), len(headers))
    t.setHorizontalHeaderLabels(headers)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            item = QTableWidgetItem(str(v))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            t.setItem(r, c, item)
    t.horizontalHeader().setStretchLastSection(True)
    t.verticalHeader().setVisible(False)
    t.resizeColumnsToContents()
    return t


def _dspin(value, lo, hi, step=1.0, decimals=3) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    return s


# --------------------------------------------------------------------------
# Processing flow editor
# --------------------------------------------------------------------------

class FlowEditor(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Processing flow (applied to the displayed gather)", parent)
        self.steps: list[dict] = []
        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(self._show_params)
        self.list.itemChanged.connect(self._toggled)
        self.add_combo = QComboBox()
        for key, proc in P.PROCESSES.items():
            self.add_combo.addItem(proc.title, key)
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove)
        up = QPushButton("Up")
        up.clicked.connect(lambda: self._move(-1))
        down = QPushButton("Down")
        down.clicked.connect(lambda: self._move(1))
        load = QPushButton("Load…")
        load.clicked.connect(self._load)
        save = QPushButton("Save…")
        save.clicked.connect(self._save)
        self.form_box = QWidget()
        self.form = QFormLayout(self.form_box)
        self.form.setContentsMargins(0, 0, 0, 0)

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(self.add_combo, 1)
        row.addWidget(add)
        lay.addLayout(row)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for b in (rm, up, down, load, save):
            row.addWidget(b)
        lay.addLayout(row)
        lay.addWidget(self.form_box)

    def set_steps(self, steps):
        self.steps = [dict(s) for s in steps]
        self._refresh()
        self.changed.emit()

    def active_steps(self):
        return [s for s in self.steps if s.get("enabled", True)]

    def _refresh(self, select=None):
        self.list.blockSignals(True)
        self.list.clear()
        for s in self.steps:
            proc = P.PROCESSES[s["op"]]
            args = ", ".join(f"{p.name}={s.get(p.name, p.default)}" for p in proc.params)
            item = QListWidgetItem(f"{proc.title}" + (f"  ({args})" if args else ""))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if s.get("enabled", True) else Qt.Unchecked)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if select is not None and 0 <= select < len(self.steps):
            self.list.setCurrentRow(select)
        else:
            self._show_params(-1)

    def _add(self):
        key = self.add_combo.currentData()
        proc = P.PROCESSES[key]
        self.steps.append({"op": key, **{p.name: p.default for p in proc.params}})
        self._refresh(len(self.steps) - 1)
        self.changed.emit()

    def _remove(self):
        i = self.list.currentRow()
        if 0 <= i < len(self.steps):
            del self.steps[i]
            self._refresh(min(i, len(self.steps) - 1))
            self.changed.emit()

    def _move(self, d):
        i = self.list.currentRow()
        j = i + d
        if 0 <= i < len(self.steps) and 0 <= j < len(self.steps):
            self.steps[i], self.steps[j] = self.steps[j], self.steps[i]
            self._refresh(j)
            self.changed.emit()

    def _toggled(self, item):
        i = self.list.row(item)
        self.steps[i]["enabled"] = item.checkState() == Qt.Checked
        self.changed.emit()

    def _show_params(self, i):
        while self.form.rowCount():
            self.form.removeRow(0)
        if not 0 <= i < len(self.steps):
            return
        step = self.steps[i]
        proc = P.PROCESSES[step["op"]]
        if proc.help:
            self.form.addRow(QLabel(proc.help))
        for p in proc.params:
            val = step.get(p.name, p.default)
            if p.kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(val))
                w.toggled.connect(lambda v, n=p.name: self._set(n, bool(v)))
            elif p.kind == "choice":
                w = QComboBox()
                w.addItems(list(p.choices))
                w.setCurrentText(str(val))
                w.currentTextChanged.connect(lambda v, n=p.name: self._set(n, v))
            elif p.kind == "picks":
                w = QLineEdit(str(val))
                w.setToolTip("Comma separated pairs, e.g. 0:0.05, 1000:0.45")
                w.editingFinished.connect(lambda w=w, n=p.name: self._set(n, w.text()))
            elif p.kind == "int":
                w = QSpinBox()
                w.setRange(-10**9, 10**9)
                w.setValue(int(val))
                w.editingFinished.connect(lambda w=w, n=p.name: self._set(n, w.value()))
            else:
                w = _dspin(float(val), -1e9, 1e9, 0.1, 4)
                w.editingFinished.connect(lambda w=w, n=p.name: self._set(n, w.value()))
            self.form.addRow(p.label, w)

    def _set(self, name, value):
        i = self.list.currentRow()
        if 0 <= i < len(self.steps) and self.steps[i].get(name) != value:
            self.steps[i][name] = value
            self._refresh(i)
            self.changed.emit()

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load processing flow", "", "Flow (*.json)")
        if path:
            try:
                flow = load_flow(path)
                for s in flow:
                    if s["op"] not in P.PROCESSES:
                        raise ValueError(f"unknown step {s['op']}")
                self.set_steps(flow)
            except Exception as exc:
                QMessageBox.warning(self, "Load flow", str(exc))

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save processing flow", "flow.json", "Flow (*.json)")
        if path:
            save_flow(path, self.steps)


# --------------------------------------------------------------------------
# Gather selector (shared widget)
# --------------------------------------------------------------------------

class GatherSelector(QWidget):
    changed = Signal()

    def __init__(self, default_key="fldr", parent=None):
        super().__init__(parent)
        self.seg: SegyFile | None = None
        self.key = QComboBox()
        self.key.addItems(GATHER_KEYS)
        self.key.setCurrentText(default_key)
        self.value = QComboBox()
        self.value.setMinimumContentsLength(8)
        prev = QPushButton("◀")
        nxt = QPushButton("▶")
        prev.setFixedWidth(32)
        nxt.setFixedWidth(32)
        prev.clicked.connect(lambda: self.step(-1))
        nxt.clicked.connect(lambda: self.step(1))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Gather"))
        lay.addWidget(self.key)
        lay.addWidget(prev)
        lay.addWidget(self.value, 1)
        lay.addWidget(nxt)
        self.key.currentTextChanged.connect(self._key_changed)
        self.value.currentIndexChanged.connect(lambda _: self.changed.emit())

    def set_file(self, seg):
        self.seg = seg
        self._key_changed(self.key.currentText(), emit=False)

    def _key_changed(self, key, emit=True):
        self.value.blockSignals(True)
        self.value.clear()
        if self.seg is not None:
            if key == "all":
                self.value.addItem("all traces", None)
            else:
                for v in self.seg.gather_values(key):
                    self.value.addItem(str(v), int(v))
        self.value.blockSignals(False)
        self.value.setEnabled(key != "all")
        if emit:
            self.changed.emit()

    def step(self, d):
        i = self.value.currentIndex() + d
        if 0 <= i < self.value.count():
            self.value.setCurrentIndex(i)

    def select(self, key, value):
        self.key.setCurrentText(key)
        i = self.value.findData(int(value))
        if i >= 0:
            self.value.setCurrentIndex(i)

    def indices(self) -> np.ndarray:
        if self.seg is None:
            return np.array([], dtype=int)
        key = self.key.currentText()
        if key == "all":
            idx = np.arange(self.seg.n_traces)
        else:
            v = self.value.currentData()
            idx = self.seg.gather_index(key, v) if v is not None else np.array([], dtype=int)
        return idx

    def label(self) -> str:
        key = self.key.currentText()
        return "all traces" if key == "all" else f"{key} = {self.value.currentText()}"


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

class HeaderTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        self.text.setFont(font)
        self.binary = QWidget()
        QVBoxLayout(self.binary)
        self.model = TraceHeaderModel()
        self.trace_view = QTableView()
        self.trace_view.setModel(self.model)
        self.trace_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.tabs.addTab(self.summary, "Summary")
        self.tabs.addTab(self.text, "Textual header")
        self.tabs.addTab(self.binary, "Binary header")
        self.tabs.addTab(self.trace_view, "Trace headers")
        QVBoxLayout(self).addWidget(self.tabs)

    def set_file(self, seg: SegyFile):
        self.text.setPlainText(seg.text_header + ("\n\n" + "\n\n".join(seg.extended_text) if seg.extended_text else ""))
        lay = self.binary.layout()
        while lay.count():
            lay.takeAt(0).widget().deleteLater()
        lay.addWidget(_table(binary_header_table(seg), ["Name", "Bytes", "Value", "Description"]))
        self.model.set_headers(seg.headers)
        lines = [f"{k:24s}: {v}" for k, v in seg.summary().items()]
        if seg.warnings:
            lines += ["", "Warnings while reading:"] + [f"  - {w}" for w in seg.warnings]
        self.summary.setPlainText("\n".join(lines))


class ViewerTab(QWidget):
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seg: SegyFile | None = None
        self.raw = None
        self.processed = None
        self.idx = np.array([], dtype=int)
        self.xs = None
        self._plotted_x = None

        self.selector = GatherSelector("fldr")
        self.selector.changed.connect(self.load_gather)
        self.xaxis = QComboBox()
        self.xaxis.addItems(["trace", "offset", "tracf", "cdp", "fldr"])
        self.mode = QComboBox()
        self.mode.addItems(["density", "wiggle", "both"])
        self.cmap = QComboBox()
        self.cmap.addItems(SEISMIC_CMAPS)
        self.clip = _dspin(99.0, 50.0, 100.0, 0.5, 1)
        for w in (self.xaxis, self.mode, self.cmap):
            w.currentIndexChanged.connect(lambda _: self.replot())
        self.clip.editingFinished.connect(self.replot)
        self.flow = FlowEditor()
        self.flow.changed.connect(self.apply_flow)

        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.addWidget(self.selector)
        f = QFormLayout()
        f.addRow("X axis", self.xaxis)
        f.addRow("Display", self.mode)
        f.addRow("Colormap", self.cmap)
        f.addRow("Clip percentile", self.clip)
        cl.addLayout(f)
        cl.addWidget(self.flow)
        cl.addStretch(1)
        controls.setMinimumWidth(330)

        self.section = FigurePanel()
        self.spectrum = FigurePanel(figsize=(6, 4))
        self.fk = FigurePanel(figsize=(6, 4))
        self.fx = FigurePanel(figsize=(6, 4))
        self.views = QTabWidget()
        self.views.addTab(self.section, "Section")
        self.views.addTab(self.spectrum, "Amplitude spectrum")
        self.views.addTab(self.fx, "F-X")
        self.views.addTab(self.fk, "F-K")
        self.views.currentChanged.connect(lambda _: self.replot())
        self.section.canvas.mpl_connect("motion_notify_event", self._on_move)

        split = QSplitter()
        split.addWidget(controls)
        split.addWidget(self.views)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def set_file(self, seg):
        self.seg = seg
        self.selector.set_file(seg)
        self.load_gather()

    def load_gather(self):
        if self.seg is None:
            return
        idx = self.selector.indices()
        if len(idx) * self.seg.ns > MAX_DISPLAY_SAMPLES:
            step = int(np.ceil(len(idx) * self.seg.ns / MAX_DISPLAY_SAMPLES))
            idx = idx[::step]
            self.status.emit(f"Large selection: showing every {step}th trace.")
        self.idx = idx
        self.raw = self.seg.read_traces(idx) if len(idx) else np.zeros((0, self.seg.ns), np.float32)
        self.apply_flow()

    def offsets(self):
        return self.seg.offsets(self.idx)

    def t0(self):
        return self.seg.delay_of(self.idx)

    def apply_flow(self):
        if self.raw is None:
            return
        try:
            self.processed = P.apply_flow(self.raw, self.seg.dt, self.offsets(), self.flow.active_steps(),
                                          self.t0()) if len(self.raw) else self.raw
        except Exception as exc:
            self.processed = self.raw
            QMessageBox.warning(self, "Processing", f"Flow could not be applied:\n{exc}")
        self.replot()

    def _x(self):
        mode = self.xaxis.currentText()
        if mode == "trace" or self.seg is None:
            return np.arange(1, len(self.idx) + 1, dtype=float), "Trace (in gather)"
        if mode == "offset":
            return self.offsets(), "Offset"
        return self.seg.headers[mode][self.idx].astype(float), mode

    def replot(self):
        if self.processed is None or self.seg is None:
            return
        data, dt, t0 = self.processed, self.seg.dt, self.t0()
        title = f"{os.path.basename(self.seg.path)} — {self.selector.label()} ({len(data)} traces)"
        view = self.views.currentWidget()
        if len(data) == 0:
            return
        if view is self.section:
            x, xl = self._x()
            fig = self.section.figure
            fig.clear()
            ax = fig.add_subplot(111)
            im = plot_section(ax, data, dt, t0, x=x, mode=self.mode.currentText(),
                              clip_percentile=self.clip.value(), cmap=self.cmap.currentText(),
                              xlabel=xl, title=title)
            if im is not None:
                fig.colorbar(im, ax=ax, shrink=0.6, label="Amplitude")
            order = np.argsort(x, kind="stable") if np.any(np.diff(x) < 0) else np.arange(len(x))
            self._plotted_x, self._plotted_order = x[order], order
            self.section.draw()
        elif view is self.spectrum:
            fig = self.spectrum.figure
            fig.clear()
            ax = fig.add_subplot(111)
            f, a = P.amplitude_spectrum(data, dt)
            plot_spectrum(ax, f, a, title=f"Average amplitude spectrum — {self.selector.label()}")
            if len(self.raw) and self.flow.active_steps():
                f0, a0 = P.amplitude_spectrum(self.raw, dt)
                a0db = 20 * np.log10(a0 / max(a0.max(), 1e-30) + 1e-12)
                ax.plot(f0, a0db, color="0.6", lw=1, label="before flow")
                ax.plot([], [], color="tab:blue", label="after flow")
                ax.legend()
            self.spectrum.draw()
        elif view is self.fx:
            fig = self.fx.figure
            fig.clear()
            ax = fig.add_subplot(111)
            f, a = P.fx_spectrum(data, dt)
            im = plot_fx(ax, np.arange(1, len(data) + 1), f, a)
            fig.colorbar(im, ax=ax, label="dB")
            self.fx.draw()
        elif view is self.fk:
            fig = self.fk.figure
            fig.clear()
            ax = fig.add_subplot(111)
            dx = P.trace_spacing(self.offsets())
            if dx <= 0:
                ax.text(0.5, 0.5, "F-K needs trace spacing (offsets are all equal)", ha="center",
                        transform=ax.transAxes)
            else:
                f, k, a = P.fk_spectrum(data, dt, dx)
                im = plot_fk(ax, f, k, a, title=f"F-K spectrum (dx = {dx:g} m)")
                fig.colorbar(im, ax=ax, label="dB")
            self.fk.draw()

    def _on_move(self, ev):
        if ev.inaxes is None or self.processed is None or self._plotted_x is None or not len(self.processed):
            return
        x = self._plotted_x
        i = int(np.clip(np.searchsorted(x, ev.xdata), 0, len(x) - 1))
        if i > 0 and abs(x[i - 1] - ev.xdata) < abs(x[i] - ev.xdata):
            i -= 1
        tr = int(self._plotted_order[i])
        j = int(round((ev.ydata - self.t0()) / self.seg.dt))
        if 0 <= j < self.seg.ns:
            h = self.seg.headers
            g = self.idx[tr]
            self.status.emit(f"t = {ev.ydata:.4f} s | trace {g + 1} (fldr {h['fldr'][g]}, tracf {h['tracf'][g]}, "
                             f"cdp {h['cdp'][g]}, offset {h['offset'][g]}) | amplitude {self.processed[tr, j]:.5g}")


class VelocityTab(QWidget):
    def __init__(self, viewer: ViewerTab, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.seg = None
        self.picks: list[tuple[float, float]] = []
        self.data = self.off = self.semb = self.vels = None
        self.t0 = 0.0

        self.selector = GatherSelector("cdp")
        self.vmin = _dspin(1400, 100, 20000, 100, 0)
        self.vmax = _dspin(4000, 100, 20000, 100, 0)
        self.dv = _dspin(25, 1, 1000, 5, 0)
        self.window = _dspin(0.04, 0.004, 1.0, 0.004, 3)
        self.stretch = _dspin(0.5, 0.05, 5.0, 0.05, 2)
        self.use_flow = QCheckBox("Apply viewer processing flow first")
        self.use_flow.setChecked(True)
        compute = QPushButton("Compute semblance")
        compute.clicked.connect(self.compute)
        auto = QPushButton("Auto pick")
        auto.clicked.connect(self.auto_pick)
        clear = QPushButton("Clear picks")
        clear.clicked.connect(self.clear_picks)
        save = QPushButton("Save picks…")
        save.clicked.connect(self.save_picks)
        load = QPushButton("Load picks…")
        load.clicked.connect(self.load_picks)
        self.pick_label = QLabel()
        self.pick_label.setWordWrap(True)

        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.addWidget(self.selector)
        f = QFormLayout()
        f.addRow("Min velocity (m/s)", self.vmin)
        f.addRow("Max velocity (m/s)", self.vmax)
        f.addRow("Velocity step (m/s)", self.dv)
        f.addRow("Semblance window (s)", self.window)
        f.addRow("NMO stretch mute", self.stretch)
        cl.addLayout(f)
        cl.addWidget(self.use_flow)
        cl.addWidget(compute)
        row = QHBoxLayout()
        row.addWidget(auto)
        row.addWidget(clear)
        cl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(load)
        row.addWidget(save)
        cl.addLayout(row)
        cl.addWidget(QLabel("Left-click on the semblance to add a pick,\nright-click to delete the nearest pick."))
        cl.addWidget(self.pick_label)
        cl.addStretch(1)
        controls.setMinimumWidth(300)

        self.panel = FigurePanel(figsize=(12, 7))
        self.panel.canvas.mpl_connect("button_press_event", self._on_click)
        split = QSplitter()
        split.addWidget(controls)
        split.addWidget(self.panel)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def set_file(self, seg):
        self.seg = seg
        if len(np.unique(seg.headers["cdp"])) <= 1:
            self.selector.key.setCurrentText("fldr")
        self.selector.set_file(seg)
        self.data = self.semb = None
        self.picks = []
        self.pick_label.setText("")
        self.panel.figure.clear()
        self.panel.draw()

    def compute(self):
        if self.seg is None:
            return
        idx = self.selector.indices()
        if len(idx) < 2:
            QMessageBox.information(self, "Velocity analysis", "Select a gather with at least 2 traces.")
            return
        seg = self.seg
        self.off = seg.offsets(idx)
        self.t0 = seg.delay_of(idx)
        data = seg.read_traces(idx)
        if self.use_flow.isChecked():
            data = P.apply_flow(data, seg.dt, self.off, self.viewer.flow.active_steps(), self.t0)
        self.data = data
        if self.vmax.value() <= self.vmin.value():
            QMessageBox.warning(self, "Velocity analysis", "Max velocity must be larger than min velocity.")
            return
        self.vels = np.arange(self.vmin.value(), self.vmax.value() + self.dv.value() / 2, self.dv.value())
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.semb = P.semblance(data, seg.dt, self.off, self.vels, self.window.value(), self.t0)
        finally:
            QApplication.restoreOverrideCursor()
        self.redraw()

    def auto_pick(self):
        if self.semb is None:
            self.compute()
        if self.semb is not None:
            self.picks = P.pick_semblance_maxima(self.semb, self.vels, self.seg.dt, self.t0, step=0.15,
                                                 min_semblance=0.2)
            self.redraw()

    def clear_picks(self):
        self.picks = []
        self.redraw()

    def velocity(self):
        if not self.picks:
            return None
        pt, pv = np.array(sorted(self.picks)).T
        return pt, pv

    def save_picks(self):
        if not self.picks:
            QMessageBox.information(self, "Save picks", "No picks yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save velocity picks", "velocity_picks.csv", "CSV (*.csv)")
        if path:
            write_velocity_picks(path, self.picks)

    def load_picks(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load velocity picks", "", "CSV/Text (*.csv *.txt);;All (*)")
        if path:
            try:
                t, v = read_velocity_picks(path)
                self.picks = list(zip(t.tolist(), v.tolist()))
                self.redraw()
            except Exception as exc:
                QMessageBox.warning(self, "Load picks", str(exc))

    def _on_click(self, ev):
        if self.semb is None or ev.inaxes is None or self.panel.toolbar.mode or ev.inaxes is not self._ax_semb:
            return
        if ev.button == 1:
            self.picks.append((float(ev.ydata), float(ev.xdata)))
        elif ev.button == 3 and self.picks:
            p = np.array(self.picks)
            vr = max(self.vels[-1] - self.vels[0], 1.0)
            tr = max(self.seg.ns * self.seg.dt, 1e-6)
            d = ((p[:, 0] - ev.ydata) / tr) ** 2 + ((p[:, 1] - ev.xdata) / vr) ** 2
            self.picks.pop(int(np.argmin(d)))
        self.redraw()

    def redraw(self):
        fig = self.panel.figure
        fig.clear()
        if self.data is None or self.semb is None:
            self.panel.draw()
            return
        ax1, ax2, ax3 = fig.subplots(1, 3, sharey=True)
        seg = self.seg
        plot_section(ax1, self.data, seg.dt, self.t0, x=self.off, xlabel="Offset (m)",
                     title=f"Gather ({self.selector.label()})")
        plot_semblance(ax2, self.semb, self.vels, seg.dt, self.t0, self.picks)
        self._ax_semb = ax2
        if self.picks:
            nmo = P.nmo_correct(self.data, seg.dt, self.off, self.velocity(), self.stretch.value(), self.t0)
            plot_section(ax3, nmo, seg.dt, self.t0, x=self.off, xlabel="Offset (m)", title="NMO corrected")
        else:
            ax3.set_title("NMO corrected (pick velocities first)")
        ax2.set_ylabel("")
        ax3.set_ylabel("")
        self.pick_label.setText("Picks (t0 s : Vrms m/s)\n" + "\n".join(f"{t:.3f} : {v:.0f}" for t, v in sorted(self.picks)))
        self.panel.draw()


class QCTab(QWidget):
    jump = Signal(int)
    run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seg = None
        self.result: QC.QCResult | None = None
        self.key = QComboBox()
        self.key.addItems(["fldr", "cdp", "ep", "iline"])
        self.z = _dspin(3.5, 1.0, 20.0, 0.5, 1)
        self.ratio = _dspin(2.0, 1.1, 100.0, 0.5, 1)
        self.spike = _dspin(12.0, 3.0, 1000.0, 1.0, 1)
        run = QPushButton("Run QC")
        run.clicked.connect(self.run_requested.emit)
        export = QPushButton("Export report…")
        export.clicked.connect(self.export)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        self.text.setFont(font)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Trace", "fldr", "tracf", "offset", "RMS", "Problems"])
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellDoubleClicked.connect(lambda r, c: self.jump.emit(int(self.table.item(r, 0).text()) - 1))
        self.panel = FigurePanel(figsize=(10, 8))

        controls = QWidget()
        cl = QVBoxLayout(controls)
        f = QFormLayout()
        f.addRow("Gather key", self.key)
        f.addRow("Noisy/weak robust z", self.z)
        f.addRow("… and amplitude ratio ≥", self.ratio)
        f.addRow("Spike ratio (vs. neighbours)", self.spike)
        cl.addLayout(f)
        cl.addWidget(run)
        cl.addWidget(export)
        cl.addWidget(QLabel("Flagged traces (double-click to view):"))
        cl.addWidget(self.table, 1)
        controls.setMinimumWidth(380)
        right = QTabWidget()
        right.addTab(self.panel, "Overview plots")
        right.addTab(self.text, "Report")
        split = QSplitter()
        split.addWidget(controls)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def settings(self):
        return QC.QCSettings(gather_key=self.key.currentText(), z_threshold=self.z.value(),
                             min_amplitude_ratio=self.ratio.value(), spike_ratio=self.spike.value())

    def set_file(self, seg):
        self.seg = seg
        self.result = None
        self.text.clear()
        self.table.setRowCount(0)
        self.panel.figure.clear()
        self.panel.draw()

    def show_result(self, result: QC.QCResult):
        self.result = result
        self.text.setPlainText(QC.report_text(self.seg, result))
        idx = result.flagged_indices()
        h = self.seg.headers
        self.table.setRowCount(min(len(idx), 20000))  # full list is in the report/CSV
        for r, i in enumerate(idx[:20000]):
            for c, v in enumerate([i + 1, h["fldr"][i], h["tracf"][i], h["offset"][i],
                                   f"{result.attributes['rms'][i]:.4g}", ", ".join(result.reasons(i))]):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        plot_qc_overview(self.panel.figure, self.seg, result)
        self.panel.draw()

    def export(self):
        if self.result is None:
            QMessageBox.information(self, "QC", "Run QC first.")
            return
        base = os.path.splitext(os.path.basename(self.seg.path))[0]
        d = QFileDialog.getExistingDirectory(self, "Folder for QC report")
        if not d:
            return
        with open(os.path.join(d, f"{base}_qc_report.txt"), "w", encoding="utf-8") as fh:
            fh.write(QC.report_text(self.seg, self.result))
        QC.write_csv(self.seg, self.result, os.path.join(d, f"{base}_qc_traces.csv"))
        self.panel.figure.savefig(os.path.join(d, f"{base}_qc_overview.png"), dpi=120)
        QMessageBox.information(self, "QC", f"Report, CSV and overview plot written to\n{d}")


class StackTab(QWidget):
    run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seg = None
        self.result = None
        self.source = QComboBox()
        self.source.addItems(["Picks from velocity analysis", "Constant velocity"])
        self.const_v = _dspin(2000, 100, 20000, 100, 0)
        self.stretch = _dspin(0.5, 0.05, 5.0, 0.05, 2)
        self.use_flow = QCheckBox("Apply viewer processing flow before NMO")
        self.use_flow.setChecked(True)
        self.read_key = QComboBox()
        self.read_key.addItems(["fldr", "cdp", "ep"])
        run = QPushButton("Compute brute stack")
        run.clicked.connect(self.run_requested.emit)
        save = QPushButton("Export stack as SEG-Y…")
        save.clicked.connect(self.export)
        self.panel = FigurePanel(figsize=(12, 7))
        controls = QWidget()
        cl = QVBoxLayout(controls)
        f = QFormLayout()
        f.addRow("Velocity", self.source)
        f.addRow("Constant velocity (m/s)", self.const_v)
        f.addRow("NMO stretch mute", self.stretch)
        f.addRow("Read data by", self.read_key)
        cl.addLayout(f)
        cl.addWidget(self.use_flow)
        cl.addWidget(run)
        cl.addWidget(save)
        cl.addWidget(QLabel("Brute stack = flow + NMO with one velocity\nfunction + CMP (cdp) stack. For QC only."))
        cl.addStretch(1)
        controls.setMinimumWidth(300)
        split = QSplitter()
        split.addWidget(controls)
        split.addWidget(self.panel)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def set_file(self, seg):
        self.seg = seg
        self.result = None
        self.panel.figure.clear()
        self.panel.draw()

    def show_result(self, result):
        self.result = result
        cdps, st, fold = result
        fig = self.panel.figure
        fig.clear()
        ax = fig.add_subplot(111)
        plot_section(ax, st, self.seg.dt, self.seg.delay, x=cdps.astype(float), xlabel="CDP",
                     title=f"Brute stack — {os.path.basename(self.seg.path)} (max fold {fold.max()})")
        self.panel.draw()

    def export(self):
        if self.result is None:
            QMessageBox.information(self, "Stack", "Compute the stack first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export stack", "brute_stack.sgy", SEGY_FILTER)
        if path:
            cdps, st, fold = self.result
            try:
                write_stack(self.seg, path, cdps, st, fold)
            except ValueError as exc:
                QMessageBox.warning(self, "Export stack", str(exc))


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, synchronous: bool = False):
        super().__init__()
        self.synchronous = synchronous  # run tasks inline (used by tests)
        self.seg: SegyFile | None = None
        self._thread = None
        self._worker = None
        self._on_done = None
        self.setWindowTitle(f"SEISGAMA {__version__} — Seismic QC")
        self.resize(1400, 900)

        self.headers_tab = HeaderTab()
        self.viewer = ViewerTab()
        self.velocity = VelocityTab(self.viewer)
        self.qc_tab = QCTab()
        self.stack_tab = StackTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.headers_tab, "Headers")
        self.tabs.addTab(self.viewer, "Seismic viewer")
        self.tabs.addTab(self.qc_tab, "QC")
        self.tabs.addTab(self.velocity, "Velocity analysis")
        self.tabs.addTab(self.stack_tab, "Brute stack")
        self.setCentralWidget(self.tabs)
        for t in (self.headers_tab, self.viewer, self.velocity, self.qc_tab, self.stack_tab):
            t.setEnabled(False)

        self.viewer.status.connect(lambda m: self.statusBar().showMessage(m))
        self.qc_tab.run_requested.connect(self.run_qc)
        self.qc_tab.jump.connect(self.jump_to_trace)
        self.stack_tab.run_requested.connect(self.run_stack)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self._cancel)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)

        self._build_browser()
        self._build_menu()
        self.statusBar().showMessage("Open a SEG-Y file (File ▸ Open) or generate a synthetic demo line.")

    # ------------------------------------------------------------------ UI
    def _build_browser(self):
        self.fs = QFileSystemModel()
        self.fs.setNameFilters(["*.sgy", "*.segy", "*.SGY", "*.SEGY"])
        self.fs.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.fs)
        for c in range(1, 4):
            self.tree.hideColumn(c)
        self.tree.setHeaderHidden(True)
        self._set_browser_root(os.path.expanduser("~"))
        self.tree.doubleClicked.connect(lambda ix: (not self.fs.isDir(ix)) and self.open_file(self.fs.filePath(ix)))
        dock = QDockWidget("Files", self)
        dock.setWidget(self.tree)
        dock.setObjectName("files")
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.files_dock = dock

    def _set_browser_root(self, folder):
        self.fs.setRootPath(folder)
        self.tree.setRootIndex(self.fs.index(folder))
        self.files_dock_title = folder
        if hasattr(self, "files_dock"):
            self.files_dock.setWindowTitle(f"Files — {folder}")

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        a = QAction("&Open SEG-Y…", self)
        a.setShortcut(QKeySequence.Open)
        a.triggered.connect(self.open_dialog)
        m.addAction(a)
        a = QAction("Open with header overrides…", self)
        a.triggered.connect(self.open_with_overrides)
        m.addAction(a)
        a = QAction("Export processed SEG-Y (viewer flow, all traces)…", self)
        a.triggered.connect(self.export_processed)
        m.addAction(a)
        a = QAction("Save current section image…", self)
        a.triggered.connect(self.save_image)
        m.addAction(a)
        m.addSeparator()
        a = QAction("Generate synthetic demo line…", self)
        a.triggered.connect(self.make_demo)
        m.addAction(a)
        m.addSeparator()
        a = QAction("&Quit", self)
        a.setShortcut(QKeySequence.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)
        v = self.menuBar().addMenu("&View")
        v.addAction(self.files_dock.toggleViewAction())
        h = self.menuBar().addMenu("&Help")
        a = QAction("About SEISGAMA", self)
        a.triggered.connect(self.about)
        h.addAction(a)

    # ------------------------------------------------------------------ tasks
    def run_task(self, fn, *args, on_done=None, label="Working…", with_progress=False, **kwargs):
        """Run ``fn`` in a background thread and call ``on_done(result)`` in the GUI thread."""
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "Another task is still running.")
            return
        if with_progress:
            kwargs["_with_progress"] = True
        if self.synchronous:
            if kwargs.pop("_with_progress", False):
                kwargs["progress"] = lambda p: False
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self._task_failed(f"{type(exc).__name__}: {exc}")
                return
            if on_done:
                on_done(result)
            return
        self.statusBar().showMessage(label)
        self.progress.setRange(0, 100 if with_progress else 0)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_btn.setVisible(with_progress)
        self._thread = QThread()
        self._worker = _Worker(fn, args, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._on_done = on_done
        # connect to bound methods of this (GUI-thread) object so Qt queues the
        # calls into the GUI thread; a plain lambda would run in the worker thread
        self._worker.finished.connect(self._task_done)
        self._worker.failed.connect(self._task_failed)
        self._thread.start()

    def _cleanup_task(self):
        self.progress.hide()
        self.cancel_btn.hide()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None

    @Slot(object)
    def _task_done(self, result):
        on_done, self._on_done = self._on_done, None
        self._cleanup_task()
        self.statusBar().showMessage("Done.", 5000)
        if on_done:
            on_done(result)

    @Slot(str)
    def _task_failed(self, msg):
        self._cleanup_task()
        self.statusBar().showMessage("Failed.", 5000)
        QMessageBox.critical(self, "SEISGAMA", msg)

    def _cancel(self):
        if self._worker is not None:
            self._worker.cancelled = True

    # ------------------------------------------------------------------ actions
    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open SEG-Y", "", SEGY_FILTER)
        if path:
            self.open_file(path)

    def open_with_overrides(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open SEG-Y", "", SEGY_FILTER)
        if not path:
            return
        from PySide6.QtWidgets import QDialog, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Header overrides (0 = use file value)")
        form = QFormLayout(dlg)
        endian = QComboBox()
        endian.addItems(["auto", "big-endian", "little-endian"])
        fmt = QSpinBox()
        fmt.setRange(0, 16)
        ns = QSpinBox()
        ns.setRange(0, 1_000_000)
        dt = _dspin(0, 0, 100000, 250, 1)
        form.addRow("Byte order", endian)
        form.addRow("Sample format code", fmt)
        form.addRow("Samples per trace", ns)
        form.addRow("Sample interval (µs)", dt)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec():
            opts = {
                "endian": {"auto": None, "big-endian": ">", "little-endian": "<"}[endian.currentText()],
                "format_code": fmt.value() or None,
                "ns": ns.value() or None,
                "dt_us": dt.value() or None,
            }
            self.open_file(path, **opts)

    def open_file(self, path, **opts):
        self.run_task(open_segy, path, on_done=self._file_opened, label=f"Reading {os.path.basename(path)}…", **opts)

    def _file_opened(self, seg: SegyFile):
        if self.seg is not None:
            self.seg.close()
        self.seg = seg
        self.setWindowTitle(f"SEISGAMA {__version__} — {seg.path}")
        if os.path.exists(seg.path):
            self._set_browser_root(os.path.dirname(os.path.abspath(seg.path)))
            self.tree.setCurrentIndex(self.fs.index(os.path.abspath(seg.path)))
        for t in (self.headers_tab, self.viewer, self.velocity, self.qc_tab, self.stack_tab):
            t.setEnabled(True)
            t.set_file(seg)
        self.tabs.setCurrentWidget(self.viewer)
        msg = f"{seg.n_traces} traces, {seg.ns} samples, dt = {seg.dt * 1000:g} ms, {seg.format_name}"
        if seg.warnings:
            msg += f" — {len(seg.warnings)} header warning(s), see Headers ▸ Summary"
        self.statusBar().showMessage(msg)

    def run_qc(self):
        if self.seg is None:
            return
        self.run_task(QC.run_qc, self.seg, self.qc_tab.settings(), on_done=self.qc_tab.show_result,
                      label="Running QC…", with_progress=True)

    def run_stack(self):
        if self.seg is None:
            return
        tab = self.stack_tab
        if tab.source.currentIndex() == 0:
            vel = self.velocity.velocity()
            if vel is None:
                QMessageBox.information(self, "Brute stack", "No velocity picks yet: pick velocities in the "
                                                             "Velocity analysis tab or choose a constant velocity.")
                return
        else:
            vel = tab.const_v.value()
        if len(np.unique(self.seg.headers["cdp"])) <= 1:
            QMessageBox.warning(self, "Brute stack", "All traces have the same CDP number: the cdp header is not "
                                                     "set, so a CMP stack cannot be formed.")
            return
        flow = self.viewer.flow.active_steps() if tab.use_flow.isChecked() else []
        self.run_task(brute_stack, self.seg, flow, vel, tab.stretch.value(), tab.read_key.currentText(),
                      on_done=tab.show_result, label="Stacking…", with_progress=True)

    def export_processed(self):
        if self.seg is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export processed SEG-Y", "processed.sgy", SEGY_FILTER)
        if not path:
            return
        if os.path.exists(path) and os.path.samefile(path, self.seg.path):
            QMessageBox.warning(self, "Export", "Choose a different file name than the input file.")
            return
        key = self.viewer.selector.key.currentText()
        key = "fldr" if key == "all" else key
        self.run_task(process_file, self.seg, path, self.viewer.flow.active_steps(), key,
                      on_done=lambda n: QMessageBox.information(self, "Export", f"{n} traces written to\n{path}"),
                      label="Exporting…", with_progress=True)

    def save_image(self):
        w = self.tabs.currentWidget()
        panel = {self.viewer: lambda: self.viewer.views.currentWidget(), self.velocity: lambda: self.velocity.panel,
                 self.qc_tab: lambda: self.qc_tab.panel, self.stack_tab: lambda: self.stack_tab.panel}.get(w)
        if panel is None:
            QMessageBox.information(self, "Save image", "Switch to a tab with a plot first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "figure.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            panel().figure.savefig(path, dpi=150)

    def make_demo(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save synthetic line", "synthetic_line.sgy", SEGY_FILTER)
        if not path:
            return
        if self.seg is not None and os.path.exists(path) and os.path.samefile(path, self.seg.path):
            QMessageBox.warning(self, "Synthetic line", "That file is currently open; choose another name.")
            return
        from ..synthetic import survey

        data, h = survey(nshots=8, nrec=48, noise=0.03)
        data[100] = 0  # plant a few problems so the QC has something to find
        data[150] *= 30
        write_segy(path, data, 0.002, h, format_code=1, text_header=(
            "C 1 SEISGAMA SYNTHETIC DEMO LINE\nC 2 Reflectors t0/Vrms: 0.30/1600 0.60/1900 0.95/2300 1.30/2700\n"
            "C 3 Planted problems: trace 101 dead, trace 151 noisy"))
        self.open_file(path)

    def jump_to_trace(self, i):
        if self.seg is None:
            return
        self.tabs.setCurrentWidget(self.viewer)
        self.viewer.selector.select("fldr", int(self.seg.headers["fldr"][i]))

    def about(self):
        QMessageBox.about(self, "About SEISGAMA",
                          f"<b>SEISGAMA {__version__}</b><br>Seismic reflection QC and basic processing "
                          "for land and marine data.<br><br>SEG-Y I/O, header inspection, trace QC, gain/AGC, "
                          "band-pass, notch, mute, Wiener deconvolution, spectra (F-X, F-K), semblance velocity "
                          "analysis, NMO and brute stack.<br><br>"
                          "<b>Please cite:</b> " + CITATION + "<br><br>"
                          "License: CC BY-NC 4.0 (non-commercial). Commercial use requires permission "
                          "from the authors (ws@ugm.ac.id).<br>"
                          "https://github.com/maswiet/SeisGama<br><br>Python " + sys.version.split()[0])

    def closeEvent(self, ev):
        if self._thread is not None:
            self._cancel()
            try:  # no result dialogs after the window is gone
                self._worker.finished.disconnect()
                self._worker.failed.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._thread.quit()
            self._thread.wait()  # tasks check the cancel flag regularly
        if self.seg is not None:
            self.seg.close()
        super().closeEvent(ev)


def run(path: str | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SEISGAMA")

    def excepthook(etype, value, tb):  # show unexpected errors instead of silently dying
        msg = "".join(traceback.format_exception(etype, value, tb))
        sys.stderr.write(msg)
        QMessageBox.critical(None, "SEISGAMA - unexpected error", msg[-3000:])

    sys.excepthook = excepthook
    win = MainWindow()
    win.show()
    if path:
        win.open_file(path)
    return app.exec()


def main() -> int:  # entry point for the seisgama-gui launcher
    return run(sys.argv[1] if len(sys.argv) > 1 else None)
