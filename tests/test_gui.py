"""Headless smoke test of the GUI (Qt offscreen platform)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from seisgama import segy  # noqa: E402
from seisgama.synthetic import survey  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    from seisgama.gui.app import MainWindow

    shown = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, _n=name, **k: shown.append((_n, a[2] if len(a) > 2 else ""))))
    w = MainWindow(synchronous=True)
    w.messages = shown
    yield w
    w.close()


def test_full_workflow(window, tmp_path, monkeypatch):
    data, h = survey(nshots=4, nrec=24, ns=500, noise=0.02)
    data[10] = 0
    p = tmp_path / "line.sgy"
    segy.write_segy(p, data, 0.002, h, format_code=1)
    w = window
    w.open_file(str(p))
    assert w.seg is not None and w.seg.n_traces == 96
    assert w.viewer.raw.shape == (24, 500)

    # processing flow in the viewer
    w.viewer.flow.set_steps([{"op": "bandpass", "f1": 3, "f2": 8, "f3": 60, "f4": 80}, {"op": "agc", "window": 0.3}])
    assert not np.allclose(w.viewer.processed, w.viewer.raw)
    for i in range(w.viewer.views.count()):
        w.viewer.views.setCurrentIndex(i)
    for mode in ("wiggle", "both", "density"):
        w.viewer.mode.setCurrentText(mode)
    w.viewer.xaxis.setCurrentText("offset")
    w.viewer.selector.step(1)
    assert w.viewer.selector.value.currentText() == "1002"
    w.viewer.selector.key.setCurrentText("all")
    assert len(w.viewer.idx) == 96

    # QC finds the dead trace and jumps to it
    w.run_qc()
    assert w.qc_tab.result is not None
    assert w.qc_tab.result.flags["dead"][10]
    w.qc_tab.jump.emit(10)
    assert w.viewer.selector.value.currentText() == "1001"

    # velocity analysis on a CDP, manual + auto picks
    v = w.velocity
    v.selector.select("cdp", 20)
    v.compute()
    assert v.semb is not None and v.semb.shape[1] == 500
    v.auto_pick()
    v.picks = [(0.3, 1600.0), (0.6, 1900.0), (0.95, 2300.0)]
    v.redraw()

    # brute stack with picks, then export
    w.run_stack()
    cdps, st, fold = w.stack_tab.result
    assert st.shape == (len(np.unique(h["cdp"])), 500) and fold.max() >= 2
    out = tmp_path / "stack.sgy"
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    w.stack_tab.export()
    s = segy.open_segy(out)
    assert s.n_traces == len(cdps)
    s.close()

    # export processed SEG-Y with the viewer flow
    out2 = tmp_path / "proc.sgy"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out2), "")))
    w.export_processed()
    s = segy.open_segy(out2)
    assert s.n_traces == 96
    np.testing.assert_array_equal(s.headers["fldr"], h["fldr"])
    s.close()
    assert not [m for m in w.messages if m[0] == "critical"], w.messages


def test_open_bad_file_shows_error(window, tmp_path):
    p = tmp_path / "bad.sgy"
    p.write_bytes(b"\xff" * 4000)
    window.open_file(str(p))
    assert window.seg is None
    assert window.messages and window.messages[-1][0] == "critical"


def test_background_thread_open(app, tmp_path):
    """The real (threaded) path: open a file and wait for the worker."""
    import time

    from seisgama.gui.app import MainWindow

    data, h = survey(nshots=1, nrec=12, ns=200)
    p = tmp_path / "t.sgy"
    segy.write_segy(p, data, 0.002, h)
    w = MainWindow()
    w.open_file(str(p))
    t0 = time.time()
    while w.seg is None and time.time() - t0 < 10:
        app.processEvents()
        time.sleep(0.01)
    assert w.seg is not None and w.seg.n_traces == 12
    w.close()
