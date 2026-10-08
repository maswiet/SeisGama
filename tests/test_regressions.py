"""Regression tests for issues found in the independent code review."""
import os

import numpy as np
import pytest

from seisgama import segy, workflow
from seisgama.synthetic import survey

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _segyio_file(path, ext_headers=0, n=4, ns=100):
    segyio = pytest.importorskip("segyio")
    spec = segyio.spec()
    spec.format = 5
    spec.samples = list(range(ns))
    spec.tracecount = n
    spec.ext_headers = ext_headers
    with segyio.create(str(path), spec) as f:
        for i in range(n):
            f.trace[i] = np.full(ns, i, np.float32)
            f.header[i] = {segyio.TraceField.TRACE_SAMPLE_COUNT: ns}
        f.bin.update(hdt=2000)


def test_extended_headers_with_revision_zero(tmp_path):
    p = tmp_path / "ext.sgy"
    _segyio_file(p, ext_headers=2)
    s = segy.open_segy(p)
    assert s.n_traces == 4 and len(s.extended_text) == 2
    np.testing.assert_array_equal(s.read_all()[:, 0], [0, 1, 2, 3])
    s.close()


def test_bogus_extended_header_count_is_ignored(tmp_path):
    p = tmp_path / "bogus.sgy"
    _segyio_file(p)
    raw = bytearray(p.read_bytes())
    raw[3500] = 1
    raw[3504:3506] = (200).to_bytes(2, "big")
    p.write_bytes(bytes(raw))
    s = segy.open_segy(p)
    assert s.n_traces == 4 and any("extended" in w for w in s.warnings)
    s.close()


def test_format_override_when_format_code_is_corrupt(tmp_path):
    p = tmp_path / "nofmt.sgy"
    _segyio_file(p)
    raw = bytearray(p.read_bytes())
    raw[3224:3226] = b"\0\0"
    p.write_bytes(bytes(raw))
    with pytest.raises(segy.SegyError):
        segy.open_segy(p)
    s = segy.open_segy(p, format_code=5)
    np.testing.assert_array_equal(s.read_all()[:, 0], [0, 1, 2, 3])
    s.close()


def test_ns_fallback_checks_second_trace_header(tmp_path):
    data, h = survey(nshots=1, nrec=4, ns=50)
    p = tmp_path / "ns.sgy"
    segy.write_segy(p, data, 0.002, h)
    raw = bytearray(p.read_bytes())
    raw[3220:3222] = (160).to_bytes(2, "big")  # 4*(240+200) = 2*(240+640): wrong ns also "fits"
    p.write_bytes(bytes(raw))
    s = segy.open_segy(p)
    assert s.ns == 50 and s.n_traces == 4
    s.close()


def test_stack_keeps_recording_delay(tmp_path):
    data, h = survey(nshots=2, nrec=12, ns=200)
    h["delrt"] = 100
    s = segy.from_arrays(data, 0.002, h)
    cdps, st, fold = workflow.brute_stack(s, [], 2000.0)
    out = tmp_path / "stack.sgy"
    workflow.write_stack(s, out, cdps, st, fold)
    r = segy.open_segy(out)
    assert r.delay == pytest.approx(0.1)
    r.close()


def test_stack_refuses_to_overwrite_input(tmp_path):
    data, h = survey(nshots=1, nrec=6, ns=100)
    p = tmp_path / "in.sgy"
    segy.write_segy(p, data, 0.002, h)
    s = segy.open_segy(p)
    cdps, st, fold = workflow.brute_stack(s, [], 2000.0)
    with pytest.raises(ValueError):
        workflow.write_stack(s, p, cdps, st, fold)
    s.close()


def test_process_uses_each_gathers_delay():
    data, h = survey(nshots=2, nrec=6, ns=100)
    h["delrt"][6:] = 40  # second shot recorded 40 ms later
    s = segy.from_arrays(np.ones_like(data), 0.002, h)
    out = workflow.P.apply_flow(s.read_traces(slice(6, 12)), s.dt, s.offsets(slice(6, 12)),
                                [{"op": "tpow", "power": 1.0}], s.delay_of(slice(6, 12)))
    assert out[0, 0] == pytest.approx(0.04)


def test_plot_with_all_equal_offsets_shows_every_trace():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from seisgama.plotting import plot_section

    fig, ax = plt.subplots()
    data = np.random.default_rng(0).normal(size=(12, 50))
    plot_section(ax, data, 0.002, x=np.zeros(12), xlabel="Offset")
    lo, hi = ax.get_xlim()
    assert hi - lo >= 11
    plt.close(fig)


def test_qc_trid_hydrophone_is_seismic():
    from seisgama import qc

    data, h = survey(nshots=1, nrec=10, ns=200)
    h["trid"] = 11
    h["trid"][3] = 2
    r = qc.run_qc(segy.from_arrays(data, 0.002, h))
    assert list(np.flatnonzero(r.flags["not_seismic"])) == [3]


def test_gui_clears_picks_and_guards_overwrite(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    app = QApplication.instance() or QApplication([])
    from seisgama.gui.app import MainWindow

    msgs = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, _n=name, **k: msgs.append(_n)))
    data, h = survey(nshots=2, nrec=12, ns=200)
    p1, p2 = tmp_path / "a.sgy", tmp_path / "b.sgy"
    segy.write_segy(p1, data, 0.002, h)
    segy.write_segy(p2, data, 0.002, h)
    w = MainWindow(synchronous=True)
    w.open_file(str(p1))
    w.velocity.picks = [(0.3, 1600.0)]
    w.stack_tab.source.setCurrentIndex(1)
    w.run_stack()
    size = p1.stat().st_size
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(p1), "")))
    w.stack_tab.export()  # must refuse: p1 is open
    w.make_demo()  # must refuse too
    assert p1.stat().st_size == size and msgs.count("warning") == 2
    w.open_file(str(p2))
    assert w.velocity.picks == []
    w.close()
    app.processEvents()
