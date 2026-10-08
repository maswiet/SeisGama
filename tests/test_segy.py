from pathlib import Path

import numpy as np
import pytest

from seisgama import segy
from seisgama.synthetic import survey

DATA = Path(__file__).parent / "data"
EXAMPLES = ["lineE.sgy", "bigEndianIEEEFloat.sgy"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_matches_segyio(name):
    segyio = pytest.importorskip("segyio")
    path = DATA / name
    with segy.open_segy(path) as s, segyio.open(path, ignore_geometry=True) as ref:
        assert s.n_traces == ref.tracecount
        assert s.ns == len(ref.samples)
        assert s.dt == pytest.approx(segyio.tools.dt(ref) / 1e6)
        np.testing.assert_array_equal(s.read_all(), segyio.tools.collect(ref.trace[:]))
        for name_, f in [("fldr", segyio.TraceField.FieldRecord), ("offset", segyio.TraceField.offset),
                         ("cdp", segyio.TraceField.CDP), ("sx", segyio.TraceField.SourceX),
                         ("gy", segyio.TraceField.GroupY), ("scalco", segyio.TraceField.SourceGroupScalar),
                         ("dt", segyio.TraceField.TRACE_SAMPLE_INTERVAL), ("delrt", segyio.TraceField.DelayRecordingTime)]:
            np.testing.assert_array_equal(s.headers[name_], ref.attributes(f)[:], err_msg=name_)
        assert s.warnings == []


def test_example_text_header_is_ebcdic():
    s = segy.open_segy(DATA / "lineE.sgy")
    assert s.text_encoding == "ebcdic"
    assert s.text_header.startswith("C 1 CLIENT")
    assert s.text_header.rstrip().endswith("END EBCDIC")
    assert s.format_code == 1
    s.close()


@pytest.mark.parametrize("values,expected", [(-118.625, 0xC276A000), (1.0, 0x41100000), (0.0, 0),
                                              (0.15625, 0x40280000), (-1.0, 0xC1100000)])
def test_ieee2ibm_known_values(values, expected):
    assert int(segy.ieee2ibm(np.array([values]))[0]) == expected
    assert segy.ibm2ieee(np.array([expected], dtype=np.uint32))[0] == np.float32(values)


def test_ibm_roundtrip_precision():
    rng = np.random.default_rng(1)
    x = rng.normal(size=20000) * 10.0 ** rng.uniform(-20, 20, 20000)
    y = segy.ibm2ieee(segy.ieee2ibm(x)).astype(np.float64)
    # IBM float has 24-bit mantissa with hex normalisation -> rel. error < 2^-20
    assert np.max(np.abs(y - x) / np.abs(x)) < 2.0**-20


@pytest.mark.parametrize("fmt", [1, 5])
def test_write_read_roundtrip(tmp_path, fmt):
    data, headers = survey(nshots=2, nrec=12, ns=301)
    p = tmp_path / "out.sgy"
    segy.write_segy(p, data, 0.002, headers, format_code=fmt, text_header="C 1 TEST LINE")
    s = segy.open_segy(p)
    assert s.n_traces == 24 and s.ns == 301 and s.dt == pytest.approx(0.002)
    assert s.format_code == fmt
    assert s.text_header.startswith("C 1 TEST LINE")
    rtol = 1e-6 if fmt == 1 else 0
    np.testing.assert_allclose(s.read_all(), data, rtol=rtol, atol=1e-6 * np.abs(data).max())
    for k in ("fldr", "tracf", "offset", "sx", "gx", "scalco", "selev", "gelev", "cdp"):
        np.testing.assert_array_equal(s.headers[k], headers[k])
    np.testing.assert_allclose(s.scaled("sx")[:2], [1000.0, 1000.0])
    s.close()


def test_write_from_template_preserves_binary_header(tmp_path):
    src = segy.open_segy(DATA / "lineE.sgy")
    p = tmp_path / "copy.sgy"
    segy.write_segy(p, src.read_all(), src.dt, src.headers, template=src, format_code=1)
    out = segy.open_segy(p)
    for k in ("jobid", "lino", "reno", "ntrpr", "hdt", "hns"):
        assert out.binary_header[k] == src.binary_header[k]
    np.testing.assert_array_equal(out.read_all(), src.read_all())
    out.close()
    src.close()


def _write_raw(path, text, binary, traces):
    with open(path, "wb") as fh:
        fh.write(text)
        fh.write(binary)
        for t in traces:
            fh.write(t)


def test_little_endian_and_ascii_header(tmp_path):
    ns, ntr = 50, 3
    text = "C 1 ASCII HEADER".ljust(3200).encode("ascii")
    b = bytearray(400)
    b[16:18] = (1000).to_bytes(2, "little")
    b[20:22] = ns.to_bytes(2, "little")
    b[24:26] = (5).to_bytes(2, "little")
    data = np.arange(ntr * ns, dtype=np.float32).reshape(ntr, ns)
    traces = []
    for i in range(ntr):
        h = np.zeros(1, dtype=segy.trace_header_dtype("<"))
        h["fldr"] = 7
        h["tracf"] = i + 1
        h["ns"] = ns
        h["dt"] = 1000
        traces.append(h.tobytes() + data[i].astype("<f4").tobytes())
    p = tmp_path / "le.sgy"
    _write_raw(p, text, bytes(b), traces)
    s = segy.open_segy(p)
    assert s.endian == "<" and s.text_encoding == "ascii"
    assert s.text_header.startswith("C 1 ASCII HEADER")
    np.testing.assert_array_equal(s.read_all(), data)
    np.testing.assert_array_equal(s.headers["tracf"], [1, 2, 3])
    assert any("little-endian" in w for w in s.warnings)
    s.close()


def test_recovers_from_wrong_binary_ns_and_zero_dt(tmp_path):
    data, headers = survey(nshots=1, nrec=4, ns=100)
    p = tmp_path / "bad.sgy"
    segy.write_segy(p, data, 0.002, headers)
    raw = bytearray(p.read_bytes())
    raw[3216:3218] = b"\0\0"  # dt = 0
    raw[3220:3222] = (999).to_bytes(2, "big")  # wrong ns
    p.write_bytes(bytes(raw))
    s = segy.open_segy(p)
    assert s.ns == 100 and s.dt == pytest.approx(0.002)
    assert len(s.warnings) >= 2
    np.testing.assert_allclose(s.read_all(), data)
    s.close()


def test_truncated_last_trace(tmp_path):
    data, headers = survey(nshots=1, nrec=5, ns=100)
    p = tmp_path / "trunc.sgy"
    segy.write_segy(p, data, 0.002, headers)
    raw = p.read_bytes()
    p.write_bytes(raw[:-50])
    s = segy.open_segy(p)
    assert s.n_traces == 4
    assert any("trailing" in w for w in s.warnings)
    s.close()


def test_extended_text_headers(tmp_path):
    data, headers = survey(nshots=1, nrec=3, ns=64)
    p = tmp_path / "ext.sgy"
    segy.write_segy(p, data, 0.002, headers)
    raw = bytearray(p.read_bytes())
    raw[3504:3506] = (1).to_bytes(2, "big")
    ext = segy.encode_text_header("((SEG: EndText))")
    p.write_bytes(bytes(raw[:3600]) + ext + bytes(raw[3600:]))
    s = segy.open_segy(p)
    assert len(s.extended_text) == 1 and s.n_traces == 3
    np.testing.assert_allclose(s.read_all(), data)
    s.close()


def test_not_segy(tmp_path):
    p = tmp_path / "junk.sgy"
    p.write_bytes(b"\xff" * 5000)
    with pytest.raises(segy.SegyError):
        segy.open_segy(p)
    p.write_bytes(b"x" * 100)
    with pytest.raises(segy.SegyError):
        segy.open_segy(p)


def test_scalar_factor():
    np.testing.assert_allclose(segy.scalar_factor([0, 1, 10, -10, -100]), [1, 1, 10, 0.1, 0.01])


def test_offsets_from_coordinates_when_header_empty():
    data, headers = survey(nshots=1, nrec=4, ns=32)
    headers["offset"] = 0
    s = segy.from_arrays(data, 0.002, headers)
    np.testing.assert_allclose(s.offsets(), [50, 75, 100, 125])
