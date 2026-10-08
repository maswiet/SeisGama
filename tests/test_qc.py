import numpy as np

from seisgama import qc, segy
from seisgama.synthetic import survey


def make_bad_survey():
    data, h = survey(nshots=3, nrec=40, ns=500, noise=0.05)
    data = data.copy()
    data[5] = 0  # dead
    data[50] *= 40  # noisy
    data[60] *= 0.01  # weak
    data[70, 300] = 500  # spike
    peak = np.abs(data[90]).max()
    data[90, 100:110] = peak * 3  # clipped plateau at the new peak
    data[100, 10] = np.nan
    h = h.copy()
    h["trid"][110] = 2  # dead per header
    return data, h


def test_qc_flags_planted_problems():
    data, h = make_bad_survey()
    s = segy.from_arrays(data, 0.002, h)
    r = qc.run_qc(s)
    expect = {"dead": 5, "noisy": 50, "weak": 60, "spike": 70, "clipped": 90, "nonfinite": 100, "not_seismic": 110}
    for flag, i in expect.items():
        assert r.flags[flag][i], flag
    # no false alarms on the remaining traces
    planted = set(expect.values())
    others = [i for i in r.flagged_indices() if i not in planted]
    assert others == [], [(i, r.reasons(i)) for i in others]
    assert r.summary["traces"] == 120
    assert r.summary["good_traces"] == 120 - len(planted)


def test_header_checks_detect_geometry_errors():
    data, h = survey(nshots=2, nrec=20, ns=200)
    h = h.copy()
    h["offset"][3] += 500  # wrong offset
    h["scalco"][7] = 7  # invalid scalar
    h = np.delete(h, 30)  # missing trace in 2nd shot
    data = np.delete(data, 30, axis=0)
    s = segy.from_arrays(data, 0.002, h)
    msgs = " | ".join(m for _, m in qc.header_checks(s))
    assert "header offset differs" in msgs
    assert "invalid coordinate scalar" in msgs
    assert "fewer traces than the usual 20" in msgs


def test_clean_survey_has_no_flags_and_reports(tmp_path):
    data, h = survey(nshots=2, nrec=30, ns=300)
    s = segy.from_arrays(data, 0.002, h)
    r = qc.run_qc(s)
    assert r.bad.sum() == 0
    levels = {lvl for lvl, _ in r.checks}
    assert "ERROR" not in levels
    txt = qc.report_text(s, r)
    assert "SEISGAMA QC REPORT" in txt
    p = tmp_path / "qc.csv"
    qc.write_csv(s, r, p)
    lines = p.read_text().splitlines()
    assert len(lines) == 61 and lines[0].startswith("trace,fldr")


def test_qc_cancel():
    data, h = survey(nshots=1, nrec=10, ns=100)
    s = segy.from_arrays(data, 0.002, h)
    try:
        qc.run_qc(s, progress=lambda p: True, chunk=2)
    except InterruptedError:
        pass
    else:
        raise AssertionError("expected cancellation")


def test_process_refuses_to_overwrite_input(tmp_path):
    import pytest

    from seisgama.workflow import process_file

    data, h = survey(nshots=1, nrec=4, ns=50)
    p = tmp_path / "in.sgy"
    segy.write_segy(p, data, 0.002, h)
    s = segy.open_segy(p)
    with pytest.raises(ValueError):
        process_file(s, p, [{"op": "agc"}])
    np.testing.assert_allclose(s.read_all(), data)
    s.close()


def _raw_gathers(kind, nshots=3, nrec=96, ns=2000, dt=0.002, seed=0):
    """Raw-looking field gathers: strong early events, amplitude decay, coherent noise."""
    from seisgama.synthetic import ricker, shot_gather

    rng = np.random.default_rng(seed)
    t = np.arange(ns) * dt
    out, heads = [], []
    for s in range(nshots):
        if kind == "marine":
            refl = ((0.8, 1500.0, 20.0), (1.4, 1800.0, 3.0), (2.1, 2300.0, 2.0), (3.0, 2800.0, 1.5))
            d, off = shot_gather(nrec=nrec, dx=12.5, near_offset=100, ns=ns, dt=dt, reflectors=refl, freq=30)
            vdir, fdir, adir = 1500.0, 30.0, 50.0
        else:
            refl = ((0.3, 1800.0, 1.0), (0.7, 2200.0, 0.8), (1.2, 2700.0, 0.6))
            d, off = shot_gather(nrec=nrec, dx=10, near_offset=10, ns=ns, dt=dt, reflectors=refl, freq=30)
            vdir, fdir, adir = 2000.0, 25.0, 30.0
        d = d.astype(np.float64)
        w = ricker(fdir, dt)
        for i, x in enumerate(off):  # direct wave / first break
            k = int(x / vdir / dt)
            if k < ns - len(w):
                d[i, k : k + len(w)] += adir * w
            if kind == "land":  # ground roll: slow, low frequency, dispersive cone
                tg = x / 400.0
                env = np.exp(-((t - tg - 0.15) / 0.12) ** 2)
                d[i] += 15 * env * np.sin(2 * np.pi * 9 * (t - tg))
        d /= np.maximum(t, 0.05)[None, :]  # spherical divergence decay
        d += 0.05 * rng.standard_normal(d.shape)
        if kind == "marine":
            d += 0.3 * np.sin(2 * np.pi * 1.5 * t[None, :] + rng.uniform(0, 6, (nrec, 1)))  # swell
        h = segy.empty_trace_headers(nrec)
        h["fldr"] = 100 + s
        h["tracf"] = np.arange(1, nrec + 1)
        h["offset"] = off.astype(int)
        h["trid"] = 11 if kind == "marine" else 1
        out.append(d.astype(np.float32))
        heads.append(h)
    return np.concatenate(out), np.concatenate(heads)


import pytest  # noqa: E402


@pytest.mark.parametrize("kind", ["marine", "land"])
def test_no_false_alarms_on_raw_field_like_gathers(kind):
    data, h = _raw_gathers(kind)
    s = segy.from_arrays(data, 0.002, h)
    r = qc.run_qc(s)
    flagged = [(i, r.reasons(i)) for i in r.flagged_indices()]
    assert flagged == [], flagged[:10]


@pytest.mark.parametrize("kind", ["marine", "land"])
def test_detects_problems_in_raw_field_like_gathers(kind):
    data, h = _raw_gathers(kind)
    data[20, 1500] = 30 * np.abs(data[20]).max()  # spike late in the trace
    data[40] = 0  # dead channel
    data[130] *= 8  # noisy channel
    s = segy.from_arrays(data, 0.002, h)
    r = qc.run_qc(s)
    assert r.flags["spike"][20] and r.flags["dead"][40] and r.flags["noisy"][130]
    assert r.bad.sum() == 3, [(i, r.reasons(i)) for i in r.flagged_indices()]
