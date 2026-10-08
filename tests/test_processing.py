import numpy as np
import pytest

from seisgama import processing as P
from seisgama.synthetic import minimum_phase_wavelet, shot_gather

DT = 0.002


def test_functions_do_not_modify_input():
    d, off = shot_gather(nrec=10, ns=200)
    ref = d.copy()
    P.agc(d, DT)
    P.gain_tpow(d, DT)
    P.bandpass(d, DT, 5, 10, 40, 60)
    P.deconvolution(d, DT)
    P.nmo_correct(d, DT, off, 2000.0)
    P.mute(d, DT, off, [(0, 0.1), (1000, 0.5)])
    np.testing.assert_array_equal(d, ref)


def test_gain_tpow_and_exp():
    d = np.ones((2, 101))
    g = P.gain_tpow(d, 0.01, power=2.0)
    assert g[0, 50] == pytest.approx(0.5**2)
    assert g[0, 0] == pytest.approx(0.01**2)  # t=0 clamped to dt, no inf/nan
    e = P.gain_exp(d, 0.01, alpha=2.0)
    assert e[0, 100] == pytest.approx(np.exp(2.0), rel=1e-6)
    # uses seconds (the C# version divided microseconds by 1e7 instead of 1e6)
    assert P.gain_tpow(d, 0.004, power=1.0)[0, 100] == pytest.approx(0.4)


def test_agc_equalises_amplitudes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2000))
    x[0, 1000:] *= 100
    y = P.agc(x, DT, window=0.2)
    r1 = np.sqrt(np.mean(y[0, 200:800] ** 2))
    r2 = np.sqrt(np.mean(y[0, 1200:1800] ** 2))
    assert r1 == pytest.approx(1, rel=0.15) and r2 == pytest.approx(1, rel=0.15)
    assert np.all(np.isfinite(P.agc(np.zeros((2, 100)), DT)))


def test_bandpass_response():
    t = np.arange(2000) * DT
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 120 * t)
    y = P.bandpass(x, DT, 15, 20, 40, 50)[0]
    f, a = P.amplitude_spectrum(y[200:-200], DT)
    amp = lambda fr: a[np.argmin(np.abs(f - fr))]
    assert amp(30) > 50 * amp(5)
    assert amp(30) > 50 * amp(120)
    with pytest.raises(ValueError):
        P.bandpass(x, DT, 10, 5, 40, 50)
    with pytest.raises(ValueError):
        P.bandpass(x, DT, 10, 20, 300, 400)  # above Nyquist (250 Hz)


def test_notch_removes_50hz():
    t = np.arange(4000) * DT
    x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 20 * t)
    y = P.notch(x, DT, 50, 30)[0]
    ref = np.sin(2 * np.pi * 20 * t)
    assert np.sqrt(np.mean((y - ref)[500:-500] ** 2)) < 0.05


def test_mute_top_split_spread():
    d = np.ones((4, 501), dtype=np.float32)
    off = np.array([-1000.0, -100.0, 100.0, 1000.0])
    m = P.mute(d, DT, off, [(0, 0.1), (1000, 0.5)], taper=0.0)
    # symmetric for negative offsets (C# version crashed on negative offsets)
    np.testing.assert_array_equal(m[0], m[3])
    assert m[0, 249] == 0 and m[0, 251] == 1
    assert m[1, int(0.14 / DT) - 2] == 0 and m[1, int(0.14 / DT) + 1] == 1
    b = P.mute(d, DT, off, [(0, 0.5), (1000, 0.5)], mode="bottom", taper=0.0)
    assert b[0, 249] == 1 and b[0, 251] == 0


def test_spiking_decon_compresses_wavelet():
    rng = np.random.default_rng(3)
    ns = 2000
    refl = np.zeros((20, ns))
    for i in range(20):
        idx = rng.choice(np.arange(50, ns - 200), 40, replace=False)
        refl[i, idx] = rng.normal(size=40)
    w = minimum_phase_wavelet(DT, f0=25)
    data = np.array([np.convolve(r, w)[:ns] for r in refl])
    out = P.deconvolution(data, DT, operator_length=0.1, prewhite=0.1)

    def spikiness(x):  # varimax / kurtosis-like norm
        return np.sum(x**4) / np.sum(x**2) ** 2

    assert spikiness(out) > 3 * spikiness(data)
    # correlation of output with true reflectivity should be high
    c = np.corrcoef(out.ravel(), refl.ravel())[0, 1]
    assert c > 0.8
    ens = P.deconvolution(data, DT, operator_length=0.1, ensemble=True)
    assert np.corrcoef(ens.ravel(), refl.ravel())[0, 1] > 0.8


def test_predictive_decon_attenuates_multiples():
    ns, period = 1500, 100  # 0.2 s water-layer reverberation
    prim = np.zeros(ns)
    prim[100] = 1.0
    x = np.zeros(ns)
    for k in range(8):
        if 100 + k * period < ns:
            x[100 + k * period] = (-0.6) ** k
    w = minimum_phase_wavelet(DT, f0=30)
    trace = np.convolve(x, w)[:ns]
    out = P.deconvolution(trace, DT, operator_length=0.3, gap=0.02, prewhite=0.1)[0]
    before = np.abs(trace[100 + period : 100 + period + 40]).max()
    after = np.abs(out[100 + period : 100 + period + 40]).max()
    assert after < 0.25 * before
    # primary is preserved (first `gap` samples of the wavelet pass unchanged)
    np.testing.assert_allclose(out[100:110], trace[100:110], atol=1e-6)


def test_decon_dead_trace_is_safe():
    d = np.zeros((2, 500))
    assert np.all(P.deconvolution(d, DT) == 0)


def test_autocorrelation_matches_numpy():
    x = np.random.default_rng(0).normal(size=300)
    r = P.autocorrelation(x, 20)[0]
    ref = np.correlate(x, x, mode="full")[299:319]
    np.testing.assert_allclose(r, ref, rtol=1e-10, atol=1e-10)


def test_nmo_flattens_reflection():
    v = 2000.0
    d, off = shot_gather(nrec=40, dx=25, near_offset=0, ns=1001, reflectors=((0.8, v, 1.0),), freq=30)
    c = P.nmo_correct(d, DT, off, v, stretch_mute=None)
    picks = np.argmax(np.abs(c), axis=1) * DT
    np.testing.assert_allclose(picks, 0.8, atol=1.5 * DT)
    # with velocity function picks
    c2 = P.nmo_correct(d, DT, off, (np.array([0.0, 2.0]), np.array([v, v])), stretch_mute=None)
    np.testing.assert_allclose(c, c2)


def test_nmo_stretch_mute():
    d = np.ones((2, 500), dtype=np.float32)
    c = P.nmo_correct(d, DT, [0.0, 1000.0], 2000.0, stretch_mute=0.3)
    assert np.all(c[0] == 1)  # zero offset untouched
    # stretch (tx-t0)/t0 <= 0.3  <=> t0 >= x/v / sqrt(1.3^2-1) = 0.5/0.83 = 0.6 s
    assert c[1, int(0.55 / DT)] == 0 and c[1, int(0.65 / DT)] != 0


def test_semblance_finds_velocities():
    refl = ((0.4, 1700.0, 1.0), (0.9, 2200.0, 1.0), (1.3, 2600.0, 1.0))
    d, off = shot_gather(nrec=60, dx=25, ns=801, reflectors=refl, freq=25, noise=0.05, split_spread=True)
    vels = np.arange(1400, 3201, 25.0)
    s = P.semblance(d, DT, off, vels, window=0.03)
    assert s.shape == (len(vels), 801)
    assert 0 <= s.min() and s.max() <= 1
    for t0, v, _ in refl:
        it = int(round(t0 / DT))
        best = vels[np.argmax(s[:, it - 5 : it + 6].max(axis=1))]
        assert abs(best - v) <= 75, (t0, v, best)
    picks = P.pick_semblance_maxima(s, vels, DT, step=0.2, min_semblance=0.3)
    assert len(picks) >= 3


def test_stack_by_key_with_fold():
    d = np.array([[1, 1, 0], [3, 1, 0], [5, 5, 5]], dtype=np.float32)
    keys, st, fold = P.stack(d, [10, 10, 20])
    np.testing.assert_array_equal(keys, [10, 20])
    np.testing.assert_allclose(st, [[2, 1, 0], [5, 5, 5]])
    np.testing.assert_array_equal(fold[0], [2, 2, 0])


def test_fk_spectrum_locates_linear_event():
    ntr, ns, dx = 64, 512, 10.0
    f0, k0 = 40.0, 0.02  # apparent velocity 2000 m/s
    t = np.arange(ns) * DT
    x = np.arange(ntr) * dx
    d = np.cos(2 * np.pi * (f0 * t[None, :] - k0 * x[:, None]))
    f, k, a = P.fk_spectrum(d, DT, dx)
    i, j = np.unravel_index(np.argmax(a), a.shape)
    assert f[i] == pytest.approx(f0, abs=1.0)
    assert abs(k[j]) == pytest.approx(k0, abs=0.002)
    assert np.all(np.diff(k) > 0)


def test_apply_flow_and_validation():
    d, off = shot_gather(nrec=8, ns=300)
    out = P.apply_flow(d, DT, off, [
        {"op": "dc"}, {"op": "tpow", "power": 1.5}, {"op": "bandpass", "f1": 3, "f2": 8, "f3": 50, "f4": 70},
        {"op": "agc", "window": 0.3}, {"op": "mute", "picks": "0:0.05, 1000:0.4"},
        {"op": "decon", "operator_length": 0.08}, {"op": "nmo", "picks": "0:1600,1.0:2400"},
        {"op": "agc", "enabled": False},
    ])
    assert out.shape == d.shape and np.all(np.isfinite(out))
    with pytest.raises(ValueError):
        P.apply_flow(d, DT, off, [{"op": "nope"}])
    with pytest.raises(ValueError):
        P.apply_flow(d, DT, off, [{"op": "agc", "windw": 1}])
