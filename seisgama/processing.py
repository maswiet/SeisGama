"""Basic seismic processing for QC (time domain, pre-stack).

All functions take ``data`` as a 2-D float array (traces x samples) and the
sample interval ``dt`` in **seconds**; they return a new array and never modify
the input in place.  Times are absolute (``t0`` = recording delay of sample 0).

References: Yilmaz, *Seismic Data Analysis* (SEG, 2001), ch. 1-3;
Neidell & Taner (1971) for semblance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import linalg, signal
from scipy.ndimage import uniform_filter1d

_EPS = 1e-30


def _as2d(data) -> np.ndarray:
    a = np.asarray(data, dtype=np.float64)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if a.ndim != 2:
        raise ValueError("data must be 1-D or 2-D (traces x samples)")
    return a


def _out(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32)


def time_axis(ns: int, dt: float, t0: float = 0.0) -> np.ndarray:
    return t0 + np.arange(ns) * dt


def _check_dt(dt: float) -> None:
    if not dt > 0:
        raise ValueError("sample interval dt must be > 0")


# --------------------------------------------------------------------------
# Amplitude
# --------------------------------------------------------------------------

def remove_dc(data) -> np.ndarray:
    """Subtract the mean of every trace."""
    a = _as2d(data)
    return _out(a - a.mean(axis=1, keepdims=True))


def gain_tpow(data, dt: float, power: float = 2.0, t0: float = 0.0) -> np.ndarray:
    """Spherical-divergence style gain: multiply by t**power (t in s)."""
    _check_dt(dt)
    a = _as2d(data)
    t = np.maximum(time_axis(a.shape[1], dt, t0), dt)  # avoid 0**negative
    return _out(a * t**power)


def gain_exp(data, dt: float, alpha: float = 1.0, t0: float = 0.0) -> np.ndarray:
    """Exponential gain: multiply by exp(alpha * t) (alpha in 1/s)."""
    _check_dt(dt)
    a = _as2d(data)
    t = time_axis(a.shape[1], dt, t0)
    return _out(a * np.exp(alpha * t))


def agc(data, dt: float, window: float = 0.5) -> np.ndarray:
    """Automatic gain control with a sliding RMS window of ``window`` seconds."""
    _check_dt(dt)
    a = _as2d(data)
    n = max(1, int(round(window / dt)))
    n += (n + 1) % 2  # odd -> centred window
    energy = uniform_filter1d(a * a, size=n, axis=1, mode="nearest")
    rms = np.sqrt(np.maximum(energy, 0.0))
    floor = 1e-6 * rms.max(axis=1, keepdims=True) + _EPS
    return _out(np.where(rms > floor, a / np.maximum(rms, floor), 0.0))


def trace_balance(data, method: str = "rms") -> np.ndarray:
    """Scale every trace to unit RMS (``rms``) or unit peak (``max``)."""
    a = _as2d(data)
    if method == "rms":
        s = np.sqrt(np.mean(a * a, axis=1, keepdims=True))
    elif method == "max":
        s = np.max(np.abs(a), axis=1, keepdims=True)
    else:
        raise ValueError("method must be 'rms' or 'max'")
    return _out(np.where(s > _EPS, a / np.maximum(s, _EPS), 0.0))


# --------------------------------------------------------------------------
# Frequency filtering
# --------------------------------------------------------------------------

def _nfft(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(2 * n, 2))))


def bandpass(data, dt: float, f1: float, f2: float, f3: float, f4: float) -> np.ndarray:
    """Zero-phase trapezoidal (Ormsby-type) band-pass with cosine tapers.

    Pass band is f2..f3 Hz, tapering to zero at f1 and f4.
    """
    _check_dt(dt)
    nyq = 0.5 / dt
    if not (0 <= f1 <= f2 < f3 <= f4):
        raise ValueError("band-pass corners must satisfy 0 <= f1 <= f2 < f3 <= f4")
    if f3 > nyq:
        raise ValueError(f"f3={f3} Hz exceeds the Nyquist frequency {nyq:g} Hz")
    a = _as2d(data)
    ns = a.shape[1]
    nfft = _nfft(ns)
    f = np.fft.rfftfreq(nfft, dt)
    h = np.zeros_like(f)
    h[(f >= f2) & (f <= f3)] = 1.0
    if f2 > f1:
        m = (f >= f1) & (f < f2)
        h[m] = np.sin(0.5 * np.pi * (f[m] - f1) / (f2 - f1)) ** 2
    if f4 > f3:
        m = (f > f3) & (f <= f4)
        h[m] = np.cos(0.5 * np.pi * (f[m] - f3) / (f4 - f3)) ** 2
    spec = np.fft.rfft(a, n=nfft, axis=1) * h
    return _out(np.fft.irfft(spec, n=nfft, axis=1)[:, :ns])


def notch(data, dt: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Zero-phase notch filter (e.g. 50/60 Hz power-line noise on land data)."""
    _check_dt(dt)
    nyq = 0.5 / dt
    if not 0 < freq < nyq:
        raise ValueError(f"notch frequency must be between 0 and Nyquist ({nyq:g} Hz)")
    b, a_ = signal.iirnotch(freq, q, fs=1.0 / dt)
    a = _as2d(data)
    padlen = min(a.shape[1] - 1, 3 * max(len(a_), len(b)))
    return _out(signal.filtfilt(b, a_, a, axis=1, padlen=padlen))


# --------------------------------------------------------------------------
# Mute
# --------------------------------------------------------------------------

def _mute_times(offsets, picks: Sequence[Sequence[float]], use_abs: bool) -> np.ndarray:
    picks = np.asarray(picks, dtype=np.float64)
    if picks.ndim != 2 or picks.shape[1] != 2 or len(picks) == 0:
        raise ValueError("mute picks must be a list of (offset, time) pairs")
    order = np.argsort(picks[:, 0])
    px, pt = picks[order, 0], picks[order, 1]
    x = np.abs(offsets) if use_abs else np.asarray(offsets, dtype=np.float64)
    return np.interp(x, px, pt)  # constant extrapolation outside the picks


def mute(data, dt: float, offsets, picks, mode: str = "top", taper: float = 0.02,
         t0: float = 0.0, use_abs_offset: bool = True) -> np.ndarray:
    """Top or bottom mute defined by (offset, time[s]) picks, linearly interpolated.

    A cosine taper of ``taper`` seconds is applied on the live side of the mute.
    """
    _check_dt(dt)
    a = _as2d(data)
    offsets = np.asarray(offsets, dtype=np.float64)
    if len(offsets) != a.shape[0]:
        raise ValueError("one offset per trace is required")
    tm = _mute_times(offsets, picks, use_abs_offset)[:, None]
    t = time_axis(a.shape[1], dt, t0)[None, :]
    if mode == "top":
        d = t - tm  # >= 0 is live
    elif mode == "bottom":
        d = tm - t
    else:
        raise ValueError("mode must be 'top' or 'bottom'")
    if taper > 0:
        w = np.clip(d / taper, 0.0, 1.0)
        w = np.sin(0.5 * np.pi * w) ** 2
    else:
        w = (d >= 0).astype(np.float64)
    return _out(a * w)


# --------------------------------------------------------------------------
# Deconvolution
# --------------------------------------------------------------------------

def autocorrelation(x: np.ndarray, nlag: int) -> np.ndarray:
    """One-sided autocorrelation r[0..nlag-1] of each row (no wrap-around)."""
    a = _as2d(x)
    n = a.shape[1]
    nfft = _nfft(n)
    spec = np.fft.rfft(a, n=nfft, axis=1)
    r = np.fft.irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :nlag]
    if nlag > n:
        r[:, n:] = 0.0
    return r


def wiener_filter(r: np.ndarray, nop: int, gap: int = 1, prewhite: float = 0.1) -> np.ndarray:
    """Prediction-error filter from an autocorrelation ``r`` (length >= nop+gap).

    ``gap`` = 1 gives spiking (whitening) deconvolution; ``gap`` > 1 gives
    predictive (gapped) deconvolution.  ``prewhite`` is in percent of r[0].
    """
    if r[0] <= 0:
        out = np.zeros(gap + nop)
        out[0] = 1.0
        return out
    rr = np.array(r[:nop], dtype=np.float64)
    rr[0] *= 1.0 + prewhite / 100.0
    g = np.asarray(r[gap : gap + nop], dtype=np.float64)
    a = linalg.solve_toeplitz(rr, g)
    pef = np.zeros(gap + nop)
    pef[0] = 1.0
    pef[gap : gap + nop] = -a
    return pef


def deconvolution(data, dt: float, operator_length: float = 0.12, gap: float | None = None,
                  prewhite: float = 0.1, window: tuple[float, float] | None = None,
                  t0: float = 0.0, ensemble: bool = False) -> np.ndarray:
    """Wiener-Levinson spiking / predictive deconvolution.

    Parameters
    ----------
    operator_length : filter length in seconds.
    gap : prediction distance in seconds; ``None`` or <= dt -> spiking decon.
    prewhite : white-noise level in percent (typ. 0.1 - 1).
    window : (t_start, t_end) design window in seconds; default whole trace.
    ensemble : design one operator from the summed autocorrelation of all traces.
    """
    _check_dt(dt)
    a = _as2d(data)
    ntr, ns = a.shape
    nop = max(2, int(round(operator_length / dt)))
    alpha = 1 if gap is None else max(1, int(round(gap / dt)))
    if window is not None:
        i0 = max(0, int(round((window[0] - t0) / dt)))
        i1 = min(ns, int(round((window[1] - t0) / dt)) + 1)
        if i1 - i0 < 2:
            raise ValueError("deconvolution design window is empty")
    else:
        i0, i1 = 0, ns
    if nop + alpha > i1 - i0:
        raise ValueError("operator length + gap is longer than the design window")
    r = autocorrelation(a[:, i0:i1], nop + alpha)
    out = np.empty_like(a)
    if ensemble:
        live = r[:, 0] > 0
        rsum = (r[live] / r[live, :1]).sum(axis=0) if np.any(live) else r[0]
        pef = wiener_filter(rsum, nop, alpha, prewhite)
        out[:] = signal.lfilter(pef, [1.0], a, axis=1)
    else:
        for i in range(ntr):
            pef = wiener_filter(r[i], nop, alpha, prewhite)
            out[i] = signal.lfilter(pef, [1.0], a[i])
    return _out(out)


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------

def amplitude_spectrum(data, dt: float, db: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Average amplitude spectrum of all traces. Returns (freqs [Hz], amplitude)."""
    _check_dt(dt)
    a = _as2d(data)
    amp = np.abs(np.fft.rfft(a, axis=1)).mean(axis=0)
    f = np.fft.rfftfreq(a.shape[1], dt)
    if db:
        amp = 20 * np.log10(amp / max(amp.max(), _EPS) + 1e-12)
    return f, amp


def dominant_frequency(data, dt: float) -> float:
    f, amp = amplitude_spectrum(data, dt)
    return float(f[np.argmax(amp[1:]) + 1]) if len(f) > 1 else 0.0


def fx_spectrum(data, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of every trace. Returns (freqs, amp[trace, freq])."""
    _check_dt(dt)
    a = _as2d(data)
    return np.fft.rfftfreq(a.shape[1], dt), np.abs(np.fft.rfft(a, axis=1))


def fk_spectrum(data, dt: float, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F-K amplitude spectrum.

    Returns (freqs >= 0 [Hz], wavenumbers [1/m] ascending, amp[freq, k]).
    """
    _check_dt(dt)
    if not dx > 0:
        raise ValueError("trace spacing dx must be > 0")
    a = _as2d(data)
    ntr, ns = a.shape
    spec = np.fft.fft2(a.T)  # (time, space)
    nf = ns // 2 + 1
    spec = spec[:nf, :]
    amp = np.fft.fftshift(np.abs(spec), axes=1)
    k = np.fft.fftshift(np.fft.fftfreq(ntr, dx))
    f = np.fft.rfftfreq(ns, dt)
    return f, k, amp


def trace_spacing(offsets) -> float:
    """Median absolute spacing between consecutive offsets (0 if undefined)."""
    d = np.abs(np.diff(np.asarray(offsets, dtype=np.float64)))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.0


# --------------------------------------------------------------------------
# Velocity analysis, NMO, stack
# --------------------------------------------------------------------------

def velocity_function(t: np.ndarray, picks_t, picks_v) -> np.ndarray:
    """Interpolate an RMS velocity function (constant extrapolation)."""
    pt = np.atleast_1d(np.asarray(picks_t, dtype=np.float64))
    pv = np.atleast_1d(np.asarray(picks_v, dtype=np.float64))
    if len(pt) != len(pv) or len(pt) == 0:
        raise ValueError("velocity picks need the same, non-zero number of times and velocities")
    if np.any(pv <= 0):
        raise ValueError("velocities must be > 0")
    order = np.argsort(pt)
    return np.interp(t, pt[order], pv[order])


def nmo_correct(data, dt: float, offsets, velocity, stretch_mute: float | None = 0.5,
                t0: float = 0.0) -> np.ndarray:
    """Normal-moveout correction with linear interpolation.

    ``velocity`` is a scalar (m/s), an array with one velocity per sample, or a
    tuple ``(times, velocities)`` of picks.  Samples whose NMO stretch
    (t_x - t_0)/t_0 exceeds ``stretch_mute`` are zeroed (None disables).
    """
    _check_dt(dt)
    a = _as2d(data)
    ntr, ns = a.shape
    offsets = np.asarray(offsets, dtype=np.float64)
    if len(offsets) != ntr:
        raise ValueError("one offset per trace is required")
    t = time_axis(ns, dt, t0)
    if isinstance(velocity, tuple):
        v = velocity_function(t, *velocity)
    else:
        v = np.broadcast_to(np.asarray(velocity, dtype=np.float64), t.shape)
        if np.any(v <= 0):
            raise ValueError("velocity must be > 0")
    out = np.zeros_like(a)
    tpos = np.maximum(t, 0.0)
    for i in range(ntr):
        tx = np.sqrt(tpos**2 + (offsets[i] / v) ** 2)
        out[i] = np.interp(tx, t, a[i], left=0.0, right=0.0)
        if stretch_mute is not None and offsets[i] != 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                stretch = (tx - tpos) / tpos
            out[i, ~(stretch <= stretch_mute)] = 0.0
    return _out(out)


def semblance(data, dt: float, offsets, velocities, window: float = 0.04,
              t0: float = 0.0, stretch_mute: float | None = None,
              min_live_fraction: float = 0.25) -> np.ndarray:
    """Semblance velocity spectrum, shape (n_velocities, n_samples), values in [0, 1].

    S(t0, v) = sum_window (sum_x a)^2 / (M * sum_window sum_x a^2),
    with M the number of live traces after NMO at each time.  Where fewer than
    ``min_live_fraction`` of the traces (and fewer than 2) are live, S is set to 0:
    otherwise a single surviving trace would give a meaningless S = 1.
    """
    _check_dt(dt)
    a = _as2d(data)
    velocities = np.asarray(velocities, dtype=np.float64)
    n = max(1, int(round(window / dt)))
    n += (n + 1) % 2
    min_live = max(2.0, min_live_fraction * a.shape[0])
    out = np.zeros((len(velocities), a.shape[1]))
    for j, v in enumerate(velocities):
        c = nmo_correct(a, dt, offsets, float(v), stretch_mute=stretch_mute, t0=t0).astype(np.float64)
        live = (c != 0).sum(axis=0).astype(np.float64)
        num = uniform_filter1d(c.sum(axis=0) ** 2, n, mode="constant")
        den = uniform_filter1d(live * (c * c).sum(axis=0), n, mode="constant")
        live_w = uniform_filter1d(live, n, mode="constant")
        ok = (den > _EPS) & (live_w >= min_live)
        out[j] = np.where(ok, num / np.maximum(den, _EPS), 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def pick_semblance_maxima(semb: np.ndarray, velocities, dt: float, t0: float = 0.0,
                          step: float = 0.1, min_semblance: float = 0.1,
                          skip: float = 0.05, relative: float = 0.3) -> list[tuple[float, float]]:
    """Simple automatic picker: peaks (in time) of the maximum semblance curve.

    Peaks must be at least ``step`` seconds apart, above ``min_semblance`` and
    ``relative`` x the strongest peak, and later than ``skip`` s after the first sample.
    """
    velocities = np.asarray(velocities, dtype=np.float64)
    smax = semb.max(axis=0)
    i_skip = int(round(skip / dt))
    smax = smax.copy()
    smax[:i_skip] = 0
    height = max(min_semblance, relative * float(smax.max()))
    peaks, _ = signal.find_peaks(smax, height=height, distance=max(1, int(round(step / dt))))
    return [(t0 + i * dt, float(velocities[np.argmax(semb[:, i])])) for i in peaks]


def stack(data, keys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack traces that share the same key (e.g. CDP), normalised by live fold.

    Returns (unique keys, stacked traces, fold per sample).
    """
    a = _as2d(data)
    keys = np.asarray(keys)
    if len(keys) != a.shape[0]:
        raise ValueError("one key per trace is required")
    uk, inv = np.unique(keys, return_inverse=True)
    total = np.zeros((len(uk), a.shape[1]))
    fold = np.zeros((len(uk), a.shape[1]))
    np.add.at(total, inv, a)
    np.add.at(fold, inv, (a != 0).astype(np.float64))
    stacked = np.where(fold > 0, total / np.maximum(fold, 1), 0.0)
    return uk, _out(stacked), fold.astype(np.int32)


# --------------------------------------------------------------------------
# Processing flows (shared by GUI and CLI)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Param:
    name: str
    default: object
    label: str
    kind: str = "float"  # float | int | bool | choice | picks
    choices: tuple = ()


@dataclass(frozen=True)
class Process:
    key: str
    title: str
    params: tuple[Param, ...]
    run: Callable[..., np.ndarray]
    help: str = ""


def _parse_picks(text) -> list[tuple[float, float]]:
    """Parse '0:0.05, 1000:0.4' or [[0,0.05],[1000,0.4]] into (x, y) pairs."""
    if isinstance(text, str):
        pairs = []
        for item in text.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            x, y = item.split(":")
            pairs.append((float(x), float(y)))
        return pairs
    return [(float(x), float(y)) for x, y in text]


PROCESSES: dict[str, Process] = {
    p.key: p
    for p in [
        Process("dc", "DC removal", (), lambda d, dt, off, t0: remove_dc(d), "Subtract trace mean."),
        Process(
            "tpow", "Gain t^n (spherical divergence)",
            (Param("power", 2.0, "Power n"),),
            lambda d, dt, off, t0, power: gain_tpow(d, dt, power, t0),
        ),
        Process(
            "exp", "Exponential gain",
            (Param("alpha", 1.0, "Alpha (1/s)"),),
            lambda d, dt, off, t0, alpha: gain_exp(d, dt, alpha, t0),
        ),
        Process(
            "agc", "AGC",
            (Param("window", 0.5, "Window (s)"),),
            lambda d, dt, off, t0, window: agc(d, dt, window),
        ),
        Process(
            "balance", "Trace balance",
            (Param("method", "rms", "Method", "choice", ("rms", "max")),),
            lambda d, dt, off, t0, method: trace_balance(d, method),
        ),
        Process(
            "bandpass", "Band-pass (Ormsby)",
            (Param("f1", 5.0, "f1 (Hz)"), Param("f2", 10.0, "f2 (Hz)"),
             Param("f3", 60.0, "f3 (Hz)"), Param("f4", 80.0, "f4 (Hz)")),
            lambda d, dt, off, t0, f1, f2, f3, f4: bandpass(d, dt, f1, f2, f3, f4),
        ),
        Process(
            "notch", "Notch filter",
            (Param("freq", 50.0, "Frequency (Hz)"), Param("q", 30.0, "Quality factor")),
            lambda d, dt, off, t0, freq, q: notch(d, dt, freq, q),
        ),
        Process(
            "mute", "Mute",
            (Param("picks", "0:0.0, 1000:0.4", "Picks offset:time(s)", "picks"),
             Param("mode", "top", "Mode", "choice", ("top", "bottom")),
             Param("taper", 0.02, "Taper (s)")),
            lambda d, dt, off, t0, picks, mode, taper: mute(d, dt, off, _parse_picks(picks), mode, taper, t0),
        ),
        Process(
            "decon", "Wiener deconvolution",
            (Param("operator_length", 0.12, "Operator length (s)"),
             Param("gap", 0.0, "Gap (s, 0 = spiking)"),
             Param("prewhite", 0.1, "Pre-whitening (%)"),
             Param("ensemble", False, "One operator per gather", "bool")),
            lambda d, dt, off, t0, operator_length, gap, prewhite, ensemble: deconvolution(
                d, dt, operator_length, gap if gap and gap > dt else None, prewhite, None, t0, ensemble),
        ),
        Process(
            "nmo", "NMO correction",
            (Param("picks", "0:1500, 1.0:2000", "Picks time(s):Vrms(m/s)", "picks"),
             Param("stretch_mute", 0.5, "Stretch mute (fraction)")),
            lambda d, dt, off, t0, picks, stretch_mute: nmo_correct(
                d, dt, off, tuple(np.array(_parse_picks(picks)).T), stretch_mute, t0),
        ),
    ]
}


def apply_flow(data, dt: float, offsets, steps: Sequence[dict], t0: float = 0.0) -> np.ndarray:
    """Apply a list of processing steps ``[{"op": "agc", "window": 0.5}, ...]``."""
    out = np.asarray(data, dtype=np.float32)
    for step in steps:
        step = dict(step)
        if not step.pop("enabled", True):
            continue
        op = step.pop("op")
        if op not in PROCESSES:
            raise ValueError(f"Unknown processing step '{op}'. Available: {', '.join(PROCESSES)}")
        proc = PROCESSES[op]
        kwargs = {p.name: step.get(p.name, p.default) for p in proc.params}
        unknown = set(step) - set(kwargs)
        if unknown:
            raise ValueError(f"Unknown parameter(s) for '{op}': {', '.join(sorted(unknown))}")
        out = proc.run(out, dt, offsets, t0, **kwargs)
    return out
