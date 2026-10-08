"""Synthetic shot gathers with known answers (for tests, demos and training)."""
from __future__ import annotations

import numpy as np

from .segy import empty_trace_headers


def ricker(freq: float, dt: float, length: float = 0.2) -> np.ndarray:
    t = np.arange(-length / 2, length / 2 + dt / 2, dt)
    a = (np.pi * freq * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def minimum_phase_wavelet(dt: float, f0: float = 30.0, damping: float = 0.6, length: float = 0.2) -> np.ndarray:
    """Causal, minimum-phase damped sinusoid (a simple marine-airgun-like wavelet)."""
    t = np.arange(0, length, dt)
    w = np.exp(-damping * 2 * np.pi * f0 * t) * np.sin(2 * np.pi * f0 * t + 0.5)
    return w / np.abs(w).max()


DEFAULT_REFLECTORS = ((0.30, 1600.0, 1.0), (0.60, 1900.0, -0.7), (0.95, 2300.0, 0.6), (1.30, 2700.0, -0.5))


def shot_gather(nrec: int = 60, dx: float = 25.0, near_offset: float = 50.0, ns: int = 751, dt: float = 0.002,
                reflectors=DEFAULT_REFLECTORS, freq: float = 25.0, noise: float = 0.0, split_spread: bool = False,
                seed: int = 0, wavelet: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Hyperbolic reflections ``t(x) = sqrt(t0^2 + x^2/v^2)``. Returns (data, offsets)."""
    if split_spread:
        half = nrec // 2
        offsets = np.concatenate([-(near_offset + dx * np.arange(half)[::-1]), near_offset + dx * np.arange(nrec - half)])
    else:
        offsets = near_offset + dx * np.arange(nrec)
    w = ricker(freq, dt) if wavelet is None else np.asarray(wavelet)
    causal = wavelet is not None
    data = np.zeros((nrec, ns))
    for t0, v, amp in reflectors:
        tx = np.sqrt(t0**2 + (offsets / v) ** 2)
        for i, ti in enumerate(tx):
            # band-limited fractional delay: put a spike by linear weights then convolve
            k = ti / dt
            k0 = int(np.floor(k))
            if 0 <= k0 < ns - 1:
                frac = k - k0
                data[i, k0] += amp * (1 - frac)
                data[i, k0 + 1] += amp * frac
    for i in range(nrec):
        full = np.convolve(data[i], w)
        start = 0 if causal else len(w) // 2
        data[i] = full[start : start + ns]
    if noise:
        rng = np.random.default_rng(seed)
        data += noise * rng.standard_normal(data.shape)
    return data.astype(np.float32), offsets.astype(np.float64)


def survey(nshots: int = 4, nrec: int = 48, dx: float = 25.0, shot_step: float = 50.0, ns: int = 751,
           dt: float = 0.002, noise: float = 0.02, seed: int = 0, **kw) -> tuple[np.ndarray, np.ndarray]:
    """A small 2-D line of end-on shots with complete trace headers. Returns (data, headers)."""
    traces, rows = [], []
    rng = np.random.default_rng(seed)
    for s in range(nshots):
        d, off = shot_gather(nrec=nrec, dx=dx, ns=ns, dt=dt, noise=noise, seed=seed + s, **kw)
        sx = 1000.0 + s * shot_step
        gx = sx + off
        h = empty_trace_headers(nrec)
        h["fldr"] = 1001 + s
        h["ep"] = 101 + s
        h["tracf"] = np.arange(1, nrec + 1)
        h["offset"] = np.round(off).astype(np.int32)
        h["scalco"] = -100
        h["sx"] = np.round(sx * 100)
        h["gx"] = np.round(gx * 100)
        h["sy"] = h["gy"] = 500000
        h["scalel"] = -10
        h["selev"] = np.round((120 + 2 * np.sin(s)) * 10)
        h["gelev"] = np.round((120 + 3 * np.sin(gx / 300.0) + rng.normal(0, 0.2, nrec)) * 10)
        h["cdp"] = np.round(((sx + gx) / 2 - 1000.0) / (dx / 2)).astype(np.int32) + 1
        h["trid"] = 1
        traces.append(d)
        rows.append(h)
    headers = np.concatenate(rows)
    headers["tracl"] = headers["tracr"] = np.arange(1, len(headers) + 1)
    headers["ns"] = ns
    headers["dt"] = int(round(dt * 1e6))
    return np.concatenate(traces), headers
