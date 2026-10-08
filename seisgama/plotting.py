"""Matplotlib plotting helpers (usable in the GUI and headless from the CLI)."""
from __future__ import annotations

import numpy as np

from .processing import time_axis
from .qc import QCResult
from .segy import SegyFile

SEISMIC_CMAPS = ("gray_r", "gray", "seismic", "RdBu", "bwr", "PuOr", "viridis", "jet")


def clip_value(data: np.ndarray, percentile: float = 99.0) -> float:
    a = np.abs(np.asarray(data))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    c = float(np.percentile(a, percentile))
    return c if c > 0 else (float(a.max()) or 1.0)


def plot_section(ax, data, dt: float, t0: float = 0.0, x=None, mode: str = "density",
                 clip_percentile: float = 99.0, cmap: str = "gray_r", xlabel: str = "Trace",
                 title: str | None = None, max_wiggles: int = 400):
    """Seismic section: variable density, wiggle (variable area) or both."""
    a = np.asarray(data, dtype=np.float32)
    ax.clear()
    if a.size == 0:
        ax.set_title("No traces")
        return None
    ntr, ns = a.shape
    t = time_axis(ns, dt, t0)
    xs = np.arange(1, ntr + 1, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    if ntr > 1 and len(np.unique(xs)) < ntr:  # equal/duplicate positions would hide traces
        xs = np.arange(1, ntr + 1, dtype=np.float64)
        xlabel = f"Trace (duplicate {xlabel.lower()} values)"
    if ntr > 1 and np.any(np.diff(xs) < 0):  # e.g. offsets of a CMP gather: sort for display
        order = np.argsort(xs, kind="stable")
        xs, a = xs[order], a[order]
    clip = clip_value(a, clip_percentile)
    im = None
    if mode in ("density", "both"):
        d = np.diff(xs)
        uniform = ntr < 2 or (np.all(d > 0) and np.ptp(d) <= 0.01 * np.median(d))
        if uniform:
            half = d[0] / 2 if ntr > 1 else 0.5
            im = ax.imshow(a.T, aspect="auto", cmap=cmap, vmin=-clip, vmax=clip, interpolation="bilinear",
                           extent=(xs[0] - half, xs[-1] + half, t[-1] + dt / 2, t[0] - dt / 2))
        else:  # irregular spacing (gaps, split spread, duplicate positions): true positions
            mid = (xs[1:] + xs[:-1]) / 2
            step = np.median(d[d > 0]) if np.any(d > 0) else 1.0
            edges = np.concatenate([[xs[0] - step / 2], mid, [xs[-1] + step / 2]])
            tedges = np.concatenate([t - dt / 2, [t[-1] + dt / 2]])
            im = ax.pcolormesh(edges, tedges, a.T, cmap=cmap, vmin=-clip, vmax=clip, shading="flat",
                               rasterized=True)
            ax.set_ylim(tedges[-1], tedges[0])
    if mode in ("wiggle", "both"):
        step = max(1, int(np.ceil(ntr / max_wiggles)))
        spacing = (np.median(np.abs(np.diff(xs))) if ntr > 1 else 1.0) * step
        scale = spacing / clip
        for i in range(0, ntr, step):
            tr = np.clip(a[i], -clip, clip) * scale
            ax.plot(xs[i] + tr, t, color="k", lw=0.5)
            ax.fill_betweenx(t, xs[i], xs[i] + tr, where=tr > 0, color="k", lw=0)
        ax.set_ylim(t[-1], t[0])
        lo, hi = (xs.min(), xs.max())
        ax.set_xlim(lo - spacing, hi + spacing)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)
    return im


def plot_spectrum(ax, freqs, amp, db: bool = True, title: str = "Average amplitude spectrum"):
    ax.clear()
    amp = np.asarray(amp, dtype=np.float64)
    if db:
        amp = 20 * np.log10(amp / max(amp.max(), 1e-30) + 1e-12)
        ax.set_ylabel("Amplitude (dB)")
        ax.set_ylim(max(amp.min(), -80), 3)
    else:
        ax.set_ylabel("Amplitude")
    ax.plot(freqs, amp, color="tab:blue", lw=1)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_xlim(0, freqs[-1] if len(freqs) else 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)


def plot_fk(ax, f, k, amp, title: str = "F-K spectrum", cmap: str = "viridis"):
    ax.clear()
    a = 20 * np.log10(np.asarray(amp) / max(np.max(amp), 1e-30) + 1e-12)
    dk = (k[1] - k[0]) if len(k) > 1 else 1.0
    df = (f[1] - f[0]) if len(f) > 1 else 1.0
    im = ax.imshow(a, aspect="auto", origin="lower", cmap=cmap, vmin=-60, vmax=0,
                   extent=(k[0] - dk / 2, k[-1] + dk / 2, f[0] - df / 2, f[-1] + df / 2))
    ax.set_xlabel("Wavenumber (1/m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return im


def plot_fx(ax, x, f, amp, title: str = "F-X spectrum", cmap: str = "viridis"):
    ax.clear()
    a = 20 * np.log10(np.asarray(amp) / max(np.max(amp), 1e-30) + 1e-12)
    x = np.asarray(x, dtype=np.float64)
    im = ax.imshow(a.T, aspect="auto", origin="lower", cmap=cmap, vmin=-60, vmax=0,
                   extent=(x.min() - 0.5, x.max() + 0.5, f[0], f[-1]))
    ax.set_xlabel("Trace")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return im


def plot_semblance(ax, semb, velocities, dt: float, t0: float = 0.0, picks=(), cmap: str = "jet"):
    ax.clear()
    ns = semb.shape[1]
    t = time_axis(ns, dt, t0)
    v = np.asarray(velocities, dtype=np.float64)
    dv = (v[1] - v[0]) if len(v) > 1 else 1.0
    im = ax.imshow(semb.T, aspect="auto", cmap=cmap, vmin=0, vmax=max(float(semb.max()), 1e-6),
                   extent=(v[0] - dv / 2, v[-1] + dv / 2, t[-1] + dt / 2, t[0] - dt / 2))
    if len(picks):
        p = np.asarray(sorted(picks))
        ax.plot(p[:, 1], p[:, 0], "w-o", ms=5, mec="k", lw=1.5)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Semblance")
    return im


def plot_qc_overview(fig, seg: SegyFile, result: QCResult):
    """Four-panel QC overview: RMS per trace, traces per gather, elevations, geometry map."""
    fig.clear()
    axs = fig.subplots(2, 2)
    n = seg.n_traces
    idx = np.arange(1, n + 1)
    rms = result.attributes["rms"]
    ax = axs[0, 0]
    ax.semilogy(idx, np.where(rms > 0, rms, np.nan), ",", color="0.4")
    colors = {"dead": "k", "noisy": "r", "weak": "b", "spike": "m", "clipped": "orange", "nonfinite": "c"}
    for name, col in colors.items():
        m = result.flags[name]
        if m.any():
            yy = np.where(rms[m] > 0, rms[m], np.nanmin(np.where(rms > 0, rms, np.nan)) if (rms > 0).any() else 1)
            ax.semilogy(idx[m], yy, "o", ms=4, color=col, label=f"{name} ({m.sum()})")
    ax.set_xlabel("Trace")
    ax.set_ylabel("RMS amplitude")
    ax.set_title("Trace RMS and flagged traces")
    if any(result.flags[k].any() for k in colors):
        ax.legend(fontsize=7)

    ax = axs[0, 1]
    key = result.gather_key
    g, c = np.unique(seg.headers[key], return_counts=True)
    ax.plot(g, c, ".-", ms=3)
    ax.set_xlabel(key)
    ax.set_ylabel("Traces")
    ax.set_title(f"Traces per gather ({key})")
    ax.grid(True, alpha=0.3)

    ax = axs[1, 0]
    ax.plot(idx, seg.scaled("selev"), label="source (selev)", lw=1)
    ax.plot(idx, seg.scaled("gelev"), label="receiver (gelev)", lw=1)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Elevation")
    ax.set_title("Elevations")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axs[1, 1]
    sx, sy, gx, gy = (seg.scaled(k) for k in ("sx", "sy", "gx", "gy"))
    ax.plot(gx, gy, "v", ms=3, color="tab:green", label="receivers", alpha=0.5)
    ax.plot(sx, sy, "*", ms=6, color="tab:red", label="sources")
    ax.set_title("Geometry map")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    return axs
