"""Quality control of SEG-Y data: trace attributes, bad-trace detection and header checks."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .segy import SegyFile

VALID_SCALARS = {0, 1, -1, 10, -10, 100, -100, 1000, -1000, 10000, -10000}

ATTRIBUTE_NAMES = ("rms", "peak", "mean", "zero_fraction", "nonfinite", "clip_fraction",
                   "spike_ratio", "dominant_freq")

FLAG_DESCRIPTIONS = {
    "dead": "Dead trace (all/near-all samples zero)",
    "nonfinite": "Contains NaN or Inf samples",
    "noisy": "RMS amplitude anomalously high within its gather",
    "weak": "RMS amplitude anomalously low within its gather",
    "spike": "Isolated spike (sample much larger than its neighbours)",
    "clipped": "Clipped samples (flat-topped at the peak value)",
    "not_seismic": "Trace id (trid 2-10) marks the trace as dead, dummy or auxiliary",
}


@dataclass
class QCSettings:
    gather_key: str = "fldr"
    z_threshold: float = 3.5  # robust z-score of the RMS anomaly for noisy/weak traces
    min_amplitude_ratio: float = 2.0  # ... and at least this factor above/below its neighbours
    neighbours: int = 11  # running-median length (traces) defining the "normal" RMS trend
    spike_ratio: float = 12.0  # sample amplitude / RMS of its +-10 neighbouring samples
    clip_fraction: float = 0.002  # fraction of clipped samples
    dead_zero_fraction: float = 0.99
    offset_tolerance: float = 0.02  # relative mismatch header offset vs coordinates
    min_gather_size: int = 6  # below this the global statistics are used


@dataclass
class QCResult:
    attributes: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    checks: list[tuple[str, str]] = field(default_factory=list)  # (level, message)
    summary: dict = field(default_factory=dict)
    gather_key: str = "fldr"

    @property
    def bad(self) -> np.ndarray:
        any_bad = np.zeros(len(self.attributes["rms"]), dtype=bool)
        for f in self.flags.values():
            any_bad |= f
        return any_bad

    def flagged_indices(self) -> np.ndarray:
        return np.flatnonzero(self.bad)

    def reasons(self, i: int) -> list[str]:
        return [name for name, f in self.flags.items() if f[i]]


def _clip_fraction(a: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """Fraction of samples in runs of >= 3 consecutive samples at the trace peak."""
    at_peak = np.abs(a) >= 0.9999 * peak[:, None]
    at_peak &= peak[:, None] > 0
    run = at_peak[:, 1:-1] & at_peak[:, :-2] & at_peak[:, 2:]
    return run.sum(axis=1) / max(a.shape[1], 1)


def local_spike_ratio(a: np.ndarray, rms: np.ndarray, half_window: int = 10) -> np.ndarray:
    """Largest ratio |a_i| / RMS of its neighbours (+-half_window samples, a_i excluded).

    A real reflection or first break is a wavelet spanning several samples, so its
    neighbours are strong too and the ratio stays small even when the event is much
    stronger than the rest of the trace (direct wave, water bottom, ground roll).
    An isolated spike has weak neighbours and gives a large ratio.
    """
    n = 2 * half_window + 1
    ns = a.shape[1]
    if ns < 3:
        return np.zeros(len(a))
    sq = a * a
    local_sum = uniform_filter1d(sq, size=min(n, ns), axis=1, mode="constant") * min(n, ns)
    excl = np.maximum(local_sum - sq, 0.0) / (min(n, ns) - 1)
    floor = (1e-3 * rms[:, None]) ** 2 + 1e-60
    ratio = np.abs(a) / np.sqrt(np.maximum(excl, floor))
    return np.where(rms > 0, ratio.max(axis=1), 0.0)


def trace_attributes(data: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    a = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(a)
    nonfinite = (~finite).sum(axis=1)
    a = np.where(finite, a, 0.0)
    rms = np.sqrt(np.mean(a * a, axis=1))
    peak = np.max(np.abs(a), axis=1)
    spike = local_spike_ratio(a, rms)
    spec = np.abs(np.fft.rfft(a - a.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(a.shape[1], dt)
    dom = freqs[np.argmax(spec, axis=1)] if spec.shape[1] else np.zeros(len(a))
    return {
        "rms": rms,
        "peak": peak,
        "mean": a.mean(axis=1),
        "zero_fraction": (a == 0).mean(axis=1),
        "nonfinite": nonfinite.astype(np.float64),
        "clip_fraction": _clip_fraction(a, peak),
        "spike_ratio": spike,
        "dominant_freq": np.where(rms > 0, dom, 0.0),
    }


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad <= 0:
        mad = np.std(x) or 1.0
    return (x - med) / mad


def run_qc(seg: SegyFile, settings: QCSettings | None = None,
           progress: Callable[[int], bool | None] | None = None, chunk: int = 2048) -> QCResult:
    """Run the full QC. ``progress(percent)`` may return True to cancel."""
    s = settings or QCSettings()
    n = seg.n_traces
    attrs = {k: np.zeros(n) for k in ATTRIBUTE_NAMES}
    for start, stop, data in seg.iter_chunks(chunk):
        for k, v in trace_attributes(data, seg.dt).items():
            attrs[k][start:stop] = v
        if progress is not None and progress(int(100 * stop / n)):
            raise InterruptedError("QC cancelled")

    flags = {k: np.zeros(n, dtype=bool) for k in FLAG_DESCRIPTIONS}
    flags["nonfinite"] = attrs["nonfinite"] > 0
    flags["dead"] = (attrs["rms"] == 0) | (attrs["zero_fraction"] >= s.dead_zero_fraction)
    trid = seg.headers["trid"]
    # SEG-Y rev1: 1 = seismic, 11 = pressure (hydrophone), 12-14 = multicomponent, 15+ = other
    # sensors; 2-10 are dead, dummy, time break, uphole, sweep, timing, water break, gun signatures.
    flags["not_seismic"] = (trid >= 2) & (trid <= 10)

    live = ~(flags["dead"] | flags["nonfinite"])
    logrms = np.log10(np.where(attrs["rms"] > 0, attrs["rms"], 1.0))
    keys = seg.headers[s.gather_key] if s.gather_key in seg.headers.dtype.names else np.zeros(n)
    # Amplitude anomaly = log10(RMS) minus the running median of its neighbours within
    # the same gather (trace order).  This removes the natural decay with offset/time
    # so that only traces that differ from their neighbours are flagged.
    resid = np.zeros(n)
    for k in np.unique(keys):
        idx = np.flatnonzero((keys == k) & live)
        if len(idx) >= 3:
            size = min(s.neighbours, len(idx) - (len(idx) + 1) % 2)
            trend = median_filter(logrms[idx], size=max(size, 1), mode="nearest")
            resid[idx] = logrms[idx] - trend
    z = np.zeros(n)
    if live.sum() >= 3:
        r = resid[live]
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        z[live] = (r - np.median(r)) / max(mad, 1e-12)
    floor = np.log10(s.min_amplitude_ratio)
    flags["noisy"] = live & (z > s.z_threshold) & (resid > floor)
    flags["weak"] = live & (z < -s.z_threshold) & (resid < -floor)
    flags["spike"] = live & (attrs["spike_ratio"] > s.spike_ratio)
    flags["clipped"] = live & (attrs["clip_fraction"] > s.clip_fraction)
    attrs["rms_anomaly_z"] = z

    checks = header_checks(seg, s)
    result = QCResult(attributes=attrs, flags=flags, checks=checks, gather_key=s.gather_key)
    bad = result.bad
    result.summary = {
        "traces": n,
        "gathers": int(len(np.unique(keys))),
        "good_traces": int(n - bad.sum()),
        "good_percent": round(float(100.0 * (n - bad.sum()) / max(n, 1)), 2),
        **{f"flag_{k}": int(v.sum()) for k, v in flags.items()},
        "median_rms": float(np.median(attrs["rms"][live])) if live.any() else 0.0,
        "median_dominant_freq_hz": float(np.median(attrs["dominant_freq"][live])) if live.any() else 0.0,
    }
    return result


def traces_per_gather(seg: SegyFile, key: str = "fldr") -> tuple[np.ndarray, np.ndarray]:
    return np.unique(seg.headers[key], return_counts=True)


def header_checks(seg: SegyFile, s: QCSettings | None = None) -> list[tuple[str, str]]:
    """Consistency checks on binary and trace headers. Returns (level, message)."""
    s = s or QCSettings()
    h = seg.headers
    out: list[tuple[str, str]] = [("WARNING", w) for w in seg.warnings]
    n = seg.n_traces

    ns_bad = (h["ns"] != seg.ns) & (h["ns"] != 0)
    if ns_bad.any():
        out.append(("WARNING", f"{ns_bad.sum()} trace(s) have header ns different from {seg.ns}."))
    dt_us = int(round(seg.dt * 1e6))
    dt_bad = (h["dt"] != dt_us) & (h["dt"] != 0)
    if dt_bad.any():
        out.append(("WARNING", f"{dt_bad.sum()} trace(s) have header dt different from {dt_us} us."))
    if np.any(h["delrt"] != 0):
        out.append(("INFO", f"Non-zero recording delay (delrt) on {np.count_nonzero(h['delrt'])} trace(s); "
                            f"first trace delay {h['delrt'][0]} ms."))
    if len(np.unique(h["delrt"])) > 1:
        out.append(("WARNING", "Recording delay (delrt) varies between traces; display uses the first trace."))

    for name, label in (("scalco", "coordinate"), ("scalel", "elevation")):
        bad = ~np.isin(h[name], list(VALID_SCALARS))
        if bad.any():
            out.append(("ERROR", f"{bad.sum()} trace(s) have an invalid {label} scalar ({name}), "
                                 f"e.g. {h[name][bad][0]}."))

    key = s.gather_key
    if key in h.dtype.names:
        g, counts = np.unique(h[key], return_counts=True)
        vals, freq = np.unique(counts, return_counts=True)
        mode = int(vals[freq == freq.max()].max())
        short = g[counts < mode]
        out.append(("INFO", f"{len(g)} gather(s) by {key}; traces per gather min/mode/max = "
                            f"{counts.min()}/{mode}/{counts.max()}."))
        if len(short):
            out.append(("WARNING", f"{len(short)} gather(s) have fewer traces than the usual {mode}: "
                                   f"{key} = {', '.join(map(str, short[:20]))}{' ...' if len(short) > 20 else ''}."))
        if np.any(np.diff(h[key]) < 0):
            out.append(("INFO", f"Traces are not sorted by {key}."))

    pair = h[["fldr", "tracf"]]
    if np.any(h["tracf"] != 0):
        _, cnt = np.unique(pair, return_counts=True)
        dup = int((cnt > 1).sum())
        if dup:
            out.append(("WARNING", f"{dup} duplicated (fldr, tracf) pair(s) - duplicate traces?"))

    sx, sy, gx, gy = (seg.scaled(k) for k in ("sx", "sy", "gx", "gy"))
    has_coords = np.any(sx != 0) or np.any(gx != 0) or np.any(sy != 0) or np.any(gy != 0)
    if not has_coords:
        out.append(("WARNING", "Source and receiver coordinates are all zero (no geometry in headers)."))
    else:
        zero_src = (sx == 0) & (sy == 0)
        zero_rec = (gx == 0) & (gy == 0)
        if zero_src.any():
            out.append(("WARNING", f"{zero_src.sum()} trace(s) have zero source coordinates."))
        if zero_rec.any():
            out.append(("WARNING", f"{zero_rec.sum()} trace(s) have zero receiver coordinates."))
        off_h = np.abs(h["offset"].astype(np.float64))
        if np.any(off_h != 0):
            dist = np.hypot(gx - sx, gy - sy)
            tol = np.maximum(s.offset_tolerance * np.maximum(dist, off_h), 1.0)
            mism = np.abs(dist - off_h) > tol
            if mism.any():
                i = np.flatnonzero(mism)[0]
                out.append(("WARNING", f"{mism.sum()} trace(s): header offset differs from the source-receiver "
                                       f"distance computed from coordinates (e.g. trace {i + 1}: "
                                       f"{off_h[i]:g} vs {dist[i]:.1f})."))
    if np.all(h["offset"] == 0):
        out.append(("WARNING", "All header offsets are zero."))

    for name, label in (("selev", "source"), ("gelev", "receiver")):
        e = seg.scaled(name)
        if np.any(e != 0) and n >= 5:
            zz = np.abs(_robust_z(e))
            spikes = zz > 6
            if spikes.any():
                out.append(("WARNING", f"{spikes.sum()} {label} elevation value(s) are outliers "
                                       f"(range {e.min():g} .. {e.max():g})."))
    if np.any(h["cdp"] != 0) and len(np.unique(h["cdp"])) > 1:
        _, fold = np.unique(h["cdp"], return_counts=True)
        out.append(("INFO", f"CDP fold min/median/max = {fold.min()}/{int(np.median(fold))}/{fold.max()}."))
    return out


def report_text(seg: SegyFile, result: QCResult) -> str:
    lines = ["SEISGAMA QC REPORT", "=" * 60]
    for k, v in seg.summary().items():
        lines.append(f"{k:28s}: {v}")
    lines += ["", "Trace QC summary", "-" * 60]
    for k, v in result.summary.items():
        label = k
        if k.startswith("flag_"):
            label = f"{k} ({FLAG_DESCRIPTIONS.get(k[5:], '')})"
        lines.append(f"{label:28s}: {v}")
    lines += ["", "Header checks", "-" * 60]
    lines += [f"[{lvl}] {msg}" for lvl, msg in result.checks] or ["No problems found."]
    idx = result.flagged_indices()
    lines += ["", f"Flagged traces ({len(idx)})", "-" * 60]
    h = seg.headers
    for i in idx[:500]:
        lines.append(f"trace {i + 1:7d}  fldr {h['fldr'][i]:6d}  tracf {h['tracf'][i]:5d}  "
                     f"offset {h['offset'][i]:8d}  : {', '.join(result.reasons(i))}")
    if len(idx) > 500:
        lines.append(f"... {len(idx) - 500} more (see CSV)")
    return "\n".join(lines) + "\n"


def write_csv(seg: SegyFile, result: QCResult, path) -> None:
    h = seg.headers
    hdr_cols = ["fldr", "tracf", "ep", "cdp", "offset", "trid"]
    attr_cols = list(result.attributes)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trace"] + hdr_cols + ["sx", "sy", "gx", "gy", "selev", "gelev"] + attr_cols + ["flags"])
        coords = {k: seg.scaled(k) for k in ("sx", "sy", "gx", "gy", "selev", "gelev")}
        for i in range(seg.n_traces):
            w.writerow([i + 1] + [int(h[c][i]) for c in hdr_cols]
                       + [f"{coords[k][i]:.3f}" for k in coords]
                       + [f"{result.attributes[c][i]:.6g}" for c in attr_cols]
                       + [";".join(result.reasons(i))])
