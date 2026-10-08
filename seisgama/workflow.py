"""Whole-file operations shared by the GUI and the command line."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from . import processing as P
from .segy import SegyFile, SegyWriter

Progress = Callable[[int], "bool | None"]


def contiguous_runs(keys: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index ranges of consecutive traces sharing the same key."""
    keys = np.asarray(keys)
    if len(keys) == 0:
        return []
    edges = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(keys)]])
    return list(zip(starts.tolist(), stops.tolist()))


def _check_cancel(progress: Progress | None, pct: int) -> None:
    if progress is not None and progress(pct):
        raise InterruptedError("cancelled")


def check_not_input(seg: SegyFile, out_path) -> None:
    import os

    if os.path.exists(out_path) and os.path.exists(seg.path) and os.path.samefile(out_path, seg.path):
        raise ValueError("The output file must be different from the input file")


def process_file(seg: SegyFile, out_path, steps: Sequence[dict], gather_key: str = "fldr",
                 format_code: int = 5, progress: Progress | None = None) -> int:
    """Apply a processing flow gather by gather and write a new SEG-Y file.

    Trace order and trace headers are preserved. Returns the number of traces written.
    """
    check_not_input(seg, out_path)
    runs = contiguous_runs(seg.headers[gather_key])
    with SegyWriter(out_path, seg.ns, seg.dt, template=seg, format_code=format_code) as w:
        for i, (a, b) in enumerate(runs):
            data = seg.read_traces(slice(a, b))
            out = P.apply_flow(data, seg.dt, seg.offsets(slice(a, b)), steps, seg.delay_of(a))
            w.write(out, seg.headers[a:b])
            _check_cancel(progress, int(100 * (i + 1) / len(runs)))
        return w.n_written


def brute_stack(seg: SegyFile, steps: Sequence[dict], velocity, stretch_mute: float = 0.5,
                gather_key: str = "fldr", cdp_key: str = "cdp",
                progress: Progress | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-stack flow + NMO with one velocity function + CMP stack.

    ``velocity`` is a scalar or a (times, velocities) tuple.
    Returns (cdp numbers, stacked traces, fold per sample).
    """
    cdps = seg.headers[cdp_key]
    ucdp = np.unique(cdps)
    pos = {c: i for i, c in enumerate(ucdp.tolist())}
    total = np.zeros((len(ucdp), seg.ns))
    fold = np.zeros((len(ucdp), seg.ns), dtype=np.int32)
    runs = contiguous_runs(seg.headers[gather_key])
    delay0 = seg.delay
    for i, (a, b) in enumerate(runs):
        sl = slice(a, b)
        t0 = seg.delay_of(a)
        data = P.apply_flow(seg.read_traces(sl), seg.dt, seg.offsets(sl), steps, t0)
        nmo = P.nmo_correct(data, seg.dt, seg.offsets(sl), velocity, stretch_mute, t0)
        if t0 != delay0:  # align gathers recorded with a different delay to the first one
            shift = int(round((t0 - delay0) / seg.dt))
            nmo = np.roll(nmo, shift, axis=1)
            if shift > 0:
                nmo[:, :shift] = 0
            elif shift < 0:
                nmo[:, shift:] = 0
        rows = np.array([pos[c] for c in cdps[sl].tolist()])
        np.add.at(total, rows, nmo)
        np.add.at(fold, rows, (nmo != 0).astype(np.int32))
        _check_cancel(progress, int(100 * (i + 1) / len(runs)))
    stacked = np.where(fold > 0, total / np.maximum(fold, 1), 0.0).astype(np.float32)
    return ucdp, stacked, fold


def stack_headers(seg: SegyFile, cdps: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Trace headers for a stacked section (keeps the recording delay of the input)."""
    from .segy import empty_trace_headers

    h = empty_trace_headers(len(cdps))
    h["cdp"] = h["ep"] = h["tracf"] = cdps
    h["tracl"] = h["tracr"] = np.arange(1, len(cdps) + 1)
    h["nhs"] = np.minimum(fold.max(axis=1), 32767)
    h["trid"] = 1
    h["delrt"] = seg.headers["delrt"][0] if seg.n_traces else 0
    h["counit"] = seg.headers["counit"][0] if seg.n_traces else 0
    return h


def write_stack(seg: SegyFile, path, cdps, stacked, fold) -> None:
    from .segy import write_segy

    check_not_input(seg, path)
    write_segy(path, stacked, seg.dt, stack_headers(seg, cdps, fold))


def load_flow(path) -> list[dict]:
    """Read a processing flow from JSON: a list of steps or {"steps": [...]}."""
    import json

    with open(path, encoding="utf-8") as fh:
        flow = json.load(fh)
    if isinstance(flow, dict):
        flow = flow.get("steps", [])
    if not isinstance(flow, list) or not all(isinstance(s, dict) and "op" in s for s in flow):
        raise ValueError("flow file must contain a list of steps like {\"op\": \"agc\", \"window\": 0.5}")
    return flow


def save_flow(path, steps: Sequence[dict]) -> None:
    import json

    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"steps": list(steps)}, fh, indent=2)


def read_velocity_picks(path) -> tuple[np.ndarray, np.ndarray]:
    """Two-column text/CSV file: time (s), velocity (m/s). Lines starting with # are ignored."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line[0].isalpha():
                continue
            t, v = line.replace(",", " ").split()[:2]
            rows.append((float(t), float(v)))
    if not rows:
        raise ValueError(f"no velocity picks found in {path}")
    arr = np.array(sorted(rows))
    return arr[:, 0], arr[:, 1]


def write_velocity_picks(path, picks) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("time_s,velocity_ms\n")
        for t, v in sorted(picks):
            fh.write(f"{t:.4f},{v:.1f}\n")
