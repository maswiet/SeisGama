"""Command-line interface.  Run ``seisgama --help``."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import __version__


def _open(args):
    from .segy import open_segy

    return open_segy(args.file, endian=getattr(args, "endian", None), format_code=getattr(args, "format_code", None),
                     ns=getattr(args, "ns", None), dt_us=getattr(args, "dt_us", None))


def _add_open_options(p):
    p.add_argument("file", help="SEG-Y file")
    g = p.add_argument_group("header overrides (for files with wrong headers)")
    g.add_argument("--endian", choices=["<", ">"], help="force byte order")
    g.add_argument("--format-code", type=int, help="force data sample format code")
    g.add_argument("--ns", type=int, help="force number of samples per trace")
    g.add_argument("--dt-us", type=float, help="force sample interval in microseconds")


def _figure(width=11, height=8):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(width, height))


def _gather(seg, key, value):
    if key == "all":
        return np.arange(seg.n_traces)
    if value is None:
        value = seg.gather_values(key)[0]
    idx = seg.gather_index(key, value)
    if len(idx) == 0:
        raise SystemExit(f"No traces with {key} = {value}. Available: {seg.gather_values(key)[:20]} ...")
    return idx


def cmd_info(args):
    from .segy import binary_header_table, trace_header_table

    with _open(args) as seg:
        print(seg.text_header)
        print()
        for k, v in seg.summary().items():
            print(f"{k:24s}: {v}")
        for w in seg.warnings:
            print(f"WARNING: {w}")
        if args.binary:
            print("\nBinary header")
            for name, pos, val, desc in binary_header_table(seg):
                print(f"  {pos:>11s} {name:9s} {val:>10d}  {desc}")
        if args.trace is not None:
            print(f"\nTrace header {args.trace}")
            for name, pos, val, desc in trace_header_table(seg, args.trace - 1):
                print(f"  {pos:>7s} {name:8s} {val:>11d}  {desc}")
    return 0


def cmd_qc(args):
    from . import qc
    from .plotting import plot_qc_overview

    with _open(args) as seg:
        settings = qc.QCSettings(gather_key=args.key, z_threshold=args.z, spike_ratio=args.spike_ratio)
        res = qc.run_qc(seg, settings)
        os.makedirs(args.outdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.file))[0]
        txt = qc.report_text(seg, res)
        rpt = os.path.join(args.outdir, f"{base}_qc_report.txt")
        csvp = os.path.join(args.outdir, f"{base}_qc_traces.csv")
        png = os.path.join(args.outdir, f"{base}_qc_overview.png")
        with open(rpt, "w", encoding="utf-8") as fh:
            fh.write(txt)
        qc.write_csv(seg, res, csvp)
        fig = _figure(12, 9)
        plot_qc_overview(fig, seg, res)
        fig.savefig(png, dpi=120)
        if args.json:
            print(json.dumps({"summary": res.summary, "checks": res.checks}, indent=2))
        else:
            print(txt)
        print(f"Written: {rpt}\n         {csvp}\n         {png}")
        errors = any(level == "ERROR" for level, _ in res.checks)
        return 2 if errors else 0


def cmd_plot(args):
    from . import processing as P
    from .plotting import plot_section, plot_spectrum
    from .workflow import load_flow

    with _open(args) as seg:
        idx = _gather(seg, args.key, args.value)
        data = seg.read_traces(idx)
        t0 = seg.delay_of(idx)
        if args.flow:
            data = P.apply_flow(data, seg.dt, seg.offsets(idx), load_flow(args.flow), t0)
        fig = _figure()
        if args.spectrum:
            ax1, ax2 = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
            f, a = P.amplitude_spectrum(data, seg.dt)
            plot_spectrum(ax2, f, a)
        else:
            ax1 = fig.add_subplot(111)
        title = f"{os.path.basename(args.file)}  {args.key}={args.value if args.key != 'all' else 'all'}"
        plot_section(ax1, data, seg.dt, t0, mode=args.mode, clip_percentile=args.clip, cmap=args.cmap,
                     title=title)
        fig.tight_layout()
        fig.savefig(args.output, dpi=args.dpi)
        print(f"Written: {args.output}")
    return 0


def cmd_process(args):
    from .workflow import load_flow, process_file

    flow = load_flow(args.flow)
    with _open(args) as seg:
        n = process_file(seg, args.output, flow, gather_key=args.key, format_code=args.out_format)
    print(f"Processed {n} traces -> {args.output}")
    return 0


def cmd_velan(args):
    from . import processing as P
    from .plotting import plot_section, plot_semblance
    from .workflow import load_flow, write_velocity_picks

    with _open(args) as seg:
        idx = _gather(seg, args.key, args.value)
        data = seg.read_traces(idx)
        off = seg.offsets(idx)
        t0 = t0_of(idx)
        if args.flow:
            data = P.apply_flow(data, seg.dt, off, load_flow(args.flow), t0)
        vels = np.arange(args.vmin, args.vmax + args.dv / 2, args.dv)
        semb = P.semblance(data, seg.dt, off, vels, args.window, t0)
        picks = P.pick_semblance_maxima(semb, vels, seg.dt, t0, step=args.pick_step,
                                        min_semblance=args.min_semblance)
        fig = _figure(14, 8)
        ax1, ax2, ax3 = fig.subplots(1, 3)
        plot_section(ax1, data, seg.dt, t0, x=off, xlabel="Offset (m)", title="Gather")
        plot_semblance(ax2, semb, vels, seg.dt, t0, picks)
        if picks:
            pt, pv = np.array(picks).T
            nmo = P.nmo_correct(data, seg.dt, off, (pt, pv), args.stretch, t0)
            plot_section(ax3, nmo, seg.dt, t0, x=off, xlabel="Offset (m)", title="NMO (auto picks)")
        fig.tight_layout()
        fig.savefig(args.output, dpi=110)
        print(f"Written: {args.output}")
        for t, v in picks:
            print(f"  t0 = {t:6.3f} s   Vrms = {v:7.1f} m/s")
        if args.picks_out:
            write_velocity_picks(args.picks_out, picks)
            print(f"Picks written: {args.picks_out}")
    return 0


def cmd_stack(args):
    from .plotting import plot_section
    from .workflow import brute_stack, load_flow, read_velocity_picks, write_stack

    flow = load_flow(args.flow) if args.flow else []
    vel = read_velocity_picks(args.velocity) if os.path.exists(args.velocity) else float(args.velocity)
    with _open(args) as seg:
        cdps, st, fold = brute_stack(seg, flow, vel, args.stretch, gather_key=args.key)
        if args.output.lower().endswith((".sgy", ".segy")):
            write_stack(seg, args.output, cdps, st, fold)
        else:
            fig = _figure(12, 7)
            plot_section(fig.add_subplot(111), st, seg.dt, seg.delay, x=cdps, xlabel="CDP",
                         title=f"Brute stack - {os.path.basename(args.file)}")
            fig.tight_layout()
            fig.savefig(args.output, dpi=120)
        print(f"Stacked {len(cdps)} CDPs -> {args.output}")
    return 0


def cmd_synthetic(args):
    from .segy import write_segy
    from .synthetic import survey

    data, h = survey(nshots=args.shots, nrec=args.receivers, noise=args.noise)
    write_segy(args.output, data, 0.002, h, format_code=1,
               text_header="C 1 SEISGAMA SYNTHETIC TEST LINE\nC 2 Reflectors t0/Vrms: 0.30/1600 0.60/1900 0.95/2300 1.30/2700")
    print(f"Written synthetic line ({len(data)} traces): {args.output}")
    return 0


def cmd_gui(args):
    from .gui.app import run

    return run(args.file)


def build_parser() -> argparse.ArgumentParser:
    from . import CITATION

    p = argparse.ArgumentParser(prog="seisgama", description="SeisGama - seismic reflection QC and basic processing",
                                epilog=f"Please cite: {CITATION}  License: CC BY-NC 4.0.")
    p.add_argument("--version", action="version", version=f"seisgama {__version__}")
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("gui", help="start the graphical application (default)")
    g.add_argument("file", nargs="?")
    g.set_defaults(func=cmd_gui)

    i = sub.add_parser("info", help="print text/binary header and file summary")
    _add_open_options(i)
    i.add_argument("--binary", action="store_true", help="print the binary header")
    i.add_argument("--trace", type=int, help="print the header of trace N (1-based)")
    i.set_defaults(func=cmd_info)

    q = sub.add_parser("qc", help="run QC; write report, per-trace CSV and overview plot")
    _add_open_options(q)
    q.add_argument("-o", "--outdir", default=".")
    q.add_argument("--key", default="fldr", help="gather key for amplitude statistics (default fldr)")
    q.add_argument("--z", type=float, default=3.5, help="robust z threshold for noisy/weak traces")
    q.add_argument("--spike-ratio", type=float, default=12.0,
                   help="sample / RMS of neighbouring samples above which a spike is flagged")
    q.add_argument("--json", action="store_true", help="print summary as JSON")
    q.set_defaults(func=cmd_qc)

    pl = sub.add_parser("plot", help="plot a gather to PNG/PDF/SVG")
    _add_open_options(pl)
    pl.add_argument("-o", "--output", required=True)
    pl.add_argument("--key", default="fldr", help="header key selecting the gather, or 'all'")
    pl.add_argument("--value", type=int, help="key value (default: first)")
    pl.add_argument("--flow", help="processing flow JSON applied before plotting")
    pl.add_argument("--mode", choices=["density", "wiggle", "both"], default="density")
    pl.add_argument("--clip", type=float, default=99.0, help="clip percentile")
    pl.add_argument("--cmap", default="gray_r")
    pl.add_argument("--dpi", type=int, default=120)
    pl.add_argument("--spectrum", action="store_true", help="add amplitude spectrum panel")
    pl.set_defaults(func=cmd_plot)

    pr = sub.add_parser("process", help="apply a processing flow (JSON) to the whole file")
    _add_open_options(pr)
    pr.add_argument("-o", "--output", required=True)
    pr.add_argument("--flow", required=True)
    pr.add_argument("--key", default="fldr", help="gather key (for ensemble operations and mute)")
    pr.add_argument("--out-format", type=int, choices=[1, 5], default=5, help="1=IBM float, 5=IEEE float")
    pr.set_defaults(func=cmd_process)

    v = sub.add_parser("velan", help="semblance velocity analysis of one gather")
    _add_open_options(v)
    v.add_argument("-o", "--output", required=True)
    v.add_argument("--key", default="cdp")
    v.add_argument("--value", type=int)
    v.add_argument("--flow", help="processing flow JSON applied first")
    v.add_argument("--vmin", type=float, default=1400)
    v.add_argument("--vmax", type=float, default=4000)
    v.add_argument("--dv", type=float, default=25)
    v.add_argument("--window", type=float, default=0.04, help="semblance window (s)")
    v.add_argument("--stretch", type=float, default=0.5, help="NMO stretch mute")
    v.add_argument("--pick-step", type=float, default=0.15, help="auto-pick time step (s)")
    v.add_argument("--min-semblance", type=float, default=0.2)
    v.add_argument("--picks-out", help="write auto picks to CSV")
    v.set_defaults(func=cmd_velan)

    s = sub.add_parser("stack", help="brute stack (flow + NMO + CMP stack)")
    _add_open_options(s)
    s.add_argument("-o", "--output", required=True, help=".sgy/.segy for SEG-Y, otherwise an image")
    s.add_argument("--velocity", required=True, help="constant velocity (m/s) or picks CSV (time,velocity)")
    s.add_argument("--flow", help="pre-stack processing flow JSON")
    s.add_argument("--stretch", type=float, default=0.5)
    s.add_argument("--key", default="fldr", help="key used to read the data in gathers")
    s.set_defaults(func=cmd_stack)

    sy = sub.add_parser("synthetic", help="write a small synthetic test line")
    sy.add_argument("output")
    sy.add_argument("--shots", type=int, default=6)
    sy.add_argument("--receivers", type=int, default=48)
    sy.add_argument("--noise", type=float, default=0.03)
    sy.set_defaults(func=cmd_synthetic)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["gui", *([] if argv is None else argv)])
    from .segy import SegyError

    try:
        return int(args.func(args) or 0)
    except (SegyError, ValueError, OSError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
