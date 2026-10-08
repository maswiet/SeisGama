# SeisGama

**Free software for seismic reflection data QC and basic processing, for land and marine data.**
Runs on **Windows, macOS and Linux**.

SeisGama was developed by the Seismology Research Group, Physics Department, Universitas Gadjah Mada,
Yogyakarta, Indonesia. The software is described in our paper
([PDF in this repository](docs/Irnaka_2018_SEISGAMA_IJG.pdf)):

> Irnaka, T. M., Wahyudi, W., Hartantyo, E., Mufaqih, A. A., Anggraini, A., & Suryanto, W. (2018).
> SEISGAMA: A Free C# Based Seismic Data Processing Software Platform.
> *International Journal of Geophysics*, 2018, 2913591. https://doi.org/10.1155/2018/2913591

Version 2 is a complete rewrite of the original C#/.NET version in Python, so that it runs on
all major operating systems. It also extends the original with trace QC, filters and deconvolution.

![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)
[![DOI](https://img.shields.io/badge/DOI-10.1155%2F2018%2F2913591-blue.svg)](https://doi.org/10.1155/2018/2913591)

## Features

* **SEG-Y input/output**
  - Reads rev 0/1/2 files: IBM/IEEE float, integer formats, big/little-endian, EBCDIC/ASCII headers and
    extended textual headers.
  - Memory-mapped, so multi-GB files open instantly.
  - Common header errors (ns/dt = 0, wrong ns, truncated files) are detected, corrected and reported.
* **Header inspection**: textual header, binary header, and a table of all trace headers.
* **Seismic viewer**
  - Variable density and wiggle display.
  - Gather selection by FFID, CDP, EP, channel or inline/crossline.
  - Amplitude spectrum, F-X and F-K displays.
  - Time, trace and amplitude readout under the cursor.
* **Processing flow**
  - DC removal, t^n gain, exponential gain, AGC and trace balance.
  - Ormsby band-pass and 50/60 Hz notch filters.
  - Top/bottom mute.
  - Wiener spiking/predictive deconvolution.
  - NMO correction.

  Flows are saved as JSON and can be applied to the whole file, with export to SEG-Y.
* **Trace QC**
  - Detects dead, noisy, weak, spiky, clipped, NaN/Inf and auxiliary traces.
  - Header checks: ns/dt consistency, coordinate/elevation scalars, traces per shot, duplicates,
    offset vs. coordinates, elevation outliers and CDP fold.
  - Exports a report (TXT), a per-trace table (CSV) and overview plots (PNG).
* **Velocity analysis**: semblance, automatic and mouse picking, NMO preview.
* **Brute stack**: flow + NMO + CMP stack, exported to SEG-Y.

## Download and install

### Ready-to-run application (no Python needed)
Go to **[Releases](https://github.com/maswiet/SeisGama/releases)** and download the file for your system:

| System | File | Start |
|---|---|---|
| Windows 10/11 | `SeisGama-Windows.zip` | unzip, run `SEISGAMA\SEISGAMA.exe` |
| macOS (Apple Silicon) | `SeisGama-macOS.zip` | unzip, right-click `SEISGAMA.app` → *Open* (first time only) |
| Linux (x86-64) | `SeisGama-Linux.tar.gz` | extract, run `SEISGAMA/SEISGAMA` |

The applications are not code-signed. Windows SmartScreen may ask for confirmation
(*More info → Run anyway*), and macOS needs the right-click → *Open* the first time.

### From source (Python 3.9 or newer)
```bash
git clone https://github.com/maswiet/SeisGama.git
```
```bash
cd SeisGama
```
```bash
python -m pip install .
```
```bash
seisgama
```
On Linux, the Qt system libraries are needed. On Ubuntu/Debian:
```bash
sudo apt install libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libfontconfig1
```

## Quick start (graphical application)

1. **File ▸ Open SEG-Y…**, or **File ▸ Generate synthetic demo line…** to try SeisGama without your own data.
2. **Headers**: summary and warnings, textual/binary header, trace headers.
3. **Seismic viewer**: choose a gather and add steps to the *Processing flow* (e.g. Band-pass → AGC).
4. **QC**: *Run QC*. Double-click a flagged trace to display it. *Export report…* writes TXT, CSV and PNG.
5. **Velocity analysis**: choose a CDP, then *Compute semblance*. Use *Auto pick*, or left-click the
   semblance to add a pick (right-click deletes one). *Save picks…* writes a CSV file.
6. **Brute stack**: *Compute brute stack*, then *Export stack as SEG-Y…*.

If a file has wrong headers, use **File ▸ Open with header overrides…** to force the byte order,
format code, number of samples or sample interval.

## Command line (batch processing)

```bash
seisgama info line.sgy --binary --trace 1          # headers
seisgama qc line.sgy -o qc_out                     # QC report + CSV + PNG
seisgama plot line.sgy --key fldr --value 1001 --flow examples/flow_land.json --spectrum -o shot.png
seisgama process line.sgy --flow examples/flow_marine.json -o line_proc.sgy
seisgama velan line.sgy --key cdp --value 200 -o velan.png --picks-out picks.csv
seisgama stack line.sgy --velocity picks.csv -o stack.sgy     # or --velocity 2000
seisgama synthetic test.sgy                        # synthetic test line
```
`seisgama qc` exits with code 2 when the headers contain errors, which is useful in automatic QC scripts.

A processing flow is a JSON list of steps. The parameter names are the same as in the application.
See [examples/flow_land.json](examples/flow_land.json) and [examples/flow_marine.json](examples/flow_marine.json):
```json
{"steps": [
  {"op": "tpow", "power": 2.0},
  {"op": "bandpass", "f1": 5, "f2": 10, "f3": 60, "f4": 80},
  {"op": "decon", "operator_length": 0.16, "gap": 0.024, "prewhite": 0.1, "ensemble": false},
  {"op": "agc", "window": 0.5}
]}
```
Available steps: `dc`, `tpow`, `exp`, `agc`, `balance`, `bandpass`, `notch`, `mute`, `decon`, `nmo`.

## How to cite

If you use SeisGama in research, teaching, theses or reports, please cite the paper:

```bibtex
@article{irnaka2018seisgama,
  title   = {{SEISGAMA}: A Free {C\#} Based Seismic Data Processing Software Platform},
  author  = {Irnaka, Theodosius Marwan and Wahyudi, Wahyudi and Hartantyo, Eddy and
             Mufaqih, Adien Akhmad and Anggraini, Ade and Suryanto, Wiwit},
  journal = {International Journal of Geophysics},
  volume  = {2018},
  pages   = {2913591},
  year    = {2018},
  doi     = {10.1155/2018/2913591}
}
```
GitHub also shows a *Cite this repository* button, based on [CITATION.cff](CITATION.cff).

## License

SeisGama is licensed under
**[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)**.

* **Allowed**: using, copying, modifying and sharing SeisGama for **non-commercial** purposes,
  such as research, teaching, study and non-profit work. You must give credit by citing the paper above.
* **Not allowed without permission**: **commercial use**, for example paid services, projects for clients,
  or inclusion in a commercial product. For commercial use, please contact the authors first:
  **Wiwit Suryanto, ws@ugm.ac.id**.

The paper itself is open access under CC BY 4.0. The test data in `tests/data` are under the MIT License
(see [tests/data/README.md](tests/data/README.md)).

## Development

```bash
python -m pip install -e ".[test]"
```
```bash
python -m pytest
```
The test suite runs without a display when `QT_QPA_PLATFORM=offscreen` is set.

To build the stand-alone application into `dist/`:
```bash
python -m pip install -e ".[build]"
```
```bash
python packaging/build.py
```
Every push runs the tests on Windows, macOS and Linux. Pushing a version tag (e.g. `v2.0.1`)
builds the applications and publishes them as a GitHub Release (`.github/workflows/ci.yml`).

Repository layout:
```
seisgama/      source code (segy, processing, qc, workflow, plotting, cli, gui)
tests/         automated tests and example SEG-Y files
examples/      example processing flows
packaging/     PyInstaller build script
docs/          the SEISGAMA paper (PDF)
```

## Ringkasan (Bahasa Indonesia)

SeisGama adalah perangkat lunak gratis untuk QC dan pengolahan awal data seismik refleksi, baik darat
maupun laut. Aplikasi ini berjalan di Windows, macOS dan Linux.

* **Unduh** aplikasi siap pakai di halaman [Releases](https://github.com/maswiet/SeisGama/releases).
* **Lisensi CC BY-NC 4.0**: bebas dipakai untuk riset, pendidikan, dan kegiatan non-komersial, dengan
  menyitasi paper di atas. **Penggunaan komersial wajib meminta izin** ke ws@ugm.ac.id.
