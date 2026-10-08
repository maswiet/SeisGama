"""SEG-Y reader/writer (rev 0, rev 1 and the common parts of rev 2).

Design goals
------------
* Pure numpy, no compiled extensions -> runs identically on Linux, macOS and Windows.
* Large files: trace samples are memory-mapped and only converted on demand,
  trace headers are read once in chunks into a compact native-endian table.
* Robust against the header errors that are common in field data
  (wrong/zero ``ns`` or ``dt`` in the binary header, little-endian files,
  ASCII instead of EBCDIC text header, extended textual headers,
  truncated last trace).  Every automatic correction is recorded in
  ``SegyFile.warnings`` so the QC operator can see it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

TEXT_HEADER_SIZE = 3200
BINARY_HEADER_SIZE = 400
TRACE_HEADER_SIZE = 240

# --------------------------------------------------------------------------
# Header definitions (byte positions are 1-based, as written in the standard)
# --------------------------------------------------------------------------

#: (name, first byte, numpy type, description)
TRACE_HEADER_FIELDS: list[tuple[str, int, str, str]] = [
    ("tracl", 1, "i4", "Trace sequence number within line"),
    ("tracr", 5, "i4", "Trace sequence number within file"),
    ("fldr", 9, "i4", "Original field record number (FFID)"),
    ("tracf", 13, "i4", "Trace number within field record (channel)"),
    ("ep", 17, "i4", "Energy source point number"),
    ("cdp", 21, "i4", "CDP/CMP ensemble number"),
    ("cdpt", 25, "i4", "Trace number within CDP ensemble"),
    ("trid", 29, "i2", "Trace identification code"),
    ("nvs", 31, "i2", "Number of vertically summed traces"),
    ("nhs", 33, "i2", "Number of horizontally stacked traces"),
    ("duse", 35, "i2", "Data use (1=production, 2=test)"),
    ("offset", 37, "i4", "Source-receiver offset"),
    ("gelev", 41, "i4", "Receiver group elevation"),
    ("selev", 45, "i4", "Surface elevation at source"),
    ("sdepth", 49, "i4", "Source depth below surface"),
    ("gdel", 53, "i4", "Datum elevation at receiver group"),
    ("sdel", 57, "i4", "Datum elevation at source"),
    ("swdep", 61, "i4", "Water depth at source"),
    ("gwdep", 65, "i4", "Water depth at group"),
    ("scalel", 69, "i2", "Scalar for elevations and depths"),
    ("scalco", 71, "i2", "Scalar for coordinates"),
    ("sx", 73, "i4", "Source X coordinate"),
    ("sy", 77, "i4", "Source Y coordinate"),
    ("gx", 81, "i4", "Group X coordinate"),
    ("gy", 85, "i4", "Group Y coordinate"),
    ("counit", 89, "i2", "Coordinate units"),
    ("wevel", 91, "i2", "Weathering velocity"),
    ("swevel", 93, "i2", "Subweathering velocity"),
    ("sut", 95, "i2", "Uphole time at source (ms)"),
    ("gut", 97, "i2", "Uphole time at group (ms)"),
    ("sstat", 99, "i2", "Source static correction (ms)"),
    ("gstat", 101, "i2", "Group static correction (ms)"),
    ("tstat", 103, "i2", "Total static applied (ms)"),
    ("laga", 105, "i2", "Lag time A (ms)"),
    ("lagb", 107, "i2", "Lag time B (ms)"),
    ("delrt", 109, "i2", "Delay recording time (ms)"),
    ("muts", 111, "i2", "Mute time start (ms)"),
    ("mute", 113, "i2", "Mute time end (ms)"),
    ("ns", 115, "u2", "Number of samples in this trace"),
    ("dt", 117, "u2", "Sample interval (microseconds)"),
    ("gain", 119, "i2", "Gain type of field instruments"),
    ("igc", 121, "i2", "Instrument gain constant (dB)"),
    ("igi", 123, "i2", "Instrument early/initial gain (dB)"),
    ("corr", 125, "i2", "Correlated (1=no, 2=yes)"),
    ("sfs", 127, "i2", "Sweep frequency at start (Hz)"),
    ("sfe", 129, "i2", "Sweep frequency at end (Hz)"),
    ("slen", 131, "i2", "Sweep length (ms)"),
    ("styp", 133, "i2", "Sweep type"),
    ("stas", 135, "i2", "Sweep taper length at start (ms)"),
    ("stae", 137, "i2", "Sweep taper length at end (ms)"),
    ("tatyp", 139, "i2", "Taper type"),
    ("afilf", 141, "i2", "Alias filter frequency (Hz)"),
    ("afils", 143, "i2", "Alias filter slope (dB/oct)"),
    ("nofilf", 145, "i2", "Notch filter frequency (Hz)"),
    ("nofils", 147, "i2", "Notch filter slope (dB/oct)"),
    ("lcf", 149, "i2", "Low-cut frequency (Hz)"),
    ("hcf", 151, "i2", "High-cut frequency (Hz)"),
    ("lcs", 153, "i2", "Low-cut slope (dB/oct)"),
    ("hcs", 155, "i2", "High-cut slope (dB/oct)"),
    ("year", 157, "i2", "Year data recorded"),
    ("day", 159, "i2", "Day of year"),
    ("hour", 161, "i2", "Hour of day"),
    ("minute", 163, "i2", "Minute of hour"),
    ("sec", 165, "i2", "Second of minute"),
    ("timbas", 167, "i2", "Time basis code"),
    ("trwf", 169, "i2", "Trace weighting factor"),
    ("grnors", 171, "i2", "Geophone group number of roll switch position one"),
    ("grnofr", 173, "i2", "Geophone group number of first trace of original record"),
    ("grnlof", 175, "i2", "Geophone group number of last trace of original record"),
    ("gaps", 177, "i2", "Gap size (total number of groups dropped)"),
    ("otrav", 179, "i2", "Overtravel taper code"),
    ("cdpx", 181, "i4", "CDP X coordinate (rev1)"),
    ("cdpy", 185, "i4", "CDP Y coordinate (rev1)"),
    ("iline", 189, "i4", "Inline number (rev1)"),
    ("xline", 193, "i4", "Crossline number (rev1)"),
    ("sp", 197, "i4", "Shotpoint number (rev1)"),
    ("scalsp", 201, "i2", "Scalar for shotpoint number (rev1)"),
    ("trunit", 203, "i2", "Trace value measurement unit (rev1)"),
]

#: (name, first byte in file, numpy type, description)
BINARY_HEADER_FIELDS: list[tuple[str, int, str, str]] = [
    ("jobid", 3201, "i4", "Job identification number"),
    ("lino", 3205, "i4", "Line number"),
    ("reno", 3209, "i4", "Reel number"),
    ("ntrpr", 3213, "i2", "Number of data traces per ensemble"),
    ("nart", 3215, "i2", "Number of auxiliary traces per ensemble"),
    ("hdt", 3217, "u2", "Sample interval (microseconds)"),
    ("dto", 3219, "u2", "Sample interval of original field recording (us)"),
    ("hns", 3221, "u2", "Number of samples per data trace"),
    ("nso", 3223, "u2", "Number of samples per trace, original recording"),
    ("format", 3225, "i2", "Data sample format code"),
    ("fold", 3227, "i2", "Ensemble fold"),
    ("tsort", 3229, "i2", "Trace sorting code"),
    ("vscode", 3231, "i2", "Vertical sum code"),
    ("hsfs", 3233, "i2", "Sweep frequency at start (Hz)"),
    ("hsfe", 3235, "i2", "Sweep frequency at end (Hz)"),
    ("hslen", 3237, "i2", "Sweep length (ms)"),
    ("hstyp", 3239, "i2", "Sweep type code"),
    ("schn", 3241, "i2", "Trace number of sweep channel"),
    ("hstas", 3243, "i2", "Sweep trace taper length at start (ms)"),
    ("hstae", 3245, "i2", "Sweep trace taper length at end (ms)"),
    ("htatyp", 3247, "i2", "Taper type"),
    ("hcorr", 3249, "i2", "Correlated data traces"),
    ("bgrcv", 3251, "i2", "Binary gain recovered"),
    ("rcvm", 3253, "i2", "Amplitude recovery method"),
    ("mfeet", 3255, "i2", "Measurement system (1=m, 2=ft)"),
    ("polyt", 3257, "i2", "Impulse signal polarity"),
    ("vpol", 3259, "i2", "Vibratory polarity code"),
    ("ext_ns", 3269, "i4", "Extended number of samples (rev2)"),
    ("rev", 3501, "u1", "SEG-Y format revision (major)"),
    ("revminor", 3502, "u1", "SEG-Y format revision (minor)"),
    ("trflag", 3503, "i2", "Fixed length trace flag"),
    ("exth", 3505, "i2", "Number of extended textual headers"),
]

#: data sample format code -> (numpy dtype without byte order, bytes per sample, name)
SAMPLE_FORMATS: dict[int, tuple[str, int, str]] = {
    1: ("u4", 4, "4-byte IBM floating point"),
    2: ("i4", 4, "4-byte two's complement integer"),
    3: ("i2", 2, "2-byte two's complement integer"),
    5: ("f4", 4, "4-byte IEEE floating point"),
    6: ("f8", 8, "8-byte IEEE floating point"),
    8: ("i1", 1, "1-byte two's complement integer"),
    9: ("i8", 8, "8-byte two's complement integer"),
    10: ("u4", 4, "4-byte unsigned integer"),
    11: ("u2", 2, "2-byte unsigned integer"),
    12: ("u8", 8, "8-byte unsigned integer"),
    16: ("u1", 1, "1-byte unsigned integer"),
}

ELEVATION_FIELDS = ("gelev", "selev", "sdepth", "gdel", "sdel", "swdep", "gwdep")
COORDINATE_FIELDS = ("sx", "sy", "gx", "gy", "cdpx", "cdpy")


class SegyError(Exception):
    """Raised when a file cannot be interpreted as SEG-Y."""


def _header_dtype(fields, endian: str, itemsize: int, base: int = 0) -> np.dtype:
    return np.dtype(
        {
            "names": [f[0] for f in fields],
            "formats": [endian + f[2] if f[2] not in ("u1", "i1") else f[2] for f in fields],
            "offsets": [f[1] - 1 - base for f in fields],
            "itemsize": itemsize,
        }
    )


def trace_header_dtype(endian: str = ">") -> np.dtype:
    """240-byte trace header as a numpy structured dtype."""
    return _header_dtype(TRACE_HEADER_FIELDS, endian, TRACE_HEADER_SIZE)


NATIVE_TRACE_HEADER_DTYPE = np.dtype([(f[0], f[2]) for f in TRACE_HEADER_FIELDS])


def empty_trace_headers(n: int) -> np.ndarray:
    """Zeroed native-endian trace header table for ``n`` traces."""
    return np.zeros(n, dtype=NATIVE_TRACE_HEADER_DTYPE)


# --------------------------------------------------------------------------
# IBM System/360 floating point
# --------------------------------------------------------------------------

def ibm2ieee(words: np.ndarray) -> np.ndarray:
    """Convert IBM single precision floats (given as uint32 words) to float32."""
    u = np.asarray(words).astype(np.uint32, copy=False)
    sign = np.where((u >> 31).astype(bool), -1.0, 1.0)
    expo = ((u >> 24) & 0x7F).astype(np.int64) - 64
    mant = (u & 0x00FFFFFF).astype(np.float64)
    # value = sign * mant/2^24 * 16^expo = sign * mant * 2^(4*expo - 24)
    with np.errstate(over="ignore"):
        out = sign * np.ldexp(mant, (4 * expo - 24).astype(np.int32))
    return out.astype(np.float32)


def ieee2ibm(values: np.ndarray) -> np.ndarray:
    """Convert floats to IBM single precision words (uint32), rounding to nearest."""
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    sign = (x < 0).astype(np.uint32) << 31
    a = np.abs(x)
    m, e2 = np.frexp(a)  # a = m * 2**e2, 0.5 <= m < 1
    e16 = -((-e2) // 4)  # ceil(e2/4)
    shift = 4 * e16 - e2  # 0..3
    mant = np.rint(np.ldexp(m, (24 - shift).astype(np.int32))).astype(np.int64)
    carry = mant >= (1 << 24)
    mant = np.where(carry, mant >> 4, mant)
    e16 = e16 + carry
    expo = e16 + 64
    # underflow -> 0, overflow -> largest IBM number
    under = (expo < 0) | (a == 0)
    over = expo > 127
    mant = np.where(under, 0, np.where(over, 0xFFFFFF, mant))
    expo = np.where(under, 0, np.where(over, 127, expo))
    words = sign | (expo.astype(np.uint32) << 24) | mant.astype(np.uint32)
    return np.where(under, np.uint32(0), words).astype(np.uint32)


# --------------------------------------------------------------------------
# Text header
# --------------------------------------------------------------------------

def _printable_score(text: str) -> int:
    return sum(ch.isalnum() or ch in " .,:;-_/()=+*#'\"" for ch in text)


def decode_text_header(raw: bytes) -> tuple[str, str]:
    """Decode a 3200-byte textual header. Returns (text with newlines, encoding)."""
    ebc = raw.decode("cp037", errors="replace")
    asc = raw.decode("latin-1", errors="replace")
    if _printable_score(ebc) > _printable_score(asc):
        text, enc = ebc, "ebcdic"
    else:
        text, enc = asc, "ascii"
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    lines = [text[i : i + 80].rstrip() for i in range(0, len(text), 80)]
    return "\n".join(lines), enc


def encode_text_header(text: str, ebcdic: bool = True) -> bytes:
    """Encode text into a 3200-byte (40 x 80) textual header."""
    lines = text.splitlines()[:40]
    lines += [""] * (40 - len(lines))
    card = "".join(line[:80].ljust(80) for line in lines)
    card = card.encode("ascii", errors="replace").decode("ascii")
    return card.encode("cp037" if ebcdic else "ascii")


def default_text_header(extra: Iterable[str] = ()) -> str:
    lines = ["C 1 SEG-Y written by SEISGAMA", "C 2"]
    for i, line in enumerate(extra, start=len(lines) + 1):
        lines.append(f"C{i:2d} {line}")
    while len(lines) < 39:
        lines.append(f"C{len(lines) + 1:2d}")
    lines.append("C40 END TEXTUAL HEADER")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------

@dataclass
class SegyFile:
    """An opened SEG-Y file.

    ``headers`` is a native-endian structured array with one row per trace.
    Sample values are read lazily with :meth:`read_traces` / :meth:`read_all`.
    """

    path: str
    text_header: str
    text_encoding: str
    binary_header: dict
    binary_raw: bytes
    headers: np.ndarray
    endian: str
    format_code: int
    ns: int
    dt: float  # seconds
    data_offset: int
    extended_text: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _mm: np.memmap | None = field(default=None, repr=False)
    _data: np.ndarray | None = field(default=None, repr=False)  # in-memory data (not file backed)

    # ---------------------------------------------------------------- basics
    @property
    def n_traces(self) -> int:
        return len(self.headers)

    @property
    def format_name(self) -> str:
        return SAMPLE_FORMATS.get(self.format_code, ("", 0, f"code {self.format_code}"))[2]

    @property
    def delay(self) -> float:
        """Recording delay (s) of the first trace (header ``delrt``)."""
        return float(self.headers["delrt"][0]) / 1000.0 if self.n_traces else 0.0

    def delay_of(self, index) -> float:
        """Recording delay (s) of the first trace in ``index``."""
        d = np.atleast_1d(self.headers["delrt"][index])
        return float(d[0]) / 1000.0 if len(d) else 0.0

    @property
    def times(self) -> np.ndarray:
        return self.delay + np.arange(self.ns) * self.dt

    def close(self) -> None:
        if self._mm is not None:
            mm = self._mm
            self._mm = None
            del mm

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- data
    def read_traces(self, index) -> np.ndarray:
        """Return the samples of the selected traces as float32 (ntr, ns)."""
        if self._data is not None:
            out = self._data[index]
            return np.array(out, dtype=np.float32, copy=True, ndmin=2)
        if self._mm is None:
            raise SegyError("File is closed")
        raw = np.asarray(self._mm["d"][index])
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        return _convert_samples(raw, self.format_code)

    def read_all(self) -> np.ndarray:
        return self.read_traces(slice(None))

    def iter_chunks(self, chunk: int = 4096):
        """Yield (start, stop, data) over all traces, ``chunk`` traces at a time."""
        for start in range(0, self.n_traces, chunk):
            stop = min(start + chunk, self.n_traces)
            yield start, stop, self.read_traces(slice(start, stop))

    # ---------------------------------------------------------------- headers
    def header(self, name: str, index=slice(None)) -> np.ndarray:
        return self.headers[name][index]

    def scaled(self, name: str, index=slice(None)) -> np.ndarray:
        """Header value with the SEG-Y scalar applied (coordinates/elevations)."""
        vals = self.headers[name][index].astype(np.float64)
        if name in COORDINATE_FIELDS:
            return vals * scalar_factor(self.headers["scalco"][index])
        if name in ELEVATION_FIELDS:
            return vals * scalar_factor(self.headers["scalel"][index])
        return vals

    def offsets(self, index=slice(None)) -> np.ndarray:
        """Signed offsets; computed from coordinates when the header is empty."""
        off = self.headers["offset"][index].astype(np.float64)
        if np.all(off == 0):
            sx, sy = self.scaled("sx", index), self.scaled("sy", index)
            gx, gy = self.scaled("gx", index), self.scaled("gy", index)
            src_ok = np.any(sx != 0) or np.any(sy != 0)
            rec_ok = np.any(gx != 0) or np.any(gy != 0)
            if src_ok and rec_ok:
                return np.hypot(gx - sx, gy - sy)
        return off

    def gather_values(self, key: str) -> np.ndarray:
        return np.unique(self.headers[key])

    def gather_index(self, key: str, value) -> np.ndarray:
        return np.flatnonzero(self.headers[key] == value)

    def summary(self) -> dict:
        h = self.headers
        info = {
            "file": os.path.basename(self.path),
            "size_mb": round(os.path.getsize(self.path) / 1e6, 2) if os.path.exists(self.path) else None,
            "traces": self.n_traces,
            "samples_per_trace": self.ns,
            "sample_interval_ms": round(self.dt * 1000, 6),
            "record_length_ms": round((self.ns - 1) * self.dt * 1000, 3),
            "format": f"{self.format_code} ({self.format_name})",
            "byte_order": "big-endian" if self.endian == ">" else "little-endian",
            "text_header_encoding": self.text_encoding,
            "segy_revision": f"{self.binary_header.get('rev', 0)}.{self.binary_header.get('revminor', 0)}",
            "extended_text_headers": len(self.extended_text),
        }
        if self.n_traces:
            for key in ("fldr", "cdp", "tracf"):
                u = np.unique(h[key])
                info[f"{key}_range"] = f"{u.min()} .. {u.max()} ({len(u)} unique)"
            off = self.offsets()
            info["offset_range"] = f"{off.min():g} .. {off.max():g}"
        return info


def scalar_factor(scalar) -> np.ndarray:
    """SEG-Y scalar convention: >0 multiply, <0 divide, 0 -> 1."""
    s = np.asarray(scalar, dtype=np.float64)
    return np.where(s > 0, s, np.where(s < 0, 1.0 / np.where(s == 0, 1, -s), 1.0))


def _convert_samples(raw: np.ndarray, format_code: int) -> np.ndarray:
    if format_code == 1:
        return ibm2ieee(raw)
    return raw.astype(np.float32)


def _detect_endian(binary: bytes) -> tuple[str, int]:
    b = binary[24:26]
    big = int.from_bytes(b, "big", signed=True)
    little = int.from_bytes(b, "little", signed=True)
    if big in SAMPLE_FORMATS or big == 4:
        return ">", big
    if little in SAMPLE_FORMATS or little == 4:
        return "<", little
    raise SegyError(
        f"Unknown data sample format code (bytes 3225-3226 = {big} big-endian / {little} little-endian). "
        "The file is probably not SEG-Y or the binary header is corrupt; "
        "you can force the format with the 'format_code' option."
    )


def _guess_endian(binary: bytes) -> str:
    """Byte order from the plausibility of dt and ns when the format code is unusable."""
    def plausible(order):
        dt = int.from_bytes(binary[16:18], order)
        ns = int.from_bytes(binary[20:22], order)
        return (0 < dt <= 20000) + (0 < ns <= 40000)
    return "<" if plausible("little") > plausible("big") else ">"


def _looks_like_text(block: bytes) -> bool:
    """True if a 3200-byte block is EBCDIC or ASCII text (not binary trace data)."""
    if len(block) != TEXT_HEADER_SIZE:
        return False
    if not block.strip(b"\x00"):  # empty block (some writers fill with zero bytes)
        return True
    best = max(sum(ch.isprintable() for ch in block.decode(enc, errors="replace")) for enc in ("cp037", "latin-1"))
    return best > 0.9 * TEXT_HEADER_SIZE


def read_binary_header(raw: bytes, endian: str) -> dict:
    """Parse the 400-byte binary header (raw bytes start at file byte 3201)."""
    dt = _header_dtype(BINARY_HEADER_FIELDS, endian, BINARY_HEADER_SIZE, base=TEXT_HEADER_SIZE)
    rec = np.frombuffer(raw, dtype=dt, count=1)[0]
    return {name: int(rec[name]) for name, *_ in BINARY_HEADER_FIELDS}


def open_segy(
    path: str | os.PathLike,
    *,
    endian: str | None = None,
    format_code: int | None = None,
    ns: int | None = None,
    dt_us: float | None = None,
    ignore_extended_headers: bool = False,
) -> SegyFile:
    """Open a SEG-Y file.

    Parameters allow overriding wrong binary-header values (common in field data).
    """
    path = os.fspath(path)
    size = os.path.getsize(path)
    if size < TEXT_HEADER_SIZE + BINARY_HEADER_SIZE:
        raise SegyError(f"File too small to be SEG-Y ({size} bytes)")
    with open(path, "rb") as fh:
        text_raw = fh.read(TEXT_HEADER_SIZE)
        bin_raw = fh.read(BINARY_HEADER_SIZE)
        warn: list[str] = []

        if endian is None or format_code is None:
            try:
                det_endian, det_format = _detect_endian(bin_raw)
            except SegyError:
                if format_code is None:
                    raise
                det_endian, det_format = (endian or _guess_endian(bin_raw)), format_code
        else:
            det_endian, det_format = endian, format_code
        endian = endian or det_endian
        bh = read_binary_header(bin_raw, endian)
        fmt = format_code if format_code is not None else (det_format if endian == det_endian else bh["format"])
        if fmt == 4:
            raise SegyError("Sample format 4 (fixed point with gain) is obsolete and not supported")
        if fmt not in SAMPLE_FORMATS:
            raise SegyError(f"Unsupported sample format code {fmt}")
        if endian == "<":
            warn.append("File is little-endian (non-standard but supported).")

        text, enc = decode_text_header(text_raw)

        # Extended textual headers.  Officially rev1+, but many writers (segyio among
        # them) set the count with rev = 0, or store the revision as 16-bit 0x0001.
        # Accept them whenever the blocks fit in the file and actually contain text.
        ext_texts: list[str] = []
        data_offset = TEXT_HEADER_SIZE + BINARY_HEADER_SIZE
        n_ext = bh["exth"]
        if n_ext and ignore_extended_headers:
            warn.append(f"{n_ext} extended textual header(s) ignored on request.")
        elif n_ext > 0:
            if data_offset + n_ext * TEXT_HEADER_SIZE + TRACE_HEADER_SIZE > size:
                warn.append(f"Binary header announces {n_ext} extended textual headers, more than the file "
                            "can hold; ignored.")
            else:
                blocks = [fh.read(TEXT_HEADER_SIZE) for _ in range(n_ext)]
                if bh["rev"] >= 1 or all(_looks_like_text(b) for b in blocks):
                    ext_texts = [decode_text_header(b)[0] for b in blocks]
                    if bh["rev"] < 1:
                        warn.append(f"{n_ext} extended textual header(s) found although the revision is 0.")
                else:
                    warn.append(f"Binary header announces {n_ext} extended textual headers but the revision is 0 "
                                "and the blocks are not text; ignored.")
                    fh.seek(data_offset)
        elif n_ext == -1 and bh["rev"] >= 1:  # variable number, terminated by an EndText stanza
            pos = data_offset
            while pos + TEXT_HEADER_SIZE + TRACE_HEADER_SIZE <= size and len(ext_texts) < 1000:
                t = decode_text_header(fh.read(TEXT_HEADER_SIZE))[0]
                ext_texts.append(t)
                pos += TEXT_HEADER_SIZE
                if "ENDTEXT" in t.upper().replace(" ", ""):
                    break
            else:
                raise SegyError("Could not find the end ((SEG: EndText)) of the extended textual headers")
        elif n_ext < 0:
            warn.append(f"Invalid extended header count {n_ext}; ignored.")
        data_offset += TEXT_HEADER_SIZE * len(ext_texts)
        fh.seek(data_offset)

        first_th = fh.read(TRACE_HEADER_SIZE) if size >= data_offset + TRACE_HEADER_SIZE else b""

        def second_trace_ns(n: int) -> int | None:
            """ns field of the 2nd trace header if traces had n samples (None if beyond EOF)."""
            pos = data_offset + TRACE_HEADER_SIZE + n * SAMPLE_FORMATS[fmt][1]
            if pos + TRACE_HEADER_SIZE > size:
                return None
            fh.seek(pos + 114)
            return int.from_bytes(fh.read(2), "big" if endian == ">" else "little")

        bps = SAMPLE_FORMATS[fmt][1]
        payload = size - data_offset

        def fits(n: int) -> bool:
            return n > 0 and payload % (TRACE_HEADER_SIZE + n * bps) == 0

        def consistent(n: int) -> bool:
            n2 = second_trace_ns(n)
            return n2 is None or n2 == 0 or n2 == n

        th_dtype = trace_header_dtype(endian)
        first = np.frombuffer(first_th, dtype=th_dtype, count=1)[0] if len(first_th) == TRACE_HEADER_SIZE else None

        # --- number of samples
        if ns is None:
            candidates = []
            if bh["hns"]:
                candidates.append(("binary header", bh["hns"]))
            if bh["ext_ns"] > 0 and bh["rev"] >= 2:
                candidates.append(("binary header (extended)", bh["ext_ns"]))
            if first is not None and int(first["ns"]):
                candidates.append(("first trace header", int(first["ns"])))
            if not candidates:
                raise SegyError("Number of samples is zero in both binary and trace headers; pass ns= explicitly")
            chosen = (next((c for c in candidates if fits(c[1]) and consistent(c[1])), None)
                      or next((c for c in candidates if fits(c[1])), None)
                      or next((c for c in candidates if consistent(c[1])), candidates[0]))
            if chosen[1] != candidates[0][1]:
                warn.append(
                    f"ns={candidates[0][1]} from {candidates[0][0]} does not match the file; "
                    f"using ns={chosen[1]} from {chosen[0]}."
                )
            ns = chosen[1]
            if first is not None and int(first["ns"]) and int(first["ns"]) != ns:
                warn.append(f"Binary header ns={ns} differs from first trace header ns={int(first['ns'])}.")

    # --- sample interval
    if dt_us is None:
        dt_us = bh["hdt"]
        if first is not None and int(first["dt"]) and dt_us and int(first["dt"]) != dt_us:
            warn.append(f"Binary header dt={dt_us} us differs from trace header dt={int(first['dt'])} us; using binary header.")
        if not dt_us and first is not None:
            dt_us = int(first["dt"])
            if dt_us:
                warn.append(f"Binary header dt is 0; using trace header dt={dt_us} us.")
        if not dt_us:
            raise SegyError("Sample interval is zero in binary and trace headers; pass dt_us= explicitly")

    trace_bytes = TRACE_HEADER_SIZE + ns * bps
    n_traces = payload // trace_bytes
    rest = payload - n_traces * trace_bytes
    if rest:
        warn.append(
            f"File size is not a whole number of traces: {rest} trailing bytes ignored "
            "(truncated last trace or variable trace length)."
        )
    if n_traces <= 0:
        raise SegyError("No complete trace found in file")

    rec_dtype = np.dtype([("h", (np.void, TRACE_HEADER_SIZE)), ("d", (endian + SAMPLE_FORMATS[fmt][0] if bps > 1 else SAMPLE_FORMATS[fmt][0], ns))])
    mm = np.memmap(path, dtype=rec_dtype, mode="r", offset=data_offset, shape=(n_traces,))

    headers = empty_trace_headers(n_traces)
    chunk = max(1, 32 * 1024 * 1024 // trace_bytes)
    for start in range(0, n_traces, chunk):
        stop = min(start + chunk, n_traces)
        raw = np.ascontiguousarray(mm["h"][start:stop]).view(th_dtype).reshape(-1)
        for name in NATIVE_TRACE_HEADER_DTYPE.names:
            headers[name][start:stop] = raw[name]

    seg = SegyFile(
        path=path,
        text_header=text,
        text_encoding=enc,
        binary_header=bh,
        binary_raw=bin_raw,
        headers=headers,
        endian=endian,
        format_code=fmt,
        ns=int(ns),
        dt=float(dt_us) / 1e6,
        data_offset=data_offset,
        extended_text=ext_texts,
        warnings=warn,
        _mm=mm,
    )
    return seg


def from_arrays(data: np.ndarray, dt: float, headers: np.ndarray | None = None, text: str | None = None,
                path: str = "<memory>") -> SegyFile:
    """Build an in-memory SegyFile (useful for synthetic data and processed results)."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (traces x samples)")
    ntr, ns = data.shape
    if headers is None:
        headers = empty_trace_headers(ntr)
        headers["tracl"] = headers["tracr"] = np.arange(1, ntr + 1)
        headers["tracf"] = np.arange(1, ntr + 1)
        headers["fldr"] = 1
    headers = headers.copy()
    headers["ns"] = ns
    headers["dt"] = int(round(dt * 1e6))
    bh = {name: 0 for name, *_ in BINARY_HEADER_FIELDS}
    bh.update(hdt=int(round(dt * 1e6)), hns=ns, format=5, rev=1)
    return SegyFile(path=path, text_header=text or default_text_header(), text_encoding="ascii",
                    binary_header=bh, binary_raw=b"\0" * BINARY_HEADER_SIZE, headers=headers, endian=">",
                    format_code=5, ns=ns, dt=float(dt), data_offset=0, _data=data)


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------

class SegyWriter:
    """Streaming big-endian SEG-Y rev1 writer (traces can be appended in chunks).

    If ``template`` is given, its binary header is copied and only the sample
    interval, number of samples, format, revision and extended-header count are
    updated.
    """

    def __init__(self, path, ns: int, dt: float, *, text_header: str | None = None,
                 format_code: int = 5, template: SegyFile | None = None):
        if format_code not in (1, 5):
            raise ValueError("Only format 1 (IBM float) and 5 (IEEE float) can be written")
        if not 0 < ns <= 65535:
            raise ValueError("SEG-Y rev1 supports 1..65535 samples per trace")
        self.dt_us = int(round(dt * 1e6))
        if not 0 < self.dt_us <= 65535:
            raise ValueError("Sample interval out of range for SEG-Y")
        self.ns = ns
        self.format_code = format_code
        self.n_written = 0
        if text_header is None:
            text_header = template.text_header if template is not None else default_text_header()
        binary = self._binary_header(template)
        self._fh = open(path, "wb")
        try:
            self._fh.write(encode_text_header(text_header))
            self._fh.write(binary)
        except Exception:
            self._fh.close()
            raise

    def _binary_header(self, template: SegyFile | None) -> bytes:
        bdt = _header_dtype(BINARY_HEADER_FIELDS, ">", BINARY_HEADER_SIZE, base=TEXT_HEADER_SIZE)
        if template is not None and len(template.binary_raw) == BINARY_HEADER_SIZE and template.endian == ">":
            rec = np.frombuffer(bytearray(template.binary_raw), dtype=bdt).copy()
        else:
            rec = np.zeros(1, dtype=bdt)
            if template is not None:  # re-encode (e.g. little-endian) template values
                for name in bdt.names:
                    rec[name] = template.binary_header.get(name, 0)
        rec["hdt"] = self.dt_us
        rec["hns"] = self.ns
        rec["format"] = self.format_code
        rec["rev"] = 1
        rec["revminor"] = 0
        rec["trflag"] = 1
        rec["exth"] = 0
        rec["ext_ns"] = 0
        return rec.tobytes()

    def write(self, data: np.ndarray, headers: np.ndarray | None = None) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ntr, ns = data.shape
        if ns != self.ns:
            raise ValueError(f"expected {self.ns} samples per trace, got {ns}")
        th = np.zeros(ntr, dtype=trace_header_dtype(">"))
        if headers is not None:
            if len(headers) != ntr:
                raise ValueError("headers and data have a different number of traces")
            for name in th.dtype.names:
                if name in headers.dtype.names:
                    th[name] = headers[name]
        else:
            seq = self.n_written + np.arange(1, ntr + 1)
            th["tracl"] = th["tracr"] = th["tracf"] = seq
            th["fldr"] = 1
        th["ns"] = ns
        th["dt"] = self.dt_us
        samples = ieee2ibm(data).astype(">u4") if self.format_code == 1 else data.astype(">f4")
        rec = np.empty(ntr, dtype=[("h", th.dtype), ("d", samples.dtype, ns)])
        rec["h"] = th
        rec["d"] = samples
        rec.tofile(self._fh)
        self.n_written += ntr

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_segy(
    path: str | os.PathLike,
    data: np.ndarray,
    dt: float,
    headers: np.ndarray | None = None,
    *,
    text_header: str | None = None,
    format_code: int = 5,
    template: SegyFile | None = None,
) -> None:
    """Write a complete big-endian SEG-Y rev1 file (see :class:`SegyWriter`)."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("data must be 2-D (traces x samples)")
    with SegyWriter(path, data.shape[1], dt, text_header=text_header, format_code=format_code,
                    template=template) as w:
        w.write(data, headers)


def trace_header_table(seg: SegyFile, index: int) -> list[tuple[str, str, int, str]]:
    """Rows (name, bytes, value, description) of one trace header, for display."""
    rows = []
    for name, byte, typ, desc in TRACE_HEADER_FIELDS:
        size = np.dtype(typ).itemsize
        rows.append((name, f"{byte}-{byte + size - 1}", int(seg.headers[name][index]), desc))
    return rows


def binary_header_table(seg: SegyFile) -> list[tuple[str, str, int, str]]:
    rows = []
    for name, byte, typ, desc in BINARY_HEADER_FIELDS:
        size = np.dtype(typ).itemsize
        rows.append((name, f"{byte}-{byte + size - 1}", seg.binary_header.get(name, 0), desc))
    return rows

