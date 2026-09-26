"""Minimal reader for MoTeC i2 .ld log files (as written by Assetto Corsa loggers).

Layout (little-endian), following the open-source ldparser project:
- file header: u32 marker, 4 pad bytes, u32 pointer to the first channel's metadata, u32 data pointer, ...
- channel metadata is a linked list of 124-byte records:
  prev_ptr, next_ptr, data_ptr, data_len (u32 each), counter, dtype_a, dtype, freq (u16 each),
  shift, mul, scale, dec (i16 each), name[32], short_name[8], unit[12], 40 pad bytes
- value = (raw / scale * 10**-dec + shift) * mul
"""
import struct
from pathlib import Path

import numpy as np
import pandas as pd

_META = struct.Struct("<IIIIHHHHhhhh32s8s12s40x")


def _dtype(dtype_a: int, dtype: int):
    if dtype_a == 0x07:
        return {2: np.float16, 4: np.float32}.get(dtype)
    if dtype_a in (0x00, 0x03, 0x05):
        return {2: np.int16, 4: np.int32}.get(dtype)
    return None


def read_channels(path: str | Path) -> dict[str, tuple[int, np.ndarray]]:
    """Return {channel name: (frequency Hz, values)}."""
    buf = Path(path).read_bytes()
    ptr = struct.unpack_from("<I", buf, 8)[0]
    out: dict[str, tuple[int, np.ndarray]] = {}
    seen = set()
    while ptr and ptr not in seen and ptr + _META.size <= len(buf):
        seen.add(ptr)
        (_prev, nxt, data_ptr, data_len, _cnt, dtype_a, dtype, freq,
         shift, mul, scale, dec, name, _short, _unit) = _META.unpack_from(buf, ptr)
        dt = _dtype(dtype_a, dtype)
        name = name.split(b"\0", 1)[0].decode("latin-1").strip()
        if dt is not None and data_len and freq:
            raw = np.frombuffer(buf, dtype=dt, count=data_len, offset=data_ptr).astype(np.float64)
            vals = (raw / (scale or 1) * 10.0 ** -dec + shift) * (mul or 1)
            out[name] = (int(freq), vals)
        ptr = nxt
    return out


def to_frame(path: str | Path, hz: int | None = None) -> pd.DataFrame:
    """Resample every channel onto one common time base (default: the most common channel rate)."""
    ch = read_channels(path)
    if not ch:
        return pd.DataFrame()
    if hz is None:
        hz = pd.Series([f for f, _ in ch.values()]).mode().iloc[0]
    duration = max(len(v) / f for f, v in ch.values())
    t = np.arange(0, duration, 1.0 / hz)
    cols = {"t": t}
    for name, (f, v) in ch.items():
        tv = np.arange(len(v)) / f
        cols[name] = np.interp(t, tv, v, right=np.nan)
    return pd.DataFrame(cols)
