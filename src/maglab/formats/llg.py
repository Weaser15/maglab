from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .header import Header


def read_llg_movie_header(filepath: Path | str) -> Header:
    """Return header metadata for LLG micromagnetics simulator file .llg_movie"""
    filepath = Path(filepath)

    with filepath.open(mode="rb") as file:
        # Parse through the version and boilerplate
        ver = file.read(64).decode("utf=8")
        if "LLG" not in ver.upper():
            raise ValueError("File is not LLG Movie!")
        nx, ny, nz = struct.unpack("<3i", file.read(12))
        file.seek(4 + 24, 1)  # Skip some
        dx, dy, dz = struct.unpack("<3d", file.read(24))
    valuedim = 3
    nnodes = nx * ny * nz * valuedim
    # Header is all of the things that were parsed here as well as some extra unnecessary stuff.
    header_length = 64 + 16 + 48 + nnodes * 8

    return Header(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        valuedim=valuedim,
        value="m",
        value_unit="1",
        header_length=header_length,
    )


def read_llg_movie(filepath: Path | str) -> tuple[np.ndarray, Header, np.ndarray, np.ndarray]:
    """Return magnetisation array data and header metadata for
    LLG micromagnetics simulator file .llg_movie."""
    filepath = Path(filepath)

    header = read_llg_movie_header(filepath)
    nx, ny, nz, valuedim = header.get_dims()
    nnodes = nx * ny * nz * valuedim
    # Each frame contains metadata in addition to the array data
    bytes_per_frame = 4 + 8 + (valuedim * 8) + (valuedim * 8) + (4 * 8) + (nnodes * 8)
    hcount = (filepath.stat().st_size - header.header_length) // bytes_per_frame
    # For each field state, there is some stuff, the applied field, average magnetisation and
    # Magnetisation for each cell.
    dtype = np.dtype(
        [
            ("stuff", "V12"),
            ("field", "<f8", 3),
            ("mag_avg", "<f8", 3),
            ("mag", "<f8", (nnodes, 3)),
        ]
    )
    data = np.fromfile(filepath, dtype=dtype, count=hcount, offset=header.header_length)
    return data["mag"].reshape(nz, ny, nx, 3), header, data["field"], data["mag_avg"]


def read_llg_movie_array(filepath: Path | str) -> np.ndarray:
    """Return magnetisation array data for LLG micromagnetics simulator file .llg_movie."""
    return read_llg_movie(filepath)[0]
