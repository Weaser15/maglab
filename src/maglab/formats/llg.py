from pathlib import Path
import struct

import numpy as np


def read_llg_movie_header(filepath: Path | str) -> dict:
    """Return header metadata for LLG micromagnetics simulator file .llg_movie"""
    filepath = Path(filepath)

    header = {}
    with filepath.open(mode="rb") as file:
        # Parse through the version and boilerplate
        ver = file.read(64).decode("utf=8")
        if "LLG" not in ver.upper():
            raise ValueError("File is not LLG Movie!")
        header["nx"], header["ny"], header["nz"] = struct.unpack("<3i", file.read(12))
        file.seek(4 + 24, 1)  # Skip some
        header["dx"], header["dy"], header["dz"] = struct.unpack("<3d", file.read(24))
        nnodes = header["dx"] * header["dy"] * header["dz"]
    # Header is all of the things that were parsed here as well as some extra unnecessary stuff.
    header["header_length"] = 64 + 16 + 48 + nnodes * 3 * 8
    return header


def read_llg_movie(filepath: Path | str) -> tuple[np.ndarray, dict]:
    """Return magnetisation array data and header metadata for
    LLG micromagnetics simulator file .llg_movie."""
    filepath = Path(filepath)

    header = read_llg_movie_header(filepath)
    nx, ny, nz = header["nx"], header["ny"], header["nz"]
    nnodes = nx * ny * nz
    # Each frame contains metadata in addition to the array data
    bytes_per_frame = 4 + 8 + (3 * 8) + (3 * 8) + (4 * 8) + (3 * nnodes * 8)
    hcount = (filepath.stat().st_size - header["header_length"]) // bytes_per_frame
    # For each field state, there is some stuff, the applied field, average magnetisation and
    # Magnetisation for each cell.
    fmt = np.dtype(
        [
            ("stuff", "V12"),
            ("field", "<f8", 3),
            ("mag_avg", "<f8", 3),
            ("mag", "<f8", (nnodes, 3)),
        ]
    )
    data = np.fromfile(filepath, dtype=fmt, count=hcount, offset=header["header_length"])
    header["field"] = data["field"]
    header["mag_avg"] = data["mag_avg"]
    return data["mag"], header


def read_llg_movie_array(filepath: Path | str) -> np.ndarray:
    """Return magnetisation array data for LLG micromagnetics simulator file .llg_movie."""
    return read_llg_movie(filepath)[0]
