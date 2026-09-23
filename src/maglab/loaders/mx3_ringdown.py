from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import spectral
from ..formats import mumax3
from .mumax3 import get_mx3_files, load_multiple_ovf_array
from .utils import compute_dot_vectors


def load_spec_array(
    dirpath: Path | str,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zslice: int | slice | list | None = None,
    mask: Any = slice(None),
    indexes: list[int] | np.ndarray | None = None,
    comp: str = "",
    max_workers: int | None = None,
):
    arr = load_multiple_ovf_array(dirpath, direction, zslice, mask, indexes, comp, max_workers)
    return spectral.time_to_freq(arr)


def load_dispersion_array(
    dirpath: Path | str,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    zslice: int | slice | list | None = None,
    mask: Any = slice(None),
    indexes: list[int] | np.ndarray | None = None,
    comp: str = "",
    max_workers: int | None = None,
):
    spec = load_spec_array(dirpath, direction, zslice, mask, indexes, comp, max_workers)
    return spectral.real_to_k(spec)


def load_frequencies(filepath: Path | str) -> np.ndarray:
    time = mumax3.read_table(filepath)["t (s)"].to_numpy()
    return spectral.frequencies(len(time), np.diff(time).mean())


def load_reciprocal_axes(dirpath: Path | str, comp: str = ""):
    dirpath = Path(dirpath)
    frequencies = load_frequencies(dirpath / "table.txt")
    header = mumax3.read_ovf_header(get_mx3_files(dirpath, comp)[0])
    kx = spectral.wavevectors(header.nx, header.dx)
    ky = spectral.wavevectors(header.ny, header.dy)
    kz = spectral.wavevectors(header.nz, header.dz)
    return frequencies, kx, ky, kz


def load_table_ringdown(
    filepath: Path | str,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    table = mumax3.read_table(filepath)
    time = table["t (s)"].to_numpy()
    mag = compute_dot_vectors(table, names=["m"], direction=direction)[0]
    return spectral.table_ringdown(mag, time)
