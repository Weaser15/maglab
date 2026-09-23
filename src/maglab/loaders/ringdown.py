from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import ringdown
from ..formats import mumax3
from ..loaders.mumax3 import get_mx3_files, load_multiple_ovf_array
from ..loaders.utils import compute_dot_vectors


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
    return ringdown.calc_spectrum(arr).copy()


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
    return ringdown.calc_dispersion(spec).copy()


def load_frequencies(filepath: Path | str) -> np.ndarray:
    time = mumax3.read_table(filepath)["t (s)"].to_numpy()
    return ringdown.calc_frequencies(len(time), np.diff(time).mean())


def load_reciprocal_axes(dirpath: Path | str, comp: str = ""):
    dirpath = Path(dirpath)
    frequencies = load_frequencies(dirpath / "table.txt")

    header = mumax3.read_ovf_header(get_mx3_files(dirpath, comp)[0])
    kx = ringdown.calc_wavevectors(header.nx, header.dx)
    ky = ringdown.calc_wavevectors(header.ny, header.dy)
    kz = ringdown.calc_wavevectors(header.nz, header.dz)
    return frequencies, kx, ky, kz


def load_mag_avg_ringdown(
    filepath: Path | str,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    table = mumax3.read_table(filepath)
    time = table["t (s)"].to_numpy()
    mag = compute_dot_vectors(table, names=["m"], direction=direction)[0]
    return ringdown.avg_mag_ringdown(mag, time)
