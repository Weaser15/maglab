from pathlib import Path

from ..analysis import ringdown
from ..formats import mumax3
from ..loaders.mumax3 import load_multiple_ovf_array
from ..loaders.utils import compute_dot_vectors


def load_mag_avg_ringdown(
    filepath: Path | str,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    table = mumax3.read_table(filepath)
    time = table["t (s)"].to_numpy()
    mag = compute_dot_vectors(table, names=["m"], direction=direction)[0]
    return ringdown.avg_mag_ringdown(mag, time)


def load_ringdown(
    dirpath: Path | str,
    direction=(0.0, 0.0, 1.0),
    zslice: int | slice | list | None = None,
    comp: str = "",
    max_workers: int | None = None,
):
    dirpath = Path(dirpath)

    tablepath = dirpath / "table.txt"
    time = mumax3.read_table(tablepath)["t (s)"].to_numpy()
    arr = load_multiple_ovf_array(
        dirpath=dirpath,
        direction=direction,
        zslice=zslice,
        comp=comp,
        max_workers=max_workers,
    )
    return ringdown.ringdown(arr, time)
