from pathlib import Path

import numpy as np

from ..analysis import ringdown
from ..formats import mumax3


def load_mag_avg_ringdown(
    filepath: Path | str,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    table = mumax3.read_table(filepath)
    time = table["t (s)"].to_numpy()
    mag = np.dot(table[["mx ()", "my ()", "mz ()"]].to_numpy(), direction)
    return ringdown.avg_mag_ringdown(mag, time)
