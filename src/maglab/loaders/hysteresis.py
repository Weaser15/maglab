from pathlib import Path

import numpy as np
import pandas as pd

from ..formats import mumax3
from .utils import compute_dot_vectors


def load_hysteresis(
    filepath: Path | str, region: int | None = None, repeat: bool = True
) -> pd.DataFrame:
    table = mumax3.read_table(filepath)
    mag_name = f"m.region{region}" if region else "m"
    field, mag = compute_dot_vectors(table, ["B_ext", mag_name], direction=mag_name)

    if repeat:
        field = np.concatenate((field, -field))
        mag = np.concatenate((mag, -mag))

    return pd.DataFrame({"field": field, "mag": mag})
