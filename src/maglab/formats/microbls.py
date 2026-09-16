from pathlib import Path

import numpy as np
import pandas as pd


def read_microbls(filepath: Path | str) -> pd.DataFrame:
    """Read a microBLS file obtained by BrilliantStar program."""
    # uBLS files do not contain any header, and columns are structured as
    # (frequency, position 1, position 2, ...).
    # Turning it into long format, i.e. make position a single column
    df = pd.read_table(filepath, sep="\t", header=None)
    df.columns = ("frequency", *range(df.shape[1] - 1))
    df = df.melt(id_vars="frequency", var_name="position", value_name="intensity")
    return df[["position", "frequency", "intensity"]]


def read_fieldlist(filepath: Path | str) -> np.ndarray:
    """Read a fieldlist containing fields swept separated by newlines."""
    return np.loadtxt(filepath)
