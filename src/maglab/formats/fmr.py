from pathlib import Path

import pandas as pd


def read_diode_fmr(filepath: Path | str) -> pd.DataFrame:
    """Read a single diode FMR scan and return a pandas DataFrame."""
    df = pd.read_table(
        filepath,
        header=None,
        names=["field", "dP/dH"],
        dtype={"field": "float64", "dP/dH": "float64"},
        engine="c",
    )
    return df


def read_vna_fmr(filepath: Path | str) -> pd.DataFrame:
    """Read a single vna FMR scan and return a pandas DataFrame."""
    df = pd.read_table(
        filepath,
        header=None,
        names=["frequency", "S21"],
        dtype={"frequency": "float64", "S21": "float64"},
        engine="c",
    )
    df["frequency"] /= 1e9  # Convert to GHz
    return df
