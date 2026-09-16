from pathlib import Path

import pandas as pd


def read_moke_scan(filepath: Path | str) -> pd.DataFrame:
    """Reads a low-field MOKE table (currently up to v3.1) and outputs a DataFrame."""
    filepath = Path(filepath)
    df = pd.read_table(
        filepath,
        comment="#",
        header=None,
        sep=r"\s+",
        names=["field", "mag"],
        dtype={"field": "float32", "mag": "float32"},
        engine="c",
    )
    return df
