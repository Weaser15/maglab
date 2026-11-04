from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from ..formats.moke import read_moke_scan
from ..processing.moke import align_hysteresis_data, get_centre_offset, interpolate_hysteresis
from .utils import apply_template


def load_moke_scan(
    filepath: Path | str, unit: Literal["mT", "Oe"] = "Oe", spacing: float | None = 0.5
) -> pd.DataFrame:
    df = read_moke_scan(filepath)

    field = df["field"].to_numpy(copy=True)
    mag = df["mag"].to_numpy(copy=True)
    field *= 10 ** (unit.lower() == "oe")  # raw data in mT; convert to Oe if needed
    field -= get_centre_offset(field, mag)
    field, mag = align_hysteresis_data(field, mag)
    field, mag = interpolate_hysteresis(field, mag, new_x_spacing=spacing)

    return pd.DataFrame({"field": field, "mag": mag})


def load_moke(
    path: Path | str,
    unit: Literal["mT", "Oe"] = "Oe",
    spacing: float | None = 0.5,
    template: Literal["name", "name_angle", "name_pattern_angle"] = "name",
) -> pd.DataFrame:
    path = Path(path)

    if path.is_file():
        return load_moke_scan(path, unit=unit, spacing=spacing)

    files = list(path.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No MOKE files found in {path}!")

    dfs: list[pd.DataFrame] = []

    def _load(file: Path):
        return load_moke_scan(file, unit=unit, spacing=spacing).assign(name=file.stem)

    with ThreadPoolExecutor() as executor:
        for future in tqdm(
            as_completed(executor.submit(_load, f) for f in files),
            total=len(files),
            desc="Loading MOKE files",
            unit=" files",
        ):
            dfs.append(future.result())
    df = pd.concat(dfs, axis=0)
    df.insert(0, "name", df.pop("name"))

    if template.lower() == "name_angle":
        return apply_template(df, "name", r"(?P<name>.+)_(?P<angle>\d+)$", [str, float])
    if template.lower() == "name_pattern_angle":
        return apply_template(
            df, "name", r"(?P<name>.+)_(?P<pattern>\d+-\d+)_(?P<angle>[\d.]+)$", [str, str, float]
        )
    return df
