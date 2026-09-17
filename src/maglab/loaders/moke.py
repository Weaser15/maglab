from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from ..formats.moke import read_moke_scan
from .utils import apply_template


def _get_centre_offset(x: np.ndarray, y: np.ndarray) -> float:
    halfarg = len(x) // 2
    x_btm, y_btm = x[:halfarg], y[:halfarg]
    x_top, y_top = x[halfarg:], y[halfarg:]

    # Rotated top branch: negate x and y, sort ascending
    x_top_rot = -x_top
    y_top_rot = -y_top

    # Resample both onto a common uniform grid
    x_min = max(x_btm[0], x_top_rot[0])
    x_max = min(x_btm[-1], x_top_rot[-1])
    n = (len(x_btm) + len(x_top)) * 2
    x_common = np.linspace(x_min, x_max, n)

    k, bc_type = 3, "natural"
    y_btm_r = make_interp_spline(x_btm, y_btm, k=k, bc_type=bc_type)(x_common)
    y_top_r = make_interp_spline(x_top_rot, y_top_rot, k=k, bc_type=bc_type)(x_common)

    # Cross-correlate to find the shift
    corr = np.correlate(y_btm_r - y_btm_r.mean(), y_top_r - y_top_r.mean(), mode="full")
    shift_samples = np.argmax(corr) - (n - 1)
    dx = x_common[1] - x_common[0]

    # Shift between branches = 2 * H_eb, so halve it
    return shift_samples * dx / 2


def _align_hysteresis_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Make data start and end with negative y.
    if y[x.argmin()] > 0.0:
        x = -x

    if x[0] > 0:
        x, y = -x, -y

    halfarg = len(x) // 2
    if y[:halfarg].mean() > y[halfarg:].mean():
        x, y = np.flip(x), np.flip(y)

    return x, y


def _interpolate_hysteresis(
    x: np.ndarray, y: np.ndarray, new_x_spacing: float | None
) -> tuple[np.ndarray, np.ndarray]:

    halfarg = len(x) // 2

    x_btm, x_top = x[:halfarg], x[halfarg:]
    y_btm, y_top = y[:halfarg], y[halfarg:]

    x_btm, y_btm = _interpolate_data(x_btm, y_btm, new_x_spacing)
    x_top, y_top = _interpolate_data(x_top, y_top, new_x_spacing, reverse=True)

    return np.hstack((x_btm, x_top)), np.hstack((y_btm, y_top))


def _interpolate_data(
    x: np.ndarray, y: np.ndarray, new_x_spacing: float | None, reverse: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if new_x_spacing is None:
        return x, y

    if reverse:
        x, y = np.flip(x), np.flip(y)

    x_max = min(
        abs(x[0] // new_x_spacing * new_x_spacing),
        abs(x[-1] // new_x_spacing * new_x_spacing),
    )
    x_interp = np.arange(-x_max, x_max + new_x_spacing, new_x_spacing)
    y_interp = np.interp(x_interp, x, y)

    if reverse:
        x_interp, y_interp = np.flip(x_interp), np.flip(y_interp)

    return x_interp, y_interp


def load_moke_scan(
    filepath: Path | str, unit: Literal["mT", "Oe"] = "Oe", spacing: float | None = 0.5
) -> pd.DataFrame:
    df = read_moke_scan(filepath)

    field = df["field"].to_numpy(copy=True)
    mag = df["mag"].to_numpy(copy=True)
    field *= 10 ** (unit.lower() == "oe")  # raw data in mT; convert to Oe if needed
    field -= _get_centre_offset(field, mag)
    field, mag = _align_hysteresis_data(field, mag)
    field, mag = _interpolate_hysteresis(field, mag, new_x_spacing=spacing)

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
