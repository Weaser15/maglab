from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..formats import mumax3
from .utils import compute_dot_vectors


def get_mx3_files(
    dirpath: Path | str, comp: str = "", indexes: list[int] | np.ndarray | None = None
):
    dirpath = Path(dirpath)
    pattern = f"m*{comp}*.ovf" if comp else "m*.ovf"
    files = sorted(dirpath.glob(pattern), key=lambda x: int(x.stem[-6:]))
    if indexes is not None:
        files = [f for f in files if int(f.stem[-6:]) in indexes]
    return files


def load_ovf_array(
    filepath: Path | str,
    direction: tuple[float, float, float] | None = None,
    zslice: int | slice | list | None = None,
):
    # Load array and apply z slice
    arr = mumax3.read_ovf_array(filepath)[zslice, ...]
    # Can only apply direction if array vector is size 3, not 1.
    if (direction is not None) and (arr.shape[-1] > 1):
        # Normalise direction and if direction is along a single component select just that.
        norm_direction = np.array(direction) / np.linalg.norm(direction)
        single_direction = np.isclose(norm_direction, 1.0)
        if not np.any(single_direction):
            arr = np.dot(arr, norm_direction)[..., np.newaxis]
        else:
            arr = arr[..., single_direction]
    return arr.copy()


def load_multiple_ovf_array(
    dirpath: Path | str,
    direction: tuple[float, float, float] | None = None,
    zslice: int | slice | list | None = None,
    indexes: list[int] | np.ndarray | None = None,
    comp: str = "",
    max_workers: int | None = None,
):
    # Select files
    files = get_mx3_files(dirpath, comp, indexes=indexes)

    # Get metadata from a header.
    header = mumax3.read_ovf_header(files[0])
    nx, ny, nz = int(header["xnodes"]), int(header["ynodes"]), int(header["znodes"])
    valuedim = int(header["valuedim"])

    # Change the number of cells along z depending on selection.
    if zslice is not None:
        nz = len(np.arange(nz)[zslice])
    # If direction
    valuedim = 1 if direction is not None else valuedim
    arr = np.empty((len(files), nz, ny, nx, valuedim), dtype=np.float32)

    # Parallelise the loading process by splitting into multiple workers
    def _load(args):
        i, file = args
        arr[i] = load_ovf_array(file, direction=direction, zslice=zslice)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load, (i, f)): i for i, f in enumerate(files)}
        # Add loading bar with tqdm
        for future in tqdm(
            as_completed(futures), total=len(files), desc="Loading OVF files", unit=" files"
        ):
            future.result()
    return arr


def get_state_idx(table: pd.DataFrame | Path | str, bool_name: str = "save_m"):
    if not isinstance(table, pd.DataFrame):
        table = mumax3.read_table(table)

    # Select first column that matches bool_name
    mask = table.columns.str.contains(bool_name)
    if np.any(mask):
        column = table.loc[:, table.columns.str.contains(bool_name)].iloc[:, 0]
        bool_vals = column.to_numpy().astype(bool)
        return pd.Series(np.cumsum(bool_vals), dtype="Int32").where(bool_vals)
    else:
        return pd.Series(np.arange(len(mask)), dtype="Int32")


def get_indexes_by_param(
    table: pd.DataFrame | Path | str,
    values: float | list[float] | np.ndarray,
    *,
    tableparam: str | None = None,
    vectorparam: str | None = None,
    bool_name: str = "save_m",
):
    if not isinstance(table, pd.DataFrame):
        table = mumax3.read_table(table)

    state_idx = get_state_idx(table, bool_name).to_numpy()
    values = np.array([values]) if isinstance(values, float) else np.array(values)
    if tableparam:
        column = table.loc[:, table.columns.str.contains(tableparam)].to_numpy()
    elif vectorparam:
        column = compute_dot_vectors(table, [vectorparam], vectorparam)[0]
    else:
        raise ValueError("tableparam and vectorparam not specified!")
    # Filter out the entries that do not match a state
    column = column[~np.isnan(state_idx)]
    has_matched = np.isclose(values[:, np.newaxis], column, atol=1e-5)
    non_matches = values[~np.any(has_matched, axis=1)]
    if np.any(non_matches):
        print(f"Could not match: {non_matches}")

    indexes = np.where(np.sum(has_matched, axis=0))[0]
    return indexes, values[np.any(has_matched, axis=1)]
