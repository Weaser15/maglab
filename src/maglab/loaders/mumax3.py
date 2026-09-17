from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..formats import mumax3


def get_mx3_files(dirpath: Path | str, comp: str = ""):
    dirpath = Path(dirpath)
    pattern = f"m*{comp}*.ovf" if comp else "m*.ovf"
    return sorted(dirpath.glob(pattern))


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
    comp: str = "",
    max_workers: int | None = None,
):
    # Select files
    files = get_mx3_files(dirpath, comp)

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
