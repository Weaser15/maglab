"""
Helpers for common ways of plotting arrays representing states in simulations.
"""

from pathlib import Path
from typing import Literal

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from ..viz.colour import get_lightness_colours
from ..viz.state import plot_arrows, plot_streamlines, resample, to_image


def plot_single_state(
    arr: np.ndarray,
    kind: Literal[
        "real",
        "imag",
        "phase",
        "magnitude",
        "power",
        "log_magnitude",
        "log_power",
        "lightness",
    ] = "real",
    comp: int | tuple[float, ...] | None = None,
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    filepath: Path | str | None = None,
    rescale: float | None = None,
    phase_offset: float | None = None,
    remove_bg: bool = True,
    ax: Axes | None = None,
    cell_size_in: float = 0.01,
    px_per_cell: float = 1.0,
    arrow_params: dict | bool = False,
    streamline_params: dict | bool = False,
):
    """Assumes that arr is an array of shape (ny, nx, valuedim)."""

    if (ndim := len(arr.shape)) > 3:
        raise ValueError(f"Array can only be 2 or 3 dimensions. Received {ndim}!")
    if ndim < 2:
        raise ValueError(f"Array can only be 2 or 3 dimensions. Received {ndim}!")

    scalar_kwargs = {
        "arr": arr,
        "kind": kind,
        "comp": comp,
        "cmap": cmap,
        "clim": clim,
        "rescale": rescale,
        "remove_bg": remove_bg,
        "phase_offset": phase_offset,
        "ax": ax,
        "cell_size_in": cell_size_in,
        "px_per_cell": px_per_cell,
    }

    vector_kwargs = {
        "arr": arr,
        "kind": kind,
        "cmap": cmap,
        "clim": clim,
        "rescale": rescale,
        "remove_bg": remove_bg,
        "ax": ax,
        "cell_size_in": cell_size_in,
        "px_per_cell": px_per_cell,
    }

    if ndim == 2:
        ax, _ = scalar_state(**scalar_kwargs)
    elif ndim == 3:
        if (arr.shape[2] == 1) or (comp is not None):  # type: ignore
            ax, _ = scalar_state(**scalar_kwargs)
        else:
            ax, _ = vector_state(**vector_kwargs)
    else:
        raise ValueError

    if ndim == 3 and arr.shape[2] > 1:  # type: ignore
        if arrow_params:
            arrow_params = {} if arrow_params is True else arrow_params
            print(arrow_params)
            plot_arrows(arr, ax=ax, **arrow_params)
        elif streamline_params:
            streamline_params = {} if streamline_params is True else streamline_params
            plot_streamlines(arr, ax=ax, **streamline_params)

    return ax, to_image(ax, filepath, cell_size_in=cell_size_in, px_per_cell=px_per_cell)


def scalar_state(
    arr: np.ndarray,
    kind: Literal[
        "real", "imag", "phase", "magnitude", "power", "log_magnitude", "log_power"
    ] = "real",
    comp: int | tuple[float, ...] | None = None,
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    filepath: Path | str | None = None,
    rescale: float | None = None,
    phase_offset: float | None = None,
    remove_bg: bool = True,
    ax: Axes | None = None,
    cell_size_in: float = 0.01,
    px_per_cell: float = 1.0,
):
    """Assumes that arr is an array of shape (ny, nx, 1) or (ny, nx)."""
    # Filter out the bad cases. Afterwards you only have a 2D array of size (ny, nx)
    if (ndim := len(arr.shape)) < 2:
        raise ValueError(f"Scalar `arr` shape expects 2 (or 3) dimensions, but received {ndim}")
    if ndim > 3:
        raise ValueError(f"Scalar `arr` shape expects 2 (or 3) dimensions, but received {ndim}")
    if ndim == 3:
        nvaluedim = arr.shape[2]
        if nvaluedim == 1:
            arr = arr[..., 0]
        else:
            if isinstance(comp, int):
                arr = arr[..., comp]
            elif comp is None:
                raise ValueError(
                    f"Cannot process an array with {nvaluedim} vector components as scalar! Specify `comp`."
                )
            else:
                arr = np.dot(arr, np.array(comp) / np.linalg.norm(comp))

    if rescale is not None:
        arr = resample(arr, (), factors=rescale)

    carr = arr.copy()
    if kind.lower() in ("real", "imag", "phase"):
        if phase_offset is None:
            phase_offset = float(np.angle(carr.flat[np.argmax(np.abs(carr))]))
        assert isinstance(phase_offset, float)
        phase_mult: float = np.exp(-1j * phase_offset)
    # Convert array from data to colour
    match kind.lower():
        case "real":
            carr *= phase_mult  # type: ignore
            carr = np.real(carr)
            cmap = "RdBu_r" if cmap is None else cmap
        case "imag":
            carr *= phase_mult  # type: ignore
            carr = np.imag(carr)
            cmap = "RdBu_r" if cmap is None else cmap
        case "phase":
            carr *= phase_mult  # type: ignore
            carr = np.angle(carr)
            cmap = "hsv" if cmap is None else cmap
        case "magnitude":
            carr = np.abs(carr)
            cmap = "plasma" if cmap is None else cmap
        case "power":
            carr = np.abs(carr) ** 2
            cmap = "plasma" if cmap is None else cmap
        case "log_magnitude":
            carr = np.log10(np.abs(carr))
            cmap = "plasma" if cmap is None else cmap
        case "log_power":
            carr = np.log10(np.abs(carr) ** 2)
            cmap = "plasma" if cmap is None else cmap
        case _:
            raise ValueError(f"Do not recognise `kind` = {kind}")

    if remove_bg:
        carr[carr == 0.0] = np.nan

    if ax is None:
        _, ax = plt.subplots()

    clim = (np.nanmin(carr), np.nanmax(carr)) if clim is None else clim
    ax.imshow(carr, origin="lower", cmap=cmap, clim=clim)
    ax.set_xlim(*ax.get_xlim())
    ax.set_ylim(*ax.get_ylim())

    return ax, to_image(
        ax, filepath, clean=True, px_per_cell=px_per_cell, cell_size_in=cell_size_in
    )


def vector_state(
    arr: np.ndarray,
    kind: Literal["lightness"] = "lightness",
    cmap: str | None = None,
    clim: tuple[float, float] | None = None,
    filepath: Path | str | None = None,
    rescale: float | None = None,
    remove_bg: bool = True,
    ax: Axes | None = None,
    cell_size_in: float = 0.01,
    px_per_cell: float = 1.0,
):
    if (ndim := len(arr.shape)) < 3:
        raise ValueError(f"Vector `arr` shape expects 3 dimensions, but received {ndim}")
    if ndim > 3:
        raise ValueError(f"Vector `arr` shape expects 3 dimensions, but received {ndim}")
    if (nvaluedim := arr.shape[2]) == 1:
        raise ValueError(
            f"Vector `arr` shape expects last dimension to be larger than 1, but is {nvaluedim}"
        )

    if rescale is not None:
        arr = resample(arr, (), factors=rescale)

    carr = arr.copy()
    match kind.lower():
        case "lightness":
            carr = get_lightness_colours(arr, lightness_clim=clim, filter=remove_bg)
        case _:
            raise NotImplementedError("Only lightness implemented at the moment!")

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(carr, origin="lower", cmap=cmap)
    ax.set_xlim(*ax.get_xlim())
    ax.set_ylim(*ax.get_ylim())

    return ax, to_image(
        ax, filepath, clean=True, cell_size_in=cell_size_in, px_per_cell=px_per_cell
    )
