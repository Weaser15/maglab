from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .colour import get_lightness_colours
from .streamlines import evenly_spaced_streamlines

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def resample(
    arr: np.ndarray, newshape: Sequence[int], *, factors: float | Sequence[float] | None = None
):
    if factors is None:
        factors = [t / s for t, s in zip(newshape, arr.shape[:-1], strict=True)] + [
            1 for _ in range(len(arr.shape) - len(newshape))
        ]

    return zoom(arr, factors)


def plot_arrows(arr: np.ndarray, narrows: tuple[int, int], ax, sf: float | None = None, **kwargs):

    if sf is not None:
        narrows = (int(arr.shape[0] * sf), int(arr.shape[1] * sf))
        print(narrows)

    arrows_arr = resample(arr, narrows)

    x = np.linspace(0, arr.shape[0] - 1, narrows[0])
    y = np.linspace(0, arr.shape[1] - 1, narrows[1])
    u = arrows_arr[..., 0].T
    v = arrows_arr[..., 1].T
    ax.quiver(x, y, u, v, **kwargs)


def plot_lightness(
    arr: np.ndarray,
    ax: Axes | None = None,
    clim: tuple[float, float] | None = None,
    filter: bool = True,
) -> Axes:

    if ax is None:
        _, ax = plt.subplots()
    carr = get_lightness_colours(arr, lightness_clim=clim, filter=filter)
    ax.imshow(carr, origin="lower")
    ax.set_xlim(*ax.get_xlim())
    ax.set_ylim(*ax.get_ylim())
    return ax


def plot_magnitude(
    arr: np.ndarray,
    ax: Axes | None = None,
    cmap: str = "plasma",
    clim: tuple[float, float] | None = None,
    filter: bool = True,
) -> Axes:

    if ax is None:
        _, ax = plt.subplots()

    if len(arr.shape) == 2:
        arr_ = arr
    elif len(arr.shape) == 3:
        arr_ = np.linalg.norm(arr, axis=-1)
    else:
        raise ValueError("arr must be shape 2 or 3!")

    if clim is None:
        clim = (-1, 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=clim[0], vmax=clim[1]))  # type: ignore
    carr = sm.to_rgba(arr_)
    # cmap_ = plt.get_cmap(cmap)
    # carr = cmap_(arr_)
    ax.imshow(carr, origin="lower")
    return ax


def plot_streamlines(
    arr: np.ndarray,
    d_sep: float = 10.0,
    d_test: float | None = None,
    step: float = 1.0,
    max_steps: int = 1000,
    ax: Axes | None = None,
    arrow: bool = True,
    arrowstyle="-|>",
    arrow_every=1,
    **kwargs,
) -> None:

    if ax is None:
        _, ax = plt.subplots()

    kwargs.setdefault("colors", "black")
    kwargs.setdefault("linewidths", 0.8)

    streamlines = evenly_spaced_streamlines(
        arr, d_sep=d_sep, d_test=d_test, step=step, max_steps=max_steps
    )

    lc = LineCollection(streamlines, **kwargs)
    ax.add_collection(lc)

    if arrow:
        color = kwargs.get("colors", "black")
        lw = kwargs.get("linewidths", 0.8)
        for line in streamlines[::arrow_every]:
            if len(line) < 2:
                continue
            # midpoint of the line
            mid = max(0, len(line) // 2 - 1)
            p1 = line[mid]
            p2 = line[mid + 1]
            ax.annotate(
                "",
                xytext=p1,
                xy=p2,
                arrowprops={
                    "arrowstyle": arrowstyle,
                    "color": color,
                    "lw": lw,
                },
            )


def to_image(
    ax: Axes | None = None,
    filepath: Path | str | None = None,
    clean: bool = True,
    dpi: float | None = None,
    sf: float | None = None,
    figsize: tuple[float | None, float | None] = (None, None),
    fix_ratio: bool = True,
) -> Image.Image:
    buffer = BytesIO()
    ax = plt.gca() if ax is None else ax
    fig: Figure = ax.figure  # type: ignore
    if clean:
        ax.axis("off")
        ax.set_frame_on(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    if not fix_ratio:
        ax.set_aspect("auto")

    im_arr: np.ndarray = ax.get_images()[0].get_array()  # type: ignore
    yx_ratio = im_arr.shape[0] / im_arr.shape[1]  # Axes transposed for image

    print(fig.get_size_inches())

    if sf is not None and not any(figsize):
        figsize = (im_arr.shape[1] * sf, im_arr.shape[0] * sf)

    if figsize[0] is not None and figsize[1] is None:
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[0] * yx_ratio)
    elif figsize[0] is None and figsize[1] is not None:
        fig.set_figheight(figsize[1])
        fig.set_figwidth(figsize[1] / yx_ratio)
    elif figsize[0] is not None and figsize[1] is not None:
        fig.set_size_inches(figsize[0], figsize[1])

    # Make sure below maximum (10, 10) inches.
    max_size = 10.0
    if fig.get_size_inches()[0] > 10.0:
        old_size = fig.get_size_inches()
        fig.set_size_inches(max_size, old_size[1] / old_size[0] * max_size)
    elif fig.get_size_inches()[1] > 10.0:
        old_size = fig.get_size_inches()
        fig.set_size_inches(old_size[0] / old_size[1] * max_size, max_size)

    print("Image size is:", fig.get_size_inches(), "inches")

    fig.savefig(buffer, format="png", pad_inches=0, transparent=True, dpi=dpi)  # type: ignore
    buffer.seek(0)
    image = Image.open(buffer)
    if filepath is not None:
        image.save(filepath, dpi=None if dpi is None else (dpi, dpi))
    return image
