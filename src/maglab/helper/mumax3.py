from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..loaders.hysteresis import load_hysteresis
from ..loaders.mumax3 import get_state_idx, load_multiple_ovf_array
from ..viz.state import plot_lightness, plot_streamlines, to_image


def plot_hysteresis(
    dirpath: Path | str,
    savedir: Path | str,
    name: str = "hyst",
    inch_per_px: float = 1e-3,
    dpi: int = 200,
    streamlines: bool = False,
):

    dirpath = Path(dirpath)
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    df = load_hysteresis(dirpath / "table.txt", repeat=False)
    ax.plot(df["field"] * 1e4, df["mag"])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Field (Oe)")
    ax.set_ylabel(r"$M/M_\text{S}$")
    fig.savefig(savedir / f"{name}.png", dpi=dpi)

    state_idx = get_state_idx(dirpath / "table.txt")
    fields = df.loc[np.where(state_idx)[0], "field"] * 1e4
    arr = load_multiple_ovf_array(dirpath, zslice=0)
    for field, state in zip(fields, arr):
        state = state[0, ...]
        ax = plot_lightness(state)
        if streamlines:
            plot_streamlines(state, ax=ax, d_sep=10, arrows=True)
        to_image(ax, filepath=savedir / f"{name}_{field:.0f}Oe.png", sf=inch_per_px, dpi=dpi)
        plt.close()
