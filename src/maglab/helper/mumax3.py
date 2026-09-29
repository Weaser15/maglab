from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.spectral import coherent_power, detect_modes, incoherent_power
from ..loaders.hysteresis import load_hysteresis
from ..loaders.mumax3 import get_state_idx, load_multiple_ovf_array
from ..loaders.mx3_ringdown import load_frequencies, load_spec_array
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

    df = load_hysteresis(dirpath / "table.txt", repeat=False)
    df.to_csv(savedir / f"{name}.csv")

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

    fig, ax = plt.subplots()
    ax.plot(df["field"] * 1e4, df["mag"])
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Field (Oe)")
    ax.set_ylabel(r"$M/M_\text{S}$")
    fig.savefig(savedir / f"{name}.png", dpi=dpi)


def plot_excitation(
    dirpath: Path | str,
    savedir: Path | str,
    name: str = "ringdown",
    coherent: bool = False,
    direction: tuple[float, float, float] = (0, 0, 1),
    min_prom_db: int = 4,
    inch_per_px: float = 1e-3,
    dpi: int = 200,
):
    dirpath = Path(dirpath)
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    arr = load_spec_array(dirpath, direction=direction)
    psd = coherent_power(arr) if coherent else incoherent_power(arr)
    freqs = load_frequencies(dirpath / "table.txt")
    idx = detect_modes(psd, min_prom_db)

    for i in idx:
        f = freqs[i]
        state = arr[i, ...].mean(axis=0)
        state *= np.exp(-1j * np.angle(state.flat[np.argmax(np.abs(state))]))
        v = np.abs(state.real).max()
        state[state == 0.0] = np.nan

        fig, ax = plt.subplots()
        ax.imshow(state.real, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())
        to_image(ax, filepath=savedir / f"{name} {f:.2f}GHz.png", sf=inch_per_px, dpi=dpi)
        plt.close()

    df = pd.DataFrame({"frequency": freqs, "power": psd})
    df.to_csv(savedir / f"{name}.csv")
    fig, ax = plt.subplots()
    ax.plot(freqs / 1e9, psd)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Power")

    ax.scatter(freqs[idx] / 1e9, psd[idx], color="k")
    fig.savefig(savedir / f"{name}.png", dpi=dpi)
