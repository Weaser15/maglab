from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..analysis.spectral import (
    coherent_power,
    detect_modes,
    filter_k,
    incoherent_power,
    time_to_freq,
)
from ..formats.mumax3 import read_table
from ..helper.state import plot_single_state
from ..loaders.hysteresis import load_hysteresis
from ..loaders.mumax3 import get_indexes_by_param, get_state_idx, load_multiple_ovf_array
from ..loaders.mx3_ringdown import load_frequencies, load_spec_array
from ..snakemake.utils import apply_mask
from ..viz.state import plot_lightness, plot_streamlines, to_image

# def plot_hysteresis(
#     dirpath: Path | str,
#     savedir: Path | str,
#     name: str = "hyst",
#     inch_per_px: float = 1e-3,
#     dpi: int = 200,
#     streamlines: bool = False,
# ):

#     dirpath = Path(dirpath)
#     savedir = Path(savedir)
#     savedir.mkdir(parents=True, exist_ok=True)

#     df = load_hysteresis(dirpath / "table.txt", repeat=False)
#     df.to_csv(savedir / f"{name}.csv")

#     state_idx = get_state_idx(dirpath / "table.txt")
#     fields = df.loc[np.where(state_idx)[0], "field"] * 1e4
#     arr = load_multiple_ovf_array(dirpath, zslice=0)
#     for field, state in zip(fields, arr):
#         state = state[0, ...]
#         ax = plot_lightness(state)
#         if streamlines:
#             plot_streamlines(state, ax=ax, d_sep=10, arrows=True)
#         to_image(ax, filepath=savedir / f"{name}_{field:.0f}Oe.png", sf=inch_per_px, dpi=dpi)
#         plt.close()

#     fig, ax = plt.subplots()
#     ax.plot(df["field"] * 1e4, df["mag"])
#     ax.set_xlim(-1000, 1000)
#     ax.set_ylim(-1, 1)
#     ax.set_xlabel("Field (Oe)")
#     ax.set_ylabel(r"$M/M_\text{S}$")
#     fig.savefig(savedir / f"{name}.png", dpi=dpi)


def plot_excitation(
    dirpath: Path | str,
    savedir: Path | str,
    name: str = "ringdown",
    masks: tuple = (None,),
    kind: Literal["real", "imag", "power", "log_power", "magnitude", "log_magnitude"] = "real",
    k_filter: np.ndarray | None = None,
    coherent: bool = False,
    direction: tuple[float, float, float] = (0, 0, 1),
    min_height: float = 1e-2,
    cell_size_in: float = 0.01,
):
    # Remove the 6-digit number and the beginning index (so glob works later)
    # This should find the different layers
    dirpath = Path(dirpath)
    comps = list({f.stem[1:-6] for f in dirpath.glob("*.ovf")})
    comps.sort()

    # Create the save directory
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    table = read_table(dirpath / "table.txt")
    time = table["t (s)"].to_numpy()
    freqs = np.round(load_frequencies(dirpath / "table.txt") / 1e9, 5)
    dfs = []
    for i, comp in enumerate(comps):
        if k_filter is not None:
            full_arr = load_multiple_ovf_array(dirpath, direction=direction, comp=comp)
            avg_axes = tuple(np.arange(len(full_arr.shape))[1:])
        for j, mask in enumerate(masks):
            base_name = f"{name}_layer{i}_mask{j}"
            # Load the magnetisation from the OVF files. Then plot the time series of magnetisation.
            # Afterwards convert time-to frequency. Complicated when applying k-filters.
            fig, ax = plt.subplots()
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Magnetization (norm.)")
            time_series_name = f"{base_name}_timeseries.png"
            if k_filter is not None:
                ax.plot(time, np.mean(apply_mask(full_arr, mask), axis=avg_axes))
                arr = apply_mask(filter_k(time_to_freq(full_arr), k_filter), mask, crop_first=True)
            else:
                arr = load_multiple_ovf_array(dirpath, direction=direction, comp=comp, mask=mask)
                avg_axes = tuple(np.arange(len(arr.shape))[1:])
                # arr = apply_mask(full_arr, mask, crop_first=True)
                ax.plot(time, np.mean(arr, axis=avg_axes))
                arr = time_to_freq(arr)
            fig.savefig(savedir / time_series_name)
            plt.close(fig)

            power = coherent_power(arr) if coherent else incoherent_power(arr)
            idxs, _ = find_peaks(power, height=min_height * np.max(power))

            dfs.append(pd.DataFrame({"frequency": freqs, "power": power, "layer": i, "mask": j}))

            fig, ax = plt.subplots()
            ax.plot(freqs, power)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Power")
            ax.scatter(freqs[idxs], power[idxs], color="k")
            fig.savefig(savedir / f"{name}_layer{i}_mask{j}.png")
            plt.close(fig)

            # Plot each peak mode
            for idx in idxs:
                f = freqs[idx]
                state_name = f"{base_name}_{f:.2f}GHz.png"
                state = arr[idx, ...].mean(axis=0)
                plot_single_state(
                    state,
                    kind,
                    px_per_cell=1,
                    cell_size_in=cell_size_in,
                    filepath=savedir / state_name,
                )
                plt.close()

    csv_name = f"{name}.csv"
    pd.concat(dfs, ignore_index=True).to_csv(savedir / csv_name)
