from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..analysis.spectral import (
    coherent_power,
    filter_k,
    incoherent_power,
    time_to_freq,
)
from ..formats.mumax3 import read_table
from ..loaders.hysteresis import load_hysteresis
from ..loaders.mumax3 import get_indexes_by_param, get_state_idx, load_multiple_ovf_array
from ..loaders.mx3_ringdown import load_frequencies, load_spec_array
from ..processing.state import apply_mask
from .state import plot_single_state


def plot_hysteresis(
    dirpath: Path | str,
    savedir: Path | str,
    name: str = "hyst",
    masks: tuple | dict = (None,),
    cell_size_in: float = 0.01,
    px_per_cell: int = 4,
    arrow_params: bool | dict = False,
    streamline_params: bool | dict = False,
):

    dirpath = Path(dirpath)
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    tablepath = dirpath / "table.txt"
    hyst_df = load_hysteresis(tablepath, repeat=False)
    state_idx = np.where(get_state_idx(tablepath))[0]

    fields = hyst_df.loc[state_idx, "field"] * 1e4
    full_arr = load_multiple_ovf_array(dirpath, zslice=0)

    # Get first field where magnetisation state is saved.
    table = read_table(tablepath)
    field_vector: np.ndarray = np.asarray(
        table.loc[state_idx[0], ("B_extx (T)", "B_exty (T)", "B_extz (T)")]
    )
    field_vector = field_vector / np.linalg.norm(field_vector)

    dfs = []
    for j, mask in enumerate(masks):
        if isinstance(masks, dict):
            # mask is the key when it's a dict, not the value (which would be the mask).
            maskname = mask
            mask = masks[mask]
        else:
            maskname = f"mask{j}"

        basename = f"{name}_{maskname}"

        arr = apply_mask(full_arr, mask)
        mag = -np.mean(np.dot(arr, field_vector), axis=(1, 2, 3))

        fig, ax = plt.subplots()
        ax.plot(fields, mag)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Field (Oe)")
        ax.set_ylabel(r"$M/M_\text{S}$")
        fig.savefig(savedir / f"{basename}.png", dpi=200)

        dfs.append(pd.DataFrame({"mask": maskname, "field": fields, "mag": mag}))

        for field, state in zip(fields, arr):
            state = state[-1, ...]
            state_save_name = f"{basename}_H{field:.0f}Oe.png"
            plot_single_state(
                state,
                filepath=savedir / state_save_name,
                kind="lightness",
                px_per_cell=px_per_cell,
                cell_size_in=cell_size_in,
                arrow_params=arrow_params,
                streamline_params=streamline_params,
            )
            plt.close()

    pd.concat(dfs).to_csv(savedir / f"{name}.csv", index=False)


def plot_excitation(
    dirpath: Path | str,
    savedir: Path | str,
    name: str = "ringdown",
    masks: tuple | dict = (None,),
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
            if isinstance(masks, dict):
                # mask is the key when it's a dict, not the value (which would be the mask).
                maskname = mask
                mask = masks[mask]
            else:
                maskname = f"mask{j}"

            base_name = f"{name}_layer{i}_{maskname}"
            # Load the magnetisation from the OVF files. Then plot the time series of magnetisation.
            # Afterwards convert time-to frequency. Complicated when applying k-filters.
            fig, ax = plt.subplots()
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Magnetization (norm.)")
            time_series_name = f"{base_name}_timeseries.png"
            if k_filter is not None:
                ax.plot(time, np.mean(apply_mask(full_arr, mask), axis=avg_axes))  # pyright: ignore[reportPossiblyUnboundVariable]
                arr = apply_mask(filter_k(time_to_freq(full_arr), k_filter), mask, crop_first=True)  # pyright: ignore[reportPossiblyUnboundVariable]
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

            dfs.append(
                pd.DataFrame({"mask": maskname, "layer": i, "frequency": freqs, "power": power})
            )

            fig, ax = plt.subplots()
            ax.plot(freqs, power)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Power")
            ax.scatter(freqs[idxs], power[idxs], color="k")
            fig.savefig(savedir / f"{name}_layer{i}_{maskname}.png")
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
    pd.concat(dfs, ignore_index=True).to_csv(savedir / csv_name, index=False)
