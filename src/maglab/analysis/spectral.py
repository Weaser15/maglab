"""Spectral analysis of magnetisation time series.

Conventions
-----------
- Magnetisation arrays: (nt, nz, ny, nx[, ncomp]), as loaded from mumax3.
  Select a component (m[.., 2]) before coherent / incoherent reductions.
- Temporal axis is axis 0. Before rfft, this is time. After rfft it is frequency.
- Spatial axes default to (2, 3) = (y, x).
- Units: Hz, rad/m. k-space arrays are fftshifted (k = 0 at centre).

"""

import numpy as np
import pandas as pd
import scipy.fft as fft

T_AXIS = 0
XY_AXES = (2, 3)

# --- helpers -----------------------------------------------------------------


def _other_axes(ndim: int, axis: int | list[int]) -> tuple[int, ...]:
    """Return all of the other axes of an array except axis."""
    axis = [axis] if isinstance(axis, int) else axis
    return tuple(i for i in range(ndim) if i not in axis)


def _expand(mask: np.ndarray, ndim: int, axes: tuple[int, ...]) -> np.ndarray:
    """Reshape mask so its dims sit on `axes` of an ndim-array (size 1 elsewhere)"""
    shape = [mask.shape[i] if i in axes else 1 for i in range(ndim)]
    return mask.reshape(shape)


# --- coordinates -------------------------------------------------------------


def frequencies(nt: int, dt: float) -> np.ndarray:
    """Calculate the frequencies of a spectrum from the number of times and the time difference."""
    return fft.rfftfreq(nt, dt)


def wavevectors(ni: int, di: float) -> np.ndarray:
    """Calculate the wavevectors (rad) from the number of cells and the distance difference."""
    return 2 * np.pi * fft.fftshift(fft.fftfreq(ni, di))


# --- transforms --------------------------------------------------------------


def time_to_freq(arr: np.ndarray, axis: int = T_AXIS, remove_dc: bool = True) -> np.ndarray:
    """Convert a time axis to a frequency axis. By default, axis = 0."""
    spec = fft.rfft(arr, axis=axis, workers=-1)
    if remove_dc:
        # Make it so that axis of interest is in index 0, then set first value to 0.
        np.moveaxis(spec, axis, 0)[0] = 0
    return spec


def real_to_k(arr: np.ndarray, axes: tuple[int, ...] = XY_AXES) -> np.ndarray:
    """Transform axes from real space to k (wavevector) space."""
    return fft.fftshift(
        fft.fftn(fft.ifftshift(arr, axes=axes), axes=axes, workers=-1, overwrite_x=True),
        axes=axes,
    )


def k_to_real(arr: np.ndarray, axes: tuple[int, ...] = XY_AXES) -> np.ndarray:
    """Transform axes from k (wavevector) space to real space."""
    return fft.fftshift(
        fft.ifftn(fft.ifftshift(arr, axes=axes), axes=axes, workers=-1, overwrite_x=True),
        axes=axes,
    )


def filter_k(arr: np.ndarray, mask: np.ndarray, axes: tuple[int, ...] = XY_AXES) -> np.ndarray:
    """Multiply by a k-space mask and transform back."""
    return k_to_real(real_to_k(arr, axes) * _expand(mask, arr.ndim, axes), axes)


# --- reductions --------------------------------------------------------------


def incoherent_power(spec: np.ndarray, f_axis: int = T_AXIS) -> np.ndarray:
    """|.|^2 at each point, then average over all non-frequency axes."""
    # Get all axes, then remove the f_axis.
    return (abs(spec) ** 2).mean(axis=_other_axes(spec.ndim, f_axis))


def coherent_power(spec: np.ndarray, f_axis: int = T_AXIS) -> np.ndarray:
    """Average over all non-frequency axes, then |.|^2."""
    # Get all axes, then remove the f_axis.
    return abs(spec.mean(axis=_other_axes(spec.ndim, f_axis))) ** 2


# --- adapters ----------------------------------------------------------------


def table_ringdown(mag: np.ndarray | pd.Series, time: np.ndarray | pd.Series) -> pd.DataFrame:
    """Perform FFT on a table containing ringdown simulations. Outputs a DataFrame."""
    mag, time = np.asarray(mag), np.asarray(time)
    spec = time_to_freq(time)
    freqs = frequencies(len(time), np.diff(time).mean())
    return pd.DataFrame({"frequency": freqs, "power": spec})
