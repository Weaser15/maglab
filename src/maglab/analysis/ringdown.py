"""Spectral analysis of magnetisation time series.

Conventions
-----------
- Magnetisation arrays: (nt, nz, ny, nx[, ncomp]), as loaded from mumax3.
  Select a component (m[.., 2]) before coherent / incoherent reductions.
- Temporal axis is axis 0. Before rfft, this is time. After rfft it is frequency.
- Spatial axes default to (2, 3) = (y, x).
- Units: Hz, rad/m. k-space arrays are fftshifted (k = 0 at centre).

"""

from typing import Any

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


# --- coordinates -------------------------------------------------------------


def calc_frequencies(nt: int, dt: float) -> np.ndarray:
    """Calculate the frequencies of a spectrum from the number of times and the time difference."""
    return fft.rfftfreq(nt, dt)


def calc_wavevectors(ni: int, di: float) -> np.ndarray:
    """Calculate the wavevectors (rad) from the number of cells and the distance difference."""
    return 2 * np.pi * fft.fftshift(fft.fftfreq(ni, di))


# --- transforms --------------------------------------------------------------


def table_ringdown(mag: np.ndarray | pd.Series, time: np.ndarray | pd.Series) -> pd.DataFrame:
    """Perform FFT on a table containing ringdown simulations. Outputs a DataFrame."""
    frequencies = calc_frequencies(len(time), np.diff(time).mean())
    psd = np.abs(fft.rfft(mag)) ** 2
    return pd.DataFrame({"frequency": frequencies, "absorption": psd})


def calc_spectrum(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """Calculate the array with temporal axis FFT'd. By default axis = 0."""
    spec = fft.rfft(arr, axis=axis, workers=-1)
    spec[0] = 0.0
    return spec


def calc_incoherent(spec: np.ndarray, f_axis: int = 0) -> np.ndarray:
    """Calculate the incoherent spectrum. This means squaring the amplitude at each point,
    and then averaging over all space. Frequency axis by default on axis 0."""
    # Get all axes, then remove the f_axis.
    return (abs(spec) ** 2).mean(axis=_other_axes(spec.ndim, f_axis))


def calc_coherent(spec: np.ndarray, f_axis: int = 0) -> np.ndarray:
    """Calculate the coherent spectrum. This means averaging over all space and then
    squaring the amplitude at each frequency. Frequency axis by default on axis 0."""
    # Get all axes, then remove the f_axis.
    return abs(spec.mean(axis=_other_axes(spec.ndim, f_axis))) ** 2


def calc_dispersion(spec: np.ndarray, axes: tuple[int, ...] = (2, 3)):
    """Calculate the dipsersion along spatial axes. Input the spectrum array.
    Performs fftshift and ifftshift to centre the dispersion."""
    return fft.fftshift(
        fft.fftn(fft.ifftshift(spec, axes=axes), axes=axes, workers=-1, overwrite_x=True),
        axes=axes,
    )


def calc_inv_dispersion(disp: np.ndarray, axes: tuple[int, ...] = (2, 3)):
    """Calculate the spectrum along spatial axes. Input the dispersion array.
    Performs fftshift and ifftshift to centre the spectrum."""
    return fft.fftshift(
        fft.ifftn(fft.ifftshift(disp, axes=axes), axes=axes, workers=-1, overwrite_x=True),
        axes=axes,
    )


def apply_wavevector_filter(arr: np.ndarray, filt: Any, axes: tuple[int, ...] = (2, 3)):
    """Perform a FFT on the spatial axes to get reciprocal vectors, apply a filter
    by multiplying the array and then return to spatial by applying inverse FFT."""
    arr = calc_dispersion(arr, axes=axes)
    arr *= filt
    arr = calc_inv_dispersion(arr, axes=axes)
    return arr


def calc_masked_wavevector_spectrum(
    arr: np.ndarray, filt: Any = 1, f_axis: int = 0, s_axes: tuple[int, int] = (2, 3)
):
    """Calculate the frequency spectrum and apply a wavevector filter."""
    arr = calc_spectrum(arr, axis=f_axis)
    return apply_wavevector_filter(arr, filt=filt, axes=s_axes)
