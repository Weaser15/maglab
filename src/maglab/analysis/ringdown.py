from typing import cast

import numpy as np
import pandas as pd
import scipy.fft as fft


def avg_mag_ringdown(time: np.ndarray, mag: np.ndarray):

    ntime = len(time)
    dt = np.ptp(time) / ntime
    frequencies = np.round(fft.fftfreq(ntime, dt) * 1e-9, 4)
    mask = frequencies > 0

    m_fft = cast(np.ndarray, fft.rfft(mag))
    psd = m_fft[mask] ** 2
    freq = frequencies[mask]
    return pd.DataFrame({"frequency": freq, "absorption": psd})


def dispersion(arr: np.ndarray, time: np.ndarray, dx: float, dy: float):

    ntime = len(time)
    dt = np.ptp(time) / ntime

    # Reciprocal distances
    Fs = 1 / dt
    Fskx = 2 * np.pi / dx
    Fsky = 2 * np.pi / dy

    # Time, number of z, number of y, number of x, number of components (1 or 3)
    _, _, ny, nx, _ = arr.shape

    # Reciprocal coordinates
    f = np.arange(-ntime // 2, ntime // 2) * (Fs / ntime)
    kx = np.arange(-nx // 2, nx // 2) * (Fskx / nx)
    ky = np.arange(-ny // 2, ny // 2) * (Fsky / ny)

    arr_shifted = np.fft.ifftshift(arr, axes=(2, 3))
    absorption = np.fft.fftn(arr_shifted)
    absorption = np.fft.fftshift(absorption, axes=(0, 2, 3))

    return absorption, (f, kx, ky)
