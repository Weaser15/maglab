from typing import cast

import numpy as np
import pandas as pd
import scipy.fft as fft


def avg_mag_ringdown(mag: np.ndarray, time: np.ndarray):

    ntime = len(time)
    dt = np.ptp(time) / ntime

    frequencies = np.round(fft.rfftfreq(ntime, dt) * 1e-9, 4)
    m_fft = cast(np.ndarray, fft.rfft(mag))
    psd = np.abs(m_fft) ** 2
    return pd.DataFrame({"frequency": frequencies, "absorption": psd})


# def ringdown(
#     arr: np.ndarray, time: np.ndarray
# ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
#     ntime = len(time)
#     dt = np.ptp(time) / ntime
#     frequencies = np.round(fft.rfftfreq(ntime, dt) * 1e-9, 4)[1:]

#     arr = arr.squeeze(axis=-1)
#     arr = arr - arr.mean(axis=0, keepdims=True)
#     arr_fft = cast(np.ndarray, fft.rfft(arr, axis=0))[1:]

#     psd = np.abs(arr_fft) ** 2
#     phase = np.angle(arr_fft)
#     del arr

#     psd_spectrum = np.mean(psd, axis=(1, 2, 3))
#     psd[psd == 0.0] = np.nan

#     return (psd, phase), (frequencies, psd_spectrum)


def ringdown(
    arr: np.ndarray, time: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    ntime = len(time)
    dt = np.diff(time).mean()

    arr = arr.squeeze(axis=-1)
    arr -= arr.mean(axis=0, keepdims=True)
    spec: np.ndarray = fft.rfft(arr, axis=0)  # type:ignore
    freqs = fft.rfftfreq(ntime, dt).round(6)
    import matplotlib.pyplot as plt

    plt.plot(time, arr.mean(axis=(1, 2, 3)))
    plt.show()

    bls = (np.abs(spec) ** 2).mean(axis=(1, 2, 3))
    fmr = np.abs(fft.rfft(arr.mean(axis=(1, 2, 3)))) ** 2
    return spec, freqs, bls, fmr


def dispersion(arr: np.ndarray, time: np.ndarray, dx: float, dy: float):

    ntime = len(time)
    dt = np.ptp(time) / ntime

    # Reciprocal distances
    Fs = 1 / dt
    Fskx = 2 * np.pi / dx
    Fsky = 2 * np.pi / dy

    # Time, number of z, number of y, number of x, number of components (1 or 3)
    _, _, ny, nx, *_ = arr.shape

    # Reciprocal coordinates
    f = np.arange(-ntime // 2, ntime // 2) * (Fs / ntime)
    kx = np.arange(-nx // 2, nx // 2) * (Fskx / nx)
    ky = np.arange(-ny // 2, ny // 2) * (Fsky / ny)

    # Compute FFT, then shift to centre.
    arr_shifted = np.fft.ifftshift(arr, axes=(2, 3))
    absorption = np.fft.fftn(arr_shifted)
    absorption = np.fft.fftshift(absorption, axes=(0, 2, 3))

    return absorption, (f, kx, ky)
