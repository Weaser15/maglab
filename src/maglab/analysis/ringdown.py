import numpy as np
import pandas as pd
import scipy.fft as fft


def avg_mag_ringdown(mag: np.ndarray, time: np.ndarray) -> pd.DataFrame:

    frequencies = calc_frequencies(len(time), np.diff(time).mean())
    psd = np.abs(fft.rfft(mag)) ** 2
    return pd.DataFrame({"frequency": frequencies, "absorption": psd})


def calc_spectrum(arr: np.ndarray) -> np.ndarray:
    spec = fft.rfft(arr, axis=0, workers=-1)
    spec[0] = 0.0
    return spec


def calc_incoherent(spec: np.ndarray) -> np.ndarray:
    return (abs(spec) ** 2).mean(axis=tuple(range(1, spec.ndim)))


def calc_coherent(spec: np.ndarray) -> np.ndarray:
    return abs(spec.mean(axis=tuple(range(1, spec.ndim)))) ** 2


def calc_dispersion(spec: np.ndarray, axes: tuple = (2, 3)):
    return fft.fftshift(fft.fftn(fft.ifftshift(spec, axes=axes), axes=axes, workers=-1), axes=axes)


def calc_frequencies(nt: int, dt: float) -> np.ndarray:
    return fft.rfftfreq(nt, dt)


def calc_wavevectors(ni: int, di: float) -> np.ndarray:
    return 2 * np.pi * fft.fftshift(fft.fftfreq(ni, di))


def calc_masked_wavevector_array(arr: np.ndarray, axes: tuple = (1, 2), filt=1):
    arr = calc_spectrum(arr)
    arr = arr[:, 0, :, :, 0]
    arr = fft.ifftshift(arr, axes=axes)
    arr = fft.fftn(arr, axes=axes, workers=-1)
    arr = fft.fftshift(arr, axes=axes)
    arr *= filt
    arr = fft.ifftshift(arr, axes=axes)
    arr = fft.ifftn(arr, axes=axes, workers=-1)
    arr = fft.fftshift(arr, axes=axes)
    return arr
