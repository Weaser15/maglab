from typing import cast

import numpy as np
import scipy.fft as fft


def averaged_magnetisation_ringdown(time: np.ndarray, mag: np.ndarray):

    ntime = len(time)
    dt = np.ptp(time) / ntime
    frequencies = np.round(fft.fftfreq(ntime, dt) * 1e-9, 4)

    m_fft = cast(np.ndarray, fft.rfft(mag))
    psd = abs(m_fft) ** 2
