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
