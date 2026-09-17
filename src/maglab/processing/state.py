import numpy as np


def select_rectangular(arr: np.ndarray, xrange: tuple[float, float], yrange: tuple[float, float]):
    xmin, xmax = xrange
    ymin, ymax = yrange
    carr = arr
    if xmin < xmax:
        carr = carr[:, :, :, xmin:xmax, :]
    else:
        carr = np.concatenate((carr[:, :, :, xmin:, :], carr[:, :, :, :xmax, :]), axis=3)
    if ymin < ymax:
        carr = carr[:, :, ymin:ymax, :, :]
    else:
        carr = np.concatenate((carr[:, :, ymin:, :, :], carr[:, :, :ymax, :, :]), axis=2)
    return carr
