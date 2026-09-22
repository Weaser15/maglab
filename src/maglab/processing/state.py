from typing import Any

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


def boundary_slice_mask(shape: np.ndarray | tuple[int, ...], slices: list[slice]):
    shape = shape.shape if isinstance(shape, np.ndarray) else shape
    mask = []
    for size, dim_slice in zip(shape, slices):
        start, stop, step = dim_slice.start, dim_slice.stop, dim_slice.step
        if start is None or stop is None:
            mask.append(dim_slice)
            continue
        if start < stop:
            mask.append(dim_slice)
        else:
            indices1, indices2 = (start, size, step), (0, stop, step)
            mask.append(np.concatenate((np.arange(*indices1), np.arange(*indices2))))
    return tuple(mask)


def get_mask_shape(shape: np.ndarray | tuple[int, ...], mask: Any):
    shape = shape.shape if isinstance(shape, np.ndarray) else shape
    mask_shape = []
    if isinstance(mask, slice):
        return (len(np.empty(shape[0])[mask]), *shape[1:])
    for size, masked_row in zip(shape, mask):
        if isinstance(masked_row, np.ndarray | list):
            mask_shape.append(len(masked_row))
        elif isinstance(masked_row, slice):
            mask_shape.append(len(np.empty(size)[masked_row]))
        else:
            raise NotImplementedError(f"Currently type {type(masked_row)} is not supported!")
    mask_shape.extend(shape[len(mask) - len(shape) :])
    return tuple(mask_shape)
