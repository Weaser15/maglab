from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import ndimage


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
    # If slice, will only operate on the first dimension
    if isinstance(mask, slice):
        return (len(np.empty(shape[0])[mask]), *shape[1:])
    # Otherwise must be an iterable with either slices or lists / numpy arrays.
    # This will iterate to the smaller of the two axes.
    for size, masked_row in zip(shape, mask):
        if isinstance(masked_row, np.ndarray | list):
            mask_shape.append(len(masked_row))
        elif isinstance(masked_row, slice):
            mask_shape.append(len(np.empty(size)[masked_row]))
        else:
            raise NotImplementedError(f"Currently type {type(masked_row)} is not supported!")
    if len(mask) < len(shape):
        mask_shape.extend(shape[len(mask) - len(shape) :])
    return tuple(mask_shape)


def apply_bool_mask(arr: np.ndarray, bool_mask: np.ndarray, has_pbc: bool = True):
    # Only works for 2D arr and bool_mask! Make sure bool mask is the same shape as arr

    if (arr_ndims := len(arr.shape)) > 3:
        raise ValueError(f"`arr` must have at most 3 dimensions! Has {arr_ndims}.")

    if arr.shape[:2] != bool_mask.shape[:2]:
        raise ValueError("`arr` and `bool_mask` must have the same shape for first 2 dimensions!")

    # Find objects does not work on boolean arrays. Need to convert to integer.
    # Y axis is axis=0 and X axis is axis=1
    arr = arr.copy()
    arr[~bool_mask] = 0.0  # Set everywhere mask isn't True to 0
    ny, nx, *_ = bool_mask.shape
    bbox: tuple[slice, slice] = ndimage.find_objects(bool_mask.astype(np.int8))[0]  # type:ignore

    spans_x = bbox[1].start == 0 and bbox[1].stop == nx
    spans_y = bbox[0].start == 0 and bbox[0].stop == ny
    if not (spans_x or spans_y) or not has_pbc:
        return arr[bbox].copy()

    if spans_x:
        # Check if there is a gap.
        break_points_x = np.where(np.diff(np.any(bool_mask, axis=0)))[0] + 1
        if len(break_points_x) == 2:
            indices_x = (*np.arange(break_points_x[1], nx), *np.arange(break_points_x[0]))
        else:
            raise NotImplementedError("There must be only two break points (single gap)")
    else:
        indices_x = bbox[1]

    if spans_y:
        # Check if there is a gap.
        break_points_y = np.where(np.diff(np.any(bool_mask, axis=1)))[0] + 1
        if len(break_points_y) == 2:
            indices_y = (*np.arange(break_points_y[1], ny), *np.arange(break_points_y[0]))
        else:
            raise NotImplementedError("There must be only two break points (single gap)")
    else:
        indices_y = bbox[0]

    return arr[indices_y, indices_x].copy()
