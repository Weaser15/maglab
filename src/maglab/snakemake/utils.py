from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from ..processing.state import apply_bool_mask, boundary_slice_mask


def auto_bool_masks(arr: np.ndarray, has_pbc: bool = True) -> list[np.ndarray]:

    labels, n = ndimage.label(arr)

    if not has_pbc:
        return [labels[labels == i] for i in range(1, n + 1)]

    # ndimage labels regions from [0, 1, ..., n].
    # 0 - only False values. 1 and above - seperate regions of True values
    bool_masks = []
    for i in range(1, n + 1):
        # Check to see if they are joined. Currently checks if any cells of label i on a border
        # line up with another on the other side.
        xmins, xmaxs, ymins, ymaxs = (
            labels[:, 0] == i,
            labels[:, -1] == i,
            labels[0, :] == i,
            labels[-1, :] == i,
        )
        # Check continuity along y-borders
        i_to_convert_y = []
        if np.any(labels[0, :] == i):
            unique = np.unique(labels[-1, np.where(ymins)[0]])
            i_to_convert_y.append(unique[unique != 0][0])
        elif np.any(labels[-1, :] == i):
            unique = np.unique(labels[0, np.where(ymaxs)[0]])
            i_to_convert_y.append(unique[unique != 0][0])
        labels[np.isin(labels, i_to_convert_y)] = i

        # Check continuity along x-borders
        i_to_convert_x = []
        if np.any(labels[:, 0] == i):
            unique = np.unique(labels[np.where(xmins)[0], -1])
            i_to_convert_x.append(unique[unique != 0][0])
        elif np.any(labels[:, -1] == i):
            unique = np.unique(labels[np.where(xmaxs)[0], 0])
            i_to_convert_x.append(unique[unique != 0][0])
        labels[np.isin(labels, i_to_convert_x)] = i

        # Discard empty masks ( converted by other labels )
        if np.any(labels == i):
            bool_masks.append(labels == i)
    return bool_masks


def generate_masks_from_roi(
    arr_shape: tuple[int, ...], roi, maskpath: Path | str, has_pbc: bool = True
) -> list[Any]:
    if roi is None:
        return [slice(None)]
    if not isinstance(roi, dict):
        raise NotImplementedError

    if roi["type"] == "slice":
        t = slice(None)
        z = slice(None)
        y = slice(roi["ymin"], roi["ymax"])
        x = slice(roi["xmin"], roi["xmax"])
        comps = slice(None)
        return [boundary_slice_mask(arr_shape, [t, z, y, x, comps])]

    if roi["type"] == "auto":
        # For mumax3, masks have False (black) where there is material, and True (white) where there isn't
        mask_arr = ~np.asarray(Image.open(maskpath))
        bool_masks = auto_bool_masks(mask_arr, has_pbc)  # 2D for discretising y and x respectively
        return [b[None, :, :, None, None] for b in bool_masks]  # Fill empty axes

    if roi["type"] == "shape":
        arr = ~np.asarray(Image.open(roi["path"]))
        return [arr[None, :, :, None, None]]

    raise NotImplementedError


def apply_mask(
    arr,
    mask: Callable | np.ndarray | slice | tuple,
    crop_first: bool = False,
    has_pbc: bool = True,
):

    if isinstance(mask, Callable):
        if crop_first:
            print("Cannot crop first dimension of mask as it is a function!")
        return mask(arr)
    if isinstance(mask, slice):
        # Slice can only be 1D, so if first dim cropped, it doesn't apply.
        return arr if crop_first else arr[mask]
    if isinstance(mask, tuple):
        # If crop_first = True -> starts from after first, else from beginning
        return arr[mask[crop_first:]]
    if isinstance(mask, np.ndarray):
        # First case - boolean mask (when dtype == bool):
        if mask.dtype == bool:
            if crop_first and mask.shape[0] != 1:
                print("Warning: First dimension has shape when cropping bool mask!")
            return apply_bool_mask(arr, mask[0, ...] if crop_first else mask, has_pbc=has_pbc)
        else:
            raise NotImplementedError
    raise NotImplementedError


def get_mask_applied_shape(
    shape,
    mask: Callable | np.ndarray | slice | tuple,
    crop_first: bool = False,
    has_pbc: bool = True,
):
    return apply_mask(np.empty(shape, dtype=bool), mask, crop_first, has_pbc).shape
