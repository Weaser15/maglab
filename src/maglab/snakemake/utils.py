from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from ..loaders.mumax3 import load_ovf_array
from ..processing.state import boundary_slice_mask


def auto_bool_masks(arr: np.ndarray, has_pbc: bool = True) -> list[np.ndarray]:

    if (ndims := len(arr.shape)) > 3:
        raise ValueError(f"`arr` must havve at most 3 dimensions! Has {ndims}")

    arr = np.sum(np.square(arr), axis=-1).astype(bool) if ndims == 3 else arr.astype(bool)

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


def generate_masks_from_rois(
    rois: dict[dict, Any],
    arr_shape: tuple[int, int],
    maskpath: Path | str,
    has_pbc: bool = True,
) -> dict[str, Any]:
    """Convert rois to their representation."""
    if len(rois) == 0:
        return {"full": slice(None)}
    mask_dict = {}
    for name, roi in rois.items():
        if roi is None:
            mask_dict[name] = slice(None)
        if not isinstance(roi, dict):
            raise NotImplementedError(f"RoI {name} should be dict, but is type {type(roi)}!")
        match roi["type"].lower():
            case "slice":
                y = slice(roi["ymin"], roi["ymax"])
                x = slice(roi["xmin"], roi["xmax"])
                mask_dict[name] = boundary_slice_mask(arr_shape, [y, x])
            case "auto":
                mask_arr = ~np.asarray(Image.open(maskpath))
                if mask_arr.shape[:2] != arr_shape[:2]:
                    raise NotImplementedError(
                        "Currently only supports a mask that has exact simulation cell size!"
                    )
                bool_masks = auto_bool_masks(mask_arr, has_pbc)
                names = [f"{name}{i:03d}" for i in range(len(bool_masks))]
                mask_dict.update(dict(zip(names, bool_masks)))
            case "shape":
                arr = ~np.asarray(Image.open(roi["path"]))
                mask_dict[name] = arr
            case "full":
                mask_dict[name] = slice(None)
            case _:
                raise NotImplementedError(
                    "Currently only types ('slice', 'auto', 'shape' and 'full') are supported"
                )
    return mask_dict


def convert_to_bool_mask(
    shape: tuple[int, ...],
    mask: np.ndarray | slice | tuple,
    crop_first: bool = False,
):
    # Shape of the 2D mask (NY, NX)
    test_arr = np.zeros(shape, dtype=bool)
    if isinstance(mask, slice):
        if not crop_first:
            test_arr[mask] = True
    elif isinstance(mask, tuple):
        test_arr[mask[crop_first:]] = True
    elif isinstance(mask, np.ndarray) and mask.dtype == bool:
        if mask.shape[:2] == shape:
            test_arr = mask
        else:
            raise ValueError(
                f"Boolean `mask` is different shape from `shape`: {mask.shape} vs {shape}!"
            )
    else:
        raise NotImplementedError("Only supports slice, tuple and boolean masks!")

    return test_arr


def save_bool_mask(mask: np.ndarray, savedir: Path | str, name: str = "mask"):
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(~mask).save(savedir / f"{name}.png")


def auto_masks_from_ovf(path: Path | str, has_pbc: bool = True):
    path = Path(path)
    if path.is_dir():
        path = next(path.glob("*.ovf"))
    arr = load_ovf_array(path, zslice=0)
    return auto_bool_masks(arr, has_pbc=has_pbc)
