# def generate_masks_from_roi(
#     arr_shape: tuple[int, ...], roi, maskpath: Path | str, has_pbc: bool = True
# ) -> list[Any]:
#     if roi is None or roi["type"] == "full":
#         return [slice(None)]
#     if not isinstance(roi, dict):
#         raise NotImplementedError(f"Is type {type(roi)}")

#     if roi["type"] == "slice":
#         t = slice(None)
#         z = slice(None)
#         y = slice(roi["ymin"], roi["ymax"])
#         x = slice(roi["xmin"], roi["xmax"])
#         comps = slice(None)
#         return [boundary_slice_mask(arr_shape, [t, z, y, x, comps])]

#     if roi["type"] == "auto":
#         # For mumax3, masks have False (black) where there is material, and True (white) where there isn't
#         mask_arr = ~np.asarray(Image.open(maskpath))
#         if mask_arr.shape[:2] != arr_shape[2:4]:
#             raise NotImplementedError(
#                 "Currently only supports a mask that has exact simulation cell size!"
#             )
#         bool_masks = auto_bool_masks(mask_arr, has_pbc)  # 2D for discretising y and x respectively
#         return [b[None, :, :, None, None] for b in bool_masks]  # Fill empty axes

#     if roi["type"] == "shape":
#         arr = ~np.asarray(Image.open(roi["path"]))
#         return [arr[None, :, :, None, None]]

#     raise NotImplementedError


# def save_bool_masks(masks: list[np.ndarray], savedir: Path | str, name: str = "mask"):
#     savedir = Path(savedir)
#     savedir.mkdir(parents=True, exist_ok=True)
#     # Save them as mask1.png, mask2.png, ...
#     # Need to first find out the indices to use for saving.
#     used_indices = [int(m.stem.removeprefix(name)) for m in savedir.glob(f"{name}*.png")]
#     # Assuming that used indices are not just random numbers, and will be close to what is needed.
#     indices = np.arange(len(used_indices) + len(masks))
#     indices = indices[~np.isin(indices, used_indices)]
#     assert len(indices) == len(masks)

#     for mask, index in zip(masks, indices):
#         Image.fromarray(~mask).save(savedir / f"{name}{index}.png")
