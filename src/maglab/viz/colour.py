import numpy as np


def inplane_angle(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    angle_array = np.arctan2(arr[..., 1], arr[..., 0])
    angle_array[angle_array < 0] += 2 * np.pi
    return angle_array


def normalise_to_range(
    arr: np.ndarray, to_range: tuple[float, float], from_range: tuple[float, float] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    if from_range is None:
        from_range = (arr.min(), arr.max())
    arr = (arr - from_range[0]) / (from_range[1] - from_range[0])
    return arr * (to_range[1] - to_range[0]) + to_range[0]


def hls_to_rgb(h: np.ndarray, l: np.ndarray, s: np.ndarray) -> np.ndarray:
    if h.shape != l.shape or l.shape != s.shape:
        raise ValueError("Arrays need to be the same shape!")

    m2 = np.where(l <= 0.5, l * (1.0 + s), l + s - l * s)
    m1 = 2.0 * l - m2
    hues = np.stack([h + 1 / 3, h, h - 1 / 3], axis=-1) % 1.0
    f = np.clip(np.minimum(6 * hues, 4 - 6 * hues), 0, 1)
    return m1[..., np.newaxis] + (m2 - m1)[..., np.newaxis] * f


def get_lightness_colours(
    arr: np.ndarray,
    lightness: np.ndarray | None = None,
    saturation: np.ndarray | None = None,
    lightness_clim: tuple[float, float] | None = None,
    filter: bool = True,
) -> np.ndarray:
    hue = normalise_to_range(inplane_angle(arr), (0, 1), from_range=(0, 2 * np.pi))
    lightness_clim = (0, 1) if lightness_clim is None else lightness_clim
    lightness = normalise_to_range(arr[..., 2], lightness_clim) if lightness is None else lightness
    saturation = (
        np.ones_like(hue) if saturation is None else normalise_to_range(saturation, (0, 1))
    )

    rgb = hls_to_rgb(hue, lightness, saturation)
    rgba = np.empty((*rgb.shape[:-1], 4))
    alpha = ~np.logical_and(arr[..., 0] == 0, arr[..., 1] == 0) if filter else np.ones_like(hue)
    rgba = np.concatenate((rgb, alpha[..., np.newaxis]), axis=-1, dtype=np.float32)

    return rgba
