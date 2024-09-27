import numpy as np
import PIL


def is_not_tissue(tile: PIL.Image.Image) -> bool:
    arr_tile = np.array(tile.convert("L"))
    arr_tile_rgb = np.array(tile.convert("RGB"))
    return (
        has_too_many_white(arr_tile)
        or not is_green_weaker(arr_tile_rgb)
        or has_too_many_black(arr_tile)
    )


def has_too_many_white(
    arr_tile: np.ndarray,
    color_threshold: int = 240,
    proportion_threshold: float = 0.85,
) -> bool:
    mask = arr_tile > color_threshold
    prop_white_pixels = float(mask.mean())
    is_background = prop_white_pixels > proportion_threshold
    return is_background


def has_too_many_black(
    arr_tile: np.ndarray,
    color_threshold: int = 100,
    proportion_threshold: float = 0.5,
) -> bool:
    mask = arr_tile < color_threshold
    prop_white_pixels = float(mask.mean())
    is_background = prop_white_pixels > proportion_threshold
    return is_background


def is_mostly_black(
    arr_tile: np.ndarray,
    color_threshold: int = 100,
) -> bool:
    return float(arr_tile.mean()) < color_threshold


def is_green_weaker(arr_tile_rgb: np.ndarray, mean_gap: int = 3) -> bool:
    mean_per_channel = arr_tile_rgb.mean(axis=tuple(range(arr_tile_rgb.ndim - 1)))
    red, green, blue = mean_per_channel
    return green + mean_gap < np.mean([red, blue])
