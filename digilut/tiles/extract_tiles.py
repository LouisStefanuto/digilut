import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import typer
from openslide.deepzoom import DeepZoomGenerator

from digilut.logs import get_logger
from digilut.tiles.filter_tiles import is_not_tissue

app = typer.Typer()
logger = get_logger(__name__)


def save_tile(
    tiles: DeepZoomGenerator,
    level: int,
    adress: tuple[int, int],
    output_dir: Path,
) -> None:
    # Retrieve the tile
    temp_tile = tiles.get_tile(level, adress)

    # Check if the tile is blank
    if not is_not_tissue(temp_tile):
        tile_name = os.path.join(output_dir, "%d_%d" % adress) + ".png"

        # Save the tile if it's not blank
        arr = np.array(temp_tile)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tile_name, arr)


def save_thumbnail(
    slide: openslide.OpenSlide,
    output_name: Path,
    size: tuple[int, int] = (500, 500),
) -> None:
    """Save a PNG thumbnail of the slide.

    Args:
        slide (openslide.OpenSlide): _description_
        size (tuple[int, int], optional): _description_. Defaults to (500, 500).
    """
    output_name.parent.mkdir(exist_ok=True, parents=True)

    thumbnail = slide.get_thumbnail(size=size)
    thumbnail.save(output_name)
    logger.info(f"Thumbnail saved: {output_name}")


def extract_tiles(
    path_tiff: Path, output_dir: Path, tile_size: int = 1024, parallel: bool = True
) -> None:
    """Extract PNG tiles from tiff slide. Removes all the background slides.

    Args:
        path_tiff (str): path to TIFF file to extract
    """
    logger.info(f"Starting to tile. File: {path_tiff}")
    dir_tiles = output_dir / "tiles"
    dir_tiles.mkdir(exist_ok=True, parents=True)

    # Create tiles from slide
    slide = openslide.OpenSlide(path_tiff)
    tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)

    # Get the maximum resolution level (the best one) and the number of tiles at this resolution
    MAX_RES_LEVEL = tiles.level_count - 1
    cols, rows = tiles.level_tiles[MAX_RES_LEVEL]

    # ------------------

    # Parallelize the tile saving
    if parallel:
        logger.info("Parallelization enabled.")

        # Specify the number of workers based on your CPU capabilities
        num_workers = min(32, os.cpu_count() + 4)  # type: ignore

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for row in range(rows):
                for col in range(cols):
                    args = (tiles, MAX_RES_LEVEL, (col, row), dir_tiles)
                    futures.append(executor.submit(save_tile, *args))

            # Optionally wait for all futures to complete
            for future in futures:
                future.result()
    else:
        logger.info("Parallelization diasbled.")

        for row in range(rows):
            for col in range(cols):
                save_tile(tiles, MAX_RES_LEVEL, (col, row), dir_tiles)

    logger.info("Finished tiling.")

    # ------------------

    # Save a low resolution PNG of the slide
    thumbnail_name = output_dir / "info" / "thumbnail.png"
    save_thumbnail(slide, output_name=thumbnail_name)
    # Save a mask of the selected tiles
    mask_name = output_dir / "info" / "mask_tiles.png"
    generate_mask(dir_tiles, output_name=mask_name, mask_shape=(rows, cols))


def generate_mask(
    dir_tiles: Path,
    output_name: Path,
    mask_shape: tuple[int, int],
) -> None:
    # Initialize an empty list to store the tuples
    tiles_saved = []

    # Iterate over each file in the directory
    for filename in os.listdir(dir_tiles):
        stem = Path(filename).stem

        # Split the filename by underscores
        parts = stem.split("_")

        # Extract X and Y from the parts
        col, row = int(parts[0]), int(parts[1])
        tiles_saved.append((col, row))

    # Fill with 1 if selected tile. ! Swap col/row !
    mask = np.zeros(mask_shape)
    for col, row in tiles_saved:
        mask[row, col] = 1

    # Save mask image
    output_name.parent.mkdir(exist_ok=True, parents=True)
    plt.figure()
    plt.imshow(mask)
    plt.savefig(output_name)
