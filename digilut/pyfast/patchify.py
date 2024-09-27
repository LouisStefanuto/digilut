import datetime
import os
import time
from pathlib import Path

import cv2
import fast
import numpy as np
import pandas as pd
import typer
from PIL import Image

from digilut.logs import get_logger

app_pyfast = typer.Typer(help="Pyfast processing commands to patchify .tif WSI images.")
logger = get_logger(__name__)


def save_arr(arr: np.ndarray, patch_path: Path, save_engine: str) -> None:
    if save_engine == "pillow":
        # Convert the NumPy array to a PIL Image
        img = Image.fromarray(arr)
        # Save the image
        img.save(patch_path)

    elif save_engine == "cv2":  # x2-4 faster
        # FAST returns PIL images (RGB), open-cv expects BGR images
        cv2.imwrite(patch_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    else:
        raise ValueError(f"Mode not implemented: {save_engine}")


def patchify_slide(
    tiff_path: Path,
    output_dir: Path,
    save_engine: str = "cv2",
    patch_size: int = 256,
    level: int = 0,
    overlap_percent: float = 0.0,
    img_format: str = "jpg",
) -> None:
    """
    Exports the TIFF file into PNG tiles of tissue.

    Args:
        tiff_path (Path): tiff slide to patchify
        output_dir (Path): output directory for the patches
        save_engine (str, optional): engine used to save patches. Defaults to "cv2".
        patch_size (int, optional): size of the patches. Defaults to 256.
        level (int, optional): Zoom level. 0 is the most zoomed in. Defaults to 0.
        overlap_percent (float, optional): Overlap % for sliding window tiling. Defaults to 0.0.
        img_format (str, optional): Format of the output images. Defaults to "jpg".
    """
    # Ensure the output directories exists
    subfolder_patches = output_dir / "patches"
    subfolder_info = output_dir / "info"
    subfolder_patches.mkdir(parents=True, exist_ok=True)
    subfolder_info.mkdir(parents=True, exist_ok=True)

    # Initialize the importer, segmentation, and patch generator
    importer = fast.WholeSlideImageImporter.create(tiff_path.as_posix())
    tissueSegmentation = fast.TissueSegmentation.create().connect(importer)
    patchGenerator: fast.PatchGenerator = (
        fast.PatchGenerator.create(
            patch_size, patch_size, level=level, overlapPercent=overlap_percent
        )
        .connect(0, importer)
        .connect(1, tissueSegmentation)
    )

    start_time = time.time()
    logger.info(f"Starting to generate patches from slide: {tiff_path}")

    patch_records = []

    patch: fast.fast.Image
    for patch in fast.DataStream(patchGenerator):
        # Keep track of the tile position and size, it is possible to emulate the filenames produced
        patch_level = int(patch.getFrameData("patch-level"))

        patch_width, patch_height = (
            int(patch.getFrameData("patch-width")),
            int(patch.getFrameData("patch-height")),
        )
        patch_overlap_x, patch_overlap_y = (
            int(patch.getFrameData("patch-overlap-x")),
            int(patch.getFrameData("patch-overlap-y")),
        )
        patch_id_x = int(patch.getFrameData("patchid-x"))
        patch_id_y = int(patch.getFrameData("patchid-y"))

        # Compute (x,y) position
        x_pos = patch_id_x * (patch_width - patch_overlap_x * 2) + patch_overlap_x
        y_pos = patch_id_y * (patch_height - patch_overlap_y * 2) + patch_overlap_y

        x_pos = (patch_level + 1) * x_pos
        y_pos = (patch_level + 1) * y_pos

        # Define the file path for the current tile
        patch_name = f"{patch_id_x}_{patch_id_y}_{x_pos}_{y_pos}_{patch_level}_{patch_width}_{patch_height}.{img_format}"
        patch_path = subfolder_patches / patch_name

        # Save patch as image
        arr = np.array(patch)
        save_arr(arr, patch_path, save_engine)

        # Save patch metadata for easier reuse
        record = {
            "patchName": patch_name,
            "IDx": patch_id_x,
            "IDy": patch_id_y,
            "x": x_pos,
            "y": y_pos,
            "patch_width": patch_width,
            "patch_height": patch_height,
            "level": level,
            "path": patch_path,
        }
        patch_records.append(record)

    # Export metadata to CSV
    csv_metadata = Path(subfolder_info / "metadata.csv")
    pd.DataFrame.from_records(patch_records).to_csv(csv_metadata)

    # Final logs
    logger.info("Finished patch generation.")
    logger.info(
        f"Output dir {subfolder_patches} contains {len(os.listdir(subfolder_patches))} tiles"
    )
    execution_time = int(time.time() - start_time)
    logger.info(
        f"Execution time: {str(datetime.timedelta(seconds=execution_time))} seconds"
    )
