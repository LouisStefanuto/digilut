import datetime
import time
from pathlib import Path

import fast
import pandas as pd
import typer
from tqdm import tqdm

from digilut.logs import get_logger

logger = get_logger(__name__)

app = typer.Typer(help="Create meatadata for all patches, without creating patch imgs")


def create_patch_generator(
    tiff_path: Path,
    patch_size: int = 256,
    level: int = 0,
    overlap_percent: float = 0.0,
) -> fast.PatchGenerator:
    # Initialize the importer, segmentation, and patch generator
    importer = fast.WholeSlideImageImporter.create(tiff_path.as_posix())

    # Create patch generator
    tissueSegmentation = fast.TissueSegmentation.create().connect(importer)
    patchGenerator: fast.PatchGenerator = (
        fast.PatchGenerator.create(
            patch_size,
            patch_size,
            level=level,
            overlapPercent=overlap_percent,
        )
        .connect(0, importer)
        .connect(1, tissueSegmentation)
    )
    return patchGenerator


def get_patch_info(patch: fast.fast.Image) -> dict:
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

    return {
        "level": patch_level,
        "width": patch_width,
        "height": patch_height,
        "overlap_x": patch_overlap_x,
        "overlap_y": patch_overlap_y,
        "idx": patch_id_x,
        "idy": patch_id_y,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "patch_name": f"{patch_id_x}_{patch_id_y}_{x_pos}_{y_pos}_{patch_level}_{patch_width}_{patch_height}",
    }


def pre_patchify_slide(
    tiff_path: Path,
    patch_size: int = 256,
    level: int = 0,
    overlap_percent: float = 0.0,
) -> list:
    # Create pyfast iterator
    patch_generator = create_patch_generator(
        tiff_path,
        patch_size,
        level,
        overlap_percent,
    )
    # Iterate over generator
    slide_patches = []

    patch: fast.fast.Image
    for patch in fast.DataStream(patch_generator):
        # Export patch metadata for easier reuse
        patch_info = get_patch_info(patch)
        patch_info["slide"] = tiff_path.stem
        slide_patches.append(patch_info)

    # Export metadata to CSV
    return slide_patches


@app.command()
def pre_patchify_dataset(
    csv_bboxes_cleaned: Path,
    folder_slides: Path,
    csv_patches_output: Path,
    patch_size: int = 256,
    level: int = 0,
    overlap_percent: float = 0.0,
):
    print("Started extracting patches metadata.")

    # Load bbox csv file
    df_bboxes = pd.read_csv(csv_bboxes_cleaned, index_col=0)
    # Get unique slide names
    slide_names = set(df_bboxes.filename)

    # Init list of records
    dataset_patches = []

    # For each tiff file, pre-patchify it and get metadata
    for slide_name in tqdm(slide_names):
        start_time = time.time()
        print("Extract patches metadata from {} ...".format(slide_name))

        tiff_path = folder_slides / slide_name
        patches = pre_patchify_slide(tiff_path, patch_size, level, overlap_percent)
        dataset_patches.append(patches)

        execution_time = int(time.time() - start_time)
        execution_time = datetime.timedelta(seconds=execution_time)  # type: ignore
        print("Execution time: {} seconds".format(str(execution_time)))

    # Export to patches from whole dataset to csv
    print("Exporting ...")
    pd.DataFrame.from_records(dataset_patches).to_csv(csv_patches_output)

    print("Finished extracting patches metadata.")


if __name__ == "__main__":
    app()
