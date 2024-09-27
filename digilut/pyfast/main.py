import datetime
import time
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm
from typing_extensions import Annotated

from digilut.logs import get_logger
from digilut.pyfast.labels import create_tile_labels_v2

from .patchify import patchify_slide

app_pyfast = typer.Typer(help="Pyfast processing commands to patchify .tif WSI images.")
logger = get_logger(__name__)


@app_pyfast.command(name="patchify-slide")
def slide(
    tiff_path: Annotated[Path, typer.Argument(help="Path to the slide")],
    output_dir: Annotated[
        Path, typer.Argument(help="Output folder where patches will be saved")
    ],
    save_engine: Annotated[
        str, typer.Option(help="Engine to save image. 'cv2' (recommended) OR 'pillow'")
    ] = "cv2",
    patch_size: Annotated[
        int, typer.Option(help="Patch size (width and hieght)")
    ] = 256,
    level: Annotated[
        int, typer.Option(help="Zoom level. 0 is the best resolution")
    ] = 0,
    overlap_percent: Annotated[
        float, typer.Option(help="Percentage of overlap between patches")
    ] = 0.0,
    img_format: Annotated[
        str,
        typer.Option(
            help="Image format. PNG is better (no artifact) but it x5 heavier. Values: 'jpg', 'png'"
        ),
    ] = "jpg",
) -> None:
    """Exports the TIFF file into PNG tiles of tissue."""

    start_time = time.time()
    logger.info(f"Starting to generate patches from slide: {tiff_path}")

    patchify_slide(
        tiff_path=tiff_path,
        output_dir=output_dir,
        save_engine=save_engine,
        patch_size=patch_size,
        level=level,
        overlap_percent=overlap_percent,
        img_format=img_format,
    )

    execution_time = int(time.time() - start_time)
    logger.info(
        f"Execution time: {str(datetime.timedelta(seconds=execution_time))} seconds"
    )


@app_pyfast.command()
def patchify_dataset(
    csv_path: Annotated[Path, typer.Argument(help="Path to the CSV file")] = Path(
        "data/train.csv"
    ),
    images_dir: Annotated[
        Path, typer.Argument(help="Folder containing the .tif slide images")
    ] = Path("data/images"),
    output_dir: Annotated[
        Path,
        typer.Argument(help="Output dir. Each slide will have a {outputdir}/{slide}"),
    ] = Path("outputs"),
    save_engine: Annotated[
        str, typer.Option(help="Engine to save image. 'cv2' (recommended) OR 'pillow'")
    ] = "cv2",
    patch_size: Annotated[
        int, typer.Option(help="Patch size (width and hieght)")
    ] = 256,
    level: Annotated[
        int, typer.Option(help="Zoom level. 0 is the best resolution")
    ] = 0,
    overlap_percent: Annotated[
        float, typer.Option(help="Percentage of overlap between patches")
    ] = 0.0,
    img_format: Annotated[
        str,
        typer.Option(
            "--img_format",
            "-f",
            help="Image format. PNG is better (no artifact) but it x5 heavier. Values: 'jpg', 'png'",
        ),
    ] = "jpg",
) -> None:
    """Extracts tiles from a dataset of slides. Calls patchify-slide over each slide in the folder."""
    df = pd.read_csv(csv_path)
    slides = df.filename.unique()

    logger.info(f"Dataset contains {len(slides)} unique slides.")

    run_records = []
    for slide in tqdm(slides):
        successful_tiling = True
        try:
            patchify_slide(
                tiff_path=images_dir / slide,
                output_dir=output_dir / Path(slide).stem,
                level=level,
                patch_size=patch_size,
                overlap_percent=overlap_percent,
                save_engine=save_engine,
                img_format=img_format,
            )
        except Exception:
            logger.warning(f"Patch extraction failed for {slide}")
            successful_tiling = False

        record = {"slide": slide, "sucessfulTiling": successful_tiling}
        run_records.append(record)

    # Helps debug if some slides fail
    pd.DataFrame.from_records(run_records).to_csv(output_dir / "run_records.csv")


@app_pyfast.command()
def labels(
    csv_bboxes: Path,
    folder_patches: Path,
    csv_labels: Path,
    threshold_iou: float = 0.1,
) -> None:
    """
    Create labels for the tiles.

    A tile is positive if it intersects at more that XX% with a bounding box.
    """
    create_tile_labels_v2(csv_bboxes, folder_patches, csv_labels, threshold_iou)


if __name__ == "__main__":
    app_pyfast()
