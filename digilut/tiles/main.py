import datetime
import time
from pathlib import Path

import pandas as pd
import typer
from typing_extensions import Annotated

from digilut.logs import get_logger
from digilut.tiles.create_labels import create_tile_labels

from .extract_tiles import extract_tiles

app_tiles = typer.Typer(help="Commands for tiles.")
logger = get_logger(__name__)


@app_tiles.command()
def extract_from_dataset(
    csv_path: Annotated[Path, typer.Argument(help="Path to the CSV file")] = Path(
        "data/train.csv"
    ),
    output_dir: Annotated[Path, typer.Argument(help="Outputdir")] = Path("outputs"),
    tile_size: Annotated[
        int, typer.Option(help="Size of the tiles (height and width)")
    ] = 1024,
    parallel: Annotated[bool, typer.Option(help="Enable multiprocessing.")] = True,
) -> None:
    """Extract tiles a dataset of tiles. Calls extract_from_image over a folder of slides."""
    df = pd.read_csv(csv_path)
    slides = df.filename.unique()

    logger.info(f"Train dataset contains {len(slides)} unique slides.")

    for tiff_path in slides:
        extract_tiles(tiff_path, output_dir, tile_size, parallel)


@app_tiles.command()
def extract_from_image(
    path_tiff: Annotated[Path, typer.Argument(help="Path to the TIFF file")],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Output folder. The output will we saved in {outputdir}/{path_tiff.stem}"
        ),
    ],
    tile_size: Annotated[
        int, typer.Option(help="Size of the tiles (height and width)")
    ] = 1024,
    parallel: Annotated[
        bool, typer.Option("--no-parallel", "-p", help="Disable multiprocessing.")
    ] = True,
) -> None:
    """Extract tiles from the slide."""
    logger.info("Launching the tile command ...")
    start_time = time.time()

    sub_folder = output_dir / path_tiff.stem
    extract_tiles(path_tiff, sub_folder, tile_size, parallel)

    execution_time = int(time.time() - start_time)
    logger.info(f"Ended tiling: {path_tiff}")
    logger.info(
        f"Execution time: {str(datetime.timedelta(seconds=execution_time))} seconds"
    )


@app_tiles.command(deprecated=True)
def generate_labels(
    csv_bboxes: Annotated[Path, typer.Argument(help="Bounding box file.")],
    slide_folder: Annotated[
        Path,
        typer.Argument(
            help="Slide folder, that contains an `info` and a `tiles` subfolder."
        ),
    ],
    iou_thres: Annotated[
        float,
        typer.Option(
            help="Threshold Intersection over Union. If tile IOU > with a bounding box, the tile is labbeled positive."
        ),
    ] = 0.2,
) -> None:
    """Create the labels for the tiles. V1 Deprecated. Use digilut pyfast"""
    logger.info("Launching the create_tiles_labels command ...")
    start_time = time.time()

    tiles_folder = slide_folder / "tiles"
    output_csv = slide_folder / "info" / "labels.csv"
    create_tile_labels(csv_bboxes, tiles_folder, output_csv, iou_thres)

    execution_time = int(time.time() - start_time)
    logger.info(
        f"Execution time: {str(datetime.timedelta(seconds=execution_time))} seconds"
    )


if __name__ == "__main__":
    app_tiles()
