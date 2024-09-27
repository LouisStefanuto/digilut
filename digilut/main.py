from pathlib import Path

import typer
from pyfiglet import Figlet
from typing_extensions import Annotated

from digilut.bbox_filtering import clean
from digilut.logs import get_logger
from digilut.pyfast.main import app_pyfast
from digilut.tiles.main import app_tiles
from digilut.undersample import app_undersample
from digilut.viz.extract_rois import extract_rois

app = typer.Typer(help="Entrypoint of Digilut's main CLI.")
app.add_typer(app_tiles, name="tiles", deprecated=True)
app.add_typer(app_pyfast, name="pyfast")
app.add_typer(app_undersample, name="undersample")


logger = get_logger(__name__)


@app.command()
def credits() -> None:
    """Print credits with style."""
    FORMAT = "slant"
    TEXT = "DigiLut"
    f = Figlet(font=FORMAT)
    print("Welcome to")
    print(f.renderText(TEXT))
    print("LouisStefanuto. 2024.")


@app.command()
def clean_bbox(
    csv_bboxes: Annotated[Path, typer.Argument(help="Bounding box csv to clean ")],
    csv_output: Annotated[Path, typer.Argument(help="Name of the cleaned csv")],
    show_plots: Annotated[bool, typer.Option(help="Plot figures")] = False,
    train: Annotated[
        bool,
        typer.Option(
            help="In train mode run some check on rows, disabled for validation."
        ),
    ] = True,
):
    """
    Remove obvious labelling mistakes from the input bounding box csv and save a new
    cleaned dataset csv.
    """
    clean(csv_bboxes, csv_output, show_plots, train)


@app.command()
def export_rois(
    csv_file: Annotated[Path, typer.Argument(help="Path to the CSV file with bboxes.")],
    image_dir: Annotated[Path, typer.Argument(help="Dir containing the TIFF images.")],
    output_dir: Annotated[Path, typer.Argument(help="Dir to save the extracted PNGs.")],
):
    """
    Extract ROIs from TIFF images based on bounding boxes in a CSV file.
    """
    extract_rois(csv_file, image_dir, output_dir)


if __name__ == "__main__":
    app()
