from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import typer
from skimage.transform import resize

from digilut.logs import get_logger

app = typer.Typer()

logger = get_logger(__name__)


def load_slide(slide_path: Path, downscale_factor: int):
    slide = openslide.OpenSlide(slide_path)
    thumbnail_dimensions = (
        slide.dimensions[0] // downscale_factor,
        slide.dimensions[1] // downscale_factor,
    )
    thumbnail = slide.get_thumbnail(size=thumbnail_dimensions)
    image = np.array(thumbnail)
    logger.info(f"Slide: {slide_path.name}")
    logger.info(f"Shape thumbnail: {image.shape}")
    return image, slide


def load_predictions(predictions_csv: Path, slide_name: str):
    df = pd.read_csv(predictions_csv)
    df = df[df["slide"] == slide_name]
    logger.debug(df.describe().T)
    return df


def create_heatmap(df: pd.DataFrame, slide_dimensions: tuple, patch_size: int):
    heatmap_dimensions = (
        1 + slide_dimensions[1] // patch_size,
        1 + slide_dimensions[0] // patch_size,
    )
    heatmap = np.zeros(heatmap_dimensions)
    logger.info(f"Heatmap dimensions: {heatmap_dimensions}")

    for _, row in df.iterrows():
        try:
            heatmap[int(row["IDy"]), int(row["IDx"])] = row["prediction"] * 255
        except IndexError as e:
            logger.warning(e)

    # Normalize the heatmap
    heatmap_normalized = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    return heatmap_normalized


def resize_heatmap(heatmap: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Resize the heatmap to the size of the image, to plot them one over the other.

    Args:
        heatmap (np.ndarray): heatmap
        image_shape (tuple): target shape

    Returns:
        np.ndarray: resized heatmap
    """
    heatmap_resized = resize(
        heatmap,
        image_shape[:2],
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return heatmap_resized


def draw_ground_truth_bboxes(
    ax, gt_bboxes_csv: Path, slide_name: str, downscale_factor: int
):
    """Add the ground truth bounding boxes to the plot.

    Args:
        ax (_type_): matplotlib plot
        gt_bboxes_csv (Path): ground truth bounding boxes dataframe
        slide_name (str): name of the slide, without the file extension
        downscale_factor (int): Scaling factor used to reduce the size of the WSI, for the thumbnail.
    """
    df_gt_bboxes = pd.read_csv(gt_bboxes_csv, index_col=0)
    df_gt_bboxes = df_gt_bboxes[df_gt_bboxes["slideName"] == slide_name]

    for _, row in df_gt_bboxes.iterrows():
        bbox = (
            (row.x1 // downscale_factor, row.y1 // downscale_factor),
            max(1, (row.x2 - row.x1) // downscale_factor),
            max(1, (row.y2 - row.y1) // downscale_factor),
        )
        rect = patches.Rectangle(
            *bbox,
            linewidth=0.1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        logger.info(f"Painted bbox: {bbox}")


@app.command()
def heatmap(
    slide_path: Path,
    predictions_csv: Path,
    gt_bboxes_csv: Path,
    downscale_factor: int = 16,
    patch_size: int = 512,
    output_folder: Path = Path("heatmaps"),
) -> None:
    """Generate a heatmap of the patch predictions for the slide.

    Save the plot as a PNG image.

    Args:
        slide_path (Path): Path to the TIF slide.
        predictions_csv (Path): Dataframe with the prediction scores for each patch, with their locations in the slide.
        gt_bboxes_csv (Path): Dataframe with the ground truth bounding boxes.
        downscale_factor (int, optional): Scaling factor used to reduce the size of the WSI, for the thumbnail. Defaults to 16.
        patch_size (int, optional): Size of the patches used at inference time, converted to level 0 dimensions. Defaults to 512.
        output_folder (Path, optional): Output folder. Defaults to Path("heatmaps").
    """

    # Load slide image and its dimensions
    image, slide = load_slide(slide_path, downscale_factor)

    # Load predictions
    df = load_predictions(predictions_csv, slide_path.stem)

    # Create heatmap
    heatmap_normalized = create_heatmap(df, slide.dimensions, patch_size)

    # Resize heatmap
    heatmap_resized = resize_heatmap(heatmap_normalized, image.shape)

    logger.info(f"Image shape: {image.shape}")
    logger.info(f"Mask shape: {heatmap_normalized.shape}")

    # -------------

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image, the heatmap and the bboxes
    ax.imshow(image)
    ax.imshow(heatmap_resized, cmap="jet", alpha=0.25)
    draw_ground_truth_bboxes(ax, gt_bboxes_csv, slide_path.stem, downscale_factor)

    # Save as PNG
    output_path = output_folder / f"heatmap_{slide_path.stem}.png"
    plt.savefig(output_path, dpi=2000, pad_inches=0)
    plt.close()


if __name__ == "__main__":
    app()
