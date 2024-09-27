import os
from pathlib import Path

import pandas as pd
from shapely.geometry import box

from digilut.logs import get_logger

logger = get_logger(__name__)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    b1 = box(*box1)
    b2 = box(*box2)
    intersection = b1.intersection(b2).area
    union = b1.union(b2).area
    return intersection / union if union > 0 else 0


def get_tile_bounds(tile_path: Path, tile_size: int = 1024):
    """Get the bounding box coordinates of the tile."""
    stem = tile_path.stem
    col, row = stem.split("_")
    col, row = int(col), int(row)  # type: ignore
    x1, y1 = col * tile_size, row * tile_size
    return (x1, y1, x1 + tile_size - 1, y1 + tile_size - 1)  # type: ignore


def create_tile_labels(
    csv_bboxes: Path,
    tiles_folder: Path,
    output_csv: Path,
    iou_thres: float,
) -> None:
    """Create the labels for the tiles."""
    filename = tiles_folder.parent.name
    logger.info(f"Image name: {filename}")

    # Read bounding boxes
    df_bboxes = pd.read_csv(csv_bboxes)
    df_bboxes = df_bboxes[df_bboxes.filename == filename + ".tif"]

    # Create a list to store tile information and labels
    results = []

    # Get tile paths
    tiles = os.listdir(tiles_folder)
    for tile in tiles:
        tile_bounds = get_tile_bounds(Path(tile))
        tile_intersects = False
        for _, row in df_bboxes.iterrows():
            bbox = (row["x1"], row["y1"], row["x2"], row["y2"])
            if calculate_iou(tile_bounds, bbox) > iou_thres:
                tile_intersects = True
                break

        results.append({"tile": tile, "label": 1 if tile_intersects else 0})

    # Create DataFrame for output
    df_output = pd.DataFrame(results)
    df_output.to_csv(output_csv, index=False)
    logger.info(f"Found {df_output.label.sum()} positives tiles.")
    logger.info(f"Output written to {output_csv}")
