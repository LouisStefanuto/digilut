from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from shapely import box
from tqdm import tqdm

from digilut.logs import get_logger

logger = get_logger(__name__)


@dataclass
class PatchMetadata:
    slide: str
    patchName: str
    intersectionScore: float
    label: int


def create_tile_labels_v2(
    csv_bboxes: Path,
    folder_patches: Path,
    csv_labels: Path,
    threshold_iou: float = 0.1,
) -> None:
    """
    Create labels for the tiles.

    A tile is positive if it intersects at more that XX% with a bounding box.

    Args:
        csv_bboxes (Path): bounding box file
        folder_patches (Path): folder containing the patch images
        csv_labels (Path): output file returned, contains a mapping tile/label
        threshold_iou (float, optional): threshold of intersection to flag positive. Defaults to 0.1.
    """
    # Read bounding boxes
    df_bboxes = pd.read_csv(csv_bboxes)

    # For each unique slide
    slide_names = df_bboxes["filename"].unique()
    logger.info(f"Nb unique slides with annotations: {len(slide_names)}")
    logger.info("Associating labels to patches ...")

    records = []

    for slide_name in tqdm(slide_names):
        # Get bboxes for this slide
        slide_bboxes = df_bboxes[df_bboxes.filename == slide_name]

        # Get patches for this slide
        slide_folder = folder_patches / Path(slide_name).stem

        # Get slide CSV info
        slide_info = pd.read_csv(slide_folder / "info" / "metadata.csv")

        # For each patch
        for _, patch_info in slide_info.iterrows():
            # patch_info = [patchName, IDx, IDy, x, y, patch_width,patch_height, level, path]
            # Compute the patch coordinates
            x1, y1, level, w, h = (
                patch_info.x,
                patch_info.y,
                patch_info.level,
                patch_info.patch_width,
                patch_info.patch_height,
            )
            x2, y2 = x1 + (level + 1) * w, y1 + (level + 1) * h
            patch_box = box(x1, y1, x2, y2)

            # Set label to 0 by default
            patch_label = 0
            score = 0.0

            # Check intersection with each bbox
            for _, row_gt in slide_bboxes.iterrows():
                box_gt = box(row_gt.x1, row_gt.y1, row_gt.x2, row_gt.y2)

                # Criterion: positive if more that XX% of the patch is in the bbox_gt
                intersection = patch_box.intersection(box_gt).area
                score = intersection / patch_box.area
                if score >= threshold_iou:
                    # Set label to 1
                    patch_label = 1
                    break

            patch_mtd = PatchMetadata(
                slide=Path(slide_name).stem,
                patchName=Path(patch_info.patchName).stem,
                intersectionScore=round(score, 4),
                label=patch_label,
            )
            records.append(asdict(patch_mtd))

    logger.info("Creating dataframe ...")
    df_labels = pd.DataFrame.from_records(records)

    logger.info("Polishing columns ...")

    # Add slide and patient ID/Name
    right = df_bboxes[["slideName", "slideID", "patientName", "patientID"]]
    right.drop_duplicates(inplace=True)
    if right.slideName.duplicated().sum() != 0:
        raise ValueError("Duplicated rows in your output label file. Check it.")

    LEFT_ON, RIGHT_ON = "slide", "slideName"
    final_df = df_labels.merge(right, how="inner", left_on=LEFT_ON, right_on=RIGHT_ON)
    final_df.drop(columns=[LEFT_ON], inplace=True)

    # Move slide name to first place for readibility
    final_df.insert(0, RIGHT_ON, final_df.pop(RIGHT_ON))

    logger.info("Exporting to CSV ...")
    final_df.to_csv(csv_labels)

    logger.info("Finished label generation.")
