from pathlib import Path

import openslide
import pandas as pd


def extract_rois(csv_file: Path, image_dir: Path, output_dir: Path):
    """
    Extract regions of interest (ROIs) from TIFF images based on the bounding boxes
    provided in the CSV file and save them as PNG files.

    Args:
        csv_file (Path): Path to the CSV file containing bounding boxes.
        image_dir (Path): Path to the directory containing TIFF images.
        output_dir (Path): Path to the directory where PNG files will be saved.
    """
    # Load CSV file containing bounding boxes
    df = pd.read_csv(csv_file)

    # Ensure the output directory exists
    output_dir.mkdir(exist_ok=True)

    # Loop through each unique filename in the dataset
    for filename in df["filename"].unique():
        # Open the TIFF image using OpenSlide
        slide_path = image_dir / filename
        slide = openslide.OpenSlide(slide_path)

        # Filter the dataframe for bounding boxes belonging to the current image
        bboxes = df[df.filename == filename]

        # Loop through each bounding box and extract the region of interest
        for BBOX_ID, bbox in bboxes.iterrows():
            # Define the location and size of the region
            location = (bbox.x1, bbox.y1)  # top left pixel (col, row)
            level = 0  # zoom level
            size = (bbox.x2 - bbox.x1, bbox.y2 - bbox.y1)  # width, height

            # Extract the region from the slide
            region = slide.read_region(location, level, size)

            # Convert the region to an RGB image (removing alpha channel if necessary)
            region_rgb = region.convert("RGB")

            # Save the region as a PNG file with a systematic name
            output_filename = f"{filename.replace('.tif', '')}_{BBOX_ID}.png"
            output_path = output_dir / output_filename
            region_rgb.save(output_path)

            print(f"Saved: {output_path}")
