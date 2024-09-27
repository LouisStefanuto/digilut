"""Tools to export patches based on if they are TF, FP, FN ..."""

import shutil
from pathlib import Path

import pandas as pd
import typer

from digilut.logs import get_logger

app = typer.Typer()
logger = get_logger(__name__)


def classify_images(
    row,
    true_positive_dir: Path,
    false_positive_dir: Path,
    false_negative_dir: Path,
    threshold: float = 0.5,
):
    """
    Classify the images into true positives, false positives, or false negatives
    based on prediction and label values from a dataframe row.

    Parameters:
    - row: A row from the pandas DataFrame containing 'imgPath', 'prediction', and 'labels'.
    - true_positive_dir: Path to the directory where true positive images will be stored.
    - false_positive_dir: Path to the directory where false positive images will be stored.
    - false_negative_dir: Path to the directory where false negative images will be stored.
    """
    img_path = Path(row["imgPath"])  # Convert imgPath to a Path object
    prediction = row["prediction"]
    label = row["labels"]

    # Classify based on prediction and label
    if prediction >= threshold:
        if label == 1:
            # True Positive
            shutil.copy(img_path, true_positive_dir)
        elif label == 0:
            # False Positive
            shutil.copy(img_path, false_positive_dir)
    else:
        if label == 1:
            # False Negative
            shutil.copy(img_path, false_negative_dir)


@app.command()
def main(csv_file: Path, threshold: float = 0.5):
    """
    Main function to classify and organize images based on a CSV file's prediction and label values.

    Parameters:
    - csv_file: Path to the CSV file containing 'imgPath', 'prediction', and 'labels'.

    The CSV file must have the following columns:
    - imgPath: Path to the image file.
    - prediction: The predicted value (a float, e.g., between 0 and 1).
    - labels: The ground truth label (0 or 1).

    Images are copied into respective folders: 'true_positive', 'false_positive', and 'false_negative'.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Define directory paths using pathlib
    main_folder = Path("pred_analysis")
    true_positive_dir = main_folder / "true_positive"
    false_positive_dir = main_folder / "false_positive"
    false_negative_dir = main_folder / "false_negative"

    # Create folders if they don't exist
    for directory in [true_positive_dir, false_positive_dir, false_negative_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    # Apply classification to each row in the dataframe
    df.apply(
        lambda row: classify_images(
            row, true_positive_dir, false_positive_dir, false_negative_dir, threshold
        ),
        axis=1,
    )

    logger.info("Images classified and copied successfully.")


if __name__ == "__main__":
    app()
