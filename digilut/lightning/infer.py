from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from digilut.lightning.config import load_config
from digilut.lightning.dataset import PatchDataset, get_transform_val
from digilut.lightning.model import PatchClassificationModel
from digilut.lightning.utils import prepare_dataset
from digilut.logs import get_logger

app = typer.Typer()
logger = get_logger(__name__)


def parse_fn(path_str: str) -> tuple:
    p = Path(path_str)
    slide_name, stem = p.parent.parent, p.stem
    idx, idy, x, y, level, patch_width, patch_height = [int(x) for x in stem.split("_")]
    return slide_name.stem, idx, idy, x, y, level, patch_width, patch_height


@app.command()
def infer(config_path: Path) -> None:
    config = load_config(config_path)

    # Load and prepare the test data
    df = pd.read_csv(config.test_data_path, index_col=0)
    df = prepare_dataset(df, config.test_image_dir)

    # Define the test dataset and dataloader
    test_dataset = PatchDataset(df, get_transform_val())
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    n_samples = len(test_dataset)
    n_batches = len(test_dataset) // config.batch_size
    logger.info(f"Dataset: {n_samples:,} samples")
    logger.info(f"Dataloader: {n_batches:,} batches")

    # Load the trained model from the checkpoint
    model = PatchClassificationModel.load_from_checkpoint(config.checkpoint_path)
    logger.info(f"Model loaded: {config.checkpoint_path}")

    device = torch.device(config.device)
    model.to(device)
    logger.info(f"Moved model to {device}.")

    model.eval()
    logger.info("Switched model to eval mode.")

    predictions = []
    image_paths = []
    labels = []

    # Inference loop
    with torch.no_grad():
        for batch in tqdm(test_loader, total=n_batches):
            # Get model predictions
            batch_img_paths, inputs, batch_labels = batch
            outputs = model(inputs.to(device))
            preds = torch.sigmoid(outputs)
            preds = preds.float().cpu().numpy()

            image_paths.extend(batch_img_paths)
            labels.extend(batch_labels.tolist())
            predictions.extend(preds.flatten().tolist())

    # Save the predictions to a CSV file
    output_df = pd.DataFrame(
        {
            "imgPath": image_paths,
            "prediction": np.round(predictions, 5),
            "labels": labels,
        }
    )

    # Add useful columns for downstream tasks
    COLS = ["slide", "IDx", "IDy", "x", "y", "level", "patch_width", "patch_height"]
    output_df[COLS] = output_df["imgPath"].apply(parse_fn).tolist()

    # Order columns for output
    ORDERED_COLS = ["imgPath"] + COLS + ["prediction", "labels"]
    output_df[ORDERED_COLS].to_csv(config.predictions_path, index=False)
    logger.info(f"Predictions saved to {config.predictions_path}")


if __name__ == "__main__":
    app()
