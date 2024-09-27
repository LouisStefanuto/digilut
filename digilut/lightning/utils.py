from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight


def prepare_dataset(
    df: pd.DataFrame, root_dir: Path, patches_subfolder: str = "patches"
):
    """
    Takes a openend dataframe as input and returns the dataframe expected by the
    pytorch datasets
    """
    expected_cols_present = set(["slideName", "patchName", "label"]) <= set(df.columns)
    if not expected_cols_present:
        raise ValueError("Missing mendatory columns in label CSV file.")

    df["imgPath"] = (
        root_dir.as_posix()
        + "/"
        + df.slideName
        + "/"
        + patches_subfolder
        + "/"
        + df.patchName
        + ".jpg"
    )
    return df


def compute_loss_weights(labels: pd.Series) -> torch.Tensor:
    # Calculate the class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    class_weights = torch.Tensor([class_weights[1] / class_weights[0]])
    print("Class weights:", class_weights)
    return class_weights
