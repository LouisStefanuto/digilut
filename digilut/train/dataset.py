from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self, labels_df: pd.DataFrame, root_dir: Path, name: str, transform=None
    ):
        """
        Expected columns in the CSV:
        - imgPath
        - label
        """
        self.labels_df = labels_df
        self._check_csv_cols(["imgPath", "label"])
        self.root_dir = root_dir
        self.transform = transform
        print("Created dataset {}.".format(name))

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image = Image.open(row.imgPath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return np.array(image), torch.tensor(row.label)

    def _check_csv_cols(self, cols: list):
        """
        Checks if cols are present in th csv
        """
        expected_cols_present = set(cols) <= set(self.labels_df.columns)
        if not expected_cols_present:
            raise ValueError("Missing mendatory columns in label CSV file.")


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
        root_dir.name
        + "/"
        + df.slideName
        + "/"
        + patches_subfolder
        + "/"
        + df.patchName
        + ".jpg"
    )
    return df[["imgPath", "label"]]
