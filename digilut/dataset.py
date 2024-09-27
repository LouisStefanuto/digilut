from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from digilut.embeddings import EmbeddedPatch


class NpyDataset:
    def __init__(self, csv_labels: Path, folder_embeddings: Path) -> None:
        self.df = pd.read_csv(csv_labels, index_col=0)

        # Add path embeddings
        self.df["embeddingPath"] = (
            folder_embeddings.as_posix()
            + "/"
            + self.df["slideName"]
            + "-"
            + self.df["patchName"]
            + ".npy"
        )

        # Sanity check
        self.complete = are_all_npy_existing(self.df["embeddingPath"])

        # Create attributes
        self.X, self.y, fields = self._load_dataset()
        self.names, self.slide_ids, self.patient_ids = fields

    def _load_dataset(self) -> tuple[np.array, np.array, tuple[list, list, list]]:
        """
        Loads the npy files of the dataframe and return them as np.array
        """
        # Matrices
        embeddings: list[np.array] = []
        labels: list[np.array] = []
        # Metadata
        names: list[str] = []
        slide_ids: list[int] = []
        patient_ids: list[int] = []

        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            embedded_patch = EmbeddedPatch.from_npy(row.embeddingPath)

            embeddings.append(embedded_patch.embedding)
            labels.append(row.label)

            names.append(embedded_patch.name)
            slide_ids.append(embedded_patch.slide_id)
            patient_ids.append(embedded_patch.patient_id)

        X = np.array(embeddings)
        y = np.array(labels)

        # Now X and y are ready for training with scikit-learn models
        return X, y, (names, slide_ids, patient_ids)


def are_all_npy_existing(expected_files: pd.Series) -> bool:
    is_complete = True

    for p in expected_files:
        if not Path(p).exists():
            print("{} is missing".format(p))
            is_complete = False

    return is_complete


def count_occ_labels(y: np.array) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
