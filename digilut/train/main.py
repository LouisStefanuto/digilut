import datetime
import time
from pathlib import Path

import pandas as pd
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from digilut.train.data_aug import get_transform
from digilut.train.dataset import MyDataset, prepare_dataset
from digilut.train.model import ClassificationModel


def train(
    csv_file_train: Path = Path("processed/labels_train_undersampled_train.csv"),
    csv_file_test: Path = Path("processed/labels_train_undersampled_test.csv"),
):
    # Training parameters
    root_dir = Path("patches")
    batch_size = 64
    num_workers = 4
    num_classes = 1
    max_epochs = 50
    lr = 5e-4
    accelerator = "mps"  # Change to "cuda" or "cpu" as needed

    # ------------------

    start_time = time.time()

    labels_df_train = pd.read_csv(csv_file_train)
    labels_df_train = prepare_dataset(labels_df_train, root_dir)
    labels_df_val = pd.read_csv(csv_file_test)
    labels_df_val = prepare_dataset(labels_df_val, root_dir)

    train_dataset = MyDataset(
        labels_df=labels_df_train,
        root_dir=root_dir,
        transform=get_transform(train=True),
        name="train",
    )
    val_dataset = MyDataset(
        labels_df=labels_df_val,
        root_dir=root_dir,
        transform=get_transform(train=False),
        name="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = ClassificationModel(num_classes=num_classes, lr=lr)

    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        accelerator=accelerator,
        callbacks=None,
    )
    trainer.fit(model, train_loader, val_loader)

    execution_time = int(time.time() - start_time)
    execution_time = datetime.timedelta(seconds=execution_time)  # type:ignore
    print("Execution time: {} seconds".format(str(execution_time)))


if __name__ == "__main__":
    train()
