from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import pytorch_lightning as pl
import typer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold

from digilut.lightning.config import load_config
from digilut.lightning.dataset import create_dataloaders
from digilut.lightning.model import PatchClassificationModel
from digilut.lightning.utils import compute_loss_weights, prepare_dataset

app = typer.Typer()


@app.command()
def main(config_path: Path) -> None:
    config = load_config(config_path)

    # Define experiment name
    mlflow.set_experiment(config.experiment_name)

    # Load the data
    df = pd.read_csv(config.train_data_path, index_col=0)
    df = prepare_dataset(df, Path(config.train_image_dir))

    # Cross validation
    sgkf = StratifiedGroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    X = df[["imgPath", "label"]]
    y = df["label"]
    groups = df["patientID"]

    for fold, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
        print(f"Fold {fold}:")

        # Create dataloaders
        train_df, val_df = df.iloc[train_index], df.iloc[test_index]
        train_loader, val_loader = create_dataloaders(train_df, val_df, config)

        # Compute class balance factor
        class_weights = compute_loss_weights(train_df["label"])

        # Initialize the Lightning model
        model = PatchClassificationModel(
            class_weights=class_weights,
            learning_rate=config.learning_rate,
            dropout=config.dropout,
        )

        # Initialize a trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator=config.device,
            callbacks=[
                ModelCheckpoint(
                    dirpath="checkpoints", monitor="val_loss", mode="min", save_top_k=1
                ),
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=config.early_stopping_patience,
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
            logger=TensorBoardLogger(config.log_dir, name=config.name_model),
        )

        # Start a new MLflow run
        mlflow.start_run(run_name=f"run_{trainer.logger.version}_fold_{fold+1}")

        mlflow.log_params(config.model_dump())
        mlflow.log_param("n_train", len(train_loader.dataset))
        mlflow.log_param("n_val", len(val_loader.dataset))
        mlflow.log_param("class_weights", float(class_weights[0]))

        trainer.fit(model, train_loader, val_loader)

        mlflow.end_run()


if __name__ == "__main__":
    app()
