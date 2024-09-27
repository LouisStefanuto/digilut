from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    batch_size: int = Field(description="Batch size for training.")
    checkpoint_path: Path = Field(description="Path to the checkpoint for inference.")
    device: str = Field(description="Device to use for training (e.g., 'cpu', 'cuda').")
    dropout: float = Field(ge=0, le=1, description="Dropout during training.")
    early_stopping_patience: int = Field(
        ge=1, description="Patience for early stopping."
    )
    experiment_name: str = Field(description="Name of the MLFlow experiment.")
    learning_rate: float = Field(ge=0, description="Learning rate for the optimizer.")
    log_dir: Path = Field(description="Directory for TensorBoard logs.")
    max_epochs: int = Field(ge=1, description="Maximum number of epochs for training.")
    n_splits: int = Field(ge=1, description="Number of splits for cross validation.")
    name_model: str = Field(description="Name of the model for logging.")
    predictions_path: Path = Field(description="Path to CSV for inference predictions.")
    seed: int = Field(description="Seed for random operations.")
    train_data_path: Path = Field(description="Path to the training data CSV file.")
    train_image_dir: Path = Field(
        description="Path to the directory containing training images."
    )
    test_data_path: Path = Field(description="Path to the test data CSV file.")
    test_image_dir: Path = Field(
        description="Path to the directory containing test images."
    )
    threshold_classif: float = Field(
        ge=0,
        le=1,
        description="Threshold above which a sample is classified as positive.",
    )


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
