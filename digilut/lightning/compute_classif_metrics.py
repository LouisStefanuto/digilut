from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import typer
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
)

from digilut.lightning.config import load_config
from digilut.logs import get_logger

app = typer.Typer()
logger = get_logger(__name__)


@app.command()
def compute_metrics(config_path: Path = Path("config.yaml")):
    """Evaluate the model based on predictions saved in a CSV file.

    Compute metrics and generate plots, logged via MLflow."""

    # Load the configuration
    config = load_config(config_path)

    # Convert Pydantic model to dict for logging
    config_dict = config.model_dump()

    # Start MLflow run
    mlflow.start_run(run_name="eval_test")
    mlflow.set_tag("mlflow.note.content", "Model evaluation: metrics and ROC curve.")

    # Load predictions and calculate metrics
    df = pd.read_csv(config.predictions_path)
    df["y_pred"] = (df.prediction > config.threshold_classif).astype(int)

    # Generate classification report
    classif_report = classification_report(
        y_true=df.labels, y_pred=df.y_pred, output_dict=True
    )

    # Compute classification metrics (precision, recall, f1-score, etc.)
    precision, recall, f1, _ = precision_recall_fscore_support(
        df.labels, df.y_pred, average="binary"
    )

    # Define ROC curve metrics
    fpr, tpr, _ = roc_curve(y_true=df.labels, y_score=df.prediction)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

    # Log inputs and outputs to MLFLOW
    NB_DECIMALS = 4
    mlflow.log_params(config_dict)
    mlflow.log_metric("precision", round(precision, NB_DECIMALS))
    mlflow.log_metric("recall", round(recall, NB_DECIMALS))
    mlflow.log_metric("f1_score", round(f1, NB_DECIMALS))
    mlflow.log_metric("n_positive", classif_report["1"]["support"])
    mlflow.log_metric("n_negative", classif_report["0"]["support"])

    # End the MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    app()
