from datetime import datetime
from pathlib import Path

import joblib
import typer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from typing_extensions import Annotated

from digilut.dataset import NpyDataset, count_occ_labels

app = typer.Typer(help="Launch a training on a balanced dataset of labelled patches.")


@app.command()
def train(
    csv_labels: Annotated[Path, typer.Argument(help="Path to the labels")],
    folder_embeddings: Annotated[Path, typer.Argument(help="Path to the embeddings")],
    n_folds: Annotated[int, typer.Option(help="Nb of folds. Must >= nb patients")] = 5,
    models_folder: Annotated[Path, typer.Option(help="Folder for models")] = Path(
        "models"
    ),
):
    models_folder.mkdir(exist_ok=True, parents=True)

    npy_dataset = NpyDataset(csv_labels, folder_embeddings)

    # Check the shapes of X and y
    print(f"Shape of X: {npy_dataset.X.shape}")
    print(f"Shape of y: {npy_dataset.y.shape}")

    # Extract dataset info for splitting
    patch_indices = npy_dataset.df.index
    patch_slides = npy_dataset.df.slideName

    verbose = False
    n_repeats = 1
    val_metrics = []

    cv_start_time = datetime.now()

    for repeat in range(n_repeats):
        print(f"Running cross-validation #{repeat+1}")

        # GroupKFold for creating folds, to avoid contamination
        cv_skfold = GroupKFold(n_splits=n_folds)
        cv_splits = cv_skfold.split(X=patch_indices, groups=patch_slides)

        for i, (train_indices, val_indices) in enumerate(cv_splits):
            fold_start_time = datetime.now()
            print(f"Running cross-validation on split #{i+1}")

            # Create datasets
            X_train = npy_dataset.X[train_indices, :]
            y_train = npy_dataset.y[train_indices]
            X_val = npy_dataset.X[val_indices, :]
            y_val = npy_dataset.y[val_indices]

            print("Train: {}".format(X_train.shape))
            print("Val: {}".format(X_val.shape))
            print("Labels train: {}".format(count_occ_labels(y_train)))
            print("Labels val: {}".format(count_occ_labels(y_val)))

            model = MLPClassifier(
                random_state=42,
                verbose=verbose,
                hidden_layer_sizes=(64),
                learning_rate_init=1e-4,
                learning_rate="adaptive",
                alpha=1e-3,
                max_iter=100,
                shuffle=True,
                tol=1e-7,
            )

            model.fit(X_train, y_train)

            # Predictions on the validation set
            y_val_pred = model.predict(X_val)
            # y_val_proba = model.predict_proba(X_val)[:, 1]

            # Evaluate the model
            accuracy = accuracy_score(y_val, y_val_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_val_pred
            )
            f2 = fbeta_score(y_val, y_val_pred, beta=2)
            auc = roc_auc_score(y_val, y_val_pred)

            # Store metrics
            val_metrics.append(
                {
                    "fold": i,
                    "repeat": repeat,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "f2": f2,
                    "auc": auc,
                }
            )

            # Print metrics
            print("Validation Metrics:")
            print("- Accuracy: {:.4f}".format(accuracy))
            print("- Precision: {}".format(precision))
            print("- Recall: {}".format(recall))
            print("- F1 Score: {}".format(f1))
            print("- F2 Score: {}".format(f2))
            print("- ROC-AUC: {}".format(auc))

            # print(classification_report(y_val, y_val_pred))
            print("Confusion matrix:\n{}".format(confusion_matrix(y_val, y_val_pred)))

            # Save the model after training
            model_filename = f"{models_folder}/mlp_model_repeat{repeat+1}_fold{i+1}.pkl"
            joblib.dump(model, model_filename)
            print(f"Model saved to {model_filename}")

            # Append fold local metrics
            fold_end_time = datetime.now()
            fold_running_time = fold_end_time - fold_start_time
            print("\n--------- Finished in {} ---------\n".format(fold_running_time))

    cv_end_time = datetime.now()
    cv_total_time = cv_end_time - cv_start_time
    print("Total cross-validation time: {}".format(cv_total_time))


if __name__ == "__main__":
    app()
