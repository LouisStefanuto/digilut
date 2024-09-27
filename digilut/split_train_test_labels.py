from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

app_split = typer.Typer(help="Split the labels.csv dataset in two patient cohorts.")


@app_split.command()
def split_train_test(
    csv_labels: Annotated[Path, typer.Argument(help="CSV input")] = Path(
        "patches/labels.csv"
    ),
    output_dir: Annotated[Path, typer.Argument(help="Folder output")] = Path("patches"),
    test_size: float = 0.2,
    seed: int = 1234,
):
    df = pd.read_csv(csv_labels, index_col=0)

    patient_ids = list(set(df.patientID))
    patient_id_train, patient_id_test = train_test_split(
        patient_ids, test_size=test_size, random_state=seed
    )

    df_train = df[df.patientID.isin(patient_id_train)]
    df_test = df[df.patientID.isin(patient_id_test)]

    df_train.to_csv(output_dir / (csv_labels.stem + "_train.csv"))
    print("Saved", output_dir / (csv_labels.stem + "_train.csv"))
    df_test.to_csv(output_dir / (csv_labels.stem + "_test.csv"))
    print("Saved", output_dir / (csv_labels.stem + "_test.csv"))


if __name__ == "__main__":
    app_split()
