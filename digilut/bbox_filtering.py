import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def detect_neg_coords(df: pd.DataFrame) -> pd.DataFrame:
    # Neg values
    df["x1_neg"] = df.x1 < 0
    df["x2_neg"] = df.x2 < 0
    df["y1_neg"] = df.y1 < 0
    df["y2_neg"] = df.y2 < 0
    return df


def detect_outside_coords(df: pd.DataFrame) -> pd.DataFrame:
    # Outside of range
    df["x1_out"] = df.x1 > df.max_x
    df["x2_out"] = df.x2 > df.max_x
    df["y1_out"] = df.y1 > df.max_y
    df["y2_out"] = df.y2 > df.max_y
    return df


def detect_weird_shapes(df: pd.DataFrame, ratio: float = 10) -> pd.DataFrame:
    # Outside of range
    df["height"] = df.x2 - df.x1
    df["width"] = df.y2 - df.y1
    df["ratio"] = df["height"] / df["width"]
    df["weird_ratio"] = (df.ratio > 100) | (df.ratio < 1 / 100.0)
    return df


def split_bbox_name(bbox_name: str) -> tuple:
    bbox_name = Path(bbox_name).stem

    stem, suffix, x1, y1, x2, y2 = bbox_name.split("_")
    slide_name = stem + "_" + suffix + ".tif"

    return slide_name, int(x1), int(y1), int(x2), int(y2)


def remove_bbox_organizers(df: pd.DataFrame) -> pd.DataFrame:
    print("Length before remove:", len(df))

    BBOXES_TO_REMOVE = "data/Boundig Box IDs"

    bboxes = os.listdir(BBOXES_TO_REMOVE)
    print("Nb boxes to remove:", len(bboxes))

    for bbox in bboxes:
        slide_name, x1, y1, x2, y2 = split_bbox_name(bbox)

        matching_rows = df[
            (df.filename == slide_name)
            & (df.x1 == x1)
            & (df.x2 == x2)
            & (df.y1 == y1)
            & (df.y2 == y2)
        ]

        if len(matching_rows) != 1:
            raise ValueError(f"It should be only one match. Found {len(matching_rows)}")

        df = df.drop(matching_rows.index)

    print("Length after remove:", len(df))

    return df


def add_patient_info(df: pd.DataFrame) -> pd.DataFrame:
    df["slideName"] = df["filename"].str.rsplit(".", n=1).str[0]
    df["slideID"] = pd.Categorical(df["slideName"]).codes

    df["patientName"] = df["filename"].str.rsplit("_", n=1).str[0]
    df["patientID"] = pd.Categorical(df["patientName"]).codes

    return df


def clean(
    csv_bboxes: Path, csv_output: Path, show_plots: bool = False, train: bool = True
) -> None:
    """
    Remove obvious labelling mistakes from the input bounding box csv and save a new
    cleaned dataset csv.

    Args:
        csv_bboxes (Path): bounding box file (csv) to clean
        csv_output (Path): cleaned dataset name
        show_plots (bool, optional): show plots. Defaults to False.
    """
    # Open bboxes csv file
    df = pd.read_csv(csv_bboxes)
    if not train:
        # Add fake cols for x1 x2 y1 y2
        df["x1"] = 100
        df["x2"] = 200
        df["y1"] = 100
        df["y2"] = 200

    print(df.head())
    print("-------------------")

    # Boxes outside of the image
    df = add_patient_info(df)
    df = detect_neg_coords(df)
    df = detect_outside_coords(df)
    df = detect_weird_shapes(df)
    if train:
        df = remove_bbox_organizers(df)

    # Summary
    df["anomaly"] = (
        df.x1_neg
        + df.x2_neg
        + df.y1_neg
        + df.y2_neg
        + df.x1_out
        + df.x2_out
        + df.y1_out
        + df.y2_out
    )

    print(df.describe().T)
    print("-------------------")

    # Print anomalies
    print("Anomaly boxes")
    anormal_rows = df[df.anomaly == 1]
    print(f"Dropping {len(anormal_rows)} row(s)")
    df = df.drop(anormal_rows.index)

    print("-------------------")
    output_columns = [
        "filename",
        "slideName",
        "slideID",
        "patientName",
        "patientID",
        "x1",
        "x2",
        "y1",
        "y2",
        "max_x",
        "max_y",
    ]
    df_cleaned = df[output_columns]

    if not train:
        # Replace fake values by explicit unkonwon values
        df_cleaned.loc[:, "x1"] = -1
        df_cleaned.loc[:, "x2"] = -1
        df_cleaned.loc[:, "y1"] = -1
        df_cleaned.loc[:, "y2"] = -1

    df_cleaned.to_csv(csv_output)
    print(f"Cleaned dataset. Shape: {df_cleaned.shape}")
    print(df_cleaned.head())

    if show_plots:
        plt.figure()
        plt.scatter(df.width, df.height)
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.show()

        plt.figure()
        df.plot.hist(subplots=True)
        plt.show()
