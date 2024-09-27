from pathlib import Path

import pandas as pd
import typer
from imblearn.under_sampling import RandomUnderSampler
from typing_extensions import Annotated

app_undersample = typer.Typer(help="Undersample patches to solve class imbalance")


def print_proportions(patches: pd.DataFrame, slide_name: str, prefix: str) -> None:
    nb_positives, nb_total = patches.label.sum(), len(patches)

    print(
        "{} - {} undersampling: {} positive VS {} negative samples".format(
            slide_name, prefix, nb_positives, nb_total - nb_positives
        )
    )


@app_undersample.command()
def run(
    csv_patches: Annotated[Path, typer.Argument(help="Path to unbalanced CSV file.")],
    output_balanced_patches: Annotated[
        Path, typer.Argument(help="Path to balanced CSV file.")
    ],
    ratio: Annotated[
        float,
        typer.Option(
            help="Desired ratio of the number of samples in the minority class over the number of samples in the majority class after resampling."
        ),
    ] = None,  # type:ignore
    random_seed: Annotated[int, typer.Option(help="Random seed.")] = 1234,
) -> None:
    """
    Takes as input the set of possible patches and returns a subset of them that will
    be used for building the training dataset.

    For each slide folder, checks the patches metadata.csv
    Keep N positive and N negative patches.
    """

    # List slides with an info .csv file
    all_patches = pd.read_csv(csv_patches, index_col=0)

    # Get unique slide names
    slide_names = set(all_patches.slideName)

    # Init the list
    balanced_patches = []

    strat = ratio if ratio else "auto"  # type: ignore
    random_under_sampler = RandomUnderSampler(
        sampling_strategy=strat, random_state=random_seed
    )

    # For each slide
    for slide_name in slide_names:
        # Get the rows matching
        patches = all_patches[all_patches.slideName == slide_name]
        print_proportions(patches, slide_name, "Before")

        if patches.label.sum() == 0:
            print("Found no positive patches for slide {}".format(slide_name))
        else:
            # Subsample the negative patches
            patches, _ = random_under_sampler.fit_resample(X=patches, y=patches.label)
            balanced_patches.append(patches)
            print_proportions(patches, slide_name, "After")

    # Export the list into a new labels, shorter .csv file
    pd.concat(balanced_patches, axis=0).to_csv(output_balanced_patches)


if __name__ == "__main__":
    app_undersample()
