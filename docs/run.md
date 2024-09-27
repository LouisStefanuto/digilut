# Run

!!! note ""
    Here is the workflow to prepare the datasets, train models, and infer on the test dataset.

Clean

    digilut clean-bbox data/train.csv data/train_cleaned.csv --train
    digilut clean-bbox data/validation.csv data/validation_cleaned.csv --no-train

Embed

    python digilut/embeddings.py dataset-v2 ../patches_train_small_level1/ ../patches_train_small_level1/labels_undersampled.csv cpu --output-folder ../embeddings_train

Undersample

    ...

Train

    ...

Infer

    ...

Eval

    ...
