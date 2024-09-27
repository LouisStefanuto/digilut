# About

## Tools used in the project

- Data pre-processing and analysis
    - [PyFAST](https://github.com/smistad/FAST): a library for TIFF image
    - [Openslide](https://openslide.org): a standard lib to open and plot WSI images

- Model Training
    - [Pytorch](https://pytorch.org/docs/stable/index.html) and [Lightning](https://lightning.ai/docs/pytorch/stable/) for training
    - [MLflow](https://mlflow.org/docs/latest/index.html) for experiment monitoring and model registry
    - [Tensorboard](https://www.tensorflow.org/tensorboard?hl=fr) for in-run loss and metrics monitoring

- Documentation
    - [Mkdocs](https://www.mkdocs.org): to generate static documentation websites from markdown files
    - [Mkdocs-Material](https://squidfunk.github.io/mkdocs-material/): a theme for Mkdocs

- Miscellaneous
    - [Typer](https://typer.tiangolo.com): for a fancy CLI
    - [Pydantic](https://docs.pydantic.dev/latest/): for config validation
    - [Github Actions](https://docs.github.com/en/actions): for CI/CD
    - [Pytest](https://docs.pytest.org/en/stable/): for testing some parts of the project
    - [Precommit](https://pre-commit.com): for quality checks before committing to the repo
