# Digilut Data Challenge (2024)

## 💪 Motivation

The [**Digilut challenge**](https://app.trustii.io/datasets/1526), a collaborative project between Trustii.io and Foch Hospital and supported by the Health Data Hub and Bpifrance, aims to develop a medical decision support algorithm to diagnose graft rejection episodes in lung transplant patients by **detecting pathological regions** (A lesions) on transbronchial biopsies.

![slide](docs/assets/slide.png)

## 🧪 Quick installation

Install poetry:

    brew install poetry

Clone the project:

    git clone https://github.com/LouisStefanuto/digilut.git

Create a virtual environment and install deps:

    cd digilut
    poetry install --with dev,docs

For a more detailed install, refer to the [install section](https://louisstefanuto.github.io/digilut/install) of the documentation.

## 📊 Download the dataset

The datasets are available on the [competition webpage](https://app.trustii.io/datasets/1526). Note that you have to create a Trustii account first.

- 412 **annotated slides** (~200 GB): *needed*
- 1680 **non-annotated slides** (~3TB): *not used in this project*

## 📚 Documentation

For more info about the project, visit the [open documentation](https://louisstefanuto.github.io/digilut/).

You can also host it locally by running:

    mkdocs serve

## 👋 Contributing

If you encounter any issues with the repo or have suggestions for improvements, feel free to open an issue or submit a pull request.
