# New to Typer?

This package leverages [**Typer**](https://typer.tiangolo.com) to create a Command Line Interface (CLI). Typer is a library that simplifies the creation of CLI applications by using Python 3.6+ type hints. It provides an intuitive and powerful way to define commands, arguments, and options. It is a more concise alternative to `argparse`.

The CLI is the recommended way to interact with the main python scripts of the repo.

---

## Installation

The CLI is automatically installed when you install the project with `poetry`:

    poetry install

## Use the CLI

Using the CLI is straightforward. You can either use the alias `digilut` (recommended and fancy) or you can directly call the main python script:

    digilut --help
    # OR
    python digilut/main.py --help

A help panel appears and walks you through the commands of the CLI.

Each command is a wrapper around a function in a Python script.

## CLI Configuration

The CLI's configuration (alias) is defined in the `pyproject.toml` file, under the `[tool.poetry.scripts]` section:

    [tool.poetry.scripts]
    digilut = "digilut.main:app"

> That creates a `digilut` CLI, that links to the Typer `app` in `digilut.main.py`.

## Typer autodoc

Typer comes with auto documentation tools. Typer can parse the commands docstrings and type hints and convert them into Markdown. I wrapped the command line into a shell script for convenience.

    scripts/autodoc_typer.sh

This will generate the `doc/cli/commands.md` file.

Run this command everytime you update/create/remove a command.
