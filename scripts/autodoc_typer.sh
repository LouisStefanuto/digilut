#!/bin/bash

# Run this script to generate the documentation page of the CLI.
# This uses the built-in documentation feature of Typer.
# More details: https://typer.tiangolo.com/tutorial/package/#generate-docs

poetry run typer digilut.main utils docs --output docs/cli/commands.md --name digilut
