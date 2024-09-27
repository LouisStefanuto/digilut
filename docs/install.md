# Install

!!! note ""
    This tutorial is for MacOS users. For Windows/Linux, take a look at this [**section**](./trustii.md).

1. Install [Poetry](https://python-poetry.org/docs/) and Python 3.12

        brew install poetry
        brew install python@3.12

2. Clone the repo

        git clone ...
        cd digilut

3. The project makes use of PyFAST for slide processing. Follow this [tutorial](https://fast.eriksmistad.no/install.html). For MacOS users, here is the command to run

        brew install openslide libomp

4. Create a virtual environment

        poetry install --with docs,dev

5. Test your installation

    ```console
    source $(poetry env info --path)/bin/activate 
    digilut --help
    ```

    You should be prompted something like:

    ```console
    Usage: digilut [OPTIONS] COMMAND [ARGS]...                                                                                                        
                                                                                                                                                    
    Entrypoint of Digilut's main CLI.                                                                                                                 
                                                                                                                                                    
    ╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --install-completion          Install completion for the current shell.                                                                         │
    │ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                  │
    │ --help                        Show this message and exit.                                                                                       │
    ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Commands ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ clean-bbox    Remove obvious labelling mistakes from the input bounding box csv and save a new cleaned dataset csv.                             │
    │ credits       Print credits with style.                                                                                                         │
    │ pyfast        Pyfast processing commands to patchify .tif WSI images.                                                                           │
    │ tiles         Commands for tiles.                                                                                                               │
    │ undersample   Undersample patches to solve class imbalance                                                                                      │
    ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

    ```
