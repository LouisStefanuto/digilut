# Trustii

!!! note ""
    During the competition, the organizers gave access to a JupyterHub instance. Here are the steps to run the project on the platform. The commands are different because it was a Debian VM.

To run on the Trustii platform, follow these steps:

1. Open a terminal

2. Clone the repo

        git clone ...
        cd digilut

3. Create a Conda env to force the python version

    ```console
    conda create -n env-digitut python=3.12
    conda init
    source ~/.bashrc
    conda activate env_digilut
    ```

4. Install poetry

    ```console
    pip install poetry -q
    ```

5. Install FAST deps:

    Install deps (enter yes if asked):

    ```console
    sudo apt install libgl1 libopengl0 libopenslide0 libusb-1.0-0 libxcb-xinerama0
    ```

    If errors encountered, run the following commands (enter yes if asked), then try again the previous command.

    ```console
    sudo apt update
    sudo apt --fix-broken install
    sudo apt install libpocl2
    ```

    The last dependency of FAST is OpenCL. It depends on your CPU. Check your CPU specs:

    ```console
    cat /proc/cpuinfo
    ```

    Then download the relevant driver. [More doc here](https://fast.eriksmistad.no/install-ubuntu-linux.html) (for me it was Intel on Trustii).

    ```console
    # Intel driver opencl. Other option is to go manual: https://github.com/intel/compute-runtime/releases (not sure if sudo apt-get install intel-gmmlib is needed, I ran it)
    sudo apt-get install intel-opencl-icd
    ```

6. Finally, install package deps

    ```console
    poetry install --with dev,docs
    ```

7. Test your installation:

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
