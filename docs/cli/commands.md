# `digilut`

Entrypoint of Digilut's main CLI.

**Usage**:

```console
$ digilut [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `clean-bbox`: Remove obvious labelling mistakes from the...
* `credits`: Print credits with style.
* `pyfast`: Pyfast processing commands to patchify...
* `tiles`: Commands for tiles.
* `undersample`: Undersample patches to solve class imbalance

## `digilut clean-bbox`

Remove obvious labelling mistakes from the input bounding box csv and save a new
cleaned dataset csv.

**Usage**:

```console
$ digilut clean-bbox [OPTIONS] CSV_BBOXES CSV_OUTPUT
```

**Arguments**:

* `CSV_BBOXES`: Bounding box csv to clean   [required]
* `CSV_OUTPUT`: Name of the cleaned csv  [required]

**Options**:

* `--show-plots / --no-show-plots`: Plot figures  [default: no-show-plots]
* `--train / --no-train`: In train mode run some check on rows, disabled for validation.  [default: train]
* `--help`: Show this message and exit.

## `digilut credits`

Print credits with style.

**Usage**:

```console
$ digilut credits [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `digilut pyfast`

Pyfast processing commands to patchify .tif WSI images.

**Usage**:

```console
$ digilut pyfast [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `labels`: Create labels for the tiles.
* `patchify-dataset`: Extracts tiles from a dataset of slides.
* `patchify-slide`: Exports the TIFF file into PNG tiles of...

### `digilut pyfast labels`

Create labels for the tiles.

A tile is positive if it intersects at more that XX% with a bounding box.

**Usage**:

```console
$ digilut pyfast labels [OPTIONS] CSV_BBOXES FOLDER_PATCHES CSV_LABELS
```

**Arguments**:

* `CSV_BBOXES`: [required]
* `FOLDER_PATCHES`: [required]
* `CSV_LABELS`: [required]

**Options**:

* `--threshold-iou FLOAT`: [default: 0.1]
* `--help`: Show this message and exit.

### `digilut pyfast patchify-dataset`

Extracts tiles from a dataset of slides. Calls patchify-slide over each slide in the folder.

**Usage**:

```console
$ digilut pyfast patchify-dataset [OPTIONS] [CSV_PATH] [IMAGES_DIR] [OUTPUT_DIR]
```

**Arguments**:

* `[CSV_PATH]`: Path to the CSV file  [default: data/train.csv]
* `[IMAGES_DIR]`: Folder containing the .tif slide images  [default: data/images]
* `[OUTPUT_DIR]`: Output dir. Each slide will have a {outputdir}/{slide}  [default: outputs]

**Options**:

* `--save-engine TEXT`: Engine to save image. 'cv2' (recommended) OR 'pillow'  [default: cv2]
* `--patch-size INTEGER`: Patch size (width and hieght)  [default: 256]
* `--level INTEGER`: Zoom level. 0 is the best resolution  [default: 0]
* `--overlap-percent FLOAT`: Percentage of overlap between patches  [default: 0.0]
* `-f, --img_format TEXT`: Image format. PNG is better (no artifact) but it x5 heavier. Values: 'jpg', 'png'  [default: jpg]
* `--help`: Show this message and exit.

### `digilut pyfast patchify-slide`

Exports the TIFF file into PNG tiles of tissue.

**Usage**:

```console
$ digilut pyfast patchify-slide [OPTIONS] TIFF_PATH OUTPUT_DIR
```

**Arguments**:

* `TIFF_PATH`: Path to the slide  [required]
* `OUTPUT_DIR`: Output folder where patches will be saved  [required]

**Options**:

* `--save-engine TEXT`: Engine to save image. 'cv2' (recommended) OR 'pillow'  [default: cv2]
* `--patch-size INTEGER`: Patch size (width and hieght)  [default: 256]
* `--level INTEGER`: Zoom level. 0 is the best resolution  [default: 0]
* `--overlap-percent FLOAT`: Percentage of overlap between patches  [default: 0.0]
* `--img-format TEXT`: Image format. PNG is better (no artifact) but it x5 heavier. Values: 'jpg', 'png'  [default: jpg]
* `--help`: Show this message and exit.

## `digilut tiles`

Commands for tiles.

**Usage**:

```console
$ digilut tiles [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `extract-from-dataset`: Extract tiles a dataset of tiles.
* `extract-from-image`: Extract tiles from the slide.
* `generate-labels`: (Deprecated) Create the labels for the tiles.

### `digilut tiles extract-from-dataset`

Extract tiles a dataset of tiles. Calls extract_from_image over a folder of slides.

**Usage**:

```console
$ digilut tiles extract-from-dataset [OPTIONS] [CSV_PATH] [OUTPUT_DIR]
```

**Arguments**:

* `[CSV_PATH]`: Path to the CSV file  [default: data/train.csv]
* `[OUTPUT_DIR]`: Outputdir  [default: outputs]

**Options**:

* `--tile-size INTEGER`: Size of the tiles (height and width)  [default: 1024]
* `--parallel / --no-parallel`: Enable multiprocessing.  [default: parallel]
* `--help`: Show this message and exit.

### `digilut tiles extract-from-image`

Extract tiles from the slide.

**Usage**:

```console
$ digilut tiles extract-from-image [OPTIONS] PATH_TIFF OUTPUT_DIR
```

**Arguments**:

* `PATH_TIFF`: Path to the TIFF file  [required]
* `OUTPUT_DIR`: Output folder. The output will we saved in {outputdir}/{path_tiff.stem}  [required]

**Options**:

* `--tile-size INTEGER`: Size of the tiles (height and width)  [default: 1024]
* `-p, --no-parallel`: Disable multiprocessing.  [default: True]
* `--help`: Show this message and exit.

### `digilut tiles generate-labels`

Create the labels for the tiles. V1 Deprecated. Use digilut pyfast

**Usage**:

```console
$ digilut tiles generate-labels [OPTIONS] CSV_BBOXES SLIDE_FOLDER
```

**Arguments**:

* `CSV_BBOXES`: Bounding box file.  [required]
* `SLIDE_FOLDER`: Slide folder, that contains an `info` and a `tiles` subfolder.  [required]

**Options**:

* `--iou-thres FLOAT`: Threshold Intersection over Union. If tile IOU > with a bounding box, the tile is labbeled positive.  [default: 0.2]
* `--help`: Show this message and exit.

## `digilut undersample`

Undersample patches to solve class imbalance

**Usage**:

```console
$ digilut undersample [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `run`: Takes as input the set of possible patches...

### `digilut undersample run`

Takes as input the set of possible patches and returns a subset of them that will
be used for building the training dataset.

For each slide folder, checks the patches metadata.csv
Keep N positive and N negative patches.

**Usage**:

```console
$ digilut undersample run [OPTIONS] CSV_PATCHES OUTPUT_BALANCED_PATCHES
```

**Arguments**:

* `CSV_PATCHES`: [required]
* `OUTPUT_BALANCED_PATCHES`: [required]

**Options**:

* `--sampling-strategy FLOAT`
* `--random-seed INTEGER`: [default: 1234]
* `--help`: Show this message and exit.
