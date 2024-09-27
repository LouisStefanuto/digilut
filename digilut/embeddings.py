import datetime
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import timm
import torch
import typer
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from tqdm import tqdm

from digilut.logs import get_logger

app = typer.Typer()

logger = get_logger(__name__)


@dataclass
class EmbeddedPatch:
    name: str
    slide_id: int
    patient_id: int
    embedding: np.array

    @classmethod
    def from_npy(cls, path: Path) -> "EmbeddedPatch":
        embedded_dict: dict = np.load(path, allow_pickle=True).item()
        return EmbeddedPatch(**embedded_dict)


class Device(Enum):
    MPS = "mps"
    CPU = "cpu"
    GPU = "gpu"


def load_model() -> VisionTransformer:
    # Load model from the hub
    model = timm.create_model(
        model_name="hf-hub:1aurent/vit_base_patch16_224.owkin_pancancer",
        pretrained=True,
    )
    return model


def get_transforms(model):
    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms


def embed_slide(
    patches_folder: Path,
    output_folder: Path,
    model: VisionTransformer,
    transforms: Callable,
    device: Device,
) -> None:
    """
    Embed all the patches of a slide and save the embeddings in a folder as .npy files.


    Args:
        patches_folder (Path): path to the patch folder, it should contain the JPN/PNG images
        output_folder (Path): output folder for the .npy embeddings
        model (timm.models.vision_transformer.VisionTransformer): embedding model instance
        transforms (Callable): model processing pipeline instance
        device (Device, optional): acceleration device. cpu/gpu/mps.
    """

    # Ensure output_folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each image in the folder
    imgs = list(patches_folder.glob("*.jpg"))

    for image_path in tqdm(imgs, total=len(imgs)):
        try:
            # Load image
            img = Image.open(image_path).convert("RGB")

            # Apply transformations
            data = transforms(img).unsqueeze(0).to(device.value)

            # Generate embeddings
            with torch.no_grad():
                output = model(data)

            # Store embeddings
            embedding = output.squeeze(0).cpu().numpy()
            output_file = output_folder / (image_path.stem + ".npy")

            np.save(output_file, embedding)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    # Save embeddings to file
    print(f"Embeddings saved to {output_folder}")


@app.command(name="dataset")
def embed_dataset(
    dataset_folder: Path,
    device: Device,
    embed_sub_folder: str = "embeddings",
) -> None:
    """
    Embed dataset.
    Iterate over all slide folders and run the embed command for each.

    The embeddings are saved in {dataset_folder}/{slide_folder}/{embed_sub_folder}

    Args:
        dataset_folder (Path): dataset folders, that contains one slide subfolder per slide in the dataset
        embed_sub_folder (str, optional): subfolder created for the embeddings in the slide folder. Defaults to "embeddings".
        device (Device, optional): acceleration device. cpu/gpu/mps
    """
    # Get model
    model = load_model()
    model.eval()
    model.to(device.value)
    # Get model preprocessing transformations
    transforms = get_transforms(model)

    slide_folders = [entry for entry in dataset_folder.iterdir() if entry.is_dir()]
    logger.info("Found {} slides folders to embed".format(len(slide_folders)))

    for slide_folder in slide_folders:
        embed_slide(
            patches_folder=slide_folder / "patches",
            output_folder=slide_folder / embed_sub_folder,
            model=model,
            transforms=transforms,
            device=device,
        )


@app.command(name="dataset-v2")
def embed_dataset_v2(
    patch_folder: Path,
    csv_dataset: Path,
    device: Device,
    output_folder: Path = Path("embeddings"),
) -> None:
    """
    Embed dataset.
    Iterate over all slide folders and run the embed command for each. Version 2.
    """
    # Create output dir
    output_folder.mkdir(exist_ok=True, parents=True)

    start_time = time.time()

    # Load dataset
    patches = pd.read_csv(csv_dataset, index_col=0)

    # Add volumn path to jpg/png name
    patch_name = (
        patches["slideName"] + "/" + "patches" + "/" + (patches["patchName"] + ".jpg")
    )
    patches["patchPath"] = patch_folder / patch_name

    # Add column npy name
    patches["embeddingPath"] = output_folder / (
        patches["slideName"] + "-" + patches["patchName"]
    )

    logger.info(patches)
    logger.info("Label file ready.")
    logger.info("Loading model ...")

    # Get model
    model = load_model()
    model.eval()
    model.to(device.value)
    # Get model preprocessing transformations
    transforms = get_transforms(model)
    logger.info("Model loaded.")

    # For each row in dataset
    for _, row in tqdm(patches.iterrows(), total=len(patches)):
        # Unpack patch fields
        patch_path: Path = row["patchPath"]

        # Load image
        img = Image.open(patch_path).convert("RGB")

        # Apply transformations
        data = transforms(img).unsqueeze(0).to(device.value)

        # Generate embeddings
        with torch.no_grad():
            output = model(data)

        # Store embeddings as a dict of arr + metadata
        embedding: np.array = output.squeeze(0).cpu().numpy()

        embedded_patch = EmbeddedPatch(
            name=patch_path.as_posix(),
            slide_id=row["slideID"],
            patient_id=row["patientID"],
            embedding=embedding,
        )
        np.save(row["embeddingPath"], asdict(embedded_patch))

    logger.info("Finished embedding.")

    execution_time = int(time.time() - start_time)
    execution_time = datetime.timedelta(seconds=execution_time)  # type: ignore
    logger.info("Execution time: {} seconds".format(str(execution_time)))


if __name__ == "__main__":
    app()
