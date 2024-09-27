import matplotlib.pyplot as plt
import numpy as np
import tifffile


class TiffSlideMetadata:
    def __init__(self, file_path: str) -> None:
        self.path = file_path

    def print(self) -> None:
        """Prints the metadata of a TIFF slide.

        Args:
            file_path (str): the path to the TIFF slide file.
        """
        print(f"Metadata of slide: {self.path}")
        try:
            with tifffile.TiffFile(self.path) as tif:
                # Print basic information about the TIFF file
                print(f"Number of pages: {len(tif.pages)}")
                for page in tif.pages:
                    print("----------")
                    print(f"Page {page.index}:")
                    print(f"Shape: {page.shape}")
                    print(f"Data type: {page.dtype}")
                    print(f"Compression: {page.compression}")
        except Exception as e:
            print(f"Error reading metadata: {e}")


def plot_img(image: np.ndarray) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Shape: {image.shape}")
    # plt.axis("off")  # Hide axes
    plt.show()
