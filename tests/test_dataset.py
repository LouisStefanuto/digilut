import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from digilut.lightning.dataset import PatchDataset, create_dataloaders


# Helper function to create a dummy dataframe
def create_dummy_dataframe():
    return pd.DataFrame(
        {"imgPath": ["dummy_path1.jpg", "dummy_path2.jpg"], "label": [0, 1]}
    )


# Mock an image
@pytest.fixture
def mock_image():
    mock_img = Image.new("RGB", (64, 64))  # Create a dummy RGB image
    return mock_img


# Test PatchDataset __getitem__
def test_patchdataset_getitem(monkeypatch, mock_image):
    # Mock the Image.open method
    def mock_open(filepath):
        return mock_image

    # Apply monkeypatch to replace Image.open with mock_open
    monkeypatch.setattr(Image, "open", mock_open)

    # Arrange
    dummy_df = create_dummy_dataframe()
    dataset = PatchDataset(dummy_df, transform=transforms.ToTensor())

    # Act
    img_path, image, label = dataset[0]

    # Assert
    assert img_path == "dummy_path1.jpg"
    assert image.shape == (
        3,
        64,
        64,
    )  # Tensor shape corresponds to the image dimensions (3 channels, 64x64)
    assert label == 0


# Test Dataloader creation
def test_create_dataloaders(monkeypatch, mock_image):
    # Mock the Image.open method
    def mock_open(filepath):
        return mock_image

    # Apply monkeypatch to replace Image.open with mock_open
    monkeypatch.setattr(Image, "open", mock_open)

    # Arrange
    dummy_train_df = create_dummy_dataframe()
    dummy_val_df = create_dummy_dataframe()

    # Mock configuration object
    class MockConfig:
        batch_size = 2

    mock_config = MockConfig()

    # Act
    train_loader, val_loader = create_dataloaders(
        dummy_train_df, dummy_val_df, mock_config
    )

    # Assert - Check the train_loader is a DataLoader instance
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Ensure data loaders have the correct batch size
    for batch in train_loader:
        assert len(batch[0]) == 2  # Batch size is 2
        break  # We only need to check the first batch

    for batch in val_loader:
        assert len(batch[0]) == 2  # Batch size is 2
        break  # We only need to check the first batch
