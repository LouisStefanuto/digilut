import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.v2 import ColorJitter, GaussianBlur

from digilut.lightning.config import Config


# Custom dataset
class PatchDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row.imgPath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return row.imgPath, image, row.label


def create_dataloaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, config: Config
) -> tuple[DataLoader, DataLoader]:
    train_dataset = PatchDataset(train_df, transform=get_transform_train())
    val_dataset = PatchDataset(val_df, transform=get_transform_val())

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def get_transform_train():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # RandomRotation(45),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            GaussianBlur(kernel_size=(7, 7), sigma=(0.001, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_transform_val():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
