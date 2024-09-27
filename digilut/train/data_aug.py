from torchvision import transforms
from torchvision.transforms import v2


def get_transform_v1():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_transform_v2(train: bool):
    if train:
        transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = v2.Compose(
            [
                transforms.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transform


def get_transform(train: bool, version: str = "v2"):
    if version == "v1":
        return get_transform_v1()
    elif version == "v2":
        return get_transform_v2(train)
    else:
        raise ValueError("Invalid transform version")
