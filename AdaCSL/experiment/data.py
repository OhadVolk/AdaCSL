import os
from typing import Dict

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from constants import TRAIN, TEST, VAL

def data_transforms(phase: str):
    if phase == TRAIN:
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if phase == VAL:
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if phase == TEST:
        transform = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    return transform


def get_dataloaders(data_dir: str) -> Dict[str, DataLoader]:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms(x))
                      for x in [TRAIN, VAL, TEST]}
    dataloaders = {TRAIN: DataLoader(image_datasets[TRAIN], batch_size=64, shuffle=True),
                   VAL: DataLoader(image_datasets[VAL], batch_size=256, shuffle=True),
                   TEST: DataLoader(image_datasets[TEST], batch_size=256, shuffle=True)}
    return dataloaders
