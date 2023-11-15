import os
from typing import Dict

import pandas as pd
import torch
from constants import TRAIN, TEST, VAL
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms


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


class TabularDataset(Dataset):

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.x = self.df.drop(columns=LABEL_COL)
        self.y = self.df[LABEL_COL]

    def __getitem__(self, idx: int) -> (torch.Tensor, int):
        batch = self.x.iloc[idx]
        batch_tensor = torch.Tensor(batch)
        label = self.y.iloc[idx]

        return batch_tensor, label

    def get_len(self):
        return self.__len__()

    def __len__(self) -> int:
        return len(self.df)
