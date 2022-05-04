from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import TRAIN
from experiment.net import resnet18


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def get_resnet(device: torch.device) -> nn.Module:
    resnet = resnet18(pretrained=True)
    resnet.fc = nn.Linear(512, 2)
    resnet = resnet.to(device)
    return resnet


def get_cost_matrix(dataloaders: Dict[str, DataLoader], rho: Union[int, float]) -> pd.DataFrame:
    classes, counts = np.unique(dataloaders[TRAIN].dataset.targets, return_counts=True)

    class_inverse_priors = {c: 1 / (count / len(dataloaders[TRAIN].dataset.targets)) for c, count in
                            zip(list(classes), counts)}
    cost_matrix = pd.DataFrame(data=[[0, rho * class_inverse_priors[1]], [class_inverse_priors[0], 0]],
                               index=['predict negative (0)', 'predict positive (1)'],
                               columns=['actual negative (0)', 'actual positive (1)'])
    return cost_matrix
