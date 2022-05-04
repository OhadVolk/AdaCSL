import numpy as np
from torch import optim

from constants import TRAIN, TEST, VAL
from experiment.data import get_dataloaders
from experiment.utils import get_device, get_resnet, get_cost_matrix
from src.adacsl import adaptive_cross_entropy
from src.adacsl_engine import run_adacsl_engine


def main():
    data_dir = "../input/breast-cancerreduced/BreastCancer/"  #
    device = get_device()
    dataloaders = get_dataloaders(data_dir)
    rho = 1
    cost_matrix = get_cost_matrix(dataloaders, rho)
    resnet = get_resnet(device)
    criterion = adaptive_cross_entropy

    lr = 0.001
    weight_decay = 0.01
    optimizer = optim.SGD(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    cost_tag_matrix = cost_matrix.copy()
    edges = np.arange(0.0, 1.0, 0.1)
    num_epochs = 30

    train_metrics_total, val_metrics_total = run_adacsl_engine(resnet,
                                                               tr_dataloader=dataloaders[TRAIN],
                                                               val_dataloader=dataloaders[VAL],
                                                               test_dataloader=dataloaders[TEST],
                                                               optimizer=optimizer,
                                                               criterion=criterion,
                                                               num_epochs=num_epochs,
                                                               cost_matrix=cost_matrix,
                                                               cost_tag_matrix=cost_tag_matrix,
                                                               edges=edges,
                                                               device=device)


if __name__ == "__main__":
    main()
