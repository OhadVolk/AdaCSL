import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from typing import Callable, Tuple, Union

from AdaCSL.adacsl import update_cost_matrix, get_lambda_psi_from_buckets
from metrics import get_cost_from_tensors, get_adacsl_metrics


def train_adacsl(model: nn.Module,
                 dataloader: DataLoader,
                 optimizer: Optimizer,
                 criterion: Callable,
                 epoch: int,
                 num_epochs: int,
                 cost_matrix: np.ndarray,
                 cost_tag_matrix: np.ndarray,
                 device: torch.device) -> Tuple[float, float, float]:
    print("Epoch: {}/{}".format(epoch + 1, num_epochs))
    print("=" * 10)
    print('Cost Matrix: ', cost_tag_matrix)

    model.train()

    dataset_size: int = len(dataloader.dataset)

    running_loss = 0.0
    running_corrects = 0
    total_cost = 0.0

    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        probabilities = model(inputs)

        _, preds = torch.max(probabilities, 1)
        loss = criterion(probabilities=probabilities, y=labels, cost_matrix=cost_tag_matrix)
        cost = get_cost_from_tensors(y_true=labels, y_pred=preds, cost_matrix=cost_matrix)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_cost += cost

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    print('Train:  {:.4f}: , Accuracy: {:.4f}, Cost: {:.4f}'.format(epoch_loss, epoch_acc, total_cost))

    return epoch_loss, epoch_acc, total_cost


def evaluate_adacsl(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: Callable,
                    edges: np.ndarray,
                    cost_matrix: np.ndarray,
                    cost_tag_matrix: np.ndarray,
                    device: torch.device,
                    interval: Union[int, float] = 0.1,
                    bucket_interval: Union[int, float] = 0.005) -> Tuple[np.ndarray, float, float, float]:
    model.eval()

    dataset_size: int = len(dataloader.dataset)

    all_probabilities = torch.Tensor([])
    all_labels = torch.Tensor([])

    running_loss = 0.0
    running_corrects = 0
    total_cost = 0.0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            all_labels = torch.cat((all_labels, labels))
            inputs = inputs.to(device)
            labels = labels.to(device)

            probabilities = model(inputs)
            all_probabilities = torch.cat((all_probabilities, probabilities.detach().cpu()))
            _, preds = torch.max(probabilities, 1)

            cost = get_cost_from_tensors(y_true=labels, y_pred=preds, cost_matrix=cost_matrix)
            loss = criterion(probabilities=probabilities, y=labels, cost_matrix=cost_tag_matrix)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_cost += cost

    lambda_psi = get_lambda_psi_from_buckets(all_probabilities, all_labels, edges, interval, bucket_interval,
                                             cost_matrix)
    print('lambda psi', lambda_psi)
    cost_tag_matrix = update_cost_matrix(cost_tag_matrix, lambda_psi)

    ### metrics
    _, all_preds = torch.max(all_probabilities, 1)
    metrics = get_adacsl_metrics(y_true=all_labels,
                                 probabilities=all_probabilities,
                                 y_pred=all_preds,
                                 cost_matrix=cost_matrix)
    print('Valid')
    print('acc: ', metrics[0], 'ce: ', metrics[1], 'auc: ', metrics[2], 'f1: ', metrics[3])
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    print('Validation: {:.4f}: , Accuracy: {:.4f}, Cost: {:.4f}'.format(epoch_loss, epoch_acc, total_cost))

    return cost_tag_matrix, epoch_loss, epoch_acc, total_cost


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             criterion: Callable,
             cost_matrix: np.ndarray,
             device: torch.device) -> Tuple[float, float, float]:
    model.eval()

    dataset_size = len(dataloader.dataset)

    all_probabilities = torch.Tensor([])
    all_labels = torch.Tensor([])

    running_loss = 0.0
    running_corrects = 0
    total_cost = 0.0

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            all_labels = torch.cat((all_labels, labels))
            inputs = inputs.to(device)
            labels = labels.to(device)

            probabilities = model(inputs)
            all_probabilities = torch.cat((all_probabilities, probabilities.detach().cpu()))
            _, preds = torch.max(probabilities, 1)

            cost = get_cost_from_tensors(y_true=labels, y_pred=preds, cost_matrix=cost_matrix)
            loss = criterion(probabilities, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_cost += cost

    ### metrics
    _, all_preds = torch.max(all_probabilities, 1)
    metrics = get_adacsl_metrics(y_true=all_labels,
                                 probabilities=all_probabilities,
                                 y_pred=all_preds,
                                 cost_matrix=cost_matrix)
    print('Test')
    print('acc: ', metrics[0], 'ce: ', metrics[1], 'auc: ', metrics[2], 'f1: ', metrics[3])
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    print('Test: {:.4f}: , Accuracy: {:.4f}, Cost: {:.4f}'.format(epoch_loss, epoch_acc, total_cost))

    return epoch_loss, epoch_acc, total_cost


def run_adacsl_engine(model: nn.Module,
                      tr_dataloader: DataLoader,
                      val_dataloader: DataLoader,
                      test_dataloader: DataLoader,
                      optimizer: Optimizer,
                      criterion: Callable,
                      num_epochs: int,
                      cost_matrix: np.ndarray,
                      cost_tag_matrix: np.ndarray,
                      edges: np.ndarray,
                      device: torch.device):
    for epoch in range(num_epochs):
        train_metrics = train_adacsl(model=model,
                                     dataloader=tr_dataloader,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     epoch=epoch,
                                     num_epochs=num_epochs,
                                     cost_matrix=cost_matrix,
                                     cost_tag_matrix=cost_tag_matrix,
                                     device=device)

        cost_tag_matrix, *val_metrics = evaluate_adacsl(model=model,
                                                        dataloader=val_dataloader,
                                                        criterion=criterion,
                                                        edges=edges,
                                                        cost_matrix=cost_matrix,
                                                        cost_tag_matrix=cost_tag_matrix,
                                                        device=device)
        ce_criterion = nn.CrossEntropyLoss()
        evaluate(model=model,
                 dataloader=test_dataloader,
                 criterion=ce_criterion,
                 cost_matrix=cost_matrix,
                 device=device)
