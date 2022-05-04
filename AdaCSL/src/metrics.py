from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
import numpy as np
import torch
from typing import List

def detach_from_cuda_to_numpy(cuda_tensor: torch.Tensor) -> np.ndarray:
    np_array = cuda_tensor.detach().cpu().numpy()
    return np_array


def get_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    return accuracy


def get_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    auc = roc_auc_score(y_true=y_true, y_score=probabilities)
    return auc


def get_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    return f1


def get_cross_entropy(y_true: np.array, probabilities: np.array) -> float:
    cross_entropy = log_loss(y_true=y_true, y_pred=probabilities)
    return cross_entropy


def get_cost(y_pred: np.ndarray, y_true: np.ndarray, cost_matrix: np.ndarray) -> float:
    cost_list = [cost_matrix[i, j] for i, j in zip(y_pred, y_true)]
    cost = sum(cost_list)
    return cost


def get_cost_from_tensors(y_true: torch.Tensor, y_pred: torch.Tensor, cost_matrix: np.ndarray) -> float:
    y_true = detach_from_cuda_to_numpy(y_true).astype(int)
    y_pred = detach_from_cuda_to_numpy(y_pred).astype(int)
    cost = get_cost(y_pred=y_pred, y_true=y_true, cost_matrix=cost_matrix)
    return cost

def get_adacsl_metrics(y_true: np.ndarray,
                probabilities: np.ndarray,
                y_pred: np.ndarray,
                cost_matrix: np.ndarray) -> List[float]:

    accuracy = get_accuracy(y_true=y_true, y_pred=y_pred)
    cross_entropy = get_cross_entropy(y_true=y_true, probabilities=probabilities)
    auc = get_auc(y_true=y_true, probabilities=probabilities[:, 1])
    f1 = get_f1(y_true=y_true, y_pred=y_pred)
    return [accuracy, cross_entropy, auc, f1]


def get_metrics_from_tensors(y_true: torch.Tensor,
                             probabilities: torch.Tensor,
                             y_pred: torch.Tensor,
                             cost_matrix: np.ndarray) -> List[float,]:

    y_true = detach_from_cuda_to_numpy(y_true).astype(int)
    probabilities = detach_from_cuda_to_numpy(probabilities).astype(float)
    y_pred = detach_from_cuda_to_numpy(y_pred).astype(int)

    metrics = get_adacsl_metrics(y_true, probabilities, y_pred, cost_matrix)
    return metrics