import numpy as np
import torch
from typing import Union, Collection, List

from metrics import get_cost_from_tensors


def adaptive_cross_entropy(probabilities: torch.Tensor, y: torch.Tensor, cost_matrix: np.ndarray) -> torch.Tensor:
    probabilities = torch.clamp(probabilities, min=1e-5, max=1 - 1e-5)
    y_hat = probabilities[:, 1]

    c10 = cost_matrix[1, 0]
    c01 = cost_matrix[0, 1]

    loss = -(y * torch.log(y_hat)) - (c10 / c01) * ((1 - y) * torch.log(1 - y_hat))

    return torch.mean(loss)


def update_cost_matrix(cost_matrix: np.ndarray, lambda_psi: float) -> np.ndarray:
    cost_matrix[1, 0] *= lambda_psi

    return cost_matrix


def get_new_probas(probas: torch.Tensor, t_actual: float) -> torch.Tensor:
    pos_prob = probas[:, 1]
    new_pos_prob = pos_prob / (((t_actual / (1 - t_actual)) * (1 - pos_prob)) + pos_prob)
    new_neg_prob = 1 - new_pos_prob
    new_probas = torch.stack((new_neg_prob, new_pos_prob), dim=1)
    return new_probas


def get_best_t_actual(probas: torch.Tensor,
                      labels: torch.Tensor,
                      t_actuals: Union[Collection[float]],
                      cost_matrix: np.ndarray) -> float:
    if len(probas[:, 1]) == 0:
        best_t_actual = 0.5
    else:
        t_cost_dict = {k: None for k in t_actuals}

        for t_actual in t_cost_dict.keys():
            new_probas = get_new_probas(probas, t_actual)
            _, new_preds = torch.max(new_probas, 1)
            t_actual_cost = get_cost_from_tensors(y_true=labels, y_pred=new_preds, cost_matrix=cost_matrix)
            t_cost_dict[t_actual] = t_actual_cost

        best_t_actual = min(t_cost_dict, key=t_cost_dict.get)
        if best_t_actual == 0:
            best_t_actual = 0.5

    return best_t_actual


def get_e(t_actual: float, t_tag: float = 0.5) -> float:
    numerator = -(t_tag - t_actual)
    denominator = t_tag * (1 - t_tag)
    power_of = numerator / denominator
    e = np.exp(power_of)
    return e


def get_lambda(probas: np.ndarray, labels: np.ndarray, t_actuals: np.ndarray, cost_matrix) -> float:
    best_t_actual = get_best_t_actual(probas, labels, t_actuals, cost_matrix)
    lambda_ = get_e(best_t_actual)
    bucket_size = len(probas)
    return lambda_ * bucket_size


def get_bucket(probas: np.ndarray,
               labels: np.ndarray,
               start_edge: Union[int, float],
               end_edge: Union[int, float],
               bucket_interval: Union[int, float]):
    start_mask = probas[:, 1] >= start_edge
    end_mask = probas[:, 1] < end_edge

    bucket_probas = probas[start_mask & end_mask]
    bucket_labels = labels[start_mask & end_mask]
    bucket_t_actuals = np.arange(start_edge, end_edge, bucket_interval)

    return bucket_probas, bucket_labels, bucket_t_actuals


def get_lambda_psi_from_buckets(probas: np.ndarray,
                                labels: np.ndarray,
                                edges: List[Union[int, float]],
                                interval: Union[int, float],
                                bucket_interval: Union[int, float],
                                cost_matrix: np.ndarray) -> Union[float, int]:
    total_size = len(probas)
    bucket_lambdas = []
    for start_edge in edges:
        end_edge = start_edge + interval
        bucket_probas, bucket_labels, bucket_t_actuals = get_bucket(probas, labels, start_edge, end_edge,
                                                                    bucket_interval)
        bucket_lambda = get_lambda(bucket_probas, bucket_labels, bucket_t_actuals, cost_matrix)
        bucket_lambdas.append(bucket_lambda)

    lambda_psi = sum(bucket_lambdas)
    lambda_psi = lambda_psi / total_size
    return lambda_psi
