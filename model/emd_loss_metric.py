# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += sum(torch.abs(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss_func(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def emd_loss_metric(p, q, r=2):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    # print(loss_vector)
    return sum(loss_vector) / mini_batch_size


def emd_dis(x, y_true, dist_r = 1):
    cdf_x = torch.cumsum(x, dim=-1)
    cdf_ytrue = torch.cumsum(y_true, dim=-1)
    if dist_r == 2:
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
    else:
        samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
    loss = torch.mean(samplewise_emd)
    return loss


def compute_mse(p, q):
    mse_fn = torch.nn.MSELoss(reduction='mean')
    mse_value = mse_fn(p, q)
    return mse_value
