# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def single_emd_loss(p, q, throshold, C, r=2):
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
    M = 0.0
    for i in range(1, length + 1):
        emd_loss += sum(torch.abs(p[:i] - q[:i])) ** r
    emd_loss = (emd_loss / length) ** (1. / r)
    if emd_loss < throshold:
        emd_loss = emd_loss
    else:
        M += (C + emd_loss)
        emd_loss = M*emd_loss
    return emd_loss


def ada_emd_loss(p, q, C, throshold):
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
        loss_vector.append(single_emd_loss(p[i], q[i], throshold, C, r=2))
    return sum(loss_vector) / mini_batch_size
