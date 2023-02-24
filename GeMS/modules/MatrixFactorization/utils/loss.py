GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch


def bpr_loss(positive_score, negative_score):
    """
    Bayesian Personalised Ranking loss
    Args:
        positive_score: (tensor<float>) predicted scores for known positive items
        negative_score: (tensor<float>) predicted scores for negative sample items
    Returns:
        loss: (float) the mean value of the summed loss
    """
    eps = 1e-7 # Smooth the argument of the log to prevent potential numerical underflows
    loss = -torch.log(torch.sigmoid(positive_score - negative_score) + eps)
    return loss.mean()
