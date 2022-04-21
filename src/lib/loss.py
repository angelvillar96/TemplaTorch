"""
Loss functions and loss-related utils
"""

import torch.nn as nn
from CONFIG import LOSSES


def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    if loss_type not in LOSSES:
        raise NotImplementedError(f"Loss {loss_type} not available. Use one of {LOSSES}")

    if loss_type in ["mse", "l2"]:
        loss = nn.MSELoss()
    elif loss_type in ["mae", "l1"]:
        loss = nn.L1Loss()
    elif loss_type in ["cross_entropy", "ce"]:
        loss = nn.CrossEntropyLoss()

    return loss


#
