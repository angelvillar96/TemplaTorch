"""
Loss functions and loss-related utils
"""

import torch.nn as nn

import lib.metrics as metrics
from CONFIG import CONFIG


LOSSES = ["mse", "l2", "mae", "l1", "cross_entropy", "ce"]

def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    assert loss_type in LOSSES,  f"""ERROR! Loss {loss_type} not available.
        Use one of the following: {LOSSES}"""

    if loss_type in ["mse", "l2"]:
        loss = nn.MSELoss()
    elif loss_type in ["mae", "l1"]:
        loss = nn.L1Loss()
    elif loss_type in ["cross_entropy", "ce"]:
        loss = nn.CrossEntropyLoss()

    return loss


#
