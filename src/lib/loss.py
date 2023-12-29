"""
Loss functions and loss-related utils
"""

import torch
import torch.nn as nn
from lib.logger import log_info, print_


class LossTracker:
    """
    Class for computing, weighting and tracking several loss functions

    Args:
    -----
    loss_params: dict
        Loss section of the experiment paramteres JSON file
    """

    def __init__(self, loss_params):
        """ Loss tracker initializer """
        assert isinstance(loss_params, list), f"Loss_params must be a list, and not {type(loss_params)}"
        LOSSES = LOSS_DICT.keys()
        for loss in loss_params:
            if loss["type"] not in LOSSES:
                raise NotImplementedError(f"Loss {loss['type']} not implemented. Use one of {LOSSES}")

        print_(f"Instanciating LossTracker module:")
        self.loss_computers = {}
        for loss in loss_params:
            loss_type, loss_weight = loss["type"], loss["weight"]
            self.loss_computers[loss_type] = {}
            self.loss_computers[loss_type]["metric"] = get_loss(loss_type, **loss)
            self.loss_computers[loss_type]["weight"] = loss_weight
            print_(f"  --> Loss {loss_type} with weight {loss_weight}")
        self.reset()
        return

    def reset(self):
        """ Reseting loss tracker """
        self.loss_values = {}
        self.loss_values["_total"] = []
        return

    def __call__(self, **kwargs):
        """ Wrapper for calling accumulate """
        self.accumulate(**kwargs)

    def accumulate(self, **kwargs):
        """ Computing the different metrics and adding them to the results list """
        total_loss = 0
        for loss in self.loss_computers:
            loss_val = self.loss_computers[loss]["metric"](**kwargs)
            # if loss is dictionary, we save each of the keys as a different loss
            if isinstance(loss_val, dict):
                cur_total_loss = 0.
                for k, cur_loss_val in loss_val.items():
                    if k not in self.loss_values:
                        self.loss_values[k] = []
                    self.loss_values[k].append(cur_loss_val)
                    cur_total_loss = cur_total_loss + cur_loss_val
                total_loss = total_loss + cur_total_loss
            # loss is simply a tensor
            else:
                if loss not in self.loss_values:
                    self.loss_values[loss] = []
                self.loss_values[loss].append(loss_val)
                total_loss = total_loss + loss_val * self.loss_computers[loss].get("weight", 1.)
        self.loss_values["_total"].append(total_loss)
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        self.loss_values["mean_loss"] = {}
        for loss in self.loss_values.keys():
            if loss == "mean_loss":
                continue
            self.loss_values["mean_loss"][loss] = torch.stack(self.loss_values[loss]).mean()
        self.loss_values["mean_loss"]["_total"] = torch.stack(self.loss_values["_total"]).mean()
        return

    def get_last_losses(self, total_only=False):
        """ Fetching the last computed loss value for each loss function """
        if total_only:
            last_losses = self.loss_values["_total"][-1]
        else:
            last_losses = {loss: loss_vals[-1] for loss, loss_vals in self.loss_values.items()}
        return last_losses

    def summary(self, log=True, get_results=True):
        """ Printing and fetching the results """
        if log:
            log_info("LOSS VALUES:")
            log_info("--------")
            for loss, loss_value in self.loss_values["mean_loss"].items():
                log_info(f"  {loss}:  {round(loss_value.item(), 5)}")

        return_val = self.loss_values["mean_loss"] if get_results else None
        return return_val


def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    LOSSES = LOSS_DICT.keys()
    if loss_type not in LOSSES:
        raise NotImplementedError(f"Loss {loss_type} not available. Use one of {LOSSES}")

    # fetching loss class
    print(f"creating loss function of type: {loss_type}")
    loss_class = LOSS_DICT[loss_type]

    # handling args
    loss_params = {}

    # instanciating loss class
    loss = loss_class(**loss_params)
    return loss



##############
### LOSSES ###
##############


class Loss(nn.Module):
    """
    Base class for custom loss functions
    """

    REQUIRED_ARGS = []

    def __init__(self):
        super().__init__()

    def _unpack_kwargs(self, **kwargs):
        """
        Fetching the required arguments from kwargs
        """
        out = []
        for arg in self.REQUIRED_ARGS:
            if arg not in kwargs:
                raise ValueError(f"Required '{arg = }' not in {kwargs.keys() = } in {self.__class__.__name__}")
            out.append(kwargs[arg])
        return out


class MSELoss(Loss):
    """ Overriding MSE Loss """

    REQUIRED_ARGS = ["preds", "targets"]

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """ Computing loss """
        preds, targets = self._unpack_kwargs(**kwargs)
        loss = self.mse(preds, targets)
        return loss


class L1Loss(Loss):
    """ Overriding L1Loss Loss """

    REQUIRED_ARGS = ["preds", "targets"]

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, **kwargs):
        """ Computing loss """
        preds, targets = self._unpack_kwargs(**kwargs)
        loss = self.mae(preds, targets)
        return loss


class CrossEntropyLoss(Loss):
    """ Overriding Cross Entropy Loss"""

    REQUIRED_ARGS = ["preds", "targets"]

    def __init__(self):
        """ Module initializer """
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        """ Computing loss """
        preds, targets = self._unpack_kwargs(**kwargs)
        loss = self.ce(preds, targets)
        return loss



################
### LOSS MAP ###
################


LOSS_DICT = {
    "mse": MSELoss,
    "mae": L1Loss,
    "cross_entropy": CrossEntropyLoss,
}
