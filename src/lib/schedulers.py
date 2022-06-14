"""
Implementation of learning rate schedulers, early stopping and other utils
for improving optimization
"""

from lib.logger import print_
import models.model_utils as model_utils


class ExponentialLRSchedule:
    """
    Exponential LR Scheduler that decreases the learning rate by multiplying it
    by an exponentially decreasing decay factor:
        LR = LR * gamma ^ (step/total_steps)

    Args:
    -----
    optimizer: torch.optim
        Optimizer to schedule
    init_lr: float
        base learning rate to decrease with the exponential scheduler
    gamma: float
        exponential decay factor
    total_steps: int/float
        number of optimization steps to optimize for. Once this is reached,
        lr is not decreased anymore
    """

    def __init__(self, optimizer, init_lr, gamma=0.5, total_steps=100_000):
        """ Module initializer """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.total_steps = total_steps
        return

    def update_lr(self, step):
        """ Computing exponential lr update """
        new_lr = self.init_lr * self.gamma ** (step / self.total_steps)
        return new_lr

    def step(self, iter):
        """ Scheduler step """
        if(iter < self.total_steps):
            for params in self.optimizer.param_groups:
                params["lr"] = self.update_lr(iter)
        elif(iter == self.total_steps):
            print_(f"Finished exponential decay due to reach of {self.total_steps} steps")
        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        """ Loading state dictinary """
        self.init_lr = state_dict["init_lr"]
        self.gamma = state_dict["gamma"]
        self.total_steps = state_dict["total_steps"]
        return


class LRWarmUp:
    """
    Class for performing learning rate warm-ups. We increase the learning rate
    during the first few iterations until it reaches the standard LR

    Args:
    -----
    init_lr: float
        initial learning rate
    warmup_steps: integer
        number of optimization steps to warm up for
    max_epochs: integer
        maximum number of epochs to warmup. It overrides 'warmup_step'
    """

    def __init__(self, init_lr, warmup_steps, max_epochs=1):
        """ Initializer """
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.active = True

    def __call__(self, iter, epoch, optimizer):
        """ Computing actual learning rate and updating optimizer """
        if(iter > self.warmup_steps or epoch >= self.max_epochs):
            if(self.active):
                self.active = False
                lr = self.init_lr
                print_("Finished learning rate warmup period...")
        else:
            lr = self.init_lr * (iter / self.warmup_steps)
            for params in optimizer.param_groups:
                params["lr"] = lr

        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict


class EarlyStop:
    """
    Implementation of an early stop criterion

    Args:
    -----
    mode: string ['min', 'max']
        whether we validate based on maximizing or minmizing a metric
    delta: float
        threshold to consider improvements
    patience: integer
        number of epochs without improvement to trigger early stopping
    """

    def __init__(self, mode="min", delta=1e-6, patience=7):
        """ Early stopper initializer """
        assert mode in ["min", "max"]
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.counter = 0

        if(mode == "min"):
            self.best = 1e15
            self.criterion = lambda x: x < (self.best - self.min_delta)
        elif(mode == "max"):
            self.best = 1e-15
            self.criterion = lambda x: x < (self.best - self.min_delta)

        return

    def __call__(self, value):
        """
        Comparing current metric agains best past results and computing if we
        should early stop or not

        Args:
        -----
        value: float
            validation metric measured by the early stopping criterion

        Returns:
        --------
        stop_training: boolean
            If True, we should early stop. Otherwise, metric is still improving
        """
        are_we_better = self.criterion(value)
        if(are_we_better):
            self.counter = 0
            self.best = value
        else:
            self.counter = self.counter + 1

        stop_training = True if(self.counter >= self.patience) else False

        return stop_training


class Freezer:
    """
    Class for freezing and unfreezing nn.Module given number of epochs

    Args:
    -----
    module: nn.Module
        nn.Module to freeze or unfreeze
    frozen_epochs: integer
        Number of initial epochs in which the model is kept frozen
    """

    def __init__(self, module, frozen_epochs=0):
        """ Module initializer """
        self.module = module
        self.frozen_epochs = frozen_epochs
        self.is_frozen = False

    def __call__(self, epoch):
        """ """
        if epoch < self.frozen_epochs and self.is_frozen is False:
            print_(f"  --> Still in frozen epochs  {epoch} < {self.frozen_epochs}. Freezing module...")
            model_utils.freeze_params(self.module)
            self.is_frozen = True
        elif epoch >= self.frozen_epochs and self.is_frozen is True:
            print_(f"  --> Finished frozen epochs {epoch} = {self.frozen_epochs}. Unfreezing module...")
            model_utils.unfreeze_params(self.module)
            self.is_frozen = False
        return
