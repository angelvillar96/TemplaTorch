"""
Implementation of learning rate schedulers, early stopping and other utils
for improving optimization
"""

from lib.logger import print_
import models.model_utils as model_utils


def update_scheduler(scheduler, exp_params, control_metric=None, iter=-1, end_epoch=False):
    """
    Updating the learning rate scheduler

    Args:
    -----
    scheduler: torch.optim
        scheduler to evaluate
    exp_params: dictionary
        dictionary containing the experiment parameters
    control_metric: float/torch Tensor
        Last computed validation metric.
        Needed for plateau scheduler
    iter: float
        number of optimization step.
        Needed for cyclic, cosine and exponential schedulers
    end_epoch: boolean
        True after finishing a validation epoch or certain number of iterations.
        Triggers schedulers such as plateau or fixed-step
    """
    scheduler_type = exp_params["training"]["scheduler"]
    if(scheduler_type == "plateau" and end_epoch):
        scheduler.step(control_metric)
    elif(scheduler_type in ["step", "multi_step"] and end_epoch):
        scheduler.step()
    elif(scheduler_type == "exponential" and not end_epoch):
        scheduler.step(iter)
    elif(scheduler_type == "" or scheduler_type is None):
        pass
    else:
        pass
        # print(f"Unrecognized scheduler type {scheduler_type}...")
    return


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
    during the first few iterations until it reaches the standard LR.
    Learning rate warmup is stopped either whenever the 'warmup_steps' or 'max_epochs' is reached.

    Args:
    -----
    init_lr: float
        initial learning rate
    warmup_steps: integer
        number of optimization steps to warm up for
    max_epochs: integer
        maximum number of epochs to warmup. It overrides 'warmup_step'
    """

    def __init__(self, init_lr, warmup_steps=-1, max_epochs=1):
        """ Initializer """
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.active = True
        self.final_step = -1

    def __call__(self, iter, epoch, optimizer):
        """ Computing actual learning rate and updating optimizer """
        if(iter > self.warmup_steps):
            if(self.active):
                self.final_step = iter
                self.active = False
                lr = self.init_lr
                print_("Finished learning rate warmup period...")
                print_(f"  --> Reached iter {iter} >= {self.warmup_steps}")
                print_(f"  --> Reached at epoch {epoch}")
        elif(epoch >= self.max_epochs):
            if(self.active):
                self.final_step = iter
                self.active = False
                lr = self.init_lr
                print_("Finished learning rate warmup period:")
                print_(f"  --> Reached epoch {epoch} >= {self.max_epochs}")
                print_(f"  --> Reached at iter {iter}")
        else:
            if iter >= 0:
                lr = self.init_lr * (iter / self.warmup_steps)
                for params in optimizer.param_groups:
                    params["lr"] = lr
        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        """ Loading state dictinary """
        self.init_lr = state_dict.init_lr
        self.warmup_steps = state_dict.warmup_steps
        self.max_epochs = state_dict.max_epochs
        self.active = state_dict.active
        self.final_step = state_dict.final_step
        return


class WarmupVSScehdule:
    """
    Wrapper that orquestrates the learning rate warmup and learning rate scheduling.
      - During warmup iterations --> calling LR-Warmup
      - After warmup is done --> Updating learning rate based on schedule

    Args:
    -----
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    lr_warmup: LRWarmUp Object
        Module that takes care of the learning rate warmup functionalities
    """

    def __init__(self, optimizer, lr_warmup, scheduler):
        """ Module intializer """
        self.optimizer = optimizer
        self.lr_warmup = lr_warmup
        self.scheduler = scheduler

    def __call__(self, iter, epoch, exp_params, end_epoch, control_metric=None):
        """
        Calling either LR-Warmup or LR-Scheduler depending on the current
        training epoch or iteration

        Args:
        -----
        iter: int
            Current training batch being processed by the model
        epoch: int
            Current training epoch being processed
        exp_params: directory
            Experiment parameters
        end_epoch: bool
            True after finishing a validation epoch or certain number of iterations.
            Used for triggering schedulers such as plateau, step or multi-step
        control_metric: float/torch Tensor
            Last computed validation metric. Needed for plateau scheduler
        """
        if self.lr_warmup is not None and self.lr_warmup.active:
            self.lr_warmup(iter=iter, epoch=epoch, optimizer=self.optimizer)
        else:
            final_warmup_step = 0 if self.lr_warmup is None else self.lr_warmup.final_step
            update_scheduler(
                    scheduler=self.scheduler,
                    exp_params=exp_params,
                    iter=iter - final_warmup_step,
                    end_epoch=end_epoch,
                    control_metric=control_metric
                )
        return


class EarlyStop:
    """
    Implementation of an early stop criterion.
    This module keeps track of the number of epochs without improvement on certain validation metric,
    and stops the training procedure if the metric has not improved for a number of epochs.

    Args:
    -----
    mode: string ['min', 'max']
        whether we validate based on maximizing or minimizing a metric
    delta: float
        threshold to consider improvements. Currently hard-coded
    use_early_stop: bool
        If True, early stopping functionalities are computed. Otherwise, this module simply
        counts the number of epochs with improvement on the validation metric
    patience: integer
        number of epochs without improvement to trigger early stopping
    """

    def __init__(self, mode="min", delta=1e-6, use_early_stop=True, patience=10):
        """ Early stopper initializer """
        assert mode in ["min", "max"]
        self.mode = mode
        self.delta = delta
        self.use_early_stop = use_early_stop
        self.patience = patience
        self.counter = 0

        if(mode == "min"):
            self.best = 1e15
            self.criterion = lambda x: x < (self.best - self.delta)
        elif(mode == "max"):
            self.best = 1e-15
            self.criterion = lambda x: x > (self.best + self.delta)

        if self.use_early_stop:
            print_("Setting up Early Stopping")
            print_(f"  --> Patience {self.patience}")
            print_(f"  --> Mode {self.mode}")
        else:
            print_("Not using Early Stopping")

        return

    def __call__(self, value, epoch=0, writer=None):
        """
        Comparing current metric agains best past results and computing if we
        should early stop or not

        Args:
        -----
        value: float
            validation metric measured by the early stopping criterion
        wirter: TensorboardWriter
            If not None, TensorboardWriter to log the early-stopper counter

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
        if stop_training and self.use_early_stop:
            print_(f"It has been {self.patience} epochs without improvement")
            print_("  --> Trigering early stopping")

        # logging to Tensorboar a counter of epochs without improvements
        if writer is not None:
            writer.add_scalar(
                name="Learning/Early-Stopping Counter",
                val=self.counter,
                step=epoch + 1,
            )

        if not self.use_early_stop:
            stop_training = False
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


#
