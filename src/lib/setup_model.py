"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import traceback
import torch

from lib.logger import log_function, print_
from lib.schedulers import LRWarmUp, ExponentialLRSchedule
from lib.utils import create_directory
from models import MODEL_MAP


@log_function
def setup_model(model_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    model_name = model_params["model_name"]
    model_params = model_params["model_params"]
    if model_name not in MODEL_MAP.keys():
        raise NotImplementedError(f"'{model_name = }' not in recognized models: {MODEL_MAP.keys()}")

    model = MODEL_MAP[model_name](**model_params)

    return model


def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    trainer=self_,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except


@log_function
def save_checkpoint(trainer, finished=False, savedir="models", savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    trainer: Trainer or BaseTrainer
        Trainer object with all the modules to save
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{trainer.epoch}.pth"

    create_directory(trainer.exp_path, savedir)
    savepath = os.path.join(trainer.exp_path, savedir, checkpoint_name)

    scheduler_data = "" if trainer.scheduler is None else trainer.scheduler.state_dict()
    torch.save({
            'epoch': trainer.epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data,
            "lr_warmup": trainer.warmup_scheduler.lr_warmup
        }, savepath)

    return


@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)
    # loading model parameters. Try catch is used to allow different dicts
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        model.load_state_dict(checkpoint)

    # returning only model for transfer learning or returning also optimizer for resuming training
    if(only_model):
        return model

    # loading saved state for optimizer, scheduler, and lr warmup
    optimizer, scheduler = kwargs["optimizer"], kwargs["scheduler"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if "lr_warmup" in kwargs:
        lr_warmup = kwargs["lr_warmup"]
        lr_warmup.load_state_dict(checkpoint['lr_warmup'])
    else:
        lr_warmup = None
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, lr_warmup, epoch


@log_function
def setup_optimization(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters
    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model

    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    lr_warmup: LRWarmUp Object
        Module that takes care of the learning rate warmup functionalities
    """

    # setting up optimizer and LR-scheduler
    optimizer = setup_optimizer(parameters=model.parameters(), exp_params=exp_params)
    scheduler = setup_scheduler(exp_params=exp_params, optimizer=optimizer)
    lr_warmup = setup_lr_warmup(exp_params=exp_params)

    return optimizer, scheduler, lr_warmup


def setup_optimizer(parameters, exp_params):
    """ Instanciating a new optimizer """
    lr = exp_params["training"]["lr"]
    momentum = exp_params["training"]["momentum"]
    optimizer = exp_params["training"]["optimizer"]
    nesterov = exp_params["training"]["nesterov"]

    # SGD-based optimizer
    if(optimizer == "adam"):
        print_("Using the Adam optimizer")
        print_(f"  --> Learning rate: {lr}")
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif(optimizer == "adamw"):
        print_("Using the AdamW optimizer")
        print_(f"  --> Learning rate: {lr}")
        optimizer = torch.optim.AdamW(parameters, lr=lr)
    else:
        weight_decay = exp_params["training"].get("weight_decay", 0.0005)
        print_("Using the AdamW optimizer")
        print_(f"  --> Learning rate: {lr}")
        print_(f"  --> Momentum:      {momentum}")
        print_(f"  --> Nesterov:      {nesterov}")
        print_(f"  --> Weight decay:  {weight_decay}")
        optimizer = torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay
            )

    return optimizer


def setup_scheduler(exp_params, optimizer):
    """ Instanciating a new scheduler """
    lr = exp_params["training"]["lr"]
    lr_factor = exp_params["training"]["lr_factor"]
    patience = exp_params["training"]["patience"]
    scheduler = exp_params["training"]["scheduler"]

    if(scheduler == "plateau"):
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"  --> Patience: {patience}")
        print_(f"  --> Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="min",
                verbose=True
            )
    elif(scheduler == "step"):
        print_("Setting up Step LR-Scheduler")
        print_(f"  --> Step Size: {patience}")
        print_(f"  --> Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )
    elif(scheduler == "multi_step"):
        print_("Setting up MultiStepLR LR-Scheduler")
        print_(f"  --> Milestones: {patience}")
        print_(f"  --> Factor:     {lr_factor}")
        if not isinstance(patience, list):
            raise ValueError(f"Milestones ({patience}) must be a list of increasing integers...")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                milestones=patience
            )
    elif(scheduler == "exponential"):
        print_("Setting up Exponential LR-Scheduler")
        print_(f"  --> Init LR: {lr}")
        print_(f"  --> Factor:  {lr_factor}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = None

    return scheduler


@log_function
def setup_lr_warmup(exp_params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.

    Example:
    -------
        #  Learning rate is initialized with 3e-4 * (1/1000). For the first 1000 iterations
        #  or first epoch, the learning rate is updated to 3e-4 * (iter/1000).
        # after the warmup period, learning rate is fixed at 3e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
        lr_warmup = LRWarmUp(init_lr=3e-4, warmup_steps=1000, max_epochs=1)
        ...
        lr_warmup(iter=cur_iter, epoch=cur_epoch, optimizer=optimizer)  # updating lr
    """
    use_warmup = exp_params["training"]["lr_warmup"]
    lr = exp_params["training"]["lr"]
    if(use_warmup):
        warmup_steps = exp_params["training"]["warmup_steps"]
        warmup_epochs = exp_params["training"]["warmup_epochs"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps, max_epochs=warmup_epochs)
        print_("Setting up learning rate warmup:")
        print_(f"  Target LR:     {lr}")
        print_(f"  Warmup Steps:  {warmup_steps}")
        print_(f"  Warmup Epochs: {warmup_epochs}")
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
        print_("Not using learning rate warmup...")
    return lr_warmup


#
