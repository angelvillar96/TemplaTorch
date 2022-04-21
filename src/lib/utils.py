"""
Utils methods for bunch of purposes, including
    - Reading/writing files
    - Creating directories
    - Timestamp
    - Handling tensorboard
"""

import os
import random
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.logger import log_function
from CONFIG import CONFIG


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = CONFIG["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def clear_cmd():
    """Clearning command line window"""
    os.system('cls' if os.name == 'nt' else 'clear')
    return


@log_function
def create_directory(dir_path, dir_name=None):
    """
    Creating a folder in given path.
    """
    if(dir_name is not None):
        dir_path = os.path.join(dir_path, dir_name)
    if(not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    return


def timestamp():
    """ Obtaining the current timestamp in an human-readable way """

    timestamp = str(datetime.datetime.now()).split('.')[0] \
                                            .replace(' ', '_') \
                                            .replace(':', '-')

    return timestamp


@log_function
def log_architecture(model, exp_path, fname="model_architecture.txt"):
    """
    Printing architecture modules into a txt file (fname) located at the given path
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"

    savepath = os.path.join(exp_path, fname)
    with open(savepath, "w") as f:
        f.write("")
    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(model))

    return


class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    Args:
    -----
    logdir: string
        path where the tensorboard logs will be stored
    """
    def __init__(self, logdir):
        """ Initializing tensorboard writer """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        """ Adding a scalar for plot """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name:val for (val_name,val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure):
        """ Adding a whole new figure to the tensorboard"""
        self.writer.add_figure(tag=tag, figure=figure)
        return

    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val[-1] for key,val in dict.items()}
        else:
            dict = {key: val[-1] for key,val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name, dict, step)
        return

#
