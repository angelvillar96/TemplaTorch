"""
Creating experiment directory and initalizing it with defauls

@author: Angel Villar-Corrales
"""

import os

from lib.arguments import create_experiment_arguments
from lib.config import create_exp_config_file
from lib.logger import Logger
from lib.utils import create_directory, timestamp, clear_cmd

from CONFIG import CONFIG

def initialize_experiment():
    """
    Creating experiment directory and initalizing it with defauls
    """

    # reading command line args
    args = create_experiment_arguments()
    exp_dir = args.exp_directory
    exp_name = f"experiment_{timestamp()}"
    exp_path = os.path.join(CONFIG["paths"]["experiments_path"], exp_dir, exp_name)

    # creating directories
    create_directory(exp_path)
    _ = Logger(exp_path)  # initialize logger once exp_dir is created
    create_directory(dir_path=exp_path, dir_name="plots")
    create_directory(dir_path=exp_path, dir_name="tensorboard_logs")

    create_exp_config_file(exp_path=exp_path)

    return


if __name__ == "__main__":
    clear_cmd()
    initialize_experiment()

#
