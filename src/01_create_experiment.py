"""
Creating experiment directory and initalizing it with defauls

@author: Angel Villar-Corrales
"""

import os
import traceback
from lib.arguments import create_experiment_arguments
from lib.callbacks import create_callbacks_file
from lib.config import Config
from lib.logger import Logger, print_
from lib.utils import create_directory, delete_directory, timestamp, clear_cmd

from CONFIG import CONFIG


def initialize_experiment():
    """
    Creating experiment directory and initalizing it with defauls
    """
    # reading command line args
    args = create_experiment_arguments()
    exp_dir, config, exp_name = args.exp_directory, args.config, args.name
    exp_name = f"experiment_{timestamp()}" if exp_name is None or len(exp_name) < 1 else exp_name
    exp_path = os.path.join(CONFIG["paths"]["experiments_path"], exp_dir, exp_name)
    if os.path.exists(exp_path):
        print(f"Experiment {exp_name} already in experiment directory {exp_dir}...")
        exit()

    # creating directories
    create_directory(exp_path)
    _ = Logger(exp_path)  # initialize logger once exp_dir is created
    create_directory(dir_path=exp_path, dir_name="plots")
    create_directory(dir_path=exp_path, dir_name="tboard_logs")

    # creating default callbacks file
    create_callbacks_file(dir_path=exp_path)

    try:
        cfg = Config()
        cfg.create_exp_config_file(
                exp_path=exp_path,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                config=config
            )
        print_(f"  --> {args.model_name = }")
        print_(f"  --> {args.dataset_name = }")
        print_(f"Experiment {exp_name} created successfully")
    except Exception:
        print_("An error has occurred...\n Removing experiment directory")
        delete_directory(dir_path=exp_path)
        message = traceback.format_exc()
        print(message)

    return


if __name__ == "__main__":
    clear_cmd()
    initialize_experiment()

#
