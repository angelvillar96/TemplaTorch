"""
Updating experiment parameters json file with new parameters
from the DEFAULTS dictionary
"""

import os
import json
from lib.arguments import get_directory_argument
from lib.config import Config


def update_params(exp_path):
    """
    Updating parameters
    """

    exp_path, _ = get_directory_argument()
    cfg = Config(exp_path=exp_path)
    exp_params = cfg.load_exp_config_file()
    exp_params = cfg.update_config(exp_params)

    params_file = os.path.join(exp_path, "experiment_params.json")
    with open(params_file, "w") as f:
        json.dump(exp_params, f)

    return


if __name__ == "__main__":
    exp_path = get_directory_argument()
    update_params(exp_path)

#
