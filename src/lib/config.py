"""
Methods to manage parameters and configurations
"""

import os
import json

from CONFIG import DEFAULTS


def create_exp_config_file(exp_path):
    """
    Creating a JSON file with exp configs in the experiment path
    """
    assert os.path.exists(exp_path), f"ERROR!: exp_path {exp_path} does not exists..."

    exp_config = os.path.join(exp_path, "experiment_params.json")
    with open(exp_config, "w") as file:
        json.dump(DEFAULTS, file)

    return


def load_exp_config_file(exp_path):
    """
    Loading the JSON file with exp configs
    """

    exp_config = os.path.join(exp_path, "experiment_params.json")
    assert os.path.exists(exp_path), f"ERROR! exp_path {exp_config} does not exist..."

    with open(exp_config,) as file:
        exp_params = json.load(file)

    return exp_params


def update_config(exp_params):
    """
    Updating an experiments parameters file with new-configurations from CONFIG.
    """

    groups = ["dataset", "model", "training", "loss"]

    # TODO: Add recursion to make it always work
    # adding missing parameters to the exp_params dicitionary
    for group in groups:
        for k in DEFAULTS[group].keys():
            if(k not in exp_params[group]):
                if(isinstance(DEFAULTS[group][k], (dict))):
                    exp_params[group][k] = {}
                else:
                    exp_params[group][k] = DEFAULTS[group][k]

            if(isinstance(DEFAULTS[group][k], dict) ):
                for q in DEFAULTS[group][k].keys():
                    if(q not in exp_params[group][k]):
                        exp_params[group][k][q] = DEFAULTS[group][k][q]
    return exp_params

#
