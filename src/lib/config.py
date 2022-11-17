"""
Methods to manage parameters and configurations

TODO:
 - Add support to change any values from the command line
"""

import os
import json

from lib.logger import print_
from lib.utils import timestamp
from CONFIG import DEFAULTS, CONFIG, MODELS


class Config(dict):
    """
    Module that creates, saves, and loads the experiemnts_parameters json/dictionary
    associated to each experiment.
    """

    _default_values = DEFAULTS
    _help = "Potentially you can add here comments for what your configs are"
    _config_groups = ["dataset", "model", "training", "loss", "metrics"]

    def create_exp_config_file(self, exp_path=None, model_name=None, config=None):
        """
        Creating a JSON file with exp configs in the experiment path
        """
        if model_name is not None and model_name not in MODELS:
            raise NotImplementedError(f"Given model name {model_name} not in models: {MODELS}...")

        # creating from predetermined config
        if config is not None:
            config_file = os.path.join(CONFIG["paths"]["configs_path"], config)
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Given config file {config_file} does not exist...")

            with open(config_file) as file:
                self = json.load(file)
                self["_general"] = {}
                self["_general"]["exp_path"] = exp_path
            print_(f"Creating experiment parameters file from config {config}...")

        # creating from defaults, given the model name
        for key in self._default_values.keys():
            if key == "model":
                self[key] = self._get_model_dict(model_name)
            else:
                self[key] = self._default_values[key]
        self["_general"] = {}
        self["_general"]["exp_path"] = exp_path
        self["_general"]["created_time"] = timestamp()

        # storing on experiment directory
        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(self, file)
        return

    def _get_model_dict(self, model_name):
        """
        Obtaining a dictionary with the model parameters only relevant to the given model name
        """
        model_params = {}
        default_model_params = self._default_values["model"]
        for key, val in default_model_params.items():
            if model_name is None:
                model_params[key] = val
            elif isinstance(val, dict) and key == model_name:
                model_params[key] = val
            elif not isinstance(val, dict):
                model_params[key] = val

        if model_name is not None:
            model_params["model_name"] = model_name
        return model_params

    def load_exp_config_file(self, exp_path=None):
        """
        Loading the JSON file with exp configs
        """
        exp_config = os.path.join(exp_path, "experiment_params.json")
        if not os.path.exists(exp_config):
            raise FileNotFoundError(f"ERROR! exp. configs file {exp_config} does not exist...")

        with open(exp_config) as file:
            self = json.load(file)
        return self

    def update_config(self, exp_params):
        """
        Updating an experiments parameters file with newly added configurations from CONFIG.
        """
        # TODO: Add recursion to make it always work
        for group in Config._config_groups:
            for k in Config._default_values[group].keys():
                if(k not in exp_params[group]):
                    if(isinstance(Config._default_values[group][k], (dict))):
                        exp_params[group][k] = {}
                    else:
                        exp_params[group][k] = Config._default_values[group][k]

                if(isinstance(Config._default_values[group][k], dict)):
                    for q in Config._default_values[group][k].keys():
                        if(q not in exp_params[group][k]):
                            exp_params[group][k][q] = Config._default_values[group][k][q]
        return exp_params

    def save_exp_config_file(self, exp_path=None, exp_params=None):
        """ Dumping experiment parameters into path """
        exp_path = self["_general"]["exp_path"] if exp_path is None else exp_path
        exp_params = self if exp_params is None else exp_params

        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(exp_params, file)
        return

#
