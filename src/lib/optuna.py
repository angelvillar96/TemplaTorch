"""
Utils files for handling optuna studies
"""
import os
import json
import optuna
import torch

from CONFIG import OPTUNA
from lib.logger import print_, log_info
import lib.utils as utils


def create_optuna_values_file(exp_path):
    """ Creating optuna value file """
    exp_config = os.path.join(exp_path, "optuna_values.json")
    with open(exp_config, "w") as file:
        json.dump(OPTUNA, file)
    return


def load_optuna_values_file(exp_path):
    """ Creating optuna value file """
    exp_config = os.path.join(exp_path, "optuna_values.json")
    with open(exp_config) as file:
        data = json.load(file)
    return data


def load_optuna_study(exp_path, study_name=None):
    """ Loading and unpickeling an Optuna study """
    study_file = os.path.join(exp_path, "optuna.db")
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_file}")
    return study


def suggest_values(trial, exp_path, exp_params, verbose=False):
    """
    Suggesting values for several hyper-parameters to optimize, and updatiing the
    experiment parameters dictionary
    """
    values = load_optuna_values_file(exp_path)

    for key, values in values.items():
        # selecting value
        if values["val"][0] == values["val"][1]:
            val = values["val"][0]
        elif values["type"] == "int":
            val = trial.suggest_int(
                    name=key,
                    low=values["val"][0],
                    high=values["val"][1],
                    log=values["log"]
                )
        elif values["type"] == "float":
            val = trial.suggest_float(
                    name=key,
                    low=values["val"][0],
                    high=values["val"][1],
                    log=values["log"]
                )
        elif values["type"] == "cat_or_log":
            val = get_cat_or_log(
                    trial=trial,
                    min_val=values["val"][0],
                    max_val=values["val"][1],
                    categorical=values["categorical"],
                    name=key
                )
        else:
            raise NotImplementedError("ERROR")

        # logging
        func = print_ if verbose else log_info
        func(f"  --> Setting param {key}: {values['path']} with value {val}")

        # updating exp_params
        utils.set_in_dict(params=exp_params, key_list=values["path"], value=val)

    return exp_params


def get_cat_or_log(trial, min_val, max_val, categorical, name):
    """
    Sampling a value from a categorical distribution with values logarithmically distributed,
    or directly sampling from a log-distribution
    """
    if min_val == 0 and max_val == 0:
        value = 0
    elif categorical:
        min_ = torch.log10(torch.tensor(min_val))
        max_ = torch.log10(torch.tensor(max_val))
        steps = (max_ - min_ + 1).int().item()
        val_list = torch.logspace(min_, max_, steps=steps).tolist()
        val_list = val_list + [0]  # adding value 0
        value = trial.suggest_categorical(name, val_list)
    else:
        value = trial.suggest_float(name, min_val, max_val, log=True)
    return value


def log_optuna_stats(study):
    """ """
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print_("Study statistics: ")
    print_(f"    Number of finished trials: {len(study.trials)}")
    print_(f"    Number of complete trials: {len(complete_trials)}")
    print_("")

    trial = study.best_trial
    print_(f"Best trial: Trial #{trial.number}")
    print_(f"  Value: {trial.value}")
    print_("  Params: ")
    for key, value in trial.params.items():
        print_(f"    {key}: {value}")

    return

#
