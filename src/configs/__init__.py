""" Configs """

import os
import json
from CONFIG import CONFIG


def get_available_configs():
    """
    Getting a list with the name of the available config files
    """
    config_path = CONFIG["paths"]["configs_path"]
    files = sorted(os.listdir(config_path))
    available_configs = [f[:-5] for f in files if f[-5:] == ".json"]
    return available_configs


def get_available_datasets_configs():
    """
    Getting a list with the available dataset configs
    """
    config_path = os.path.join(CONFIG["paths"]["configs_path"], "datasets")
    files = sorted(os.listdir(config_path))
    available_configs = [f[:-5] for f in files if f[-5:] == ".json"]
    return available_configs


def get_available_model_configs():
    """
    Getting a list with the available model configs
    """
    config_path = os.path.join(CONFIG["paths"]["configs_path"], "models")
    files = sorted(os.listdir(config_path))
    available_configs = [f[:-5] for f in files if f[-5:] == ".json"]
    return available_configs


def get_dataset_config(dataset_name):
    """
    Fetching a dataset config (i.e. a dictionary) given the dataset name
    """
    all_db_configs = get_available_datasets_configs()
    if dataset_name not in all_db_configs:
        raise ValueError(f"No dataset-config for dataset {dataset_name} in {all_db_configs}...")

    print(f"  --> Loading dataset parameters for dataset {dataset_name}")
    config_path = os.path.join(CONFIG["paths"]["configs_path"], "datasets", f"{dataset_name}.json")
    with open(config_path) as f:
        dataset_params = json.load(f)
    dataset_config = {"dataset_name": dataset_name, **dataset_params}
    return dataset_config


def get_model_config(model_name):
    """
    Fetching a model config (i.e. a dictionary) given the model name
    """
    all_model_configs = get_available_model_configs()
    if model_name not in all_model_configs:
        raise ValueError(f"No model-config for model {model_name} in {all_model_configs}...")

    print(f"  --> Loading model parameters for model {model_name}")
    config_path = os.path.join(CONFIG["paths"]["configs_path"], "models", f"{model_name}.json")
    with open(config_path) as f:
        model_params = json.load(f)
    model_config = {"model_name": model_name, "model_params": model_params}
    return model_config


#
