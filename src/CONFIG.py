"""
Global configurations
"""

import os

# High level configurations, such as paths or random seed
CONFIG = {
    "num_workers": 4,
    "random_seed": 13,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "..", "datasets"),
        "experiments_path": os.path.join(os.getcwd(), "experiments"),
        "configs_path": os.path.join(os.getcwd(), "src", "configs"),
        "base_path": os.path.join(os.getcwd(), "src", "base"),
    }
}


# Supported datasets, models, metrics, and so on
AUGMENTATIONS = ["mirror", "noise", "color_jitter", "rotate", "scale"]


# Specific configurations and default values
DEFAULTS = {
    "dataset": None,
    "model": {
        "model_name": None,
        "model_params": None
    },
    "loss": [
        {
            "type": "cross_entropy",
            "weight": 1
        },
    ],
    "metrics": {
        "metrics": "classification"
    },
    "training": {  # training related parameters
        "num_epochs": 100,      # number of epochs to train for
        "save_frequency": 5,    # saving a checkpoint after these eoochs ()
        "log_frequency": 100,   # logging stats after this amount of updates
        "batch_size": 32,
        "lr": 1e-3,
        "optimizer": "adam",     # optimizer parameters: name, L2-reg, momentum
        "momentum": 0,
        "weight_decay": 0,
        "nesterov": False,
        "scheduler": "exponential",  # learning rate scheduler parameters
        "lr_factor": 0.5,            # Meaning depends on scheduler. See lib/model_setup.py
        "patience": 5,
        "lr_warmup": True,       # learning rate warmup parameters (2 epochs or 2000 iters default)
        "warmup_steps": 2000,
        "warmup_epochs": 2,
        "early_stopping": True,
        "early_stopping_patience": 10
    }
}


# ranges for optuna hyper-param study
OPTUNA = {
    "lr": {   # training parameters
        "val": [1e-5, 1e-1],
        "type": "float",
        "categorical": False,
        "log": True,
        "path": ["training", "lr"]
    },
    "batch_size": {
        "val": [4, 64],
        "type": "int",
        "categorical": False,
        "log": False,
        "path": ["training", "batch_size"]
    },
    "lr_factor": {
        "val": [1e-2, 0.9],
        "type": "float",
        "categorical": False,
        "log": True,
        "path": ["training", "lr_factor"]
    },
    "patience": {
        "val": [1, 5],
        "type": "int",
        "categorical": False,
        "log": False,
        "path": ["training", "patience"]
    },
    "weight_decay": {
        "val": [0, 0],
        "type": "float",
        "categorical": False,
        "log": True,
        "path": ["training", "weight_decay"]
    },
    "lambda": {
        "val": [1, 1],
        "type": "cat_or_log",
        "categorical": False,
        "log": True,
        "path": ["loss", 0, "weight"]
    },
}
