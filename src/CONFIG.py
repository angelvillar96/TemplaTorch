"""
Global configurations
"""

import os

# High level configurations, such as paths or random seed
CONFIG = {
    "random_seed": 13,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "..", "datasets"),
        "experiments_path": os.path.join(os.getcwd(), "experiments"),
        "configs_path": os.path.join(os.getcwd(), "src", "configs"),
    }
}


# Supported datasets, models, metrics, and so on
DATASETS = ["mnist"]
LOSSES = ["mse", "l2", "mae", "l1", "cross_entropy", "ce"]
METRICS = ["accuracy"]
MODELS = ["ConvNet"]


# Specific configurations and default values
DEFAULTS = {
    "dataset": {
        "dataset_name": "mnist",
        "shuffle_train": True,
        "shuffle_eval": False
    },
    "model": {
        "model_name": "ConvNet",
        "ConvNet": {
            "channels": [1, 32, 64, 128],
            "out_size": 10
        }
    },
    "loss": [
        {
            "type": "cross_entropy",
            "weight": 1
        },
    ],
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
        "lr_warmup": True,       # learning rate warmup parameters (2 epochs or 200 iters default)
        "warmup_steps": 2000,
        "warmup_epochs": 2
    }
}
