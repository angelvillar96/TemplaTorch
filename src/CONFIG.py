"""
Global configurations
"""

import os

CONFIG = {
    "random_seed": 13,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "..", "..", "datasets"),
        "experiments_path": os.path.join(os.getcwd(), "..", "experiments"),
        }
}


DEFAULTS = {
    "dataset": {
        "dataset_name": "mnist",
        "shuffle_train": True,
        "shuffle_eval": False
    },
    "model": {
        "model_name": "sample_model"
    },
    "loss": {
        "loss_type": "cross_entropy"
    },
    "training": {  # training related parameters
        "num_epochs": 100,      # number of epochs to train for
        "save_frequency": 5,    # saving a checkpoint after these eoochs ()
        "log_frequency": 100,   # logging stats after this amount of updates
        "batch_size": 32,
        "lr": 1e-3,
        "lr_factor": 1,
        "optimizer": "adam",     # optimizer parameters: name, L2-reg, momentum
        "momentum": 0,
        "weight_decay": 0,
        "nesterov": False,
        "scheduler": "exponential", # learning rate scheduler parameters
        "lr_factor": 0.5,           # Meaning depends on scheduler. See lib/model_setup.py
        "patience": 5,
        "lr_warmup": True,       # learning rate warmup parameters (2 epochs or 200 iters default)
        "warmup_steps": 2000,
        "warmup_epochs": 2
    }
}
