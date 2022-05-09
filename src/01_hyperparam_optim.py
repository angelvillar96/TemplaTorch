"""
Optuna hyper-parameter optimization
"""
import os
from copy import deepcopy
import numpy as np
import optuna

from lib.arguments import get_optuna_arguments
from lib.config import Config
from lib.logger import Logger, print_
import lib.optuna as optuna_utils
import lib.utils as utils
TRAINER = __import__('02_train')


def objective(trial, optuna_path, logger, num_epochs=5):
    """ Objective to optimize in an Optuna hyper-param optimization study """
    trial_number = trial.number
    logger.log_info(f"Starting Trial {trial_number}")
    cfg = Config(optuna_path)
    general_params = cfg.load_exp_config_file()

    # creating exp_directory
    exp_path = os.path.join(optuna_path, f"trial_{trial_number}")
    utils.create_directory(exp_path)
    cfg.create_exp_config_file(exp_path=exp_path)
    exp_params = cfg.load_exp_config_file(exp_path=exp_path)
    exp_params = deepcopy(general_params)
    exp_params["training"]["num_epochs"] = num_epochs
    exp_params["training"]["log_frequency"] = 1e9  # to avoid logging all the time if b_size is small

    # adjusting values
    print_("Obtaining values for current trial")
    exp_params = optuna_utils.suggest_values(trial, optuna_path, exp_params, verbose=False)
    cfg.save_exp_config_file(exp_path=exp_path, exp_params=exp_params)

    # training current model
    try:
        trainer = TRAINER.Trainer(exp_path=exp_path)
        trainer.load_data()
        trainer.setup_model()
        trainer.training_loop()

        # final validation
        trainer.model.eval()
        trainer.valid_epoch(num_epochs)
        metric = np.min(trainer.validation_losses)
    except:
        return 1e8

    return metric


def optuna_optimization(exp_path, num_epochs=5, num_trials=50):
    """
    Performing an Optuna hyper-param optimization study

    Args:
    -----
    exp_path: string
        directory where the study will be performed
    """
    logger = Logger(exp_path=exp_path)
    values_file = os.path.join(exp_path, "optuna_values.json")
    if(not os.path.exists(values_file)):
        print("Creating Optuna_Values.json and basic exp_params.json in experiment directory")
        print("  Modify such files with the desired parameters and hyper-parameter ranges...")
        print("  Re-run then script to start the study...")
        optuna_utils.create_optuna_values_file(exp_path)
        cfg = Config(exp_path=exp_path)
        cfg.create_exp_config_file()
        return

    storage_path = os.path.join(exp_path, "optuna.db")
    if(os.path.exists(storage_path)):
        message = "An Optuna study already exists. Are you sure you want to continue?..."
        utils.press_yes_to_continue(message)

    logger.log_info("Starting Hyper Parameter optimization procedure", message_type="new_exp")
    logger.log_info("Starting Hyper Parameter optimization procedure", message_type="new_exp")
    logger.log_info(f"   Number of Trials: {num_trials}")
    logger.log_info(f"   Number of Epochs: {num_epochs}")

    study = optuna.create_study(
            direction="minimize",
            study_name="optuna_study",
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True
        )
    func = lambda trial: objective(trial, optuna_path=exp_path, logger=logger, num_epochs=num_epochs)
    study.optimize(func, n_trials=num_trials)

    # logging stats of the best trial
    optuna_utils.log_optuna_stats(study)

    # Saving study and best trial
    study_path = os.path.join(exp_path, "optuna_study.pkl")
    best_trial_path = os.path.join(exp_path, "best_trial.pkl")
    utils.save_pickle_file(study_path, study)
    utils.save_pickle_file(best_trial_path, study.best_trial.params)
    return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_directory, num_epochs, num_trials = get_optuna_arguments()
    optuna_optimization(
            exp_path=exp_directory,
            num_epochs=num_epochs,
            num_trials=num_trials
        )



#
