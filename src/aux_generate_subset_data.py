"""
Visualizing some data and annotations to make sure that dataset, augmentations and
data loaders are working as expected
"""

import os
import torch
import numpy as np
import data

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.logger import Logger
from lib.utils import clear_cmd, create_directory
import lib.visualizations as visualizations


N_TEST_IMGS = 5
SAMPLES_PER_IMG = 5


def main():
    """
    Simple logic for loading experiment parameters, loading data, saving some examples
    """
    exp_path, _ = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting visualization of a subset of the test set", message_type="new_exp")

    # loading exp_params and creating directory to store plots
    cfg = Config()
    exp_params = cfg.load_exp_config_file(exp_path)
    plots_path = os.path.join(exp_path, "plots", "data_test_plots")
    create_directory(plots_path)

    # loading test set
    shuffle_test = exp_params["dataset"]["shuffle_eval"]
    test_set = data.load_data(exp_params=exp_params, split="train")

    for i in range(N_TEST_IMGS):
        idx = np.random.randint(low=0, high=len(test_set)) if shuffle_test else i
        imgs = torch.stack([test_set[idx][0] for _ in range(SAMPLES_PER_IMG)])
        classes = [test_set[idx][1] for _ in range(SAMPLES_PER_IMG)]
        visualizations.visualize_sequence(
                sequence=imgs,
                savepath=os.path.join(plots_path, f"test_img_{i+1}.png"),
                n_channels=1,
                n_cols=5,
                add_title=True,
                titles=[f"Class: {c}" for c in classes]
            )
    return


if __name__ == "__main__":
    clear_cmd()
    main()
