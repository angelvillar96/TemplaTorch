"""
Evaluating a model checkpoint
"""

import os
from tqdm import tqdm
import torch

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class Evaluator:
    """ Class for evaluating a model """

    def __init__(self, exp_path, checkpoint):
        """ Initializing the evaluator object """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint

        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        self.metric_tracker = MetricTracker(metrics=self.exp_params["metrics"]["metrics"])
        return

    def load_data(self):
        """ Loading dataset and fitting data-loader for iterating in a batch-like fashion """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        test_set = data.load_data(exp_params=self.exp_params, split="test")
        self.test_loader = data.build_data_loader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        print_(f"  --> Length test set: {len(test_set)}")
        print_(f"  --> Num. Batches test: {len(self.test_loader)}")
        return

    def setup_model(self):
        """ Initializing model and loading pretrained paramters """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(self.model, exp_path=self.exp_path)
        self.model = self.model.eval().to(self.device)

        checkpoint_path = os.path.join(self.models_path, self.checkpoint)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        self.model = self.model.eval()
        return

    @torch.no_grad()
    def evaluate(self):
        """ Evaluating model epoch loop """
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))

        # iterating test set and accumulating the results
        for i, (imgs, targets) in progress_bar:
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            preds = self.model(imgs)
            self.metric_tracker.accumulate(preds=preds, targets=targets)

            progress_bar.set_description(f"Iter {i}/{len(self.test_loader)}")

        self.metric_tracker.aggregate()
        _ = self.metric_tracker.summary()
        self.metric_tracker.save_results(exp_path=self.exp_path, checkpoint_name=self.checkpoint)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting evaluation procedure", message_type="new_exp")
    logger.log_git_hash()

    print_("Initializing Evaluator...")
    evaluator = Evaluator(exp_path=exp_path, checkpoint=args.checkpoint)
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and loading pretrained parameters")
    evaluator.setup_model()
    print_("Starting evaluation")
    evaluator.evaluate()


#
