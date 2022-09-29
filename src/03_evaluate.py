"""
Evaluating a model checkpoint
"""

import torch

from base.baseEvaluator import BaseEvaluator
from lib.arguments import get_directory_argument
from lib.logger import Logger, print_
import lib.utils as utils


class Evaluator(BaseEvaluator):
    """ Class for evaluating a model """

    @torch.no_grad()
    def forward_eval(self, imgs, targets, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        imgs: torch Tensor
            Images to feed as input to the model
        targets: torch Tensor
            Targets that the model should predict
        """
        imgs, targets = imgs.to(self.device), targets.to(self.device)
        preds = self.model(imgs)
        self.metric_tracker.accumulate(preds=preds, targets=targets)
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
