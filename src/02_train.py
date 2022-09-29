"""
Training and Validation a model
"""

from base.baseTrainer import BaseTrainer
from lib.arguments import get_directory_argument
from lib.logger import Logger
import lib.utils as utils


class Trainer(BaseTrainer):
    """
    Class for training and validating a model
    """

    def forward_loss_metric(self, imgs, targets, training=False):
        """
        Computing a forwad pass through the model, and (if necessary) the loss values and metrics

        Args:
        -----
        imgs: torch Tensor
            Images to feed as input to the model
        targets: torch Tensor
            Targets that the model should predict
        training: bool
            If True, model is in training mode

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        # forward pass
        imgs, targets = imgs.to(self.device), targets.to(self.device)
        preds = self.model(imgs)

        # loss and optimization
        self.loss_tracker(preds=preds, targets=targets)
        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # computing metrics
        self.metric_tracker.accumulate(preds=preds, targets=targets)
        return preds, loss


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting training procedure", message_type="new_exp")
    logger.log_git_hash()

    print("Initializing Trainer...")
    trainer = Trainer(exp_path=exp_path, checkpoint=args.checkpoint, resume_training=args.resume_training)
    print("Loading dataset...")
    trainer.load_data()
    print("Setting up model and optimizer")
    trainer.setup_model()
    print("Starting to train")
    trainer.training_loop()


#
