"""
Training and Validation a model
"""

import os
from tqdm import tqdm
import torch

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.logger import Logger, print_, log_function, for_all_methods, log_info
from lib.loss import LossTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class Trainer:
    """
    Class for training and validating a model
    """

    def __init__(self, exp_path, checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = self.exp_params["training"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        train_set = data.load_data(exp_params=self.exp_params, split="train")
        valid_set = data.load_data(exp_params=self.exp_params, split="valid")
        self.train_loader = data.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.valid_loader = data.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)

        # loading optimizer, scheduler and loss
        optimizer, scheduler = setup_model.setup_optimizer(exp_params=self.exp_params, model=model)
        loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        epoch = 0

        # loading pretrained model and other necessary objects for resuming training or fine-tuning
        if self.checkpoint is not None:
            print_(f"Loading pretrained parameters from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
            if self.resume_training:
                model, optimizer, scheduler, epoch = loaded_objects
                print_(f"Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects

        self.model = model
        self.optimizer, self.scheduler, self.epoch = optimizer, scheduler, epoch
        self.loss_tracker = loss_tracker
        return

    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss/CE_comb_loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )

            # updating learning rate scheduler if loss increases or plateaus
            setup_model.update_scheduler(
                    scheduler=self.scheduler,
                    exp_params=self.exp_params,
                    control_metric=self.validation_losses[-1]
                )

            # saving model checkpoint if reached saving frequency
            if(epoch % save_frequency == 0 and epoch != 0):
                print_("Saving model checkpoint")
                setup_model.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        exp_path=self.exp_path,
                        savedir="models"
                    )

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        setup_model.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir="models",
                finished=True
            )
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        self.loss_tracker.reset()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, (imgs, targets) in progress_bar:
            iter_ = len(self.train_loader) * epoch + i
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            preds = self.model(imgs)

            self.loss_tracker(preds=preds, targets=targets)
            loss = self.loss_tracker.get_last_losses(total_only=True)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                self.writer.log_full_dictionary(
                        dict=self.loss_tracker.get_last_losses(),
                        step=iter_,
                        plot_name="Train Loss",
                        dir="Train Loss Iter",
                    )
                self.writer.add_scalar(
                        name="Learning Rate",
                        val=self.optimizer.param_groups[0]['lr'],
                        step=iter_
                    )

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Train Loss",
                dir="Train Loss",
            )
        self.training_losses.append(average_loss_vals["_total"].item())
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        self.loss_tracker.reset()
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        for i, (imgs, targets) in progress_bar:
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            preds = self.model(imgs)
            self.loss_tracker(preds=preds, targets=targets)
            loss = self.loss_tracker.get_last_losses(total_only=True)
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Valid Loss",
                dir="Valid Loss",
            )
        self.validation_losses.append(average_loss_vals["_total"].item())
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting training procedure", message_type="new_exp")

    print("Initializing Trainer...")
    trainer = Trainer(exp_path=exp_path, checkpoint=args.checkpoint, resume_training=args.resume_training)
    print("Loading dataset...")
    trainer.load_data()
    print("Setting up model and optimizer")
    trainer.setup_model()
    print("Starting to train")
    trainer.training_loop()


#
