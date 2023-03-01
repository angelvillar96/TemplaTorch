"""
Base trainer from which all trainer classes inherit.
Basically it removes the scaffolding that is repeat across all training modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.loss import LossTracker
from lib.metrics import MetricTracker
from lib.schedulers import WarmupVSScehdule, EarlyStop
from lib.setup_model import emergency_save
from models.model_utils import GradientInspector
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class BaseTrainer:
    """
    Base Class for training and validating a model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    checkpoint: string/None
        Name of a model checkpoint stored in the models/ directory of the experiment directory.
        If given, the model is initialized with the parameters of such checkpoint.
        This can be used to continue training or for transfer learning.
    resume_training: bool
        If True, saved checkpoint states from the optimizer, scheduler, ... are restored
        in order to continue training from the checkpoint
    """

    def __init__(self, exp_path, checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config()
        self.exp_params = self.cfg.load_exp_config_file(exp_path)
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        # setting paths and creating subdirectories
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

        # initializing datasets and data loaders
        train_set = data.load_data(exp_params=self.exp_params, split="train")
        valid_set = data.load_data(exp_params=self.exp_params, split="valid")
        self.train_loader = data.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        print_(f"  --> Length training set: {len(train_set)}")
        print_(f"  --> Num. Batches train: {len(self.train_loader)}")
        self.valid_loader = data.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        print_(f"  --> Length validation set: {len(valid_set)}")
        print_(f"  --> Num. Batches validation: {len(self.valid_loader)}")
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
        optimizer, scheduler, lr_warmup = setup_model.setup_optimization(
                exp_params=self.exp_params,
                model=model
            )
        epoch = 0

        # loading pretrained model and other necessary objects for resuming training or fine-tuning
        if self.checkpoint is not None:
            print_(f"  --> Loading pretrained parameters from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                model, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"  --> Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects

        # training and evaluation objects
        self.model = model
        self.epoch = epoch
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        self.metric_tracker = MetricTracker(metrics=self.exp_params["metrics"]["metrics"])
        self.warmup_scheduler = WarmupVSScehdule(
                optimizer=self.optimizer,
                lr_warmup=lr_warmup,
                scheduler=scheduler
            )

        self.GradientInspector = GradientInspector(
                writer=self.writer,
                stats=["Max", "Mean", "Var", "Norm"],
                layers=[],
                names=[]
            )
        return

    @emergency_save
    def training_loop(self):
        """
        Repearting the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training"]["num_epochs"]
        save_frequency = self.exp_params["training"]["save_frequency"]

        # iterating for the desired number of epochs
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.eval()
            self.valid_epoch(epoch)
            self.train()
            self.train_epoch(epoch)
            stop_training = self.early_stopping(
                    value=self.validation_losses[-1],
                    writer=self.writer,
                    epoch=epoch
                )
            if stop_training:
                break

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss/CE_comb_loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )

            # updating learning rate scheduler or lr-warmup
            self.warmup_scheduler(
                    iter=-1,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=True,
                    control_metric=self.validation_losses[-1]
                )

            # saving backup model checkpoint and (if reached saving frequency) epoch checkpoint
            setup_model.save_checkpoint(  # Gets overriden every epoch: checkpoint_last_saved.pth
                    trainer=self,
                    savedir="models",
                    savename="checkpoint_last_saved.pth"
                )
            if(epoch % save_frequency == 0 and epoch != 0):  # checkpoint_epoch_xx.pth
                print_("Saving model checkpoint")
                setup_model.save_checkpoint(
                        trainer=self,
                        savedir="models"
                    )

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        setup_model.save_checkpoint(
                trainer=self,
                savedir="models",
                finished=not stop_training
            )
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        self.loss_tracker.reset()
        self.metric_tracker.reset()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, (imgs, targets) in progress_bar:
            iter_ = len(self.train_loader) * epoch + i

            # updating learning rate scheduler or lr-warmup
            self.warmup_scheduler(
                    iter=iter_,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=False,
                    control_metric=self.validation_losses[-1]
                )

            # forward pass
            _, loss = self.forward_loss_metric(
                    imgs=imgs,
                    targets=targets,
                    training=True
                )

            # logging and plots
            if(iter_ % self.exp_params["training"]["log_frequency"] == 0):
                self.writer.log_full_dictionary(
                        dict=self.loss_tracker.get_last_losses(),
                        step=iter_,
                        plot_name="Train Loss",
                        dir="Train Loss Iter",
                    )
                self.writer.add_scalar(
                        name="Learning/Learning Rate",
                        val=self.optimizer.param_groups[0]['lr'],
                        step=iter_
                    )

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        # logging loss to tensorboard
        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Train Loss",
                dir="Train Loss",
            )
        self.training_losses.append(average_loss_vals["_total"].item())

        # logging metrics to tensorboard
        self.metric_tracker.aggregate()
        metrics = self.metric_tracker.summary(verbose=False)
        self.writer.log_full_dictionary(
                dict=metrics,
                step=epoch + 1,
                plot_name="Train Metrics",
                dir="Train Metrics",
            )
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        self.loss_tracker.reset()
        self.metric_tracker.reset()
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        for i, (imgs, targets) in progress_bar:

            # forward pass
            _, loss = self.forward_loss_metric(
                    imgs=imgs,
                    targets=targets,
                    training=False
                )
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

        # logging valid loss to tensorboard
        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Valid Loss",
                dir="Valid Loss",
            )
        self.validation_losses.append(average_loss_vals["_total"].item())

        # logging metrics to tensorboard
        self.metric_tracker.aggregate()
        metrics = self.metric_tracker.summary(verbose=False)
        self.writer.log_full_dictionary(
                dict=metrics,
                step=epoch + 1,
                plot_name="Valid Metrics",
                dir="Valid Metrics",
            )
        return

    def train(self):
        """ Setting all the components to training mode """
        self.model.train()

    def eval(self):
        """ Setting all the components to evaluation mode """
        self.model.eval()

#
