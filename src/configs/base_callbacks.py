"""
Default callbacks file
"""

from lib.callbacks import Callback
from lib.schedulers import EarlyStop, WarmupVSScehdule
from lib.setup_model import save_checkpoint


class SaveCheckpoint(Callback):
    """
    Saving checkpoint callback
    """

    def on_epoch_end(self, trainer):
        """ Saving emergency checkpoint at the end of every epoch """
        save_checkpoint(
            trainer=trainer,
            savedir="models",
            savename="checkpoint_last_saved.pth"
        )


class EarlyStoppingCallback(Callback):
    """
    Early stopping callback
    """

    def __init__(self, trainer):
        """ Callback initializer """
        self.early_stopping = EarlyStop(
            mode="min",
            use_early_stop=trainer.exp_params["training"]["early_stopping"],
            patience=trainer.exp_params["training"]["early_stopping_patience"]
        )

    def on_epoch_end(self, trainer):
        """ Calling early stopping on epoch end """
        stop_training = self.early_stopping(
                value=trainer.validation_losses[-1],
                writer=trainer.writer,
                epoch=trainer.epoch
            )
        return {"stop_training": stop_training}


class WarmupScheduleCallback(Callback):
    """
    Warmup vs Schedule callback
    """

    def __init__(self, trainer):
        """ Callback initializer """
        self.warmup_scheduler = WarmupVSScehdule(
                optimizer=trainer.optimizer,
                lr_warmup=trainer.lr_warmup,
                scheduler=trainer.scheduler
            )

    def on_batch_start(self, trainer):
        """ """
        self.warmup_scheduler(
                iter=trainer.iter_,
                epoch=trainer.epoch,
                exp_params=trainer.exp_params,
                end_epoch=False,
                control_metric=trainer.validation_losses[-1]
            )

    def on_epoch_end(self, trainer):
        """ Updating at the end of every epoch """
        self.warmup_scheduler(
                iter=-1,
                epoch=trainer.epoch,
                exp_params=trainer.exp_params,
                end_epoch=True,
                control_metric=trainer.validation_losses[-1]
            )
