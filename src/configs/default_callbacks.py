"""
Default callbacks file
"""

from lib.callbacks import Callback
from lib.schedulers import EarlyStop


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

    def __call__(self):
        """
        """
