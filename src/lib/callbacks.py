"""
Callbacks
"""

import os
import shutil
from CONFIG import CONFIG


def create_callbacks_file(dir_path):
    """
    Creating default callbacks file
    """
    shutil.copyfile(
            os.path.join(CONFIG["paths"]["configs_path"], "callbacks.py"),
            os.path.join(dir_path, "callbacks.py")
        )
    return


class Callback:
    """
    Base class from which all callbacks inherit
    """

    def __init__(self):
        """ Callback initializer """
        pass

    def __call__(self):
        """ """
        pass


class Callbacks:
    """
    Module for registering callbacks from the callbacks file, and orquestrating calls
    to such callbacks
    """

    def __init__(self):
        """ Module initalizer """
        self.callbacks = []
        return

    def initialize_callbacks(self, trainer):
        """ Initializing all callbacks """
        for callback in self.callbacks:
            callback(trainer)

    def on_epoch_start(self, trainer):
        """ Calling callbacks activated on epoch start """
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_epoch_end(self, trainer):
        """ Calling callbacks activated on epoch end """
        for callback in self.callbacks:
            callback.on_epoch_end(trainer)

    def on_batch_start(self, trainer):
        """ Calling callbacks activated on batch start """
        for callback in self.callbacks:
            callback.on_batch_start(trainer)

    def on_batch_end(self, trainer):
        """ Calling callbacks activated on batch end """
        for callback in self.callbacks:
            callback.on_batch_end(trainer)


#
