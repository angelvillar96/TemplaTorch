"""
Callbacks
"""

import os
import sys
import inspect
import shutil

from lib.logger import print_
from CONFIG import CONFIG


def create_callbacks_file(dir_path):
    """
    Creating default callbacks file
    """
    shutil.copyfile(
            os.path.join(CONFIG["paths"]["configs_path"], "base_callbacks.py"),
            os.path.join(dir_path, "callbacks.py")
        )
    return


class Callback:
    """
    Base class from which all callbacks inherit
    """

    def __init__(self, trainer):
        """ Callback initializer """
        pass

    def on_epoch_start(self, trainer):
        """ Called at the beginning of every epoch """
        pass

    def on_epoch_end(self, trainer):
        """ Called at the end of every epoch """
        pass

    def on_batch_start(self, trainer):
        """ Called at the beginning of every batch """
        pass

    def on_batch_end(self, trainer):
        """ Called at the end of every batch """
        pass

    def on_log_frequency(self, trainer):
        """ Called at the end of every batch if current iteration is multiple of log frequency """
        pass


class Callbacks:
    """
    Module for registering callbacks from the callbacks file, and orquestrating calls
    to such callbacks
    """

    def __init__(self, trainer=None):
        """ Module initalizer """
        self.callbacks = []
        if trainer is not None:
            self.register_callbacks(trainer=trainer)
        return

    def register_callbacks(self, trainer):
        """ Loading callbacks from the callbacks file """
        sys.path.append(trainer.exp_path)
        import callbacks
        for name, cls in inspect.getmembers(callbacks, inspect.isclass):
            if not issubclass(cls, Callback) or cls is Callback:
                continue
            print_(f"Registering callback {name}: {cls}")
            self.callbacks.append(cls)

    def initialize_callbacks(self, trainer):
        """ Initializing all callbacks """
        initialized_callbacks = []
        for callback in self.callbacks:
            initialized_callbacks.append(
                    callback(trainer=trainer)
                )
        self.callbacks = initialized_callbacks
        return

    def on_epoch_start(self, trainer):
        """ Calling callbacks activated on epoch start """
        for callback in self.callbacks:
            callback.on_epoch_end(trainer=trainer)

    def on_epoch_end(self, trainer):
        """ Calling callbacks activated on epoch end """
        return_data = {}
        for callback in self.callbacks:
            out = callback.on_epoch_end(trainer=trainer)
            if out is not None:
                return_data = {**return_data, **out}
        return return_data

    def on_batch_start(self, trainer):
        """ Calling callbacks activated on batch start """
        for callback in self.callbacks:
            callback.on_batch_start(trainer=trainer)

    def on_batch_end(self, trainer):
        """ Calling callbacks activated on batch end """
        for callback in self.callbacks:
            callback.on_batch_end(trainer=trainer)

    def on_log_frequency(self, trainer):
        """ Calling callbacks activated on batch end """
        if(trainer.iter_ % trainer.exp_params["training"]["log_frequency"] == 0):
            for callback in self.callbacks:
                callback.on_log_frequency(trainer=trainer)


#
