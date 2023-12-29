"""
Computation of different metrics
"""

import os
import json
import torch
from lib.logger import print_, log_info
from lib.utils import create_directory


class MetricTracker:
    """
    Class for computing several evaluation metrics
    """

    def __init__(self, metrics=["accuracy"]):
        """ Module initializer """
        assert isinstance(metrics, (list, str)), f"Metrics argument must be string/list, not {type(metrics)}"
        METRICS = METRICS_DICT.keys()
        if isinstance(metrics, str):
            if metrics not in METRIC_SETS.keys():
                raise ValueError(f"If str, metrics must be one of {METRIC_SETS.keys()}, not {metrics}")
        else:
            for metric in metrics:
                if metric not in METRICS:
                    raise NotImplementedError(f"Metric {metric} not implemented. Use one of {METRICS}")

        metrics = METRIC_SETS[metrics] if isinstance(metrics, str) else metrics
        
        self.metric_computers = {}
        print_("Using evaluation metrics:")
        for metric in metrics:
            print_(f"  --> {metric}")
            if metric not in METRICS_DICT.keys():
                raise NotImplementedError(f"Unknown metric {metric}. Use one of {METRICS_DICT.keys()} ...")
            self.metric_computers[metric] = METRICS_DICT[metric]()
        self.reset()
        return

    def reset(self):
        """ Reseting results and metric computers """
        self.results = {m: None for m in self.metric_computers.keys()}
        for m in self.metric_computers.values():
            m.reset()
        return

    def accumulate(self, preds, targets):
        """ Computing the different metrics and adding them to the results list """
        for _, metric_computer in self.metric_computers.items():
            metric_computer.accumulate(preds=preds, targets=targets)
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        for metric, metric_computer in self.metric_computers.items():
            self.results[metric] = metric_computer.aggregate()
        return

    def summary(self, get_results=True, verbose=True):
        """ Printing and fetching the results """
        fprint = print_ if verbose is True else log_info
        fprint("RESULTS:")
        fprint("--------")
        for metric in self.metric_computers.keys():
            fprint(f"  {metric}:  {round(self.results[metric], 3)}")
        return self.results

    def save_results(self, exp_path, checkpoint_name):
        """ Storing results into JSON file """
        checkpoint_name = checkpoint_name.split(".")[0]
        results_file = os.path.join(exp_path, "results", f"{checkpoint_name}.json")
        create_directory(dir_path=exp_path, dir_name="results")

        with open(results_file, "w") as file:
            json.dump(self.results, file)
        return


######################
### METRIC CLASSES ###
######################



class Metric:
    """ Base class for metrics """

    def __init__(self):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        raise NotImplementedError("Base class does not implement 'reset' functionality")

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'aggregate' functionality")


class Accuracy(Metric):
    """ Accuracy computer """

    LOWER_BETTER = False

    def __init__(self):
        """ """
        self.correct = 0
        self.total = 0
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.correct = 0
        self.total = 0

    def accumulate(self, preds, targets):
        """ Computing metric """
        preds = torch.argmax(preds, dim=-1)
        cur_correct = len(torch.where(preds == targets)[0])
        cur_total = len(preds)
        self.correct += cur_correct
        self.total += cur_total

    def aggregate(self):
        """ Computing average accuracy """
        accuracy = self.correct / self.total
        return accuracy


class MSE(Metric):
    """ Mean Squared Error computer """

    LOWER_BETTER = True

    def __init__(self):
        """ """
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        cur_mse = (preds.float() - targets.float()).pow(2).mean()
        self.values.append(cur_mse)
        return cur_mse

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean().item()
        return mean_values


###################
### METRICS MAP ###
###################


METRIC_SETS = {
    "classification": ["accuracy"],
    "regression": ["mse"]
}

METRICS_DICT = {
    "accuracy": Accuracy,
    "mse": MSE,
}

#
