"""
Computation of different metrics
"""

import os
import json
import torch
from lib.utils import create_directory
from CONFIG import METRICS


class MetricTracker:
    """
    Class for computing several evaluation metrics
    """

    def __init__(self, metrics=["accuracy"]):
        """ Module initializer """
        assert isinstance(metrics, list), f"Metrics argument must be a list, and not {type(metrics)}"
        for metric in metrics:
            if metric not in METRICS:
                raise NotImplementedError(f"Metric {metric} not implemented. Use one of {METRICS}")

        self.metric_computers = {metric: self._get_metric(metric) for m in metrics}
        self.reset_results()
        return

    def reset_results(self):
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

    def summary(self, get_results=True):
        """ Printing and fetching the results """
        print("RESULTS:")
        print("--------")
        for metric in self.metric_computers.keys():
            print(f"  {metric}:  {round(self.results[metric], 3)}")
        return self.results

    def save_results(self, exp_path, checkpoint_name):
        """ Storing results into JSON file """
        checkpoint_name = checkpoint_name.split(".")[0]
        results_file = os.path.join(exp_path, "results", f"{checkpoint_name}.json")
        create_directory(dir_path=exp_path, dir_name="results")

        with open(results_file, "w") as file:
            json.dump(self.results, file)
        return

    def _get_metric(self, metric):
        """ """
        if metric == "accuracy":
            metric_computer = Accuracy()
        else:
            raise NotImplementedError(f"Unknown metric {metric}. Use one of {METRICS} ...")
        return metric_computer


class Metric:
    """
    Base class for metrics
    """

    def __init__(self):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")


class Accuracy(Metric):
    """ Accuracy computer """

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
        cur_correct = len(torch.where(preds == targets)[0])
        cur_total = len(preds)
        self.correct += cur_correct
        self.total += cur_total

    def aggregate(self):
        """ Computing average accuracy """
        accuracy = self.correct / self.total
        return accuracy


#
