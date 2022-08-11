"""
Computation of different metrics
"""

import os
import json
import torch
from lib.utils import create_directory
from CONFIG import METRICS, METRIC_SETS


class MetricTracker:
    """
    Class for computing several evaluation metrics
    """

    def __init__(self, metrics=["accuracy"]):
        """ Module initializer """
        assert isinstance(metrics, (list, str)), f"Metrics argument must be string/list, not {type(metrics)}"
        if isinstance(metrics, str):
            if metrics not in METRIC_SETS.keys():
                raise ValueError(f"If str, metrics must be one of {METRIC_SETS.keys()}, not {metrics}")
        else:
            for metric in metrics:
                if metric not in METRICS:
                    raise NotImplementedError(f"Metric {metric} not implemented. Use one of {METRICS}")

        metrics = METRIC_SETS[metrics] if isinstance(metrics, str) else metrics
        self.metric_computers = {m: get_metric(m) for m in metrics}
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


def get_metric(self, metric):
    """ """
    if metric == "accuracy":
        metric_computer = Accuracy()
    elif metric == "mse":
        metric_computer = MSE()
    else:
        raise NotImplementedError(f"Unknown metric {metric}. Use one of {METRICS} ...")
    return metric_computer


class Metric:
    """ Base class for metrics """

    def __init__(self):
        """ Metric initializer """
        self.values = []
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        self.values = []

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_value = all_values.mean()
        return mean_value

    def _shape_check(self, x, name="Preds"):
        """
        Checking the shape of the imputs. It must be one of the following:
        """
        if len(x.shape) not in [3, 4, 5]:
            raise ValueError(f"{name} has shape {x.shape}, but it must have one of the folling shapes\n"
                             " - (B, F, C, H, W) for frame or heatmap prediction.\n"
                             " - (B, F, D) or (B, F, N_joints, N_coords) for pose skeleton prediction")


class Accuracy(Metric):
    """ Accuracy computer. Completely overrides the base metric """

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
        if len(preds.shape) == 2:
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

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if len(preds.shape) == 5 and len(targets.shape) == 5:
            cur_mse = (preds - targets).pow(2).mean(dim=(-1, -2, -3))
        elif len(preds.shape) == 3 and len(targets.shape) == 3:
            cur_mse = (preds - targets).pow(2).mean(dim=-1)
        self.values.append(cur_mse)
        return


#
