"""
Template for custom datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SampleDataset(Dataset):
    """
    Sample dataset
    """

    def __init__(self):
        """ Dataset Initializer """
        self.data = []
        self.targets = []

    def __len__(self):
        """ Obtaining number of examples in the dataset """
        return len(self.data)

    def __getitem__(self, i):
        """ Fetching example i from the dataset """
        image = self.data[i]
        label = self.targets[i]
        return image, label
