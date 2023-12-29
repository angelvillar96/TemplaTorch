"""
Accessing datasets from script
"""

from .SampleDataset import SampleDataset

DATASET_MAP = {
    "SampleDataset": SampleDataset
}

from .load_data import load_data, build_data_loader