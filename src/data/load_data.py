"""
Methods for loading specific datasets, fitting data loaders and other
"""

from torch.utils.data import DataLoader

from lib.augmentations import Augmentator
from data import DATASET_MAP
from CONFIG import CONFIG


def load_data(exp_params, split="train"):
    """
    Loading a dataset given the parameters

    Args:
    -----
    exp_params: dictionary
        dict with the experiment specific parameters
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    in_channels: integer
        number of channels in the dataset samples (e.g. 1 for B&W, 3 for RGB)
    """
    DATASETS = DATASET_MAP.keys()
    dataset_params = exp_params["dataset"]
    dataset_name = dataset_params["dataset_name"]
    if dataset_name not in DATASETS:
        raise NotImplementedError(f"Dataset'{dataset_name}' is not available. Use one of {DATASETS}...")

    # setting up augmentations
    augmentator = Augmentator(augment_params=exp_params["dataset"]["augment_params"])
    if split == "train":
        augmentator.train()
    else:
        augmentator.eval()

    # instanciating dataset
    dataset = DATASET_MAP[dataset_name](
        root=CONFIG["paths"]["data_path"],
        split=split,
        augmentator=augmentator,
        **dataset_params
    )

    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"]
        )

    return data_loader
