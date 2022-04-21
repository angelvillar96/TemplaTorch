"""
Methods for loading specific datasets, fitting data loaders and other
"""

DATASETS = ["mnist"]

def load_data(exp_params, split="train", transform=None):
    """
    Loading a dataset given the parameters

    Args:
    -----
    exp_params: dictionary
        dict with the experiment specific parameters
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')
    transform: Torch Transforms
        Compose of torchvision transforms to apply to the data

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    in_channels: integer
        number of channels in the dataset samples (e.g. 1 for B&W, 3 for RGB)
    """

    dataset_name = exp_params["dataset"]["dataset_name"]

    if(dataset_name == "mnist"):
        dataset = datasets.MNIST(
                root=CONFIG["paths"]["data_path"],
                train=train,
                download=True,
                transform=transform
            )
        in_channels = 3
    else:
        raise NotImplementedError(f"""ERROR! Dataset'{dataset_name}' is not available.
            Please use one of the following: {DATASETS}...""")

    return dataset, in_channels


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
            shuffle=shuffle
        )

    return data_loader
