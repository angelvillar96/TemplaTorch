"""
Template for custom datasets
"""

import torchvision
from torch.utils.data import Dataset
from CONFIG import CONFIG


class SampleDataset(Dataset):
    """
    Sample dataset that instanciates the MNIST dataset
    """

    def __init__(self, root, split="train", augmentator=None, **kwargs):
        """ Dataset Initializer """
        self.root = root
        self.split = split
        self.augmentator = augmentator

        self.db = torchvision.datasets.MNIST(
                root=CONFIG["paths"]["data_path"],
                train=split == "train",
                download=True,
                transform=torchvision.transforms.ToTensor()
            )

        # setting augmentations on training or evaluation model
        if self.split == "train":
            augmentator.train()
        else:
            augmentator.eval()
        return

    def __len__(self):
        """ Obtaining number of examples in the dataset """
        return len(self.db)

    def __getitem__(self, i):
        """ Fetching example i from the dataset """
        img, lbl = self.db[i]
        img, lbl, _ = self.augmentator(img, lbl)
        return img, lbl
