"""
Implementation of data augmentations and other related methods
"""

import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from CONFIG import AUGMENTATIONS


class Augmentator:
    """
    Class for augmenting an image from the dataset into two 'views' from that same
    image. The transformations must not change the contained objects (e.g., no crops).

    Args:
    -----
    augment_params: dictionary
        augmentations chunk from the experiment parameters
    """

    def __init__(self, augment_params={}):
        """ Initializer of the augmentator """
        add_augmentations = augment_params.get("use_augments", [])
        self.training = True

        # instanciating all augmentations
        tf_list = []
        for augment in add_augmentations:
            augment_params = augment_params.get(augment, {})
            tf = self._get_augment(augment, augment_params)
            tf_list.append(tf)
        self.augmentations = tf_list

        print("Using Augmentatorch:")
        for aug in self.augmentations:
            print(f"    {aug}")
        return

    def _get_augment(self, augment, augment_params):
        """
        Fetching the augmentation
        """
        # common parameters
        on_train = augment_params.get("on_train", True)
        on_eval = augment_params.get("on_eval", False)

        if augment == "mirror":
            mirror_prob = augment_params.get("mirror_prob", 0.5)
            tf = RandomHorizontalFlip(
                    mirror_prob=mirror_prob,
                    on_train=on_train,
                    on_eval=on_eval,
                )
        elif augment == "color_jitter":
            jitter_strength = augment_params.get("jitter_strength", 0.5)
            tf = ColorJitter(
                    brightness=round(0.8 * jitter_strength, 4),
                    contrast=round(0.8 * jitter_strength, 4),
                    hue=round(0.8 * jitter_strength, 4),
                    saturation=round(0.2 * jitter_strength, 4),
                    on_train=on_train,
                    on_eval=on_eval,
                )
        elif augment == "rotate":
            rotate_degrees = augment_params.get("rotate_degrees", 20)
            tf = RandomRotation(
                    degrees=rotate_degrees,
                    on_train=on_train,
                    on_eval=on_eval,
                )
        elif augment == "noise":
            noise_std = augment_params.get("noise_std", 0.15)
            tf = AddNoise(
                    mean=0,
                    std=noise_std,
                    on_train=on_train,
                    on_eval=on_eval,
                )
        elif augment == "scale":
            output_size = augment_params.get("output_size", (64, 64))
            min_scale_factor = augment_params.get("min_scale_factor", 0.75)
            max_scale_factor = augment_params.get("max_scale_factor", 2.)
            tf = Scale(
                    output_size=output_size,
                    min_scale_factor=min_scale_factor,
                    max_scale_factor=max_scale_factor,
                    on_train=on_train,
                    on_eval=on_eval,
                )
        else:
            raise NotImplementedError(f"Unrecognized {augment = }. Only {AUGMENTATIONS} supported")

        return tf

    def _augment(self, x, y):
        """
        Applying list of augmentations to one image

        Args:
        -----
        x: torch Tensor
            Images to augment. Shape is (..., C, H, W)
        y: torch Tensor
            Labels of the images to augment.

        Returns:
        --------
        x: torch Tensor
            Augmented images to augment.
        y: torch Tensor
            Augmented labels, if necessary (e.g. horiz flip or scaling)
        params: dict
            Actual parameters used for augmentation
        """
        params = {}
        for augmentation in self.augmentations:
            x, y = augmentation(x, y)
            params = {**params, **augmentation.get_params()}
        return x, y, params

    def __getitem__(self, i):
        """
        Fetching augmentation 'i'
        """
        return self.augmentations[i]

    def __call__(self, x, y):
        """
        Applying random augmentations to one image

        Args:
        -----
        x: torch Tensor
            Images to augment. Shape is (..., C, H, W)
        y: torch Tensor
            Labels of the images to augment.
        """
        x, y, params = self._augment(x, y)
        return x, y, params

    def __repr__(self):
        """ For displaying nicely """
        message = f"Compose of {len(self.augmentations)} transforms:\n"
        for tf in self.augmentations:
            message += f"    {self.__class__.__name__}\n"
        message = message[:-1]  # removing last line break
        return message

    def train(self):
        """ Setting training state to True """
        self.training = True
        for augment in self.augmentations:
            augment.training = True

    def eval(self):
        """ Setting training state to True """
        self.training = False
        for augment in self.augmentations:
            augment.training = False


class Augmentation:
    """
    Base class for self-implemented augmentations

    Args:
    -----
    params: list
        list of parameters used for the augmentation
    """

    def __init__(self, on_train, on_eval, params):
        """ Module initializer """
        self.on_train = on_train
        self.on_eval = on_eval
        self.training = True

        self.params = params
        self.log = {}
        return

    def __call__(self, x, y):
        """ Auctually augmenting one image"""
        raise NotImplementedError("Base class does not implement augmentations")

    def log_params(self, values):
        """ Saving the exact sampled value for the current augment """
        assert len(values) == len(self.params), \
            f"ERROR! Length of value ({len(values)}) and params ({len(self.params)}) do not match"
        self.log = {p: v for p, v in zip(self.params, values)}
        return

    def get_params(self):
        """ Fetching parameters and values """
        return self.log

    def should_augment_label(self, x, y):
        """
        Checking whether label should be augmented. Only done if (H, W) are the same in img and label
        """
        should_augment = True
        if type(x) != type(y):
            should_augment = False
        elif x.shape[-2:] == y.shape[-2:]:
            should_augment = False
        return should_augment

    def apply_augment(self):
        """
        Determines, given the training state, whether the augmentation should be applied or not
        """
        apply_augment = False
        if self.on_train and self.training:
            apply_augment = True
        if self.on_eval and not self.training:
            apply_augment = True
        return apply_augment


class RandomHorizontalFlip(Augmentation):
    """
    Horizontally mirroring an image given a certain probability
    """

    PARAMS = ["mirror_prob"]
    AUGMENT_LABEL = True

    def __init__(self, on_train, on_eval, mirror_prob=0.5):
        """ Augmentation initializer """
        super().__init__(on_train=on_train, on_eval=on_eval, params=self.PARAMS)
        self.mirror_prob = mirror_prob

    def __call__(self, x, y):
        """ Mirroring the image """
        if not self.apply_augment():
            return x, y

        mirror = np.random.rand() < self.mirror_prob
        self.log_params([mirror])
        x_augment = F.hflip(x) if mirror else x
        if self.should_augment_label(x, y):
            y = F.hflip(y) if mirror else y
        return x_augment, y

    def __repr__(self):
        """ String representation """
        str = f"RandomHorizontalFlip(mirror_prob={self.mirror_prob})"
        return str


class RandomRotation(Augmentation):
    """
    Rotating an image for certaing angle
    """

    PARAMS = ["degrees"]
    AUGMENT_LABEL = True

    def __init__(self, on_train, on_eval, degrees=20):
        """ Augmentation initializer """
        super().__init__(on_train=on_train, on_eval=on_eval, params=self.PARAMS)
        self.degrees = degrees

    def __call__(self, x, y):
        """ Rotating the image by a random sample angle """
        if not self.apply_augment():
            return x, y

        random_angle = (np.random.rand() * self.degrees*2) - self.degrees
        self.log_params([random_angle])
        x_augment = F.rotate(x, random_angle)
        if self.should_augment_label(x, y):
            y = F.rotate(y, random_angle)
        return x_augment, y

    def __repr__(self):
        """ String representation """
        str = f"RandomRotation(degrees={self.degrees})"
        return str


class ColorJitter(Augmentation):
    """ Color Jittering augmentation """

    PARAMS = ["brightness", "contrast", "hue", "saturation"]
    AUGMENT_LABEL = False

    def __init__(self, on_train, on_eval, brightness, contrast, hue, saturation):
        """ Augmentation Initializer """
        super().__init__(on_train=on_train, on_eval=on_eval, params=self.PARAMS)
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.tf = transforms.ColorJitter(brightness, contrast, hue, saturation)
        return

    def __call__(self, x, y):
        """ Augmenting """
        if not self.apply_augment():
            return x, y

        self.log_params(["?", "?", "?", "?"])
        x_augment = self.tf(x)
        return x_augment, y

    def __repr__(self):
        """ String representation """
        str = f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, hue={self.hue}," +\
              f"saturation={self.saturation})"
        return str


class AddNoise(Augmentation):
    """
    Custom augmentation to add some random Gaussiabn Noise to an Image
    """

    PARAMS = ["mean", "std"]
    AUGMENT_LABEL = False

    def __init__(self, on_train, on_eval, mean=0., std=0.3):
        """ Initializer """
        super().__init__(on_train=on_train, on_eval=on_eval, params=self.PARAMS)
        self.std = std
        self.mean = mean
        self.rand_std = lambda: torch.rand(1)*self.std
        return

    def __call__(self, x, y):
        """ Actually adding noise to the image """
        if not self.apply_augment():
            return x, y

        random_std = self.rand_std()
        self.log_params([0.0, random_std])
        noise = torch.randn(x.shape) * random_std + self.mean
        noisy_x = x + noise.to(x.device)
        noisy_x = noisy_x.clamp(0, 1)
        return noisy_x, y

    def __repr__(self):
        """ String representation """
        str = f"AddNoise(mean={self.mean}, std={self.std})"
        return str


class Scale(Augmentation):
    """
    Scaling the image to a given size
    """

    PARAMS = ["output_size", "scale_factor"]
    AUGMENT_LABEL = True

    def __init__(self, on_train, on_eval, output_size=(64, 64), min_scale_factor=0.75, max_scale_factor=0.75):
        """ Initializer """
        super().__init__(on_train=on_train, on_eval=on_eval, params=self.PARAMS)
        self.output_size = output_size
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor

        self.get_scale_factor = lambda: np.random.uniform(min_scale_factor, max_scale_factor)
        return

    def __call__(self, x, y):
        """ Actually scaling the image """
        if not self.apply_augment():
            return x, y

        scale_factor = self.get_scale_factor()
        output_size = (int(scale_factor * self.output_size[0]), int(scale_factor * self.output_size[1]))
        self.log_params([output_size, scale_factor])

        scaled_x = F.resize(x, output_size, interpolation=transforms.InterpolationMode.BILINEAR)
        if self.should_augment_label(x, y):
            y = F.resize(y, output_size, interpolation=transforms.InterpolationMode.NEAREST)
        return scaled_x, y

    def __repr__(self):
        """ String representation """
        str = f"Scale(output_size={self.output_size}, scale_factor=[{self.min_scale_factor}," +\
              f"{self.min_scale_factor}])"
        return str


#
