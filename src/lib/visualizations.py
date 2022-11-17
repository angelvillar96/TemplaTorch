"""
Utils methods for data visualization
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors


COLORS = ["blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "forestgreen", "springgreen",
          "aqua", "royalblue", "navy", "darkviolet", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]


def visualize_sequence(sequence, savepath=None, add_title=True, add_axis=False, n_cols=10,
                       size=3, n_channels=3, titles=None, **kwargs):
    """
    Visualizing a sequence of imgs in a grid like manner.

    Args:
    -----
    sequence: torch Tensor
        Sequence of images to visualize. Shape in (N_imgs, C, H, W)
    savepath: string ir None
        If not None, path where to store the sequence
    add_title: bool
        whether to add a title to each image
    n_cols: int
        Number of images per row in the grid
    size: int
        Size of each image in inches
    n_channels: int
        Number of channels (RGB=3, grayscale=1) in the data
    titles: list
        Titles to add to each image if 'add_title' is True
    """
    # initializing grid
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols)

    # adding super-title and resizing
    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    fig.suptitle(kwargs.pop("suptitle", ""))

    # plotting all frames from the sequence
    ims = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        f = sequence[i].permute(1, 2, 0).cpu().detach()
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title)
            else:
                a.set_title(f"Image {i}")

    # removing axis
    if(not add_axis):
        for i in range(n_cols * n_rows):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax, ims


def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    b, nc, h, w = x.shape
    zeros = torch.zeros if torch.is_tensor(x) else np.zeros
    px = zeros((b, 3, h+2*pad, w+2*pad))
    color = colors.to_rgb(color_name)
    px[:, 0, :, :] = color[0]
    px[:, 1, :, :] = color[1]
    px[:, 2, :, :] = color[2]
    if nc == 1:
        for c in range(3):
            px[:, c, pad:h+pad, pad:w+pad] = x[:, 0]
    else:
        px[:, :, pad:h+pad, pad:w+pad] = x
    return px
