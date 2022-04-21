"""
Example of a model neural network created automatically by TemplaTorch
"""

import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Simple convolutional block
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool=False, norm=True):
        """ Block initializer """
        super().__init__()
        layers = []
        conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2
            )
        layers.append(conv_layer)
        if(norm):
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if(pool):
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        """ Forward pass through block """
        y = self.block(x)
        return y


class ConvNet(nn.Module):
    """
    Simple convolutional model for testing TemplaTorch
    """

    def __init__(self, channels=[3, 32, 64, 128], out_size=10):
        """ Model Initializer """
        super().__init__()
        conv_blocks = []
        for i in range(len(channels)-1):
            cur_block = ConvBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=3,
                    pool=(i < len(channels)-1),
                    norm=True
                )
            conv_blocks.append(cur_block)
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=channels[-1], out_features=out_size)
        return

    def forward(self, x):
        """ Forward pass through model """
        feats = self.conv_blocks(x)
        pooled_feats = self.avg_pool(feats).flatten(start_dim=1)
        y = self.fc(pooled_feats)
        return y


#
