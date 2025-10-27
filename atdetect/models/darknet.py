"""
Implementation of Darknet53 backbone for object detection networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class DarknetConv(nn.Module):
    """
    Basic convolution block for Darknet architecture:
    Convolution + BatchNorm + LeakyReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class DarknetResidualBlock(nn.Module):
    """
    Residual block used in Darknet53:
    - A 1x1 convolution that reduces the channel dimension
    - A 3x3 convolution that restores the original dimension
    - A residual connection that adds the input to the output
    """

    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = in_channels // 2

        self.conv1 = DarknetConv(
            in_channels, reduced_channels, kernel_size=1, padding=0
        )
        self.conv2 = DarknetConv(
            reduced_channels, in_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        return x + residual  # Residual connection


class Darknet53(nn.Module):
    """
    Darknet53 backbone network for object detection.
    The network produces feature maps at 3 different scales (strides 8, 16, and 32).

    Args:
        input_channels: Number of input channels (default is 1 for mono16 images)
    """

    def __init__(self, input_channels=1):
        super().__init__()

        # Initial convolution layer
        self.conv1 = DarknetConv(input_channels, 32, kernel_size=3, padding=1)

        # Downsample 1: 32 -> 64 channels, /2 resolution
        self.downsample1 = DarknetConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.res_block1 = self._make_layer(64, num_blocks=1)

        # Downsample 2: 64 -> 128 channels, /4 resolution
        self.downsample2 = DarknetConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.res_block2 = self._make_layer(128, num_blocks=2)

        # Downsample 3: 128 -> 256 channels, /8 resolution
        self.downsample3 = DarknetConv(128, 256, kernel_size=3, stride=2, padding=1)
        self.res_block3 = self._make_layer(256, num_blocks=8)

        # Downsample 4: 256 -> 512 channels, /16 resolution
        self.downsample4 = DarknetConv(256, 512, kernel_size=3, stride=2, padding=1)
        self.res_block4 = self._make_layer(512, num_blocks=8)

        # Downsample 5: 512 -> 1024 channels, /32 resolution
        self.downsample5 = DarknetConv(512, 1024, kernel_size=3, stride=2, padding=1)
        self.res_block5 = self._make_layer(1024, num_blocks=4)

    def _make_layer(self, channels, num_blocks):
        """Helper to create a sequence of residual blocks."""
        layers = []
        for i in range(num_blocks):
            layers.append(DarknetResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Downsample and residual blocks
        x = self.downsample1(x)
        x = self.res_block1(x)

        x = self.downsample2(x)
        x = self.res_block2(x)

        x = self.downsample3(x)
        x = self.res_block3(x)
        route_small = x  # First feature map (stride 8)

        x = self.downsample4(x)
        x = self.res_block4(x)
        route_medium = x  # Second feature map (stride 16)

        x = self.downsample5(x)
        x = self.res_block5(x)
        route_large = x  # Third feature map (stride 32)

        return {
            "small": route_small,  # stride 8
            "medium": route_medium,  # stride 16
            "large": route_large,  # stride 32
        }
