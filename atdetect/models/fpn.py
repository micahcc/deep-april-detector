"""
Implementation of Feature Pyramid Network (FPN) for object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from atdetect.models.darknet import DarknetConv


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) implementation that takes multi-scale features
    from a backbone and creates a feature pyramid with top-down and lateral connections.

    This enhances the feature hierarchy for better object detection at different scales.
    """

    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list: List of input channel sizes for each scale (e.g., [256, 512, 1024] for Darknet53)
            out_channels: Number of output channels for each feature map
        """
        super().__init__()

        # Lateral connections (1x1 convs to adjust channel dimensions)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                DarknetConv(in_channels, out_channels, kernel_size=1, padding=0)
            )

        # Top-down connections (upsample and 3x3 convs to smooth features)
        self.top_down_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):  # One less than lateral connections
            self.top_down_convs.append(
                DarknetConv(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, features):
        """
        Args:
            features: Dictionary containing feature maps from the backbone at different scales
                      e.g., {'small': small_feature_map, 'medium': medium_feature_map, 'large': large_feature_map}

        Returns:
            Dictionary of enhanced feature maps at each scale
        """
        # Get feature maps at different scales
        feature_names = [
            "small",
            "medium",
            "large",
        ]  # Small to large in size (small has highest resolution)
        x_small = features["small"]  # Highest resolution, stride 8
        x_medium = features["medium"]  # Medium resolution, stride 16
        x_large = features["large"]  # Lowest resolution, stride 32

        # Apply lateral connections to adjust channel dimensions
        lateral_small = self.lateral_convs[0](x_small)
        lateral_medium = self.lateral_convs[1](x_medium)
        lateral_large = self.lateral_convs[2](x_large)

        # Top-down pathway (from large/low-res to small/high-res)
        # Start with the largest/deepest feature map
        top_down = lateral_large

        # Medium resolution feature map
        # Upsample top_down and add lateral connection
        top_down_upsampled = F.interpolate(
            top_down, size=lateral_medium.shape[2:], mode="nearest"
        )
        top_down = lateral_medium + top_down_upsampled
        # Apply 3x3 conv for smoothing
        top_down_medium = self.top_down_convs[0](top_down)

        # Small resolution feature map
        # Upsample top_down and add lateral connection
        top_down_upsampled = F.interpolate(
            top_down, size=lateral_small.shape[2:], mode="nearest"
        )
        top_down = lateral_small + top_down_upsampled
        # Apply 3x3 conv for smoothing
        top_down_small = self.top_down_convs[1](top_down)

        # Return dictionary of enhanced feature maps
        return {
            "small": top_down_small,  # Enhanced small feature map (highest resolution)
            "medium": top_down_medium,  # Enhanced medium feature map
            "large": lateral_large,  # Enhanced large feature map (lowest resolution)
        }
