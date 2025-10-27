"""
Detection head classes for AprilTag detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from atdetect.models.darknet import DarknetConv


class AprilTagDetectionHead(nn.Module):
    """
    Detection head for AprilTag detection.
    Produces outputs for:
    - Object classification (tag/no-tag)
    - Bounding box regression (x, y, w, h)
    - Corner point regression (8 values for 4 corner points)
    """

    def __init__(self, in_channels, num_classes=1):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of object classes (default 1 for AprilTag)
        """
        super().__init__()

        # Common feature extraction
        self.conv1 = DarknetConv(
            in_channels, in_channels // 2, kernel_size=3, padding=1
        )
        self.conv2 = DarknetConv(
            in_channels // 2, in_channels, kernel_size=3, padding=1
        )

        # Classification branch
        self.cls_conv = DarknetConv(
            in_channels, in_channels // 2, kernel_size=3, padding=1
        )
        self.cls_output = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

        # Bounding box regression branch
        self.bbox_conv = DarknetConv(
            in_channels, in_channels // 2, kernel_size=3, padding=1
        )
        self.bbox_output = nn.Conv2d(in_channels // 2, 4, kernel_size=1)  # x, y, w, h

        # Keypoint (corner) regression branch
        self.keypoint_conv = DarknetConv(
            in_channels, in_channels // 2, kernel_size=3, padding=1
        )
        self.keypoint_output = nn.Conv2d(
            in_channels // 2, 8, kernel_size=1
        )  # 4 corners * 2 coords each

    def forward(self, x):
        """
        Args:
            x: Input feature map

        Returns:
            Tuple of (class_pred, bbox_pred, keypoint_pred)
        """
        # Common layers
        feat = self.conv1(x)
        feat = self.conv2(feat)

        # Classification branch
        cls_feat = self.cls_conv(feat)
        cls_pred = self.cls_output(cls_feat)

        # Bounding box regression branch
        bbox_feat = self.bbox_conv(feat)
        bbox_pred = self.bbox_output(bbox_feat)

        # Keypoint regression branch
        keypoint_feat = self.keypoint_conv(feat)
        keypoint_pred = self.keypoint_output(keypoint_feat)

        return cls_pred, bbox_pred, keypoint_pred


class MultiScaleDetectionHead(nn.Module):
    """
    Multi-scale detection head that processes features at multiple scales.
    Creates a detection head for each feature map scale.
    """

    def __init__(self, in_channels_dict, out_channels, num_classes=1):
        """
        Args:
            in_channels_dict: Dictionary of input channels for each scale
            out_channels: Number of output channels for each detection head
            num_classes: Number of object classes
        """
        super().__init__()

        # Create detection heads for each scale
        self.detection_heads = nn.ModuleDict()

        for scale_name, in_channels in in_channels_dict.items():
            self.detection_heads[scale_name] = AprilTagDetectionHead(
                in_channels=in_channels, num_classes=num_classes
            )

    def forward(self, features):
        """
        Args:
            features: Dictionary of feature maps at different scales

        Returns:
            Dictionary of detection outputs for each scale
        """
        outputs = {}

        for scale_name, feature_map in features.items():
            outputs[scale_name] = self.detection_heads[scale_name](feature_map)

        return outputs
