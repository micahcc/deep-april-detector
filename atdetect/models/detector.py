"""
Top-level AprilTag detector model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from atdetect.models.darknet import Darknet53
from atdetect.models.fpn import FeaturePyramidNetwork
from atdetect.models.detection_head import MultiScaleDetectionHead
from atdetect.models.decoder import DetectionDecoder


class AprilTagDetector(nn.Module):
    """
    Complete AprilTag detector model combining:
    - Darknet53 backbone
    - Feature Pyramid Network
    - Multi-scale detection heads

    The model is designed to work with grayscale (mono16) images.
    """

    def __init__(self, input_channels=1, fpn_channels=256, num_classes=1):
        """
        Args:
            input_channels: Number of input channels (default 1 for mono16 images)
            fpn_channels: Number of channels in the FPN
            num_classes: Number of object classes to detect (default 1 for AprilTag)
        """
        super().__init__()

        # Backbone network
        self.backbone = Darknet53(input_channels=input_channels)

        # Feature pyramid network
        # Darknet53 feature channels at different scales: small=256, medium=512, large=1024
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024],  # small, medium, large
            out_channels=fpn_channels,
        )

        # Detection heads for each scale
        self.detection_head = MultiScaleDetectionHead(
            in_channels_dict={
                "small": fpn_channels,
                "medium": fpn_channels,
                "large": fpn_channels,
            },
            out_channels=fpn_channels,
            num_classes=num_classes,
        )
        
        # Detection decoder
        self.decoder = DetectionDecoder()

    def forward(self, x, decode=True):
        """
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            decode: Whether to decode the raw outputs to absolute coordinates

        Returns:
            If decode=True:
                Dictionary with decoded outputs:
                    boxes: Dictionary of boxes for each scale [batch_size, height*width, 4]
                    keypoints: Dictionary of keypoints for each scale [batch_size, height*width, 4, 2]
                    logits: Dictionary of class logits for each scale [batch_size, height*width, num_classes]
                    feature_sizes: Dictionary of feature map sizes for each scale
            If decode=False:
                Dictionary of raw detection outputs for each scale
        """
        # Extract features using backbone
        backbone_features = self.backbone(x)

        # Enhance features using FPN
        fpn_features = self.fpn(backbone_features)

        # Generate detection outputs
        raw_detections = self.detection_head(fpn_features)
        
        if not decode:
            return raw_detections
        
        # Get input image size
        _, _, height, width = x.shape
        image_size = (height, width)
        
        # Decode detection outputs
        decoded_detections = self.decoder(raw_detections, image_size)
        
        return decoded_detections
