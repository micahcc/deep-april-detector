"""
Decoder classes for converting network outputs to detections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class BoundingBoxDecoder(nn.Module):
    """
    Decoder for bounding box regression outputs.
    Converts relative coordinates to absolute pixel coordinates.
    """

    def __init__(self):
        super().__init__()

    def forward(self, bbox_preds, feature_sizes, image_size, strides):
        """
        Decode bounding box predictions to absolute coordinates.

        Args:
            bbox_preds: Dictionary of bounding box predictions at different scales
                Each entry has shape [batch_size, 4, height, width]
            feature_sizes: Dictionary of feature map sizes for each scale
            image_size: Original image size (height, width)
            strides: Dictionary of strides for each scale

        Returns:
            Dictionary of decoded bounding boxes for each scale
                Each entry has shape [batch_size, height*width, 4]
                where 4 represents [x1, y1, x2, y2] in absolute coordinates
        """
        decoded_bboxes = {}

        for scale_name, bbox_pred in bbox_preds.items():
            stride = strides[scale_name]
            feature_size = feature_sizes[scale_name]

            # Get batch size and feature map dimensions
            batch_size, _, height, width = bbox_pred.shape

            # Generate grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=bbox_pred.device),
                torch.arange(width, device=bbox_pred.device),
                indexing="ij",
            )

            # Reshape grid to [height*width, 2]
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid.reshape(-1, 2)

            # Reshape predictions to [batch_size, height*width, 4]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

            # Decode bbox predictions: [x_center, y_center, width, height]
            # Network outputs are relative to grid cell
            x_center = (grid[:, 0].unsqueeze(0) + bbox_pred[:, :, 0]) * stride
            y_center = (grid[:, 1].unsqueeze(0) + bbox_pred[:, :, 1]) * stride
            w = torch.exp(bbox_pred[:, :, 2]) * stride
            h = torch.exp(bbox_pred[:, :, 3]) * stride

            # Convert to [x1, y1, x2, y2] format
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # Store decoded boxes
            decoded_bboxes[scale_name] = torch.stack([x1, y1, x2, y2], dim=-1)

        return decoded_bboxes


class KeypointDecoder(nn.Module):
    """
    Decoder for keypoint regression outputs.
    Converts relative keypoint coordinates to absolute pixel coordinates.
    For AprilTags, this represents the 4 corners of the tag.
    """

    def __init__(self):
        super().__init__()

    def forward(self, keypoint_preds, feature_sizes, image_size, strides):
        """
        Decode keypoint predictions to absolute coordinates.

        Args:
            keypoint_preds: Dictionary of keypoint predictions at different scales
                Each entry has shape [batch_size, 8, height, width]
                where 8 represents 4 corners with (x,y) coordinates
            feature_sizes: Dictionary of feature map sizes for each scale
            image_size: Original image size (height, width)
            strides: Dictionary of strides for each scale

        Returns:
            Dictionary of decoded keypoints for each scale
                Each entry has shape [batch_size, height*width, 4, 2]
                where 4 represents the 4 corners and 2 represents (x,y) coordinates
        """
        decoded_keypoints = {}

        for scale_name, keypoint_pred in keypoint_preds.items():
            stride = strides[scale_name]
            feature_size = feature_sizes[scale_name]

            # Get batch size and feature map dimensions
            batch_size, _, height, width = keypoint_pred.shape

            # Generate grid coordinates
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=keypoint_pred.device),
                torch.arange(width, device=keypoint_pred.device),
                indexing="ij",
            )

            # Reshape grid to [height*width, 2]
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid.reshape(-1, 2)

            # Reshape predictions to [batch_size, height*width, 8]
            keypoint_pred = keypoint_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 8)

            # Split keypoints into x and y coordinates
            kpts_x = keypoint_pred[:, :, 0::2]  # [batch_size, height*width, 4]
            kpts_y = keypoint_pred[:, :, 1::2]  # [batch_size, height*width, 4]

            # Decode keypoint predictions: Add grid offset and multiply by stride
            for i in range(4):  # 4 corners
                kpts_x[:, :, i] = (grid[:, 0].unsqueeze(0) + kpts_x[:, :, i]) * stride
                kpts_y[:, :, i] = (grid[:, 1].unsqueeze(0) + kpts_y[:, :, i]) * stride

            # Reshape to [batch_size, height*width, 4, 2] for 4 corners with (x,y) coordinates
            corners = torch.stack(
                [
                    torch.stack([kpts_x[:, :, 0], kpts_y[:, :, 0]], dim=-1),
                    torch.stack([kpts_x[:, :, 1], kpts_y[:, :, 1]], dim=-1),
                    torch.stack([kpts_x[:, :, 2], kpts_y[:, :, 2]], dim=-1),
                    torch.stack([kpts_x[:, :, 3], kpts_y[:, :, 3]], dim=-1),
                ],
                dim=2,
            )  # [batch_size, height*width, 4, 2]

            # Store decoded keypoints
            decoded_keypoints[scale_name] = corners

        return decoded_keypoints


class ClassificationDecoder(nn.Module):
    """
    Decoder for classification outputs.
    Preserves raw logits without filtering.
    """

    def __init__(self):
        super().__init__()

    def forward(self, class_preds):
        """
        Process classification predictions.

        Args:
            class_preds: Dictionary of classification predictions at different scales
                Each entry has shape [batch_size, num_classes, height, width]

        Returns:
            Dictionary of reshaped logits for each scale
                Each entry has shape [batch_size, height*width, num_classes]
        """
        logits = {}

        for scale_name, class_pred in class_preds.items():
            # Get batch size and feature map dimensions
            batch_size, num_classes, height, width = class_pred.shape

            # Reshape to [batch_size, height*width, num_classes]
            reshaped_logits = class_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, num_classes
            )

            # For single-class case, can optionally simplify to [batch_size, height*width]
            if num_classes == 1:
                reshaped_logits = reshaped_logits.squeeze(-1)

            # Store logits
            logits[scale_name] = reshaped_logits

        return logits


class DetectionDecoder(nn.Module):
    """
    Main decoder class that combines bounding box, keypoint, and classification decoders.
    Preserves all outputs for loss calculation.
    """

    def __init__(self):
        super().__init__()

        self.bbox_decoder = BoundingBoxDecoder()
        self.keypoint_decoder = KeypointDecoder()
        self.class_decoder = ClassificationDecoder()

        # Define strides for each scale
        self.strides = {
            "small": 8,  # 1/8 of original image size
            "medium": 16,  # 1/16 of original image size
            "large": 32,  # 1/32 of original image size
        }

    def forward(self, detection_outputs, image_size):
        """
        Decode detection outputs to get all predictions for loss calculation.

        Args:
            detection_outputs: Dictionary of detection outputs for each scale
                Each entry is a tuple of (class_pred, bbox_pred, keypoint_pred)
            image_size: Original image size (height, width)

        Returns:
            Dictionary with all decoded outputs:
                boxes: Dictionary of boxes for each scale [batch_size, height*width, 4]
                keypoints: Dictionary of keypoints for each scale [batch_size, height*width, 4, 2]
                logits: Dictionary of class logits for each scale [batch_size, height*width, num_classes]
                feature_sizes: Dictionary of feature map sizes for each scale
        """
        # Split outputs into separate dictionaries
        class_preds = {}
        bbox_preds = {}
        keypoint_preds = {}
        feature_sizes = {}

        for scale_name, outputs in detection_outputs.items():
            class_pred, bbox_pred, keypoint_pred = outputs

            # Store predictions
            class_preds[scale_name] = class_pred
            bbox_preds[scale_name] = bbox_pred
            keypoint_preds[scale_name] = keypoint_pred

            # Get feature map size
            _, _, height, width = class_pred.shape
            feature_sizes[scale_name] = (height, width)

        # Decode predictions without filtering
        decoded_bboxes = self.bbox_decoder(
            bbox_preds, feature_sizes, image_size, self.strides
        )
        decoded_keypoints = self.keypoint_decoder(
            keypoint_preds, feature_sizes, image_size, self.strides
        )
        decoded_logits = self.class_decoder(class_preds)

        # Return all outputs
        return {
            "boxes": decoded_bboxes,
            "keypoints": decoded_keypoints,
            "logits": decoded_logits,
            "feature_sizes": feature_sizes,
        }
