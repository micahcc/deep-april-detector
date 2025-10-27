"""
Loss functions for AprilTag detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import (
    sigmoid_focal_loss,
    box_iou,
    generalized_box_iou_loss,
    clip_boxes_to_image,
)
from typing import Dict, List, Tuple, Optional


class AprilTagLoss(nn.Module):
    """
    Combined loss function for AprilTag detection, including:
    - Focal loss for classification
    - GIoU loss for bounding boxes
    - L1/Smooth L1 loss for keypoints

    This function takes merged predictions across scales for simplified processing.
    """

    def __init__(
        self,
        cls_weight=1.0,
        box_weight=1.0,
        keypoint_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_iou_threshold=0.5,
        neg_iou_threshold=0.4,
    ):
        """
        Args:
            cls_weight: Weight for classification loss
            box_weight: Weight for bounding box loss
            keypoint_weight: Weight for keypoint loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            pos_iou_threshold: IoU threshold for positive samples
            neg_iou_threshold: IoU threshold for negative samples
        """
        super().__init__()

        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.keypoint_weight = keypoint_weight

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

    def compute_classification_loss(self, pred_logits, target_boxes=None, ious=None):
        """
        Compute classification loss using focal loss.
        Always computes classification loss regardless of inputs.

        Args:
            pred_logits: Predicted logits tensor, shape [P]
            target_boxes: Optional target boxes tensor, shape [N, 4]
            ious: Optional IoU values between predictions and targets, shape [P, N]

        Returns:
            cls_loss: Classification loss value
            pos_mask: Boolean mask for positive samples, shape [P]
            neg_mask: Boolean mask for negative samples, shape [P]
            max_iou_indices: Indices of predictions with max IoU for each target
        """
        device = pred_logits.device
        pos_mask = torch.zeros_like(pred_logits, dtype=torch.bool)
        neg_mask = torch.ones_like(
            pred_logits, dtype=torch.bool
        )  # Default all to negative
        max_iou_indices = None

        # If there are targets and IoUs, determine positive and negative samples
        if target_boxes is not None and target_boxes.shape[0] > 0 and ious is not None:
            # For each target, find the highest IoU prediction
            max_iou_values, max_iou_indices = ious.max(dim=0)  # [N]

            # Assign positive samples (those with highest IoU for each target)
            pos_mask[max_iou_indices] = True

            # Also assign as positive any prediction with IoU > threshold
            for i in range(target_boxes.shape[0]):
                high_iou_mask = ious[:, i] > self.pos_iou_threshold
                pos_mask = pos_mask | high_iou_mask

            # Assign negative samples (those with IoU < neg_threshold with all targets)
            max_iou_per_pred, _ = ious.max(dim=1)  # [P]
            neg_mask = max_iou_per_pred < self.neg_iou_threshold

            # Remove any overlap
            neg_mask = neg_mask & (~pos_mask)

        # Create target tensor (1 for positive samples, 0 for negative samples)
        cls_targets = torch.zeros_like(pred_logits)
        cls_targets[pos_mask] = 1.0

        # Compute focal loss
        cls_loss = sigmoid_focal_loss(
            pred_logits,
            cls_targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )

        return cls_loss, pos_mask, neg_mask, max_iou_indices

    def compute_bbox_and_keypoint_loss(
        self,
        pred_boxes,
        pred_keypoints,
        target_boxes,
        target_keypoints,
        max_iou_indices,
    ):
        """
        Compute bounding box and keypoint losses for matched predictions.

        Args:
            pred_boxes: Predicted boxes tensor, shape [P, 4]
            pred_keypoints: Predicted keypoints tensor, shape [P, 4, 2]
            target_boxes: Target boxes tensor, shape [N, 4]
            target_keypoints: Target keypoints tensor, shape [N, 4, 2]
            max_iou_indices: Indices of predictions with max IoU for each target, shape [N]

        Returns:
            box_loss: Bounding box loss value
            keypoint_loss: Keypoint loss value
        """
        # For each target, get the highest IoU prediction
        pred_boxes_matched = pred_boxes[max_iou_indices]  # [N, 4]

        # Compute GIoU loss using torchvision
        box_loss = generalized_box_iou_loss(
            pred_boxes_matched, target_boxes, reduction="mean"
        )

        # Compute keypoint loss (L1)
        pred_keypoints_matched = pred_keypoints[max_iou_indices]  # [N, 4, 2]
        keypoint_loss = (
            F.l1_loss(pred_keypoints_matched, target_keypoints, reduction="none")
            .sum(dim=(1, 2))
            .mean()
        )

        return box_loss, keypoint_loss

    def forward(self, predictions, targets, image_size):
        """
        Compute the combined loss on merged predictions.

        Args:
            predictions: Dictionary of merged predictions containing:
                - boxes: Tensor of predicted boxes, shape (B, P, 4)
                - keypoints: Tensor of predicted keypoints, shape (B, P, 4, 2)
                - logits: Tensor of predicted class logits, shape (B, P)
            targets: Dictionary of targets containing:
                - boxes: Tensor of target boxes, shape (B, N, 4)
                - keypoints: Tensor of target keypoints, shape (B, N, 4, 2)
                - labels: Tensor of target labels, shape (B, N)
            image_size: Tuple of (height, width) for clipping boxes

        Returns:
            loss_dict: Dictionary of losses containing:
                - cls_loss: Classification loss
                - box_loss: Bounding box loss
                - keypoint_loss: Keypoint loss
                - total_loss: Combined total loss
        """
        batch_size = targets["boxes"].shape[0]
        device = targets["boxes"].device

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_keypoint_loss = torch.tensor(0.0, device=device)

        num_pos_samples = 0

        # Process each batch
        for batch_idx in range(batch_size):
            # Get predictions for this batch
            pred_boxes = predictions["boxes"][batch_idx]  # [P, 4]
            pred_keypoints = predictions["keypoints"][batch_idx]  # [P, 4, 2]
            pred_logits = predictions["logits"][batch_idx]  # [P]

            # Get targets for this batch
            target_boxes = targets["boxes"][batch_idx]  # [N, 4]
            target_keypoints = targets["keypoints"][batch_idx]  # [N, 4, 2]
            target_labels = targets["labels"][batch_idx]  # [N]

            # Filter out padded targets
            if target_boxes.shape[0] > 0:
                valid_mask = target_labels >= 0
                target_boxes = target_boxes[valid_mask]
                target_keypoints = target_keypoints[valid_mask]
                target_labels = target_labels[valid_mask]

            # Clip boxes to image bounds
            pred_boxes = clip_boxes_to_image(pred_boxes, image_size)
            target_boxes = clip_boxes_to_image(target_boxes, image_size)

            # Compute IoU if there are targets
            ious = None
            if target_boxes.shape[0] > 0:
                ious = box_iou(pred_boxes, target_boxes)  # [P, N]

            # 1. Classification Loss - always computed
            cls_loss, pos_mask, neg_mask, max_iou_indices = (
                self.compute_classification_loss(pred_logits, target_boxes, ious)
            )
            total_cls_loss = total_cls_loss + cls_loss

            # Count positive samples
            num_pos = pos_mask.sum().item()
            num_pos_samples += num_pos

            # 2. Bounding Box and Keypoint Loss (only if positive samples and targets)
            if (
                num_pos > 0
                and target_boxes.shape[0] > 0
                and max_iou_indices is not None
            ):
                box_loss, keypoint_loss = self.compute_bbox_and_keypoint_loss(
                    pred_boxes,
                    pred_keypoints,
                    target_boxes,
                    target_keypoints,
                    max_iou_indices,
                )
                total_box_loss = total_box_loss + box_loss
                total_keypoint_loss = total_keypoint_loss + keypoint_loss

        # Normalize losses by batch size
        total_cls_loss = total_cls_loss / batch_size

        if num_pos_samples > 0:
            total_box_loss = total_box_loss / batch_size
            total_keypoint_loss = total_keypoint_loss / batch_size

        # Apply weights
        cls_loss_weighted = self.cls_weight * total_cls_loss
        box_loss_weighted = self.box_weight * total_box_loss
        keypoint_loss_weighted = self.keypoint_weight * total_keypoint_loss

        # Total loss
        total_loss = cls_loss_weighted + box_loss_weighted + keypoint_loss_weighted

        # Return loss dict
        return {
            "cls_loss": total_cls_loss,
            "box_loss": total_box_loss,
            "keypoint_loss": total_keypoint_loss,
            "cls_loss_weighted": cls_loss_weighted,
            "box_loss_weighted": box_loss_weighted,
            "keypoint_loss_weighted": keypoint_loss_weighted,
            "total_loss": total_loss,
            "num_pos": num_pos_samples,
        }
