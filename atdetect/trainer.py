"""
Trainer module for AprilTag detector.

This module includes a Trainer class that handles:
- Model training with multi-GPU support
- Performance tracking for data loading, inference, and backprop times
- Periodic logging of metrics
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from atdetect.models.detector import AprilTagDetector
from atdetect.models.loss import AprilTagLoss


class Trainer:
    """
    Trainer class for AprilTag detector models.

    Features:
    - Multi-GPU training with DistributedDataParallel
    - Performance tracking (data loading, inference, backprop times)
    - Periodic logging
    - Checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        log_interval: int = 10,
        save_interval: int = 1,
        checkpoint_dir: str = "checkpoints",
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        eval_interval: int = 5,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation data
            device: Device to train on ("cuda" or "cpu")
            learning_rate: Learning rate for Adam optimizer
            weight_decay: Weight decay for Adam optimizer
            num_epochs: Number of epochs to train for
            log_interval: How often (in batches) to log training metrics
            save_interval: How often (in epochs) to save model checkpoints
            checkpoint_dir: Directory to save model checkpoints
            distributed: Whether to use distributed training
            rank: Process rank for distributed training
            world_size: Total number of processes for distributed training
            eval_interval: How often (in epochs) to run validation
        """
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.eval_interval = eval_interval

        # Create loss function
        self.criterion = AprilTagLoss().to(device)

        # Set up the model
        self.model = model.to(device)
        if distributed:
            self.model = DDP(self.model, device_ids=[rank])

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Create checkpoint directory
        if not distributed or rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize metrics tracking
        self.metrics = {
            "data_time": [],
            "forward_time": [],
            "backward_time": [],
            "loss": [],
            "cls_loss": [],
            "box_loss": [],
            "keypoint_loss": [],
        }

    def train(self):
        """Run the training loop for the specified number of epochs."""
        logging.info("Starting training for %i epochs", self.num_epochs)

        for epoch in range(self.num_epochs):
            self._train_epoch(epoch)

            # Run validation
            if (
                self.val_dataloader is not None
                and (epoch + 1) % self.eval_interval == 0
            ):
                self._validate_epoch(epoch)

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        start_time = time.time()

        # Reset epoch metrics
        epoch_metrics = {
            "data_time": 0.0,
            "forward_time": 0.0,
            "backward_time": 0.0,
            "loss": 0.0,
            "cls_loss": 0.0,
            "box_loss": 0.0,
            "keypoint_loss": 0.0,
        }

        # Set up distributed sampler if using distributed training
        if self.distributed:
            self.train_dataloader.sampler.set_epoch(epoch)

        # Iterate through batches
        num_batches = len(self.train_dataloader)
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Measure data loading time
            data_load_end = time.time()
            data_time = data_load_end - start_time
            epoch_metrics["data_time"] += data_time

            # Get inputs and targets
            images = batch["images"].to(self.device)
            targets = {
                "boxes": [b.to(self.device) for b in batch["bboxes"]],
                "keypoints": [k.to(self.device) for k in batch["keypoints"]],
                "labels": [c.to(self.device) for c in batch["class_nums"]],
            }

            # Forward pass
            forward_start = time.time()
            predictions = self.model(images)
            forward_end = time.time()
            forward_time = forward_end - forward_start
            epoch_metrics["forward_time"] += forward_time

            # Compute loss
            image_size = (images.shape[2], images.shape[3])  # (height, width)
            loss_dict = self.criterion(predictions, targets, image_size)
            loss = loss_dict["total_loss"]

            # Backward pass and optimize
            self.optimizer.zero_grad()
            backward_start = time.time()
            loss.backward()
            self.optimizer.step()
            backward_end = time.time()
            backward_time = backward_end - backward_start
            epoch_metrics["backward_time"] += backward_time

            # Update metrics
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["cls_loss"] += loss_dict["cls_loss"].item()
            epoch_metrics["box_loss"] += loss_dict["box_loss"].item()
            epoch_metrics["keypoint_loss"] += loss_dict["keypoint_loss"].item()

            # Log progress
            if (batch_idx + 1) % self.log_interval == 0 or (
                batch_idx + 1
            ) == num_batches:
                if not self.distributed or self.rank == 0:
                    self._log_progress(
                        epoch,
                        batch_idx,
                        num_batches,
                        epoch_metrics,
                        batch_size=images.shape[0],
                    )

            # Reset timer for next batch's data loading
            start_time = time.time()

        # Average metrics for the epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            self.metrics[key].append(epoch_metrics[key])

        # Log epoch summary
        if not self.distributed or self.rank == 0:
            self._log_epoch_summary(epoch, epoch_metrics)

    def _validate_epoch(self, epoch: int):
        """
        Run validation for one epoch.

        Args:
            epoch: Current epoch number
        """
        self.model.eval()
        val_metrics = {
            "loss": 0.0,
            "cls_loss": 0.0,
            "box_loss": 0.0,
            "keypoint_loss": 0.0,
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Get inputs and targets
                images = batch["image"].to(self.device)
                targets = {
                    "boxes": batch["boxes"].to(self.device),
                    "keypoints": batch["keypoints"].to(self.device),
                    "labels": batch["labels"].to(self.device),
                }

                # Forward pass
                predictions = self.model(images)

                # Compute loss
                image_size = (images.shape[2], images.shape[3])  # (height, width)
                loss_dict = self.criterion(predictions, targets, image_size)

                # Update metrics
                val_metrics["loss"] += loss_dict["total_loss"].item()
                val_metrics["cls_loss"] += loss_dict["cls_loss"].item()
                val_metrics["box_loss"] += loss_dict["box_loss"].item()
                val_metrics["keypoint_loss"] += loss_dict["keypoint_loss"].item()

        # Average metrics
        num_batches = len(self.val_dataloader)
        for key in val_metrics:
            val_metrics[key] /= num_batches

        # Log validation results
        if not self.distributed or self.rank == 0:
            logging.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Validation Loss: {val_metrics['loss']:.4f} | "
                f"Cls: {val_metrics['cls_loss']:.4f} | "
                f"Box: {val_metrics['box_loss']:.4f} | "
                f"Keypoint: {val_metrics['keypoint_loss']:.4f}"
            )

    def _save_checkpoint(self, epoch: int):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
        """
        if not self.distributed or self.rank == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": (
                    self.model.module.state_dict()
                    if self.distributed
                    else self.model.state_dict()
                ),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": self.metrics,
            }

            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_epoch_{epoch+1}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    def _log_progress(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        metrics: Dict[str, float],
        batch_size: int,
    ):
        """
        Log training progress within an epoch.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            num_batches: Total number of batches in the epoch
            metrics: Dictionary of metrics to log
            batch_size: Batch size
        """
        samples_processed = (batch_idx + 1) * batch_size * self.world_size
        total_samples = num_batches * batch_size * self.world_size
        progress = 100.0 * (batch_idx + 1) / num_batches

        avg_loss = metrics["loss"] / (batch_idx + 1)
        avg_cls_loss = metrics["cls_loss"] / (batch_idx + 1)
        avg_box_loss = metrics["box_loss"] / (batch_idx + 1)
        avg_keypoint_loss = metrics["keypoint_loss"] / (batch_idx + 1)

        avg_data_time = metrics["data_time"] / (batch_idx + 1)
        avg_forward_time = metrics["forward_time"] / (batch_idx + 1)
        avg_backward_time = metrics["backward_time"] / (batch_idx + 1)

        logging.info(
            f"Epoch {epoch+1}/{self.num_epochs} | "
            f"Batch {batch_idx+1}/{num_batches} ({progress:.1f}%) | "
            f"Samples {samples_processed}/{total_samples} | "
            f"Loss: {avg_loss:.4f} (C: {avg_cls_loss:.4f}, B: {avg_box_loss:.4f}, K: {avg_keypoint_loss:.4f}) | "
            f"Times (s): Data: {avg_data_time:.4f}, Fwd: {avg_forward_time:.4f}, Bwd: {avg_backward_time:.4f}"
        )

    def _log_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """
        Log summary of an epoch.

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to log
        """
        logging.info(
            f"Epoch {epoch+1}/{self.num_epochs} Summary | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Cls: {metrics['cls_loss']:.4f} | "
            f"Box: {metrics['box_loss']:.4f} | "
            f"Keypoint: {metrics['keypoint_loss']:.4f} | "
            f"Times (s): Data: {metrics['data_time']:.4f}, Fwd: {metrics['forward_time']:.4f}, Bwd: {metrics['backward_time']:.4f}"
        )
