#!/usr/bin/env python3
"""
Training script for AprilTag detector.

This script:
1. Sets up distributed training environment if requested
2. Initializes the model, dataset, and data loaders
3. Runs the training loop with logging and checkpointing

Example usage:
    # Single GPU training
    python -m atdetect.train --config configs/train_config.yaml

    # Multi-GPU training with DDP
    python -m torch.distributed.launch --nproc_per_node=4 \
        -m atdetect.train --config configs/train_config.yaml --distributed
"""

import os
import sys
import logging
import argparse
import yaml
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from atdetect.models.detector import AprilTagDetector
from atdetect.april_tag_data_loader import AprilTagDataLoader 
from atdetect.trainer import Trainer
from atdetect.train_config import TrainConfig


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def get_num_classes_from_tag_type(tag_type: str) -> int:
    """
    Get the number of classes based on the tag type.

    Args:
        tag_type: The AprilTag type (tag16h5, tagCircle21h7, tagStandard41h12)

    Returns:
        Number of classes for the tag type
    """
    # Map tag types to their class counts
    # In AprilTag detection, each tag ID is treated as a separate class
    tag_type_to_classes = {
        "tag16h5": 30,  # 16h5 has 30 possible tag IDs (0-29)
        "tagCircle21h7": 35,  # tagCircle21h7 has 35 possible tag IDs (0-34)
        "tagStandard41h12": 587,  # tagStandard41h12 has 587 possible tag IDs (0-586)
    }

    if tag_type not in tag_type_to_classes:
        raise ValueError(f"Unknown tag type: {tag_type}")

    # Add one for background class (0 = background, 1+ = tag IDs)
    return tag_type_to_classes[tag_type] + 1


def load_config(config_path: str) -> TrainConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the config file

    Returns:
        TrainConfig object containing configuration parameters
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return TrainConfig(**config_dict)


def setup_distributed(rank: int, world_size: int):
    """
    Initialize the distributed environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    # Set up environment variables for PyTorch distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for GPU training
        world_size=world_size,
        rank=rank,
    )


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, config: TrainConfig, distributed: bool):
    """
    Worker function for training process.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: TrainConfig object containing configuration parameters
        distributed: Whether to use distributed training
    """
    if distributed:
        # Initialize the process group
        setup_distributed(rank, world_size)

    # Set up device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Get number of classes from tag type
    num_classes = get_num_classes_from_tag_type(config.tag_type)

    # Create model
    model = AprilTagDetector(
        input_channels=1,  # Default to 1 for mono16 images
        fpn_channels=256,  # Default FPN channels
        num_classes=num_classes,
    )

    # Create datasets
    # Use standard data directory structure
    data_dir = os.path.join("data")
    train_dir = os.path.join(data_dir, "train")

    # Ensure train directory exists
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        logging.warning(f"Created train directory: {train_dir}")

    train_dataset = AprilTagDataLoader(
        config.tag_type
    )
    
    val_dataset = None
    if config.val_split > 0:
        val_dir = os.path.join(data_dir, "val")
        # Only create validation dataset if directory exists
        if os.path.exists(val_dir):
            val_dataset = AprilTagDataset(
                data_dir=val_dir,
                transform=None,  # No transforms needed as images are generated
                split="val"
            )
        else:
            logging.warning(
                "Validation split %.2f specified but no validation directory found at %s", 
                config.val_split, val_dir
            )

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,  # Default to 4 workers
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset) if distributed else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,  # Default to 4 workers
            pin_memory=True,
        )

    # Only rank 0 process should log
    if rank == 0 or not distributed:
        logging.info("Training on %i GPUs", torch.cuda.device_count())
        logging.info("Total batch size: %i", config.batch_size * world_size)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=0.0001,  # Default weight decay
        num_epochs=config.epochs,
        log_interval=10,  # Default log interval
        save_interval=5,  # Default save interval
        checkpoint_dir=config.checkpoint_dir or "output/checkpoints",
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        eval_interval=5,  # Default eval interval
    )

    # Run training
    trainer.train()

    # Clean up
    if distributed:
        cleanup_distributed()


def main():
    """Main entry point for training script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train AprilTag detector")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )
    args = parser.parse_args()

    # Set up logging
    setup_logging()

    # Load config
    config = load_config(args.config)

    # Get world size for distributed training
    if args.distributed:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = torch.cuda.device_count()
    else:
        world_size = 1

    # Start training
    if args.distributed:
        if "RANK" in os.environ:
            # When launched with torch.distributed.launch
            rank = int(os.environ["RANK"])
            train_worker(rank, world_size, config, args.distributed)
        else:
            # When launched directly
            mp.spawn(
                train_worker,
                args=(world_size, config, args.distributed),
                nprocs=world_size,
                join=True,
            )
    else:
        # Single GPU or CPU training
        train_worker(0, 1, config, False)


if __name__ == "__main__":
    main()
