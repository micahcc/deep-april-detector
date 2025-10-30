#!/usr/bin/env python3
import argparse
import sys
import dataclasses
import yaml
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

# Import configuration classes
from atdetect.train_config import TrainConfig
from atdetect.eval_config import EvalConfig
from atdetect.export_config import ExportConfig

# Import data loader components
from atdetect.april_tag_data_loader import AprilTagDataLoader
from atdetect.trainer import Trainer
from atdetect.models import load_model

UINT16_MAX = 2**16 - 1


def load_config(config_path: str, config_class):
    """Load and parse a YAML config file into the specified dataclass."""
    try:
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)

        # Check for required fields
        missing_fields = []
        for field in config_class.__annotations__:
            if (
                field not in config_dict
                and field not in getattr(config_class, "__dataclass_fields__", {})
                or (
                    field in getattr(config_class, "__dataclass_fields__", {})
                    and config_class.__dataclass_fields__[field].default
                    == dataclasses._MISSING_TYPE
                )
            ):
                missing_fields.append(field)

        if missing_fields:
            print(
                f"Error: Missing required fields in config: {', '.join(missing_fields)}"
            )
            sys.exit(1)

        return config_class(**config_dict)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Invalid YAML in config file: {config_path}")
        sys.exit(1)
    except TypeError as e:
        print(f"Error: Invalid config format: {e}")
        sys.exit(1)


def make_examples(args):
    """Handle the make-examples subcommand for generating synthetic tag examples."""
    config = load_config(args.config, TrainConfig)
    count = args.count
    print(f"Generating examples of {config.tag_type} tags")
    print(
        f"Model: {config.model_name}, Learning rate: {config.learning_rate}, Batch size: {config.batch_size}"
    )

    # Create data loader with complex backgrounds
    data_loader = AprilTagDataLoader(
        tag_type=config.tag_type,
        min_scale_factor=1.5,
        max_scale_factor=5.0,
        tags_per_image=(1, 3),
        brightness_variation_range=0.4,  # Allow +/-40% brightness variation in tags
    )

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Generate and save a few samples
    for i in range(count):
        sample = data_loader.generate_sample()

        # Perform min-max scaling for better visualization of 16-bit images
        image_float = sample.image.astype(np.float32)

        # Scale each color channel independently for better contrast
        normalized_image = np.zeros_like(image_float)
        channel_min = np.min(image_float)
        channel_max = np.max(image_float)

        normalized_image = (
            (image_float - channel_min) / (channel_max - channel_min) * 255.0
        )

        # Convert to 8-bit for visualization
        image_8bit = normalized_image.astype(np.uint8)

        # Draw bounding boxes and keypoints for visualization
        vis_image = image_8bit.copy()
        for annotation in sample.annotations:
            # Draw bounding box
            bbox = annotation.bbox
            vis_image_pil = Image.fromarray(vis_image)
            draw = ImageDraw.Draw(vis_image_pil)

            draw.rectangle(
                [
                    (int(bbox.x_min), int(bbox.y_min)),
                    (int(bbox.x_max), int(bbox.y_max)),
                ],
                outline=UINT16_MAX,
                width=2,
            )

            # Draw keypoints
            for kp in annotation.keypoints:
                draw.ellipse(
                    [(int(kp.x) - 5, int(kp.y) - 5), (int(kp.x) + 5, int(kp.y) + 5)],
                    outline=UINT16_MAX,
                )

            # Draw class information
            # No built-in font, use simple text drawing
            draw.text(
                (int(bbox.x_min), int(bbox.y_min) - 10),
                f"{annotation.class_name} #{annotation.class_num}",
                outline=UINT16_MAX,
            )

            # Update the numpy array with the modified image
            vis_image = np.array(vis_image_pil)

        # Save both raw image and visualization
        Image.fromarray(image_8bit).save(f"output/sample_{i}.png")
        Image.fromarray(vis_image).save(f"output/sample_{i}_annotated.png")

    print(
        f"Generated {count} sample images in the 'output' directory for visualization."
    )


def evaluate(args):
    """Handle the eval subcommand."""
    config = load_config(args.config, EvalConfig)
    print(f"Evaluating model on {config.tag_type} tag templates")
    print(f"Model path: {config.model_path}, Batch size: {config.batch_size}")
    # TODO: Implement actual evaluation logic


def export(args):
    """Handle the export subcommand."""
    config = load_config(args.config, ExportConfig)
    print(
        f"Exporting {config.tag_type} detector model to {config.format.upper()} format"
    )
    print(
        f"Model path: {config.model_path}, Output path: {config.output_path}, Quantize: {config.quantize}"
    )
    # TODO: Implement actual export logic


def setup_logging():
    """Set up logging configuration."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def get_num_classes_from_tag_type(tag_type: str) -> int:
    """Get the number of classes based on the tag type."""
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


def setup_distributed(rank: int, world_size: int):
    """Initialize the distributed environment."""
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


def synthetic_image_collate_fn(batch):
    """Custom collate function for SyntheticImage objects.

    This function processes a batch of SyntheticImage objects and converts them
    to a format compatible with PyTorch DataLoader.

    Args:
        batch: A list of SyntheticImage objects from the dataset.

    Returns:
        A dictionary containing batched tensors for images, bboxes, class_nums and keypoints.
    """
    images = []
    bboxes = []
    class_nums = []
    keypoints = []

    target_width = 0
    target_height = 0
    target_channels = 0
    for sample in batch:
        target_height = max(target_height, sample.image.shape[0])
        target_width = max(target_width, sample.image.shape[1])
        if len(sample.image.shape) == 2:
            sample.image = np.expand_dims(sample.image, axis=2)
        target_channels = max(target_channels, 1)

    # make one big array, (B, C, H, W)
    images = torch.zeros((len(batch), target_height, target_width, target_channels))
    for b, sample in enumerate(batch):
        height = sample.image.shape[0]
        width = sample.image.shape[1]
        channels = sample.image.shape[2]

        img_tensor = torch.from_numpy(sample.image.astype(np.float32))
        images[b, 0:height, 0:width, 0:channels] = img_tensor[:, :, :]

        # Process annotations
        sample_bboxes = []
        sample_class_nums = []
        sample_keypoints = []
        for ann in sample.annotations:
            # Get bbox coordinates
            bbox = ann.bbox
            sample_bboxes.append(
                torch.tensor(
                    [bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                    dtype=torch.float32,
                )
            )

            # Get class number
            sample_class_nums.append(torch.tensor(ann.class_num, dtype=torch.long))

            # Process keypoints if available
            kps = torch.tensor(
                [[kp.x, kp.y] for kp in ann.keypoints], dtype=torch.float32
            )
            sample_keypoints.append(kps)

        bboxes.append(torch.stack(sample_bboxes))
        class_nums.append(torch.stack(sample_class_nums))
        keypoints.append(torch.stack(sample_keypoints))

    return {
        "images": images,
        "bboxes": bboxes,  # List of lists of tensors
        "class_nums": class_nums,  # List of lists of tensors
        "keypoints": keypoints,  # List of lists of tensors
    }


def train_worker(rank: int, world_size: int, config: TrainConfig, distributed: bool):
    """Worker function for training process."""
    if distributed:
        # Initialize the process group
        setup_distributed(rank, world_size)

    # Set up device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Get number of classes from tag type
    num_classes = get_num_classes_from_tag_type(config.tag_type)

    # Create model

    model = load_model(config.model_config_path)

    # Create datasets
    # Use standard data directory structure
    data_dir = os.path.join("data")
    train_dir = os.path.join(data_dir, "train")

    # Ensure train directory exists
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
        import logging

        logging.warning(f"Created train directory: {train_dir}")

    train_dataset = AprilTagDataLoader(config.tag_type)

    val_dataset = None
    if config.val_split > 0:
        val_dir = os.path.join(data_dir, "val")
        # Only create validation dataset if directory exists
        if os.path.exists(val_dir):
            val_dataset = AprilTagDataLoader(config.tag_type, templates_dir=val_dir)
        else:
            import logging

            logging.warning(
                "Validation split %.2f specified but no validation directory found at %s",
                config.val_split,
                val_dir,
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
        collate_fn=synthetic_image_collate_fn,
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
            collate_fn=synthetic_image_collate_fn,
        )

    # Only rank 0 process should log
    if rank == 0 or not distributed:
        import logging

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


def train(args):
    """Handle the train subcommand."""
    config = load_config(args.config, TrainConfig)
    print(f"Training model on {config.tag_type} tag templates")
    print(
        f"Model: {config.model_name}, Learning rate: {config.learning_rate}, Batch size: {config.batch_size}"
    )

    # Set up logging
    setup_logging()

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


def main():
    """CLI entrypoint with subcommands for train, eval, and export."""
    parser = argparse.ArgumentParser(
        description="Deep April Tag Detector - CLI tool for training, evaluation, and export"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config", "-c", required=True, help="Path to train config YAML file"
    )
    train_parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )
    train_parser.set_defaults(func=train)

    # Make-examples subcommand
    make_examples_parser = subparsers.add_parser(
        "make-examples", help="Generate synthetic tag examples"
    )
    make_examples_parser.add_argument(
        "--config", "-c", required=True, help="Path to train config YAML file"
    )
    make_examples_parser.add_argument(
        "--count", "-C", default=10, type=int, help="How many to generate"
    )
    make_examples_parser.set_defaults(func=make_examples)

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument(
        "--config", "-c", required=True, help="Path to eval config YAML file"
    )
    eval_parser.set_defaults(func=evaluate)

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export a model")
    export_parser.add_argument(
        "--config", "-c", required=True, help="Path to export config YAML file"
    )
    export_parser.set_defaults(func=export)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
