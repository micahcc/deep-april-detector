import argparse
import sys
import dataclasses
import yaml
import os
import numpy as np
import cv2

# Import configuration classes
from atdetect.train_config import TrainConfig
from atdetect.eval_config import EvalConfig
from atdetect.export_config import ExportConfig

# Import data loader components
from atdetect.background_complexity import BackgroundComplexity
from atdetect.april_tag_data_loader import AprilTagDataLoader


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


def train(args):
    """Handle the train subcommand."""
    config = load_config(args.config, TrainConfig)
    print(f"Training model on {config.tag_type} tag templates")
    print(
        f"Model: {config.model_name}, Learning rate: {config.learning_rate}, Batch size: {config.batch_size}"
    )

    # Create data loader with complex backgrounds
    data_loader = AprilTagDataLoader(
        tag_type=config.tag_type,
        min_scale_factor=1.5,
        max_scale_factor=5.0,
        tags_per_image=(1, 3),
        bg_complexity=BackgroundComplexity.COMPLEX,  # Use complex backgrounds for better training
        brightness_variation_range=0.4,  # Allow +/-40% brightness variation in tags
    )

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Generate and save a few samples
    for i in range(5):
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
            cv2.rectangle(
                vis_image,
                (int(bbox.x_min), int(bbox.y_min)),
                (int(bbox.x_max), int(bbox.y_max)),
                (0, 255, 0),  # Green color
                2,
            )

            # Draw keypoints
            for kp in annotation.keypoints:
                cv2.circle(
                    vis_image,
                    (int(kp.x), int(kp.y)),
                    5,  # Radius
                    (0, 0, 255),  # Red color
                    -1,  # Filled
                )

            # Draw class information
            cv2.putText(
                vis_image,
                f"{annotation.class_name} #{annotation.class_num}",
                (int(bbox.x_min), int(bbox.y_min) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),  # Yellow color
                1,
            )

        # Save both raw image and visualization
        cv2.imwrite(f"output/sample_{i}.png", image_8bit)
        cv2.imwrite(f"output/sample_{i}_annotated.png", vis_image)

    print(f"Generated 5 sample images in the 'output' directory for visualization.")
    print(
        f"Images use min-max scaling per channel for optimal visualization of 16-bit data."
    )

    # TODO: Implement actual training logic using template images for the specified tag type


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
    train_parser.set_defaults(func=train)

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
