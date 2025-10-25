import argparse
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import yaml


@dataclass
class TrainConfig:
    """Configuration for training."""
    model_name: str
    learning_rate: float
    batch_size: int
    epochs: int
    data_path: str
    val_split: float = 0.2
    checkpoint_dir: Optional[str] = None
    

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    model_path: str
    data_path: str
    batch_size: int = 64
    output_file: Optional[str] = None


@dataclass
class ExportConfig:
    """Configuration for export."""
    model_path: str
    output_path: str
    format: str = "onnx"  # onnx, tflite, etc.
    quantize: bool = False


def load_config(config_path: str, config_class):
    """Load and parse a YAML config file into the specified dataclass."""
    try:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        # Check for required fields
        missing_fields = []
        for field in config_class.__annotations__:
            if field not in config_dict and field not in getattr(config_class, "__dataclass_fields__", {}) or \
               (field in getattr(config_class, "__dataclass_fields__", {}) and \
                config_class.__dataclass_fields__[field].default == dataclasses._MISSING_TYPE):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"Error: Missing required fields in config: {', '.join(missing_fields)}")
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
    print(f"Training with config: {config}")
    # TODO: Implement actual training logic


def evaluate(args):
    """Handle the eval subcommand."""
    config = load_config(args.config, EvalConfig)
    print(f"Evaluating with config: {config}")
    # TODO: Implement actual evaluation logic


def export(args):
    """Handle the export subcommand."""
    config = load_config(args.config, ExportConfig)
    print(f"Exporting with config: {config}")
    # TODO: Implement actual export logic


def main():
    """CLI entrypoint with subcommands for train, eval, and export."""
    parser = argparse.ArgumentParser(
        description="Deep April Tag Detector - CLI tool for training, evaluation, and export"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    
    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", "-c", required=True, help="Path to train config YAML file")
    train_parser.set_defaults(func=train)
    
    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--config", "-c", required=True, help="Path to eval config YAML file")
    eval_parser.set_defaults(func=evaluate)
    
    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export a model")
    export_parser.add_argument("--config", "-c", required=True, help="Path to export config YAML file")
    export_parser.set_defaults(func=export)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
        
    args.func(args)


if __name__ == "__main__":
    main()
