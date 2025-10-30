"""
Configuration for training.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class TrainConfig:
    """Configuration for training."""

    model_name: str
    model_config_path: str
    learning_rate: float
    batch_size: int
    epochs: int
    tag_type: str  # One of: tag16h5, tagCircle21h7, or tagStandard41h12
    val_split: float = 0.2
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        valid_tag_types = ["tag16h5", "tagCircle21h7", "tagStandard41h12"]
        if self.tag_type not in valid_tag_types:
            raise ValueError(f"tag_type must be one of {', '.join(valid_tag_types)}")
