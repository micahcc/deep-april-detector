"""
Configuration for evaluation.
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    model_path: str
    tag_type: str  # Same tag type used for training
    batch_size: int = 64
    output_file: Optional[str] = None

    def __post_init__(self):
        valid_tag_types = ["tag16h5", "tagCircle21h7", "tagStandard41h12"]
        if self.tag_type not in valid_tag_types:
            raise ValueError(f"tag_type must be one of {', '.join(valid_tag_types)}")
