"""
Configuration for exporting models.
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ExportConfig:
    """Configuration for export."""

    model_path: str
    output_path: str
    tag_type: str  # Same tag type used for training
    format: str = "onnx"  # onnx, tflite, etc.
    quantize: bool = False

    def __post_init__(self):
        valid_tag_types = ["tag16h5", "tagCircle21h7", "tagStandard41h12"]
        if self.tag_type not in valid_tag_types:
            raise ValueError(f"tag_type must be one of {', '.join(valid_tag_types)}")
