"""
Bounding box class for object detection.
"""
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """Bounding box in [x_min, y_min, x_max, y_max] format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
