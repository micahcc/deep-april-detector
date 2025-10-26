"""
Synthetic image class with annotations.
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from atdetect.april_tag_annotation import AprilTagAnnotation

@dataclass
class SyntheticImage:
    """Synthetic image with AprilTag annotations."""

    image: np.ndarray  # H x W x C image (16-bit)
    annotations: List[AprilTagAnnotation]
    height: int
    width: int