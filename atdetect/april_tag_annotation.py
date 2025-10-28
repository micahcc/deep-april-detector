"""
Annotation class for AprilTag detections.
"""

from dataclasses import dataclass
from typing import List
from atdetect.bounding_box import BoundingBox
from atdetect.key_point import KeyPoint


@dataclass
class AprilTagAnnotation:
    """Annotation for a single AprilTag instance."""

    class_name: str  # e.g., 'tag16h5'
    class_num: int  # Numeric ID of the tag
    bbox: BoundingBox
    keypoints: List[KeyPoint]  # 4 keypoints, one for each corner
