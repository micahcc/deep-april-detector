"""
Keypoint class for representing points of interest.
"""
from dataclasses import dataclass

@dataclass
class KeyPoint:
    """Keypoint with x, y coordinates."""

    x: float
    y: float
