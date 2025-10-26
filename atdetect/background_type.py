"""
Background type enumeration.
"""
from enum import Enum, auto

class BackgroundType(Enum):
    """
    Enum for different types of backgrounds.
    """
    SOLID = auto()
    GRADIENT = auto()
    NOISE = auto()
    PATTERN = auto()
    SHAPES = auto()
