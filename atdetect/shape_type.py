"""
Shape type enumeration.
"""
from enum import Enum, auto

class ShapeType(Enum):
    """
    Enum for different shape types.
    """
    CIRCLE = auto()
    RECTANGLE = auto()
    LINE = auto()