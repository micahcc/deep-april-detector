"""
Direction enumeration.
"""
from enum import Enum, auto

class Direction(Enum):
    """
    Enum for different directions.
    """
    HORIZONTAL = auto()
    VERTICAL = auto()
    DIAGONAL = auto()