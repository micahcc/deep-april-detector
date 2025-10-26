"""
Pattern type enumeration.
"""
from enum import Enum, auto

class PatternType(Enum):
    """
    Enum for different pattern types.
    """
    GRID = auto()
    STRIPES = auto()
    CHECKER = auto()