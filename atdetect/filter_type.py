"""
Filter type enumeration.
"""
from enum import Enum, auto

class FilterType(Enum):
    """
    Enum for different filter types.
    """
    BLUR = auto()
    SHARPEN = auto()
    EMBOSS = auto()
    NONE = auto()
