"""
Noise type enumeration.
"""
from enum import Enum, auto

class NoiseType(Enum):
    """
    Enum for different noise types.
    """
    GAUSSIAN = auto()
    PERLIN = auto()