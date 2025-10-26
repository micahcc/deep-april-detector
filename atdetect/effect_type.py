"""
Effect type enumeration.
"""
from enum import Enum, auto

class EffectType(Enum):
    """
    Enum for different effect types.
    """
    MANDELBROT = auto()
    NOISE = auto()
    LINEAR_GRADIENT = auto()
    RADIAL_GRADIENT = auto()
    COMBINED = auto()