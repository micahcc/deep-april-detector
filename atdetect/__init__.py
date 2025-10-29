"""
Deep April Tag Detector package.
"""

# Import enum classes
from atdetect.background_type import BackgroundType
from atdetect.direction import Direction
from atdetect.shape_type import ShapeType
from atdetect.pattern_type import PatternType
from atdetect.noise_type import NoiseType
from atdetect.filter_type import FilterType
from atdetect.gradient_type import GradientType
from atdetect.effect_type import EffectType

# Import config classes
from atdetect.train_config import TrainConfig
from atdetect.eval_config import EvalConfig
from atdetect.export_config import ExportConfig

# Import dataclasses
from atdetect.bounding_box import BoundingBox
from atdetect.key_point import KeyPoint
from atdetect.april_tag_annotation import AprilTagAnnotation
from atdetect.synthetic_image import SyntheticImage

# Import main classes
from atdetect.april_tag_data_loader import AprilTagDataLoader

# Import main module functions
from atdetect.background_generators import (
    UINT16_MAX,
    generate_random_background,
    create_solid_background,
    create_linear_gradient_background,
    create_radial_gradient_background,
    create_gaussian_noise_background,
    create_perlin_noise_background,
    create_grid_pattern_background,
    create_stripes_pattern_background,
    create_checker_pattern_background,
    create_shape_background,
    apply_noise_texture,
    apply_mandelbrot_effect,
    apply_noise_effect,
    apply_filter_effect,
)
