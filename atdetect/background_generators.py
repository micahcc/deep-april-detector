"""
Background generators for AprilTag detection.

This module contains functions to generate various background patterns and effects
for synthetic AprilTag training images. All functions operate on uint16 grayscale images.
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageFilter
from typing import Tuple, List, Optional, Union, Callable

from atdetect.background_type import BackgroundType
from atdetect.direction import Direction
from atdetect.shape_type import ShapeType
from atdetect.pattern_type import PatternType
from atdetect.noise_type import NoiseType
from atdetect.filter_type import FilterType
from atdetect.gradient_type import GradientType
from atdetect.effect_type import EffectType

# Define global constants
UINT16_MAX = np.iinfo(np.uint16).max


def create_solid_background(
    height: int, width: int, color_range: Tuple[int, int] = (0, UINT16_MAX)
) -> np.ndarray:
    """
    Generate a solid color background.

    Args:
        height: Image height
        width: Image width
        color_range: Range for random color generation (min, max)

    Returns:
        Solid color background as uint16 array
    """
    bg_color = random.randint(color_range[0], color_range[1])
    background = np.ones((height, width), dtype=np.uint16) * bg_color
    return background


def create_linear_gradient_background(
    height: int,
    width: int,
    direction: Direction = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a linear gradient background.

    Args:
        height: Image height
        width: Image width
        direction: Gradient direction (HORIZONTAL, VERTICAL, or DIAGONAL)
        color_range: Range for random color generation (min, max)

    Returns:
        Linear gradient background as uint16 array
    """
    if direction is None:
        direction = random.choice(list(Direction))

    # Create a base background
    background = np.zeros((height, width), dtype=np.uint16)

    # Generate start and end colors
    start_color = random.randint(color_range[0], color_range[1])
    end_color = random.randint(color_range[0], color_range[1])

    if direction == Direction.HORIZONTAL:
        # Horizontal gradient (left to right)
        for i in range(width):
            ratio = i / width
            color = int(start_color * (1 - ratio) + end_color * ratio)
            background[:, i] = color

    elif direction == Direction.VERTICAL:
        # Vertical gradient (top to bottom)
        for i in range(height):
            ratio = i / height
            color = int(start_color * (1 - ratio) + end_color * ratio)
            background[i, :] = color

    else:  # Direction.DIAGONAL
        # Diagonal gradient
        for i in range(height):
            for j in range(width):
                ratio = (i / height + j / width) / 2
                color = int(start_color * (1 - ratio) + end_color * ratio)
                background[i, j] = color

    return background


def create_radial_gradient_background(
    height: int,
    width: int,
    center: Tuple[int, int] = None,
    bright_center: bool = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a radial gradient background.

    Args:
        height: Image height
        width: Image width
        center: Center point of the gradient (x, y)
        bright_center: If True, center is bright and edges are dark; if False, vice versa
        color_range: Range for random color generation (min, max)

    Returns:
        Radial gradient background as uint16 array
    """
    # Create a base background
    background = np.zeros((height, width), dtype=np.uint16)

    # Set defaults for random parameters
    if center is None:
        center = (random.randint(0, width - 1), random.randint(0, height - 1))

    if bright_center is None:
        bright_center = random.choice([True, False])

    # Generate colors
    start_color = random.randint(color_range[0], color_range[1])
    end_color = random.randint(color_range[0], color_range[1])

    if not bright_center:
        start_color, end_color = end_color, start_color

    # Calculate maximum distance from center to corner
    max_dist = np.sqrt((width - 1) ** 2 + (height - 1) ** 2)

    # Generate radial gradient
    y_coords, x_coords = np.ogrid[:height, :width]
    distances = np.sqrt((x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2)
    normalized_distances = distances / max_dist  # 0 to 1

    # Apply gradient
    background = (
        start_color * (1 - normalized_distances) + end_color * normalized_distances
    ).astype(np.uint16)

    return background


def create_gaussian_noise_background(
    height: int,
    width: int,
    mean: int = None,
    std: int = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a Gaussian noise background.

    Args:
        height: Image height
        width: Image width
        mean: Mean value for the Gaussian distribution
        std: Standard deviation for the Gaussian distribution
        color_range: Range for random color generation (min, max)

    Returns:
        Gaussian noise background as uint16 array
    """
    if mean is None:
        mean = random.randint(color_range[0], color_range[1])

    if std is None:
        std = random.randint(500, 3000)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, (height, width))

    # Clip to valid range and convert to uint16
    return np.clip(noise, color_range[0], color_range[1]).astype(np.uint16)


def create_perlin_noise_background(
    height: int,
    width: int,
    scale: int = None,
    octaves: int = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a Perlin-like noise background using frequency interpolation.

    Args:
        height: Image height
        width: Image width
        scale: Base scale factor for the noise (larger means more zoomed in)
        octaves: Number of noise layers to combine
        color_range: Range for random color generation (min, max)

    Returns:
        Perlin-like noise background as uint16 array
    """
    if scale is None:
        scale = random.randint(5, 30)

    if octaves is None:
        octaves = random.randint(1, 3)

    # Generate base noise
    base_height = height // scale + 1
    base_width = width // scale + 1
    base = np.random.randint(color_range[0], color_range[1], (base_height, base_width))

    # Resize to full size using bilinear interpolation
    scaled = cv2.resize(base, (width, height), interpolation=cv2.INTER_LINEAR)

    # Add octaves for more detail
    for i in range(1, octaves):
        small_scale = scale // (2**i)
        if small_scale < 1:
            break

        small_height = height // small_scale + 1
        small_width = width // small_scale + 1
        small = np.random.randint(0, 2000, (small_height, small_width))
        small_scaled = cv2.resize(
            small, (width, height), interpolation=cv2.INTER_LINEAR
        )
        scaled = scaled + small_scaled / (2**i)

    # Clip to valid range and convert to uint16
    return np.clip(scaled, color_range[0], color_range[1]).astype(np.uint16)


def create_grid_pattern_background(
    height: int,
    width: int,
    grid_size: int = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a grid pattern background.

    Args:
        height: Image height
        width: Image width
        grid_size: Size of grid cells
        color_range: Range for random color generation (min, max)

    Returns:
        Grid pattern background as uint16 array
    """
    if grid_size is None:
        grid_size = random.randint(20, 100)

    # Generate colors
    base_color = random.randint(color_range[0], color_range[1])
    alt_color = random.randint(color_range[0], color_range[1])

    # Create base background with base color
    background = np.ones((height, width), dtype=np.uint16) * base_color

    # Draw grid lines with alternate color
    for i in range(0, height, grid_size):
        if i + 2 <= height:
            background[i : i + 2, :] = alt_color

    for j in range(0, width, grid_size):
        if j + 2 <= width:
            background[:, j : j + 2] = alt_color

    return background


def create_stripes_pattern_background(
    height: int,
    width: int,
    stripe_width: int = None,
    direction: Direction = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a striped pattern background.

    Args:
        height: Image height
        width: Image width
        stripe_width: Width of each stripe
        direction: Direction of stripes (HORIZONTAL or VERTICAL)
        color_range: Range for random color generation (min, max)

    Returns:
        Striped pattern background as uint16 array
    """
    if stripe_width is None:
        stripe_width = random.randint(10, 50)

    if direction is None:
        direction = random.choice([Direction.HORIZONTAL, Direction.VERTICAL])

    # Generate colors
    base_color = random.randint(color_range[0], color_range[1])
    alt_color = random.randint(color_range[0], color_range[1])

    # Create base stripe pattern - alternating values [base_color, alt_color]
    base_pattern = np.array([base_color, alt_color], dtype=np.uint16)

    if direction == Direction.HORIZONTAL:
        # Repeat each value stripe_width times horizontally
        stripes = np.repeat(base_pattern, stripe_width)
        # Tile to fill the width
        stripes = np.tile(stripes, (width // (2 * stripe_width)) + 1)
        # Crop to exact width
        stripes = stripes[:width]
        # Repeat vertically for all rows
        background = np.tile(stripes, (height, 1))
    else:  # VERTICAL
        # Repeat each value stripe_width times
        stripes = np.repeat(base_pattern, stripe_width)
        # Tile to fill the height
        stripes = np.tile(stripes, (height // (2 * stripe_width)) + 1)
        # Crop to exact height
        stripes = stripes[:height]
        # Reshape for vertical stripes
        background = np.tile(stripes[:, np.newaxis], (1, width))

    return background


def create_checker_pattern_background(
    height: int,
    width: int,
    check_size: int = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a checker pattern background.

    Args:
        height: Image height
        width: Image width
        check_size: Size of each checker square
        color_range: Range for random color generation (min, max)

    Returns:
        Checker pattern background as uint16 array
    """
    if check_size is None:
        check_size = random.randint(20, 80)

    # Generate colors
    base_color = random.randint(color_range[0], color_range[1])
    alt_color = random.randint(color_range[0], color_range[1])

    # Create a basic 2x2 checker pattern
    basic_checker = np.array(
        [[base_color, alt_color], [alt_color, base_color]], dtype=np.uint16
    )

    # Repeat each value check_size times in both dimensions
    checker_h = np.repeat(basic_checker, check_size, axis=0)
    checker = np.repeat(checker_h, check_size, axis=1)

    # Tile to fill the entire image
    tile_rows = (height // (2 * check_size)) + 1
    tile_cols = (width // (2 * check_size)) + 1
    checker = np.tile(checker, (tile_rows, tile_cols))

    # Crop to exact dimensions
    return checker[:height, :width]


def create_shape_background(
    height: int,
    width: int,
    num_layers: int = None,
    shapes_per_layer: int = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a background with random shapes.

    Args:
        height: Image height
        width: Image width
        num_layers: Number of shape layers to add
        shapes_per_layer: Number of shapes per layer
        color_range: Range for random color generation (min, max)

    Returns:
        Background with random shapes as uint16 array
    """
    if num_layers is None:
        num_layers = random.randint(2, 5)

    if shapes_per_layer is None:
        shapes_per_layer = random.randint(3, 10)

    # Create a base background
    background = np.ones((height, width), dtype=np.uint16) * random.randint(
        color_range[0], color_range[1]
    )

    # Add layers of shapes
    for _ in range(num_layers):
        for _ in range(shapes_per_layer):
            # Choose a random shape type
            shape_type = random.choice(list(ShapeType))
            shape_color = random.randint(color_range[0], color_range[1])

            if shape_type == ShapeType.CIRCLE:
                # Random circle
                center_x = random.randint(0, width - 1)
                center_y = random.randint(0, height - 1)
                radius = random.randint(10, max(50, min(height, width) // 4))
                thickness = random.choice([-1, random.randint(1, 5)])  # -1 means filled

                # Draw circle
                cv2.circle(
                    background, (center_x, center_y), radius, shape_color, thickness
                )

            elif shape_type == ShapeType.RECTANGLE:
                # Random rectangle
                x1 = random.randint(0, width - 1)
                y1 = random.randint(0, height - 1)
                x2 = random.randint(x1, min(x1 + width // 2, width - 1))
                y2 = random.randint(y1, min(y1 + height // 2, height - 1))
                thickness = random.choice([-1, random.randint(1, 5)])  # -1 means filled

                # Draw rectangle
                cv2.rectangle(background, (x1, y1), (x2, y2), shape_color, thickness)

            else:  # ShapeType.LINE
                # Random line
                x1 = random.randint(0, width - 1)
                y1 = random.randint(0, height - 1)
                x2 = random.randint(0, width - 1)
                y2 = random.randint(0, height - 1)
                thickness = random.randint(1, 5)

                # Draw line
                cv2.line(background, (x1, y1), (x2, y2), shape_color, thickness)

    return background


def apply_noise_texture(
    background: np.ndarray,
    amplitude: int = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Apply subtle noise texture to an existing background.

    Args:
        background: Input background image
        amplitude: Amplitude of the noise
        color_range: Range for clipping the result (min, max)

    Returns:
        Background with added noise texture as uint16 array
    """
    height, width = background.shape

    if amplitude is None:
        amplitude = random.randint(500, 2000)

    # Generate noise
    noise = np.random.normal(0, amplitude, (height, width))

    # Apply noise
    result = background.astype(np.int32) + noise

    # Clip to valid range and convert to uint16
    return np.clip(result, color_range[0], color_range[1]).astype(np.uint16)


def apply_mandelbrot_effect(
    background: np.ndarray,
    blend_factor: float = None,
) -> np.ndarray:
    """
    Apply Mandelbrot fractal effect to a background image.

    Args:
        background: Input background image
        blend_factor: Factor for blending the effect (0-1)

    Returns:
        Background with Mandelbrot effect as uint16 array
    """
    height, width = background.shape

    if blend_factor is None:
        blend_factor = random.uniform(0.3, 0.7)

    # Convert to 8-bit for PIL compatibility
    scaled_image = (background / 256).astype(np.uint8)
    pil_image = Image.fromarray(scaled_image)

    # Create Mandelbrot effect
    try:
        effect_img = pil_image.effect_mandelbrot(
            (
                random.uniform(-2.0, 0.5),
                random.uniform(-1.5, 1.5),
                random.uniform(-1.0, 2.0),
                random.uniform(-1.5, 1.5),
            ),
            width,
            random.randint(100, 500),
        )

        # Blend with original
        blended = Image.blend(pil_image, effect_img, blend_factor)

        # Convert back to 16-bit
        return np.array(blended).astype(np.uint16) * 256

    except Exception as e:
        print(f"Mandelbrot effect error: {e}")
        return background


def apply_noise_effect(
    background: np.ndarray,
    noise_size: int = None,
    blend_factor: float = None,
) -> np.ndarray:
    """
    Apply noise effect to a background image.

    Args:
        background: Input background image
        noise_size: Size of the noise
        blend_factor: Factor for blending the effect (0-1)

    Returns:
        Background with noise effect as uint16 array
    """
    height, width = background.shape

    if noise_size is None:
        noise_size = random.randint(5, 30)

    if blend_factor is None:
        blend_factor = random.uniform(0.2, 0.6)

    # Convert to 8-bit for PIL compatibility
    scaled_image = (background / 256).astype(np.uint8)
    pil_image = Image.fromarray(scaled_image)

    try:
        # Apply noise effect
        noise_img = pil_image.effect_noise((width, height), noise_size)

        # Blend with original
        blended = Image.blend(pil_image, noise_img, blend_factor)

        # Convert back to 16-bit
        return np.array(blended).astype(np.uint16) * 256

    except Exception as e:
        print(f"Noise effect error: {e}")
        return background


def apply_filter_effect(
    background: np.ndarray,
    filter_type: FilterType = None,
) -> np.ndarray:
    """
    Apply filter effect to a background image.

    Args:
        background: Input background image
        filter_type: Type of filter to apply

    Returns:
        Background with filter effect as uint16 array
    """
    if filter_type is None:
        filter_type = random.choice(list(FilterType))

    if filter_type == FilterType.NONE:
        return background

    # Convert to 8-bit for PIL compatibility
    scaled_image = (background / 256).astype(np.uint8)
    pil_image = Image.fromarray(scaled_image)

    try:
        if filter_type == FilterType.BLUR:
            filtered = pil_image.filter(
                ImageFilter.GaussianBlur(random.uniform(0.5, 2.0))
            )
        elif filter_type == FilterType.SHARPEN:
            filtered = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        else:  # FilterType.EMBOSS
            filtered = pil_image.filter(ImageFilter.EMBOSS)

        # Convert back to 16-bit
        return np.array(filtered).astype(np.uint16) * 256

    except Exception as e:
        print(f"Filter effect error: {e}")
        return background


def generate_random_background(
    height: int,
    width: int,
    bg_type: BackgroundType = None,
    color_range: Tuple[int, int] = (0, UINT16_MAX),
) -> np.ndarray:
    """
    Generate a random background based on the specified type.

    Args:
        height: Image height
        width: Image width
        bg_type: Type of background to generate
        color_range: Range for random color generation (min, max)

    Returns:
        Random background as uint16 array
    """
    if bg_type is None:
        bg_type = random.choice(list(BackgroundType))

    if bg_type == BackgroundType.SOLID:
        background = create_solid_background(height, width, color_range)

    elif bg_type == BackgroundType.GRADIENT:
        gradient_type = random.choice(list(GradientType))

        if gradient_type == GradientType.LINEAR:
            background = create_linear_gradient_background(
                height, width, None, color_range
            )
        else:  # GradientType.RADIAL
            background = create_radial_gradient_background(
                height, width, None, None, color_range
            )

    elif bg_type == BackgroundType.NOISE:
        noise_type = random.choice(list(NoiseType))

        if noise_type == NoiseType.GAUSSIAN:
            background = create_gaussian_noise_background(
                height, width, None, None, color_range
            )
        else:  # NoiseType.PERLIN
            background = create_perlin_noise_background(
                height, width, None, None, color_range
            )

    elif bg_type == BackgroundType.PATTERN:
        pattern_type = random.choice(list(PatternType))

        if pattern_type == PatternType.GRID:
            background = create_grid_pattern_background(
                height, width, None, color_range
            )
        elif pattern_type == PatternType.STRIPES:
            background = create_stripes_pattern_background(
                height, width, None, None, color_range
            )
        else:  # PatternType.CHECKER
            background = create_checker_pattern_background(
                height, width, None, color_range
            )

    else:  # BackgroundType.SHAPES
        background = create_shape_background(height, width, None, None, color_range)

    # Add subtle noise texture
    if random.random() < 0.5:
        background = apply_noise_texture(background, None, color_range)

    # Apply random PIL effect
    if random.random() < 0.25:
        effect_type = random.choice(
            [
                lambda img: apply_mandelbrot_effect(img),
                lambda img: apply_noise_effect(img),
                lambda img: apply_filter_effect(img),
            ]
        )

        background = effect_type(background)

    return background
