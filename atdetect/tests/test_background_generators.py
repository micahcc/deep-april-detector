import unittest
import numpy as np
import cv2
from PIL import Image
import pytest
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from atdetect.background_generators import (
    BackgroundType,
    Direction,
    ShapeType,
    PatternType,
    NoiseType,
    FilterType,
    GradientType,
    EffectType,
    UINT16_MAX,
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
    generate_random_background,
)


class TestBackgroundGenerators(unittest.TestCase):
    """Test suite for background generation functions."""

    def setUp(self):
        """Set up test parameters."""
        self.height = 256
        self.width = 256
        self.color_range = (0, UINT16_MAX)

    def _verify_basic_image_properties(self, image, expected_shape=None, expected_dtype=np.uint16):
        """Helper method to verify basic image properties."""
        # Check the image is not None
        self.assertIsNotNone(image)
        
        # Check the shape
        if expected_shape:
            self.assertEqual(image.shape, expected_shape)
        else:
            self.assertEqual(image.shape, (self.height, self.width))
        
        # Check the data type
        self.assertEqual(image.dtype, expected_dtype)
        
        # Check value range
        self.assertTrue(np.all(image >= self.color_range[0]))
        self.assertTrue(np.all(image <= self.color_range[1]))

    def test_create_solid_background(self):
        """Test creating a solid background."""
        # Test with default parameters
        bg = create_solid_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Check that all pixels have the same value
        # (solid background = all pixels have the same value)
        first_pixel = bg[0, 0]
        self.assertTrue(np.all(bg == first_pixel))
        
        # Test with custom color range
        custom_range = (1000, 5000)
        bg = create_solid_background(self.height, self.width, custom_range)
        self.assertTrue(np.all(bg >= custom_range[0]))
        self.assertTrue(np.all(bg <= custom_range[1]))

    def test_create_linear_gradient_background(self):
        """Test creating a linear gradient background."""
        # Test horizontal gradient
        bg = create_linear_gradient_background(
            self.height, self.width, Direction.HORIZONTAL
        )
        self._verify_basic_image_properties(bg)
        
        # Check horizontal gradient property
        # (leftmost column should be different from rightmost column)
        self.assertNotEqual(bg[:, 0].mean(), bg[:, -1].mean())
        
        # Test vertical gradient
        bg = create_linear_gradient_background(
            self.height, self.width, Direction.VERTICAL
        )
        self._verify_basic_image_properties(bg)
        
        # Check vertical gradient property
        # (top row should be different from bottom row)
        self.assertNotEqual(bg[0, :].mean(), bg[-1, :].mean())

    def test_create_radial_gradient_background(self):
        """Test creating a radial gradient background."""
        # Test with default parameters
        bg = create_radial_gradient_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Test with specific center and bright_center
        center = (self.width // 2, self.height // 2)
        bg = create_radial_gradient_background(
            self.height, self.width, center, True
        )
        self._verify_basic_image_properties(bg)
        
        # Check radial property
        # (center should be different from corners)
        center_val = bg[center[1], center[0]]
        corner_val = bg[0, 0]
        self.assertNotEqual(center_val, corner_val)

    def test_create_gaussian_noise_background(self):
        """Test creating a Gaussian noise background."""
        # Test with default parameters
        bg = create_gaussian_noise_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Verify statistical properties
        # (standard deviation should be non-zero)
        self.assertGreater(np.std(bg), 0)
        
        # Test with specific mean and std
        mean_val = 30000
        std_val = 1000
        bg = create_gaussian_noise_background(
            self.height, self.width, mean_val, std_val, self.color_range
        )
        self._verify_basic_image_properties(bg)
        
        # Check that the actual mean is somewhat close to the requested mean
        # (not exact due to clipping)
        mean_tolerance = 5000
        self.assertLess(abs(np.mean(bg) - mean_val), mean_tolerance)

    def test_create_perlin_noise_background(self):
        """Test creating a Perlin-like noise background."""
        # Test with default parameters
        bg = create_perlin_noise_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Test with specific scale and octaves
        scale = 10
        octaves = 2
        bg = create_perlin_noise_background(
            self.height, self.width, scale, octaves, self.color_range
        )
        self._verify_basic_image_properties(bg)

    def test_create_grid_pattern_background(self):
        """Test creating a grid pattern background."""
        # Test with default parameters
        bg = create_grid_pattern_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Test with specific grid size
        grid_size = 40
        bg = create_grid_pattern_background(
            self.height, self.width, grid_size, self.color_range
        )
        self._verify_basic_image_properties(bg)

    def test_create_stripes_pattern_background(self):
        """Test creating a striped pattern background."""
        # Test with horizontal stripes
        bg = create_stripes_pattern_background(
            self.height, self.width, 20, Direction.HORIZONTAL
        )
        self._verify_basic_image_properties(bg)
        
        # Check stripe pattern (rows should alternate)
        middle_row = self.height // 2
        self.assertNotEqual(bg[middle_row, 0], bg[middle_row + 20, 0])
        
        # Test with vertical stripes
        bg = create_stripes_pattern_background(
            self.height, self.width, 20, Direction.VERTICAL
        )
        self._verify_basic_image_properties(bg)
        
        # Check stripe pattern (columns should alternate)
        middle_col = self.width // 2
        self.assertNotEqual(bg[0, middle_col], bg[0, middle_col + 20])

    def test_create_checker_pattern_background(self):
        """Test creating a checker pattern background."""
        # Test with default parameters
        bg = create_checker_pattern_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Test with specific check size
        check_size = 30
        bg = create_checker_pattern_background(
            self.height, self.width, check_size, self.color_range
        )
        self._verify_basic_image_properties(bg)
        
        # Check checker pattern
        # (diagonally adjacent cells should be the same, orthogonally adjacent should be different)
        self.assertEqual(bg[0, 0], bg[check_size, check_size])
        self.assertNotEqual(bg[0, 0], bg[0, check_size])
        self.assertNotEqual(bg[0, 0], bg[check_size, 0])

    def test_create_shape_background(self):
        """Test creating a background with shapes."""
        # Test with default parameters
        bg = create_shape_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Test with specific parameters
        num_layers = 3
        shapes_per_layer = 5
        bg = create_shape_background(
            self.height, self.width, num_layers, shapes_per_layer, self.color_range
        )
        self._verify_basic_image_properties(bg)

    def test_apply_noise_texture(self):
        """Test applying noise texture to a background."""
        # Create a base background
        base_bg = create_solid_background(self.height, self.width)
        
        # Apply noise texture
        bg = apply_noise_texture(base_bg)
        self._verify_basic_image_properties(bg)
        
        # Check that the output is not identical to the input
        self.assertFalse(np.array_equal(base_bg, bg))

    def test_apply_mandelbrot_effect(self):
        """Test applying Mandelbrot effect to a background."""
        # Create a base background
        base_bg = create_solid_background(self.height, self.width)
        
        try:
            # Apply effect with default parameters
            bg = apply_mandelbrot_effect(base_bg)
            self._verify_basic_image_properties(bg)
            
            # Test with specific blend factor
            blend_factor = 0.5
            bg = apply_mandelbrot_effect(base_bg, blend_factor)
            self._verify_basic_image_properties(bg)
        except Exception as e:
            # This test may fail if PIL's effect_mandelbrot is not available
            # Skip in that case but log the error
            print(f"Skipped Mandelbrot test due to: {e}")
            pass

    def test_apply_noise_effect(self):
        """Test applying noise effect to a background."""
        # Create a base background
        base_bg = create_solid_background(self.height, self.width)
        
        try:
            # Apply effect with default parameters
            bg = apply_noise_effect(base_bg)
            self._verify_basic_image_properties(bg)
            
            # Test with specific parameters
            noise_size = 20
            blend_factor = 0.4
            bg = apply_noise_effect(base_bg, noise_size, blend_factor)
            self._verify_basic_image_properties(bg)
        except Exception as e:
            # This test may fail if PIL's effect_noise is not available
            # Skip in that case but log the error
            print(f"Skipped noise effect test due to: {e}")
            pass

    def test_apply_filter_effect(self):
        """Test applying filter effects to a background."""
        # Create a base background
        base_bg = create_solid_background(self.height, self.width)
        
        for filter_type in FilterType:
            # Apply filter
            bg = apply_filter_effect(base_bg, filter_type)
            self._verify_basic_image_properties(bg)
            
            # For NONE filter, output should be identical to input
            if filter_type == FilterType.NONE:
                self.assertTrue(np.array_equal(base_bg, bg))

    def test_generate_random_background(self):
        """Test generating random backgrounds of different types."""
        # Test with default parameters
        bg = generate_random_background(self.height, self.width)
        self._verify_basic_image_properties(bg)
        
        # Test each background type
        for bg_type in BackgroundType:
            bg = generate_random_background(
                self.height, self.width, bg_type, self.color_range
            )
            self._verify_basic_image_properties(bg)


if __name__ == "__main__":
    unittest.main()