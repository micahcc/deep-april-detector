"""
Data loader for AprilTag templates.
"""

import os
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import math
from typing import List, Tuple, Dict, Optional, NamedTuple

from atdetect.bounding_box import BoundingBox
from atdetect.key_point import KeyPoint
from atdetect.april_tag_annotation import AprilTagAnnotation
from atdetect.synthetic_image import SyntheticImage


class TemplateInfo(NamedTuple):
    """Container for template data and metadata."""

    template: np.ndarray
    mask: np.ndarray
    class_num: int
    keypoints: List[KeyPoint]


# Import background generation functionality
from atdetect.background_type import BackgroundType
from atdetect.direction import Direction
from atdetect.filter_type import FilterType
from atdetect.gradient_type import GradientType
from atdetect.noise_type import NoiseType
from atdetect.pattern_type import PatternType
from atdetect.shape_type import ShapeType
from atdetect.effect_type import EffectType

from atdetect.background_generators import (
    UINT16_MAX,
    generate_random_background,
)


class AprilTagDataLoader:
    """Data loader for AprilTag templates."""

    def __init__(
        self,
        tag_type: str,
        templates_dir: str = "templates",
        min_scale_factor: float = 1.5,
        max_scale_factor: float = 5.0,
        min_img_size: int = 256,
        max_img_size: int = 1024,
        tags_per_image: Tuple[int, int] = (1, 3),
        bg_color_range: Tuple[int, int] = (5000, 20000),  # 16-bit range
        brightness_variation_range: float = 0.3,
        min_rotation: float = 0.0,  # Minimum rotation angle in degrees
        max_rotation: float = 360.0,  # Maximum rotation angle in degrees
        skew_range: float = 0.1,  # Range of skew transformation (0-1),
        grid_rows: Tuple[int, int] = (1, 4),  # Range of grid rows
        grid_cols: Tuple[int, int] = (1, 4),  # Range of grid columns
        grid_spacing_range: Tuple[int, int] = (5, 20),
        grid_border: int = 20,  # Border around the entire grid in pixels
        brightness_scale_range: Tuple[float, float] = (
            0.5,
            2.0,
        ),  # Range for brightness scaling
        invert_probability: float = 0.2,  # Probability of inverting the templates
    ):  # Controls the amount of brightness variation
        """
        Initialize AprilTag data loader.

        Args:
            tag_type: Type of AprilTag (tag16h5, tagCircle21h7, or tagStandard41h12).
            templates_dir: Directory containing template images.
            min_scale_factor: Minimum scale factor for templates.
            max_scale_factor: Maximum scale factor for templates.
            min_img_size: Minimum image size (both H and W).
            max_img_size: Maximum image size (both H and W).
            tags_per_image: Range (min, max) of number of tags per image.
            bg_color_range: Range of background color (min, max) for 16-bit images.
        """
        self.tag_type = tag_type
        self.templates_dir = Path(templates_dir) / tag_type
        self.min_scale_factor = max(
            1.0, min_scale_factor
        )  # Ensure scale factor is at least 1.0
        self.max_scale_factor = max(self.min_scale_factor + 0.5, max_scale_factor)
        self.min_img_size = max(64, min_img_size)  # Ensure minimum size is at least 64
        self.max_img_size = max(self.min_img_size + 64, max_img_size)
        self.tags_per_image = tags_per_image
        self.bg_color_range = bg_color_range
        self.brightness_variation_range = max(
            0.0, min(1.0, brightness_variation_range)
        )  # Clamp to [0, 1]
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.skew_range = max(0.0, min(1.0, skew_range))  # Clamp to [0, 1]

        # Grid layout parameters
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_spacing_range = grid_spacing_range
        self.grid_border = grid_border

        # Brightness scaling parameters
        self.brightness_scale_range = brightness_scale_range
        self.invert_probability = max(
            0.0, min(1.0, invert_probability)
        )  # Clamp to [0, 1]

        # Load template paths and class numbers
        self.template_paths, self.class_nums = self._load_templates()

    def _load_templates(self) -> Tuple[List[Path], List[int]]:
        """Load template paths and extract class numbers."""
        if not self.templates_dir.exists():
            raise ValueError(f"Templates directory not found: {self.templates_dir}")

        paths = []
        class_nums = []

        for file_path in self.templates_dir.glob("*.png"):
            # Parse class number from filename (e.g., tag16h5_0.png -> 0)
            try:
                class_num = int(file_path.stem.split("_")[-1])
                paths.append(file_path)
                class_nums.append(class_num)
            except ValueError:
                print(f"Skipping {file_path}: Could not parse class number")

        if not paths:
            raise ValueError(f"No template images found in {self.templates_dir}")

        return paths, class_nums

    @staticmethod
    def _scale_template(
        template: np.ndarray, mask: np.ndarray, scale_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale template and mask using nearest neighbor interpolation."""
        h, w = template.shape[:2]
        new_h, new_w = max(1, int(h * scale_factor)), max(
            1, int(w * scale_factor)
        )  # Ensure size is at least 1x1
        template_pil = Image.fromarray(template)
        mask_pil = Image.fromarray(mask)
        scaled_template = np.array(
            template_pil.resize((new_w, new_h), resample=Image.NEAREST)
        )
        scaled_mask = np.array(mask_pil.resize((new_w, new_h), resample=Image.NEAREST))
        return scaled_template, scaled_mask

    @staticmethod
    def _transform_template(
        template: np.ndarray,
        mask: np.ndarray,
        annotations: List[AprilTagAnnotation],
        angle: float,
        skew_range: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[KeyPoint]]:
        """Apply rotation and skew transformations to template, mask, and keypoints.

        Args:
            template: Template image to transform
            mask: Mask to transform
            keypoints: Keypoints to transform
            angle: Rotation angle in degrees (clockwise)
            skew_range: Range of skew transformation (-skew_range to +skew_range)

        Returns:
            Tuple of (transformed_template, transformed_mask, transformed_keypoints)
        """

        # Get dimensions of the template
        h, w = template.shape[:2]
        center = (w // 2, h // 2)  # Center of the template

        # Calculate rotation matrix manually (2x3 matrix)
        angle_rad = math.radians(-angle)  # Negative angle for clockwise rotation
        skew_x = random.uniform(-skew_range, skew_range)
        skew_y = random.uniform(-skew_range, skew_range)
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        # Create rotation matrix [cos, -sin, x_shift; sin, cos, y_shift]
        rotation_matrix = np.zeros((3, 3), dtype=np.float32)
        rotation_matrix[0, 0] = cos_val
        rotation_matrix[0, 1] = -sin_val
        rotation_matrix[1, 0] = sin_val
        rotation_matrix[1, 1] = cos_val
        rotation_matrix[0, 2] = (1 - cos_val) * center[0] + sin_val * center[1]
        rotation_matrix[1, 2] = -sin_val * center[0] + (1 - cos_val) * center[1]

        # Calculate new dimensions after rotation to ensure the entire template is visible
        # Convert rotation matrix to radians
        angle_rad = math.radians(angle)
        cos_angle = abs(math.cos(angle_rad))
        sin_angle = abs(math.sin(angle_rad))

        # Calculate new dimensions for rotation
        new_w = int(w * cos_angle * (1 + skew_x) + h * sin_angle)
        new_h = int(w * sin_angle + h * cos_angle * (1 + skew_y))

        # Adjust rotation matrix to account for new dimensions
        rotation_matrix[0, 2] += (new_w - w) // 2
        rotation_matrix[1, 2] += (new_h - h) // 2
        rotation_matrix[2, 2] = 1

        # Create skew transformation matrix (3x3 for homography)
        # [1, skew_x, 0]   [x]   [x + skew_x*y]
        # [skew_y, 1, 0] * [y] = [skew_y*x + y]
        # [0, 0, 1]        [1]   [1]
        skew_matrix = np.array(
            [[1.0, skew_x, 0], [skew_y, 1.0, 0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        transform_matrix = skew_matrix @ rotation_matrix

        # Rotate keypoints
        transformed_annotations = []
        for an in annotations:
            transformed_keypoints = []
            x_min = float("inf")
            x_max = -float("inf")
            y_min = float("inf")
            y_max = -float("inf")
            for kp in an.keypoints:
                # Translate keypoint to center, then apply tfm matrix
                kp_matrix = np.array([kp.x, kp.y, 1.0])
                transformed_x, transformed_y, _ = transform_matrix @ kp_matrix
                transformed_keypoints.append(
                    KeyPoint(x=float(transformed_x), y=float(transformed_y))
                )
                x_min = min(transformed_x, x_min)
                x_max = max(transformed_x, x_max)
                y_min = min(transformed_y, y_min)
                y_max = max(transformed_y, y_max)

            transformed_annotations.append(
                AprilTagAnnotation(
                    class_name=an.class_name,
                    class_num=an.class_num,
                    bbox=BoundingBox(
                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
                    ),
                    keypoints=transformed_keypoints,
                )
            )

        # Apply transform
        # convert to uint32 otherwise transform fails silently (black image)
        template_pil = Image.fromarray(template.astype(np.int32))
        mask_pil = Image.fromarray(mask)

        transform_matrix_inv = np.linalg.inv(transform_matrix)
        pil_params = (
            transform_matrix_inv[0, 0],
            transform_matrix_inv[0, 1],
            transform_matrix_inv[0, 2],
            transform_matrix_inv[1, 0],
            transform_matrix_inv[1, 1],
            transform_matrix_inv[1, 2],
        )

        transformed_template_pil = template_pil.transform(
            (new_w, new_h),
            Image.AFFINE,
            pil_params,
            resample=Image.BICUBIC,
            fillcolor=0,
        )

        transformed_mask_pil = mask_pil.transform(
            (new_w, new_h),
            Image.AFFINE,
            pil_params,
            resample=Image.BICUBIC,
            fillcolor=0,
        )

        # transformed_mask_pil.save("mask.png", format="PNG")
        # transformed_template_pil.save("tftemplate.png", format="PNG")
        # template_pil.save("template.png", format="PNG")
        transformed_template = (
            np.array(transformed_template_pil).clip(0, UINT16_MAX).astype(np.uint16)
        )
        transformed_mask = np.array(transformed_mask_pil)
        # transformed_template_pil2 = Image.fromarray(transformed_template)
        # transformed_template_pil2.save("tftemplate2.png", format="PNG")
        return transformed_template, transformed_mask, transformed_annotations

    def _create_template_grid(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[AprilTagAnnotation]]:
        """Create a grid of AprilTag templates.

        Returns:
            Tuple containing:
                - Combined grid template image
                - Combined grid mask
                - List of annotations for each template in the grid
        """
        # Decide on grid dimensions, ensuring we don't exceed the number of available templates
        num_templates = len(self.template_paths)

        # First choose a random number of columns within the specified range
        cols = random.randint(self.grid_cols[0], min(self.grid_cols[1], num_templates))

        # Then choose the rows, ensuring rows * cols <= num_templates
        max_possible_rows = (
            num_templates // cols
        )  # Integer division to ensure we don't exceed num_templates
        rows = random.randint(
            self.grid_rows[0], min(self.grid_rows[1], max_possible_rows)
        )

        # Assert that our dimensions don't exceed available templates
        assert (
            rows * cols <= num_templates
        ), "Grid dimensions exceed available templates"

        # Use a single scale factor for all templates in the grid for uniformity
        scale_factor = random.uniform(self.min_scale_factor, self.max_scale_factor)

        # Generate templates for the grid (all with same scale factor)
        # Use random.sample to select unique indices without replacement
        template_indices = random.sample(range(len(self.template_paths)), rows * cols)

        # Generate a template for each selected index
        templates_info = []
        for idx in template_indices:
            template_info = self._generate_template_by_index(
                idx, scale_factor=scale_factor
            )
            if template_info is not None:
                templates_info.append(template_info)

        # If any templates failed to generate, we still have the assertion
        # that will catch this case during development

        # Reshape templates into grid layout
        templates_grid = [
            templates_info[i : i + cols] for i in range(0, len(templates_info), cols)
        ]

        # Calculate the maximum height for each row and maximum width for each column
        row_heights = []
        for row in templates_grid:
            max_height = 0
            for template_info in row:
                max_height = max(max_height, template_info.template.shape[0])
            row_heights.append(max_height)

        col_widths = [0] * cols
        for row in templates_grid:
            for col_idx, template_info in enumerate(row):
                col_widths[col_idx] = max(
                    col_widths[col_idx], template_info.template.shape[1]
                )

        # Use a single random spacing for the entire grid
        grid_spacing = random.randint(
            self.grid_spacing_range[0], self.grid_spacing_range[1]
        )

        # Calculate total grid dimensions with spacing and border
        grid_width = sum(col_widths) + grid_spacing * (cols - 1) + 2 * self.grid_border
        grid_height = (
            sum(row_heights) + grid_spacing * (rows - 1) + 2 * self.grid_border
        )

        # Create empty grid template and mask
        grid_template = np.zeros((grid_height, grid_width), dtype=np.uint16)
        grid_mask = np.zeros((grid_height, grid_width), dtype=np.uint8)

        annotations = []

        # Place templates in the grid
        y_offset = self.grid_border
        for row_idx, row in enumerate(templates_grid):
            x_offset = self.grid_border
            for col_idx, template_info in enumerate(row):
                # Get template data
                template = template_info.template
                mask = template_info.mask
                class_num = template_info.class_num
                keypoints = template_info.keypoints

                # Center the template in its cell
                template_h, template_w = template.shape[:2]
                center_y = y_offset + (row_heights[row_idx] - template_h) // 2
                center_x = x_offset + (col_widths[col_idx] - template_w) // 2

                # Place template in the grid
                grid_template[
                    center_y : center_y + template_h, center_x : center_x + template_w
                ] = template
                grid_mask[
                    center_y : center_y + template_h, center_x : center_x + template_w
                ] = mask

                # Adjust keypoints for placement in the grid
                placed_keypoints = [
                    KeyPoint(x=kp.x + center_x, y=kp.y + center_y) for kp in keypoints
                ]

                # Calculate bounding box based on keypoints
                kp_coords = np.array([[kp.x, kp.y] for kp in placed_keypoints])
                x_min = float(np.min(kp_coords[:, 0]))
                y_min = float(np.min(kp_coords[:, 1]))
                x_max = float(np.max(kp_coords[:, 0]))
                y_max = float(np.max(kp_coords[:, 1]))

                bbox = BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                )

                # Create annotation
                annotation = AprilTagAnnotation(
                    class_name=self.tag_type,
                    class_num=class_num,
                    bbox=bbox,
                    keypoints=placed_keypoints,
                )
                annotations.append(annotation)

                x_offset += col_widths[col_idx] + grid_spacing

            y_offset += row_heights[row_idx] + grid_spacing

        return grid_template, grid_mask, annotations

    def _generate_template(self, scale_factor: Optional[float] = None) -> TemplateInfo:
        """Generate a single untransformed template.

        Args:
            scale_factor: Optional scale factor to apply. If None, a random scale
                         factor will be chosen.

        Returns:
            TemplateInfo containing the template, mask, class number, and keypoints
        """
        # Choose random template
        idx = random.randint(0, len(self.template_paths) - 1)
        return self._generate_template_by_index(idx, scale_factor)

    def _generate_template_by_index(
        self, idx: int, scale_factor: Optional[float] = None
    ) -> Optional[TemplateInfo]:
        """Generate a template for a specific index.

        Args:
            idx: Index of template to use
            scale_factor: Optional scale factor to apply. If None, a random scale
                         factor will be chosen.

        Returns:
            TemplateInfo containing the template, mask, class number, and keypoints
        """
        if idx < 0 or idx >= len(self.template_paths):
            print(f"Invalid template index: {idx}")
            return None

        template_path = self.template_paths[idx]
        class_num = self.class_nums[idx]

        # Load template and convert to 16-bit
        template_pil = Image.open(str(template_path))
        if template_pil.mode != "I":  # Check if already 16-bit grayscale
            # Convert to grayscale if needed
            if template_pil.mode != "L":
                template_pil = template_pil.convert("L")
            # Convert to 16-bit
            template_arr = np.array(template_pil)
            template = template_arr.astype(np.uint16) * 256
        else:
            template = np.array(template_pil)
        if template is None:
            print(f"Failed to load template: {template_path}")
            return None

        # Make sure the template is 16-bit
        old_range = (template.min(), template.max())
        template = (
            UINT16_MAX
            * (template.astype(np.float32) - old_range[0])
            / (old_range[1] - old_range[0])
        ).astype(np.uint16)

        template_h, template_w = template.shape[:2]

        # Create a full mask for the entire template (255 for the entire tag)
        mask = np.ones((template_h, template_w), dtype=np.uint8) * 255

        # Initial keypoints are the corners of the template
        # Use integers to ensure exact pixel boundaries
        keypoints = [
            KeyPoint(x=0, y=0),  # Top-left
            KeyPoint(x=template_w - 1, y=0),  # Top-right
            KeyPoint(x=template_w - 1, y=template_h - 1),  # Bottom-right
            KeyPoint(x=0, y=template_h - 1),  # Bottom-left
        ]

        # Apply scaling if needed
        if scale_factor is None:
            # Choose random scale factor
            scale_factor = random.uniform(self.min_scale_factor, self.max_scale_factor)

        # Scale template and mask
        scaled_template, scaled_mask = self._scale_template(
            template, mask, scale_factor
        )
        scaled_keypoints = [
            KeyPoint(x=kp.x * scale_factor, y=kp.y * scale_factor) for kp in keypoints
        ]

        return TemplateInfo(
            template=scaled_template,
            mask=scaled_mask,
            class_num=class_num,
            keypoints=scaled_keypoints,
        )

    def _apply_random_brightness_scaling(self, template: np.ndarray) -> np.ndarray:
        """Apply random brightness scaling to a template, possibly inverting it.

        Args:
            template: Template image to adjust

        Returns:
            Brightness scaled template
        """
        # Choose a random scale factor for brightness
        scale_factor = random.uniform(
            self.brightness_scale_range[0], self.brightness_scale_range[1]
        )

        # Randomly decide whether to invert the template
        invert = random.random() < self.invert_probability

        if invert:
            # Invert the template (65535 - value)
            template = UINT16_MAX - template

        # Scale the brightness
        scaled_template = np.clip(
            template.astype(np.float32) * scale_factor, 0.0, UINT16_MAX
        ).astype(np.uint16)

        return scaled_template

    def _generate_templates(self, num_templates: int) -> List[TemplateInfo]:
        """Generate multiple transformed templates.

        Args:
            num_templates: Number of templates to generate

        Returns:
            List of TemplateInfo objects
        """
        templates = []
        for _ in range(num_templates):
            template_info = self._generate_template()
            if template_info is not None:
                templates.append(template_info)

        return templates

    def _apply_brightness_variation(
        self, template: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Apply brightness variation to a template.

        Args:
            template: Template image to adjust
            mask: Mask for the template

        Returns:
            Brightness adjusted template
        """
        h, w = template.shape[:2]

        # Choose between linear or radial gradient for brightness
        gradient_type = random.choice(list(GradientType))

        # Create a brightness gradient map as a grayscale image
        gradient_map = Image.new("I", (w, h))
        gradient_draw = ImageDraw.Draw(gradient_map)

        # Random brightness gradient range from -variation to +variation
        min_brightness = 1.0 - self.brightness_variation_range
        max_brightness = 1.0 + self.brightness_variation_range

        if gradient_type == GradientType.LINEAR:
            # Choose gradient direction
            direction = random.choice(list(Direction))

            if direction == Direction.HORIZONTAL:
                for i in range(w):
                    ratio = i / max(1, w - 1)
                    brightness = int(
                        (min_brightness + (max_brightness - min_brightness) * ratio)
                        * 65535
                    )
                    gradient_draw.line([(i, 0), (i, h - 1)], fill=brightness)
            elif direction == Direction.VERTICAL:
                for i in range(h):
                    ratio = i / max(1, h - 1)
                    brightness = int(
                        (min_brightness + (max_brightness - min_brightness) * ratio)
                        * 65535
                    )
                    gradient_draw.line([(0, i), (w - 1, i)], fill=brightness)
            elif direction == Direction.DIAGONAL:
                # Create custom diagonal gradient
                for i in range(h):
                    for j in range(w):
                        # Calculate diagonal ratio
                        ratio = (i / max(1, h - 1) + j / max(1, w - 1)) / 2
                        brightness = int(
                            (min_brightness + (max_brightness - min_brightness) * ratio)
                            * 65535
                        )
                        gradient_draw.point((j, i), fill=brightness)
        elif gradient_type == GradientType.RADIAL:
            # Choose center point for radial gradient
            center_x = random.randint(0, w - 1)
            center_y = random.randint(0, h - 1)

            # Randomly choose between bright center or dark center
            bright_center = random.choice([True, False])
            inner_brightness = max_brightness if bright_center else min_brightness
            outer_brightness = min_brightness if bright_center else max_brightness

            # Create custom radial gradient
            max_dist = math.sqrt(w**2 + h**2) / 2
            for i in range(h):
                for j in range(w):
                    # Calculate distance from center
                    dist = math.sqrt((j - center_x) ** 2 + (i - center_y) ** 2)
                    ratio = min(1.0, dist / max_dist)
                    brightness = int(
                        (inner_brightness * (1 - ratio) + outer_brightness * ratio)
                        * 65535
                    )
                    gradient_draw.point((j, i), fill=brightness)

        # Convert gradient to numpy and resize to match template dimensions
        gradient_array = np.array(gradient_map).astype(np.float32)

        # Resize gradient to match template dimensions using PIL
        template_shape = template.shape
        gradient_pil = Image.fromarray(gradient_array.astype(np.uint32))
        gradient_resized_pil = gradient_pil.resize(
            (template_shape[1], template_shape[0]), resample=Image.BILINEAR
        )
        gradient_resized = np.array(gradient_resized_pil).astype(np.float32)

        # Normalize gradient to 0-1 range
        gradient_normalized = gradient_resized / 65535.0

        # Ensure gradient doesn't have all near-zero values
        if np.mean(gradient_normalized) < 0.01:
            # Fallback to a reasonable brightness value
            gradient_normalized = np.ones_like(gradient_normalized) * 0.5
            print("Warning: Gradient values too low, using fallback value")

        # Apply brightness adjustment using vectorized numpy operations
        # Map gradient from 0-1 range to -brightness_variation to +brightness_variation
        brightness_variation = self.brightness_variation_range * UINT16_MAX / 2
        brightness_adjustment = (gradient_normalized * 2.0 - 1.0) * brightness_variation

        # Add the brightness adjustment (clamped to uint16 range)
        adjusted_array = np.clip(
            template.astype(np.float32) + brightness_adjustment,
            0.0,
            UINT16_MAX,
        )

        # Convert back to uint16 for the final template
        return adjusted_array.astype(np.uint16)

    def generate_sample(self) -> SyntheticImage:
        """Generate a synthetic image with AprilTag templates arranged in a grid."""
        # Choose random image size divisible by 64
        min_img_size = max(1, self.min_img_size)
        max_img_size = max(1, self.max_img_size)
        img_width = 64 * math.ceil(random.randint(min_img_size, max_img_size) / 64)
        img_height = 64 * math.ceil(random.randint(min_img_size, max_img_size) / 64)

        # Create interesting background image (16-bit)
        image = self._generate_background(img_height, img_width)

        # Create a grid of templates (all templates are just scaled, not transformed)
        grid_template, grid_mask, grid_annotations = self._create_template_grid()

        # Apply random brightness scaling to the entire grid
        grid_template = self._apply_random_brightness_scaling(grid_template)

        # Apply brightness variation to the grid before transformation
        grid_template = self._apply_brightness_variation(grid_template, grid_mask)

        # Apply transformations (rotation and skew) to the entire grid as a single unit
        angle = random.uniform(self.min_rotation, self.max_rotation)
        transformed_grid, transformed_mask, transformed_annotations = (
            self._transform_template(
                grid_template,
                grid_mask,
                grid_annotations,
                angle,
                self.skew_range,
            )
        )

        # Get dimensions of the transformed grid
        grid_h, grid_w = transformed_grid.shape[:2]

        # Choose random position for the grid in the final image (background)
        # Allow grid to be partially off-screen
        x_pos = random.randint(-grid_w // 2, img_width - grid_w // 2)
        y_pos = random.randint(-grid_h // 2, img_height - grid_h // 2)

        # Paste the transformed grid into the image
        # Create intersection region between the image and grid
        x_min_img = max(0, x_pos)
        x_max_img = min(img_width, x_pos + grid_w)
        y_min_img = max(0, y_pos)
        y_max_img = min(img_height, y_pos + grid_h)

        # Calculate corresponding region in the transformed grid
        x_min_grid = max(0, -x_pos)
        x_max_grid = min(grid_w, img_width - x_pos)
        y_min_grid = max(0, -y_pos)
        y_max_grid = min(grid_h, img_height - y_pos)

        # Check if there's a valid intersection
        if (
            x_max_img > x_min_img
            and y_max_img > y_min_img
            and x_max_grid > x_min_grid
            and y_max_grid > y_min_grid
        ):

            # Get the relevant portions of the image and grid
            img_region = image[y_min_img:y_max_img, x_min_img:x_max_img]
            grid_region = transformed_grid[y_min_grid:y_max_grid, x_min_grid:x_max_grid]
            mask_region = (
                transformed_mask[y_min_grid:y_max_grid, x_min_grid:x_max_grid].astype(
                    np.float32
                )
                / 255.0
            )

            # Use the mask to combine the grid and image (maintaining single channel)
            image[y_min_img:y_max_img, x_min_img:x_max_img] = (
                (grid_region.astype(np.float32) * mask_region)
                + (img_region.astype(np.float32) * (1 - mask_region))
            ).astype(np.uint16)

        # Adjust annotations for placement in the final image
        final_annotations = []
        for annotation in transformed_annotations:
            # Adjust keypoints
            placed_keypoints = [
                KeyPoint(x=kp.x + x_pos, y=kp.y + y_pos) for kp in annotation.keypoints
            ]

            # Adjust bounding box
            bbox = annotation.bbox
            placed_bbox = BoundingBox(
                x_min=bbox.x_min + x_pos,
                y_min=bbox.y_min + y_pos,
                x_max=bbox.x_max + x_pos,
                y_max=bbox.y_max + y_pos,
            )

            # Check if the annotation is at least partially visible in the image
            if (
                placed_bbox.x_max > 0
                and placed_bbox.x_min < img_width
                and placed_bbox.y_max > 0
                and placed_bbox.y_min < img_height
            ):

                # Clip bounding box to image boundaries if necessary
                clipped_bbox = BoundingBox(
                    x_min=max(0, placed_bbox.x_min),
                    y_min=max(0, placed_bbox.y_min),
                    x_max=min(img_width, placed_bbox.x_max),
                    y_max=min(img_height, placed_bbox.y_max),
                )

                # Filter keypoints to only include those visible in the image
                visible_keypoints = [
                    kp
                    for kp in placed_keypoints
                    if 0 <= kp.x < img_width and 0 <= kp.y < img_height
                ]

                # Only include the annotation if we have at least some keypoints visible
                if visible_keypoints:
                    # Create new annotation with clipped bbox and visible keypoints
                    placed_annotation = AprilTagAnnotation(
                        class_name=annotation.class_name,
                        class_num=annotation.class_num,
                        bbox=clipped_bbox,
                        keypoints=visible_keypoints,
                    )
                    final_annotations.append(placed_annotation)

        return SyntheticImage(
            image=image,
            annotations=final_annotations,
            height=img_height,
            width=img_width,
        )

    def __iter__(self):
        """Iterator for generating an infinite stream of samples."""
        while True:
            yield self.generate_sample()

    def _generate_background(self, height: int, width: int) -> np.ndarray:
        """
        Generate a complex background image with textures, gradients, or patterns.

        Args:
            height: Height of the image
            width: Width of the image

        Returns:
            16-bit background image with interesting patterns
        """
        # Use the dedicated background generator module
        background = generate_random_background(height, width, self.bg_color_range)

        return background

    def __getitem__(self, idx) -> List[SyntheticImage]:
        """Generate a batch of synthetic images."""
        return self.generate_sample()

    def __len__(self) -> int:
        return 10000
