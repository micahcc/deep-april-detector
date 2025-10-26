"""
Data loader for AprilTag templates.
"""

import os
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import math

from atdetect.bounding_box import BoundingBox
from atdetect.key_point import KeyPoint
from atdetect.april_tag_annotation import AprilTagAnnotation
from atdetect.synthetic_image import SyntheticImage

# Import background generation functionality
from atdetect.background_complexity import BackgroundComplexity
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
    apply_mandelbrot_effect,
    apply_noise_effect,
    apply_filter_effect,
    create_radial_gradient_background,
    create_linear_gradient_background,
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
        bg_complexity: BackgroundComplexity = BackgroundComplexity.MEDIUM,
        brightness_variation_range: float = 0.3,
        min_rotation: float = 0.0,  # Minimum rotation angle in degrees
        max_rotation: float = 360.0,  # Maximum rotation angle in degrees
        skew_range: float = 0.1,  # Range of skew transformation (0-1)
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
        self.bg_complexity = bg_complexity
        self.brightness_variation_range = max(
            0.0, min(1.0, brightness_variation_range)
        )  # Clamp to [0, 1]
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.skew_range = max(0.0, min(1.0, skew_range))  # Clamp to [0, 1]

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
        scaled_template = cv2.resize(
            template, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return scaled_template, scaled_mask

    @staticmethod
    def _transform_template(
        template: np.ndarray,
        mask: np.ndarray,
        keypoints: List[KeyPoint],
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

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(
            center, -angle, 1.0
        )  # Negative angle for clockwise rotation

        # Calculate new dimensions after rotation to ensure the entire template is visible
        # Convert rotation matrix to radians
        angle_rad = math.radians(angle)
        cos_angle = abs(math.cos(angle_rad))
        sin_angle = abs(math.sin(angle_rad))

        # Calculate new dimensions for rotation
        new_w = int(w * cos_angle + h * sin_angle)
        new_h = int(w * sin_angle + h * cos_angle)

        # Adjust rotation matrix to account for new dimensions
        rotation_matrix[0, 2] += (new_w - w) // 2
        rotation_matrix[1, 2] += (new_h - h) // 2

        # Apply rotation to template and mask
        rotated_template = cv2.warpAffine(
            template,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        rotated_mask = cv2.warpAffine(
            mask,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # Rotate keypoints
        rotated_keypoints = []
        for kp in keypoints:
            # Translate keypoint to center, then apply rotation matrix
            kp_matrix = np.array([kp.x, kp.y, 1.0])
            rotated_x, rotated_y = rotation_matrix @ kp_matrix
            rotated_keypoints.append(KeyPoint(x=float(rotated_x), y=float(rotated_y)))

        # Now apply skew transformation
        # Create random skew factors
        skew_x = random.uniform(-skew_range, skew_range)
        skew_y = random.uniform(-skew_range, skew_range)

        # Create skew transformation matrix (3x3 for homography)
        # [1, skew_x, 0]   [x]   [x + skew_x*y]
        # [skew_y, 1, 0] * [y] = [skew_y*x + y]
        # [0, 0, 1]        [1]   [1]
        skew_matrix = np.array(
            [[1.0, skew_x, 0], [skew_y, 1.0, 0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        # Calculate the bounds after skew
        corners = np.array(
            [
                [0, 0, 1],  # Top-left
                [new_w - 1, 0, 1],  # Top-right
                [new_w - 1, new_h - 1, 1],  # Bottom-right
                [0, new_h - 1, 1],  # Bottom-left
            ]
        )

        # Apply skew to corners
        skewed_corners = []
        for corner in corners:
            skewed_corner = skew_matrix @ corner
            skewed_corners.append([skewed_corner[0], skewed_corner[1]])

        skewed_corners = np.array(skewed_corners)

        # Calculate new bounds
        min_x, min_y = np.min(skewed_corners, axis=0).astype(int)
        max_x, max_y = np.ceil(np.max(skewed_corners, axis=0)).astype(int)

        # Ensure positive coordinates by adding translation if needed
        tx, ty = 0, 0
        if min_x < 0:
            tx = -min_x
        if min_y < 0:
            ty = -min_y

        # Add translation to skew_matrix
        translation_matrix = np.array(
            [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        # Combine translation and skew
        final_skew_matrix = translation_matrix @ skew_matrix

        # Calculate new dimensions
        final_w = max_x - min_x + 1 + int(tx)
        final_h = max_y - min_y + 1 + int(ty)

        # Apply skew transformation to the rotated template and mask
        # We need the 2x3 portion of the 3x3 matrix for warpAffine
        skew_matrix_2x3 = final_skew_matrix[:2, :]

        final_template = cv2.warpAffine(
            rotated_template,
            skew_matrix_2x3,
            (final_w, final_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        final_mask = cv2.warpAffine(
            rotated_mask,
            skew_matrix_2x3,
            (final_w, final_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # Apply skew to rotated keypoints
        final_keypoints = []
        for kp in rotated_keypoints:
            # Convert to homogeneous coordinates and apply skew
            kp_matrix = np.array([kp.x, kp.y, 1.0])
            skewed_point = final_skew_matrix @ kp_matrix
            final_keypoints.append(
                KeyPoint(x=float(skewed_point[0]), y=float(skewed_point[1]))
            )

        return final_template, final_mask, final_keypoints

    def generate_sample(self) -> SyntheticImage:
        """Generate a synthetic image with AprilTag templates."""
        # Choose random image size divisible by 64
        min_img_size = max(1, self.min_img_size)
        max_img_size = max(1, self.max_img_size)
        img_width = 64 * math.ceil(random.randint(min_img_size, max_img_size) / 64)
        img_height = 64 * math.ceil(random.randint(min_img_size, max_img_size) / 64)

        # Create interesting background image (16-bit)
        image = self._generate_background(img_height, img_width)

        # Decide how many tags to place
        num_tags = random.randint(self.tags_per_image[0], self.tags_per_image[1])

        annotations = []

        # Generate and place tags
        for _ in range(num_tags):
            # Choose random template
            idx = random.randint(0, len(self.template_paths) - 1)
            template_path = self.template_paths[idx]
            class_num = self.class_nums[idx]

            # Load template and convert to 16-bit
            template = cv2.imread(
                str(template_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
            )
            if template is None:
                print(f"Failed to load template: {template_path}")
                continue

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

            # Choose random scale factor
            scale_factor = random.uniform(self.min_scale_factor, self.max_scale_factor)

            # Scale template and mask
            scaled_template, scaled_mask = self._scale_template(
                template, mask, scale_factor
            )
            scaled_keypoints = [
                KeyPoint(x=kp.x * scale_factor, y=kp.y * scale_factor)
                for kp in keypoints
            ]

            # Choose a random rotation angle
            rotation_angle = random.uniform(self.min_rotation, self.max_rotation)

            # Apply transforms (rotation and skew) to the scaled template, mask, and keypoints
            transformed_template, transformed_mask, transformed_keypoints = (
                self._transform_template(
                    scaled_template,
                    scaled_mask,
                    scaled_keypoints,
                    rotation_angle,
                    self.skew_range,
                )
            )

            # Use the transformed template, mask, and keypoints
            warped_template = transformed_template
            warped_mask = transformed_mask
            warped_keypoints = transformed_keypoints

            # Get dimensions of the warped template
            warped_h, warped_w = warped_template.shape[:2]

            # Check if warped template is too large for the image
            if warped_w >= img_width or warped_h >= img_height:
                continue

            # Choose random position for the tag in the final image
            # Allow it to be placed anywhere in the image
            x_pos = random.randint(0, img_width - warped_w)
            y_pos = random.randint(0, img_height - warped_h)

            # Use the full template without cropping
            cropped_template = warped_template
            cropped_mask = warped_mask

            # Since we're not cropping, there's no min_x/min_y offset
            min_x, min_y = 0, 0
            max_x, max_y = warped_w - 1, warped_h - 1

            # Check if mask has non-zero values
            if not np.any(cropped_mask):
                print("Warning: Mask is all zeros, skipping this template")
                continue

            # Adjust keypoints for placement only (no cropping)
            placed_keypoints = [
                KeyPoint(x=kp.x + x_pos, y=kp.y + y_pos) for kp in warped_keypoints
            ]

            # Calculate bounding box for the placed tag based on the rotated keypoints
            # Since we've rotated the template, we need to use the keypoints to determine the bounding box
            kp_coords = np.array(
                [[kp.x + x_pos, kp.y + y_pos] for kp in warped_keypoints]
            )
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

            # Use the rotated keypoints for placement in the final image
            placed_keypoints = [
                KeyPoint(x=kp.x + x_pos, y=kp.y + y_pos) for kp in warped_keypoints
            ]

            # No need to convert to PIL, work directly with numpy arrays
            # Keep template as uint16 for maximum precision
            template_array = cropped_template

            # Choose between linear or radial gradient for brightness
            gradient_type = random.choice(list(GradientType))

            # Create a brightness gradient map as a grayscale image
            gradient_map = Image.new("I", (warped_w, warped_h))
            gradient_draw = ImageDraw.Draw(gradient_map)

            # Random brightness gradient range from -variation to +variation
            min_brightness = 1.0 - self.brightness_variation_range
            max_brightness = 1.0 + self.brightness_variation_range

            if gradient_type == GradientType.LINEAR:
                # Choose gradient direction
                direction = random.choice(list(Direction))

                if direction == Direction.HORIZONTAL:
                    for i in range(warped_w):
                        ratio = i / max(1, warped_w - 1)
                        brightness = int(
                            (min_brightness + (max_brightness - min_brightness) * ratio)
                            * 65535
                        )
                        gradient_draw.line([(i, 0), (i, warped_h - 1)], fill=brightness)
                elif direction == Direction.VERTICAL:
                    for i in range(warped_h):
                        ratio = i / max(1, warped_h - 1)
                        brightness = int(
                            (min_brightness + (max_brightness - min_brightness) * ratio)
                            * 65535
                        )
                        gradient_draw.line([(0, i), (warped_w - 1, i)], fill=brightness)
                elif direction == Direction.DIAGONAL:
                    # Create custom diagonal gradient
                    for i in range(warped_h):
                        for j in range(warped_w):
                            # Calculate diagonal ratio
                            ratio = (
                                i / max(1, warped_h - 1) + j / max(1, warped_w - 1)
                            ) / 2
                            brightness = int(
                                (
                                    min_brightness
                                    + (max_brightness - min_brightness) * ratio
                                )
                                * 65535
                            )
                            gradient_draw.point((j, i), fill=brightness)
            elif gradient_type == GradientType.RADIAL:
                # Choose center point for radial gradient
                center_x = random.randint(0, warped_w - 1)
                center_y = random.randint(0, warped_h - 1)

                # Randomly choose between bright center or dark center
                bright_center = random.choice([True, False])
                inner_brightness = max_brightness if bright_center else min_brightness
                outer_brightness = min_brightness if bright_center else max_brightness

                # Create custom radial gradient
                max_dist = math.sqrt(warped_w**2 + warped_h**2) / 2
                for i in range(warped_h):
                    for j in range(warped_w):
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

            # Resize gradient to match template dimensions
            template_shape = template_array.shape
            gradient_resized = cv2.resize(
                gradient_array,
                (template_shape[1], template_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Normalize gradient to 0-1 range
            gradient_normalized = gradient_resized / 65535.0

            # Ensure gradient doesn't have all near-zero values
            if np.mean(gradient_normalized) < 0.01:
                # Fallback to a reasonable brightness value
                gradient_normalized = np.ones_like(gradient_normalized) * 0.5
                print("Warning: Gradient values too low, using fallback value")

            # Apply brightness adjustment using vectorized numpy operations
            # This applies the brightness gradient to the template additively
            # Map gradient from 0-1 range to -brightness_variation to +brightness_variation
            brightness_variation = self.brightness_variation_range * UINT16_MAX / 2
            brightness_adjustment = (
                gradient_normalized * 2.0 - 1.0
            ) * brightness_variation

            # Add the brightness adjustment (clamped to uint16 range)
            adjusted_array = np.clip(
                template_array.astype(np.float32) + brightness_adjustment,
                0.0,
                UINT16_MAX,
            )

            # Convert back to uint16 for the final template
            brightness_adjusted_template = adjusted_array.astype(np.uint16)

            # Place the tag on the image using the mask
            roi = image[y_pos : y_pos + warped_h, x_pos : x_pos + warped_w]
            # Convert mask to binary (0 or 1)
            binary_mask = cropped_mask > 0
            # Apply mask to place the tag
            roi[binary_mask] = brightness_adjusted_template[binary_mask]

            # Create annotation
            annotation = AprilTagAnnotation(
                class_name=self.tag_type,
                class_num=class_num,
                bbox=bbox,
                keypoints=placed_keypoints,
            )
            annotations.append(annotation)

        return SyntheticImage(
            image=image, annotations=annotations, height=img_height, width=img_width
        )

    def __iter__(self):
        """Iterator for generating an infinite stream of samples."""
        while True:
            yield self.generate_sample()

    def _apply_pil_effects(
        self, image: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """
        Apply PIL-based effects to enhance background complexity.

        Args:
            image: Input image to apply effects to
            height: Image height
            width: Image width

        Returns:
            Modified image with PIL effects applied
        """
        # Choose a random effect type
        effect_type = random.choice(list(EffectType))

        if effect_type == EffectType.MANDELBROT:
            # Apply mandelbrot effect using the dedicated function
            image = apply_mandelbrot_effect(image)

        elif effect_type == EffectType.NOISE:
            # Apply noise effect using the dedicated function
            image = apply_noise_effect(image)

        elif effect_type == EffectType.LINEAR_GRADIENT:
            # Create a linear gradient using the dedicated function
            direction = random.choice(list(Direction))
            gradient_bg = create_linear_gradient_background(
                height, width, direction, (0, UINT16_MAX)
            )

            # Blend with the original image
            blend_factor = random.uniform(0.3, 0.8)
            image = (
                image.astype(np.float32) * (1 - blend_factor)
                + gradient_bg.astype(np.float32) * blend_factor
            ).astype(np.uint16)

        elif effect_type == EffectType.RADIAL_GRADIENT:
            # Create a radial gradient using the dedicated function
            center = (random.randint(0, width - 1), random.randint(0, height - 1))
            bright_center = random.choice([True, False])
            gradient_bg = create_radial_gradient_background(
                height, width, center, bright_center, (0, UINT16_MAX)
            )

            # Blend with the original image
            blend_factor = random.uniform(0.3, 0.8)
            image = (
                image.astype(np.float32) * (1 - blend_factor)
                + gradient_bg.astype(np.float32) * blend_factor
            ).astype(np.uint16)

        elif effect_type == EffectType.COMBINED:
            # Apply multiple effects
            # First a gradient (either linear or radial)
            if random.random() < 0.5:
                gradient_bg = create_linear_gradient_background(
                    height, width, Direction.DIAGONAL, (0, UINT16_MAX)
                )
                blend_factor = random.uniform(0.3, 0.6)
                image = (
                    image.astype(np.float32) * (1 - blend_factor)
                    + gradient_bg.astype(np.float32) * blend_factor
                ).astype(np.uint16)
            else:
                center = (random.randint(0, width - 1), random.randint(0, height - 1))
                gradient_bg = create_radial_gradient_background(
                    height, width, center, True, (0, UINT16_MAX)
                )
                blend_factor = random.uniform(0.3, 0.6)
                image = (
                    image.astype(np.float32) * (1 - blend_factor)
                    + gradient_bg.astype(np.float32) * blend_factor
                ).astype(np.uint16)

            # Then add noise effect
            image = apply_noise_effect(image)

        # Apply additional filters
        filter_type = random.choice(list(FilterType))
        if filter_type != FilterType.NONE:
            image = apply_filter_effect(image, filter_type)

        return image

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
        background = generate_random_background(
            height, width, None, self.bg_color_range
        )

        return background

    def __call__(self, batch_size: int = 1) -> List[SyntheticImage]:
        """Generate a batch of synthetic images."""
        return [self.generate_sample() for _ in range(batch_size)]
