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
        perspective_transform_range: float = 0.2,
        tags_per_image: Tuple[int, int] = (1, 3),
        bg_color_range: Tuple[int, int] = (5000, 20000),  # 16-bit range
        bg_complexity: BackgroundComplexity = BackgroundComplexity.MEDIUM,
        brightness_variation_range: float = 0.3,
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
            perspective_transform_range: Range of perspective transform.
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
        self.perspective_transform_range = perspective_transform_range
        self.tags_per_image = tags_per_image
        self.bg_color_range = bg_color_range
        self.bg_complexity = bg_complexity
        self.brightness_variation_range = max(
            0.0, min(1.0, brightness_variation_range)
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
        scaled_template = cv2.resize(
            template, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return scaled_template, scaled_mask

    @staticmethod
    def _build_perspective_matrix(
        h: int, w: int, perspective_range: float
    ) -> np.ndarray:
        """
        Build a random perspective transformation matrix.

        Args:
            h: Height of the image
            w: Width of the image
            perspective_range: Range of perspective distortion (0-1)

        Returns:
            3x3 perspective transformation matrix
        """
        # Create a random center for the transform
        center_x = w / 2
        center_y = h / 2

        # Generate random rotation angle
        angle = random.uniform(-45, 45)  # degrees

        # Generate random skew factors
        skew_x = random.uniform(-perspective_range, perspective_range)
        skew_y = random.uniform(-perspective_range, perspective_range)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])  # Convert to 3x3

        # Create skew matrix
        skew_matrix = np.array(
            [[1.0, skew_x, 0], [skew_y, 1.0, 0], [0, 0, 1.0]], dtype=np.float32
        )

        # Combine transforms
        perspective_matrix = skew_matrix @ rotation_matrix

        return perspective_matrix

    @staticmethod
    def _apply_perspective_transform(
        image: np.ndarray,
        mask: np.ndarray,
        keypoints: List[KeyPoint],
        perspective_range: float,
    ) -> Tuple[np.ndarray, np.ndarray, List[KeyPoint]]:
        """
        Apply random perspective transform to image, mask, and update keypoints.

        Returns:
            Tuple of (transformed image, transformed mask, transformed keypoints)
        """
        h, w = image.shape[:2]

        # Build perspective matrix
        perspective_mat = AprilTagDataLoader._build_perspective_matrix(
            h, w, perspective_range
        )

        # Apply transform to image and mask
        transformed_img = cv2.warpPerspective(image, perspective_mat, (w, h))
        transformed_mask = cv2.warpPerspective(mask, perspective_mat, (w, h))

        # Transform keypoints
        transformed_keypoints = []
        for kp in keypoints:
            # Convert to homogeneous coordinates
            p = np.array([kp.x, kp.y, 1.0])
            # Apply transformation
            p_transformed = perspective_mat @ p
            # Convert back from homogeneous coordinates
            if p_transformed[2] != 0:
                x_new, y_new = p_transformed[:2] / p_transformed[2]
                transformed_keypoints.append(KeyPoint(x=float(x_new), y=float(y_new)))
            else:
                transformed_keypoints.append(KeyPoint(x=kp.x, y=kp.y))

        return transformed_img, transformed_mask, transformed_keypoints

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

            # Apply perspective transform to template, mask and keypoints
            warped_template, warped_mask, warped_keypoints = (
                self._apply_perspective_transform(
                    scaled_template,
                    scaled_mask,
                    scaled_keypoints,
                    self.perspective_transform_range,
                )
            )

            # Find the bounding box of the warped keypoints
            kp_coords = np.array([[kp.x, kp.y] for kp in warped_keypoints])
            min_x, min_y = np.min(kp_coords, axis=0).astype(int)
            max_x, max_y = np.max(kp_coords, axis=0).astype(int)

            # Ensure the bounding box is within image bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(warped_template.shape[1] - 1, max_x)
            max_y = min(warped_template.shape[0] - 1, max_y)

            # Calculate dimensions of the warped tag
            warped_w = max_x - min_x + 1
            warped_h = max_y - min_y + 1

            # Check if warped template is too large for the image
            if warped_w >= img_width or warped_h >= img_height:
                continue

            # Choose random position for the tag in the final image
            # Allow it to be placed anywhere in the image
            x_pos = random.randint(0, img_width - warped_w)
            y_pos = random.randint(0, img_height - warped_h)

            # Crop the warped template and mask to the keypoint bounds
            cropped_template = warped_template[min_y : max_y + 1, min_x : max_x + 1]
            cropped_mask = warped_mask[min_y : max_y + 1, min_x : max_x + 1]

            # Adjust keypoints for cropping and placement
            placed_keypoints = [
                KeyPoint(x=kp.x - min_x + x_pos, y=kp.y - min_y + y_pos)
                for kp in warped_keypoints
            ]

            # Calculate bounding box for the placed tag
            kp_coords = np.array([[kp.x, kp.y] for kp in placed_keypoints])
            x_min, y_min = np.min(kp_coords, axis=0)
            x_max, y_max = np.max(kp_coords, axis=0)
            bbox = BoundingBox(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
            )

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
                    # Draw gradient corners with proper brightness values
                    gradient_draw.rectangle(
                        [(0, 0), (warped_w - 1, warped_h - 1)],
                        fill=int(min_brightness * 65535),
                    )
                    gradient_map = Image.linear_gradient("I")
            elif gradient_type == GradientType.RADIAL:
                # Choose center point for radial gradient
                center_x = random.randint(0, warped_w - 1)
                center_y = random.randint(0, warped_h - 1)

                # Randomly choose between bright center or dark center
                bright_center = random.choice([True, False])
                inner_brightness = max_brightness if bright_center else min_brightness
                outer_brightness = min_brightness if bright_center else max_brightness

                # Create radial gradient
                gradient_map = Image.radial_gradient("I")

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

            # Apply brightness adjustment using vectorized numpy operations
            # This applies the brightness gradient to the template in one operation
            adjusted_array = template_array.astype(np.float32) * gradient_normalized

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
