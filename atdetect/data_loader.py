import os
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from enum import Enum, auto
import math

UINT16_MAX = np.iinfo(np.uint16).max


class BackgroundComplexity(Enum):
    SIMPLE = auto()
    MEDIUM = auto()
    COMPLEX = auto()


class BackgroundType(Enum):
    SOLID = auto()
    GRADIENT = auto()
    NOISE = auto()
    PATTERN = auto()
    SHAPES = auto()


class ShapeType(Enum):
    CIRCLE = auto()
    RECTANGLE = auto()
    LINE = auto()


class PatternType(Enum):
    GRID = auto()
    STRIPES = auto()
    CHECKER = auto()


class Direction(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()
    DIAGONAL = auto()


class NoiseType(Enum):
    PERLIN = auto()
    GAUSSIAN = auto()


class GradientType(Enum):
    LINEAR = auto()
    RADIAL = auto()


class FilterType(Enum):
    BLUR = auto()
    SHARPEN = auto()
    EMBOSS = auto()
    NONE = auto()


class EffectType(Enum):
    MANDELBROT = auto()
    NOISE = auto()
    LINEAR_GRADIENT = auto()
    RADIAL_GRADIENT = auto()
    COMBINED = auto()


@dataclass
class BoundingBox:
    """Bounding box in [x_min, y_min, x_max, y_max] format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class KeyPoint:
    """Keypoint with x, y coordinates."""

    x: float
    y: float


@dataclass
class AprilTagAnnotation:
    """Annotation for a single AprilTag instance."""

    class_name: str  # e.g., 'tag16h5'
    class_num: int  # Numeric ID of the tag
    bbox: BoundingBox
    keypoints: List[KeyPoint]  # 4 keypoints, one for each corner


@dataclass
class SyntheticImage:
    """Synthetic image with AprilTag annotations."""

    image: np.ndarray  # H x W x C image (16-bit)
    annotations: List[AprilTagAnnotation]
    height: int
    width: int


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
        M = AprilTagDataLoader._build_perspective_matrix(h, w, perspective_range)

        # Apply transform to image and mask
        transformed_img = cv2.warpPerspective(image, M, (w, h))
        transformed_mask = cv2.warpPerspective(mask, M, (w, h))

        # Transform keypoints
        transformed_keypoints = []
        for kp in keypoints:
            # Convert to homogeneous coordinates
            p = np.array([kp.x, kp.y, 1.0])
            # Apply transformation
            p_transformed = M @ p
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
        import ipdb

        ipdb.set_trace()
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

            if template.dtype != np.uint16:
                template = template.astype(np.uint16)
                # Scale from 8-bit to 16-bit if needed
                if template.max() <= 255:
                    template = template * 256

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
        # Convert from 16-bit numpy array to PIL Image
        # Scale down to 8-bit for PIL compatibility
        scaled_image = (image / 256).astype(np.uint8)
        pil_image = Image.fromarray(scaled_image)

        # Choose a random effect
        effect_type = random.choice(list(EffectType))

        if effect_type == EffectType.MANDELBROT:
            # Create a mandelbrot effect
            import ipdb

            ipdb.set_trace()
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
            pil_image = Image.blend(pil_image, effect_img, random.uniform(0.3, 0.7))

        elif effect_type == EffectType.NOISE:
            import ipdb

            # Apply noise effect
            noise_img = pil_image.effect_noise((width, height), random.randint(5, 30))
            pil_image = Image.blend(pil_image, noise_img, random.uniform(0.2, 0.6))

        elif effect_type == EffectType.LINEAR_GRADIENT:
            import ipdb

            # Create a linear gradient
            direction_map = {
                Direction.HORIZONTAL: "left-to-right",
                Direction.VERTICAL: "top-to-bottom",
                Direction.DIAGONAL: "diagonal",
            }
            direction = random.choice(list(Direction))
            gradient_direction = direction_map[direction]
            color1 = (random.randint(0, UINT16_MAX),)
            color2 = (random.randint(0, UINT16_MAX),)

            gradient_img = pil_image.linear_gradient(gradient_direction, color1, color2)
            pil_image = Image.blend(pil_image, gradient_img, random.uniform(0.3, 0.8))

        elif effect_type == EffectType.RADIAL_GRADIENT:
            import ipdb

            # Create a radial gradient
            color1 = random.randint(0, UINT16_MAX)
            color2 = random.randint(0, UINT16_MAX)

            center = (random.randint(0, width), random.randint(0, height))
            radius = random.uniform(0.3, 1.0) * max(width, height) / 2

            # PIL only uses "radial" for the type parameter
            gradient_img = pil_image.radial_gradient(
                "radial", color1, color2, center, radius
            )
            pil_image = Image.blend(pil_image, gradient_img, random.uniform(0.3, 0.8))

        elif effect_type == EffectType.COMBINED:
            import ipdb

            # Apply multiple effects
            # First a gradient (either linear or radial)
            if random.random() < 0.5:
                color1 = random.randint(0, UINT16_MAX)
                color2 = random.randint(0, UINT16_MAX)
                # Map the Direction enum to PIL's gradient direction string
                gradient_img = pil_image.linear_gradient(
                    direction_map[Direction.DIAGONAL], color1, color2
                )
                pil_image = Image.blend(
                    pil_image, gradient_img, random.uniform(0.3, 0.6)
                )
            else:
                color1 = random.randint(0, UINT16_MAX)
                color2 = random.randint(0, UINT16_MAX)

                center = (random.randint(0, width), random.randint(0, height))
                radius = random.uniform(0.3, 1.0) * max(width, height) / 2

                gradient_img = pil_image.radial_gradient(
                    "radial", color1, color2, center, radius
                )
                pil_image = Image.blend(
                    pil_image, gradient_img, random.uniform(0.3, 0.6)
                )

            # Then add noise
            noise_img = pil_image.effect_noise((width, height), random.randint(5, 30))
            pil_image = Image.blend(pil_image, noise_img, random.uniform(0.2, 0.5))

        # Apply additional filters for more complexity
        filter_type = random.choice(list(FilterType))
        if filter_type == FilterType.BLUR:
            pil_image = pil_image.filter(
                ImageFilter.GaussianBlur(random.uniform(0.5, 2.0))
            )
        elif filter_type == FilterType.SHARPEN:
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        elif filter_type == FilterType.EMBOSS:
            pil_image = pil_image.filter(ImageFilter.EMBOSS)

        # Convert back to numpy array and scale back to 16-bit
        result = np.array(pil_image).astype(np.uint16) * 256

        # Ensure the result has the right shape
        if result.shape[0] != height or result.shape[1] != width:
            print(
                f"Warning: PIL effect resulted in wrong shape: {result.shape}, expected {(height, width, 3)}"
            )
            return image  # Return the original image if there's a shape mismatch

        return result

    def _generate_background(self, height: int, width: int) -> np.ndarray:
        """
        Generate a complex background image with textures, gradients, or patterns.

        Args:
            height: Height of the image
            width: Width of the image

        Returns:
            16-bit background image with interesting patterns
        """
        import ipdb

        ipdb.set_trace()
        # Create a base background
        background = np.ones((height, width), dtype=np.uint16) * random.randint(
            0, UINT16_MAX
        )

        # Choose a random background type based on complexity
        # Choose randomly between gradient, noise, or simple patterns
        bg_type = random.choice(list(BackgroundType))

        if bg_type == BackgroundType.SOLID:
            # Simple solid color background
            bg_color = random.randint(self.bg_color_range[0], self.bg_color_range[1])
            background = background * bg_color
        if bg_type == BackgroundType.GRADIENT:
            # Create a random gradient background
            start_color = np.array(
                [random.randint(self.bg_color_range[0], self.bg_color_range[1])]
            )
            end_color = np.array(
                [random.randint(self.bg_color_range[0], self.bg_color_range[1])]
            )

            # Choose gradient direction (horizontal, vertical, diagonal)
            direction = random.choice(list(Direction))

            if direction == Direction.HORIZONTAL:
                # Horizontal gradient
                for i in range(width):
                    ratio = i / width
                    color = start_color * (1 - ratio) + end_color * ratio
                    background[:, i, :] = color

            elif direction == Direction.VERTICAL:
                # Vertical gradient
                for i in range(height):
                    ratio = i / height
                    color = start_color * (1 - ratio) + end_color * ratio
                    background[i, :, :] = color

            elif direction == Direction.DIAGONAL:
                # Diagonal gradient
                for i in range(height):
                    for j in range(width):
                        ratio = (i / height + j / width) / 2
                        color = start_color * (1 - ratio) + end_color * ratio
                        background[i, j, :] = color

        elif bg_type == BackgroundType.NOISE:
            # Create noise background
            noise_type = random.choice(list(NoiseType))

            if noise_type == NoiseType.GAUSSIAN:
                # Gaussian noise
                mean = random.randint(self.bg_color_range[0], self.bg_color_range[1])
                std = random.randint(500, 3000)  # Standard deviation for noise

                noise = np.random.normal(mean, std, (height, width))
                noise = np.clip(noise, self.bg_color_range[0], self.bg_color_range[1])
                background = noise.astype(np.uint16)

            elif noise_type == NoiseType.PERLIN:
                # Generate base noise
                scale = random.randint(5, 30)  # Controls the scale of the noise
                octaves = random.randint(1, 3)  # Number of layers of noise

                base = np.random.randint(
                    self.bg_color_range[0],
                    self.bg_color_range[1],
                    (height // scale + 1, width // scale + 1),
                ).astype(np.uint16)
                scaled = cv2.resize(
                    base, (width, height), interpolation=cv2.INTER_LINEAR
                )

                # Add octaves for more detail
                for i in range(1, octaves):
                    small_scale = scale // (2**i)
                    if small_scale < 1:
                        break

                    small = np.random.randint(
                        0,
                        2000,
                        (height // small_scale + 1, width // small_scale + 1),
                    ).astype(np.uint16)
                    small_scaled = cv2.resize(
                        small, (width, height), interpolation=cv2.INTER_LINEAR
                    )
                    scaled = scaled + small_scaled / (2**i)

                background = np.clip(
                    scaled, self.bg_color_range[0], self.bg_color_range[1]
                ).astype(np.uint16)

        elif bg_type == BackgroundType.PATTERN:
            # Create simple geometric patterns
            pattern_type = random.choice(list(PatternType))
            base_color = random.randint(self.bg_color_range[0], self.bg_color_range[1])
            alt_color = random.randint(self.bg_color_range[0], self.bg_color_range[1])

            if pattern_type == PatternType.GRID:
                # Grid pattern
                grid_size = random.randint(20, 100)
                background.fill(base_color)

                # Draw grid lines
                for i in range(0, height, grid_size):
                    background[i : i + 2, :] = alt_color
                for j in range(0, width, grid_size):
                    background[:, j : j + 2] = alt_color

            elif pattern_type == PatternType.STRIPES:
                # Striped pattern
                stripe_width = random.randint(10, 50)
                direction = random.choice(list(Direction))

                # Create alternating stripes using np.repeat
                if direction == Direction.HORIZONTAL:
                    # Create base stripe pattern - alternating values [base_color, alt_color]
                    base_pattern = np.array([base_color, alt_color], dtype=np.uint16)
                    # Repeat each value stripe_width times horizontally
                    stripes = np.repeat(base_pattern, stripe_width)
                    # Tile to fill the width
                    stripes = np.tile(stripes, (width // (2 * stripe_width)) + 1)
                    # Crop to exact width
                    stripes = stripes[:width]
                    # Repeat vertically for all rows
                    background = np.tile(stripes, (height, 1))
                elif direction == Direction.VERTICAL:
                    # Create base stripe pattern - alternating values [base_color, alt_color]
                    base_pattern = np.array([base_color, alt_color], dtype=np.uint16)
                    # Repeat each value stripe_width times
                    stripes = np.repeat(base_pattern, stripe_width)
                    # Tile to fill the height
                    stripes = np.tile(stripes, (height // (2 * stripe_width)) + 1)
                    # Crop to exact height
                    stripes = stripes[:height]
                    # Reshape for vertical stripes
                    background = np.tile(stripes[:, np.newaxis], (1, width))
            elif pattern_type == PatternType.CHECKER:  # checker pattern
                # Checker pattern using np.repeat and tiling
                check_size = random.randint(20, 80)

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
                background = checker[:height, :width]

        elif bg_type == BackgroundType.SHAPES:
            # Determine number of layers to add (instead of recursively calling)
            num_layers = random.randint(2, 5)

            # Add layers of random shapes or additional textures
            for _ in range(num_layers):
                # Each layer has a different number of shapes
                num_shapes = random.randint(
                    3, 10
                )  # Fewer shapes per layer, but multiple layers

                # For each shape in this layer
                for _ in range(num_shapes):
                    # Choose a random shape type
                    shape_type = random.choice(list(ShapeType))
                    shape_color = np.array(
                        [random.randint(self.bg_color_range[0], self.bg_color_range[1])]
                    )

                    if shape_type == ShapeType.CIRCLE:
                        # Random circle
                        center_x = random.randint(0, width)
                        center_y = random.randint(0, height)
                        radius = random.randint(10, max(50, min(height, width) // 4))
                        thickness = random.choice(
                            [-1, random.randint(1, 5)]
                        )  # -1 means filled

                        # Draw circle (safely handling out-of-bounds)
                        cv2.circle(
                            background,
                            (center_x, center_y),
                            radius,
                            shape_color.tolist(),
                            thickness,
                        )

                    elif shape_type == ShapeType.RECTANGLE:
                        # Random rectangle
                        x1 = random.randint(0, width)
                        y1 = random.randint(0, height)
                        x2 = random.randint(x1, min(x1 + width // 2, width))
                        y2 = random.randint(y1, min(y1 + height // 2, height))
                        thickness = random.choice(
                            [-1, random.randint(1, 5)]
                        )  # -1 means filled

                        # Draw rectangle (safely handling out-of-bounds)
                        cv2.rectangle(
                            background,
                            (x1, y1),
                            (x2, y2),
                            shape_color.tolist(),
                            thickness,
                        )

                    elif shape_type == ShapeType.LINE:
                        # Random line
                        x1 = random.randint(0, width)
                        y1 = random.randint(0, height)
                        x2 = random.randint(0, width)
                        y2 = random.randint(0, height)
                        thickness = random.randint(1, 5)

                        # Draw line (safely handling out-of-bounds)
                        cv2.line(
                            background,
                            (x1, y1),
                            (x2, y2),
                            shape_color.tolist(),
                            thickness,
                        )

            # Add a layer of subtle noise for texture
            noise_amplitude = random.randint(500, 2000)
            noise = np.random.normal(0, noise_amplitude, (height, width))
            channel = background.astype(np.int32) + noise
            background = np.clip(
                channel, self.bg_color_range[0], self.bg_color_range[1]
            ).astype(np.uint16)

            # 25% chance to apply PIL-based effects
            if random.random() < 0.25:
                background = self._apply_pil_effects(background, height, width)

        return background

    def __call__(self, batch_size: int = 1) -> List[SyntheticImage]:
        """Generate a batch of synthetic images."""
        return [self.generate_sample() for _ in range(batch_size)]
