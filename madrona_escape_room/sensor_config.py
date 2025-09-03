"""
Sensor configuration presets for Madrona Escape Room.
Provides standard configurations for RGB, depth, and lidar sensors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .generated_constants import RenderMode


class SensorType(Enum):
    """Type of sensor configuration."""

    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"
    LIDAR_HORIZONTAL = "lidar_horizontal"
    LIDAR_360 = "lidar_360"
    CUSTOM = "custom"


@dataclass
class SensorConfig:
    """
    Sensor configuration parameters.

    Attributes:
        width: Horizontal resolution in pixels
        height: Vertical resolution in pixels
        vertical_fov: Vertical field of view in degrees
        horizontal_fov: Horizontal FOV in degrees (computed from vertical_fov and aspect)
        render_mode: RenderMode.RGBD or RenderMode.Depth
        sensor_type: Type of sensor for documentation/clarity
        name: Optional descriptive name
    """

    width: int
    height: int
    vertical_fov: float
    render_mode: RenderMode
    sensor_type: SensorType = SensorType.CUSTOM
    name: Optional[str] = None

    @property
    def horizontal_fov(self) -> float:
        """Calculate horizontal FOV from vertical FOV and aspect ratio."""
        import math

        aspect_ratio = self.width / self.height
        # Using standard perspective projection formula
        # tan(h_fov/2) = aspect_ratio * tan(v_fov/2)
        v_fov_rad = math.radians(self.vertical_fov)
        tan_half_v = math.tan(v_fov_rad / 2)
        tan_half_h = aspect_ratio * tan_half_v
        h_fov_rad = 2 * math.atan(tan_half_h)
        return math.degrees(h_fov_rad)

    @property
    def aspect_ratio(self) -> float:
        """Get the aspect ratio (width/height)."""
        return self.width / self.height

    @property
    def total_pixels(self) -> int:
        """Total number of pixels in the sensor."""
        return self.width * self.height

    def __str__(self) -> str:
        """String representation of the sensor config."""
        name = self.name or f"{self.sensor_type.value}"

        # Handle both enum objects and integer constants from generated_constants
        if hasattr(self.render_mode, "name"):
            mode_str = self.render_mode.name
        else:
            # Map integer values to names for generated constants
            mode_map = {0: "RGBD", 1: "Depth"}
            mode_str = mode_map.get(self.render_mode, f"Unknown({self.render_mode})")

        return (
            f"{name}: {self.width}x{self.height}, "
            f"V-FOV: {self.vertical_fov:.1f}°, H-FOV: {self.horizontal_fov:.1f}°, "
            f"Mode: {mode_str}"
        )

    # ===== Standard Presets =====

    @classmethod
    def rgb_default(cls) -> "SensorConfig":
        """Standard 64x64 RGB camera with 100° FOV."""
        return cls(
            width=64,
            height=64,
            vertical_fov=100.0,
            render_mode=RenderMode.RGBD,
            sensor_type=SensorType.RGB,
            name="RGB Camera (Default)",
        )

    @classmethod
    def rgb_high_res(cls) -> "SensorConfig":
        """High resolution 128x128 RGB camera."""
        return cls(
            width=128,
            height=128,
            vertical_fov=100.0,
            render_mode=RenderMode.RGBD,
            sensor_type=SensorType.RGB,
            name="RGB Camera (High-Res)",
        )

    @classmethod
    def rgb_narrow_fov(cls) -> "SensorConfig":
        """Narrow field of view RGB camera for focused perception."""
        return cls(
            width=64,
            height=64,
            vertical_fov=45.0,
            render_mode=RenderMode.RGBD,
            sensor_type=SensorType.RGB,
            name="RGB Camera (Narrow FOV)",
        )

    @classmethod
    def depth_default(cls) -> "SensorConfig":
        """Standard 64x64 depth sensor."""
        return cls(
            width=64,
            height=64,
            vertical_fov=100.0,
            render_mode=RenderMode.Depth,
            sensor_type=SensorType.DEPTH,
            name="Depth Sensor (Default)",
        )

    @classmethod
    def depth_high_res(cls) -> "SensorConfig":
        """High resolution 128x128 depth sensor."""
        return cls(
            width=128,
            height=128,
            vertical_fov=100.0,
            render_mode=RenderMode.Depth,
            sensor_type=SensorType.DEPTH,
            name="Depth Sensor (High-Res)",
        )

    @classmethod
    def rgbd_default(cls) -> "SensorConfig":
        """Combined RGB+Depth sensor at standard resolution."""
        return cls(
            width=64,
            height=64,
            vertical_fov=100.0,
            render_mode=RenderMode.RGBD,
            sensor_type=SensorType.RGBD,
            name="RGBD Sensor (Default)",
        )

    @classmethod
    def lidar_horizontal_128(cls) -> "SensorConfig":
        """
        128-beam horizontal lidar with 120° FOV.
        Uses 1.55° vertical FOV to achieve 120° horizontal with 128:1 aspect ratio.
        """
        return cls(
            width=128,
            height=1,
            vertical_fov=1.55,
            render_mode=RenderMode.Depth,
            sensor_type=SensorType.LIDAR_HORIZONTAL,
            name="Horizontal Lidar (128-beam, 120° FOV)",
        )

    @classmethod
    def lidar_horizontal_64(cls) -> "SensorConfig":
        """
        64-beam horizontal lidar with 120° FOV.
        Lower resolution version for faster processing.
        """
        return cls(
            width=64,
            height=1,
            vertical_fov=3.1,  # Double the vertical FOV for half the horizontal resolution
            render_mode=RenderMode.Depth,
            sensor_type=SensorType.LIDAR_HORIZONTAL,
            name="Horizontal Lidar (64-beam, 120° FOV)",
        )

    @classmethod
    def lidar_horizontal_256(cls) -> "SensorConfig":
        """
        256-beam high-resolution horizontal lidar with 120° FOV.
        """
        return cls(
            width=256,
            height=1,
            vertical_fov=0.775,  # Half the vertical FOV for double the horizontal resolution
            render_mode=RenderMode.Depth,
            sensor_type=SensorType.LIDAR_HORIZONTAL,
            name="Horizontal Lidar (256-beam, 120° FOV)",
        )

    @classmethod
    def lidar_multi_layer(cls, layers: int = 16, beams_per_layer: int = 128) -> "SensorConfig":
        """
        Multi-layer lidar (like Velodyne) with configurable layers and beams.

        Args:
            layers: Number of vertical layers (default: 16)
            beams_per_layer: Horizontal beams per layer (default: 128)
        """
        # Calculate vertical FOV to maintain reasonable horizontal FOV
        # For 16 layers, we want about 30° vertical coverage
        vertical_fov = 30.0 if layers == 16 else (30.0 * layers / 16)

        return cls(
            width=beams_per_layer,
            height=layers,
            vertical_fov=vertical_fov,
            render_mode=RenderMode.Depth,
            sensor_type=SensorType.LIDAR_360,
            name=f"Multi-Layer Lidar ({layers}x{beams_per_layer})",
        )

    @classmethod
    def custom(
        cls,
        width: int,
        height: int,
        vertical_fov: float,
        render_mode: RenderMode = RenderMode.RGBD,
        name: Optional[str] = None,
    ) -> "SensorConfig":
        """
        Create a custom sensor configuration.

        Args:
            width: Horizontal resolution
            height: Vertical resolution
            vertical_fov: Vertical field of view in degrees
            render_mode: RenderMode.RGBD or RenderMode.Depth
            name: Optional descriptive name
        """
        return cls(
            width=width,
            height=height,
            vertical_fov=vertical_fov,
            render_mode=render_mode,
            sensor_type=SensorType.CUSTOM,
            name=name or f"Custom ({width}x{height})",
        )

    def to_manager_kwargs(self) -> dict:
        """
        Convert to kwargs for SimManager initialization.

        Returns:
            Dictionary with manager configuration parameters
        """
        return {
            "enable_batch_renderer": True,
            "batch_render_view_width": self.width,
            "batch_render_view_height": self.height,
            "custom_vertical_fov": self.vertical_fov,
            "render_mode": self.render_mode,
        }

    def validate(self) -> None:
        """
        Validate the sensor configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid resolution: {self.width}x{self.height}")

        if self.vertical_fov <= 0 or self.vertical_fov > 180:
            raise ValueError(f"Invalid vertical FOV: {self.vertical_fov}°")

        if self.horizontal_fov > 180:
            raise ValueError(
                f"Computed horizontal FOV ({self.horizontal_fov:.1f}°) exceeds 180°. "
                f"Adjust vertical FOV or aspect ratio."
            )

        # Warn about extreme aspect ratios
        if self.aspect_ratio > 100 or self.aspect_ratio < 0.01:
            print(
                f"Warning: Extreme aspect ratio {self.aspect_ratio:.2f} may cause rendering issues"
            )

        # Warn about very high resolutions
        if self.total_pixels > 256 * 256:
            print(f"Warning: High resolution ({self.width}x{self.height}) may impact performance")


# Convenience exports
RGB_DEFAULT = SensorConfig.rgb_default()
DEPTH_DEFAULT = SensorConfig.depth_default()
LIDAR_128 = SensorConfig.lidar_horizontal_128()
LIDAR_64 = SensorConfig.lidar_horizontal_64()
