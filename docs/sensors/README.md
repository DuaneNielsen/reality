# Sensor Systems Documentation

This directory contains documentation for the various sensor systems available in Madrona Escape Room, including RGB cameras, depth sensors, and lidar configurations.

## Quick Start

```python
from madrona_escape_room import SimManager, sensor_config

# Use a predefined sensor configuration
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=4,
    **sensor_config.LIDAR_128.to_manager_kwargs()
)

# Access sensor data
mgr.step()
depth = mgr.depth_tensor()  # Shape: (worlds, agents, height, width, 1)
```

## Available Sensors

### Standard Configurations

The `sensor_config` module provides ready-to-use sensor configurations:

| Configuration | Resolution | FOV | Mode | Description |
|--------------|------------|-----|------|-------------|
| `sensor_config.RGB_DEFAULT` | 64×64 | 100° | RGBD | Standard RGB camera |
| `sensor_config.RGB_HIGH_RES` | 128×128 | 100° | RGBD | High-resolution RGB |
| `sensor_config.DEPTH_DEFAULT` | 64×64 | 100° | Depth | Standard depth sensor |
| `sensor_config.DEPTH_HIGH_RES` | 128×128 | 100° | Depth | High-resolution depth |
| `sensor_config.RGBD_DEFAULT` | 64×64 | 100° | RGBD | Combined RGB+Depth |
| `sensor_config.LIDAR_128` | 128×1 | 120° H | Depth | 128-beam horizontal lidar |
| `sensor_config.LIDAR_64` | 64×1 | 120° H | Depth | 64-beam horizontal lidar |
| `sensor_config.LIDAR_256` | 256×1 | 120° H | Depth | High-res horizontal lidar |

### Custom Configurations

Create custom sensor configurations for specific needs:

```python
from madrona_escape_room import SensorConfig, RenderMode

# Custom narrow FOV camera
narrow_camera = SensorConfig.custom(
    width=96,
    height=96,
    vertical_fov=45.0,
    render_mode=RenderMode.RGBD,
    name="Narrow FOV Camera"
)

# Multi-layer lidar (like Velodyne)
multi_lidar = SensorConfig.lidar_multi_layer(
    layers=16,
    beams_per_layer=128
)

# Use in SimManager
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=4,
    **narrow_camera.to_manager_kwargs()
)
```

## Sensor Data Format

### Tensor Dimensions

All sensor data follows the same tensor layout:

```python
# Tensor shape: (num_worlds, num_agents, height, width, channels)
# 
# Where:
#   - num_worlds: Number of parallel simulation worlds
#   - num_agents: Agents per world (currently 1)
#   - height: Vertical resolution in pixels
#   - width: Horizontal resolution in pixels  
#   - channels: 1 for depth, 4 for RGBA

depth_tensor = mgr.depth_tensor()  # (..., 1) channel
rgb_tensor = mgr.rgb_tensor()      # (..., 4) channels RGBA
```

### Coordinate System

```python
# Spatial mapping in sensor images:
value = sensor_data[world, agent, y, x, channel]

# Where:
# y=0: Top of view (looking up)
# y=height//2: Center vertically (straight ahead)
# y=height-1: Bottom of view (looking down)
# 
# x=0: Left side of view
# x=width//2: Center horizontally (straight ahead)
# x=width-1: Right side of view
```

### Depth Values

- **Format**: Float32 linear distances
- **Units**: World space units (not normalized)
- **Range**: 0.001 to 20,000 units
- **Calculation**: Euclidean distance from camera to surface

## Camera Positioning

Cameras are positioned relative to the agent:

```
Agent position: (x, y, z=0)
Camera position: (x, y, z=1.5)  # 1.5 units above agent
Camera rotation: Follows agent's facing direction
```

## Testing with Pytest

The test framework provides convenient markers for sensor configurations:

```python
import pytest

@pytest.mark.depth_default
def test_with_default_depth(cpu_manager):
    """Test using default 64×64 depth sensor."""
    mgr = cpu_manager
    # Manager automatically configured with depth sensor
    
@pytest.mark.lidar_128
def test_with_horizontal_lidar(cpu_manager):
    """Test using 128-beam horizontal lidar."""
    mgr = cpu_manager
    # Manager configured with lidar settings
    
@pytest.mark.rgbd_default
def test_with_rgbd(cpu_manager):
    """Test with combined RGB and depth."""
    mgr = cpu_manager
    # Both RGB and depth tensors available
```

Available test markers:
- `@pytest.mark.rgb_default` - 64×64 RGB camera
- `@pytest.mark.rgb_high_res` - 128×128 RGB camera  
- `@pytest.mark.depth_default` - 64×64 depth sensor
- `@pytest.mark.depth_high_res` - 128×128 depth sensor
- `@pytest.mark.rgbd_default` - 64×64 RGBD sensor
- `@pytest.mark.lidar_128` - 128-beam horizontal lidar
- `@pytest.mark.lidar_64` - 64-beam horizontal lidar
- `@pytest.mark.lidar_256` - 256-beam horizontal lidar

For custom configurations in tests:

```python
@pytest.mark.depth_sensor(96, 48, 60.0)  # width, height, vertical_fov
def test_custom_sensor(cpu_manager):
    """Test with custom 96×48 sensor with 60° FOV."""
    mgr = cpu_manager
```

## Performance Considerations

### Memory Usage

| Resolution | Memory per Agent | Notes |
|------------|-----------------|-------|
| 64×64 | ~16 KB | Default, fast |
| 128×128 | ~65 KB | High detail |
| 256×256 | ~262 KB | Very high detail |
| 128×1 | ~512 B | Horizontal lidar |

Memory is allocated on GPU when using CUDA execution mode.

### Optimization Tips

1. **Resolution**: Lower resolution = faster rendering, less memory
2. **Render Mode**: Depth-only mode is faster than RGBD if RGB not needed
3. **Batch Size**: All worlds render simultaneously, consider total GPU memory
4. **Aspect Ratio**: Extreme ratios (e.g., 128×1) may have performance implications

## Field of View (FOV) Calculations

The relationship between vertical FOV, horizontal FOV, and aspect ratio:

```python
# Horizontal FOV is computed from vertical FOV and aspect ratio:
tan(h_fov/2) = aspect_ratio × tan(v_fov/2)

# Examples:
# 64×64 with 100° V-FOV → 100° H-FOV (square)
# 128×64 with 100° V-FOV → 144° H-FOV (wide)
# 128×1 with 1.55° V-FOV → 120° H-FOV (lidar)
```

## Common Use Cases

### Obstacle Detection

```python
def detect_obstacles(depth_tensor, max_distance=10.0):
    """Simple obstacle detection using depth sensor."""
    depth_np = depth_tensor.to_numpy()
    
    # Check center region for obstacles
    h, w = depth_np.shape[2:4]
    center_region = depth_np[0, 0, h//2-5:h//2+5, w//2-5:w//2+5, 0]
    
    # Path is clear if all distances > threshold
    path_clear = np.all(center_region > max_distance)
    nearest = np.min(center_region)
    
    return path_clear, nearest
```

### Horizontal Scanning (Lidar-like)

```python
# Use horizontal lidar configuration
config = sensor_config.LIDAR_128

mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=1,
    **config.to_manager_kwargs()
)

mgr.step()
depth = mgr.depth_tensor()
scan_line = depth.to_numpy()[0, 0, 0, :, 0]  # 128 distance measurements
```

### Combined RGB and Depth

```python
# Use RGBD mode for both sensors
config = sensor_config.RGBD_DEFAULT

mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=4,
    **config.to_manager_kwargs()
)

mgr.step()
rgb = mgr.rgb_tensor()    # Visual data
depth = mgr.depth_tensor() # Distance data

# Pixels correspond between sensors
# rgb[w,a,y,x,:] matches depth[w,a,y,x,0]
```

## Documentation Index

### Available Guides

- **[DEPTH_SENSOR_QUICKSTART.md](DEPTH_SENSOR_QUICKSTART.md)** - Detailed depth sensor usage and examples
- Additional guides coming soon:
  - LIDAR_CONFIGURATION.md - Advanced lidar setups
  - CAMERA_CONFIGURATION.md - Custom FOV and positioning
  - SENSOR_FUSION.md - Combining multiple sensors

### Related Documentation

- [../architecture/ECS_ARCHITECTURE.md](../architecture/ECS_ARCHITECTURE.md) - How sensors integrate with ECS
- [../development/CPP_CODING_STANDARDS.md](../development/CPP_CODING_STANDARDS.md) - Modifying sensor C++ code
- [../tools/VIEWER_GUIDE.md](../tools/VIEWER_GUIDE.md) - Visualizing sensor data

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Empty tensor | Ensure `enable_batch_renderer=True` or use sensor config |
| All zeros | Check scene has geometry within sensor range |
| Wrong shape | Verify resolution settings match expectations |
| GPU memory error | Reduce resolution or number of worlds |
| No RGB data in Depth mode | Use RGBD mode for both sensors |

### Debug Checklist

1. Verify batch renderer is enabled
2. Check tensor shapes match configuration
3. Ensure scene has geometry within range (0.001 - 20,000 units)
4. For custom FOV, verify calculations are correct
5. For GPU mode, check available memory

## API Reference

### SensorConfig Class

```python
class SensorConfig:
    width: int              # Horizontal resolution
    height: int             # Vertical resolution  
    vertical_fov: float     # Vertical field of view in degrees
    render_mode: RenderMode # RGBD or Depth
    
    # Computed properties
    @property
    def horizontal_fov(self) -> float
    @property
    def aspect_ratio(self) -> float
    
    # Factory methods
    @classmethod
    def rgb_default(cls) -> SensorConfig
    @classmethod
    def depth_default(cls) -> SensorConfig
    @classmethod
    def lidar_horizontal_128(cls) -> SensorConfig
    @classmethod
    def custom(cls, width, height, vertical_fov, render_mode) -> SensorConfig
    
    # Convert to manager parameters
    def to_manager_kwargs(self) -> dict
```

### SimManager Sensor Methods

```python
class SimManager:
    def depth_tensor(self) -> Tensor
        """Get depth sensor data."""
        
    def rgb_tensor(self) -> Tensor
        """Get RGB camera data."""
```

## Future Enhancements

Planned improvements for the sensor system:

- [ ] 360° panoramic sensors
- [ ] Multiple cameras per agent
- [ ] Configurable camera positions
- [ ] Infrared/thermal sensors
- [ ] Point cloud output format
- [ ] Sensor noise models
- [ ] Ray-based range sensors