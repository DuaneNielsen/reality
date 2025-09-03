# Depth Sensor Quickstart Guide

This guide provides detailed information about using depth sensors in Madrona Escape Room for spatial perception and navigation tasks.

> **Note**: For general sensor information and configuration options, see the [Sensors README](README.md).

## Overview

The depth sensor provides agents with distance measurements from their camera viewpoint, similar to how depth cameras or time-of-flight sensors work in real robotics. Unlike traditional Z-buffer depth, Madrona provides **linear world-space distances** that are ideal for RL spatial reasoning.

## Quick Start with SensorConfig

```python
from madrona_escape_room import SimManager, sensor_config, ExecMode

# Use predefined depth sensor configuration
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=4,
    **sensor_config.DEPTH_DEFAULT.to_manager_kwargs()
)

# Or use high-resolution depth
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=4,
    **sensor_config.DEPTH_HIGH_RES.to_manager_kwargs()
)

# Access depth data
mgr.step()
depth_tensor = mgr.depth_tensor()
depth_np = depth_tensor.to_numpy()
print(f"Depth shape: {depth_np.shape}")  # (4, 1, 128, 128, 1) for high-res
```

## Depth Data Format

### Data Characteristics
- **Format**: Float32 values
- **Units**: World distance units (linear, not logarithmic)
- **Range**: 0.001 to 20,000 world units
- **Calculation**: Euclidean distance from camera to surface

### Understanding Depth Values
```python
# Extract depth for a single agent in world 0
agent_depth = depth_np[0, 0, :, :, 0]  # Shape: (64, 64)

# Center pixel (agent's forward direction)
forward_distance = agent_depth[32, 32]
print(f"Distance straight ahead: {forward_distance:.2f} world units")

# Find nearest and farthest points
nearest = np.min(agent_depth)
farthest = np.max(agent_depth)
print(f"Nearest obstacle: {nearest:.2f} units")
print(f"Farthest visible: {farthest:.2f} units")
```

## Interpreting Depth Values

### Spatial Correspondence
```python
# Extract depth for specific agent
depth_value = depth_np[world_idx, agent_idx, y, x, 0]

# Pixel positions correspond to view directions:
# Center pixel = straight ahead
# Top pixels = looking up
# Bottom pixels = looking down
# Left/right pixels = peripheral vision

# Example: Get distance straight ahead for agent 0 in world 0
h, w = depth_np.shape[2:4]
forward_distance = depth_np[0, 0, h//2, w//2, 0]
```

## Example: Obstacle Detection

```python
import numpy as np

def detect_obstacles(depth_tensor, world_idx=0, agent_idx=0, max_distance=10.0):
    """Detect obstacles within a certain range."""
    
    # Get depth data for specific agent
    if depth_tensor.isOnGPU():
        depth_np = depth_tensor.to_torch().cpu().numpy()
    else:
        depth_np = depth_tensor.to_numpy()
    
    agent_depth = depth_np[world_idx, agent_idx, :, :, 0]
    
    # Find obstacles closer than max_distance
    obstacles = agent_depth < max_distance
    
    # Get center region (agent's main forward view)
    center_h, center_w = agent_depth.shape[0] // 2, agent_depth.shape[1] // 2
    forward_region = agent_depth[center_h-5:center_h+5, center_w-5:center_w+5]
    
    # Check if path ahead is clear
    forward_clear = np.all(forward_region > max_distance)
    
    return {
        'obstacles_detected': np.any(obstacles),
        'obstacle_count': np.sum(obstacles),
        'forward_clear': forward_clear,
        'nearest_distance': np.min(agent_depth),
        'forward_distance': agent_depth[center_h, center_w]
    }

# Usage
mgr.step()
depth = mgr.depth_tensor()
obstacles = detect_obstacles(depth, world_idx=0, agent_idx=0)
print(f"Forward clear: {obstacles['forward_clear']}")
print(f"Distance ahead: {obstacles['forward_distance']:.2f}")
```

## Custom Resolution and FOV

Use `SensorConfig` for custom depth sensor configurations:

```python
from madrona_escape_room import SensorConfig, RenderMode

# Create custom depth sensor
custom_depth = SensorConfig.custom(
    width=96,
    height=72,
    vertical_fov=60.0,  # Narrower FOV for focused vision
    render_mode=RenderMode.Depth,
    name="Custom Depth Sensor"
)

# Use in simulation
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    num_worlds=4,
    **custom_depth.to_manager_kwargs()
)
```

## Depth-Specific Applications

### Navigation and Path Planning

```python
def find_clear_path(depth_tensor, turn_threshold=15.0):
    """Find clear path direction using depth sensor."""
    depth_np = depth_tensor.to_numpy()[0, 0, :, :, 0]
    h, w = depth_np.shape
    
    # Divide view into three sections
    left_region = depth_np[:, :w//3]
    center_region = depth_np[:, w//3:2*w//3]
    right_region = depth_np[:, 2*w//3:]
    
    # Calculate average distances
    left_dist = np.mean(left_region)
    center_dist = np.mean(center_region)
    right_dist = np.mean(right_region)
    
    # Recommend turn direction
    if center_dist > turn_threshold:
        return "forward"
    elif left_dist > right_dist:
        return "turn_left"
    else:
        return "turn_right"
```

### Wall Following

```python
def wall_follow_distance(depth_tensor, side="right", target_distance=3.0):
    """Calculate error for wall following behavior."""
    depth_np = depth_tensor.to_numpy()[0, 0, :, :, 0]
    h, w = depth_np.shape
    
    # Sample side region
    if side == "right":
        side_region = depth_np[h//3:2*h//3, 3*w//4:]
    else:  # left
        side_region = depth_np[h//3:2*h//3, :w//4]
    
    # Get minimum distance to wall
    wall_distance = np.min(side_region)
    error = wall_distance - target_distance
    
    return error, wall_distance
```

### Collision Prediction

```python
def time_to_collision(depth_tensor, velocity, safety_margin=0.5):
    """Estimate time until collision based on depth and velocity."""
    depth_np = depth_tensor.to_numpy()[0, 0, :, :, 0]
    h, w = depth_np.shape
    
    # Focus on forward region
    forward_region = depth_np[h//3:2*h//3, w//3:2*w//3]
    min_distance = np.min(forward_region) - safety_margin
    
    if velocity > 0:
        ttc = min_distance / velocity
    else:
        ttc = float('inf')
    
    return ttc, min_distance
```

## Advanced: Render Pipeline Architecture

### Two-Stage Rendering Process

The Madrona depth sensor uses a sophisticated two-stage rendering pipeline:

1. **Stage 1: Geometry Rendering** 
   - Uses `batch_draw_*.hlsl` shaders to render scene geometry
   - Populates a visibility buffer (vizBuffer) with depth and object data
   - Shader selection: RGB mode uses `batch_draw_rgb.hlsl`, depth-only mode uses `batch_draw_depth.hlsl`

2. **Stage 2: Depth Processing**
   - Uses `draw_deferred_*.hlsl` shaders to process the vizBuffer
   - Extracts final depth values and writes to depth tensor
   - Shader selection: RGB mode uses `draw_deferred_rgb.hlsl`, depth-only mode uses `draw_deferred_depth.hlsl`

### Render Mode Configuration

The render mode determines which shaders are used and can be modified in `src/mgr.cpp`:

```cpp
return render::RenderManager(render_api, render_dev, {
    .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
    .renderMode = render::RenderManager::Config::RenderMode::Depth,
    .agentViewWidth = mgr_cfg.batchRenderViewWidth,
    .agentViewHeight = mgr_cfg.batchRenderViewHeight,
    .numWorlds = mgr_cfg.numWorlds,
    .maxViewsPerWorld = madEscape::consts::numAgents,
    .maxInstancesPerWorld = madEscape::consts::performance::maxProgressEntries,
});
```

**RGBD Mode** (default):
- Stage 1: `batch_draw_rgb.hlsl` 
- Stage 2: `draw_deferred_rgb.hlsl`
- Generates both RGB and depth data

**Depth Mode**:  
- Stage 1: `batch_draw_depth.hlsl`
- Stage 2: `draw_deferred_depth.hlsl`
- Generates only depth data (more efficient)

### Data Flow Path

```
Python: mgr.depth_tensor()
    ↓
Manager::depthTensor() (mgr.cpp:828)
    ↓  
RenderManager::batchRendererDepthOut() (render_mgr.cpp:74)
    ↓
BatchRenderer::getDepthCUDAPtr() (batch_renderer.cpp:2432)
    ↓
depthOutputCUDA buffer (written by draw_deferred_*.hlsl)
```

## Troubleshooting

### Common Issues

**Empty depth tensor**: Ensure `enable_batch_renderer=True` when creating SimManager

**All-zero depth values**: 
- Check if vizBuffer is being populated in Stage 1 geometry rendering
- Verify camera frustum intersects scene geometry  
- For custom resolutions (e.g., 128×1), ensure FOV settings are appropriate

**GPU memory errors**: Reduce resolution or number of worlds for GPU execution

**Unexpected depth values**: Check that scenes have geometry within the 0.001-20,000 unit range

**Resolution mismatch**: Verify tensor shapes match expected (worlds, agents, height, width, 1)

**Shader debugging**: 
- 64×64 configuration: Known working baseline
- 128×1 configuration: May have camera frustum issues requiring investigation