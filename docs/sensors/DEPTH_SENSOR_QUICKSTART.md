# Depth Sensor Quickstart Guide

This guide explains how to use the depth sensor capabilities in Madrona Escape Room for reinforcement learning applications.

## Overview

The depth sensor provides agents with distance measurements from their camera viewpoint, similar to how lidar or depth cameras work in real robotics applications. Unlike traditional Z-buffer depth, Madrona provides **linear world-space distances** that are ideal for RL spatial reasoning.

## Basic Usage

### Enable Depth Sensor
```python
from madrona_escape_room import SimManager, ExecMode

# Create manager with batch renderer enabled (required for depth sensor)
mgr = SimManager(
    exec_mode=ExecMode.CPU,  # or ExecMode.CUDA
    num_worlds=4,
    enable_batch_renderer=True  # ← Required for depth sensor
)
```

### Access Depth Data
```python
import numpy as np

# Step the simulation
mgr.step()

# Get depth tensor
depth_tensor = mgr.depth_tensor()

# Convert to numpy for processing
if depth_tensor.isOnGPU():
    depth_np = depth_tensor.to_torch().cpu().numpy()
else:
    depth_np = depth_tensor.to_numpy()

print(f"Depth tensor shape: {depth_np.shape}")
# Output: (num_worlds, num_agents, height, width, 1)
# Default: (4, 1, 64, 64, 1)
```

## Camera Configuration

### Default Settings
- **Resolution**: 64×64 pixels
- **Vertical FOV**: 100° (very wide angle)
- **Horizontal FOV**: 100° (calculated from 1:1 aspect ratio)
- **Near plane**: 0.001 world units
- **Far plane**: 20,000 world units
- **Camera position**: 1.5 units above agent center
- **Camera orientation**: Follows agent rotation

### Camera Position Relative to Agent
```
Agent position: (x, y, 0)
Camera position: (x, y, 1.5)  ← 1.5 units above agent
Field of view: 100° vertical, 100° horizontal (for square resolution)
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

## Coordinate System

### Tensor Layout
- **Dimension 0**: World index
- **Dimension 1**: Agent index  
- **Dimension 2**: Image height (Y axis, top-to-bottom)
- **Dimension 3**: Image width (X axis, left-to-right)
- **Dimension 4**: Single depth channel

### Spatial Mapping
```python
# Tensor coordinate to spatial meaning:
depth_value = depth_np[world, agent, y, x, 0]

# Where:
# y=0: Top of agent's view (looking up)
# y=31: Center vertically (looking forward)  
# y=63: Bottom of agent's view (looking down)
# x=0: Left side of agent's view
# x=31: Center horizontally (straight ahead)
# x=63: Right side of agent's view
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

## Custom Resolution (Advanced)

The default 64×64 resolution can be customized for different applications:

```python
# Note: This requires C++ code modifications (see advanced documentation)
# Example configurations:

# High resolution for detailed perception
# config.batchRenderViewWidth = 128
# config.batchRenderViewHeight = 128

# Lidar-like horizontal scanning (128 beams across horizon)  
# config.batchRenderViewWidth = 128
# config.batchRenderViewHeight = 1
# Requires custom vertical FOV configuration

# Current default
# config.batchRenderViewWidth = 64
# config.batchRenderViewHeight = 64
```

## Testing with Pytest

```python
import pytest
from madrona_escape_room import SimManager, ExecMode

@pytest.mark.depth_sensor
def test_depth_sensor_basic():
    """Test basic depth sensor functionality."""
    
    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        enable_batch_renderer=True
    )
    
    # Step simulation  
    mgr.step()
    
    # Get depth data
    depth = mgr.depth_tensor()
    depth_np = depth.to_numpy()
    
    # Verify tensor properties
    assert depth_np.shape == (1, 1, 64, 64, 1)
    assert depth_np.dtype == np.float32
    assert np.all(depth_np >= 0.001)  # No values below near plane
    assert np.all(depth_np <= 20000)  # No values above far plane
```

## Performance Considerations

### Memory Usage
- **64×64 resolution**: ~16KB per agent per world (float32)
- **128×128 resolution**: ~65KB per agent per world  
- **GPU memory**: Allocated on GPU when using CUDA execution mode

### Computational Cost
- **Lower resolution**: Faster rendering, less memory, reduced spatial detail
- **Higher resolution**: Better spatial awareness, more GPU work, increased memory
- **Aspect ratio effects**: Wide/tall resolutions change horizontal/vertical FOV

## Integration with RGB Data

Depth and RGB sensors can be used together:

```python
# Get both depth and RGB data
depth = mgr.depth_tensor()      # Shape: (worlds, agents, 64, 64, 1)
rgb = mgr.rgb_tensor()          # Shape: (worlds, agents, 64, 64, 4)

# Convert to numpy
depth_np = depth.to_numpy() if not depth.isOnGPU() else depth.to_torch().cpu().numpy()
rgb_np = rgb.to_numpy() if not rgb.isOnGPU() else rgb.to_torch().cpu().numpy()

# Both sensors share the same viewpoint and resolution
# Pixel correspondence: depth_np[w,a,y,x,0] corresponds to rgb_np[w,a,y,x,:]
```

## Next Steps

- **Advanced configurations**: See `/docs/sensors/DEPTH_SENSOR_ADVANCED.md` (future)
- **Lidar simulation**: See `/docs/sensors/LIDAR_CONFIGURATION.md` (future)
- **Custom FOV settings**: See `/docs/sensors/CAMERA_CONFIGURATION.md` (future)
- **Performance optimization**: See `/docs/development/PERFORMANCE_GUIDE.md`

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