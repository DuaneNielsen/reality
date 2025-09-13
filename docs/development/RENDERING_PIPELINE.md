# Rendering Pipeline Guide: Multi-World System & Shader APIs

## Overview

This guide documents the Madrona escape room rendering pipeline, focusing on the multi-world grid system and shader interfaces. The pipeline supports rendering multiple simulation worlds either as a grid layout or individual worlds.

## Multi-World Rendering Architecture

### Two Rendering Modes

1. **Single World Mode**: Renders only the selected world at the origin
2. **Multi-World Grid Mode**: Renders all worlds in a configurable grid layout

### Control API

#### C++ Viewer API
```cpp
// Enable/disable multi-world grid with full configuration
viewer.setMultiWorldGrid(enabled, spacing, gridCols, worldScaleX, worldScaleY);

// Enable/disable with existing parameters
viewer.setMultiWorldGrid(enabled);

// Example: 2x2 grid with 10.0 spacing
viewer.setMultiWorldGrid(true, 10.0f, 2, 40.0f, 40.0f);
```

#### ViewerControl Structure
```cpp
struct ViewerControl {
    // ... other fields ...
    
    // Multi-world grid layout parameters
    bool multiWorldGrid;     // Enable grid mode
    uint32_t gridCols;      // Columns in grid layout
    float worldSpacing;     // Space between worlds
    float worldScaleX;      // World width for layout calculation
    float worldScaleY;      // World height for layout calculation
};
```

## Culling Pipeline

### CullPushConst Interface

**File**: `external/madrona/src/render/shaders/shader_common.h`

```cpp
struct CullPushConst {
    uint32_t startRenderWorldIdx;  // First world to render (inclusive)
    uint32_t endRenderWorldIdx;    // Last world to render (inclusive)  
    uint32_t numThreads;           // Thread count for work distribution
    uint32_t totalInstances;       // Total instances across ALL worlds
    uint32_t totalWorlds;          // Total number of worlds in simulation
};
```

### Culling Logic

**File**: `external/madrona/src/render/shaders/viewer_cull.hlsl`

#### Key Functions
```hlsl
// Get instance count for a specific world
uint getNumInstancesForWorld(uint world_idx)
{
    if (world_idx == pushConst.totalWorlds - 1) {
        return pushConst.totalInstances - instanceOffsets[world_idx];
    } else {
        return instanceOffsets[world_idx+1] - instanceOffsets[world_idx];
    }
}

// Get starting instance offset for a world
uint getInstanceOffsetsForWorld(uint world_idx)
{
    return instanceOffsets[world_idx];
}
```

#### Main Culling Loop
```hlsl
if (tid_local.x == 0) {
    sm.numInstances = 0;
    // Always use the same logic - iterate from start to end (inclusive)
    for (uint world = pushConst.startRenderWorldIdx; 
         world <= pushConst.endRenderWorldIdx; ++world) {
        sm.numInstances += getNumInstancesForWorld(world);
    }
    sm.instancesOffset = getInstanceOffsetsForWorld(pushConst.startRenderWorldIdx);
    sm.numInstancesPerThread = (sm.numInstances + pushConst.numThreads-1) /
                               pushConst.numThreads;
}
```

### CPU-Side Logic

**File**: `external/madrona/src/viz/viewer_renderer.cpp`

```cpp
// Determine world range based on multi-world grid setting
uint32_t start_render_world_idx, end_render_world_idx;
if (viz_ctrl.multiWorldGrid) {
    // Grid mode: render all worlds
    start_render_world_idx = 0;
    end_render_world_idx = num_worlds - 1;
} else {
    // Single world mode: render only selected world
    start_render_world_idx = viz_ctrl.worldIdx;
    end_render_world_idx = viz_ctrl.worldIdx;
}

CullPushConst cull_push_const {
    start_render_world_idx,
    end_render_world_idx,
    num_warps * 32,         // numThreads
    num_instances,          // totalInstances
    num_worlds              // totalWorlds
};
```

## World Grid Layout System

### Grid Position Calculation

**File**: `external/madrona/src/render/shaders/viewer_draw.hlsl`

```hlsl
// Calculate grid layout position for a world
float3 calculateGridPosition(uint worldId, uint gridCols, float worldWidth, 
                           float worldHeight, float spacing) {
    uint row = worldId / gridCols;
    uint col = worldId % gridCols;
    
    float x_spacing = worldWidth + spacing;
    float y_spacing = worldHeight + spacing;
    
    return float3(col * x_spacing, row * y_spacing, 0);
}

// Get world offset in grid layout
float3 getWorldGridOffset(uint worldId) {
    if (push_const.multiWorldGrid == 0) {
        return float3(0, 0, 0);
    }
    
    return calculateGridPosition(
        worldId, 
        push_const.gridCols,
        push_const.worldScaleX,
        push_const.worldScaleY,
        push_const.worldSpacing
    );
}
```

### Draw Push Constants

The draw shaders receive grid layout parameters via push constants:

```cpp
// In viewer_renderer.cpp
DrawPushConst draw_push_const {
    // ... other fields ...
    viz_ctrl.multiWorldGrid ? 1u : 0u,  // multiWorldGrid
    viz_ctrl.gridCols,                   // gridCols
    viz_ctrl.worldSpacing,               // worldSpacing
    viz_ctrl.worldScaleX,                // worldScaleX
    viz_ctrl.worldScaleY                 // worldScaleY
};
```

## Usage Examples

### Single World Rendering
```cpp
// Render only world 2
start_render_world_idx = 2;
end_render_world_idx = 2;
```

### Multi-World Grid Rendering
```cpp
// Render all worlds (0-3) in 2x2 grid
start_render_world_idx = 0;
end_render_world_idx = 3;
multiWorldGrid = true;
gridCols = 2;
worldSpacing = 10.0f;
```

### Partial World Range
```cpp
// Render worlds 1-2 only
start_render_world_idx = 1;
end_render_world_idx = 2;
```

## Key Design Principles

### 1. Inclusive Range Design
- Both `startRenderWorldIdx` and `endRenderWorldIdx` are inclusive
- Single world: `start == end`
- Multi-world: `start < end`

### 2. Unified Shader Logic
- No branching in shaders for single vs multi-world
- Single loop handles all cases consistently
- Cleaner, more maintainable code

### 3. Parameterized Grid Layout
- All grid parameters configurable via API
- No hardcoded layout values in shaders
- Flexible for different world sizes and layouts

## Debugging Tips

### Culling Issues
- Check `CullPushConst` values in debugger
- Verify `startRenderWorldIdx <= endRenderWorldIdx`
- Ensure `totalInstances` and `totalWorlds` are correct

### Grid Layout Issues
- Verify `push_const.multiWorldGrid` flag
- Check grid calculation parameters
- Use viewer keyboard controls to test single vs grid mode

### Testing Multi-World Toggle
```bash
# Test with 4 worlds
./build/viewer --num-worlds 4

# In viewer:
# - Press 'M' to toggle multi-world grid
# - Should switch between single world and 2x2 grid
# - Grid OFF: Only world 0 visible at origin
# - Grid ON: All 4 worlds visible in grid layout
```

## Pipeline Flow Summary

1. **CPU**: Determines render world range based on `viz_ctrl.multiWorldGrid`
2. **CPU**: Populates `CullPushConst` with inclusive range
3. **GPU Culling**: Processes all worlds in range with unified loop
4. **GPU Drawing**: Applies grid offset for multi-world layout
5. **Result**: Single world at origin OR all worlds in grid

This design provides a clean, efficient, and maintainable multi-world rendering system.