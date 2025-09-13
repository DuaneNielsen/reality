# Implementation Plan: Grid-Based Multi-World Rendering with Spatial Offsets

## Approach
Render all worlds simultaneously by applying spatial offsets to instance positions based on their world ID. Each world will be positioned in a grid layout in 3D space.

## Implementation Steps

### Step 1: Add Multi-World Grid Mode Toggle
**File: `src/viewer_core.hpp`** (~line 80)
```cpp
enum Key { W, A, S, D, Q, E, R, T, Space, Shift, M };  // Add 'M' key
```

**File: `src/viewer_core.hpp`** (~line 66, in Config struct)
```cpp
bool multi_world_grid = false;  // Enable multi-world grid layout
float world_spacing = 10.0f;    // Spacing between worlds
uint32_t grid_cols = 8;         // Grid columns (configurable)
```

**File: `src/viewer_core.cpp`** (in handleInput)
```cpp
if (event.type == InputEvent::KeyHit && event.key == InputEvent::M) {
    config_.multi_world_grid = !config_.multi_world_grid;
    printf("Multi-world grid: %s\n", config_.multi_world_grid ? "ON" : "OFF");
}
```

### Step 2: Pass Grid Settings Through Rendering Pipeline
**File: `external/madrona/src/viz/viewer_common.hpp`**
```cpp
struct ViewerControl {
    // ... existing fields ...
    bool multiWorldGrid = false;
    float worldSpacing = 10.0f;
    uint32_t gridCols = 8;
    float worldScaleX = 20.0f;  // Level width * world_scale
    float worldScaleY = 40.0f;  // Level height * world_scale
};
```

**File: `src/viewer.cpp`** (where ViewerControl is populated)
```cpp
// Pass grid settings from ViewerCore to ViewerControl
viz::ViewerControl viz_ctrl;
// ... existing field assignments ...
viz_ctrl.multiWorldGrid = viewer_core->config_.multi_world_grid;
viz_ctrl.worldSpacing = viewer_core->config_.world_spacing;
viz_ctrl.gridCols = viewer_core->config_.grid_cols;

// Get level dimensions for world scaling
const CompiledLevel& level = mgr->getCompiledLevel();
viz_ctrl.worldScaleX = level.width * level.world_scale;
viz_ctrl.worldScaleY = level.height * level.world_scale;
```

### Step 3: Modify DrawPushConst Structure  
**File: `external/madrona/src/render/shaders/shader_common.h`** (line 101)
```hlsl
struct DrawPushConst {
    uint32_t viewIdx;
    uint32_t worldIdx;

    uint32_t isOrtho;
    float xMax;
    float xMin;
    float yMax;
    float yMin;
    float zMax;
    float zMin;
    
    // Add grid layout parameters
    uint32_t multiWorldGrid;
    uint32_t gridCols;
    float worldSpacing;
    float worldScaleX;
    float worldScaleY;
};
```

### Step 4: Apply World Offsets in Vertex Shader
**File: `external/madrona/src/render/shaders/viewer_draw.hlsl`** (add function before vert shader ~line 186)
```hlsl
float3 getWorldGridOffset(uint worldId) {
    if (push_const.multiWorldGrid == 0) {
        return float3(0, 0, 0);
    }
    
    uint row = worldId / push_const.gridCols;
    uint col = worldId % push_const.gridCols;
    
    float cell_width = push_const.worldScaleX + push_const.worldSpacing;
    float cell_height = push_const.worldScaleY + push_const.worldSpacing;
    
    return float3(col * cell_width, row * cell_height, 0);
}

// In vert function (modify around line 204):
EngineInstanceData instance_data = unpackEngineInstanceData(
    engineInstanceBuffer[instance_id]);

// Apply world grid offset
float3 worldOffset = getWorldGridOffset(instance_data.worldID);
float3 adjustedPosition = instance_data.position + worldOffset;

// Use adjustedPosition in computeCompositeTransform:
computeCompositeTransform(adjustedPosition, instance_data.rotation,
    view_data.pos, view_data.rot,
    to_view_translation, to_view_rotation);
```

**File: `external/madrona/src/viz/viewer_renderer.cpp`** (around line 3047)
```cpp
// Update the DrawPushConst creation to include grid parameters:
DrawPushConst draw_const {
    .viewIdx = view_idx,
    .worldIdx = world_idx,
    .isOrtho = is_ortho ? 1u : 0u,
    .xMax = xMax,
    .xMin = xMin,
    .yMax = yMax,
    .yMin = yMin,
    .zMax = zMax,
    .zMin = zMin,
    // Add grid parameters from ViewerControl
    .multiWorldGrid = viz_ctrl.multiWorldGrid ? 1u : 0u,
    .gridCols = viz_ctrl.gridCols,
    .worldSpacing = viz_ctrl.worldSpacing,
    .worldScaleX = viz_ctrl.worldScaleX,
    .worldScaleY = viz_ctrl.worldScaleY,
};
```

### Step 5: Adjust Camera for Multi-World View
**File: `src/viewer.cpp`** (in the viewer loop)
```cpp
// When multi-world grid is active, adjust camera
if (viewer_core->config_.multi_world_grid) {
    const CompiledLevel& level = mgr->getCompiledLevel();
    uint32_t grid_cols = viewer_core->config_.grid_cols;
    uint32_t num_rows = (num_worlds + grid_cols - 1) / grid_cols;
    
    // Calculate grid dimensions
    float cell_width = level.width * level.world_scale + viewer_core->config_.world_spacing;
    float cell_height = level.height * level.world_scale + viewer_core->config_.world_spacing;
    float grid_width = grid_cols * cell_width;
    float grid_height = num_rows * cell_height;
    
    // Position camera to view entire grid
    float view_distance = std::max(grid_width, grid_height) * 1.5f;
    viewer.setCameraPosition(Vector3(grid_width/2, grid_height/2, view_distance));
}
```

## Command Line Usage
```bash
# View 16 worlds in a grid
./build/viewer --num-worlds 16
# Press 'M' to toggle multi-world grid mode
# Use WASD + mouse to navigate
```

## Testing Strategy
1. Start with 4 worlds (2x2 grid) to verify offsets work
2. Test with 16 worlds (4x4 grid) 
3. Scale to 64 worlds (8x8 grid)
4. Verify performance and visual clarity

## Benefits
- Simple implementation - just adding position offsets
- Works with existing rendering pipeline
- No multiple render passes needed
- Easy toggle between single/multi world view
- Scalable to hundreds of worlds