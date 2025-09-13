# Multi-World Grid Shader Interface

## Overview

The multi-world grid system enables proper spacing and layout of multiple simulation worlds in the viewer. This document describes the complete interface between the CPU application and GPU shader for grid layout control.

## Data Flow Architecture

```
Application (viewer.cpp)
    ↓ setMultiWorldGrid()
ViewerControl struct
    ↓ viewer_renderer.cpp
DrawPushConst (GPU push constants)
    ↓ viewer_draw.hlsl
Shader grid calculation
```

## Interface Components

### 1. ViewerControl Parameters

Located in: `external/madrona/src/viz/viewer_common.hpp`

```cpp
struct ViewerControl {
    // ... existing fields ...
    
    // Multi-world grid layout parameters
    bool multiWorldGrid;      // Enable/disable grid layout
    uint32_t gridCols;        // Number of columns in grid (e.g., 4 for 4x4)
    float worldSpacing;       // Buffer space between worlds (e.g., 10.0f)
    float worldScaleX;        // World width in units (level-dependent)
    float worldScaleY;        // World height in units (level-dependent)
};
```

### 2. CPU API Interface

Located in: `external/madrona/include/madrona/viz/viewer.hpp`

```cpp
class Viewer {
public:
    // Set grid parameters with full control
    void setMultiWorldGrid(bool enabled, float spacing = 10.0f, uint32_t gridCols = 4, 
                          float worldScaleX = 40.0f, float worldScaleY = 40.0f);
    
    // Toggle grid on/off only (preserves existing parameters)
    void setMultiWorldGrid(bool enabled);
};
```

### 3. GPU Push Constants

Located in: `external/madrona/src/render/shaders/shader_common.h`

```cpp
struct DrawPushConst {
    // ... existing fields ...
    
    // Grid layout parameters (passed to shader)
    uint32_t multiWorldGrid;  // 0 = disabled, 1 = enabled
    uint32_t gridCols;        // Grid columns
    float worldSpacing;       // Buffer between worlds
    float worldScaleX;        // World width
    float worldScaleY;        // World height
};
```

### 4. Shader Implementation

Located in: `external/madrona/src/render/shaders/viewer_draw.hlsl`

```hlsl
// Helper function for grid position calculation
float3 calculateGridPosition(uint worldId, uint gridCols, float worldWidth, float worldHeight, float spacing) {
    uint row = worldId / gridCols;
    uint col = worldId % gridCols;
    
    float x_spacing = worldWidth + spacing;
    float y_spacing = worldHeight + spacing;
    
    return float3(col * x_spacing, row * y_spacing, 0);
}

// Main grid offset function
float3 getWorldGridOffset(uint worldId) {
    if (push_const.multiWorldGrid == 0) {
        return float3(0, 0, 0);  // No offset when disabled
    }
    
    // Use push constant values from CPU
    return calculateGridPosition(
        worldId, 
        push_const.gridCols,
        push_const.worldScaleX,
        push_const.worldScaleY,
        push_const.worldSpacing
    );
}
```

## Parameter Calculation

### Level-Aware Scaling

World dimensions are calculated dynamically based on level data:

```cpp
// In viewer.cpp
const CompiledLevel* level = mgr.getCompiledLevel(0);
float worldScaleX = level->width * 5.0f;   // Grid cells are ~5 world units each
float worldScaleY = level->height * 5.0f;
```

### Grid Layout Examples

**4x4 Grid (16 worlds):**
- `gridCols = 4`
- World positions: (0,0), (50,0), (100,0), (150,0), (0,50), (50,50), etc.

**2x4 Grid (8 worlds):**
- `gridCols = 4` 
- World positions: (0,0), (50,0), (100,0), (150,0), (0,50), (50,50), (100,50), (150,50)

**Default Parameters:**
- `worldSpacing = 10.0f` (buffer between worlds)
- `gridCols = 4` (4 columns)
- `worldScaleX/Y = level.width/height * 5.0f` (level-dependent)

## Usage Patterns

### Runtime Control

```cpp
// Enable multi-world grid with custom parameters
viewer.setMultiWorldGrid(true, 15.0f, 3, 80.0f, 120.0f);

// Disable grid (worlds overlap at origin)
viewer.setMultiWorldGrid(false);

// Toggle via 'M' key (handled automatically in viewer.cpp)
```

### Level Support

The system automatically adapts to different level sizes:

- **16×16 default levels**: `worldScale = 16 * 5.0 = 80.0f`
- **19×48 custom levels**: `worldScaleX = 19 * 5.0 = 95.0f, worldScaleY = 48 * 5.0 = 240.0f`
- **Any custom level**: Dynamic calculation based on level dimensions

## Implementation Status

✅ **Stage 1 Complete**: Hardcoded parameterized grid function  
✅ **Stage 2 Complete**: Dynamic external parameters  

- All hardcoded values replaced with external configuration
- Level-aware world scaling implemented
- Runtime toggle functionality working
- Full parameter control through API
- Clean data flow from application to shader

## Testing

Test with different configurations:

```bash
# Test with 4 worlds (2×2 grid)
./build/viewer --num-worlds 4

# Test with 8 worlds (2×4 grid) 
./build/viewer --num-worlds 8

# Test with 16 worlds (4×4 grid)
./build/viewer --num-worlds 16

# Test with custom level
./build/viewer --num-worlds 4 --load levels/lvl11_19x48_gen.lvl

# Press 'M' key in viewer to toggle grid on/off
```

The multi-world grid system is now production-ready with full dynamic parameter support.