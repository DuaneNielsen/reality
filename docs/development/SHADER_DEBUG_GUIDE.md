# Shader Debug Guide: Debugging Depth Tensor Generation Pipeline

## Overview

This guide explains the actual depth tensor generation pipeline and how to debug infinity issues in depth calculations. **Note**: The depth tensor uses a two-stage pipeline, not direct shader output.

## Actual Pipeline

### Stage 1: Depth Rasterization
- **File**: `external/madrona/src/render/shaders/batch_draw_depth.hlsl`
- **Output**: Depth render target (LayeredTarget.depth)
- **Function**: Renders geometry depth to render target

### Stage 2: Depth Extraction (THE ACTUAL TENSOR SOURCE)
- **File**: `external/madrona/src/render/shaders/draw_deferred_depth.hlsl`
- **Type**: Compute shader
- **Input**: Depth render target from Stage 1
- **Output**: Linear buffer that becomes the depth tensor
- **Key Code**: Lines 594-596, 758

## Critical Discovery: Where to Debug

### ❌ WRONG: Debugging batch_draw_depth.hlsl
- Outputs to render target, NOT tensor
- Debug code here won't affect tensor values
- Visual debugging only affects rasterization stage

### ✅ CORRECT: Debugging draw_deferred_depth.hlsl
- Compute shader that writes to tensor buffer
- Line 598: `depthOutputBuffer[out_pixel_idx] = depth;`
- This is where tensor infinity issues occur

## Debug Strategy

### 1. Compute Shader Debug Points

The actual tensor generation happens in `draw_deferred_depth.hlsl`:

```hlsl
// Line ~594-596: Depth extraction from render target
float depth = vizBuffer[target_idx][vbuffer_pixel + 
               uint3(x_pixel_offset, y_pixel_offset, 0)];

// Line ~598: ACTUAL tensor write (debug here)
depthOutputBuffer[out_pixel_idx] = depth;
```

### 2. Debug Code Template for Compute Shader

Add this debug code in `draw_deferred_depth.hlsl` around line 598:

```hlsl
// Before: depthOutputBuffer[out_pixel_idx] = depth;

// Debug: Check for infinity/NaN before tensor write
if (isinf(depth)) {
    depth = 999999.0; // Red flag value for infinity
} else if (isnan(depth)) {
    depth = 888888.0; // Red flag value for NaN
} else if (depth > 10000.0) {
    depth = 777777.0; // Red flag for extreme values
}

depthOutputBuffer[out_pixel_idx] = depth;
```

### 3. Thread Dispatch Issues (128x1 Problem)

The compute shader uses `[numThreads(32, 32, 1)]` which may cause issues with extreme aspect ratios:

```hlsl
// Current thread group size optimized for square viewports
[numThreads(32, 32, 1)]
void renderPixel(uint3 threadId : SV_DispatchThreadID)
```

**128x1 viewport issue**: Thread dispatch calculation may not handle extreme aspect ratios correctly.

## Usage Instructions

### 1. Modify the Correct Shader

Edit `external/madrona/src/render/shaders/draw_deferred_depth.hlsl` (NOT `batch_draw_depth.hlsl`).

### 2. Add Debug Code

Insert debug validation around line 598 where `depthOutputBuffer` is written to.

### 3. Rebuild Pipeline

```bash
cd external/madrona
cmake --build build  # Rebuild Madrona with shader changes
cd ../..
make -C build -j16   # Rebuild main project
```

### 4. Run and Check Tensor Values

```bash
./build/viewer
# Check depth tensor for debug marker values:
# 999999.0 = infinity detected
# 888888.0 = NaN detected  
# 777777.0 = extreme values
```

### 5. Data Flow Debugging

The tensor path is:
```
batch_draw_depth.hlsl → render target → draw_deferred_depth.hlsl → tensor buffer
```

Debug both stages:
1. **Stage 1 Issues**: Scale/geometry problems in rasterization
2. **Stage 2 Issues**: Compute shader indexing/extraction problems

## Known Issues

### 128x1 Viewport Optimization
- **Status**: ✅ **RESOLVED** - Infinity values issue fixed
- **Remaining**: Performance optimization for compute shader thread dispatch
- **Current**: Works correctly but may be suboptimal for extreme aspect ratios

### Debug Limitations
- Requires manual shader modification and rebuild
- Debug markers may affect performance
- Cannot debug both pipeline stages simultaneously

## Build System Integration

Currently requires manual shader editing. Future improvements:

```cmake
# Proposed: Conditional debug shader compilation
if(ENABLE_SHADER_DEBUG)
    add_definitions(-DSHADER_DEBUG_MODE=1)
endif()
```

## Investigation Status

✅ **COMPLETED**: Pipeline identification  
✅ **COMPLETED**: Correct debug location identified  
✅ **COMPLETED**: 128x1 infinity values issue resolved  
🔄 **IN PROGRESS**: 128x1 performance optimization  
⏸️ **PENDING**: Automated debug mode integration  

## Reference Documents

- [Shader Pipeline Investigation](../status/shader_pipeline_actual_path.md) - Complete pipeline analysis
- [Madrona Render Architecture](external/madrona/docs/render.md) - Framework docs