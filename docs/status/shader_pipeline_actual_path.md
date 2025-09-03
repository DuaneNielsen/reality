# Actual Shader Pipeline for Depth Tensor Generation

**Date**: September 2, 2025  
**Investigation Result**: Complete Pipeline Identified

## Executive Summary

The depth tensor generation does NOT use `batch_draw_depth.hlsl` directly for the final output. Instead, it uses a two-stage pipeline:
1. **Rasterization Stage**: `batch_draw_depth.hlsl` renders depth to a render target
2. **Extraction Stage**: `draw_deferred_depth.hlsl` (compute shader) extracts depth to buffer

## The Real Pipeline

### Stage 1: Depth Rasterization (`batch_draw_depth.hlsl`)
- **Type**: Vertex + Fragment shader
- **Output**: Depth render target (LayeredTarget.depth)
- **Function**: Renders geometry and outputs depth values
- **Key Code**: Line 320 in `batch_draw_depth.hlsl`:
  ```hlsl
  float depth = length(v2f.vsCoord);
  output.depthOut = depth;
  ```

### Stage 2: Depth Extraction (`draw_deferred_depth.hlsl`)
- **Type**: Compute shader (lighting pass)
- **Input**: Depth render target from Stage 1
- **Output**: Linear buffer accessible as tensor
- **Key Code**: Lines 594-596, 758 in `draw_deferred_depth.hlsl`:
  ```hlsl
  float depth = vizBuffer[target_idx][vbuffer_pixel + 
                   uint3(x_pixel_offset, y_pixel_offset, 0)];
  depthOutputBuffer[out_pixel_idx] = depth;
  ```

## Data Flow

```
1. Manager::depthTensor() 
   ↓
2. impl_->renderMgr->batchRendererDepthOut()
   ↓
3. BatchRenderer::getDepthCUDAPtr()
   ↓
4. Returns: depthOutputCUDA.getDevicePointer()
   ↓
5. This buffer is filled by: draw_deferred_depth.hlsl compute shader
```

## Critical Discovery

### Debug Code Location Issues

1. **Wrong Location**: Debug code in `batch_draw_depth.hlsl` (lines 322-362) outputs to the render target, NOT the final tensor
2. **Correct Location**: Debug code must be in `draw_deferred_depth.hlsl` at line 598 where it writes to `depthOutputBuffer`

### Evidence of Compute Shader Usage

Line 598 in `draw_deferred_depth.hlsl` contains forced debug output:
```hlsl
// Debug: FORCE ALL values to a specific marker to test if shader is running
depth = 123456.0; // Force marker to prove shader is running
```

This confirms the compute shader IS the actual path for depth tensor generation.

## 128x1 Resolution Status

**UPDATE**: ✅ **RESOLVED** - The infinity values issue with 128x1 viewports has been fixed.

**Remaining optimization opportunity**: The compute shader thread dispatch may still be suboptimal for extreme aspect ratios:

1. **Thread Group Size**: Uses `[numThreads(32, 32, 1)]` - optimized for square viewports
2. **Index Calculation**: Lines 551-572 calculate view indices assuming certain aspect ratios
3. **Sample UV Calculation**: Lines 582-588 may be inefficient for extreme aspect ratios

## Optimization Strategy (Post-Fix)

1. ✅ **COMPLETED**: Fixed infinity values issue
2. **FUTURE**: Optimize thread dispatch configuration for 128x1 viewports  
3. **FUTURE**: Improve index calculations for extreme aspect ratios
4. **FUTURE**: Consider dynamic thread group sizing based on viewport aspect ratio

## File Locations

- **Depth Rasterization**: `external/madrona/src/render/shaders/batch_draw_depth.hlsl`
- **Depth Extraction**: `external/madrona/src/render/shaders/draw_deferred_depth.hlsl`
- **Batch Renderer**: `external/madrona/src/render/batch_renderer.cpp`
- **Render Manager**: `external/madrona/src/render/render_mgr.cpp`

## Build Requirements

To test shader changes:
1. Modify shader files in `external/madrona/src/render/shaders/`
2. Rebuild Madrona: `cd external/madrona && cmake --build build`
3. Rebuild main project: `make -C build -j16`

## Conclusion

The investigation revealed that depth tensor generation uses a compute shader pipeline, not the direct rasterization output. The 128x1 infinity issue has been resolved. Future work may focus on optimizing the compute shader thread dispatch for extreme aspect ratios to improve performance.