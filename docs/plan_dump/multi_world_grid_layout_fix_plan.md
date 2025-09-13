# Multi-World Grid Layout Fix Plan

## Problem Summary
The multi-world grid visualization is failing because all worlds are rendering on top of each other with zero offsets. Investigation revealed that while the shader logic is correct, it's receiving zero or incorrect values from the push constants that should control grid spacing.

## Current State
- **Working Version**: Hardcoded values in `viewer_draw.hlsl` successfully create proper grid spacing
- **Issue**: Push constant values (`worldSpacing`, `worldScaleX`, `worldScaleY`, `gridCols`) are not being properly passed from the CPU side to the shader
- **Root Cause**: The data flow from `ViewerControl` → `DrawPushConst` → shader is broken somewhere

## Stage 1: Create Parameterized Grid Function with Hardcoded Values

### Objective
Replace the current hardcoded implementation with a clean, parameterized function that can be easily tested and verified before connecting external parameters.

### Implementation Steps

1. **Refactor `getWorldGridOffset()` in `viewer_draw.hlsl`**:
   ```hlsl
   // Create a helper function to calculate grid layout
   float3 calculateGridPosition(uint worldId, uint gridCols, float worldWidth, float worldHeight, float spacing) {
       uint row = worldId / gridCols;
       uint col = worldId % gridCols;
       
       float x_spacing = worldWidth + spacing;
       float y_spacing = worldHeight + spacing;
       
       return float3(col * x_spacing, row * y_spacing, 0);
   }
   
   float3 getWorldGridOffset(uint worldId) {
       if (push_const.multiWorldGrid == 0) {
           return float3(0, 0, 0);
       }
       
       // Stage 1: Hardcode parameters directly for testing
       const uint GRID_COLS = 4;
       const float WORLD_WIDTH = 40.0f;   // 16 tiles * 2.5 * 1.0 scale
       const float WORLD_HEIGHT = 40.0f;  // 16 tiles * 2.5 * 1.0 scale  
       const float SPACING = 10.0f;
       
       return calculateGridPosition(worldId, GRID_COLS, WORLD_WIDTH, WORLD_HEIGHT, SPACING);
   }
   ```

2. **Test with different level sizes**:
   - For default 16x16 level: WORLD_WIDTH=40.0, WORLD_HEIGHT=40.0
   - For lvl11_19x48_gen.lvl: WORLD_WIDTH=142.5, WORLD_HEIGHT=360.0
   - Verify grid spacing works correctly with hardcoded values

3. **Build and verify**:
   - Run `./build.sh`
   - Test with `./build/viewer --num-worlds 4 --load levels/lvl11_19x48_gen.lvl`
   - Press 'M' to enable multi-world grid
   - Confirm worlds are properly spaced

## Stage 2: Connect External Parameters to Shader

### Objective
Fix the data flow from CPU to GPU so that dynamic parameters can be passed through push constants.

### Investigation Steps

1. **Trace the data flow path**:
   ```
   ViewerControl (viz_ctrl) 
      ↓
   DrawPushConst (in viewer_renderer.cpp)
      ↓
   push_const (in shader)
   ```

2. **Debug push constant values in `viewer_renderer.cpp`**:
   ```cpp
   // Add debug prints before creating DrawPushConst
   printf("DEBUG: Creating DrawPushConst with:\n");
   printf("  multiWorldGrid: %d\n", viz_ctrl.multiWorldGrid ? 1 : 0);
   printf("  gridCols: %u\n", viz_ctrl.gridCols);
   printf("  worldSpacing: %f\n", viz_ctrl.worldSpacing);
   printf("  worldScaleX: %f\n", viz_ctrl.worldScaleX);
   printf("  worldScaleY: %f\n", viz_ctrl.worldScaleY);
   
   DrawPushConst draw_const {
       // ... existing fields ...
       viz_ctrl.multiWorldGrid ? 1u : 0u,
       viz_ctrl.gridCols,
       viz_ctrl.worldSpacing,
       viz_ctrl.worldScaleX,
       viz_ctrl.worldScaleY,
   };
   ```

3. **Verify ViewerControl initialization**:
   - Check where `viz_ctrl` is populated
   - Ensure `viewer.setMultiWorldGrid()` is properly updating the values
   - Verify the values persist across frames

4. **Connect parameters in shader**:
   ```hlsl
   float3 getWorldGridOffset(uint worldId) {
       if (push_const.multiWorldGrid == 0) {
           return float3(0, 0, 0);
       }
       
       // Stage 2: Use push constant values
       return calculateGridPosition(
           worldId, 
           push_const.gridCols,
           push_const.worldScaleX,
           push_const.worldScaleY,
           push_const.worldSpacing
       );
   }
   ```

### Potential Issues to Check

1. **Push constant size mismatch**:
   - Verify `sizeof(DrawPushConst)` matches between CPU and GPU
   - Check alignment requirements for push constants

2. **ViewerControl not persisting**:
   - The values might be set temporarily but reset before rendering
   - Check if `viz_ctrl` is being copied or referenced

3. **Shader compilation caching**:
   - Clear shader cache: `rm -rf build/madrona_kernels.cache`
   - Force shader recompilation

### Testing Protocol

1. Start with Stage 1 hardcoded values to confirm shader logic works
2. Add debug output at each stage of the data flow
3. Gradually replace hardcoded values with push constant values one at a time:
   - First test `gridCols`
   - Then add `worldSpacing`
   - Finally add `worldScaleX` and `worldScaleY`
4. Identify exactly where the values become zero

### Success Criteria

- Multi-world grid displays with proper spacing for any level size
- Grid layout adapts dynamically based on level dimensions
- Works correctly with both default levels and custom levels like lvl11_19x48_gen.lvl
- No hardcoded values remain in the final implementation