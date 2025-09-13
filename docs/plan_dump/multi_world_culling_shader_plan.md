# Multi-World Culling Shader Implementation Plan

## Reading List

Before implementing this plan, review these files to understand the current architecture:

### Core Files to Read:
1. **`external/madrona/src/render/shaders/viewer_cull.hlsl`** - Current culling shader implementation
2. **`external/madrona/src/render/shaders/shader_common.h`** - CullPushConst structure definition
3. **`external/madrona/src/viz/viewer_renderer.cpp`** - Lines 2030-2070 (issueCulling function)
4. **`external/madrona/src/viz/viewer_renderer.cpp`** - Lines 2794-2920 (renderFlycamFrame function)
5. **`external/madrona/src/viz/viewer_common.hpp`** - Lines 31-56 (ViewerControl structure)

### Key Understanding Points:
- How `instanceOffsets` array organizes instances by world
- How `getNumInstancesForWorld()` and `getInstanceOffsetsForWorld()` work
- How `CullPushConst` passes parameters to the shader
- How `viz_ctrl.multiWorldGrid` flag indicates when multi-world mode is active

---

## Plan: Enable Multi-World Rendering in Culling Shader

### Problem
The culling shader currently filters instances to only render one world at a time. For multi-world grid mode, we need it to process instances from multiple consecutive worlds.

### Solution Approach
Modify the culling shader to process a contiguous range of instances when multi-world grid mode is enabled.

### Implementation Steps

#### 1. Update CullPushConst structure (shader_common.h)
Add fields to specify instance range:
- Add `startWorldIdx` - first world to render
- Add `numWorldsToRender` - how many consecutive worlds to process
- Keep existing fields for compatibility

#### 2. Modify viewer_cull.hlsl
Update the culling logic to:
- When processing multiple worlds: calculate total instances across the world range
- Set `sm.numInstances` to sum of instances from `startWorldIdx` to `startWorldIdx + numWorldsToRender - 1`
- Set `sm.instancesOffset` to offset of `startWorldIdx`
- The rest of the shader remains unchanged - it will naturally process all instances in the range

#### 3. Update issueCulling function (viewer_renderer.cpp)
Modify the function to:
- Check if `viz_ctrl.multiWorldGrid` is true
- If true: set `startWorldIdx = 0` and `numWorldsToRender = num_worlds`
- If false: set `startWorldIdx = world_idx` and `numWorldsToRender = 1`
- Pass these values in the CullPushConst

#### 4. Update renderFlycamFrame (viewer_renderer.cpp)
Pass `viz_ctrl` to `issueCulling` so it can access the `multiWorldGrid` flag.

### Expected Result
- When multi-world grid is OFF: renders only the selected world (current behavior)
- When multi-world grid is ON: renders all worlds with their grid offsets applied by the vertex shader

### Testing
After implementation, test with:
- `./build/viewer --num-worlds 4`
- Press M to toggle multi-world grid
- Should see all 4 worlds arranged in a 2x2 grid