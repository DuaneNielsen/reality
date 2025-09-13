# Culling Interface Refactor Plan

## Problem Statement

The current world culling system has several issues:
1. **Hardcoded behavior**: Always renders all worlds when `num_worlds > 1`, ignoring `viz_ctrl.multiWorldGrid` setting
2. **Confusing interface**: Redundant fields (`worldIDX` vs `startWorldIdx`), implicit mode selection
3. **Bug**: When multi-world grid is disabled, all worlds render overlapped at origin instead of showing only one world

## Root Cause

In `viewer_renderer.cpp` line 2057:
```cpp
// WRONG - Hardcoded testing code still active
bool enable_multi_world_grid = (num_worlds > 1);

// Should use the actual setting:
bool enable_multi_world_grid = viz_ctrl.multiWorldGrid;
```

## Solution: Clean CullPushConst Interface

### Step 1: Update CullPushConst Structure

**File: `external/madrona/src/render/shaders/shader_common.h`**

```cpp
// OLD:
struct CullPushConst {
    uint32_t worldIDX;
    uint32_t numViews;
    uint32_t numInstances;
    uint32_t numWorlds;
    uint32_t numThreads;
    uint32_t startWorldIdx;
    uint32_t numWorldsToRender;
};

// NEW:
struct CullPushConst {
    uint32_t startRenderWorldIdx;  // First world to render (inclusive)
    uint32_t endRenderWorldIdx;    // Last world to render (inclusive)
    uint32_t numThreads;           // Thread count for work distribution
    uint32_t totalInstances;       // Total instances across ALL worlds
    uint32_t totalWorlds;          // Total number of worlds in simulation
};
```

### Step 2: Update Shader Logic

**File: `external/madrona/src/render/shaders/viewer_cull.hlsl`**

```hlsl
// OLD getNumInstancesForWorld function:
uint getNumInstancesForWorld(uint world_idx)
{
    if (world_idx == pushConst.numWorlds - 1) {
        return pushConst.numInstances - instanceOffsets[world_idx];
    } else {
        return instanceOffsets[world_idx+1] - instanceOffsets[world_idx];
    }
}

// NEW (using totalWorlds and totalInstances):
uint getNumInstancesForWorld(uint world_idx)
{
    if (world_idx == pushConst.totalWorlds - 1) {
        return pushConst.totalInstances - instanceOffsets[world_idx];
    } else {
        return instanceOffsets[world_idx+1] - instanceOffsets[world_idx];
    }
}

// OLD instanceCull main logic:
if (tid_local.x == 0) {
    if (pushConst.numWorldsToRender > 1) {
        sm.numInstances = 0;
        for (uint world = pushConst.startWorldIdx; 
             world < pushConst.startWorldIdx + pushConst.numWorldsToRender; ++world) {
            sm.numInstances += getNumInstancesForWorld(world);
        }
        sm.instancesOffset = getInstanceOffsetsForWorld(pushConst.startWorldIdx);
    } else {
        sm.numInstances = getNumInstancesForWorld(pushConst.worldIDX);
        sm.instancesOffset = getInstanceOffsetsForWorld(pushConst.worldIDX);
    }
    sm.numInstancesPerThread = (sm.numInstances + pushConst.numThreads-1) /
                               pushConst.numThreads;
}

// NEW (simpler, consistent):
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

### Step 3: Update CPU-side Logic

**File: `external/madrona/src/viz/viewer_renderer.cpp`**

```cpp
// OLD (line ~2055-2074):
// Stage 1: Hardcoded multi-world parameters for testing
uint32_t start_world_idx, num_worlds_to_render;
bool enable_multi_world_grid = (num_worlds > 1);  // BUG: Hardcoded!
if (enable_multi_world_grid) {
    start_world_idx = 0;
    num_worlds_to_render = num_worlds;
} else {
    start_world_idx = viz_ctrl.worldIdx;
    num_worlds_to_render = 1;
}

CullPushConst cull_push_const {
    viz_ctrl.worldIdx,
    num_views,
    num_instances,
    num_worlds,
    num_warps * 32,
    start_world_idx,
    num_worlds_to_render
};

// NEW:
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

### Step 4: Add World Selection Controls (Optional Enhancement)

**File: `src/viewer.cpp`**

```cpp
// Add keyboard handling for world selection (in the input handler):
// Number keys 1-9 to select worlds 0-8
if (input.keyHit(Key::Num1)) {
    viewer.setWorldIndex(0);
}
if (input.keyHit(Key::Num2)) {
    viewer.setWorldIndex(1);
}
// ... etc up to Num9

// Add [ and ] for prev/next world
if (input.keyHit(Key::LeftBracket)) {
    uint32_t current = viewer.getCurrentWorldID();
    if (current > 0) {
        viewer.setWorldIndex(current - 1);
    }
}
if (input.keyHit(Key::RightBracket)) {
    uint32_t current = viewer.getCurrentWorldID();
    if (current < num_worlds - 1) {
        viewer.setWorldIndex(current + 1);
    }
}
```

**File: `external/madrona/include/madrona/viz/viewer.hpp`**

```cpp
// Add method to Viewer class:
void setWorldIndex(uint32_t worldIdx);
```

**File: `external/madrona/src/viz/viewer.cpp`**

```cpp
// Implementation:
void Viewer::setWorldIndex(uint32_t worldIdx)
{
    impl_->vizCtrl.worldIdx = worldIdx;
}
```

## Benefits of This Design

1. **Cleaner interface** - No redundant fields or implicit modes
2. **Bug fix** - Respects `viz_ctrl.multiWorldGrid` instead of hardcoded logic
3. **Simpler shader** - One consistent loop instead of if/else branches
4. **Better usability** - Optional world selection controls

## Key Improvements

- Explicit `startRenderWorldIdx` and `endRenderWorldIdx` (inclusive range)
- Removal of unused `numViews` field
- Elimination of redundant `worldIDX` vs `startWorldIdx` confusion
- Consistent shader logic without special cases

## Usage Examples

- **Single world 2**: `startRenderWorldIdx = 2, endRenderWorldIdx = 2`
- **All worlds (0-3)**: `startRenderWorldIdx = 0, endRenderWorldIdx = 3`
- **Range (1-2)**: `startRenderWorldIdx = 1, endRenderWorldIdx = 2`

## Test Cases

1. **Test multi-world grid toggle:**
   - Run with `--num-worlds 4`
   - Press 'M' to toggle grid on/off
   - Grid ON: Should see 4 worlds in 2x2 layout
   - Grid OFF: Should see only world 0 at origin

2. **Test world selection (if implemented):**
   - With grid OFF, press number keys 1-4
   - Should switch between individual worlds

3. **Test with different world counts:**
   - `--num-worlds 1` (single world)
   - `--num-worlds 8` (2x4 grid)
   - `--num-worlds 16` (4x4 grid)

## Files to Modify

1. `external/madrona/src/render/shaders/shader_common.h` - Update CullPushConst structure
2. `external/madrona/src/render/shaders/viewer_cull.hlsl` - Update shader logic
3. `external/madrona/src/viz/viewer_renderer.cpp` - Fix hardcoded bug, update CPU logic
4. `src/viewer.cpp` - (Optional) Add world selection controls
5. `external/madrona/include/madrona/viz/viewer.hpp` - (Optional) Add setWorldIndex method
6. `external/madrona/src/viz/viewer.cpp` - (Optional) Implement setWorldIndex

## Build and Test

After making changes:
```bash
./build.sh
./build/viewer --num-worlds 4
```

Test the 'M' key to toggle grid mode and verify:
- Grid ON: All worlds visible in grid layout
- Grid OFF: Only selected world visible at origin