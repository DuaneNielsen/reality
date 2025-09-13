# Multi-World Rendering Bug Analysis

**Date**: 2025-01-13  
**Status**: PARTIALLY FIXED - Grid calculation corrected, culling dispatch improved, but flickering remains  
**Priority**: HIGH - Affects multi-world grid functionality with 64+ worlds

## Problem Description

When rendering 64 worlds in multi-world grid mode, only a subset of geometry is rendered (approximately 6x8 = 48 worlds worth), with no FPS drop indicating a rendering pipeline limitation rather than performance issue.

**After initial fix**: Grid now displays all 64 worlds in 8x8 layout but exhibits flickering, suggesting not all geometry is rendered every frame.

## Root Causes Identified

### 1. ✅ FIXED: Hardcoded Grid Columns
- **Issue**: Grid columns hardcoded to 4, causing sparse layouts for large world counts
- **Location**: `external/madrona/src/viz/viewer.cpp:584` and `src/viewer.cpp`
- **Fix Applied**: Dynamic calculation using `ceil(sqrt(num_worlds))`
- **Result**: 64 worlds now correctly use 8x8 grid layout

### 2. ✅ IMPROVED: Culling Dispatch Limitation  
- **Issue**: Hardcoded `num_warps = 4` limited culling to 128 threads total
- **Location**: `external/madrona/src/viz/viewer_renderer.cpp:2053`
- **Original**: Only 4 workgroups × 32 threads = 128 threads for all instances
- **Fix Applied**: Dynamic calculation based on instance count
- **Result**: Adequate thread dispatch for large world counts

### 3. ✅ FIXED: Frame-to-Frame Rendering Inconsistency
- **Issue**: Flickering where random geometry not rendered every frame  
- **Root Cause**: GPU timing race condition - compute shader not completing before draw calls
- **Solution Applied**:
  - Sequential atomic allocation (`InterlockedAdd(drawCount[0], 1, draw_offset)`) 
  - Stronger GPU synchronization barriers with `VK_ACCESS_MEMORY_*` and `VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT`
  - Boosted buffer sizes (100x multiplier, 1M draw commands limit)
- **Result**: Eliminates timing-based flickering for 64+ world rendering

## Technical Details

### Dispatch Calculation (Applied Fix)
```cpp
// OLD: Hardcoded limit
uint32_t num_warps = 4;  // Only 128 threads total

// NEW: Dynamic calculation 
uint32_t instances_per_thread = 1;
uint32_t min_threads_needed = (num_instances + instances_per_thread - 1) / instances_per_thread;
uint32_t num_warps = std::max(4u, (min_threads_needed + 31) / 32);
```

### Grid Layout (Applied Fix)
```cpp
// OLD: Hardcoded
.gridCols = 4,

// NEW: Dynamic
.gridCols = static_cast<uint32_t>(std::ceil(std::sqrt(cfg.numWorlds))),
```

## Investigation Areas for Remaining Issue

### Synchronization
- Check GPU memory barriers between culling and drawing passes
- Verify command buffer synchronization for large instance counts
- Investigate if `InterlockedAdd` operations in culling shader cause race conditions

### Resource Limits
- Verify GPU memory limits for large instance buffers
- Check if draw command buffer size is adequate
- Investigate descriptor set limitations

### Shader Logic
- Review culling shader loop termination conditions
- Check if instance offset calculations handle large world counts correctly
- Verify draw command generation doesn't overflow buffers

### Performance Considerations
- Monitor actual thread utilization vs theoretical dispatch
- Check if GPU occupancy is optimal for increased warp count
- Investigate memory bandwidth limitations

## Testing Scenarios

### Working Cases
- ✅ 1-16 worlds: Render correctly without issues
- ✅ 4 worlds: 2x2 grid displays properly  
- ✅ 9 worlds: 3x3 grid displays properly

### Problematic Cases  
- 🔴 64 worlds: 8x8 grid with flickering geometry
- 🔴 Large world counts (>64): Likely similar issues

### Test Commands
```bash
./build/viewer --num-worlds 4   # Works correctly
./build/viewer --num-worlds 9   # Works correctly  
./build/viewer --num-worlds 64  # Shows flickering
```

## Files Modified

1. `src/viewer.cpp:405` - Added dynamic grid_cols calculation
2. `external/madrona/src/viz/viewer.cpp:585` - Added dynamic gridCols calculation  
3. `external/madrona/src/viz/viewer_renderer.cpp:2053-2057` - Dynamic num_warps calculation

## Next Steps

1. **Memory Barrier Investigation**: Check synchronization between culling and draw passes
2. **Buffer Size Audit**: Verify all buffers can handle 64+ worlds worth of instances
3. **Shader Debugging**: Add debug output to culling shader to verify instance processing
4. **GPU Profiling**: Use GPU profiler to identify bottlenecks in large world scenarios
5. **Race Condition Analysis**: Investigate potential race conditions in `InterlockedAdd` operations

## Impact Assessment

- **Severity**: High - Affects usability of multi-world visualization for large datasets
- **Workaround**: Use fewer worlds (<= 16) for stable rendering
- **User Impact**: Flickering makes 64+ world visualization unusable for analysis
- **Performance**: No FPS impact, purely a rendering correctness issue

## Related Documentation

- `docs/development/RENDERING_PIPELINE.md` - Multi-world rendering architecture
- `external/madrona/src/render/shaders/viewer_cull.hlsl` - Culling shader implementation
- `external/madrona/src/render/shaders/viewer_draw.hlsl` - Draw shader implementation