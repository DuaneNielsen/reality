# Multi-World Rendering Flickering - Status Report

**Date**: 2025-01-13  
**Status**: UNRESOLVED - Multiple fix attempts unsuccessful  
**Priority**: HIGH - Critical rendering issue affecting 64+ world visualization

## Problem Summary

When rendering 64 worlds in multi-world grid mode, random elements are not being rendered each frame, causing visible flickering. The 8x8 grid layout displays correctly, but individual geometry objects appear and disappear randomly between frames, making the visualization unusable for analysis.

## Symptoms

- ✅ Correct 8x8 grid layout (64 worlds arranged properly)
- ✅ No performance impact (FPS remains stable)
- 🔴 Random geometry flickering - objects appear/disappear between frames
- 🔴 Affects large world counts (64+), smaller counts (≤16) work fine

## Root Cause Investigation

Initially suspected **race conditions in InterlockedAdd operations** in the culling shader, where multiple threads compete for the same draw counter, potentially causing buffer overwrites and missing geometry.

## Fix Attempts (All Unsuccessful)

### Attempt 1: Two-Pass Culling System
**Approach**: Eliminated race conditions by using group-shared memory and single atomic operation per workgroup
- Added `threadMeshCounts[32]` array to count meshes per thread
- Used thread 0 to perform single `InterlockedAdd` per workgroup
- Pre-calculated draw offsets to avoid atomic contention

**Files Modified**:
- `external/madrona/src/render/shaders/viewer_cull.hlsl`

**Result**: ❌ No improvement in flickering behavior

### Attempt 2: Sequential Atomic Allocation
**Approach**: Simplified to one atomic operation per mesh instead of batching
- Changed from `InterlockedAdd(drawCount[0], obj.numMeshes, draw_offset)` 
- To `InterlockedAdd(drawCount[0], 1, draw_offset)` per mesh
- Added buffer overflow protection with `MAX_DRAW_COMMANDS` check

**Files Modified**:
- `external/madrona/src/render/shaders/viewer_cull.hlsl` (lines 112-119)

**Result**: ❌ No improvement in flickering behavior

### Attempt 3: Buffer Size Increases
**Approach**: Increased buffer limits to handle 64+ worlds
- Boosted `MAX_DRAW_COMMANDS` from 65,536 to 1,048,576
- Increased viewer renderer buffer multipliers from 10x to 100x
- Modified batch renderer `maxDrawsPerView` from 2,048 to 16,384

**Files Modified**:
- `external/madrona/src/render/shaders/shader_common.h` (line 5)
- `external/madrona/src/viz/viewer_renderer.cpp` (lines 1532-1533)
- `external/madrona/src/render/batch_renderer.cpp` (line 35)

**Result**: ❌ No improvement in flickering behavior

### Attempt 4: Stronger GPU Synchronization
**Approach**: Enhanced memory barriers and pipeline synchronization
- Added `VK_ACCESS_MEMORY_WRITE_BIT` and `VK_ACCESS_MEMORY_READ_BIT`
- Expanded to `VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT` 
- Added `VK_DEPENDENCY_BY_REGION_BIT` for region-based dependencies

**Files Modified**:
- `external/madrona/src/viz/viewer_renderer.cpp` (lines 2087-2104)

**Result**: ❌ No improvement in flickering behavior

## Current Analysis

The race condition hypothesis appears to be **incorrect**. Multiple comprehensive fixes targeting atomic operations, buffer sizes, and GPU synchronization have all failed to resolve the issue.

### Alternative Theories

1. **Indirect Draw Command Issues**
   - The `vkCmdDrawIndexedIndirect` call may be reading inconsistent draw counts
   - Draw command buffer may be getting corrupted despite proper atomic allocation

2. **View Frustum Culling Problems**
   - Culling logic may be incorrectly rejecting geometry for certain world positions
   - Grid positioning calculations may cause some worlds to be outside view frustum

3. **GPU Driver/Hardware Issues**
   - Specific GPU driver behavior with large draw call counts
   - Memory coherency issues on certain hardware configurations

4. **Vulkan Command Buffer Ordering**
   - Command buffer recording order may cause timing-dependent behavior
   - Multiple command buffers may be interfering with each other

## Testing Scenarios

### Working Cases
- ✅ 4 worlds: Perfect 2x2 grid rendering
- ✅ 9 worlds: Perfect 3x3 grid rendering  
- ✅ 16 worlds: Stable 4x4 grid rendering

### Problematic Cases
- 🔴 64 worlds: 8x8 grid with severe flickering
- 🔴 Assumed: Other large counts (32+, 100+) likely similar issues

## Next Investigation Steps

### 1. Indirect Draw Analysis
- Log actual draw count values being read by indirect draw commands
- Verify draw command buffer contents after culling pass
- Test with direct draw calls instead of indirect to isolate issue

### 2. Culling Logic Verification  
- Add debug output to culling shader to verify which instances are being processed
- Check if world positioning calculations are correct for all 64 worlds
- Verify instance offset calculations for large world counts

### 3. Driver/Hardware Investigation
- Test on different GPU hardware (NVIDIA vs AMD)
- Try different Vulkan driver versions
- Enable Vulkan validation layers for detailed error checking

### 4. Alternative Rendering Approaches
- Implement direct draw path bypassing compute culling entirely
- Try splitting 64 worlds across multiple smaller draw calls
- Investigate if issue persists with static geometry (no culling)

## Impact Assessment

- **Severity**: Critical - Makes 64+ world visualization completely unusable
- **Workaround**: Limited to ≤16 worlds for stable rendering
- **User Impact**: Cannot analyze large-scale multi-agent scenarios effectively
- **Performance**: No FPS impact suggests this is a correctness, not performance issue

## Conclusion

After 4 comprehensive fix attempts targeting suspected race conditions, buffer limitations, and synchronization issues, the flickering persists. The root cause is likely **not** related to atomic operations in the culling shader. Further investigation should focus on indirect draw mechanics, culling logic verification, and potential driver-level issues.