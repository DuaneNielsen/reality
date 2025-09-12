# Multi-World Grid Layout - Current State Analysis

## Overview

The multi-world grid layout system allows displaying multiple simulation worlds simultaneously in a spatial grid arrangement. Currently works well for 4 worlds but has issues with larger world counts.

## Current Implementation

### Location: `external/madrona/src/render/shaders/viewer_draw.hlsl:187-209`

The grid positioning is handled by the `getWorldGridOffset()` function:

```hlsl
float3 getWorldGridOffset(uint worldId) {
    if (push_const.multiWorldGrid == 0) {
        return float3(0, 0, 0);  // Single world mode
    }
    
    // PROBLEM: Hardcoded positions for worlds 1-3
    if (worldId == 1) return float3(50, 0, 0);    // World 1: Right
    if (worldId == 2) return float3(0, 50, 0);    // World 2: Forward  
    if (worldId == 3) return float3(50, 50, 0);   // World 3: Diagonal
    
    // Algorithmic positioning for world 4+
    uint row = worldId / push_const.gridCols;     // gridCols = 8
    uint col = worldId % push_const.gridCols;
    
    float cell_width = push_const.worldScaleX + push_const.worldSpacing;
    float cell_height = push_const.worldScaleY + push_const.worldSpacing;
    
    return float3(col * cell_width, row * cell_height, 0);
}
```

## Configuration Parameters

- **Grid Columns**: `gridCols = 8` (default)
- **World Spacing**: `worldSpacing = 1.0f` (configurable)
- **Grid Cols**: `grid_cols = 8` (in ViewerCore config)

## Current Behavior

### ✅ Works Well (4 worlds)
- **World 0**: (0, 0) - Origin
- **World 1**: (50, 0) - Hardcoded right
- **World 2**: (0, 50) - Hardcoded forward
- **World 3**: (50, 50) - Hardcoded diagonal

Creates a nice compact 2×2 grid layout.

### ❌ Problematic (8+ worlds)
- **Worlds 0-3**: Use hardcoded positions (compact 2×2 square)
- **Worlds 4-7**: Use algorithmic positions with gridCols=8
  - World 4: (4×cell_width, 0) - Far right in row 0
  - World 5: (5×cell_width, 0) - Even further right
  - World 6: (6×cell_width, 0) - Row 0, col 6
  - World 7: (7×cell_width, 0) - Row 0, col 7

**Result**: Worlds 4+ appear as a disconnected horizontal line far from worlds 0-3.

## Root Causes

1. **Inconsistent Positioning Logic**: Mix of hardcoded vs algorithmic positioning
2. **Wrong Grid Dimensions**: `gridCols=8` creates sparse layouts for small world counts
3. **Algorithm Doesn't Account for World 0**: World 0 always at origin, disrupting grid pattern
4. **Poor Default Parameters**: Large grid columns and cell sizes spread worlds too far apart

## Current Status

- **Multi-world culling shader**: ✅ Implemented and working
- **Camera controls**: ✅ Fixed and committed  
- **M key toggle**: ✅ Working for multi-world mode activation
- **Grid layout algorithm**: ❌ Needs complete rewrite

## Next Steps

1. Remove hardcoded positions for worlds 1-3
2. Implement consistent algorithmic grid positioning for all worlds
3. Calculate appropriate grid dimensions based on world count
4. Adjust default spacing and cell sizes for better visual arrangement
5. Test with various world counts (4, 6, 8, 9, 16, etc.)

## Files Involved

- `external/madrona/src/render/shaders/viewer_draw.hlsl` - Grid offset calculation
- `external/madrona/src/viz/viewer_common.hpp` - Grid parameters
- `src/viewer.cpp` - Grid configuration and M key handling
- `src/viewer_core.hpp` - ViewerCore config parameters

---
*Status as of: December 2024*
*Last updated by: Claude Code*