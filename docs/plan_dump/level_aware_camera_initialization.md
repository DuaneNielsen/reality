# Level-Aware Camera Initialization Enhancement Plan

## Goal
Enhance free camera initialization to automatically position and orient based on level dimensions for optimal viewing.

## Current State
- Camera hardcoded to position `{agentPos.x, agentPos.y - 14.0f, agentPos.z + 35.0f}`
- No consideration of level bounds or aspect ratio
- Same positioning regardless of level size/shape

## Required Changes

### 1. Add Manager Method to Expose Level Bounds
**File**: `src/mgr.hpp` + `src/mgr.cpp`
```cpp
// Add to Manager class:
struct LevelBounds {
    float min_x, max_x, min_y, max_y, min_z, max_z;
    float width() const { return max_x - min_x; }
    float length() const { return max_y - min_y; }
    float height() const { return max_z - min_z; }
};

LevelBounds getLevelBounds(int32_t world_idx = 0) const;
```

### 2. Create Smart Camera Positioning Function
**File**: `src/viewer.cpp` (new helper function)
```cpp
struct CameraSetup {
    math::Vector3 position;
    math::Vector3 lookAt;
    float yaw_rotation; // Additional rotation if needed
};

CameraSetup calculateOptimalCameraPosition(const LevelBounds& bounds, const math::Vector3& agentPos) {
    // Logic:
    // 1. Determine longest axis (width vs length)
    // 2. If length > width: rotate camera 90° to align Y-axis with screen width
    // 3. Set height proportional to largest dimension for full view
    // 4. Center camera on level bounds, not just agent
}
```

### 3. Update Camera Initialization Logic
**File**: `src/viewer.cpp` (lines ~444-452)
Replace hardcoded positioning with:
```cpp
// Get level bounds for smart positioning
auto levelBounds = mgr.getLevelBounds(0);
auto cameraSetup = calculateOptimalCameraPosition(levelBounds, agentPos);

freeFlyCamera->setPosition(cameraSetup.position);
freeFlyCamera->setLookAt(cameraSetup.lookAt);
```

## Technical Details

### Optimal Camera Algorithm
1. **Determine level aspect ratio**: `width/length`
2. **Choose orientation**: 
   - If `length > width * 1.2`: Rotate 90° (Y-axis becomes horizontal)
   - Else: Standard orientation (X-axis horizontal)
3. **Calculate height**: `height = max(width, length) * 1.5` for full coverage
4. **Position**: Center of level bounds + calculated height offset

### Benefits
- Automatic adaptation to any level size/shape
- Optimal use of monitor aspect ratio (16:9/16:10)
- Consistent "full level view" regardless of level design
- Better initial user experience

### Files to Modify
1. `src/mgr.hpp` - Add `getLevelBounds()` declaration
2. `src/mgr.cpp` - Implement `getLevelBounds()` method
3. `src/viewer.cpp` - Add `calculateOptimalCameraPosition()` and update initialization

### Testing Strategy
- Test with wide levels (large X dimension)
- Test with long levels (large Y dimension) 
- Test with square levels
- Verify camera always shows full level bounds