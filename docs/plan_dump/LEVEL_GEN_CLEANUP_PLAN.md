# Level Generation Cleanup Plan

## Overview
This plan outlines the removal of redundant entities and code from the level generation system after the removal of doors, buttons, and multi-agent support.

## Current State Analysis

### Entities Still in Use
- **Agent** (1 only)
- **Walls** (outer boundaries + room separators)
- **Floor** (static plane)
- **Cubes** (obstacles and movable blocks)

### Redundant Code Identified
- **Door logic** in `makeEndWall()` - creates gaps for non-existent doors
- **Button-related room types** - SingleButton, DoubleButton, CubeButtons
- **Multi-agent spawning logic** - spreading spawn positions for 2 agents
- **Door constants** - `doorWidth` and related calculations
- **Duplicate room functions** - `makeEmptyRoom` and `makeEmptyRoomVariant` are identical

## Implementation Steps

### Step 1: Remove Door-Related Constants
**File**: `src/level_gen.cpp`
- Remove `inline constexpr float doorWidth = consts::worldWidth / 3.f;`
- Remove any door-related position calculations

### Step 2: Simplify makeEndWall Function
**File**: `src/level_gen.cpp`
- Current: Creates two wall segments with a gap for a door
- Change to: Create a single solid wall spanning the entire width
- Remove door positioning logic and random placement

### Step 3: Consolidate Room Types
**File**: `src/level_gen.cpp`
- Update RoomType enum:
  ```cpp
  enum class RoomType : uint32_t {
      Empty,        // No obstacles
      CubeObstacle, // Fixed cube obstacles
      CubeMovable,  // Movable cubes
      NumTypes,
  };
  ```

### Step 4: Remove Redundant Room Functions
**File**: `src/level_gen.cpp`
- Delete `makeEmptyRoomVariant()` (duplicate of `makeEmptyRoom()`)
- Rename remaining functions to match new room types:
  - `makeEmptyRoom()` → keep as is
  - `makeCubeObstacleRoom()` → keep as is
  - `makeCubeRoom()` → rename to `makeCubeMovableRoom()`

### Step 5: Update Entity Count Calculations
**File**: `src/sim.cpp`
- Current calculation: `consts::numRooms * (consts::maxEntitiesPerRoom + 3) + 4`
  - The +3 was for: 2 wall segments + 1 door per room
- New calculation: `consts::numRooms * (consts::maxEntitiesPerRoom + 1) + 4`
  - The +1 is for: 1 solid wall per room

### Step 6: Simplify Agent Spawning
**File**: `src/level_gen.cpp` in `resetPersistentEntities()`
- Remove the alternating left/right spawn logic:
  ```cpp
  // Remove this:
  if (i % 2 == 0) {
      pos.x += consts::worldWidth / 4.f;
  } else {
      pos.x -= consts::worldWidth / 4.f;
  }
  ```
- Since we have only 1 agent, spawn at center

### Step 7: Update Room Structure
**File**: `src/types.hpp`
- Check Room struct for any door-related fields
- Update walls array if needed (from 2 segments to 1 wall)

### Step 8: Update makeRoom Function
**File**: `src/level_gen.cpp`
- Update switch statement to use new room types
- Remove references to button-related room types

### Step 9: Clean Up Level Generation
**File**: `src/level_gen.cpp` in `generateLevel()`
- Update the fixed level sequence to use new room types
- Or enable the random room generation with new types

## Expected Benefits
1. **Reduced memory usage** - fewer entities allocated per room
2. **Simpler code** - removed ~100 lines of redundant code
3. **Clearer intent** - room types now match actual functionality
4. **Easier maintenance** - less dead code to confuse future developers

## Testing Requirements
After implementation:
1. Build the project: `cd build && make -j8`
2. Run viewer to verify level generation: `./build/viewer`
3. Run training to ensure compatibility: `uv run python scripts/train.py --num-worlds 64 --num-updates 10`
4. Check that physics and collision still work correctly

## Potential Issues
- Ensure wall collision boxes properly block the full width
- Verify agent can still navigate through the simplified rooms
- Check that observation systems handle the reduced entity count