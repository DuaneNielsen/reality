# Plan: Add World Boundary Fields to CompiledLevel Structure

## Problem Statement
The reward normalization system currently uses a hardcoded `worldLength` constant (40.0f) to normalize agent progress rewards. This causes incorrect reward calculations when using custom levels with different dimensions. For example, a 32x17 level with scale=2.5 has actual world dimensions that don't match the hardcoded constant, leading to rewards > 1.0.

## Solution
Add world boundary fields to the `CompiledLevel` structure to store the actual world dimensions in world units. These boundaries will be:
- Minimum and maximum X coordinates (world units)
- Minimum and maximum Y coordinates (world units)
- Minimum and maximum Z coordinates (world units)

These values will be calculated during level compilation based on the grid dimensions and scale factor.

## New Fields to Add
```cpp
// World boundaries in world units (calculated from grid dimensions * scale)
float world_min_x;  // Minimum X boundary in world units
float world_max_x;  // Maximum X boundary in world units  
float world_min_y;  // Minimum Y boundary in world units
float world_max_y;  // Maximum Y boundary in world units
float world_min_z;  // Minimum Z boundary in world units
float world_max_z;  // Maximum Z boundary in world units
```

## Files to Modify

### 1. Core Structure Files

#### src/types.hpp
Add the boundary fields to the `CompiledLevel` struct after the `level_name` field:
```cpp
struct CompiledLevel {
    // ... existing header fields ...
    char level_name[MAX_LEVEL_NAME_LENGTH];
    
    // World boundaries in world units
    float world_min_x;
    float world_max_x;
    float world_min_y;
    float world_max_y;
    float world_min_z;
    float world_max_z;
    
    // ... existing spawn data ...
```

#### include/madrona_escape_room_c_api.h
Add matching fields to `MER_CompiledLevel` in the same position:
```c
typedef struct {
    // ... existing header fields ...
    char level_name[MER_MAX_NAME_LENGTH];
    
    // World boundaries in world units
    float world_min_x;
    float world_max_x;
    float world_min_y;
    float world_max_y;
    float world_min_z;
    float world_max_z;
    
    // ... existing spawn data ...
```

#### madrona_escape_room/ctypes_bindings.py
Add fields to the `MER_CompiledLevel` class:
```python
class MER_CompiledLevel(Structure):
    _fields_ = [
        # ... existing header fields ...
        ("level_name", c_char * MAX_NAME_LENGTH),
        
        # World boundaries
        ("world_min_x", c_float),
        ("world_max_x", c_float),
        ("world_min_y", c_float),
        ("world_max_y", c_float),
        ("world_min_z", c_float),
        ("world_max_z", c_float),
        
        # ... existing spawn data ...
```

### 2. Level Compilation Files

#### madrona_escape_room/level_compiler.py

Update `compile_level()` to calculate world boundaries:
```python
def compile_level(...):
    # ... existing code ...
    
    # Calculate world boundaries based on grid dimensions and scale
    # Grid coordinates are centered at origin
    half_width = width / 2.0
    half_height = height / 2.0
    
    world_min_x = -half_width * scale
    world_max_x = half_width * scale
    world_min_y = -half_height * scale  # Y is inverted in ASCII
    world_max_y = half_height * scale
    world_min_z = 0.0  # Floor level
    world_max_z = 10.0 * scale  # Reasonable max height
    
    # Add to compiled dict
    compiled = {
        # ... existing fields ...
        "world_min_x": world_min_x,
        "world_max_x": world_max_x,
        "world_min_y": world_min_y,
        "world_max_y": world_max_y,
        "world_min_z": world_min_z,
        "world_max_z": world_max_z,
        # ... rest of fields ...
    }
```

Update binary I/O functions:

In `save_compiled_level_binary()`:
```python
# After level_name, before num_spawns
f.write(struct.pack("<f", compiled["world_min_x"]))
f.write(struct.pack("<f", compiled["world_max_x"]))
f.write(struct.pack("<f", compiled["world_min_y"]))
f.write(struct.pack("<f", compiled["world_max_y"]))
f.write(struct.pack("<f", compiled["world_min_z"]))
f.write(struct.pack("<f", compiled["world_max_z"]))
```

In `load_compiled_level_binary()`:
```python
# After level_name, before num_spawns
world_min_x = struct.unpack("<f", f.read(4))[0]
world_max_x = struct.unpack("<f", f.read(4))[0]
world_min_y = struct.unpack("<f", f.read(4))[0]
world_max_y = struct.unpack("<f", f.read(4))[0]
world_min_z = struct.unpack("<f", f.read(4))[0]
world_max_z = struct.unpack("<f", f.read(4))[0]

compiled["world_min_x"] = world_min_x
# ... etc for all boundary fields
```

### 3. Default Level

#### src/default_level.cpp
Initialize the boundary fields for the default 40x40 level:
```cpp
// After level_name initialization
compiledLevel.world_min_x = -50.0f;  // (40/2) * 2.5
compiledLevel.world_max_x = 50.0f;
compiledLevel.world_min_y = -50.0f;
compiledLevel.world_max_y = 50.0f;
compiledLevel.world_min_z = 0.0f;
compiledLevel.world_max_z = 25.0f;  // 10 * 2.5
```

### 4. Simulation Usage

#### src/sim.cpp
Update the reward calculation to use actual world boundaries:
```cpp
inline void rewardSystem(...) {
    // Get world boundaries from the compiled level
    Context ctx = ctx.data();
    const CompiledLevel* level = &ctx.compiledLevel;
    
    // Update max Y reached
    if (pos.y > progress.maxY) {
        progress.maxY = pos.y;
    }
    
    // Only give reward at episode end
    if (done.v == 1 || steps_remaining.t == 0) {
        // Use actual world boundaries for normalization
        float world_length = level->world_max_y - level->world_min_y;
        float adjusted_progress = progress.maxY - level->world_min_y;
        float normalized_progress = adjusted_progress / world_length;
        
        // Clamp to [0, 1] range
        normalized_progress = fmaxf(0.0f, fminf(1.0f, normalized_progress));
        
        out_reward.v = normalized_progress;
    } else {
        out_reward.v = 0.0f;
    }
}
```

### 5. Test Files to Update

#### C++ Tests
- **tests/cpp/unit/test_levels.hpp** - Initialize boundaries in `CreateTestLevel()`
- **tests/cpp/unit/viewer_test_base.hpp** - Add boundary comparison in `CompareLevels()`
- **tests/cpp/unit/test_persistence.cpp** - Initialize boundaries
- **tests/cpp/unit/test_recording_utilities.cpp** - Initialize boundaries
- **tests/cpp/unit/test_asset_refactor.cpp** - Initialize boundaries

#### Python Tests  
- **tests/python/test_c_api_struct_validation.py** - Update struct size validation
- **tests/python/test_level_compiler_c_api.py** - Test boundary calculation and I/O

## Validation Steps

1. **Compile and Build**: Ensure project builds with new fields
2. **Test Reward Normalization**: Run `test_reward_normalization` - should now pass
3. **Test Binary I/O**: Verify .lvl files save/load correctly with boundaries
4. **Test Recording**: Verify .rec files work with new CompiledLevel size
5. **Test Different Levels**: Test with various level sizes to verify boundary calculations

## Benefits

1. **Correct Reward Normalization**: Rewards will be properly normalized to [0, 1] range for any level size
2. **Future Extensibility**: Boundaries can be used for:
   - Out-of-bounds detection
   - Camera frustum culling
   - Spatial queries
   - Level-aware AI behavior
3. **Data-Driven Design**: Removes hardcoded assumptions about world size

## Implementation Order

1. Add fields to all three structure definitions (C++, C API, Python)
2. Update level compiler to calculate boundaries
3. Update binary I/O functions
4. Update default level
5. Fix reward calculation in sim.cpp
6. Update tests
7. Verify all tests pass