# Plan: Add X/Y/Z Scaling Parameters and done_on_collide Flag to CompiledLevel

## Pre-Reading: Files to Review

Before starting implementation, familiarize yourself with these files:

### Core Structure Definitions
1. **src/types.hpp** (lines 144-195) - C++ CompiledLevel struct definition
2. **include/madrona_escape_room_c_api.h** (lines 79-126) - C API MER_CompiledLevel struct
3. **madrona_escape_room/ctypes_bindings.py** (lines 90-130) - Python MER_CompiledLevel class

### Conversion Functions
4. **madrona_escape_room/ctypes_bindings.py** 
   - `dict_to_compiled_level()` function (lines 180-250)
   - `compiled_level_to_dict()` function (lines 252-290)

### Binary Compatibility Verification
5. **src/madrona_escape_room_c_api.cpp** (lines 70-90) - Static assert checks for struct compatibility

### Default Level Generation
6. **src/default_level.cpp** (lines 15-50) - Default level initialization

### C++ Test Files
7. **tests/cpp/unit/test_file_inspector.cpp** - Uses sizeof(CompiledLevel)
8. **tests/cpp/unit/test_recording_utilities.cpp** - Creates CompiledLevel instances
9. **tests/cpp/fixtures/viewer_test_base.hpp** - May contain CompareLevels() function

### Python Test Files
10. **tests/python/test_c_api_struct_validation.py** - Contains 3 MER_CompiledLevel definitions
11. **tests/python/test_level_compiler_c_api.py** - Binary I/O roundtrip tests

## Overview
We will add four new fields to the CompiledLevel structure:
- `float x_scale` - Global X-axis scaling factor for the entire level
- `float y_scale` - Global Y-axis scaling factor for the entire level  
- `float z_scale` - Global Z-axis scaling factor for the entire level
- `bool done_on_collide` - Flag to indicate if episode should end on collision

These fields will be added after the existing `scale` field to maintain logical grouping.

## Todo List
- [ ] Step 1: Add fields to C++ CompiledLevel struct in src/types.hpp
- [ ] Step 2: Add fields to C API MER_CompiledLevel struct in include/madrona_escape_room_c_api.h
- [ ] Step 3A: Add fields to Python MER_CompiledLevel class in ctypes_bindings.py
- [ ] Step 3B: Update dict_to_compiled_level() in ctypes_bindings.py
- [ ] Step 3C: Update compiled_level_to_dict() in ctypes_bindings.py
- [ ] Step 4: Add static_asserts for new fields in madrona_escape_room_c_api.cpp
- [ ] Step 5: Initialize fields in default_level.cpp
- [ ] Step 6: Update C++ test files that create CompiledLevel
- [ ] Step 7: Update ALL MER_CompiledLevel definitions in test_c_api_struct_validation.py
- [ ] Step 8: Verify binary I/O test in test_level_compiler_c_api.py passes
- [ ] Step 9: Build project and verify static_asserts pass
- [ ] Step 10: Run all tests to verify functionality

## Implementation Steps

### Step 1: Add Fields to C++ CompiledLevel struct (src/types.hpp)
- Add after line 154 (after `float scale;`):
  ```cpp
  float x_scale;           // Global X-axis scale factor
  float y_scale;           // Global Y-axis scale factor  
  float z_scale;           // Global Z-axis scale factor
  bool done_on_collide;    // Episode ends on collision
  ```

### Step 2: Add Fields to C API MER_CompiledLevel (include/madrona_escape_room_c_api.h)
- Add after line 85 (after `float scale;`):
  ```c
  float x_scale;           // Global X-axis scale factor
  float y_scale;           // Global Y-axis scale factor
  float z_scale;           // Global Z-axis scale factor
  bool done_on_collide;    // Episode ends on collision
  ```

### Step 3: Update Python Bindings (madrona_escape_room/ctypes_bindings.py)

#### 3A: Add to MER_CompiledLevel class definition
- Add after `("scale", c_float),`:
  ```python
  ("x_scale", c_float),
  ("y_scale", c_float),
  ("z_scale", c_float),
  ("done_on_collide", c_bool),
  ```

#### 3B: Update dict_to_compiled_level() function
- Add after setting `level.scale`:
  ```python
  level.x_scale = compiled_dict.get("x_scale", 1.0)
  level.y_scale = compiled_dict.get("y_scale", 1.0)
  level.z_scale = compiled_dict.get("z_scale", 1.0)
  level.done_on_collide = compiled_dict.get("done_on_collide", False)
  ```

#### 3C: Update compiled_level_to_dict() function
- Add to returned dictionary:
  ```python
  "x_scale": level.x_scale,
  "y_scale": level.y_scale,
  "z_scale": level.z_scale,
  "done_on_collide": level.done_on_collide,
  ```

### Step 4: Add Static Asserts (src/madrona_escape_room_c_api.cpp)
**⚠️ CRITICAL: Verify binary compatibility! ⚠️**

Add after existing static_asserts (around line 86):
```cpp
static_assert(offsetof(MER_CompiledLevel, x_scale) == offsetof(CompiledLevel, x_scale),
              "x_scale offset mismatch");
static_assert(offsetof(MER_CompiledLevel, y_scale) == offsetof(CompiledLevel, y_scale),
              "y_scale offset mismatch");
static_assert(offsetof(MER_CompiledLevel, z_scale) == offsetof(CompiledLevel, z_scale),
              "z_scale offset mismatch");
static_assert(offsetof(MER_CompiledLevel, done_on_collide) == offsetof(CompiledLevel, done_on_collide),
              "done_on_collide offset mismatch");
```

**Important:** The C API uses direct struct assignment via `reinterpret_cast`:
```cpp
CompiledLevel cpp_level = *reinterpret_cast<const CompiledLevel*>(c_level);
```
This means the new fields are automatically copied as long as they exist in both structs at the same offset.

### Step 5: Initialize Fields in Default Level (src/default_level.cpp)
**REQUIRED** - Add after line 19 (after `level.scale = 1.0f;`):
```cpp
level.x_scale = 1.0f;
level.y_scale = 1.0f;
level.z_scale = 1.0f;
level.done_on_collide = false;
```

### Step 6: Update C++ Test Files

#### Files that need updating:
1. **test_file_inspector.cpp** - Uses `sizeof(CompiledLevel)`
2. **test_recording_utilities.cpp** - Creates CompiledLevel instances
3. **viewer_test_base.hpp** - May have CompareLevels() function

Initialize new fields in all CompiledLevel instances:
```cpp
level.x_scale = 1.0f;
level.y_scale = 1.0f;
level.z_scale = 1.0f;
level.done_on_collide = false;
```

### Step 7: Update Python Test Structs (test_c_api_struct_validation.py)

**IMPORTANT**: There are 3 MER_CompiledLevel class definitions in this file!

Update ALL occurrences - add after `("scale", ctypes.c_float),`:
```python
("x_scale", ctypes.c_float),
("y_scale", ctypes.c_float),
("z_scale", ctypes.c_float),
("done_on_collide", ctypes.c_bool),
```

For initialization sections:
```python
level_struct.x_scale = compiled.get("x_scale", 1.0)
level_struct.y_scale = compiled.get("y_scale", 1.0)
level_struct.z_scale = compiled.get("z_scale", 1.0)
level_struct.done_on_collide = compiled.get("done_on_collide", False)
```

### Step 8: Verify Binary I/O Test
The existing test in **test_level_compiler_c_api.py** automatically verifies ALL fields:
```bash
uv run --group dev pytest tests/python/test_level_compiler_c_api.py::TestBinaryIO -v
```
No changes needed - test should pass if implementation is correct.

### Step 9: Build and Verify
```bash
# Build using project-builder agent
# Static asserts will fail at compile time if structs don't match
```

If static_assert fails, check:
1. Field order matches exactly between structs
2. Field types and sizes match
3. No alignment/padding issues

### Step 10: Run Tests
```bash
# Run C++ tests
./build/mad_escape_tests

# Run Python tests
uv run --group dev pytest tests/python/test_c_api_struct_validation.py -v
uv run --group dev pytest tests/python/test_level_compiler_c_api.py -v
```

## Final Validation Checklist

- [ ] **Step 1**: Fields added to C++ `CompiledLevel` struct
- [ ] **Step 2**: Fields added to C API `MER_CompiledLevel` struct with identical type/size
- [ ] **Step 3**: Python bindings fully updated (class definition, dict_to_compiled_level, compiled_level_to_dict)
- [ ] **Step 4**: Static asserts added and passing (build succeeds)
- [ ] **Step 5**: Default level initializes the fields
- [ ] **Step 6**: C++ tests initialize the fields where CompiledLevel is created
- [ ] **Step 7**: ALL MER_CompiledLevel structs in test files updated (3 copies)
- [ ] **Step 8**: Binary I/O test passes (automatic verification)
- [ ] **Step 9**: Project builds without errors
- [ ] **Step 10**: All tests pass (C++ and Python)

## Common Issues and Solutions

1. **Static assert fails**: Fields not in same order or different types/sizes
2. **Python tests fail**: Missed updating one of the MER_CompiledLevel definitions in test files
3. **Binary I/O fails**: Field serialization order doesn't match between save/load
4. **Segfaults**: Array size mismatch or alignment issues
5. **Recording playback fails**: Struct size changed, old recordings incompatible

## Notes on Binary Compatibility
- The fields are placed after `scale` to maintain logical grouping
- Default values: x_scale=1.0, y_scale=1.0, z_scale=1.0, done_on_collide=false
- Binary formats that change: `.lvl` files and `.rec` files (embedded CompiledLevel)
- Old files cannot be read after adding fields due to size change

## Accessing the New Fields in System Calls

Once implemented, the new fields can be accessed from any system call:

```cpp
// In system functions (sim.cpp)
inline void mySystem(Engine &ctx, /* other components */) {
    const CompiledLevel& level = ctx.singleton<CompiledLevel>();
    
    // Access the new scaling fields
    float x_factor = level.x_scale;
    float y_factor = level.y_scale;
    float z_factor = level.z_scale;
    
    // Check collision flag
    if (level.done_on_collide) {
        // Handle collision-based episode termination
    }
}
```