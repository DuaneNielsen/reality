# Plan Template: Adding a New Field to CompiledLevel Structure

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
This plan template describes the steps needed to add a new field to the `CompiledLevel` structure in `src/types.hpp`. The `CompiledLevel` structure is a GPU-compatible data structure used to store level information, and changes to it must be synchronized across C++, C API, Python bindings, recording serialization, and file inspection tools.

## Key Architecture Points
1. **Binary Compatibility**: The C++ `CompiledLevel` and C API `MER_CompiledLevel` structs must have identical memory layout
2. **Direct Struct Copy**: The C API uses `reinterpret_cast` for direct binary copying - no manual field copying needed
3. **Static Asserts**: Compile-time checks ensure struct compatibility - new fields MUST have matching offsets
4. **Python Data Flow**: Python dict → ctypes struct → C API → direct cast to C++ struct

## Files That Need Modification

### Core Structure Files
1. **src/types.hpp** - Add field to C++ `CompiledLevel` struct
2. **include/madrona_escape_room_c_api.h** - Add field to C API `MER_CompiledLevel` struct  
3. **madrona_escape_room/ctypes_bindings.py** - Add field to Python `MER_CompiledLevel` class

### Binary I/O (Automatic)
- **No changes needed** - Binary I/O is handled automatically by C API
- The C API directly writes/reads the entire struct to/from disk
- Python's functions are just thin wrappers around C API calls

### Level Processing Files  
5. **src/default_level.cpp** - ALWAYS initialize field in default level

### Recording System Files (Automatic)
- **src/mgr.cpp** - No changes needed (writes entire struct)
- **src/replay_metadata.hpp** - No changes needed (reads entire struct)
- **src/file_inspector.cpp** - No changes needed (validates via sizeof)

## Implementation Steps

### Step 0: Create Todo List

Use your Todo tool to create the following task list:

- [ ] Step 1: Add field to C++ CompiledLevel struct in src/types.hpp
- [ ] Step 2: Add field to C API MER_CompiledLevel struct in include/madrona_escape_room_c_api.h
- [ ] Step 3A: Add field to Python MER_CompiledLevel class in ctypes_bindings.py
- [ ] Step 3B: Update dict_to_compiled_level() in ctypes_bindings.py
- [ ] Step 3C: Update compiled_level_to_dict() in ctypes_bindings.py
- [ ] Step 4: Add static_assert for new field in madrona_escape_room_c_api.cpp
- [ ] Step 5: Initialize field in default_level.cpp
- [ ] Step 6: Update C++ test files that create CompiledLevel
- [ ] Step 7: Update ALL MER_CompiledLevel definitions in test_c_api_struct_validation.py
- [ ] Step 8: Verify binary I/O test in test_level_compiler_c_api.py passes
- [ ] Step 9: Build project and verify static_asserts pass
- [ ] Step 10: Run all tests to verify functionality

### Step 1: Add Field to C++ Structure
In `src/types.hpp`, add the new field to the `CompiledLevel` struct:
```cpp
struct CompiledLevel {
    // ... existing fields ...
    
    // Example new field
    int32_t new_field_name[MAX_TILES];  // Description
};
```

### Step 2: Add Field to C API Structure  
In `include/madrona_escape_room_c_api.h`, add matching field to `MER_CompiledLevel`:
```c
typedef struct {
    // ... existing fields ...
    
    int32_t new_field_name[MER_MAX_TILES];  // Must match C++ exactly
} MER_CompiledLevel;
```

### Step 3: Update Python Bindings
In `madrona_escape_room/ctypes_bindings.py`:

#### A. Add field to `MER_CompiledLevel._fields_`:
```python
class MER_CompiledLevel(Structure):
    _fields_ = [
        # ... existing fields ...
        ("new_field_name", c_int32 * MAX_TILES),  # Must match C++ type exactly
    ]
```

#### B. Update `dict_to_compiled_level()` function to copy the new field:
```python
def dict_to_compiled_level(compiled_dict):
    # ... existing code ...
    
    # Get the actual array size for this level
    array_size = compiled_dict["array_size"]
    
    # For array fields, copy actual data and zero-fill remaining slots:
    for i in range(array_size):
        level.new_field_name[i] = compiled_dict.get("new_field_name", [0] * array_size)[i]
    
    # Zero-fill remaining slots (important for deterministic behavior)
    for i in range(array_size, MAX_TILES):
        level.new_field_name[i] = 0  # Or appropriate default
    
    # For single value fields:
    level.new_field_name = compiled_dict.get("new_field_name", default_value)
    
    # ... rest of function ...
```

#### C. Update `compiled_level_to_dict()` function for reverse conversion:
```python
def compiled_level_to_dict(level):
    # ... existing code ...
    return {
        # ... existing fields ...
        "new_field_name": list(level.new_field_name),  # For arrays
        # OR
        "new_field_name": level.new_field_name,  # For single values
    }
```

### Step 4: Add Static Assert for New Field
**⚠️ CRITICAL: Verify binary compatibility! ⚠️**

In `src/madrona_escape_room_c_api.cpp`, add a static_assert to verify the new field has the same offset in both structs. This ensures the direct struct copy will work correctly:

```cpp
// Around line 77-86, with the other static_assert checks:
static_assert(offsetof(MER_CompiledLevel, new_field_name) == offsetof(CompiledLevel, new_field_name),
              "new_field_name offset mismatch");
```

**Important Architecture Note:** The C API uses direct struct assignment via `reinterpret_cast` (line 102):

```cpp
// Direct binary copy of entire struct - no manual field copying needed
CompiledLevel cpp_level = *reinterpret_cast<const CompiledLevel*>(c_level);
```

This means:
- **You don't need to manually copy the new field in C++** - it's automatically included
- **The structs MUST remain binary compatible** - verified by static_assert checks
- **Any new field is automatically copied** as long as it exists in both structs at the same offset

**If the static_assert fails**, it means the structs are not binary compatible and you need to:
1. Ensure the field is in the same position in both structs
2. Ensure the field has the same type and size in both structs  
3. Check for any alignment/padding differences
4. Verify MAX_TILES vs MER_MAX_TILES are the same value

### Step 5: Update Default Level (REQUIRED)
In `src/default_level.cpp`, initialize the new field in the embedded default level:
```cpp
// In the default level initialization
for (int i = 0; i < MAX_TILES; i++) {
    default_level.new_field_name[i] = 0;  // Or appropriate default
}
```

### Step 6: Update C++ Tests

#### Must Update:
1. **viewer_test_base.hpp** - Add field comparison in `compareLevels()` function:
   ```cpp
   // In compareLevels() function, add comparison for new field:
   if (level1.new_field_name[i] != level2.new_field_name[i]) {
       return false;
   }
   ```

#### Usually NO Updates Needed (due to zero-initialization):
Most C++ tests use `CompiledLevel level {}` which automatically zero-initializes all fields, so new fields are handled automatically. These files typically don't need changes:
- **test_persistence.cpp** - Uses `CompiledLevel level {}`
- **test_asset_refactor.cpp** - Uses `CompiledLevel level {}`
- Other test files that create levels with `{}`

### Step 7: Update Python Test Struct Definitions

In **test_c_api_struct_validation.py**:
   - **IMPORTANT**: There are multiple MER_CompiledLevel class definitions in this file (typically 2-3)
   - Update ALL occurrences of the MER_CompiledLevel class definition
   - Add new field(s) to _fields_ list in correct position (must match C++ struct order exactly)
   - For array field initialization, follow this pattern:
     ```python
     # In _fields_ list (ALL occurrences):
     ("new_field_name", ctypes.c_float * 1024),  # MAX_TILES = 1024
     
     # In initialization section (for array fields):
     for i in range(compiled["array_size"]):
         level_struct.new_field_name[i] = compiled.get("new_field_name", [0.0] * 1024)[i]
     
     # For single value fields:
     level_struct.new_field_name = compiled.get("new_field_name", default_value)
     ```
   - The test file typically has MER_CompiledLevel defined:
     - Once around line 52-80 in test_c_api_validation
     - Again around line 142-180 in test_manager_creation_with_c_api
     - Sometimes a third time in other test methods
   
### Step 8: Verify Binary I/O Test

The existing test in **test_level_compiler_c_api.py** already verifies binary round-trip:
```python
def test_binary_roundtrip_c_api_arrays(self):
    # This test automatically verifies ALL fields are preserved
    # through save/load cycle - no changes needed!
```

Just run the test to ensure your new field works:
```bash
uv run --group dev pytest tests/python/test_level_compiler_c_api.py::TestBinaryIO -v
```

### Step 9: Build and Verify

```bash
# Build the project using the project-builder agent
# The build will fail if static_asserts don't pass
```

If static_assert fails, check:
1. Field order matches exactly between structs
2. Field types and sizes match
3. MAX_TILES == MER_MAX_TILES

### Step 10: Run Tests

```bash
# Run C++ tests
./build/mad_escape_tests

# Run Python tests
uv run --group dev pytest tests/python/test_c_api_struct_validation.py -v
uv run --group dev pytest tests/python/test_level_compiler_c_api.py -v
```

## Recording Format Impact

**Binary formats that change:**
- `.lvl` files: Direct binary dump of CompiledLevel struct
- `.rec` files: Include embedded CompiledLevel after metadata

**Size changes break compatibility** - Old files cannot be read after adding fields.

## Accessing CompiledLevel in System Calls

Once the CompiledLevel is properly initialized, it can be accessed from any system call in the simulation as a singleton:

### In System Functions (sim.cpp)
```cpp
// Access as const reference in system functions
inline void mySystem(Engine &ctx, /* other components */) {
    const CompiledLevel& level = ctx.singleton<CompiledLevel>();
    
    // Now you can access your fields:
    float boundary = level.world_max_y - level.world_min_y;
    int32_t value = level.new_field_name;
    // etc...
}
```

### In Level Generation (level_gen.cpp)
```cpp
// Access as non-const reference during level generation
static void generateLevel(Engine &ctx) {
    CompiledLevel& level = ctx.singleton<CompiledLevel>();
    
    // Can read and modify during generation:
    level.some_field = calculated_value;
}
```

### Registration Requirements
The CompiledLevel must be registered as a singleton in `Sim::registerTypes()`:
```cpp
static void registerTypes(ECSRegistry &registry) {
    // This registration is already done in sim.cpp:
    registry.registerSingleton<CompiledLevel>();
    // ... other registrations ...
}
```

### Initialization
The CompiledLevel singleton is populated in the `Sim` constructor:
```cpp
Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &world_init) {
    // This is already done in sim.cpp:
    CompiledLevel &compiled_level = ctx.singleton<CompiledLevel>();
    compiled_level = world_init.compiledLevel;  // Copy from per-world init data
    // ... rest of initialization ...
}
```

### Common Usage Examples
```cpp
// Example: Normalizing positions using world boundaries
inline void computeObservations(Engine &ctx, Position pos, SelfObservation &obs) {
    const CompiledLevel& level = ctx.singleton<CompiledLevel>();
    float world_length = level.world_max_y - level.world_min_y;
    obs.globalY = pos.y / world_length;  // Normalized position
}

// Example: Checking level properties
inline void checkLevelProperties(Engine &ctx) {
    const CompiledLevel& level = ctx.singleton<CompiledLevel>();
    if (level.width > 32 || level.height > 32) {
        // Large level, adjust algorithm
    }
}
```

## Final Validation Checklist

After completing all steps, verify:

- [ ] **Step 1**: Field added to C++ `CompiledLevel` struct
- [ ] **Step 2**: Field added to C API `MER_CompiledLevel` struct with identical type/size
- [ ] **Step 3**: Python bindings fully updated (class definition, dict_to_compiled_level, compiled_level_to_dict)
- [ ] **Step 4**: Static assert added and passing (build succeeds)
- [ ] **Step 5**: Default level initializes the field
- [ ] **Step 6**: viewer_test_base.hpp compareLevels() updated to compare new field
- [ ] **Step 7**: ALL MER_CompiledLevel structs in test files updated (usually 2-3 copies)
- [ ] **Step 8**: Binary I/O test passes (automatic verification)
- [ ] **Step 9**: Project builds without errors
- [ ] **Step 10**: All tests pass (C++ and Python)

### Common Issues and Solutions

1. **Static assert fails**: Fields not in same order or different types/sizes
2. **Python tests fail**: Missed updating one of the MER_CompiledLevel definitions in test files
3. **Binary I/O fails**: Field serialization order doesn't match between save/load
4. **Segfaults**: Array size mismatch between MAX_TILES and actual usage
5. **Recording playback fails**: Struct size changed, old recordings incompatible

