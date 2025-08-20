# Plan Template: Adding a New Field to CompiledLevel Structure

## Overview
This plan template describes the steps needed to add a new field to the `CompiledLevel` structure in `src/types.hpp`. The `CompiledLevel` structure is a GPU-compatible data structure used to store level information, and changes to it must be synchronized across C++, C API, Python bindings, recording serialization, and file inspection tools.

## Files That Need Modification

### Core Structure Files
1. **src/types.hpp** - Add field to C++ `CompiledLevel` struct
2. **include/madrona_escape_room_c_api.h** - Add field to C API `MER_CompiledLevel` struct  
3. **madrona_escape_room/ctypes_bindings.py** - Add field to Python `MER_CompiledLevel` class

### Binary I/O Files (For Python to write/read level files)
4. **madrona_escape_room/level_compiler.py** - Update only the binary I/O functions:
   - `save_compiled_level_binary()` - Write new field to .lvl files
   - `load_compiled_level_binary()` - Read new field from .lvl files

### Level Processing Files  
5. **src/level_gen.cpp** - Use new field in level generation (if needed)
6. **src/default_level.cpp** - ALWAYS initialize field in default level

### Recording System Files (Automatic)
- **src/mgr.cpp** - No changes needed (writes entire struct)
- **src/replay_metadata.hpp** - No changes needed (reads entire struct)
- **src/file_inspector.cpp** - No changes needed (validates via sizeof)

## Implementation Steps

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
In `madrona_escape_room/ctypes_bindings.py`, add field to `MER_CompiledLevel._fields_`:
```python
class MER_CompiledLevel(Structure):
    _fields_ = [
        # ... existing fields ...
        ("new_field_name", c_int32 * MAX_TILES),
    ]
```

### Step 4: Update Binary I/O Functions Only
In `madrona_escape_room/level_compiler.py`:

#### Update `save_compiled_level_binary()`:
```python
# Add serialization for new field (after existing fields)
# For single values:
f.write(struct.pack("<i", compiled.get("new_field_name", 0)))

# For arrays:
for i in range(MAX_TILES_C_API):
    value = compiled.get("new_field_name", [0]*MAX_TILES_C_API)[i]
    f.write(struct.pack("<i", value))
```

#### Update `load_compiled_level_binary()`:
```python
# Add deserialization for new field (in same order as save)
# For single values:
new_field_name = struct.unpack("<i", f.read(4))[0]

# For arrays:
new_field_name = []
for _ in range(MAX_TILES_C_API):
    new_field_name.append(struct.unpack("<i", f.read(4))[0])

# Add to returned dictionary
compiled["new_field_name"] = new_field_name
```

### Step 5: Update Level Generation (if needed)
In `src/level_gen.cpp`, use the new field if it affects entity creation:
```cpp
// Access new field from CompiledLevel
int32_t value = level->new_field_name[i];
// Use value in entity setup...
```

### Step 6: Update Default Level (REQUIRED)
In `src/default_level.cpp`, initialize the new field in the embedded default level:
```cpp
// In the default level initialization
for (int i = 0; i < MAX_TILES; i++) {
    default_level.new_field_name[i] = 0;  // Or appropriate default
}
```

## C++ Tests That Need Updates

### Must Update:
1. **test_file_inspector.cpp** - Uses `sizeof(CompiledLevel)`
2. **test_levels.hpp** - Initialize new field in `CreateTestLevel()`
3. **viewer_test_base.hpp** - Add field comparison in `CompareLevels()`

### Should Update:
4. **test_persistence.cpp** - Initialize new field
5. **test_recording_utilities.cpp** - Initialize new field
6. **test_asset_refactor.cpp** - Initialize new field
7. **test_level_utilities.cpp** - Initialize new field
8. **cpp_test_base.hpp** - Initialize new field if creating levels

## Python Tests That Need Updates

### Must Update:
1. **test_c_api_struct_validation.py** - Update struct definition and size validation
2. **test_level_compiler_c_api.py** - Test binary save/load with new field

## Recording Format Impact

**Binary formats that change:**
- `.lvl` files: Direct binary dump of CompiledLevel struct
- `.rec` files: Include embedded CompiledLevel after metadata

**Size changes break compatibility** - Old files cannot be read after adding fields.

## Validation Checklist
- [ ] C++ and C structures have identical memory layout
- [ ] Python ctypes structure matches C layout exactly  
- [ ] Binary save/load functions handle new field
- [ ] Default level initializes new field
- [ ] File inspector validates new struct size automatically
- [ ] Tests initialize new field properly
- [ ] Recording/replay works with new field
- [ ] Level files (.lvl) can be written/read by Python