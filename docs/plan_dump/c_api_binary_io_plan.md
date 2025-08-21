# C API Binary I/O Plan

## Reading List
Files to review before implementation:
- `src/mgr.cpp` (lines 1680-1720, 1850-1890) - See existing binary I/O patterns
- `src/madrona_escape_room_c_api.cpp` (lines 73-101) - Direct struct copy implementation
- `src/replay_metadata.hpp` - Binary format structures
- `src/types.hpp` (lines 144-195) - CompiledLevel struct definition
- `include/madrona_escape_room_c_api.h` (lines 78-125) - MER_CompiledLevel struct
- `madrona_escape_room/ctypes_bindings.py` (lines 194-250) - Python struct definitions
- `madrona_escape_room/ctypes_bindings.py` (lines 400-450) - dict_to_compiled_level function
- `madrona_escape_room/level_compiler.py` (lines 528-850) - Current binary I/O implementation
- `tests/python/test_recording_binary_format.py` - Manual parsing problems

## Problem Statement & Maintenance Burden

### Current Issues
1. **Missing Fields Bug**: The tile_rand_* fields were missing from manual copying code
2. **Test Failures**: Tests break due to incorrect manual offset calculations
3. **Code Duplication**: 200+ lines of error-prone binary parsing code
4. **Time Cost**: Hours of debugging per struct change

### Example: Recent tile_rand_* Bug
- Added 4 new fields to CompiledLevel struct
- Manual field copying in C API missed them
- Tests failed due to hardcoded offset calculations
- Required updating multiple files to fix

## Key Insight

**We don't need a "binary format" - the C struct IS the format**

The C++ code already does this correctly:
```cpp
// From mgr.cpp - this is all we need!
file.write(reinterpret_cast<const char*>(&level), sizeof(CompiledLevel));
file.read(reinterpret_cast<char*>(&level), sizeof(CompiledLevel));
```

Any abstraction layer on top is just maintenance burden.

## Detailed Implementation Steps

### Step 1: Add C API Functions (20 mins)

In `include/madrona_escape_room_c_api.h`:
```c
// Write compiled level to binary file
MER_EXPORT MER_Result mer_write_compiled_level(
    const char* filepath, 
    const MER_CompiledLevel* level
);

// Read compiled level from binary file
MER_EXPORT MER_Result mer_read_compiled_level(
    const char* filepath, 
    MER_CompiledLevel* level
);

// Expose existing replay write functionality
MER_EXPORT MER_Result mer_write_replay_file(
    MER_ManagerHandle handle, 
    const char* filepath
);
```

In `src/madrona_escape_room_c_api.cpp`:
```cpp
MER_Result mer_write_compiled_level(
    const char* filepath, 
    const MER_CompiledLevel* level
) {
    if (!filepath || !level) {
        return MER_ERROR_NULL_POINTER;
    }
    
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        return MER_ERROR_FILE_IO;
    }
    
    size_t written = fwrite(level, sizeof(MER_CompiledLevel), 1, f);
    fclose(f);
    
    return (written == 1) ? MER_SUCCESS : MER_ERROR_FILE_IO;
}

MER_Result mer_read_compiled_level(
    const char* filepath, 
    MER_CompiledLevel* level
) {
    if (!filepath || !level) {
        return MER_ERROR_NULL_POINTER;
    }
    
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return MER_ERROR_FILE_IO;
    }
    
    size_t read = fread(level, sizeof(MER_CompiledLevel), 1, f);
    fclose(f);
    
    return (read == 1) ? MER_SUCCESS : MER_ERROR_FILE_IO;
}
```

### Step 2: Add Python Helper (10 mins)

In `madrona_escape_room/ctypes_bindings.py`:
```python
def compiled_level_to_dict(level: MER_CompiledLevel) -> dict:
    """
    Convert MER_CompiledLevel ctypes structure to dictionary.
    Reverse of dict_to_compiled_level.
    """
    # Get actual array size for this level
    array_size = level.width * level.height
    
    return {
        # Header fields
        "num_tiles": level.num_tiles,
        "max_entities": level.max_entities,
        "width": level.width,
        "height": level.height,
        "scale": level.scale,
        "level_name": level.level_name.decode('utf-8').rstrip('\x00'),
        
        # World boundaries
        "world_min_x": level.world_min_x,
        "world_max_x": level.world_max_x,
        "world_min_y": level.world_min_y,
        "world_max_y": level.world_max_y,
        "world_min_z": level.world_min_z,
        "world_max_z": level.world_max_z,
        
        # Spawn data
        "num_spawns": level.num_spawns,
        "spawn_x": list(level.spawn_x),
        "spawn_y": list(level.spawn_y),
        "spawn_facing": list(level.spawn_facing),
        
        # Tile arrays (only copy actual data, not padding)
        "object_ids": list(level.object_ids[:array_size]),
        "tile_x": list(level.tile_x[:array_size]),
        "tile_y": list(level.tile_y[:array_size]),
        "tile_z": list(level.tile_z[:array_size]),
        "tile_persistent": list(level.tile_persistent[:array_size]),
        "tile_render_only": list(level.tile_render_only[:array_size]),
        "tile_entity_type": list(level.tile_entity_type[:array_size]),
        "tile_response_type": list(level.tile_response_type[:array_size]),
        
        # Transform arrays
        "tile_scale_x": list(level.tile_scale_x[:array_size]),
        "tile_scale_y": list(level.tile_scale_y[:array_size]),
        "tile_scale_z": list(level.tile_scale_z[:array_size]),
        "tile_rot_w": list(level.tile_rot_w[:array_size]),
        "tile_rot_x": list(level.tile_rot_x[:array_size]),
        "tile_rot_y": list(level.tile_rot_y[:array_size]),
        "tile_rot_z": list(level.tile_rot_z[:array_size]),
        
        # Randomization arrays
        "tile_rand_x": list(level.tile_rand_x[:array_size]),
        "tile_rand_y": list(level.tile_rand_y[:array_size]),
        "tile_rand_z": list(level.tile_rand_z[:array_size]),
        "tile_rand_rot_z": list(level.tile_rand_rot_z[:array_size]),
        
        # Metadata
        "array_size": array_size,
    }

# Add ctypes declarations
lib.mer_write_compiled_level.argtypes = [c_char_p, POINTER(MER_CompiledLevel)]
lib.mer_write_compiled_level.restype = c_int

lib.mer_read_compiled_level.argtypes = [c_char_p, POINTER(MER_CompiledLevel)]
lib.mer_read_compiled_level.restype = c_int
```

### Step 3: Update level_compiler.py (5 mins)

Replace entire binary I/O implementation:
```python
from .ctypes_bindings import (
    dict_to_compiled_level, 
    compiled_level_to_dict,
    MER_CompiledLevel,
    lib,
    MER_SUCCESS
)
from ctypes import byref

def save_compiled_level_binary(compiled_dict: Dict, filepath: str) -> None:
    """
    Save compiled level dictionary to binary .lvl file using C API.
    
    Args:
        compiled_dict: Output from compile_level()
        filepath: Path to .lvl file to create
    
    Raises:
        IOError: If file cannot be written
    """
    level = dict_to_compiled_level(compiled_dict)
    result = lib.mer_write_compiled_level(filepath.encode('utf-8'), byref(level))
    
    if result != MER_SUCCESS:
        raise IOError(f"Failed to write level file: {filepath} (error code: {result})")

def load_compiled_level_binary(filepath: str) -> Dict:
    """
    Load compiled level dictionary from binary .lvl file using C API.
    
    Args:
        filepath: Path to .lvl file
    
    Returns:
        Dict matching compile_level() output format
    
    Raises:
        IOError: If file cannot be read
    """
    level = MER_CompiledLevel()
    result = lib.mer_read_compiled_level(filepath.encode('utf-8'), byref(level))
    
    if result != MER_SUCCESS:
        raise IOError(f"Failed to read level file: {filepath} (error code: {result})")
    
    return compiled_level_to_dict(level)
```

### Step 4: Fix Tests (5 mins)

In `tests/python/test_recording_binary_format.py`:
```python
def test_compiled_level_structure_validation(cpu_manager):
    """Test CompiledLevel structure validation using C API"""
    mgr = cpu_manager
    
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name
    
    try:
        # Create recording
        mgr.start_recording(recording_path, seed=999)
        # ... run simulation ...
        mgr.stop_recording()
        
        # Read using C API functions instead of manual parsing
        from madrona_escape_room.ctypes_bindings import MER_CompiledLevel, lib
        from ctypes import byref
        
        # Skip metadata (192 bytes) and read CompiledLevel
        with open(recording_path, "rb") as f:
            f.seek(192)  # Skip ReplayMetadata
            
            # Read CompiledLevel directly
            level = MER_CompiledLevel()
            level_bytes = f.read(sizeof(MER_CompiledLevel))
            memmove(byref(level), level_bytes, sizeof(MER_CompiledLevel))
            
            # Now validate using struct fields, not manual offsets
            assert 0 <= level.num_tiles <= 1024
            assert level.max_entities > 0
            assert 0 <= level.num_spawns <= 8
            # etc...
    finally:
        os.unlink(recording_path)
```

## Why This Is Maintainable

1. **Zero maintenance on struct changes** - Add a field to the struct, it automatically works everywhere
2. **No binary format documentation needed** - The struct IS the documentation
3. **No synchronization bugs possible** - Single source of truth (C++ struct)
4. **Trivial debugging** - Just print struct fields, no hex editor needed
5. **Type safety via ctypes** - Compile-time verification of struct compatibility
6. **Fast** - Direct memory I/O, no serialization overhead

## What Gets Deleted

- **200+ lines of manual parsing** in `level_compiler.py`
- **All offset calculations** in test files
- **Binary format documentation** - no longer needed
- **Struct duplication** - use ctypes structs directly
- **Error-prone memcpy chains** in C API

## Testing Strategy

1. **Round-trip tests**: Write → Read → Verify all fields match
2. **Cross-validation**: Ensure Python-written files work with C++ reader
3. **Backward compatibility**: Test with existing .lvl files
4. **Performance**: Benchmark against current implementation
5. **Error cases**: Invalid files, permissions, corrupted data

## Benefits Analysis

### Implementation Time
- Add C API functions: ~20 minutes
- Add Python helper: ~10 minutes  
- Update level_compiler: ~5 minutes
- Fix tests: ~5 minutes
- **Total: ~40 minutes**

### Future Maintenance
- Time per struct field addition: **0 minutes**
- No code changes needed when adding fields
- Automatic inclusion of all fields
- No risk of forgetting to update parsers

### Bug Reduction
- Eliminates entire class of offset calculation bugs
- No more missing field bugs
- No more manual parsing errors
- Type-safe at compile time

### Code Reduction
- Remove 200+ lines of error-prone code
- Simpler, more readable codebase
- Less documentation needed

## Conclusion

This approach leverages C's native ability to write structs directly to disk, treating the struct itself as the serialization format. This is the simplest solution that could possibly work, and therefore the best solution for long-term maintainability.

The key insight: **Don't build an abstraction layer over binary I/O - just use the struct directly.**