# Level Compiler to C++ Struct Verification Test Plan

## Pre-Reading

Before implementing this plan, familiarize yourself with the following files and concepts:

### 1. Core Files to Understand

#### Python Side
- **`madrona_escape_room/level_compiler.py`**: 
  - Lines 160-443: The `compile_level()` function that creates the dictionary
  - Lines 46-48: Gets MAX_TILES and MAX_SPAWNS from C API
  - Lines 302-341: How arrays are populated with data

#### C++ Side  
- **`src/types.hpp`**:
  - Lines 144-195: The `CompiledLevel` struct definition
  - Note the static constants: MAX_TILES=1024, MAX_SPAWNS=8

#### C API Bridge
- **`include/madrona_escape_room_c_api.h`**:
  - Lines 78-125: The `MER_CompiledLevel` struct (C version)
  - Line 185: Existing `mer_validate_compiled_level()` function

- **`src/madrona_escape_room_c_api.cpp`**:
  - Lines 59-84: How `mer_create_manager()` uses the compiled level
  - Lines 73-76: Static assertions checking struct layout compatibility
  - Lines 192-203: Current validation implementation

#### Python Bindings
- **`madrona_escape_room/ctypes_bindings.py`**:
  - Lines 194-232: `MER_CompiledLevel` ctypes Structure definition
  - Lines 411-480: `dict_to_compiled_level()` function that converts Python dict to ctypes struct

### 2. Key Concepts

#### The Data Flow
1. ASCII level string → `compile_level()` → Python dictionary
2. Python dictionary → `dict_to_compiled_level()` → ctypes Structure
3. ctypes Structure → C API → C++ `CompiledLevel` struct
4. C++ struct → Used by simulation

#### Current Problem
- We suspect data might not be correctly transferred from Python dict to C++ struct
- Need a way to verify the transfer without fragile field-by-field comparisons
- Current `mer_validate_compiled_level()` only checks basic constraints, not data integrity

#### Existing Tests
- **`tests/python/test_c_api_struct_validation.py`**: Has some validation but may not catch all issues
- Lines 15-128: Basic validation test
- Lines 329-381: Multiple level validation

### 3. Critical Arrays and Their Sizes

All arrays in CompiledLevel have fixed sizes for GPU compatibility:
- Tile arrays: Size = MAX_TILES (1024)
- Spawn arrays: Size = MAX_SPAWNS (8)
- Level name: Size = 64 characters

The Python compiler creates these as full-size arrays but only populates `num_tiles` elements, zero-filling the rest.

---

## Problem Analysis

We need to verify that data from the Python level compiler (`level_compiler.py`) correctly populates the C++ `CompiledLevel` struct via the `MER_CompiledLevel` C API struct. The test should:
- Detect if there's a mismatch in data transfer
- Not be fragile or dependent on detailed struct layout
- Provide clear indication when something is wrong
- Be maintainable as the struct evolves

## Solution Design

Create a **checksum-based verification test** that computes a fingerprint of the struct data to verify integrity.

### Why Checksum Approach?

1. **Single Point of Comparison**: One number to verify instead of many fields
2. **Layout Independent**: Doesn't care about field order or padding
3. **Comprehensive**: Can cover all data in the struct
4. **Fast**: Simple arithmetic operations
5. **Deterministic**: Same input always produces same output

## Implementation Plan

### Phase 1: Add C API Checksum Function

#### 1.1 Add to `include/madrona_escape_room_c_api.h`
```c
// Compute a checksum of the compiled level data for verification
// Returns a 64-bit checksum covering all significant fields
MER_EXPORT uint64_t mer_compute_level_checksum(const MER_CompiledLevel* level);
```

#### 1.2 Implement in `src/madrona_escape_room_c_api.cpp`
```cpp
uint64_t mer_compute_level_checksum(const MER_CompiledLevel* level) {
    if (!level) return 0;
    
    uint64_t checksum = 0;
    
    // Include header fields
    checksum = checksum * 31 + level->num_tiles;
    checksum = checksum * 31 + level->max_entities;
    checksum = checksum * 31 + level->width;
    checksum = checksum * 31 + level->height;
    checksum = checksum * 31 + (uint32_t)(level->scale * 1000); // Scale to int for consistency
    
    // Include spawn data
    checksum = checksum * 31 + level->num_spawns;
    for (int i = 0; i < level->num_spawns; i++) {
        checksum = checksum * 31 + (uint32_t)(level->spawn_x[i] * 1000);
        checksum = checksum * 31 + (uint32_t)(level->spawn_y[i] * 1000);
        checksum = checksum * 31 + (uint32_t)(level->spawn_facing[i] * 1000);
    }
    
    // Include tile data (only active tiles)
    for (int i = 0; i < level->num_tiles; i++) {
        checksum = checksum * 31 + level->object_ids[i];
        checksum = checksum * 31 + (uint32_t)(level->tile_x[i] * 1000);
        checksum = checksum * 31 + (uint32_t)(level->tile_y[i] * 1000);
        checksum = checksum * 31 + (uint32_t)(level->tile_z[i] * 1000);
        checksum = checksum * 31 + (level->tile_persistent[i] ? 1 : 0);
        checksum = checksum * 31 + (level->tile_render_only[i] ? 1 : 0);
        checksum = checksum * 31 + level->tile_entity_type[i];
        checksum = checksum * 31 + level->tile_response_type[i];
        // Include scale and rotation
        checksum = checksum * 31 + (uint32_t)(level->tile_scale_x[i] * 1000);
        checksum = checksum * 31 + (uint32_t)(level->tile_scale_y[i] * 1000);
        checksum = checksum * 31 + (uint32_t)(level->tile_scale_z[i] * 1000);
    }
    
    // Include world boundaries
    checksum = checksum * 31 + (uint32_t)(level->world_min_x * 1000);
    checksum = checksum * 31 + (uint32_t)(level->world_max_x * 1000);
    checksum = checksum * 31 + (uint32_t)(level->world_min_y * 1000);
    checksum = checksum * 31 + (uint32_t)(level->world_max_y * 1000);
    
    return checksum;
}
```

### Phase 2: Update Python Bindings

#### 2.1 Add to `madrona_escape_room/ctypes_bindings.py`
```python
# Add function binding
lib.mer_compute_level_checksum.argtypes = [POINTER(MER_CompiledLevel)]
lib.mer_compute_level_checksum.restype = c_uint64

def compute_level_checksum_from_dict(compiled_dict):
    """
    Compute expected checksum from Python dictionary.
    Must match the C++ implementation logic.
    """
    checksum = 0
    
    # Include header fields
    checksum = (checksum * 31 + compiled_dict["num_tiles"]) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + compiled_dict["max_entities"]) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + compiled_dict["width"]) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + compiled_dict["height"]) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + int(compiled_dict["scale"] * 1000)) & 0xFFFFFFFFFFFFFFFF
    
    # Include spawn data
    checksum = (checksum * 31 + compiled_dict["num_spawns"]) & 0xFFFFFFFFFFFFFFFF
    for i in range(compiled_dict["num_spawns"]):
        checksum = (checksum * 31 + int(compiled_dict["spawn_x"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["spawn_y"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["spawn_facing"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
    
    # Include tile data
    for i in range(compiled_dict["num_tiles"]):
        checksum = (checksum * 31 + compiled_dict["object_ids"][i]) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["tile_x"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["tile_y"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["tile_z"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + (1 if compiled_dict["tile_persistent"][i] else 0)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + (1 if compiled_dict["tile_render_only"][i] else 0)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + compiled_dict["tile_entity_type"][i]) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + compiled_dict["tile_response_type"][i]) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["tile_scale_x"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["tile_scale_y"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
        checksum = (checksum * 31 + int(compiled_dict["tile_scale_z"][i] * 1000)) & 0xFFFFFFFFFFFFFFFF
    
    # Include world boundaries  
    checksum = (checksum * 31 + int(compiled_dict["world_min_x"] * 1000)) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + int(compiled_dict["world_max_x"] * 1000)) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + int(compiled_dict["world_min_y"] * 1000)) & 0xFFFFFFFFFFFFFFFF
    checksum = (checksum * 31 + int(compiled_dict["world_max_y"] * 1000)) & 0xFFFFFFFFFFFFFFFF
    
    return checksum
```

### Phase 3: Create Test File

#### 3.1 Create `tests/python/test_level_compiler_struct_verification.py`
```python
"""
Test to verify data integrity when transferring from Python level compiler
to C++ CompiledLevel struct through ctypes.
"""

import ctypes
import pytest
from madrona_escape_room.level_compiler import compile_level
from madrona_escape_room.ctypes_bindings import (
    dict_to_compiled_level,
    lib,
    compute_level_checksum_from_dict,
    MER_CompiledLevel,
    POINTER
)


class TestLevelCompilerStructVerification:
    """Verify Python to C++ struct data transfer integrity"""
    
    def test_simple_level_checksum(self):
        """Test with a simple deterministic level"""
        level = """####
#S.#
####"""
        
        # Compile in Python
        compiled_dict = compile_level(level, scale=2.5)
        
        # Convert to C struct
        c_struct = dict_to_compiled_level(compiled_dict)
        
        # Get checksum from C API
        c_checksum = lib.mer_compute_level_checksum(ctypes.byref(c_struct))
        
        # Compute expected checksum from Python dict
        expected_checksum = compute_level_checksum_from_dict(compiled_dict)
        
        # Verify they match
        assert c_checksum == expected_checksum, (
            f"Checksum mismatch: C API returned {c_checksum}, "
            f"expected {expected_checksum}. This indicates data transfer issue."
        )
    
    def test_complex_level_checksum(self):
        """Test with a more complex level"""
        level = """##########
#S.......#
#..####..#
#..#..#..#
#..#.C#..#
#........#
##########"""
        
        compiled_dict = compile_level(level, scale=3.0)
        c_struct = dict_to_compiled_level(compiled_dict)
        
        c_checksum = lib.mer_compute_level_checksum(ctypes.byref(c_struct))
        expected_checksum = compute_level_checksum_from_dict(compiled_dict)
        
        assert c_checksum == expected_checksum, (
            f"Complex level checksum mismatch: {c_checksum} != {expected_checksum}"
        )
    
    def test_checksum_detects_corruption(self):
        """Verify checksum detects data corruption"""
        level = """####
#S.#
####"""
        
        compiled_dict = compile_level(level)
        c_struct = dict_to_compiled_level(compiled_dict)
        
        # Get original checksum
        original_checksum = lib.mer_compute_level_checksum(ctypes.byref(c_struct))
        
        # Corrupt a field
        c_struct.num_tiles += 1
        
        # Get new checksum
        corrupted_checksum = lib.mer_compute_level_checksum(ctypes.byref(c_struct))
        
        # Verify checksums differ
        assert original_checksum != corrupted_checksum, (
            "Checksum failed to detect corruption"
        )
    
    @pytest.mark.parametrize("scale", [1.0, 2.5, 5.0])
    def test_different_scales(self, scale):
        """Test checksum with different scale values"""
        level = """###
#S#
###"""
        
        compiled_dict = compile_level(level, scale=scale)
        c_struct = dict_to_compiled_level(compiled_dict)
        
        c_checksum = lib.mer_compute_level_checksum(ctypes.byref(c_struct))
        expected_checksum = compute_level_checksum_from_dict(compiled_dict)
        
        assert c_checksum == expected_checksum, (
            f"Checksum mismatch at scale {scale}"
        )
```

### Phase 4: Integration with Existing Tests

#### 4.1 Add to `tests/python/test_c_api_struct_validation.py`

Add checksum verification to the existing `test_c_api_validation_function`:
```python
def test_c_api_validation_with_checksum(self):
    """Test C API validation with checksum verification"""
    level = """######
#S...#
#.C..#
######"""
    
    compiled = compile_level(level)
    
    # ... existing struct creation code ...
    
    # Add checksum verification
    lib.mer_compute_level_checksum.argtypes = [ctypes.POINTER(MER_CompiledLevel)]
    lib.mer_compute_level_checksum.restype = ctypes.c_uint64
    
    c_checksum = lib.mer_compute_level_checksum(ctypes.byref(level_struct))
    expected_checksum = compute_level_checksum_from_dict(compiled)
    
    assert c_checksum == expected_checksum, (
        f"Data integrity check failed: checksums don't match "
        f"({c_checksum} != {expected_checksum})"
    )
```

## Testing Strategy

### Test Cases

1. **Simple Level**: Minimal 3x3 level with one spawn
2. **Complex Level**: Larger level with multiple objects
3. **Edge Cases**: 
   - Empty tiles (all positions have empty tiles)
   - Maximum size level (32x32)
   - Multiple spawn points
4. **Scale Variations**: Test different scale values
5. **Corruption Detection**: Verify checksum changes when data is modified

### Expected Outcomes

- All tests should pass with matching checksums
- Corruption test should show different checksums
- Tests should be fast (<0.1s each)
- Clear error messages when mismatches occur

## Benefits of This Approach

1. **Simple**: Single value comparison instead of field-by-field
2. **Comprehensive**: Covers all significant data
3. **Fast**: O(n) computation where n is num_tiles
4. **Maintainable**: Easy to update when struct changes
5. **Diagnostic**: Clear indication when data transfer fails
6. **Non-fragile**: Doesn't depend on exact memory layout

## Alternative Approaches Considered

1. **Field-by-field comparison**: Too fragile, maintenance burden
2. **Binary comparison**: Would fail due to padding/alignment
3. **Hash functions (MD5/SHA)**: Overkill for this use case
4. **Random sampling**: Might miss corruption

## Implementation Notes

- Use 64-bit checksum to avoid collisions
- Multiply by prime (31) for good distribution
- Convert floats to integers (×1000) for consistency
- Only process `num_tiles` elements, not full array
- Include all semantically significant fields

## Success Criteria

1. Test reliably detects when data transfer is broken
2. Test passes when data transfer is correct
3. Test execution time < 1 second
4. Clear error messages on failure
5. Easy to maintain as struct evolves