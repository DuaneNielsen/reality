# Fix tile_rand_* Fields Copying in mer_create_manager

## Pre-Reading List

### 1. Understand the Struct Definitions
- **`src/types.hpp`** (Lines 144-195)
  - Review the C++ `CompiledLevel` struct
  - Note the `tile_rand_*` fields at lines 191-194
  - Confirm all fields that need to be transferred

- **`include/madrona_escape_room_c_api.h`** (Lines 78-125)
  - Review the C `MER_CompiledLevel` struct
  - Verify it has matching `tile_rand_*` fields at lines 121-124
  - Check that both structs are meant to be binary compatible

### 2. Current Implementation Problem
- **`src/madrona_escape_room_c_api.cpp`** (Lines 59-156)
  - Read the `mer_create_manager` function
  - Lines 71-76: See the static_assert checks confirming binary compatibility
  - Lines 88-154: Observe the manual field-by-field copying
  - **Critical**: Notice that `tile_rand_*` fields are NOT being copied

### 3. Binary Compatibility Verification
- **`src/madrona_escape_room_c_api.cpp`** (Lines 73-76)
  - Static assertions already verify size and offset compatibility
  - This confirms the structs are designed to be binary compatible

## Problem Statement

The `mer_create_manager` function manually copies fields from `MER_CompiledLevel` to `CompiledLevel`, but it's missing the `tile_rand_*` fields. Since the structs are binary compatible (verified by static_assert), we should replace the error-prone manual copying with direct assignment.

## Implementation Plan

### Step 1: Add Additional Static Asserts for Safety
Add static_assert checks for the tile_rand fields to ensure complete compatibility in `src/madrona_escape_room_c_api.cpp` after line 76:

```cpp
static_assert(offsetof(MER_CompiledLevel, tile_rand_x) == offsetof(CompiledLevel, tile_rand_x),
              "tile_rand_x offset mismatch");
static_assert(offsetof(MER_CompiledLevel, tile_rand_rot_z) == offsetof(CompiledLevel, tile_rand_rot_z),
              "tile_rand_rot_z offset mismatch");
```

### Step 2: Replace Manual Copying with Direct Assignment
Replace lines 91-152 (the entire manual copying block) with:

```cpp
// Since structs are binary compatible (verified by static_assert above),
// we can directly copy the entire struct
CompiledLevel cpp_level = *reinterpret_cast<const CompiledLevel*>(c_level);
```

This single line replaces ~60 lines of error-prone manual copying.

### Step 3: Clean Up
Remove the now-unnecessary `array_size` calculation (lines 116-117) since we're no longer using it for individual array copies.

### Step 4: Build and Test
1. Build the project to ensure compilation succeeds
2. Run existing tests to verify nothing breaks
3. Specifically run level compiler tests:
   ```bash
   uv run --group dev pytest tests/python/test_ascii_level_compiler.py -v
   uv run --group dev pytest tests/python/test_c_api_struct_validation.py -v
   ```

## Complete Code Change

The entire change in `src/madrona_escape_room_c_api.cpp` will be:

**Remove** (lines 91-152):
```cpp
CompiledLevel cpp_level;
cpp_level.num_tiles = c_level->num_tiles;
cpp_level.max_entities = c_level->max_entities;
cpp_level.width = c_level->width;
cpp_level.height = c_level->height;
cpp_level.scale = c_level->scale;
// ... ~60 lines of manual copying ...
std::memcpy(cpp_level.tile_rot_z, c_level->tile_rot_z,
           sizeof(float) * array_size);
```

**Replace with**:
```cpp
// Add safety checks for new fields
static_assert(offsetof(MER_CompiledLevel, tile_rand_x) == offsetof(CompiledLevel, tile_rand_x),
              "tile_rand_x offset mismatch");
static_assert(offsetof(MER_CompiledLevel, tile_rand_rot_z) == offsetof(CompiledLevel, tile_rand_rot_z),
              "tile_rand_rot_z offset mismatch");

// Since structs are binary compatible (verified by static_assert above),
// we can directly copy the entire struct
CompiledLevel cpp_level = *reinterpret_cast<const CompiledLevel*>(c_level);
```

## Note on Endianness

No endianness concerns here because:
1. Python creates the `MER_CompiledLevel` struct in memory using ctypes
2. The struct is passed by pointer to the C API in the same process
3. ctypes uses native byte order and alignment
4. No serialization or network transfer is involved
5. The `reinterpret_cast` just reads the same memory with a different type

Endianness only matters for the binary `.lvl` file I/O (where we use little-endian format), not for in-memory struct passing.

## Benefits

1. **Correctness**: All fields are guaranteed to be copied, including tile_rand_*
2. **Performance**: Single operation instead of multiple memcpy calls
3. **Maintainability**: No need to update when fields are added
4. **Simplicity**: 1 line instead of 60+ lines
5. **Future-proof**: Any new fields added to both structs will automatically be copied

## Testing Verification

After making the change:
1. All existing tests should continue to pass
2. The tile_rand_* fields will now be properly transferred
3. Any future randomization features will work correctly

## Commit Message

```
fix: use direct struct copy instead of manual field copying in mer_create_manager

- Replace 60+ lines of manual field copying with direct assignment
- Fixes missing tile_rand_* fields that weren't being copied
- Add static_assert checks for tile_rand field offsets
- Leverages existing binary compatibility between MER_CompiledLevel and CompiledLevel

This ensures all fields are always copied and prevents future copying bugs.
```