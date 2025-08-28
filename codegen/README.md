# Code Generation Tools

This directory contains code generation scripts used during the build process to maintain synchronization between C++ and Python code.

## Overview

These scripts automatically generate Python bindings from C++ code to ensure perfect memory layout compatibility and constant synchronization between C++ and Python. They run as post-build steps in the CMake build process.

## Scripts

### `generate_python_structs.py`

Automatically generates Python ctypes structures from compiled C++ binaries using `pahole`.

**Purpose**: Ensures Python structs have the exact same memory layout as C++ structs, eliminating manual synchronization errors.

**Usage**:
```bash
python codegen/generate_dataclass_structs.py <library_path> <output_path>

# Example (automatically run by CMake):
python codegen/generate_dataclass_structs.py \
    build/libmadrona_escape_room_c_api.so \
    madrona_escape_room/generated_dataclasses.py
```

**Requirements**:
- `pahole` tool from dwarves package (install with: `sudo apt install dwarves`)
- Compiled library with debug symbols (`-g` flag in compilation)

**Generated Output**: `madrona_escape_room/generated_dataclasses.py`

**Structs Extracted**:
- `CompiledLevel` - Level data with tile positions and spawn configurations
- `Action` - Agent action structure (movement and rotation)
- `SelfObservation` - Agent observation data
- `Done` - Episode completion flag
- `Reward` - Reward value
- `Progress` - Agent progress tracking
- `StepsRemaining` - Episode steps counter
- `ReplayMetadata` - Recording file metadata
- `ManagerConfig` - Simulation configuration

**How it works**:
1. Runs `pahole -C <struct_name> <library>` for each struct to extract memory layout
2. Parses pahole output using regex to extract:
   - Field names, types, offsets, and sizes
   - Array dimensions for array fields
3. Maps C types to ctypes equivalents:
   - `int32_t` → `ctypes.c_int32`
   - `float` → `ctypes.c_float`
   - `char[N]` → `ctypes.c_char * N`
   - Enums → `ctypes.c_int`
4. Generates ctypes.Structure classes with `_fields_` definitions
5. Adds padding fields (`_pad_N`) where gaps exist in memory layout
6. Includes size assertions to validate layout matches C++ exactly
7. Extracts `MAX_TILES` constant from CompiledLevel array field size

### `generate_python_constants.py`

Automatically generates Python constants from C++ headers using libclang AST parsing.

**Purpose**: Creates a Python module with constants that mirror the C++ namespace structure, providing a single source of truth.

**Usage**:
```bash
python codegen/generate_python_constants.py [--verbose] <consts.hpp> <types.hpp> <output.py>

# Example (automatically run by CMake):
python codegen/generate_python_constants.py \
    src/consts.hpp \
    src/types.hpp \
    madrona_escape_room/generated_constants.py
```

**Command-line Options**:
- `--verbose` - Enable detailed debug output showing AST traversal

**Requirements**:
- `libclang` Python package (install with: `pip install libclang` or `uv pip install libclang`)
- `clang.native` package for libclang.so path resolution
- System libclang library (typically installed with clang)
- Optional: `MADRONA_INCLUDE_DIR` environment variable for Madrona headers

**Generated Output**: `madrona_escape_room/generated_constants.py`

**Configuration** (in `GENERATION_CONFIG` dict):
- **namespace_classes**: Maps C++ namespaces to Python classes
  - `madEscape::consts` → `consts` class
  - `madEscape::consts::limits` → `limits` class
  - `types` → `types` class
- **aliases**: Creates module-level convenience aliases
  - `action = consts.action` (if action namespace exists)

**How it works**:
1. Creates a wrapper C++ file including all headers and dependencies
2. Parses using libclang with C++20 standard
3. Traverses AST recursively to build namespace tree
4. Extracts from each namespace:
   - `constexpr` variable declarations and literal values
   - Enum declarations with all constant values
   - Static const members from structs/classes
5. Special handling:
   - Float literals: strips 'f' suffix
   - Complex expressions: preserved as string comments
   - Python keywords: renamed with underscore suffix (e.g., `None` → `None_`)
6. Generates Python module with:
   - All enums as top-level classes (flattened hierarchy)
   - Namespace classes per configuration
   - `__slots__ = ()` to prevent runtime modification
   - `__all__` export list for clean imports

**Python API Example**:
```python
from madrona_escape_room import consts, ExecMode
from madrona_escape_room.generated_constants import limits

# Access constants matching C++ structure
consts.episodeLen           # 200
consts.worldLength          # 40.0
consts.action.move_amount.FAST  # 3
limits.maxTiles             # 1024
ExecMode.CPU                # 0
```

## CMake Integration

Both scripts run automatically as POST_BUILD commands for the `madrona_escape_room_c_api` target:

```cmake
add_custom_command(TARGET madrona_escape_room_c_api POST_BUILD
    COMMAND ${Python3_EXECUTABLE} 
        ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/generate_dataclass_structs.py
        $<TARGET_FILE:madrona_escape_room_c_api>
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/generated_dataclasses.py
    
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/generate_python_constants.py
        ${CMAKE_CURRENT_SOURCE_DIR}/consts.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/types.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/generated_constants.py
)
```

## Architecture Notes

### Separation of Concerns

The code generation scripts have a specific, limited responsibility:

**What they DO provide:**
- Exact memory layout matching C++ structs (size, offsets, padding)
- Correct field types and array dimensions
- Pre-allocated Python structures of the correct total size
- Memory-compatible ctypes structures for FFI calls

**What they DON'T provide:**
- Valid initial values for fields
- Game-specific initialization logic
- Semantic validation of data
- Business logic or behavior

**Example:** A `CompiledLevel` struct will be generated with the correct size (e.g., 84180 bytes) and all arrays pre-allocated to their maximum sizes (e.g., 1024 tiles), but the tile positions, spawn points, and other values will be uninitialized (typically zeros). It's the responsibility of the application code (like `level_compiler.py`) to fill these structures with valid game data.

This separation ensures:
- Code generators remain simple and focused on structure
- Game logic stays in application code where it belongs
- Memory layout compatibility without semantic coupling
- Easy testing of both layers independently

### Single Source of Truth
- All constants and struct layouts defined in C++ headers
- Python code automatically synchronized on each build
- No manual duplication or maintenance required

### File Management
- Generated files (`generated_dataclasses.py`, `generated_constants.py`) are tracked in git
- Ensures stable imports and consistent behavior
- Regenerated on build to stay synchronized with C++ changes

### Error Handling
- Both scripts validate their requirements (pahole, libclang) before running
- Clear error messages if dependencies are missing
- Size assertions in generated code catch layout mismatches at import time

## Troubleshooting

### pahole not found
Install the dwarves package:
```bash
sudo apt install dwarves
```

### libclang import error
Install the Python bindings:
```bash
uv pip install libclang
# or
pip install libclang
```

### Struct layout mismatch assertion
This indicates the C++ struct layout changed but Python wasn't regenerated:
1. Clean and rebuild the project
2. Ensure debug symbols are enabled (`-g` flag)
3. Check that the post-build commands are running

### Missing constants or enums
1. Verify the constants are in `consts.hpp` or `types.hpp`
2. Check they use `constexpr` or `enum` declarations
3. Run with `--verbose` flag to debug AST traversal