# Code Generation Tools

This directory contains code generation scripts used during the build process to maintain synchronization between C++ and Python code.

## Scripts

### `generate_python_structs.py`

Automatically generates Python ctypes structures from compiled C++ binaries using `pahole`.

**Purpose**: Ensures Python structs have the exact same memory layout as C++ structs, eliminating manual synchronization.

**Usage**: This script is automatically run as a post-build step when building `madrona_escape_room_c_api`.

**Requirements**:
- `pahole` (install with: `sudo apt install dwarves`)
- Debug symbols in the compiled binary (`-g` flag)

**Generated Output**: `madrona_escape_room/generated_structs.py`

**How it works**:
1. Extracts struct layouts from the compiled `.so` file using `pahole`
2. Parses the pahole output to get field offsets and sizes
3. Generates Python ctypes.Structure classes with correct padding
4. Includes size assertions to validate the layout matches

### `generate_python_constants.py`

Automatically generates Python constants from C++ headers using libclang AST parsing.

**Purpose**: Creates a clean Python namespace hierarchy that directly mirrors the C++ constant structure, providing a single source of truth for all constants.

**Usage**: This script is automatically run as a post-build step when building `madrona_escape_room_c_api`.

**Requirements**:
- `libclang` Python bindings (install with: `pip install libclang` or `uv pip install libclang`)
- System libclang library (typically installed with clang)

**Generated Output**: `madrona_escape_room/generated_constants.py`

**How it works**:
1. Parses `consts.hpp` and `types.hpp` using libclang's AST parser
2. Traverses the AST to extract:
   - `constexpr` constants from all namespaces
   - Enum values from enum declarations
   - Static constants from struct definitions
3. Builds a namespace tree matching the C++ structure
4. Generates nested Python classes that mirror the C++ namespace hierarchy
5. Provides convenience aliases for common namespaces (action, physics, rendering, etc.)

**Python API**:
```python
# Access constants using the same structure as C++
from madrona_escape_room import consts, types, action

# Examples:
consts.episodeLen           # 200
consts.worldLength          # 40.0
action.move_amount.FAST     # 3
action.rotate.SLOW_LEFT     # 1
consts.physics.gravityAcceleration  # 9.8
types.CompiledLevel.MAX_TILES       # 1024
```

## Adding New Code Generation Scripts

When adding new code generation tools:
1. Place the script in this directory
2. Update the relevant CMakeLists.txt to call it
3. Document the script's purpose and usage in this README
4. Ensure generated files are placed in appropriate locations
5. Consider adding the generated files to `.gitignore` if they shouldn't be committed