# Code Generation Tools

This directory contains code generation scripts used during the build process to maintain synchronization between C++ and Python code.

## Scripts

### `generate_python_structs.py`

Automatically generates Python ctypes structures from compiled C++ binaries using `pahole`.

**Purpose**: Ensures Python structs have the exact same memory layout as C++ structs, eliminating manual synchronization.

**Usage**: 
```bash
python codegen/generate_python_structs.py <library_path> <output_path>
# Example (run automatically by CMake):
python codegen/generate_python_structs.py build/libmadrona_escape_room_c_api.so madrona_escape_room/generated_structs.py
```

**Requirements**:
- `pahole` (install with: `sudo apt install dwarves`)
- Debug symbols in the compiled binary (`-g` flag)

**Generated Output**: `madrona_escape_room/generated_structs.py` (tracked in git)

**Structs Extracted**:
- `CompiledLevel` - Level data with tile positions and configurations
- `Action` - Agent action structure
- `SelfObservation` - Agent observation data
- `Done` - Episode completion flag
- `Reward` - Reward value
- `Progress` - Agent progress tracking
- `StepsRemaining` - Episode steps counter
- `ReplayMetadata` - Recording metadata
- `ManagerConfig` - Simulation configuration

**How it works**:
1. Runs `pahole -C <struct_name> <library>` for each struct
2. Parses pahole output using regex to extract field information (name, type, offset, size)
3. Maps C types to ctypes equivalents (int32_t → c_int32, float → c_float, etc.)
4. Handles arrays including special case for char arrays (strings)
5. Adds padding fields where gaps exist in memory layout
6. Generates ctypes.Structure classes with `_fields_` definitions
7. Includes size assertions using `ctypes.sizeof()` to validate layouts match C++ exactly
8. Extracts MAX_TILES from CompiledLevel array size for compatibility

### `generate_python_constants.py`

Automatically generates Python constants from C++ headers using libclang AST parsing.

**Purpose**: Creates a clean Python namespace hierarchy that directly mirrors the C++ constant structure, providing a single source of truth for all constants.

**Usage**: 
```bash
python codegen/generate_python_constants.py [--verbose] <consts.hpp> <types.hpp> <output.py>
# Example (run automatically by CMake):
python codegen/generate_python_constants.py src/consts.hpp src/types.hpp madrona_escape_room/generated_constants.py
```

**Command-line Options**:
- `--verbose` - Enable detailed debug output showing AST traversal and namespace discovery

**Requirements**:
- `libclang` Python bindings (install with: `pip install libclang` or `uv pip install libclang`)
- `clang.native` package for libclang.so path resolution
- System libclang library (typically installed with clang)
- Environment variable `MADRONA_INCLUDE_DIR` (optional, for Madrona headers)

**Generated Output**: `madrona_escape_room/generated_constants.py` (tracked in git)

**Configuration** (defined in `GENERATION_CONFIG`):
- **namespace_classes**: Defines which C++ namespaces become Python classes
  - `madEscape::consts` → `consts` class
  - `madEscape::consts::limits` → `limits` class  
  - `types` → `types` class
- **aliases**: Creates convenience module-level aliases
  - `action = consts.action` (if action namespace exists)

**How it works**:
1. Creates a wrapper C++ file that includes required headers and Madrona dependencies
2. Parses with libclang using C++20 standard and appropriate include paths
3. Traverses the AST recursively to build a namespace tree structure
4. Extracts from each namespace:
   - `constexpr` variables and their literal values
   - Enum declarations and all enum constant values
   - Static const members from struct/class definitions
5. Handles special cases:
   - Float literals with 'f' suffix stripped
   - Expression values preserved as strings if not evaluable
   - Python keyword conflicts renamed with underscore suffix (e.g., "None" → "None_")
6. Generates Python module with:
   - All enums as top-level classes (flattened from namespace hierarchy)
   - Namespace classes as configured in `GENERATION_CONFIG`
   - Module-level aliases for convenience
   - `__all__` export list for clean imports
   - `__slots__ = ()` on all classes to prevent runtime attribute assignment

**Python API**:
```python
# Access constants using the same structure as C++
from madrona_escape_room import consts, action, ExecMode
from madrona_escape_room.generated_constants import limits

# Examples:
consts.episodeLen           # 200
consts.worldLength          # 40.0
action.move_amount.FAST     # 3
action.rotate.SLOW_LEFT     # 1
consts.physics.gravityAcceleration  # 9.8
limits.maxTiles             # 1024 (from consts::limits namespace)
limits.maxSpawns            # 8
ExecMode.CPU                # 0 (enum value)
ExecMode.CUDA               # 1
```

## Architectural Notes

### Dependency Management
- `generated_structs.py` imports constants from `generated_constants.py` to avoid hardcoding values
- Constants like `MAX_TILES` and `MAX_SPAWNS` are defined in `consts.hpp` under the `limits` namespace
- This ensures a single source of truth for all structural limits

### File Tracking Strategy
- Both `generated_constants.py` and `generated_structs.py` are tracked in git
- This ensures stable imports and consistent Python module behavior
- Regeneration during build keeps them in sync with C++ definitions

### Python Module Organization
- Constants are directly exported from the main module: `from madrona_escape_room import ExecMode, consts`
- No intermediate `madrona` class wrapper - cleaner API
- Tensor class moved to separate `tensor.py` file for better code organization

## Adding New Code Generation Scripts

When adding new code generation tools:
1. Place the script in this directory
2. Update the relevant CMakeLists.txt to call it
3. Document the script's purpose and usage in this README
4. Ensure generated files are placed in appropriate locations
5. Consider adding the generated files to `.gitignore` if they shouldn't be committed