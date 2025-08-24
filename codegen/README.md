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

## Adding New Code Generation Scripts

When adding new code generation tools:
1. Place the script in this directory
2. Update the relevant CMakeLists.txt to call it
3. Document the script's purpose and usage in this README
4. Ensure generated files are placed in appropriate locations
5. Consider adding the generated files to `.gitignore` if they shouldn't be committed