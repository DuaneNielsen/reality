# Plan: CMake Builds Everything Including DLPack Extension

## Pre-Reading List

Before implementing this plan, familiarize yourself with these files and concepts:

### Essential Files to Read
1. **src/CMakeLists.txt** (lines 130-326) - Current build configuration for C API library and post-build commands
2. **src/dlpack_extension.cpp** - The DLPack extension source that needs to be built
3. **setup.py** - Current Python packaging setup with dual build system
4. **madrona_escape_room/ctypes_bindings.py** (lines 70-85) - How the C API library is loaded
5. **external/madrona/include/madrona/py/utils.hpp** - Madrona Python utilities used by dlpack

### Key Concepts to Understand
- **Python Extension Naming**: Extensions need the suffix `.cpython-312-x86_64-linux-gnu.so` for Python to recognize them
- **Madrona Toolchain**: Located at `external/madrona/external/madrona-toolchain/bundled-toolchain/toolchain/bin/clang++`
- **Library Dependencies**: The C API needs embree, dxcompiler, madrona_render_shader_compiler, and madrona_std_mem
- **CMake MODULE libraries**: Use `add_library(target MODULE ...)` for Python extensions
- **Post-build commands**: CMake can copy files after building targets

### Current Problems
1. setup.py tries to build dlpack extension with system compiler (`/usr/bin/x86_64-linux-gnu-g++`)
2. Madrona headers require `-DMADRONA_CLANG` and `-nostdlib++` flags
3. Libraries are scattered across build directory and need manual copying
4. Dual build system causes confusion and maintenance burden

## Goal
Have CMake handle building all components including the dlpack Python extension, ensuring consistent toolchain usage and automatic library management.

## Implementation Steps

### 1. Add DLPack Extension to CMakeLists.txt

Add a new CMake target that builds the dlpack extension as a Python module:
- Create a shared library target for `_madrona_escape_room_dlpack`
- Use the Madrona toolchain (already configured in CMake)
- Set output name with proper Python extension suffix (`.cpython-312-x86_64-linux-gnu.so`)
- Link against Python libraries and Madrona runtime
- Place the built extension directly in the package directory

### 2. Add Post-Build Library Copying

Extend the existing post-build commands to copy all required libraries:
- Copy the dlpack extension to `madrona_escape_room/` 
- Copy all dependency libraries (embree, dxcompiler, etc.) to package directory
- Ensure all libraries are copied after successful build

### 3. Simplify setup.py

Remove all compilation logic from setup.py:
- Remove `BuildExtWithMadronaToolchain` class (no longer needed)
- Remove `dlpack_extension` Extension definition
- Keep only `BuildPyWithLibrary` for copying pre-built binaries
- Update package_data to include the pre-built dlpack extension

## Detailed Changes

### CMakeLists.txt additions (after madrona_escape_room_c_api target, around line 326):

```cmake
# Python DLPack extension
add_library(madrona_escape_room_dlpack MODULE
    dlpack_extension.cpp
)

# Set proper Python module naming
set_target_properties(madrona_escape_room_dlpack PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_madrona_escape_room_dlpack"
    SUFFIX ".cpython-312-x86_64-linux-gnu.so"
)

target_include_directories(madrona_escape_room_dlpack PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/python3.12
)

target_compile_definitions(madrona_escape_room_dlpack PRIVATE
    MADRONA_CLANG
)

target_link_libraries(madrona_escape_room_dlpack PRIVATE
    madrona_common
    madrona_python_utils
)

# Copy all required libraries to package directory
add_custom_command(TARGET madrona_escape_room_dlpack POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:madrona_escape_room_dlpack>
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:madrona_escape_room_c_api>
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libembree4.so.4
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libdxcompiler.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libmadrona_render_shader_compiler.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libmadrona_std_mem.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMENT "Copying all required libraries to Python package directory"
)
```

### setup.py simplification:

1. **Remove BuildExtWithMadronaToolchain class** (lines 16-38)
2. **Remove dlpack_extension definition** (lines 87-106)
3. **Change ext_modules to empty list** (line 117):
   ```python
   ext_modules=[],  # All extensions are pre-built by CMake
   ```
4. **Remove "build_ext" from cmdclass** (line 142)
5. **Keep BuildPyWithLibrary** but it becomes simpler since libraries are pre-copied by CMake

## Benefits

1. **Single build system**: CMake handles everything
2. **Consistent toolchain**: All C++ code uses Madrona's clang
3. **Automatic dependency management**: CMake knows all library dependencies
4. **No manual steps**: Everything happens during `make`
5. **Simpler Python packaging**: setup.py just packages pre-built binaries
6. **Easier maintenance**: One place to update build configuration

## Build Process After Changes

```bash
cmake . -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j16
uv pip install -e .
```

All components will be built with the correct toolchain and automatically placed in the right locations.

## Testing Plan

After implementation:
1. Clean build directory and remove all `.so` files from package directory
2. Run CMake configuration
3. Run make and verify all libraries are copied
4. Install package with `uv pip install -e .`
5. Test import: `python -c "import madrona_escape_room"`
6. Run a simple simulation to verify dlpack extension works

## Rollback Plan

If issues arise:
1. Git diff to see all changes
2. Revert CMakeLists.txt changes
3. Restore original setup.py
4. Manually build dlpack extension as before

## Future Improvements

Once this is working:
1. Add CMake option to control Python version detection
2. Add CMake target for running Python tests
3. Consider adding CMake install target for production deployment