# ctypes Bindings Packaging Guide

This guide explains what needs to be packaged for the ctypes bindings to work correctly when distributed.

## Overview

The ctypes bindings require several shared libraries to be bundled with the Python package since they access the C API directly through `ctypes.CDLL()`.

## Required Files for Distribution

### 1. Core Python Files
- `madrona_escape_room/__init__.py` - Main Python API
- `madrona_escape_room/ctypes_bindings.py` - ctypes wrapper for C API
- `madrona_escape_room/__init__.pyi` - Type hints
- `madrona_escape_room/madrona.pyi` - Type hints for madrona submodule

### 2. Main C API Library
- `libmadrona_escape_room_c_api.so` - Primary C API library that ctypes loads
  - Built from: `src/madrona_escape_room_c_api.cpp`
  - Location: `build/libmadrona_escape_room_c_api.so`
  - **CRITICAL**: This is the main library that ctypes_bindings.py loads

### 3. Required Shared Library Dependencies
These must be bundled because they're not standard system libraries:

- `libembree4.so.4` - Intel Embree ray tracing library
- `libdxcompiler.so` - DirectX Shader Compiler  
- `libmadrona_render_shader_compiler.so` - Madrona rendering components
- `libmadrona_std_mem.so` - Madrona memory management

### 4. Python Extension Modules
- `_madrona_escape_room_dlpack.cpython-312-x86_64-linux-gnu.so` - DLPack C extension
  - Built from: `src/dlpack_extension.cpp` by CMake
  - Location: `build/_madrona_escape_room_dlpack.cpython-312-x86_64-linux-gnu.so`
  - Required for PyTorch tensor interoperability via DLPack protocol
  - **CRITICAL**: Must be imported as `from . import _madrona_escape_room_dlpack`

## Current Packaging Status

### ✅ Already Configured in setup.py
```python
package_data={
    "madrona_escape_room": [
        "*.pyi", 
        "*.so", 
        "libmadrona_escape_room_c_api.so",  # Main C API library
        "_madrona_escape_room_dlpack*.so",   # DLPack extension
        "libembree4.so.4",                   # Dependencies
        "libdxcompiler.so",
        "libmadrona_render_shader_compiler.so", 
        "libmadrona_std_mem.so"
    ],
},
```

### ✅ Implemented: CMake Build Integration
- CMake builds the DLPack extension as a MODULE library
- Post-build commands copy all libraries to the package directory
- Setup.py ensures libraries are present for both regular and editable installs

## Current Implementation

### 1. CMake DLPack Extension Build
Added to `src/CMakeLists.txt` after the C API target:
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
    # ... other libraries ...
    COMMENT "Copying all required libraries to Python package directory"
)
```

### 2. Setup.py Library Management
The setup.py ensures libraries are present for both regular and editable installs:

```python
# Helper function to ensure libraries are in package directory
def ensure_libraries_present():
    """Ensure all required libraries are in the package directory."""
    required_libs = [
        "libmadrona_escape_room_c_api.so",
        "libembree4.so.4",
        "libdxcompiler.so",
        "libmadrona_render_shader_compiler.so",
        "libmadrona_std_mem.so",
        "_madrona_escape_room_dlpack.cpython-312-x86_64-linux-gnu.so",
    ]
    
    build_dir = Path("build")
    package_dir = Path("madrona_escape_room")
    
    for lib_name in required_libs:
        dest_path = package_dir / lib_name
        if not dest_path.exists():
            src_path = build_dir / lib_name
            if src_path.exists():
                shutil.copy2(src_path, dest_path)

class BuildPyWithLibrary(build_py):
    """Custom build_py that ensures libraries are present and copies them"""
    def run(self):
        ensure_libraries_present()
        super().run()
        # Also copy to build_lib for regular installs
        
class DevelopWithLibrary(develop):
    """Custom develop command for editable installs"""
    def run(self):
        ensure_libraries_present()
        super().run()
```

### 3. Python Extension Import
The DLPack extension must be imported as a package submodule in `tensor.py`:

```python
def __dlpack__(self, stream=None):
    """Create DLPack capsule for PyTorch consumption"""
    try:
        from . import _madrona_escape_room_dlpack as dlpack_ext
    except ImportError:
        raise ImportError("DLPack extension module not found")
```

This follows Python conventions - C extensions in a package should be imported as part of the package namespace.

### 4. Platform-Specific Considerations

#### Linux (Current)
- File extension: `.so`
- RPATH: Should be set to `$ORIGIN` so libraries find each other
- LD_LIBRARY_PATH: Set in `__init__.py` to include package directory

#### Windows (Future)
- File extension: `.dll`
- Need to handle Windows DLL search path
- May need to copy additional MSVC runtime libraries

#### macOS (Future)
- File extension: `.dylib`
- Need to handle `@rpath` and `install_name_tool`

## Testing Packaging

### 1. Build Distribution Package
```bash
# Build source distribution
python -m build --sdist

# Build wheel
python -m build --wheel
```

### 2. Test Installation in Clean Environment
```bash
# Create clean virtual environment
python -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/madrona_escape_room-*.whl

# Test import and basic functionality
python -c "
import madrona_escape_room as mer
mgr = mer.SimManager(
    exec_mode=mer.madrona.ExecMode.CPU,
    gpu_id=0, num_worlds=1, rand_seed=42, auto_reset=True
)
print('✓ Package works!')
"
```

### 3. Check Included Files
```bash
# Extract wheel and check contents
unzip -l dist/madrona_escape_room-*.whl | grep "\.so"
```

Should show all required `.so` files in the package.

## Troubleshooting

### Library Not Found Errors
```
OSError: libmadrona_escape_room_c_api.so: cannot open shared object file
```

**Cause**: Main C API library not included in package
**Solution**: Ensure `BuildPyWithLibrary` copies the library

### Symbol Not Found Errors
```
OSError: undefined symbol: __nvJitLinkCreate_12_5
```

**Cause**: Missing dependency libraries (embree, dxcompiler, etc.)
**Solution**: Ensure all dependency libraries are copied and LD_LIBRARY_PATH is set

### Import Errors on Different Systems
**Cause**: Platform-specific library differences
**Solution**: Build on target platform or use universal build strategies

## Recommended Build Process

1. **Configure CMake**: `cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo`
2. **Build everything**: `make -C build -j16`
   - This builds the C API library
   - Builds the DLPack extension 
   - Copies all libraries to the package directory
3. **Install package**: `uv pip install -e .` (for development)
4. **Test installation**: `uv run pytest tests/python/test_00_ctypes_cpu_only.py`
5. **Build distribution**: `python -m build` (for release)

## Integration with CI/CD

For automated builds, ensure:
1. C++ build environment with Madrona toolchain available
2. CUDA toolkit installed (for GPU support)
3. Run CMake build which handles everything:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   make -C build -j$(nproc)
   ```
4. Test installation in clean environment:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/*.whl
   python -c "import madrona_escape_room; print('✓ Import successful')"
   ```

## Key Improvements Made

1. **Unified Build System**: CMake now builds all components including the DLPack extension
2. **Automatic Library Management**: Post-build commands ensure all libraries are in the package directory
3. **Proper Python Packaging**: The DLPack extension is imported as a package submodule following Python conventions
4. **Support for Editable Installs**: Custom develop command ensures libraries are present for `pip install -e .`
5. **No Manual Steps**: Everything happens automatically during the build process

This ensures the ctypes bindings and DLPack extension work correctly for both development and distribution.