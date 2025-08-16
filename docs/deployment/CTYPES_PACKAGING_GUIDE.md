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

### 4. Optional Extension Modules
- `_madrona_escape_room_dlpack.cpython-*.so` - DLPack C extension (if available)
  - Built from: `src/dlpack_extension.cpp`
  - Required for optimal DLPack performance
  - Falls back gracefully if missing

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

### ❌ Missing: Library Copy Step
The build system doesn't automatically copy libraries to the package directory.

## Required Packaging Fixes

### 1. Update BuildPyWithLibrary Class
The current `setup.py` has a `BuildPyWithLibrary` class but it only copies the main C API library. It needs to copy all dependencies.

**Current Code:**
```python
class BuildPyWithLibrary(build_py):
    def run(self):
        super().run()
        
        c_lib = Path("build/libmadrona_escape_room_c_api.so")
        if c_lib.exists():
            # Only copies main library
            shutil.copy2(c_lib, dest)
```

**Needs to be:**
```python
class BuildPyWithLibrary(build_py):
    def run(self):
        super().run()
        
        # Libraries to copy
        required_libs = [
            "libmadrona_escape_room_c_api.so",  # Main C API
            "libembree4.so.4",                   # Dependencies
            "libdxcompiler.so",
            "libmadrona_render_shader_compiler.so",
            "libmadrona_std_mem.so"
        ]
        
        build_dir = Path("build")
        for package in self.packages:
            if package == "madrona_escape_room":
                package_dir = Path(self.build_lib) / package
                
                for lib_name in required_libs:
                    lib_path = build_dir / lib_name
                    if lib_path.exists():
                        dest = package_dir / lib_name
                        print(f"Copying {lib_name} to {dest}")
                        shutil.copy2(lib_path, dest)
                    else:
                        print(f"Warning: {lib_name} not found in build directory")
```

### 2. Update Library Search Path in ctypes_bindings.py
The ctypes bindings should look for libraries in the package directory first:

**Current search order:**
1. `../build/libmadrona_escape_room_c_api.so` (development)
2. Package directory (distribution)

**This is correct** - the current implementation should work for both development and distribution.

### 3. Platform-Specific Considerations

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

1. **Clean build**: `rm -rf build/ dist/ *.egg-info/`
2. **Build C libraries**: `make -C build -j8`
3. **Build Python package**: `python -m build`
4. **Test in clean environment**: See testing section above
5. **Verify library inclusion**: Check wheel contents

## Integration with CI/CD

For automated builds, ensure:
1. C++ build environment available
2. CUDA toolkit installed (for GPU support)
3. All dependency libraries built before Python packaging
4. Test installation in clean environment as final step

This ensures the ctypes bindings work correctly when distributed to end users.