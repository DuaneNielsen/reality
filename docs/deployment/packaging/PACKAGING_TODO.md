# ctypes Packaging TODO

## Current Problem
The ctypes bindings work perfectly in development but will fail when distributed because the build system doesn't package all required libraries.

## What's Working ✅
1. **ctypes_bindings.py** - Correctly searches for libraries in package directory
2. **setup.py package_data** - Lists all required files for inclusion
3. **Main C API library** - Gets copied by BuildPyWithLibrary class

## What's Broken ❌
1. **Dependency libraries not copied** - Only main C API library gets copied to package
2. **Build process incomplete** - Missing step to copy all dependencies

## Required Fixes

### 1. Fix BuildPyWithLibrary Class in setup.py

**Current (incomplete):**
```python
class BuildPyWithLibrary(build_py):
    def run(self):
        super().run()
        
        # Copy the C library to the package
        c_lib = Path("build/libmadrona_escape_room_c_api.so")
        if c_lib.exists():
            # Only copies main library - MISSING DEPENDENCIES!
            shutil.copy2(c_lib, dest)
```

**Needs to be:**
```python
class BuildPyWithLibrary(build_py):
    def run(self):
        super().run()
        
        # ALL libraries that ctypes needs
        required_libs = [
            "libmadrona_escape_room_c_api.so",  # Main C API (ctypes loads this)
            "libembree4.so.4",                   # Embree ray tracing  
            "libdxcompiler.so",                  # DirectX shader compiler
            "libmadrona_render_shader_compiler.so", # Madrona rendering
            "libmadrona_std_mem.so"              # Madrona memory management
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

### 2. Test the Fix

After fixing setup.py:

```bash
# Clean rebuild
rm -rf build/ dist/ *.egg-info/
make -C build -j$(nproc)

# Build package  
python -m build

# Test in clean environment
python -m venv test_env
source test_env/bin/activate
pip install dist/madrona_escape_room-*.whl

# Verify it works
python -c "
import madrona_escape_room as mer
mgr = mer.SimManager(mer.madrona.ExecMode.CPU, 0, 1, 42, True)
print('✅ Packaged ctypes bindings work!')
"
```

### 3. Verify Library Inclusion

Check that wheel contains all libraries:
```bash
unzip -l dist/madrona_escape_room-*.whl | grep "\.so"
```

Should show:
- `libmadrona_escape_room_c_api.so` ✅ Main library
- `libembree4.so.4` ❌ Currently missing  
- `libdxcompiler.so` ❌ Currently missing
- `libmadrona_render_shader_compiler.so` ❌ Currently missing
- `libmadrona_std_mem.so` ❌ Currently missing
- `_madrona_escape_room_dlpack.cpython-*.so` ✅ DLPack extension

## Why This Matters

Without all libraries packaged:
- ✅ Development works (finds libraries in build/ directory)
- ❌ Distribution fails (missing dependencies when installed)
- ❌ Users get "library not found" errors
- ❌ ctypes.CDLL() fails to load main library due to missing dependencies

## Priority: HIGH
This is the final step to make ctypes bindings distributable. The implementation is correct, just the packaging is incomplete.

## Estimated Time: 15 minutes
Simple fix to copy all required libraries during build.