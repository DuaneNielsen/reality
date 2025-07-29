#!/usr/bin/env python3
"""
Build helper script to ensure CFFI module is built before installation
"""

import os
import sys
import subprocess
from pathlib import Path


def build_c_wrapper():
    """Build the C wrapper library"""
    build_dir = Path("build")
    if not build_dir.exists():
        print("Creating build directory...")
        build_dir.mkdir()
    
    # Run cmake if not already configured
    if not (build_dir / "CMakeCache.txt").exists():
        print("Running cmake...")
        subprocess.check_call([
            "/opt/cmake/bin/cmake", "-B", "build"
        ])
    
    # Build the C wrapper library
    print("Building C wrapper library...")
    subprocess.check_call([
        "make", "-C", "build", "madrona_escape_room_c_api", f"-j{os.cpu_count()}"
    ])


def build_cffi_module():
    """Build the CFFI module in the package directory"""
    print("Building CFFI module...")
    
    # Import and run the build script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scripts.build_cffi import build_cffi
    
    # Build CFFI module
    cffi_tmp = build_cffi()
    
    # Copy the built module to the package directory
    import shutil
    import glob
    
    built_modules = glob.glob(str(cffi_tmp / "_madrona_escape_room_cffi*.so"))
    if built_modules:
        pkg_dir = Path("madrona_escape_room")
        for module in built_modules:
            dest = pkg_dir / Path(module).name
            print(f"Copying {module} to {dest}")
            shutil.copy2(module, dest)
            
            # Also copy the C library to the package directory
            c_lib = Path("build/libmadrona_escape_room_c_api.so")
            if c_lib.exists():
                lib_dest = pkg_dir / c_lib.name
                print(f"Copying C library to {lib_dest}")
                shutil.copy2(c_lib, lib_dest)
                
            # Copy all required shared libraries
            required_libs = [
                "libembree4.so.4",
                "libdxcompiler.so", 
                "libmadrona_render_shader_compiler.so",
                "libmadrona_std_mem.so"
            ]
            
            for lib_name in required_libs:
                lib_path = Path("build") / lib_name
                if lib_path.exists():
                    lib_dest = pkg_dir / lib_name
                    print(f"Copying dependency {lib_name} to {lib_dest}")
                    shutil.copy2(lib_path, lib_dest)
            
            # On Linux, update the RPATH to find the library in the same directory
            if sys.platform == "linux":
                try:
                    # Try to set rpath using chrpath or patchelf if available
                    subprocess.run(["chrpath", "-r", "$ORIGIN", str(dest)], 
                                 capture_output=True)
                except FileNotFoundError:
                    # If chrpath not available, try patchelf
                    try:
                        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(dest)],
                                     capture_output=True)
                    except FileNotFoundError:
                        # Neither tool available, library will need to be in same directory
                        pass


if __name__ == "__main__":
    # Build everything
    build_c_wrapper()
    build_cffi_module()
    print("Build complete!")