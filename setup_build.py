#!/usr/bin/env python3
"""
Build helper script to build C API wrapper library
"""

import os
import shutil
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
        subprocess.check_call(["/opt/cmake/bin/cmake", "-B", "build"])

    # Build the C wrapper library
    print("Building C wrapper library...")
    subprocess.check_call(
        ["make", "-C", "build", "madrona_escape_room_c_api", f"-j{os.cpu_count()}"]
    )


def copy_shared_libraries():
    """Copy shared libraries to package directory for ctypes access"""
    print("Copying shared libraries to package directory...")

    pkg_dir = Path("madrona_escape_room")
    build_dir = Path("build")

    # Required libraries for ctypes bindings
    required_libs = [
        "libembree4.so.4",
        "libdxcompiler.so",
        "libmadrona_render_shader_compiler.so",
        "libmadrona_std_mem.so",
    ]

    for lib_name in required_libs:
        lib_path = build_dir / lib_name
        if lib_path.exists():
            lib_dest = pkg_dir / lib_name
            print(f"Copying dependency {lib_name} to {lib_dest}")
            shutil.copy2(lib_path, lib_dest)
        else:
            print(f"Warning: {lib_name} not found in build directory")


if __name__ == "__main__":
    # Build everything
    build_c_wrapper()
    copy_shared_libraries()
    print("Build complete! ctypes bindings will access C API library directly.")
