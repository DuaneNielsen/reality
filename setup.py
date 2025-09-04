#!/usr/bin/env python3
"""
Setup script for Madrona Escape Room with ctypes bindings
"""

import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install_lib import install_lib


# Helper function to ensure libraries are in package directory
def ensure_libraries_present():
    """Ensure all required libraries are in the package directory."""
    required_libs = [
        "libmadrona_escape_room_c_api.so",  # Main C API (ctypes loads this)
        "libembree4.so.4",  # Embree ray tracing
        "libdxcompiler.so",  # DirectX shader compiler
        "libmadrona_render_shader_compiler.so",  # Madrona rendering
        "libmadrona_std_mem.so",  # Madrona memory management
        "_madrona_escape_room_dlpack.cpython-312-x86_64-linux-gnu.so",  # DLPack extension
    ]

    build_dir = Path("build")
    package_dir = Path("madrona_escape_room")

    missing_libs = []
    for lib_name in required_libs:
        dest_path = package_dir / lib_name
        if not dest_path.exists():
            # Try to copy from build directory
            src_path = build_dir / lib_name
            if src_path.exists():
                print(f"Copying {lib_name} to package directory")
                shutil.copy2(src_path, dest_path)
            else:
                missing_libs.append(lib_name)

    if missing_libs:
        print(f"Warning: Missing libraries: {missing_libs}")
        print("Run 'make' in build directory first")
    return len(missing_libs) == 0


class BuildPyWithLibrary(build_py):
    """Custom build_py that ensures libraries are present and copies them"""

    def run(self):
        # First ensure libraries are in source directory
        ensure_libraries_present()

        # Run normal build_py
        super().run()

        # For regular installs, also copy to build_lib
        if self.build_lib:
            required_libs = [
                "libmadrona_escape_room_c_api.so",
                "libembree4.so.4",
                "libdxcompiler.so",
                "libmadrona_render_shader_compiler.so",
                "libmadrona_std_mem.so",
                "_madrona_escape_room_dlpack.cpython-312-x86_64-linux-gnu.so",
            ]

            src_dir = Path("madrona_escape_room")
            dest_dir = Path(self.build_lib) / "madrona_escape_room"
            dest_dir.mkdir(parents=True, exist_ok=True)

            for lib_name in required_libs:
                src_path = src_dir / lib_name
                if src_path.exists():
                    dest_path = dest_dir / lib_name
                    print(f"Copying {lib_name} to build directory")
                    shutil.copy2(src_path, dest_path)


class DevelopWithLibrary(develop):
    """Custom develop command for editable installs"""

    def run(self):
        ensure_libraries_present()
        super().run()


class InstallLibWithLibrary(install_lib):
    """Custom install_lib that ensures library is installed"""

    def install(self):
        # First run normal install
        outfiles = super().install()

        # The library should already be in the build directory
        # Just make sure it gets installed
        return outfiles


# Read the current version from pyproject.toml
version = "0.1.2"  # Incremented version

# Ensure libraries are present before setup
ensure_libraries_present()

setup(
    name="madrona-escape-room",
    version=version,
    description="Madrona Escape Room - High-performance 3D RL environment",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/madrona_escape_room",
    packages=find_packages(exclude=["tests", "tests.*", "external", "external.*", "scripts"])
    + find_packages(where="train_src"),
    ext_modules=[],  # All extensions are pre-built by CMake
    package_dir={"madrona_escape_room_learn": "train_src/madrona_escape_room_learn"},
    package_data={
        "madrona_escape_room": [
            "*.pyi",
            "*.so",
            "libmadrona_escape_room_c_api.so",
            "_madrona_escape_room_dlpack*.so",
            "libembree4.so.4",
            "libdxcompiler.so",
            "libmadrona_render_shader_compiler.so",
            "libmadrona_std_mem.so",
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy>=1.20.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-xdist>=3.0.0",
        ],
    },
    cmdclass={
        "build_py": BuildPyWithLibrary,
        "develop": DevelopWithLibrary,
        "install_lib": InstallLibWithLibrary,
    },
    zip_safe=False,
)
