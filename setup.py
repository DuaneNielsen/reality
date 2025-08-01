#!/usr/bin/env python3
"""
Setup script for Madrona Escape Room with ctypes bindings
"""

import os
import shutil
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from setuptools.command.install_lib import install_lib

class BuildPyWithLibrary(build_py):
    """Custom build_py that includes the C library and all dependencies"""
    def run(self):
        # First run the normal build_py
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

# DLPack extension configuration
dlpack_extension = Extension(
    "_madrona_escape_room_dlpack",
    sources=["src/dlpack_extension.cpp"],
    language="c++",
    extra_compile_args=[
        "-std=c++17",
        "-O3",
        "-fPIC", 
        "-Wall",
        "-Wextra",
    ],
    extra_link_args=[],
)

setup(
    name="madrona-escape-room",
    version=version,
    description="Madrona Escape Room - High-performance 3D RL environment",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/madrona_escape_room",
    packages=find_packages(exclude=["tests", "tests.*", "external", "external.*", "scripts"]) + find_packages(where="train_src"),
    ext_modules=[dlpack_extension],
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
            "libmadrona_std_mem.so"
        ],
    },
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "torchrl>=0.2.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-xdist>=3.0.0",
        ],
    },
    cmdclass={
        "build_py": BuildPyWithLibrary,
        "install_lib": InstallLibWithLibrary,
    },
    zip_safe=False,
)