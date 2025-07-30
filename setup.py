#!/usr/bin/env python3
"""
Setup script for Madrona Escape Room with CFFI bindings
"""

import os
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install_lib import install_lib

class BuildPyWithLibrary(build_py):
    """Custom build_py that includes the C library"""
    def run(self):
        # First run the normal build_py
        super().run()
        
        # Copy the C library to the package
        c_lib = Path("build/libmadrona_escape_room_c_api.so")
        if c_lib.exists():
            # Copy to each package in the build directory
            for package in self.packages:
                if package == "madrona_escape_room":
                    package_dir = Path(self.build_lib) / package
                    dest = package_dir / c_lib.name
                    print(f"Copying C library to {dest}")
                    shutil.copy2(c_lib, dest)


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

setup(
    name="madrona-escape-room",
    version=version,
    description="Madrona Escape Room - High-performance 3D RL environment",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/madrona_escape_room",
    packages=find_packages(exclude=["tests", "tests.*", "external", "external.*", "scripts"]) + find_packages(where="train_src"),
    package_dir={"madrona_escape_room_learn": "train_src/madrona_escape_room_learn"},
    package_data={
        "madrona_escape_room": [
            "*.pyi", 
            "*.so", 
            "libmadrona_escape_room_c_api.so", 
            "_madrona_escape_room_cffi*.so",
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
        "cffi>=1.15.0",
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