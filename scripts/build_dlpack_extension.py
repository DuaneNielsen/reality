#!/usr/bin/env python3
"""
Build script for the DLPack C extension module
"""

from pathlib import Path

from setuptools import Extension, setup


def main():
    # Get paths
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"

    # DLPack extension configuration
    dlpack_extension = Extension(
        "_madrona_escape_room_dlpack",
        sources=[str(src_dir / "dlpack_extension.cpp")],
        language="c++",
        include_dirs=[],
        libraries=[],
        library_dirs=[],
        define_macros=[],
        extra_compile_args=[
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-Wall",
            "-Wextra",
        ],
        extra_link_args=[],
    )

    # Setup configuration
    setup(
        name="_madrona_escape_room_dlpack",
        ext_modules=[dlpack_extension],
        zip_safe=False,
        python_requires=">=3.8",
    )


if __name__ == "__main__":
    main()
