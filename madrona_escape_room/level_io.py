#!/usr/bin/env python3
"""
Binary Level I/O for Madrona Escape Room

Handles reading and writing CompiledLevel structures to/from binary files.
Uses the C API for efficient binary serialization.
"""

import ctypes
from pathlib import Path
from typing import Union

from .dataclass_utils import create_compiled_level
from .generated_dataclasses import CompiledLevel


def save_compiled_level(level: CompiledLevel, filepath: Union[str, Path]) -> None:
    """
    Save CompiledLevel struct to binary .lvl file using C API.

    Args:
        level: CompiledLevel struct to save
        filepath: Path to save .lvl file

    Raises:
        IOError: If file cannot be written
    """
    from .ctypes_bindings import lib
    from .generated_constants import Result

    filepath_str = str(filepath)
    c_level = level.to_ctype()
    result = lib.mer_write_compiled_level(filepath_str.encode("utf-8"), ctypes.byref(c_level))

    if result != Result.Success:
        raise IOError(f"Failed to write level file: {filepath_str} (error code: {result})")


def load_compiled_level(filepath: Union[str, Path]) -> CompiledLevel:
    """
    Load CompiledLevel struct from binary .lvl file using C API.

    Args:
        filepath: Path to .lvl file

    Returns:
        CompiledLevel struct

    Raises:
        IOError: If file cannot be read
    """
    from .ctypes_bindings import lib
    from .generated_constants import Result

    filepath_str = str(filepath)
    level = create_compiled_level()
    c_level = level.to_ctype()
    result = lib.mer_read_compiled_level(filepath_str.encode("utf-8"), ctypes.byref(c_level))

    if result != Result.Success:
        raise IOError(f"Failed to read level file: {filepath_str} (error code: {result})")

    # Convert back to dataclass
    return CompiledLevel.from_ctype(c_level)
