#!/usr/bin/env python3
"""
Binary Level I/O for Madrona Escape Room

Handles reading and writing CompiledLevel structures to/from binary files.
Uses the C API for efficient binary serialization.
"""

import ctypes
from pathlib import Path
from typing import List, Union

from .dataclass_utils import create_compiled_level
from .generated_dataclasses import CompiledLevel


def save_compiled_levels(levels: List[CompiledLevel], filepath: Union[str, Path]) -> None:
    """
    Save list of CompiledLevel structs to binary file using unified format.

    For single levels, pass a list of one: save_compiled_levels([level], path)

    Args:
        levels: List of CompiledLevel structs to save
        filepath: Path to save file

    Raises:
        IOError: If file cannot be written
        ValueError: If levels list is empty
    """
    from .ctypes_bindings import lib
    from .generated_constants import Result

    if not levels:
        raise ValueError("Cannot save empty list of levels")

    filepath_str = str(filepath)

    # Convert all levels to ctypes structures
    c_levels = [level.to_ctype() for level in levels]

    # Create array type and populate it
    c_level_type = type(c_levels[0])
    c_levels_array = (c_level_type * len(levels))(*c_levels)

    # Call C API
    result = lib.mer_write_compiled_levels(
        filepath_str.encode("utf-8"), ctypes.cast(c_levels_array, ctypes.c_void_p), len(levels)
    )

    if result != Result.Success:
        raise IOError(f"Failed to write level file: {filepath_str} (error code: {result})")


def load_compiled_levels(filepath: Union[str, Path]) -> List[CompiledLevel]:
    """
    Load list of CompiledLevel structs from binary file using unified format.

    Returns list with one element for single-level files.

    Args:
        filepath: Path to level file

    Returns:
        List of CompiledLevel structs

    Raises:
        IOError: If file cannot be read
    """
    from .ctypes_bindings import lib
    from .generated_constants import Result

    filepath_str = str(filepath)

    # First pass: get count
    actual_count = ctypes.c_uint32(0)
    result = lib.mer_read_compiled_levels(
        filepath_str.encode("utf-8"),
        None,  # Just getting count
        ctypes.byref(actual_count),
        0,
    )

    if result != Result.Success:
        raise IOError(f"Failed to read level file: {filepath_str} (error code: {result})")

    num_levels = actual_count.value

    # Second pass: read levels
    if num_levels > 0:
        levels = [create_compiled_level() for _ in range(num_levels)]
        c_levels = [level.to_ctype() for level in levels]

        # Create array type
        c_level_type = type(c_levels[0])
        c_levels_array = (c_level_type * num_levels)(*c_levels)

        # Call C API
        result = lib.mer_read_compiled_levels(
            filepath_str.encode("utf-8"),
            ctypes.cast(c_levels_array, ctypes.c_void_p),
            ctypes.byref(actual_count),
            num_levels,
        )

        if result != Result.Success:
            raise IOError(f"Failed to read level data: {filepath_str} (error code: {result})")

        # Convert back to dataclasses
        compiled_levels = []
        for i in range(num_levels):
            compiled_level = CompiledLevel.from_ctype(c_levels_array[i])
            compiled_levels.append(compiled_level)

        return compiled_levels
    else:
        return []
