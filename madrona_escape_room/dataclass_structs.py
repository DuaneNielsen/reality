"""
Auto-generated Python dataclass structures from compiled binary.
Uses cdataclass for C compatibility with Pythonic interface.
DO NOT EDIT - this file is automatically regenerated.
"""

import ctypes
from dataclasses import dataclass, field
from typing import List, Tuple

from cdataclass import BigEndianCDataMixIn, NativeEndianCDataMixIn, meta

# Dataclass structures
# Size validation
# Import proper constants from generated_constants
from madrona_escape_room.generated_constants import limits

# Use the constant defined in consts.hpp
MAX_TILES = limits.maxTiles
MAX_SPAWNS = limits.maxSpawns


# Helper function to convert any dataclass to ctypes for C API
def to_ctypes(obj):
    """Convert dataclass to ctypes Structure for C API."""
    if hasattr(obj, "to_ctype"):
        return obj.to_ctype()
    return obj  # Already a ctypes structure
