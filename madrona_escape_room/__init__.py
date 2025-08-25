"""
Madrona Escape Room Python bindings using ctypes
Provides the same API as the original nanobind version
"""

# Note: Library path setup is handled in ctypes_bindings.py where the library is actually loaded

# Import generated constants
# Import default level creator
from .default_level import create_default_level
from .generated_constants import ExecMode, Result, TensorElementType, action, consts

# Import SimManager from manager module
from .manager import SimManager

# Import Tensor from separate module
from .tensor import Tensor

# Define public API
__all__ = [
    # Main classes
    "SimManager",
    "Tensor",
    # Level creation
    "create_default_level",
    # Constants
    "ExecMode",
    "Result",
    "TensorElementType",
    "action",
    "consts",
]
