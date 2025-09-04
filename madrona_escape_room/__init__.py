"""
Madrona Escape Room Python bindings using ctypes
Provides the same API as the original nanobind version
"""

# Note: Library path setup is handled in ctypes_bindings.py where the library is actually loaded

# Import generated constants
# Import default level creator
# Import sensor_config module as namespace
from . import sensor_config
from .default_level import create_default_level
from .generated_constants import ExecMode, RenderMode, Result, TensorElementType, action, consts

# Import SimManager and factory function from manager module
from .manager import SimManager, create_sim_manager

# Import SensorConfig class
from .sensor_config import SensorConfig

# Import Tensor from separate module
from .tensor import Tensor

# Define public API
__all__ = [
    # Main classes
    "SimManager",
    "Tensor",
    "SensorConfig",
    # Factory functions
    "create_sim_manager",
    "create_default_level",
    # Sensor config namespace
    "sensor_config",
    # Constants
    "ExecMode",
    "Result",
    "RenderMode",
    "TensorElementType",
    "action",
    "consts",
]
