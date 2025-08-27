"""
Utility functions for creating properly initialized dataclass instances.
"""

from .generated_dataclasses import CompiledLevel


def create_compiled_level() -> CompiledLevel:
    """
    Create a CompiledLevel instance with proper default values.
    
    The main issue this solves is that scale values must default to 1.0, not 0.0.
    A scale of (0, 0, 0) is invalid and causes assertion failures in the physics engine.
    
    Returns:
        CompiledLevel with properly initialized arrays:
        - tile_scale_x/y/z arrays initialized to 1.0 (not 0.0)
        - All other arrays use their normal defaults
    """
    level = CompiledLevel()
    
    # Fix the scale arrays - they MUST default to 1.0, not 0.0
    # A scale of 0 is invalid and causes physics assertions after ~177 iterations
    level.tile_scale_x = [1.0] * 1024
    level.tile_scale_y = [1.0] * 1024
    level.tile_scale_z = [1.0] * 1024
    
    return level