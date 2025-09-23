#!/usr/bin/env python3
"""
JSON Level Compiler for Madrona Escape Room

Compiles JSON level definitions to CompiledLevel format for GPU processing.
This compiler exclusively accepts JSON format with explicit tileset definitions.

Required JSON format (supports single and multi-level formats):

Option 1 - Single level with array of strings (recommended):
{
    "ascii": [
        "########",
        "#S....#",
        "########"
    ],
    "tileset": {
        "#": {"asset": "wall"},
        "C": {"asset": "cube"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,              # Optional, default 2.5
    "agent_facing": [0.0],      # Optional, radians for each agent
    "spawn_random": false,      # Optional, use random spawn positions instead of fixed
    "name": "level_name",       # Optional, default "unknown_level"
    "targets": [               # Optional, list of moving target entities
        {
            "position": [x, y, z],  # Required, target initial position
            "motion_type": "static", # Required, "static" or "figure8"
        },
        {
            "position": [x, y, z],   # Required, target initial position
            "motion_type": "figure8", # Required, "static" or "figure8"
            "params": {              # Required for figure8 motion
                "omega_x": 1.0,      # Speed control (rad/s)
                "omega_y": 6.0,      # Unused (legacy)
                "center": [x, y, z], # Center point of figure-8
                "mass": 1.0,         # Unused (legacy)
                "phase_x": 10.0,     # X-axis amplitude
                "phase_y": 10.0      # Y-axis amplitude
            }
        }
    ]
}

Option 2 - Multi-level format with shared tileset:
{
    "levels": [
        {
            "ascii": [
                "########",
                "#S....#",
                "########"
            ],
            "name": "level_1",     # Optional per-level name
            "agent_facing": [0.0]  # Optional per-level agent facing
        },
        {
            "ascii": [
                "##########",
                "#S......C#",
                "##########"
            ],
            "name": "level_2"
        }
    ],
    "tileset": {               # Shared tileset for all levels
        "#": {"asset": "wall"},
        "C": {"asset": "cube"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,              # Optional, default 2.5 (applies to all levels)
    "spawn_random": false,      # Optional, use random spawn positions for all levels
    "name": "multi_level_set"   # Optional name for the level set
}

Option 3 - Multi-level format with per-level tilesets:
{
    "levels": [
        {
            "ascii": [
                "########",
                "#S....#",
                "########"
            ],
            "tileset": {       # Per-level tileset
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "name": "simple_level"
        },
        {
            "ascii": [
                "##########",
                "#S......C#",
                "##########"
            ],
            "tileset": {       # Different tileset for this level
                "#": {"asset": "wall", "done_on_collision": true},
                "C": {"asset": "cube"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "name": "complex_level"
        }
    ],
    "scale": 2.5,              # Optional shared settings
    "name": "mixed_level_set"
}

Option 4 - String with newlines (backwards compatible):
{
    "ascii": "########\\n#S....#\\n########",
    "tileset": { ... },
    ...
}
"""

import argparse
import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .dataclass_utils import create_compiled_level

# Get constants from generated constants
from .generated_constants import limits

# Get the CompiledLevel struct
from .generated_dataclasses import CompiledLevel

# Constants
MAX_TILES = limits.maxTiles
MAX_SPAWNS = limits.maxSpawns
MAX_TARGETS = limits.maxTargets
MAX_GRID_SIZE = limits.maxGridSize
MAX_LEVEL_NAME_LENGTH = limits.maxLevelNameLength

# Backwards compatibility aliases
MAX_TILES_C_API = MAX_TILES

# Special tile type values
TILE_EMPTY = 0
TILE_SPAWN = -1

# Level dimension limits
MIN_LEVEL_WIDTH = 3
MIN_LEVEL_HEIGHT = 3

# Default tileset as JSON string - can be easily modified/extended
DEFAULT_TILESET_JSON = """
{
    "#": {"asset": "wall", "done_on_collision": false},
    "C": {"asset": "cube", "done_on_collision": true},
    "O": {"asset": "cylinder", "done_on_collision": true},
    "S": {"asset": "spawn"},
    ".": {"asset": "empty"},
    " ": {"asset": "empty"}
}
"""

# Parse the default tileset at module load time
DEFAULT_TILESET = json.loads(DEFAULT_TILESET_JSON)


def _get_asset_object_id(asset_name: str) -> int:
    """
    Get object ID for an asset by name using C API functions.

    Args:
        asset_name: Name of the asset (e.g., "wall", "cube", "cylinder")

    Returns:
        Object ID for the asset, or special values for spawn/empty

    Raises:
        ValueError: If asset name is not found
    """
    # Special cases
    if asset_name == "spawn":
        return TILE_SPAWN
    if asset_name == "empty":
        return TILE_EMPTY

    # Use C API to get correct asset IDs
    from .ctypes_bindings import get_physics_asset_object_id, get_render_asset_object_id

    # Try physics assets first
    obj_id = get_physics_asset_object_id(asset_name)
    if obj_id >= 0:
        return obj_id

    # Try render-only assets
    obj_id = get_render_asset_object_id(asset_name)
    if obj_id >= 0:
        return obj_id

    raise ValueError(f"Unknown asset name: '{asset_name}'")


def _normalize_ascii_input(ascii_data: Union[str, List[str]]) -> str:
    """
    Normalize ASCII input to a string format.

    Args:
        ascii_data: Either a string with newlines or array of strings

    Returns:
        String with newline characters
    """
    if isinstance(ascii_data, list):
        return "\n".join(ascii_data)
    else:
        return ascii_data


def _validate_tileset(tileset: Dict) -> None:
    """
    Validate a tileset definition.

    Args:
        tileset: Dictionary mapping characters to asset definitions

    Raises:
        ValueError: If tileset is invalid
    """
    if not isinstance(tileset, dict):
        raise ValueError("Tileset must be a dictionary")

    if not tileset:
        raise ValueError("Tileset cannot be empty")

    for char, tile_def in tileset.items():
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError(f"Tileset key must be single character, got: '{char}'")

        if not isinstance(tile_def, dict):
            raise ValueError(f"Tileset value for '{char}' must be a dictionary")

        if "asset" not in tile_def:
            raise ValueError(f"Tileset entry for '{char}' must have 'asset' field")

        if not isinstance(tile_def["asset"], str):
            raise ValueError(f"Asset name for '{char}' must be a string")

        # Validate optional randomization parameters
        for rand_field in ["rand_x", "rand_y", "rand_z", "rand_rot_z"]:
            if rand_field in tile_def:
                value = tile_def[rand_field]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{rand_field} for '{char}' must be a number, got {type(value)}"
                    )
                if value < 0:
                    raise ValueError(f"{rand_field} for '{char}' must be non-negative, got {value}")

        # Validate optional done_on_collision flag
        if "done_on_collision" in tile_def:
            value = tile_def["done_on_collision"]
            if not isinstance(value, bool):
                raise ValueError(
                    f"done_on_collision for '{char}' must be a boolean, got {type(value)}"
                )


def _validate_targets(targets: List[Dict]) -> None:
    """
    Validate targets array in JSON level definition.

    Args:
        targets: List of target dictionaries

    Raises:
        ValueError: If targets array is invalid
    """
    if not isinstance(targets, list):
        raise ValueError("'targets' field must be a list")

    if len(targets) > MAX_TARGETS:
        raise ValueError(f"Too many targets: {len(targets)} > {MAX_TARGETS} max")

    for i, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"Target {i} must be a dictionary")

        # Required fields
        if "position" not in target:
            raise ValueError(f"Target {i} must have 'position' field")

        position = target["position"]
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f"Target {i} 'position' must be [x, y, z] array")

        for j, coord in enumerate(position):
            if not isinstance(coord, (int, float)):
                raise ValueError(f"Target {i} position[{j}] must be a number")

        if "motion_type" not in target:
            raise ValueError(f"Target {i} must have 'motion_type' field")

        motion_type = target["motion_type"]
        if not isinstance(motion_type, str):
            raise ValueError(f"Target {i} 'motion_type' must be a string")

        # Validate motion types
        valid_motion_types = ["static", "figure8", "circular"]
        if motion_type not in valid_motion_types:
            raise ValueError(
                f"Target {i} invalid motion_type '{motion_type}', "
                f"must be one of: {valid_motion_types}"
            )

        # Validate parameters based on motion type
        if motion_type == "figure8":
            if "params" not in target:
                raise ValueError(f"Target {i} with motion_type 'figure8' must have 'params' field")

            params = target["params"]
            if not isinstance(params, dict):
                raise ValueError(f"Target {i} 'params' must be a dictionary")

            required_params = ["omega_x", "omega_y", "center", "mass"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Target {i} figure8 params missing '{param}' field")

            # Validate numeric parameters
            for param in ["omega_x", "omega_y", "mass"]:
                value = params[param]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Target {i} params['{param}'] must be a number")

            # Validate optional phase parameters (default to 0.0 if not provided)
            for param in ["phase_x", "phase_y"]:
                if param in params:
                    value = params[param]
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Target {i} params['{param}'] must be a number")

            # Validate center array
            center = params["center"]
            if not isinstance(center, list) or len(center) != 3:
                raise ValueError(f"Target {i} params['center'] must be [x, y, z] array")

            for j, coord in enumerate(center):
                if not isinstance(coord, (int, float)):
                    raise ValueError(f"Target {i} params center[{j}] must be a number")

        elif motion_type == "circular":
            if "params" not in target:
                raise ValueError(f"Target {i} with motion_type 'circular' must have 'params' field")

            params = target["params"]
            if not isinstance(params, dict):
                raise ValueError(f"Target {i} 'params' must be a dictionary")

            required_params = ["angular_velocity", "center", "radius"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"Target {i} circular params missing '{param}' field")

            # Validate required numeric parameters
            for param in ["angular_velocity", "radius"]:
                value = params[param]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Target {i} params['{param}'] must be a number")
                if param == "radius" and value <= 0:
                    raise ValueError(f"Target {i} params['radius'] must be positive")
                if param == "angular_velocity" and value == 0:
                    raise ValueError(f"Target {i} params['angular_velocity'] must be non-zero")

            # Validate optional parameters
            for param in ["randomize", "direction", "initial_angle"]:
                if param in params:
                    value = params[param]
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Target {i} params['{param}'] must be a number")
                    if param == "randomize" and value not in [0.0, 1.0]:
                        raise ValueError(f"Target {i} params['randomize'] must be 0.0 or 1.0")
                    if param == "direction" and value not in [-1.0, 1.0]:
                        raise ValueError(
                            f"Target {i} params['direction'] must be -1.0 (clockwise) "
                            f"or 1.0 (counter-clockwise)"
                        )

            # Validate center array
            center = params["center"]
            if not isinstance(center, list) or len(center) != 3:
                raise ValueError(f"Target {i} params['center'] must be [x, y, z] array")

            for j, coord in enumerate(center):
                if not isinstance(coord, (int, float)):
                    raise ValueError(f"Target {i} params center[{j}] must be a number")


def _validate_multi_level_json(data: Dict) -> None:
    """
    Validate multi-level JSON data structure.

    Args:
        data: JSON multi-level dictionary

    Raises:
        ValueError: If JSON structure is invalid
    """
    # Required fields for multi-level format
    if "levels" not in data:
        raise ValueError("Multi-level JSON must contain 'levels' field")

    # Check that we have either a global tileset or all levels have their own tilesets
    has_global_tileset = "tileset" in data

    # Validate levels array
    levels = data["levels"]
    if not isinstance(levels, list) or not levels:
        raise ValueError("'levels' field must be a non-empty list")

    # Check each level has a tileset (either its own or there's a global one)
    for i, level in enumerate(levels):
        if not isinstance(level, dict):
            raise ValueError(f"Level {i} must be a dictionary")

        level_has_tileset = "tileset" in level
        if not level_has_tileset and not has_global_tileset:
            raise ValueError(
                f"Level {i} must have a 'tileset' field since no global tileset is provided"
            )

        # Validate per-level tileset if present
        if level_has_tileset:
            try:
                _validate_tileset(level["tileset"])
            except ValueError as e:
                raise ValueError(f"Level {i} tileset validation failed: {e}")

        # Validate each level
        if not isinstance(level, dict):
            raise ValueError(f"Level {i} must be a dictionary")

        if "ascii" not in level:
            raise ValueError(f"Level {i} must contain 'ascii' field")

        # Validate ascii field (same as single level)
        ascii_data = level["ascii"]
        if isinstance(ascii_data, list):
            if not ascii_data:
                raise ValueError(f"Level {i} 'ascii' array cannot be empty")
            for j, line in enumerate(ascii_data):
                if not isinstance(line, str):
                    raise ValueError(f"Level {i} 'ascii' array line {j} must be a string")
        elif isinstance(ascii_data, str):
            if not ascii_data.strip():
                raise ValueError(f"Level {i} 'ascii' field cannot be empty")
        else:
            raise ValueError(f"Level {i} 'ascii' field must be a string or array of strings")

        # Validate optional per-level fields
        if "agent_facing" in level:
            agent_facing = level["agent_facing"]
            if not isinstance(agent_facing, list):
                raise ValueError(f"Level {i} 'agent_facing' must be a list of angles in radians")
            for j, angle in enumerate(agent_facing):
                if not isinstance(angle, (int, float)):
                    raise ValueError(f"Level {i} agent_facing[{j}]: {angle} (must be number)")

        if "name" in level:
            name = level["name"]
            if not isinstance(name, str):
                raise ValueError(f"Level {i} 'name' field must be a string")
            if len(name) > MAX_LEVEL_NAME_LENGTH:
                raise ValueError(f"Level {i} name too long: {len(name)} > {MAX_LEVEL_NAME_LENGTH}")

        # Validate per-level targets (optional)
        if "targets" in level:
            try:
                _validate_targets(level["targets"])
            except ValueError as e:
                raise ValueError(f"Level {i} targets validation failed: {e}")

    # Validate shared tileset if present
    if has_global_tileset:
        _validate_tileset(data["tileset"])

    # Validate optional shared fields
    if "scale" in data:
        scale = data["scale"]
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"Invalid scale: {scale} (must be positive number)")

    if "spawn_random" in data:
        spawn_random = data["spawn_random"]
        if not isinstance(spawn_random, bool):
            raise ValueError(f"Invalid spawn_random: {spawn_random} (must be boolean)")

    if "auto_boundary_walls" in data:
        auto_boundary_walls = data["auto_boundary_walls"]
        if not isinstance(auto_boundary_walls, bool):
            raise ValueError(
                f"Invalid auto_boundary_walls: {auto_boundary_walls} (must be boolean)"
            )

    if "boundary_wall_offset" in data:
        boundary_wall_offset = data["boundary_wall_offset"]
        if not isinstance(boundary_wall_offset, (int, float)) or boundary_wall_offset < 0:
            raise ValueError(
                f"Invalid boundary_wall_offset: {boundary_wall_offset} "
                "(must be non-negative number)"
            )

    if "name" in data:
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError("'name' field must be a string")

    # Validate shared targets (optional)
    if "targets" in data:
        _validate_targets(data["targets"])


def _validate_json_level(data: Dict) -> None:
    """
    Validate JSON level data structure.

    Args:
        data: JSON level dictionary

    Raises:
        ValueError: If JSON structure is invalid
    """
    # Check if this is multi-level format
    if "levels" in data:
        _validate_multi_level_json(data)
        return

    # Required fields for single level
    if "ascii" not in data:
        raise ValueError("JSON level must contain 'ascii' field")

    if "tileset" not in data:
        raise ValueError("JSON level must contain 'tileset' field")

    # Validate ascii field - support both string and array formats
    ascii_data = data["ascii"]
    if isinstance(ascii_data, list):
        # Array format - validate each line is a string
        if not ascii_data:
            raise ValueError("'ascii' array cannot be empty")
        for i, line in enumerate(ascii_data):
            if not isinstance(line, str):
                raise ValueError(f"'ascii' array line {i} must be a string, got {type(line)}")
    elif isinstance(ascii_data, str):
        # String format - must not be empty after stripping
        if not ascii_data.strip():
            raise ValueError("'ascii' field cannot be empty")
    else:
        raise ValueError("'ascii' field must be a string or array of strings")

    # Validate tileset
    _validate_tileset(data["tileset"])

    # Validate optional fields
    if "scale" in data:
        scale = data["scale"]
        if not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"Invalid scale: {scale} (must be positive number)")

    if "agent_facing" in data:
        agent_facing = data["agent_facing"]
        if not isinstance(agent_facing, list):
            raise ValueError("'agent_facing' must be a list of angles in radians")
        for i, angle in enumerate(agent_facing):
            if not isinstance(angle, (int, float)):
                raise ValueError(f"Invalid agent_facing[{i}]: {angle} (must be number)")

    if "spawn_random" in data:
        spawn_random = data["spawn_random"]
        if not isinstance(spawn_random, bool):
            raise ValueError(f"Invalid spawn_random: {spawn_random} (must be boolean)")

    if "auto_boundary_walls" in data:
        auto_boundary_walls = data["auto_boundary_walls"]
        if not isinstance(auto_boundary_walls, bool):
            raise ValueError(
                f"Invalid auto_boundary_walls: {auto_boundary_walls} (must be boolean)"
            )

    if "boundary_wall_offset" in data:
        boundary_wall_offset = data["boundary_wall_offset"]
        if not isinstance(boundary_wall_offset, (int, float)) or boundary_wall_offset < 0:
            raise ValueError(
                f"Invalid boundary_wall_offset: {boundary_wall_offset} "
                "(must be non-negative number)"
            )

    if "name" in data:
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError("'name' field must be a string")
        if len(name) > MAX_LEVEL_NAME_LENGTH:
            raise ValueError(f"Level name too long: {len(name)} > {MAX_LEVEL_NAME_LENGTH}")

    # Validate targets field (optional)
    if "targets" in data:
        _validate_targets(data["targets"])


def _process_tileset(tileset: Dict) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Process tileset to extract object IDs and tile properties.

    Args:
        tileset: Validated tileset dictionary

    Returns:
        Tuple of (char_to_tile mapping, char_to_properties dictionary)

    Raises:
        ValueError: If asset cannot be resolved
    """
    char_to_tile = {}
    char_to_props = {}

    for char, tile_def in tileset.items():
        asset_name = tile_def["asset"]

        # Get object ID for this asset
        try:
            char_to_tile[char] = _get_asset_object_id(asset_name)
        except ValueError as e:
            raise ValueError(f"Invalid asset '{asset_name}' for character '{char}': {e}")

        # Extract tile properties including randomization and collision
        char_to_props[char] = {
            "rand_x": tile_def.get("rand_x", 0.0),
            "rand_y": tile_def.get("rand_y", 0.0),
            "rand_z": tile_def.get("rand_z", 0.0),
            "rand_rot_z": tile_def.get("rand_rot_z", 0.0),
            "done_on_collision": tile_def.get("done_on_collision", False),
        }

    return char_to_tile, char_to_props


def _parse_ascii_to_tiles(
    ascii_str: str,
    char_to_tile: Dict[str, int],
    char_to_props: Dict[str, Dict],
    scale: float,
    width: int,
    height: int,
) -> Tuple[List[Tuple], List[Tuple], int]:
    """
    Parse ASCII string to tile and spawn data.

    Args:
        ascii_str: ASCII level string
        char_to_tile: Character to object ID mapping
        char_to_props: Character to tile properties (randomization, collision, etc.)
        scale: World units per tile
        width: Level width in tiles
        height: Level height in tiles

    Returns:
        Tuple of (tiles list, spawns list, entity_count)

    Raises:
        ValueError: If unknown character found
    """
    lines = [line.rstrip() for line in ascii_str.strip().split("\n")]

    tiles = []
    spawns = []
    entity_count = 0

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char not in char_to_tile:
                raise ValueError(f"Unknown character '{char}' at position ({x}, {y})")

            tile_type = char_to_tile[char]

            # Convert grid coordinates to world coordinates
            # Y is inverted because ASCII Y=0 is at top, but world Y+ is up
            world_x = (x - width / 2.0 + 0.5) * scale
            world_y = -(y - height / 2.0 + 0.5) * scale

            if tile_type == TILE_SPAWN:
                spawns.append((world_x, world_y))
            elif tile_type != TILE_EMPTY:
                tiles.append((world_x, world_y, tile_type, char))
                entity_count += 1

    return tiles, spawns, entity_count


def compile_ascii_level(
    ascii_str: str,
    scale: float = 2.5,
    agent_facing: Optional[List[float]] = None,
    level_name: str = "unknown_level",
) -> CompiledLevel:
    """
    Compile ASCII level string using default tileset.

    This is a convenience wrapper that uses a standard tileset for common ASCII characters.
    For custom tilesets, use compile_level() directly with JSON format.

    Args:
        ascii_str: Multi-line ASCII level string
        scale: World units per tile (default 2.5)
        agent_facing: List of agent facing angles in radians (optional)
        level_name: Name of the level (default "unknown_level")

    Returns:
        CompiledLevel struct ready for C API

    Raises:
        ValueError: If ASCII contains unknown characters or compilation fails

    Example:
        level = '''
        ##########
        #S.......#
        #..####..#
        #........#
        ##########
        '''
        compiled = compile_ascii_level(level, scale=2.5, agent_facing=[math.pi/2])
    """
    # Default tileset for common ASCII characters
    default_tileset = {
        "#": {"asset": "wall", "done_on_collision": False},
        "C": {"asset": "cube", "done_on_collision": True},
        "O": {"asset": "cylinder", "done_on_collision": True},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"},
        " ": {"asset": "empty"},
    }

    # Build JSON structure
    json_data = {"ascii": ascii_str, "tileset": default_tileset, "scale": scale, "name": level_name}

    # Add agent facing if provided
    if agent_facing is not None:
        json_data["agent_facing"] = agent_facing

    # Compile using the main compile_level function and extract single level
    compiled_levels = compile_level(json_data)
    return compiled_levels[0]  # Always single level for this function


def _add_boundary_walls(
    level: "CompiledLevel",
    start_index: int,
    width: int,
    height: int,
    scale: float,
    boundary_wall_offset: float = 0.0,
) -> int:
    """
    Add 4 boundary walls and 4 corner blocks around the level perimeter.

    Args:
        level: CompiledLevel to add walls to
        start_index: Starting tile index for boundary walls
        width: Level width in tiles
        height: Level height in tiles
        scale: World units per tile
        boundary_wall_offset: Additional offset distance beyond calculated world boundaries
            (default 0.0)

    Returns:
        Number of boundary tiles added (4 walls + 4 corners = 8)

    Raises:
        ValueError: If not enough space for boundary walls and corners
    """
    from .ctypes_bindings import get_physics_asset_object_id

    # Check if we have space for 4 boundary walls + 4 corner blocks
    if start_index + 8 > MAX_TILES:
        raise ValueError(
            f"Not enough space for boundary walls and corners: "
            f"need {start_index + 8} tiles but max is {MAX_TILES}"
        )

    wall_id = get_physics_asset_object_id("wall")
    wall_thickness = 1.0  # Wall thickness in world units
    wall_height = 2.0  # Standard wall height

    # Calculate world boundaries
    world_width = width * scale
    world_height = height * scale
    world_min_x = level.world_min_x
    world_max_x = level.world_max_x
    world_min_y = level.world_min_y
    world_max_y = level.world_max_y

    # Position walls so their inner edge aligns with world boundaries + offset
    # Wall center should be offset by half the wall thickness + boundary_wall_offset
    # outside the boundaries
    wall_half_thickness = wall_thickness / 2.0
    total_offset = wall_half_thickness + boundary_wall_offset

    # Walls need to be extended by 2 * boundary_wall_offset to span the full level boundaries
    extended_width = world_width + (2.0 * boundary_wall_offset)
    extended_height = world_height + (2.0 * boundary_wall_offset)

    # Boundary wall configurations: (position_x, position_y, scale_x, scale_y, scale_z)
    boundary_walls = [
        # North wall (top): spans extended width, inner edge at max_y + boundary_wall_offset
        (0.0, world_max_y + total_offset, extended_width, wall_thickness, wall_height),
        # South wall (bottom): spans extended width, inner edge at min_y - boundary_wall_offset
        (0.0, world_min_y - total_offset, extended_width, wall_thickness, wall_height),
        # East wall (right): spans extended height, inner edge at max_x + boundary_wall_offset
        (world_max_x + total_offset, 0.0, wall_thickness, extended_height, wall_height),
        # West wall (left): spans extended height, inner edge at min_x - boundary_wall_offset
        (world_min_x - total_offset, 0.0, wall_thickness, extended_height, wall_height),
    ]

    # Add each boundary wall
    for i, (pos_x, pos_y, scale_x, scale_y, scale_z) in enumerate(boundary_walls):
        tile_idx = start_index + i

        # Set tile properties
        level.object_ids[tile_idx] = wall_id
        level.tile_x[tile_idx] = pos_x
        level.tile_y[tile_idx] = pos_y
        level.tile_z[tile_idx] = 0.0
        level.tile_scale_x[tile_idx] = scale_x
        level.tile_scale_y[tile_idx] = scale_y
        level.tile_scale_z[tile_idx] = scale_z
        level.tile_persistent[tile_idx] = True  # Boundary walls persist across episodes
        level.tile_render_only[tile_idx] = False  # Boundary walls have physics
        level.tile_done_on_collide[tile_idx] = False  # Walls don't end episode
        level.tile_entity_type[tile_idx] = 2  # EntityType::Wall
        level.tile_response_type[tile_idx] = 2  # ResponseType::Static
        level.tile_rotation[tile_idx] = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion

        # No randomization for boundary walls (they need to be precise)
        level.tile_rand_x[tile_idx] = 0.0
        level.tile_rand_y[tile_idx] = 0.0
        level.tile_rand_z[tile_idx] = 0.0
        level.tile_rand_rot_z[tile_idx] = 0.0
        level.tile_rand_scale_x[tile_idx] = 0.0
        level.tile_rand_scale_y[tile_idx] = 0.0
        level.tile_rand_scale_z[tile_idx] = 0.0

    # Add corner blocks at the intersections of boundary walls
    corner_positions = [
        # Northeast corner
        (world_max_x + total_offset, world_max_y + total_offset),
        # Northwest corner
        (world_min_x - total_offset, world_max_y + total_offset),
        # Southeast corner
        (world_max_x + total_offset, world_min_y - total_offset),
        # Southwest corner
        (world_min_x - total_offset, world_min_y - total_offset),
    ]

    # Add each corner block
    for i, (pos_x, pos_y) in enumerate(corner_positions):
        tile_idx = start_index + 4 + i  # After the 4 boundary walls

        # Set tile properties for corner blocks
        level.object_ids[tile_idx] = wall_id  # Use wall asset for corners too
        level.tile_x[tile_idx] = pos_x
        level.tile_y[tile_idx] = pos_y
        level.tile_z[tile_idx] = 0.0
        level.tile_scale_x[tile_idx] = wall_thickness  # Square corner blocks (unchanged)
        level.tile_scale_y[tile_idx] = wall_thickness
        level.tile_scale_z[tile_idx] = wall_height
        level.tile_persistent[tile_idx] = True  # Corner blocks persist across episodes
        level.tile_render_only[tile_idx] = False  # Corner blocks have physics
        level.tile_done_on_collide[tile_idx] = False  # Corners don't end episode
        level.tile_entity_type[tile_idx] = 2  # EntityType::Wall
        level.tile_response_type[tile_idx] = 2  # ResponseType::Static
        level.tile_rotation[tile_idx] = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion

        # No randomization for corner blocks (they need to be precise)
        level.tile_rand_x[tile_idx] = 0.0
        level.tile_rand_y[tile_idx] = 0.0
        level.tile_rand_z[tile_idx] = 0.0
        level.tile_rand_rot_z[tile_idx] = 0.0
        level.tile_rand_scale_x[tile_idx] = 0.0
        level.tile_rand_scale_y[tile_idx] = 0.0
        level.tile_rand_scale_z[tile_idx] = 0.0

    return 8  # 4 boundary walls + 4 corner blocks


def _process_targets(level: CompiledLevel, targets: List[Dict]) -> None:
    """
    Process targets configuration and populate CompiledLevel target arrays.

    Args:
        level: CompiledLevel struct to populate
        targets: List of target dictionaries from JSON

    Raises:
        ValueError: If target processing fails
    """
    level.num_targets = len(targets)

    # Initialize all target arrays to zero/default values
    for i in range(MAX_TARGETS):
        level.target_x[i] = 0.0
        level.target_y[i] = 0.0
        level.target_z[i] = 0.0
        level.target_motion_type[i] = 0  # Static by default

        # Clear all parameter slots (flattened array)
        base_idx = i * 8
        for j in range(8):
            level.target_params[base_idx + j] = 0.0

    # Process each target
    for i, target in enumerate(targets):
        # Set position
        position = target["position"]
        level.target_x[i] = float(position[0])
        level.target_y[i] = float(position[1])
        level.target_z[i] = float(position[2])

        # Set motion type
        motion_type = target["motion_type"]
        if motion_type == "static":
            level.target_motion_type[i] = 0

            # Set static parameters if provided (for randomization control)
            if "params" in target:
                params = target["params"]
                base_idx = i * 8
                level.target_params[base_idx + 0] = float(params.get("omega_x", 0.0))  # omega_x
                level.target_params[base_idx + 1] = float(
                    params.get("omega_y", 0.0)
                )  # omega_y (randomization flag)
                level.target_params[base_idx + 2] = float(
                    params.get("center", [0, 0, 0])[0]
                )  # center_x
                level.target_params[base_idx + 3] = float(
                    params.get("center", [0, 0, 0])[1]
                )  # center_y
                level.target_params[base_idx + 4] = float(
                    params.get("center", [0, 0, 0])[2]
                )  # center_z
                level.target_params[base_idx + 5] = float(params.get("mass", 1.0))  # mass
                level.target_params[base_idx + 6] = float(params.get("phase_x", 0.0))  # phase_x
                level.target_params[base_idx + 7] = float(params.get("phase_y", 0.0))  # phase_y
        elif motion_type == "figure8":
            level.target_motion_type[i] = 1

            # Set figure8 parameters (flattened array)
            params = target["params"]
            base_idx = i * 8
            level.target_params[base_idx + 0] = float(params["omega_x"])  # omega_x
            level.target_params[base_idx + 1] = float(params["omega_y"])  # omega_y
            level.target_params[base_idx + 2] = float(params["center"][0])  # center_x
            level.target_params[base_idx + 3] = float(params["center"][1])  # center_y
            level.target_params[base_idx + 4] = float(params["center"][2])  # center_z
            level.target_params[base_idx + 5] = float(params["mass"])  # mass
            level.target_params[base_idx + 6] = float(params.get("phase_x", 0.0))  # phase_x
            level.target_params[base_idx + 7] = float(params.get("phase_y", 0.0))  # phase_y
        elif motion_type == "circular":
            level.target_motion_type[i] = 2

            # Set circular parameters (flattened array)
            params = target["params"]
            base_idx = i * 8
            level.target_params[base_idx + 0] = float(params["angular_velocity"])  # omega_x
            level.target_params[base_idx + 1] = float(
                params.get("randomize", 0.0)
            )  # omega_y (randomize flag)
            level.target_params[base_idx + 2] = float(params["center"][0])  # center_x
            level.target_params[base_idx + 3] = float(params["center"][1])  # center_y
            level.target_params[base_idx + 4] = float(params["center"][2])  # center_z
            level.target_params[base_idx + 5] = float(
                params.get("direction", 1.0)
            )  # mass (direction: 1=ccw, -1=cw)
            level.target_params[base_idx + 6] = float(params["radius"])  # phase_x (radius)
            level.target_params[base_idx + 7] = float(
                params.get("initial_angle", 0.0)
            )  # phase_y (initial angle)
        else:
            raise ValueError(f"Unknown motion_type: {motion_type}")


def _compile_single_level(data: Dict) -> CompiledLevel:
    """
    Internal function to compile a single level from validated JSON data.

    Args:
        data: Validated single-level JSON dictionary

    Returns:
        CompiledLevel struct
    """
    # Extract fields
    ascii_str = _normalize_ascii_input(data["ascii"])
    tileset = data["tileset"]
    scale = data.get("scale", 2.5)
    agent_facing = data.get("agent_facing", None)
    spawn_random = data.get("spawn_random", False)
    auto_boundary_walls = data.get(
        "auto_boundary_walls", False
    )  # Default to False for backward compatibility
    boundary_wall_offset = data.get("boundary_wall_offset", 0.0)  # Default to 0.0
    level_name = data.get("name", "unknown_level")
    targets = data.get("targets", [])  # Default to empty list

    # Process tileset to get mappings
    char_to_tile, char_to_props = _process_tileset(tileset)

    # Parse ASCII to get dimensions
    lines = [line.rstrip() for line in ascii_str.strip().split("\n")]
    if not lines:
        raise ValueError("Empty level string")

    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0

    if width == 0 or height == 0:
        raise ValueError("Level has zero width or height")

    # Validate dimensions
    if width < MIN_LEVEL_WIDTH or width > MAX_GRID_SIZE:
        raise ValueError(
            f"Level width {width} must be between {MIN_LEVEL_WIDTH} and {MAX_GRID_SIZE}"
        )
    if height < MIN_LEVEL_HEIGHT or height > MAX_GRID_SIZE:
        raise ValueError(
            f"Level height {height} must be between {MIN_LEVEL_HEIGHT} and {MAX_GRID_SIZE}"
        )

    # Parse ASCII to tiles and spawns
    tiles, spawns, entity_count = _parse_ascii_to_tiles(
        ascii_str, char_to_tile, char_to_props, scale, width, height
    )

    # Validate entity count (excluding empty tiles)
    if entity_count > MAX_TILES:
        raise ValueError(f"Level has too many entities: {entity_count} > {MAX_TILES} max")

    # Validate results
    if len(spawns) == 0:
        raise ValueError("No spawn points (S) found in level")

    if len(spawns) > MAX_SPAWNS:
        raise ValueError(f"Too many spawn points: {len(spawns)} > {MAX_SPAWNS}")

    # Calculate max_entities for BVH allocation
    # Must account for: level tiles + persistent entities (floor, agents, origin markers) + buffer
    max_entities = entity_count + 6 + 30

    # Create CompiledLevel struct
    level = create_compiled_level()

    # Set basic fields
    level.num_tiles = len(tiles)
    level.max_entities = max_entities
    level.width = width
    level.height = height
    level.world_scale = scale
    level.done_on_collide = False
    level.level_name = level_name.encode("utf-8")[:64]  # Ensure it fits in char[64]

    # Set world boundaries
    min_tile_center_x = -(width - 1) / 2.0 * scale
    max_tile_center_x = (width - 1) / 2.0 * scale
    min_tile_center_y = -(height - 1) / 2.0 * scale
    max_tile_center_y = (height - 1) / 2.0 * scale

    tile_half_extent = scale / 2.0

    level.world_min_x = min_tile_center_x - tile_half_extent
    level.world_max_x = max_tile_center_x + tile_half_extent
    level.world_min_y = min_tile_center_y - tile_half_extent
    level.world_max_y = max_tile_center_y + tile_half_extent
    level.world_min_z = 0.0
    level.world_max_z = 10.0 * scale

    # Set spawn data
    level.num_spawns = len(spawns)
    level.spawn_random = spawn_random
    level.auto_boundary_walls = auto_boundary_walls
    for i in range(len(spawns)):
        level.spawn_x[i] = spawns[i][0]
        level.spawn_y[i] = spawns[i][1]

        if agent_facing and i < len(agent_facing):
            level.spawn_facing[i] = agent_facing[i]
        else:
            level.spawn_facing[i] = 0.0

    # Get asset IDs for entity type detection
    from .ctypes_bindings import get_physics_asset_object_id

    wall_id = get_physics_asset_object_id("wall")
    cube_id = get_physics_asset_object_id("cube")
    cylinder_id = get_physics_asset_object_id("cylinder")

    wall_scale_factor = scale  # Scale walls to match tile spacing

    # Fill tile arrays
    for i, (world_x, world_y, tile_type, char) in enumerate(tiles):
        level.object_ids[i] = tile_type
        level.tile_x[i] = world_x
        level.tile_y[i] = world_y
        level.tile_z[i] = 0.0

        # Set tile properties
        tile_props = char_to_props.get(char, {})
        level.tile_rand_x[i] = tile_props.get("rand_x", 0.0)
        level.tile_rand_y[i] = tile_props.get("rand_y", 0.0)
        level.tile_rand_z[i] = tile_props.get("rand_z", 0.0)
        level.tile_rand_rot_z[i] = tile_props.get("rand_rot_z", 0.0)
        level.tile_done_on_collide[i] = tile_props.get("done_on_collision", False)

        # Set entity type and scale based on object type
        if tile_type == wall_id:
            level.tile_entity_type[i] = 2  # EntityType::Wall
            level.tile_scale_x[i] = wall_scale_factor
            level.tile_scale_y[i] = wall_scale_factor
            level.tile_scale_z[i] = 1.0
        elif tile_type == cube_id:
            level.tile_entity_type[i] = 1  # EntityType::Cube
            level.tile_scale_x[i] = 1.0
            level.tile_scale_y[i] = 1.0
            level.tile_scale_z[i] = 1.0
        elif tile_type == cylinder_id:
            level.tile_entity_type[i] = 0  # EntityType::None
            level.tile_scale_x[i] = 1.0
            level.tile_scale_y[i] = 1.0
            level.tile_scale_z[i] = 1.0
        else:
            level.tile_entity_type[i] = 0
            level.tile_scale_x[i] = 1.0
            level.tile_scale_y[i] = 1.0
            level.tile_scale_z[i] = 1.0

        # Set other tile properties to defaults
        level.tile_persistent[i] = False
        level.tile_render_only[i] = False
        level.tile_response_type[i] = 2  # ResponseType::Static

        # Identity quaternion for rotation (w, x, y, z)
        level.tile_rotation[i] = (1.0, 0.0, 0.0, 0.0)

    # Process targets configuration
    _process_targets(level, targets)

    # Add automatic boundary walls and corners if enabled
    if auto_boundary_walls:
        tiles_added = _add_boundary_walls(
            level, len(tiles), width, height, scale, boundary_wall_offset
        )
        level.num_tiles = len(tiles) + tiles_added

    return level


def compile_multi_level(json_data: Union[str, Dict]) -> List[CompiledLevel]:
    """
    Compile multi-level JSON definition to list of CompiledLevel structures.

    Args:
        json_data: JSON string or dictionary with multi-level definition

    Returns:
        List of CompiledLevel structs ready for C API

    Raises:
        ValueError: If JSON is invalid or compilation fails

    Example:
        multi_level = {
            "levels": [
                {
                    "ascii": ["###", "#S#", "###"],
                    "name": "level_1",
                    "agent_facing": [0.0]
                },
                {
                    "ascii": ["#####", "#S.C#", "#####"],
                    "name": "level_2"
                }
            ],
            "tileset": {
                "#": {"asset": "wall"},
                "C": {"asset": "cube"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "scale": 2.5
        }
        compiled_levels = compile_multi_level(multi_level)
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        data = json_data

    # Ensure this is actually multi-level format
    if "levels" not in data:
        raise ValueError(
            "compile_multi_level() requires multi-level JSON format with 'levels' field"
        )

    # Validate JSON structure (will call multi-level validation)
    _validate_json_level(data)

    # Extract shared fields
    shared_tileset = data.get("tileset")  # May be None if levels have their own tilesets
    shared_scale = data.get("scale", 2.5)
    shared_spawn_random = data.get("spawn_random", False)
    shared_auto_boundary_walls = data.get("auto_boundary_walls", False)
    shared_boundary_wall_offset = data.get("boundary_wall_offset", 0.0)
    level_set_name = data.get("name", "multi_level_set")

    compiled_levels = []

    # Compile each level
    for i, level_data in enumerate(data["levels"]):
        # Determine which tileset to use: per-level first, then global fallback
        if "tileset" in level_data:
            level_tileset = level_data["tileset"]
        elif shared_tileset is not None:
            level_tileset = shared_tileset
        else:
            # This should not happen if validation passed, but be defensive
            raise ValueError(f"Level {i} has no tileset and no global tileset provided")

        # Build single-level JSON for this level
        single_level_json = {
            "ascii": level_data["ascii"],
            "tileset": level_tileset,
            "scale": shared_scale,
            "spawn_random": shared_spawn_random,
            "auto_boundary_walls": shared_auto_boundary_walls,
            "boundary_wall_offset": shared_boundary_wall_offset,
        }

        # Use per-level name if provided, otherwise generate from set name
        if "name" in level_data:
            single_level_json["name"] = level_data["name"]
        else:
            single_level_json["name"] = f"{level_set_name}_level_{i+1}"

        # Use per-level agent_facing if provided
        if "agent_facing" in level_data:
            single_level_json["agent_facing"] = level_data["agent_facing"]

        # Handle targets - use per-level targets if provided, otherwise use shared targets
        if "targets" in level_data:
            single_level_json["targets"] = level_data["targets"]
        elif "targets" in data:
            single_level_json["targets"] = data["targets"]

        # Compile this individual level using the internal function
        compiled_level = _compile_single_level(single_level_json)
        compiled_levels.append(compiled_level)

    return compiled_levels


def compile_level(json_data: Union[str, Dict]) -> List[CompiledLevel]:
    """
    Compile JSON level definition to CompiledLevel format.

    Args:
        json_data: JSON string or dictionary with level definition

    Returns:
        List of CompiledLevel structs ready for C API (always returns list, even for single levels)

    Raises:
        ValueError: If JSON is invalid or compilation fails

    Example:
        json_level = {
            "ascii": "###O###\\n#S...C#\\n#######",
            "tileset": {
                "#": {"asset": "wall"},
                "C": {"asset": "cube"},
                "O": {"asset": "cylinder"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "scale": 2.5,
            "agent_facing": [0.0]
        }
        compiled = compile_level(json_level)
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        data = json_data

    # Validate JSON structure
    _validate_json_level(data)

    # Check if this is multi-level format - delegate to compile_multi_level
    if "levels" in data:
        return compile_multi_level(data)

    # Single level format - use internal helper and wrap in list
    return [_compile_single_level(data)]


# Alias for backward compatibility with old test files
compile_level_from_json = compile_level


def validate_compiled_level(compiled: CompiledLevel) -> None:
    """
    Validate compiled level data before passing to C API.

    Args:
        compiled: CompiledLevel struct from compile_level()

    Raises:
        ValueError: If validation fails
    """
    # Validate ranges
    if compiled.num_tiles < 0 or compiled.num_tiles > MAX_TILES:
        raise ValueError(f"Invalid num_tiles: {compiled.num_tiles}")

    if compiled.max_entities <= 0:
        raise ValueError(f"Invalid max_entities: {compiled.max_entities}")

    if compiled.width <= 0 or compiled.height <= 0:
        raise ValueError(f"Invalid dimensions: {compiled.width}x{compiled.height}")

    if compiled.world_scale <= 0.0:
        raise ValueError(f"Invalid scale: {compiled.world_scale}")

    if compiled.num_spawns <= 0 or compiled.num_spawns > MAX_SPAWNS:
        raise ValueError(f"Invalid num_spawns: {compiled.num_spawns}")

    # Validate array lengths - must be sized to MAX_TILES for C API compatibility
    if len(compiled.object_ids) != MAX_TILES:
        raise ValueError(
            f"Invalid object_ids array length: {len(compiled.object_ids)}, must be {MAX_TILES}"
        )

    if len(compiled.tile_x) != MAX_TILES:
        raise ValueError(
            f"Invalid tile_x array length: {len(compiled.tile_x)}, must be {MAX_TILES}"
        )

    if len(compiled.tile_y) != MAX_TILES:
        raise ValueError(
            f"Invalid tile_y array length: {len(compiled.tile_y)}, must be {MAX_TILES}"
        )


# Binary I/O functions moved to level_io.py
# Import them for backwards compatibility
def save_compiled_level_binary(compiled: CompiledLevel, filepath: str) -> None:
    """Deprecated: Use level_io.save_compiled_levels() instead."""
    from .level_io import save_compiled_levels

    save_compiled_levels([compiled], filepath)


def load_compiled_level_binary(filepath: str) -> CompiledLevel:
    """Deprecated: Use level_io.load_compiled_levels() instead."""
    from .level_io import load_compiled_levels

    levels = load_compiled_levels(filepath)
    if len(levels) != 1:
        raise ValueError(f"Expected exactly 1 level in file, got {len(levels)}")
    return levels[0]


def print_level_info(compiled: CompiledLevel) -> None:
    """Print compiled level information for debugging."""
    print("Compiled Level Info:")
    print(f"  Name: {compiled.level_name.decode('utf-8', errors='ignore')}")
    print(f"  Dimensions: {compiled.width}x{compiled.height} (scale: {compiled.world_scale})")
    print(f"  Tiles: {compiled.num_tiles}")
    print(f"  Max entities: {compiled.max_entities}")
    print(f"  Spawn points: {compiled.num_spawns}")

    for i in range(compiled.num_spawns):
        x = compiled.spawn_x[i]
        y = compiled.spawn_y[i]
        facing_rad = compiled.spawn_facing[i]
        facing_deg = facing_rad * 180.0 / math.pi
        print(f"    Spawn {i}: ({x:.1f}, {y:.1f}) facing {facing_deg:.1f}°")


def main():
    """Command line interface for level compiler."""
    parser = argparse.ArgumentParser(
        description="Compile JSON level files to binary .lvl format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile JSON file to binary level
  python -m madrona_escape_room.level_compiler level.json level.lvl

  # Display info about binary level
  python -m madrona_escape_room.level_compiler --info level.lvl

  # Test with example level
  python -m madrona_escape_room.level_compiler --test
        """,
    )

    parser.add_argument("input", nargs="?", help="Input JSON level file")
    parser.add_argument("output", nargs="?", help="Output binary .lvl file")
    parser.add_argument("--info", metavar="FILE", help="Display info about binary level file")
    parser.add_argument("--test", action="store_true", help="Run test with example level")

    args = parser.parse_args()

    try:
        if args.test:
            # Run test with example level
            print("=== Level Compiler Test ===\n")

            test_json = {
                "ascii": """##########
#S.......#
#..####..#
#..#C#...#
#..###...#
##########""",
                "tileset": {
                    "#": {"asset": "wall", "done_on_collision": False},
                    "C": {"asset": "cube", "done_on_collision": True},
                    "S": {"asset": "spawn"},
                    ".": {"asset": "empty"},
                },
                "scale": 2.5,
                "agent_facing": [math.pi / 2],  # Face right
                "name": "test_level",
            }

            print("Compiling test level...")
            compiled_levels = compile_level(test_json)
            compiled = compiled_levels[0]  # Test level is always single level
            validate_compiled_level(compiled)
            print_level_info(compiled)

            # Test save/load round-trip
            test_file = "/tmp/test_level.lvl"
            print(f"\nSaving to {test_file}...")
            save_compiled_level_binary(compiled, test_file)

            print(f"Loading from {test_file}...")
            loaded = load_compiled_level_binary(test_file)

            # Verify key fields match
            if compiled.num_tiles != loaded.num_tiles:
                raise ValueError(
                    f"Mismatch in num_tiles: {compiled.num_tiles} != {loaded.num_tiles}"
                )
            if compiled.width != loaded.width:
                raise ValueError(f"Mismatch in width: {compiled.width} != {loaded.width}")
            if compiled.height != loaded.height:
                raise ValueError(f"Mismatch in height: {compiled.height} != {loaded.height}")
            if abs(compiled.world_scale - loaded.world_scale) > 0.001:
                raise ValueError(
                    f"Mismatch in scale: {compiled.world_scale} != {loaded.world_scale}"
                )

            print("✓ Test completed successfully!")

            # Clean up
            Path(test_file).unlink()

        elif args.info:
            # Display info about binary level
            print(f"Loading binary level: {args.info}")
            compiled = load_compiled_level_binary(args.info)
            print_level_info(compiled)

        elif args.input and args.output:
            # Compile JSON to binary
            print(f"Compiling '{args.input}' to '{args.output}'")

            # Read JSON file
            with open(args.input, "r") as f:
                json_data = f.read()

            # Compile and validate
            compiled_levels = compile_level(json_data)

            if len(compiled_levels) == 1:
                # Single level
                compiled = compiled_levels[0]
                validate_compiled_level(compiled)
                save_compiled_level_binary(compiled, args.output)
                print("✓ Level compiled successfully")
                print_level_info(compiled)
            else:
                # Multi-level - save as single file containing all levels
                print(f"✓ Multi-level compilation: {len(compiled_levels)} levels")
                for compiled in compiled_levels:
                    validate_compiled_level(compiled)

                # Save all levels as multi-level binary file (still .lvl extension)
                from .level_io import save_compiled_levels

                save_compiled_levels(compiled_levels, args.output)
                print("✓ Multi-level compiled successfully")
                print(f"Contains {len(compiled_levels)} levels for curriculum learning")

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
