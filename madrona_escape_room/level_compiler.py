#!/usr/bin/env python3
"""
Python Level Compiler for Madrona Escape Room

Compiles ASCII level strings to CompiledLevel format for GPU processing.
Part of the test-driven level system that allows tests to define their
environment layouts using visual ASCII art.

Supports three input formats:
1. Plain ASCII strings for simple levels
2. JSON format with ASCII and parameters (agent facing, scale, etc.)
3. JSON format with tileset definitions for custom asset mapping
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def _get_max_tiles_from_c_api():
    """Get MAX_TILES from C API - returns the actual CompiledLevel::MAX_TILES value"""
    try:
        from .ctypes_bindings import _get_max_tiles

        return _get_max_tiles()
    except ImportError:
        # Fallback if ctypes bindings not available
        return 1024


def _get_max_spawns_from_c_api():
    """Get MAX_SPAWNS from C API - returns the actual CompiledLevel::MAX_SPAWNS value"""
    try:
        from .ctypes_bindings import _get_max_spawns

        return _get_max_spawns()
    except ImportError:
        # Fallback if ctypes bindings not available
        return 8


# Get the authoritative values from C++
MAX_TILES_C_API = _get_max_tiles_from_c_api()
MAX_SPAWNS_C_API = _get_max_spawns_from_c_api()

# Tile type constants (must match C++ TileType enum in src/level_gen.cpp)
TILE_EMPTY = 0
TILE_WALL = 1
TILE_CUBE = 2
TILE_SPAWN = 3
TILE_DOOR = 4  # Future
TILE_BUTTON = 5  # Future
TILE_GOAL = 6  # Future

# Level dimension limits for validation
MAX_LEVEL_WIDTH = 64
MAX_LEVEL_HEIGHT = 64
MIN_LEVEL_WIDTH = 3
MIN_LEVEL_HEIGHT = 3
# MAX_TOTAL_TILES now comes from C API (MAX_TILES_C_API)

# Special tile type constants (for spawn and empty, which aren't assets)
# Legacy CHAR_MAP will be built dynamically using C API asset IDs


def _get_legacy_char_map():
    """Build the legacy CHAR_MAP using actual asset IDs from C API"""
    from .ctypes_bindings import get_physics_asset_object_id

    # Get the actual asset IDs from the C API
    wall_id = get_physics_asset_object_id("wall")
    cube_id = get_physics_asset_object_id("cube")

    return {
        ".": TILE_EMPTY,
        " ": TILE_EMPTY,  # Whitespace also means empty
        "#": wall_id if wall_id >= 0 else 2,  # Use actual wall asset ID
        "C": cube_id if cube_id >= 0 else 1,  # Use actual cube asset ID
        "S": TILE_SPAWN,
    }


# Default tileset using asset names
# Each entry can optionally specify scale_x, scale_y, scale_z
DEFAULT_TILESET = {
    "#": {"asset": "wall"},  # Will be auto-scaled to match tile spacing
    "C": {"asset": "cube"},
    "O": {"asset": "cylinder"},
    "S": {"asset": "spawn"},  # Special case for agent spawn
    ".": {"asset": "empty"},  # Special case for empty space
    " ": {"asset": "empty"},  # Whitespace also means empty
}


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

    # Always use C API to get correct asset IDs
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

    for char, tile_def in tileset.items():
        if not isinstance(char, str) or len(char) != 1:
            raise ValueError(f"Tileset key must be single character, got: '{char}'")

        if not isinstance(tile_def, dict):
            raise ValueError(f"Tileset value for '{char}' must be a dictionary")

        if "asset" not in tile_def:
            raise ValueError(f"Tileset entry for '{char}' must have 'asset' field")

        if not isinstance(tile_def["asset"], str):
            raise ValueError(f"Asset name for '{char}' must be a string")


def compile_level(
    ascii_str: str,
    scale: float = 2.5,
    agent_facing: Optional[List[float]] = None,
    level_name: str = "unknown_level",
    tileset: Optional[Dict] = None,
) -> Dict:
    """
    Compile ASCII level string to dict matching MER_CompiledLevel struct.

    Args:
        ascii_str: Multi-line ASCII level definition using the character map above
        scale: World units per ASCII character (default 2.5 units per cell)
        agent_facing: List of initial facing angles in radians for each agent (optional)
                     Default is 0.0 (facing forward/north) for all agents
        level_name: Name of the level for identification (default "unknown_level")
        tileset: Optional dictionary mapping characters to asset definitions.
                If not provided, uses legacy CHAR_MAP with hardcoded tile types.

    Returns:
        Dict with fields matching MER_CompiledLevel C struct:
        - num_tiles: Number of entities to create
        - max_entities: BVH size hint (includes persistent entities)
        - width, height: Grid dimensions
        - scale: World scale factor
        - object_ids, tile_x, tile_y: Arrays of tile data (MAX_TILES_C_API elements each)
        - spawn_facing: Array of agent facing angles (MAX_SPAWNS elements)

    Raises:
        ValueError: If level is invalid (empty, too large, no spawn points, unknown chars)

    Example:
        level = '''
        ##########
        #S.......#
        #..####..#
        #........#
        ##########
        '''
        compiled = compile_level(level, agent_facing=[math.pi/2])  # Face right
        # compiled['num_tiles'] = 38 (wall tiles)
        # compiled['max_entities'] = 88 (walls + buffer for agents/floor/etc)

        # With custom tileset:
        tileset = {
            "#": {"asset": "wall"},
            "O": {"asset": "cylinder"},
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"}
        }
        compiled = compile_level(level, tileset=tileset)
    """
    # Clean up input - remove leading/trailing whitespace and normalize line endings
    lines = [line.rstrip() for line in ascii_str.strip().split("\n")]
    if not lines or all(len(line) == 0 for line in lines):
        raise ValueError("Empty level string")

    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0

    if width == 0 or height == 0:
        raise ValueError("Level has zero width or height")

    # Validate dimensions
    if width < MIN_LEVEL_WIDTH or width > MAX_LEVEL_WIDTH:
        raise ValueError(
            f"Level width {width} must be between {MIN_LEVEL_WIDTH} and {MAX_LEVEL_WIDTH}"
        )
    if height < MIN_LEVEL_HEIGHT or height > MAX_LEVEL_HEIGHT:
        raise ValueError(
            f"Level height {height} must be between {MIN_LEVEL_HEIGHT} and {MAX_LEVEL_HEIGHT}"
        )

    # Calculate total array size needed
    array_size = width * height
    if array_size > MAX_TILES_C_API:
        raise ValueError(
            f"Level too large: {width}×{height} = {array_size} tiles > "
            f"{MAX_TILES_C_API} max (from C API)"
        )

    # Determine which character map to use
    if tileset is not None:
        # Validate the provided tileset
        _validate_tileset(tileset)
        char_to_tile = {}
        for char, tile_def in tileset.items():
            asset_name = tile_def["asset"]
            try:
                char_to_tile[char] = _get_asset_object_id(asset_name)
            except ValueError as e:
                raise ValueError(f"Invalid asset '{asset_name}' for character '{char}': {e}")
    else:
        # Use legacy CHAR_MAP with correct asset IDs from C API
        char_to_tile = _get_legacy_char_map()

    tiles = []
    spawns = []
    entity_count = 0  # Count entities that need physics bodies

    # Parse ASCII grid to extract tile positions and types
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char in char_to_tile:
                tile_type = char_to_tile[char]

                # Convert grid coordinates to world coordinates (center at origin)
                # Note: Y is inverted because ASCII Y=0 is at top, but world Y+ is up
                world_x = (x - width / 2.0 + 0.5) * scale
                world_y = -(y - height / 2.0 + 0.5) * scale

                if tile_type == TILE_SPAWN:
                    spawns.append((world_x, world_y))
                    # Don't add spawn tiles to tile array - handled by agent placement
                elif tile_type != TILE_EMPTY:
                    tiles.append((world_x, world_y, tile_type))

                    # Count entities that will need physics bodies and BVH slots
                    # Note: This is a simplified count - actual requirements depend on asset
                    # properties
                    entity_count += 1
            else:
                raise ValueError(f"Unknown character '{char}' at grid position ({x}, {y})")

    # Calculate max_entities for BVH allocation
    # Must account for:
    # - Level tiles (entity_count)
    # - Persistent entities: 1 floor + 2 agents + 3 origin markers = 6
    # - All 6 persistent entities have ObjectID components and consume BVH slots
    # - Additional buffer for safety
    max_entities = entity_count + 6 + 30  # tiles + persistent + buffer

    # Validate constraints
    if len(tiles) > array_size:
        raise ValueError(
            f"Too many non-empty tiles: {len(tiles)} > {array_size} (level dimensions)"
        )
    if len(spawns) == 0:
        raise ValueError("No spawn points (S) found in level - at least one required")

    # Build arrays with MAX_TILES_C_API size for C++ compatibility
    object_ids = [0] * MAX_TILES_C_API
    tile_x = [0.0] * MAX_TILES_C_API
    tile_y = [0.0] * MAX_TILES_C_API
    tile_persistent = [False] * MAX_TILES_C_API  # Default: all tiles non-persistent

    # Fill arrays with actual tile data
    for i, (world_x, world_y, tile_type) in enumerate(tiles):
        object_ids[i] = tile_type
        tile_x[i] = world_x
        tile_y[i] = world_y
        # tile_persistent[i] remains False by default

    # Prepare spawn arrays (MAX_SPAWNS from C API)
    spawn_x = [0.0] * MAX_SPAWNS_C_API
    spawn_y = [0.0] * MAX_SPAWNS_C_API
    spawn_facing = [0.0] * MAX_SPAWNS_C_API  # Default facing forward (0 radians)
    num_spawns = min(len(spawns), MAX_SPAWNS_C_API)

    for i in range(num_spawns):
        spawn_x[i] = spawns[i][0]
        spawn_y[i] = spawns[i][1]

        # Set agent facing angles if provided
        if agent_facing and i < len(agent_facing):
            spawn_facing[i] = agent_facing[i]
        else:
            spawn_facing[i] = 0.0  # Default: face forward

    # Initialize new fields with default values (all arrays size MAX_TILES_C_API)
    tile_z = [0.0] * MAX_TILES_C_API  # Default Z=0
    tile_render_only = [False] * MAX_TILES_C_API  # Default: all have physics
    tile_entity_type = [0] * MAX_TILES_C_API  # Default: EntityType::None
    tile_response_type = [2] * MAX_TILES_C_API  # Default: ResponseType::Static
    tile_scale_x = [1.0] * MAX_TILES_C_API  # Default scale 1.0
    tile_scale_y = [1.0] * MAX_TILES_C_API
    tile_scale_z = [1.0] * MAX_TILES_C_API
    tile_rot_w = [1.0] * MAX_TILES_C_API  # Identity quaternion
    tile_rot_x = [0.0] * MAX_TILES_C_API
    tile_rot_y = [0.0] * MAX_TILES_C_API
    tile_rot_z = [0.0] * MAX_TILES_C_API

    # Set entity types based on object IDs and apply appropriate scaling
    # Get the correct asset IDs from the C API
    from .ctypes_bindings import get_physics_asset_object_id

    wall_id = get_physics_asset_object_id("wall")
    cube_id = get_physics_asset_object_id("cube")
    cylinder_id = get_physics_asset_object_id("cylinder")

    # Calculate wall scale factor based on tile spacing
    # Walls are 1x1 units but tiles are spaced at 'scale' units apart
    # We need to scale walls to fill the gap between tiles
    wall_scale_factor = scale  # Scale walls to match tile spacing

    for i, (_, _, tile_type) in enumerate(tiles):
        if tile_type == wall_id:
            tile_entity_type[i] = 2  # EntityType::Wall
            # Scale walls to match tile spacing
            tile_scale_x[i] = wall_scale_factor
            tile_scale_y[i] = wall_scale_factor
            # Keep Z scale at 1.0 to maintain wall height
            tile_scale_z[i] = 1.0
        elif tile_type == cube_id:
            tile_entity_type[i] = 1  # EntityType::Cube
            # Cubes can stay at default scale
            tile_scale_x[i] = 1.0
            tile_scale_y[i] = 1.0
            tile_scale_z[i] = 1.0
        elif tile_type == cylinder_id:
            tile_entity_type[i] = 0  # EntityType::None (generic entity)
            # Cylinders can stay at default scale
            tile_scale_x[i] = 1.0
            tile_scale_y[i] = 1.0
            tile_scale_z[i] = 1.0
        else:
            # For other assets, use EntityType::None and default scale
            tile_entity_type[i] = 0
            tile_scale_x[i] = 1.0
            tile_scale_y[i] = 1.0
            tile_scale_z[i] = 1.0

    # Return dict with compiler-calculated array sizing
    return {
        "num_tiles": len(tiles),
        "max_entities": max_entities,
        "width": width,
        "height": height,
        "scale": scale,
        "level_name": level_name,
        "num_spawns": num_spawns,
        "spawn_x": spawn_x,
        "spawn_y": spawn_y,
        "spawn_facing": spawn_facing,  # Agent facing angles in radians
        "array_size": array_size,  # NEW: compiler-calculated array size (width × height)
        "object_ids": object_ids,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_z": tile_z,
        "tile_persistent": tile_persistent,  # Persistence flags for each tile
        "tile_render_only": tile_render_only,  # Render-only flags
        "tile_entity_type": tile_entity_type,  # EntityType values
        "tile_response_type": tile_response_type,  # ResponseType values
        "tile_scale_x": tile_scale_x,  # Scale components
        "tile_scale_y": tile_scale_y,
        "tile_scale_z": tile_scale_z,
        "tile_rot_w": tile_rot_w,  # Rotation quaternion components
        "tile_rot_x": tile_rot_x,
        "tile_rot_y": tile_rot_y,
        "tile_rot_z": tile_rot_z,
        # Metadata for debugging/validation (not part of C struct)
        "_spawn_points": spawns,
        "_entity_count": entity_count,
    }


def validate_compiled_level(compiled: Dict) -> None:
    """
    Validate compiled level data before passing to C API.

    Args:
        compiled: Output from compile_level()

    Raises:
        ValueError: If validation fails
    """
    required_fields = [
        "num_tiles",
        "max_entities",
        "width",
        "height",
        "scale",
        "level_name",
        "num_spawns",
        "spawn_x",
        "spawn_y",
        "spawn_facing",
        "array_size",
        "object_ids",
        "tile_x",
        "tile_y",
        "tile_z",
        "tile_persistent",
        "tile_render_only",
        "tile_entity_type",
        "tile_response_type",
        "tile_scale_x",
        "tile_scale_y",
        "tile_scale_z",
        "tile_rot_w",
        "tile_rot_x",
        "tile_rot_y",
        "tile_rot_z",
    ]
    for field in required_fields:
        if field not in compiled:
            raise ValueError(f"Missing required field: {field}")

    # Validate against actual array size for this level
    array_size = compiled["array_size"]
    if compiled["num_tiles"] < 0 or compiled["num_tiles"] > array_size:
        raise ValueError(f"Invalid num_tiles: {compiled['num_tiles']} (must be 0-{array_size})")

    if compiled["max_entities"] <= 0:
        raise ValueError(f"Invalid max_entities: {compiled['max_entities']} (must be > 0)")

    if compiled["width"] <= 0 or compiled["height"] <= 0:
        raise ValueError(f"Invalid dimensions: {compiled['width']}x{compiled['height']}")

    if compiled["scale"] <= 0.0:
        raise ValueError(f"Invalid scale: {compiled['scale']} (must be > 0)")

    # Check array lengths match MAX_TILES_C_API (fixed size for C++ compatibility)
    tile_arrays = [
        "object_ids",
        "tile_x",
        "tile_y",
        "tile_z",
        "tile_persistent",
        "tile_render_only",
        "tile_entity_type",
        "tile_response_type",
        "tile_scale_x",
        "tile_scale_y",
        "tile_scale_z",
        "tile_rot_w",
        "tile_rot_x",
        "tile_rot_y",
        "tile_rot_z",
    ]
    for array_name in tile_arrays:
        if len(compiled[array_name]) != MAX_TILES_C_API:
            raise ValueError(
                f"Invalid {array_name} array length: {len(compiled[array_name])} "
                f"(must be {MAX_TILES_C_API} for C++ compatibility)"
            )


def save_compiled_level_binary(compiled: Dict, filepath: str) -> None:
    """
    Save compiled level dictionary to binary .lvl file.

    Args:
        compiled: Output from compile_level()
        filepath: Path to save .lvl file

    Raises:
        ValueError: If compiled level is invalid
        IOError: If file cannot be written
    """
    # Validate first
    validate_compiled_level(compiled)

    try:
        with open(filepath, "wb") as f:
            # Write header fields (matching MER_CompiledLevel struct layout)
            # int32_t num_tiles, max_entities, width, height
            f.write(struct.pack("<i", compiled["num_tiles"]))
            f.write(struct.pack("<i", compiled["max_entities"]))
            f.write(struct.pack("<i", compiled["width"]))
            f.write(struct.pack("<i", compiled["height"]))

            # float scale
            f.write(struct.pack("<f", compiled["scale"]))

            # char level_name[64] - truncate or pad to 64 bytes
            level_name = compiled["level_name"][:63]  # Ensure null termination
            level_name_bytes = level_name.encode("utf-8").ljust(64, b"\0")[:64]
            f.write(level_name_bytes)

            # Spawn data
            f.write(struct.pack("<i", compiled["num_spawns"]))

            # spawn_x array - always write MAX_SPAWNS_C_API elements
            for i in range(MAX_SPAWNS_C_API):
                f.write(struct.pack("<f", compiled["spawn_x"][i]))

            # spawn_y array - always write MAX_SPAWNS_C_API elements
            for i in range(MAX_SPAWNS_C_API):
                f.write(struct.pack("<f", compiled["spawn_y"][i]))

            # spawn_facing array - always write MAX_SPAWNS_C_API elements
            for i in range(MAX_SPAWNS_C_API):
                f.write(struct.pack("<f", compiled["spawn_facing"][i]))

            # Arrays: Write fixed-size arrays for C++ compatibility
            # Always write MAX_TILES_C_API elements regardless of actual array size
            array_size = compiled["array_size"]

            # object_ids array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size:
                    f.write(struct.pack("<i", compiled["object_ids"][i]))
                else:
                    f.write(struct.pack("<i", 0))  # TILE_EMPTY

            # tile_x array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size:
                    f.write(struct.pack("<f", compiled["tile_x"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_y array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size:
                    f.write(struct.pack("<f", compiled["tile_y"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_z array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_z" in compiled:
                    f.write(struct.pack("<f", compiled["tile_z"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_persistent array - pad with false if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_persistent" in compiled:
                    # Write as bool (1 byte)
                    f.write(struct.pack("<B", 1 if compiled["tile_persistent"][i] else 0))
                else:
                    f.write(struct.pack("<B", 0))  # Default: non-persistent

            # tile_render_only array - pad with false if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_render_only" in compiled:
                    # Write as bool (1 byte)
                    f.write(struct.pack("<B", 1 if compiled["tile_render_only"][i] else 0))
                else:
                    f.write(struct.pack("<B", 0))  # Default: has physics

            # tile_entity_type array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_entity_type" in compiled:
                    f.write(struct.pack("<i", compiled["tile_entity_type"][i]))
                else:
                    f.write(struct.pack("<i", 0))  # Default: EntityType::None

            # tile_response_type array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_response_type" in compiled:
                    f.write(struct.pack("<i", compiled["tile_response_type"][i]))
                else:
                    f.write(struct.pack("<i", 2))  # Default: ResponseType::Static

            # tile_scale_x array - pad with 1.0 if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_scale_x" in compiled:
                    f.write(struct.pack("<f", compiled["tile_scale_x"][i]))
                else:
                    f.write(struct.pack("<f", 1.0))

            # tile_scale_y array - pad with 1.0 if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_scale_y" in compiled:
                    f.write(struct.pack("<f", compiled["tile_scale_y"][i]))
                else:
                    f.write(struct.pack("<f", 1.0))

            # tile_scale_z array - pad with 1.0 if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_scale_z" in compiled:
                    f.write(struct.pack("<f", compiled["tile_scale_z"][i]))
                else:
                    f.write(struct.pack("<f", 1.0))

            # tile_rot_w array - pad with 1.0 (identity quaternion) if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rot_w" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rot_w"][i]))
                else:
                    f.write(struct.pack("<f", 1.0))  # Identity quaternion W component

            # tile_rot_x array - pad with 0.0 if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rot_x" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rot_x"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_rot_y array - pad with 0.0 if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rot_y" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rot_y"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_rot_z array - pad with 0.0 if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rot_z" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rot_z"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_rand_x array - pad with 0.0 if needed (no randomization)
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rand_x" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rand_x"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_rand_y array - pad with 0.0 if needed (no randomization)
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rand_y" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rand_y"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_rand_z array - pad with 0.0 if needed (no randomization)
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rand_z" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rand_z"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

            # tile_rand_rot_z array - pad with 0.0 if needed (no randomization)
            for i in range(MAX_TILES_C_API):
                if i < array_size and "tile_rand_rot_z" in compiled:
                    f.write(struct.pack("<f", compiled["tile_rand_rot_z"][i]))
                else:
                    f.write(struct.pack("<f", 0.0))

    except IOError as e:
        raise IOError(f"Failed to write level file '{filepath}': {e}")


def load_compiled_level_binary(filepath: str) -> Dict:
    """
    Load compiled level dictionary from binary .lvl file.

    Args:
        filepath: Path to .lvl file

    Returns:
        Dict matching compile_level() output format

    Raises:
        IOError: If file cannot be read
        ValueError: If file format is invalid
    """
    try:
        with open(filepath, "rb") as f:
            # Read header fields
            num_tiles = struct.unpack("<i", f.read(4))[0]
            max_entities = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<i", f.read(4))[0]
            height = struct.unpack("<i", f.read(4))[0]
            scale = struct.unpack("<f", f.read(4))[0]

            # Read level_name (64 bytes, null-terminated)
            level_name_bytes = f.read(64)
            level_name = level_name_bytes.rstrip(b"\0").decode("utf-8")

            # Read spawn data
            num_spawns = struct.unpack("<i", f.read(4))[0]

            spawn_x = []
            for _ in range(MAX_SPAWNS_C_API):
                spawn_x.append(struct.unpack("<f", f.read(4))[0])

            spawn_y = []
            for _ in range(MAX_SPAWNS_C_API):
                spawn_y.append(struct.unpack("<f", f.read(4))[0])

            spawn_facing = []
            for _ in range(MAX_SPAWNS_C_API):
                spawn_facing.append(struct.unpack("<f", f.read(4))[0])

            # Read fixed-size arrays (always MAX_TILES_C_API elements for C++ compatibility)
            object_ids = []
            for _ in range(MAX_TILES_C_API):
                object_ids.append(struct.unpack("<i", f.read(4))[0])

            tile_x = []
            for _ in range(MAX_TILES_C_API):
                tile_x.append(struct.unpack("<f", f.read(4))[0])

            tile_y = []
            for _ in range(MAX_TILES_C_API):
                tile_y.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_z array
            tile_z = []
            for _ in range(MAX_TILES_C_API):
                tile_z.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_persistent array
            tile_persistent = []
            for _ in range(MAX_TILES_C_API):
                tile_persistent.append(struct.unpack("<B", f.read(1))[0] != 0)

            # Read tile_render_only array
            tile_render_only = []
            for _ in range(MAX_TILES_C_API):
                tile_render_only.append(struct.unpack("<B", f.read(1))[0] != 0)

            # Read tile_entity_type array
            tile_entity_type = []
            for _ in range(MAX_TILES_C_API):
                tile_entity_type.append(struct.unpack("<i", f.read(4))[0])

            # Read tile_response_type array
            tile_response_type = []
            for _ in range(MAX_TILES_C_API):
                tile_response_type.append(struct.unpack("<i", f.read(4))[0])

            # Read tile_scale_x array
            tile_scale_x = []
            for _ in range(MAX_TILES_C_API):
                tile_scale_x.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_scale_y array
            tile_scale_y = []
            for _ in range(MAX_TILES_C_API):
                tile_scale_y.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_scale_z array
            tile_scale_z = []
            for _ in range(MAX_TILES_C_API):
                tile_scale_z.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_rot_w array
            tile_rot_w = []
            for _ in range(MAX_TILES_C_API):
                tile_rot_w.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_rot_x array
            tile_rot_x = []
            for _ in range(MAX_TILES_C_API):
                tile_rot_x.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_rot_y array
            tile_rot_y = []
            for _ in range(MAX_TILES_C_API):
                tile_rot_y.append(struct.unpack("<f", f.read(4))[0])

            # Read tile_rot_z array
            tile_rot_z = []
            for _ in range(MAX_TILES_C_API):
                tile_rot_z.append(struct.unpack("<f", f.read(4))[0])

            # Try to read new randomization fields if they exist
            # This provides backward compatibility with old files
            try:
                # Read tile_rand_x array
                tile_rand_x = []
                for _ in range(MAX_TILES_C_API):
                    tile_rand_x.append(struct.unpack("<f", f.read(4))[0])

                # Read tile_rand_y array
                tile_rand_y = []
                for _ in range(MAX_TILES_C_API):
                    tile_rand_y.append(struct.unpack("<f", f.read(4))[0])

                # Read tile_rand_z array
                tile_rand_z = []
                for _ in range(MAX_TILES_C_API):
                    tile_rand_z.append(struct.unpack("<f", f.read(4))[0])

                # Read tile_rand_rot_z array
                tile_rand_rot_z = []
                for _ in range(MAX_TILES_C_API):
                    tile_rand_rot_z.append(struct.unpack("<f", f.read(4))[0])
            except struct.error:
                # Old file format without randomization fields - use defaults (no randomization)
                tile_rand_x = [0.0] * MAX_TILES_C_API
                tile_rand_y = [0.0] * MAX_TILES_C_API
                tile_rand_z = [0.0] * MAX_TILES_C_API
                tile_rand_rot_z = [0.0] * MAX_TILES_C_API

            # Calculate expected array size for this level's dimensions
            array_size = width * height

            # Construct dictionary matching compile_level() output
            compiled = {
                "num_tiles": num_tiles,
                "max_entities": max_entities,
                "width": width,
                "height": height,
                "scale": scale,
                "level_name": level_name,
                "num_spawns": num_spawns,
                "spawn_x": spawn_x,
                "spawn_y": spawn_y,
                "spawn_facing": spawn_facing,
                "array_size": array_size,  # Add calculated array size
                "object_ids": object_ids,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_z": tile_z,
                "tile_persistent": tile_persistent,
                "tile_render_only": tile_render_only,
                "tile_entity_type": tile_entity_type,
                "tile_response_type": tile_response_type,
                "tile_scale_x": tile_scale_x,
                "tile_scale_y": tile_scale_y,
                "tile_scale_z": tile_scale_z,
                "tile_rot_w": tile_rot_w,
                "tile_rot_x": tile_rot_x,
                "tile_rot_y": tile_rot_y,
                "tile_rot_z": tile_rot_z,
                "tile_rand_x": tile_rand_x,
                "tile_rand_y": tile_rand_y,
                "tile_rand_z": tile_rand_z,
                "tile_rand_rot_z": tile_rand_rot_z,
            }

            # Validate loaded data
            validate_compiled_level(compiled)
            return compiled

    except IOError as e:
        raise IOError(f"Failed to read level file '{filepath}': {e}")
    except struct.error as e:
        raise ValueError(f"Invalid level file format '{filepath}': {e}")


def compile_level_from_json(json_data: Union[str, Dict]) -> Dict:
    """
    Compile level from JSON format with parameters.

    JSON formats supported:

    1. Simple format (uses default tileset):
    {
        "ascii": "Multi-line ASCII level string",
        "name": "level_name",  // Optional, default "unknown_level"
        "scale": 2.5,  // Optional, default 2.5
        "agent_facing": [0.0, 1.57],  // Optional, radians for each agent
    }

    2. Tileset format (custom asset mapping):
    {
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

    Args:
        json_data: JSON string or dict containing level data

    Returns:
        Dict matching compile_level() output format

    Raises:
        ValueError: If JSON is invalid or missing required fields

    Example:
        # Simple format
        json_level = {
            "ascii": "###\\n#S#\\n###",
            "scale": 1.5,
            "agent_facing": [math.pi/2]  # Face right
        }
        compiled = compile_level_from_json(json_level)

        # Tileset format
        json_level = {
            "ascii": "###O###\\n#S...C#\\n#######",
            "tileset": {
                "#": {"asset": "wall"},
                "C": {"asset": "cube"},
                "O": {"asset": "cylinder"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            }
        }
        compiled = compile_level_from_json(json_level)
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        data = json_data

    # Validate required fields
    if "ascii" not in data:
        raise ValueError("JSON level must contain 'ascii' field with level layout")

    # Extract parameters with defaults
    ascii_str = data["ascii"]
    scale = data.get("scale", 2.5)
    agent_facing = data.get("agent_facing", None)
    level_name = data.get("name", "unknown_level")
    tileset = data.get("tileset", None)

    # If no tileset provided but ASCII contains special characters, use default tileset
    if tileset is None:
        # Check if ASCII contains characters that require tileset (e.g., 'O' for cylinder)
        legacy_char_map = _get_legacy_char_map()
        special_chars = set()
        for line in ascii_str.split("\n"):
            for char in line:
                if char not in legacy_char_map and char.strip():  # Non-empty, not in legacy map
                    special_chars.add(char)

        # If special characters found, use default tileset
        if special_chars:
            tileset = DEFAULT_TILESET

    # Validate parameters
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError(f"Invalid scale: {scale} (must be positive number)")

    if agent_facing is not None:
        if not isinstance(agent_facing, list):
            raise ValueError("agent_facing must be a list of angles in radians")
        for i, angle in enumerate(agent_facing):
            if not isinstance(angle, (int, float)):
                raise ValueError(f"Invalid agent_facing[{i}]: {angle} (must be number)")

    if tileset is not None:
        _validate_tileset(tileset)

    # Compile with parameters
    return compile_level(ascii_str, scale, agent_facing, level_name, tileset)


def compile_level_to_binary(ascii_input: str, binary_output: str, scale: float = 2.5) -> None:
    """
    Compile ASCII or JSON level file to binary .lvl file.

    Args:
        ascii_input: Path to ASCII/JSON level file or level string
        binary_output: Path to output .lvl file
        scale: World units per ASCII character (ignored if JSON file)

    Raises:
        IOError: If files cannot be read/written
        ValueError: If level compilation fails
    """
    # Check if input is a file path or level string
    if Path(ascii_input).exists():
        # Read from file
        with open(ascii_input, "r") as f:
            content = f.read()

        # Check if it's JSON format (file extension or content)
        if ascii_input.endswith(".json") or content.strip().startswith("{"):
            # Compile from JSON
            compiled = compile_level_from_json(content)
        else:
            # Compile from ASCII
            compiled = compile_level(content, scale)
    else:
        # Treat as level string - check if JSON
        if ascii_input.strip().startswith("{"):
            compiled = compile_level_from_json(ascii_input)
        else:
            compiled = compile_level(ascii_input, scale)

    # Save to binary file
    save_compiled_level_binary(compiled, binary_output)


def print_level_info(compiled: Dict) -> None:
    """Print compiled level information for debugging."""
    print("Compiled Level Info:")
    print(f"  Dimensions: {compiled['width']}x{compiled['height']} (scale: {compiled['scale']})")
    print(f"  Tiles: {compiled['num_tiles']}")
    print(f"  Max entities: {compiled['max_entities']}")

    if "_spawn_points" in compiled:
        print(f"  Spawn points: {len(compiled['_spawn_points'])}")
        for i, (x, y) in enumerate(compiled["_spawn_points"]):
            facing_rad = compiled.get("spawn_facing", [0.0] * 8)[i]
            facing_deg = facing_rad * 180.0 / math.pi
            print(f"    Spawn {i}: ({x:.1f}, {y:.1f}) facing {facing_deg:.1f}°")

    if "_entity_count" in compiled:
        print(f"  Physics entities: {compiled['_entity_count']}")


def main():
    """Command line interface for level compiler."""
    parser = argparse.ArgumentParser(
        description="Compile ASCII level files to binary .lvl format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile ASCII file to binary level
  python -m madrona_escape_room.level_compiler maze.txt maze.lvl

  # Compile with custom scale
  python -m madrona_escape_room.level_compiler maze.txt maze.lvl --scale 1.5

  # Test mode - run built-in tests
  python -m madrona_escape_room.level_compiler --test

  # Load and display info about binary level
  python -m madrona_escape_room.level_compiler --info maze.lvl
        """,
    )

    parser.add_argument("input", nargs="?", help="Input ASCII level file")
    parser.add_argument("output", nargs="?", help="Output binary .lvl file")
    parser.add_argument(
        "--scale", type=float, default=2.5, help="World units per ASCII character (default: 2.5)"
    )
    parser.add_argument("--test", action="store_true", help="Run built-in test suite")
    parser.add_argument("--info", metavar="FILE", help="Display info about binary level file")

    args = parser.parse_args()

    try:
        if args.test:
            # Run built-in test suite
            run_test_suite()
        elif args.info:
            # Display info about binary level file
            print(f"Loading binary level: {args.info}")
            compiled = load_compiled_level_binary(args.info)
            print_level_info(compiled)
        elif args.input and args.output:
            # Compile ASCII to binary
            print(f"Compiling '{args.input}' to '{args.output}' (scale: {args.scale})")
            compile_level_to_binary(args.input, args.output, args.scale)
            print("✓ Level compiled successfully")

            # Load and display info
            compiled = load_compiled_level_binary(args.output)
            print_level_info(compiled)
        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_test_suite():
    """Run built-in test suite."""
    print("=== Level Compiler Test Suite ===")

    # Test 1: Simple room
    print("\n=== Test 1: Simple Room ===")
    simple_room = """
    ##########
    #S.......#
    #........#
    #........#
    ##########
    """

    try:
        compiled = compile_level(simple_room)
        print_level_info(compiled)
        validate_compiled_level(compiled)
        print("✓ Simple room compiled and validated successfully")
    except Exception as e:
        print(f"✗ Simple room failed: {e}")

    # Test 2: Obstacle course
    print("\n=== Test 2: Obstacle Course ===")
    obstacle_course = """
    ############
    #S.........#
    #..###.....#
    #..#.#.CC..#
    #..#.#.....#
    #..#.......#
    ############
    """

    try:
        compiled = compile_level(obstacle_course)
        print_level_info(compiled)
        validate_compiled_level(compiled)
        print("✓ Obstacle course compiled and validated successfully")
    except Exception as e:
        print(f"✗ Obstacle course failed: {e}")

    # Test 3: Binary I/O test
    print("\n=== Test 3: Binary I/O ===")
    try:
        # Compile to dict
        compiled = compile_level(simple_room)

        # Save to binary
        test_file = "/tmp/test_level.lvl"
        save_compiled_level_binary(compiled, test_file)
        print(f"✓ Saved binary level to {test_file}")

        # Load from binary
        loaded = load_compiled_level_binary(test_file)
        print("✓ Loaded binary level successfully")

        # Verify data matches
        for key in ["num_tiles", "max_entities", "width", "height", "scale"]:
            if compiled[key] != loaded[key]:
                raise ValueError(f"Mismatch in {key}: {compiled[key]} != {loaded[key]}")

        for i in range(compiled["num_tiles"]):
            if (
                compiled["object_ids"][i] != loaded["object_ids"][i]
                or abs(compiled["tile_x"][i] - loaded["tile_x"][i]) > 0.001
                or abs(compiled["tile_y"][i] - loaded["tile_y"][i]) > 0.001
            ):
                raise ValueError(f"Mismatch in tile {i}")

        print("✓ Binary I/O round-trip successful")

        # Clean up
        Path(test_file).unlink()

    except Exception as e:
        print(f"✗ Binary I/O test failed: {e}")

    # Test 4: JSON format test
    print("\n=== Test 4: JSON Format ===")
    json_level = {
        "ascii": """#####
#S.S#
#####""",
        "scale": 1.5,
        "agent_facing": [0.0, math.pi / 2],  # First agent faces forward, second faces right
    }

    try:
        compiled = compile_level_from_json(json_level)
        print_level_info(compiled)
        validate_compiled_level(compiled)
        print("✓ JSON level compiled and validated successfully")

        # Verify agent facing was set
        if (
            compiled["spawn_facing"][0] == 0.0
            and abs(compiled["spawn_facing"][1] - math.pi / 2) < 0.001
        ):
            print("✓ Agent facing angles correctly set")
        else:
            print("✗ Agent facing angles incorrect")
    except Exception as e:
        print(f"✗ JSON format test failed: {e}")

    # Test 5: Error cases
    print("\n=== Test 5: Error Cases ===")

    # Empty level
    try:
        compile_level("")
        print("✗ Empty level should have failed")
    except ValueError as e:
        print(f"✓ Empty level correctly rejected: {e}")

    # No spawn point
    try:
        compile_level("""
        #####
        #...#
        #####
        """)
        print("✗ No spawn point should have failed")
    except ValueError as e:
        print(f"✓ No spawn point correctly rejected: {e}")

    # Unknown character
    try:
        compile_level("""
        #####
        #S.X#
        #####
        """)
        print("✗ Unknown character should have failed")
    except ValueError as e:
        print(f"✓ Unknown character correctly rejected: {e}")

    print("\n🎉 Level compiler tests completed!")


# Example usage and test cases
if __name__ == "__main__":
    main()
