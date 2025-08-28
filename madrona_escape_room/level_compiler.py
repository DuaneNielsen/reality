#!/usr/bin/env python3
"""
JSON Level Compiler for Madrona Escape Room

Compiles JSON level definitions to CompiledLevel format for GPU processing.
This compiler exclusively accepts JSON format with explicit tileset definitions.

Required JSON format:
{
    "ascii": "###\\n#S#\\n###",
    "tileset": {
        "#": {"asset": "wall"},
        "C": {"asset": "cube"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,              # Optional, default 2.5
    "agent_facing": [0.0],      # Optional, radians for each agent
    "name": "level_name"        # Optional, default "unknown_level"
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
    "#": {"asset": "wall"},
    "C": {"asset": "cube"},
    "O": {"asset": "cylinder"},
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


def _validate_json_level(data: Dict) -> None:
    """
    Validate JSON level data structure.

    Args:
        data: JSON level dictionary

    Raises:
        ValueError: If JSON structure is invalid
    """
    # Required fields
    if "ascii" not in data:
        raise ValueError("JSON level must contain 'ascii' field")

    if "tileset" not in data:
        raise ValueError("JSON level must contain 'tileset' field")

    # Validate ascii field
    if not isinstance(data["ascii"], str):
        raise ValueError("'ascii' field must be a string")

    if not data["ascii"].strip():
        raise ValueError("'ascii' field cannot be empty")

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

    if "name" in data:
        name = data["name"]
        if not isinstance(name, str):
            raise ValueError("'name' field must be a string")
        if len(name) > MAX_LEVEL_NAME_LENGTH:
            raise ValueError(f"Level name too long: {len(name)} > {MAX_LEVEL_NAME_LENGTH}")


def _process_tileset(tileset: Dict) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """
    Process tileset to extract object IDs and randomization parameters.

    Args:
        tileset: Validated tileset dictionary

    Returns:
        Tuple of (char_to_tile mapping, char_to_rand parameters)

    Raises:
        ValueError: If asset cannot be resolved
    """
    char_to_tile = {}
    char_to_rand = {}

    for char, tile_def in tileset.items():
        asset_name = tile_def["asset"]

        # Get object ID for this asset
        try:
            char_to_tile[char] = _get_asset_object_id(asset_name)
        except ValueError as e:
            raise ValueError(f"Invalid asset '{asset_name}' for character '{char}': {e}")

        # Extract randomization parameters
        char_to_rand[char] = {
            "rand_x": tile_def.get("rand_x", 0.0),
            "rand_y": tile_def.get("rand_y", 0.0),
            "rand_z": tile_def.get("rand_z", 0.0),
            "rand_rot_z": tile_def.get("rand_rot_z", 0.0),
        }

    return char_to_tile, char_to_rand


def _parse_ascii_to_tiles(
    ascii_str: str,
    char_to_tile: Dict[str, int],
    char_to_rand: Dict[str, Dict],
    scale: float,
    width: int,
    height: int,
) -> Tuple[List[Tuple], List[Tuple], int]:
    """
    Parse ASCII string to tile and spawn data.

    Args:
        ascii_str: ASCII level string
        char_to_tile: Character to object ID mapping
        char_to_rand: Character to randomization parameters
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
        "#": {"asset": "wall"},
        "C": {"asset": "cube"},
        "O": {"asset": "cylinder"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"},
        " ": {"asset": "empty"},
    }

    # Build JSON structure
    json_data = {"ascii": ascii_str, "tileset": default_tileset, "scale": scale, "name": level_name}

    # Add agent facing if provided
    if agent_facing is not None:
        json_data["agent_facing"] = agent_facing

    # Compile using the main compile_level function
    return compile_level(json_data)


def compile_level(json_data: Union[str, Dict]) -> CompiledLevel:
    """
    Compile JSON level definition to CompiledLevel format.

    Args:
        json_data: JSON string or dictionary with level definition

    Returns:
        CompiledLevel struct ready for C API

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

    # Extract fields
    ascii_str = data["ascii"]
    tileset = data["tileset"]
    scale = data.get("scale", 2.5)
    agent_facing = data.get("agent_facing", None)
    level_name = data.get("name", "unknown_level")

    # Process tileset to get mappings
    char_to_tile, char_to_rand = _process_tileset(tileset)

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

    # Calculate total array size
    array_size = width * height
    if array_size > MAX_TILES:
        raise ValueError(
            f"Level too large: {width}×{height} = {array_size} tiles > {MAX_TILES} max"
        )

    # Parse ASCII to tiles and spawns
    tiles, spawns, entity_count = _parse_ascii_to_tiles(
        ascii_str, char_to_tile, char_to_rand, scale, width, height
    )

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

        # Set randomization parameters
        rand_params = char_to_rand.get(char, {})
        level.tile_rand_x[i] = rand_params.get("rand_x", 0.0)
        level.tile_rand_y[i] = rand_params.get("rand_y", 0.0)
        level.tile_rand_z[i] = rand_params.get("rand_z", 0.0)
        level.tile_rand_rot_z[i] = rand_params.get("rand_rot_z", 0.0)

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

    return level


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


def save_compiled_level_binary(compiled: CompiledLevel, filepath: str) -> None:
    """
    Save compiled level struct to binary .lvl file using C API.

    Args:
        compiled: CompiledLevel struct from compile_level()
        filepath: Path to save .lvl file

    Raises:
        IOError: If file cannot be written
    """
    from .ctypes_bindings import lib
    from .generated_constants import Result

    result = lib.mer_write_compiled_level(filepath.encode("utf-8"), ctypes.byref(compiled))

    if result != Result.Success:
        raise IOError(f"Failed to write level file: {filepath} (error code: {result})")


def load_compiled_level_binary(filepath: str) -> CompiledLevel:
    """
    Load compiled level struct from binary .lvl file using C API.

    Args:
        filepath: Path to .lvl file

    Returns:
        CompiledLevel struct

    Raises:
        IOError: If file cannot be read
    """
    from .ctypes_bindings import lib
    from .generated_constants import Result

    level = create_compiled_level()
    result = lib.mer_read_compiled_level(filepath.encode("utf-8"), ctypes.byref(level))

    if result != Result.Success:
        raise IOError(f"Failed to read level file: {filepath} (error code: {result})")

    return level


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
                    "#": {"asset": "wall"},
                    "C": {"asset": "cube"},
                    "S": {"asset": "spawn"},
                    ".": {"asset": "empty"},
                },
                "scale": 2.5,
                "agent_facing": [math.pi / 2],  # Face right
                "name": "test_level",
            }

            print("Compiling test level...")
            compiled = compile_level(test_json)
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
            compiled = compile_level(json_data)
            validate_compiled_level(compiled)

            # Save to binary
            save_compiled_level_binary(compiled, args.output)
            print("✓ Level compiled successfully")

            # Display info
            print_level_info(compiled)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
