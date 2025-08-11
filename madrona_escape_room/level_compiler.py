#!/usr/bin/env python3
"""
Python Level Compiler for Madrona Escape Room

Compiles ASCII level strings to CompiledLevel format for GPU processing.
Part of the test-driven level system that allows tests to define their
environment layouts using visual ASCII art.
"""

import math
from typing import Dict, List, Optional, Tuple

# Tile type constants (must match C++ TileType enum in src/level_gen.cpp)
TILE_EMPTY = 0
TILE_WALL = 1
TILE_CUBE = 2
TILE_SPAWN = 3
TILE_DOOR = 4  # Future
TILE_BUTTON = 5  # Future
TILE_GOAL = 6  # Future

# Character to tile type mapping
CHAR_MAP = {
    ".": TILE_EMPTY,
    " ": TILE_EMPTY,  # Whitespace also means empty
    "#": TILE_WALL,
    "C": TILE_CUBE,
    "S": TILE_SPAWN,
    # Future expansions:
    # 'D': TILE_DOOR,
    # 'B': TILE_BUTTON,
    # 'G': TILE_GOAL
}


def compile_level(ascii_str: str, scale: float = 2.0) -> Dict:
    """
    Compile ASCII level string to dict matching MER_CompiledLevel struct.

    Args:
        ascii_str: Multi-line ASCII level definition using the character map above
        scale: World units per ASCII character (default 2.0 units per cell)

    Returns:
        Dict with fields matching MER_CompiledLevel C struct:
        - num_tiles: Number of entities to create
        - max_entities: BVH size hint (includes persistent entities)
        - width, height: Grid dimensions
        - scale: World scale factor
        - tile_types, tile_x, tile_y: Arrays of tile data (256 elements each)

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
        compiled = compile_level(level)
        # compiled['num_tiles'] = 38 (wall tiles)
        # compiled['max_entities'] = 88 (walls + buffer for agents/floor/etc)
    """
    # Clean up input - remove leading/trailing whitespace and normalize line endings
    lines = [line.rstrip() for line in ascii_str.strip().split("\n")]
    if not lines or all(len(line) == 0 for line in lines):
        raise ValueError("Empty level string")

    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0

    if width == 0 or height == 0:
        raise ValueError("Level has zero width or height")

    tiles = []
    spawns = []
    entity_count = 0  # Count entities that need physics bodies

    # Parse ASCII grid to extract tile positions and types
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char in CHAR_MAP:
                tile_type = CHAR_MAP[char]

                # Convert grid coordinates to world coordinates (center at origin)
                world_x = (x - width / 2.0) * scale
                world_y = (y - height / 2.0) * scale

                if tile_type == TILE_SPAWN:
                    spawns.append((world_x, world_y))
                    # Don't add spawn tiles to tile array - handled by agent placement
                elif tile_type != TILE_EMPTY:
                    tiles.append((world_x, world_y, tile_type))

                    # Count entities that will need physics bodies and BVH slots
                    if tile_type in [TILE_WALL, TILE_CUBE]:
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
    if len(tiles) > 256:
        raise ValueError(f"Too many tiles: {len(tiles)} > 256 (CompiledLevel::MAX_TILES)")
    if len(spawns) == 0:
        raise ValueError("No spawn points (S) found in level - at least one required")

    # Build arrays matching C struct layout (256 elements each, zero-padded)
    tile_types = [0] * 256
    tile_x = [0.0] * 256
    tile_y = [0.0] * 256

    # Fill arrays with actual tile data
    for i, (world_x, world_y, tile_type) in enumerate(tiles):
        tile_types[i] = tile_type
        tile_x[i] = world_x
        tile_y[i] = world_y

    # Return dict matching MER_CompiledLevel struct layout
    return {
        "num_tiles": len(tiles),
        "max_entities": max_entities,
        "width": width,
        "height": height,
        "scale": scale,
        "tile_types": tile_types,
        "tile_x": tile_x,
        "tile_y": tile_y,
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
        "tile_types",
        "tile_x",
        "tile_y",
    ]
    for field in required_fields:
        if field not in compiled:
            raise ValueError(f"Missing required field: {field}")

    if compiled["num_tiles"] < 0 or compiled["num_tiles"] > 256:
        raise ValueError(f"Invalid num_tiles: {compiled['num_tiles']} (must be 0-256)")

    if compiled["max_entities"] <= 0:
        raise ValueError(f"Invalid max_entities: {compiled['max_entities']} (must be > 0)")

    if compiled["width"] <= 0 or compiled["height"] <= 0:
        raise ValueError(f"Invalid dimensions: {compiled['width']}x{compiled['height']}")

    if compiled["scale"] <= 0.0:
        raise ValueError(f"Invalid scale: {compiled['scale']} (must be > 0)")

    # Check array lengths
    for array_name in ["tile_types", "tile_x", "tile_y"]:
        if len(compiled[array_name]) != 256:
            raise ValueError(
                f"Invalid {array_name} array length: {len(compiled[array_name])} (must be 256)"
            )


def print_level_info(compiled: Dict) -> None:
    """Print compiled level information for debugging."""
    print("Compiled Level Info:")
    print(f"  Dimensions: {compiled['width']}x{compiled['height']} (scale: {compiled['scale']})")
    print(f"  Tiles: {compiled['num_tiles']}")
    print(f"  Max entities: {compiled['max_entities']}")

    if "_spawn_points" in compiled:
        print(f"  Spawn points: {len(compiled['_spawn_points'])}")
        for i, (x, y) in enumerate(compiled["_spawn_points"]):
            print(f"    Spawn {i}: ({x:.1f}, {y:.1f})")

    if "_entity_count" in compiled:
        print(f"  Physics entities: {compiled['_entity_count']}")


# Example usage and test cases
if __name__ == "__main__":
    # Test 1: Simple room
    print("=== Test 1: Simple Room ===")
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

    # Test 3: Error cases
    print("\n=== Test 3: Error Cases ===")

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
