#!/usr/bin/env python3
"""
Python Level Compiler for Madrona Escape Room

Compiles ASCII level strings to CompiledLevel format for GPU processing.
Part of the test-driven level system that allows tests to define their
environment layouts using visual ASCII art.
"""

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _get_max_tiles_from_c_api():
    """Get MAX_TILES from C API - returns the actual CompiledLevel::MAX_TILES value"""
    try:
        from .ctypes_bindings import _get_max_tiles

        return _get_max_tiles()
    except ImportError:
        # Fallback if ctypes bindings not available
        return 1024


# Get the authoritative MAX_TILES value from C++
MAX_TILES_C_API = _get_max_tiles_from_c_api()

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
        - tile_types, tile_x, tile_y: Arrays of tile data (MAX_TILES_C_API elements each)

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
    if len(tiles) > array_size:
        raise ValueError(
            f"Too many non-empty tiles: {len(tiles)} > {array_size} (level dimensions)"
        )
    if len(spawns) == 0:
        raise ValueError("No spawn points (S) found in level - at least one required")

    # Build arrays with MAX_TILES_C_API size for C++ compatibility
    tile_types = [0] * MAX_TILES_C_API
    tile_x = [0.0] * MAX_TILES_C_API
    tile_y = [0.0] * MAX_TILES_C_API

    # Fill arrays with actual tile data
    for i, (world_x, world_y, tile_type) in enumerate(tiles):
        tile_types[i] = tile_type
        tile_x[i] = world_x
        tile_y[i] = world_y

    # Return dict with compiler-calculated array sizing
    return {
        "num_tiles": len(tiles),
        "max_entities": max_entities,
        "width": width,
        "height": height,
        "scale": scale,
        "array_size": array_size,  # NEW: compiler-calculated array size (width × height)
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
        "array_size",
        "tile_types",
        "tile_x",
        "tile_y",
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
    for array_name in ["tile_types", "tile_x", "tile_y"]:
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

            # Arrays: Write fixed-size arrays for C++ compatibility
            # Always write MAX_TILES_C_API elements regardless of actual array size
            array_size = compiled["array_size"]

            # tile_types array - pad with zeros if needed
            for i in range(MAX_TILES_C_API):
                if i < array_size:
                    f.write(struct.pack("<i", compiled["tile_types"][i]))
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

            # Read fixed-size arrays (always MAX_TILES_C_API elements for C++ compatibility)
            tile_types = []
            for _ in range(MAX_TILES_C_API):
                tile_types.append(struct.unpack("<i", f.read(4))[0])

            tile_x = []
            for _ in range(MAX_TILES_C_API):
                tile_x.append(struct.unpack("<f", f.read(4))[0])

            tile_y = []
            for _ in range(MAX_TILES_C_API):
                tile_y.append(struct.unpack("<f", f.read(4))[0])

            # Calculate expected array size for this level's dimensions
            array_size = width * height

            # Construct dictionary matching compile_level() output
            compiled = {
                "num_tiles": num_tiles,
                "max_entities": max_entities,
                "width": width,
                "height": height,
                "scale": scale,
                "array_size": array_size,  # Add calculated array size
                "tile_types": tile_types,
                "tile_x": tile_x,
                "tile_y": tile_y,
            }

            # Validate loaded data
            validate_compiled_level(compiled)
            return compiled

    except IOError as e:
        raise IOError(f"Failed to read level file '{filepath}': {e}")
    except struct.error as e:
        raise ValueError(f"Invalid level file format '{filepath}': {e}")


def compile_level_to_binary(ascii_input: str, binary_output: str, scale: float = 2.0) -> None:
    """
    Compile ASCII level file to binary .lvl file.

    Args:
        ascii_input: Path to ASCII level file or ASCII string
        binary_output: Path to output .lvl file
        scale: World units per ASCII character

    Raises:
        IOError: If files cannot be read/written
        ValueError: If level compilation fails
    """
    # Check if input is a file path or ASCII string
    if Path(ascii_input).exists():
        # Read from file
        with open(ascii_input, "r") as f:
            ascii_str = f.read()
    else:
        # Treat as ASCII string
        ascii_str = ascii_input

    # Compile to dictionary
    compiled = compile_level(ascii_str, scale)

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
            print(f"    Spawn {i}: ({x:.1f}, {y:.1f})")

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
        "--scale", type=float, default=2.0, help="World units per ASCII character (default: 2.0)"
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
                compiled["tile_types"][i] != loaded["tile_types"][i]
                or abs(compiled["tile_x"][i] - loaded["tile_x"][i]) > 0.001
                or abs(compiled["tile_y"][i] - loaded["tile_y"][i]) > 0.001
            ):
                raise ValueError(f"Mismatch in tile {i}")

        print("✓ Binary I/O round-trip successful")

        # Clean up
        Path(test_file).unlink()

    except Exception as e:
        print(f"✗ Binary I/O test failed: {e}")

    # Test 4: Error cases
    print("\n=== Test 4: Error Cases ===")

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
