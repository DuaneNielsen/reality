#!/usr/bin/env python3
"""
Design a 16x64 obstacle course level with randomized elements.
This level features multiple rooms, corridors, and varied obstacles.
"""

import json
import math

from madrona_escape_room.level_compiler import compile_level
from madrona_escape_room.level_io import save_compiled_level


def create_obstacle_course_level():
    """Create a 20x48 obstacle course with winding paths and open spaces."""

    # Design the ASCII level - 20 characters wide, 48 lines tall (20*48 = 960 tiles)
    level_ascii = """
####################
#..............O...#
#..................#
#########..#########
#..................#
#...O.....O....C...#
#..................#
#.....C.......##...#
#..............#...#
#......O.......#...#
############...#...#
#......C.......#...#
#..O...........#...#
#............C.#...#
#...############...#
#..................#
#..................#
#####...############
#...#..............#
#...#..............#
#...#####..#####...#
#..................#
#....C.......O.....#
#..................#
#...#########......#
#...........#...C..#
#...........#......#
#############......#
#..................#
#..O....C....C.....#
#..................#
#......#########...#
#......#...........#
#......#...........#
#......#...#########
#......#...........#
#......#...........#
#......#########...#
#..................#
#....O.......C.....#
#..................#
#...############...#
#..................#
#..................#
#########..#########
#..................#
#.............S....#
####################"""

    # Define tileset with randomization parameters
    tileset = {
        "#": {
            "asset": "wall",
            "done_on_collision": False,
            "rand_x": 0.0,  # No randomization for walls to maintain structural integrity
            "rand_y": 0.0,
            "rand_z": 0.0,
            "rand_rot_z": 0.0,
        },
        "C": {
            "asset": "cube",
            "done_on_collision": True,
            "rand_x": 0.8,  # High randomization for cubes
            "rand_y": 0.8,
            "rand_z": 0.0,  # Keep at ground level
            "rand_rot_z": 3.14159,  # Random rotation up to 180 degrees
        },
        "O": {
            "asset": "cylinder",
            "done_on_collision": True,
            "rand_x": 0.6,  # Medium randomization for cylinders
            "rand_y": 0.6,
            "rand_z": 0.0,
            "rand_rot_z": 0.0,  # Cylinders don't need rotation randomization
        },
        "S": {
            "asset": "spawn",
            "rand_x": 0.5,  # Small spawn randomization
            "rand_y": 0.5,
            "rand_z": 0.0,
            "rand_rot_z": 0.0,
        },
        ".": {"asset": "empty"},
    }

    # Create the JSON level definition
    level_data = {
        "ascii": level_ascii,
        "tileset": tileset,
        "scale": 2.0,  # Slightly smaller scale for tighter spaces
        "agent_facing": [0.0],  # Start facing forward (north)
        "name": "obstacle_course_20x48",
    }

    return level_data


def main():
    print("Creating 20x48 obstacle course level...")

    # Create level data
    level_data = create_obstacle_course_level()

    # Print level statistics
    ascii_lines = level_data["ascii"].strip().split("\n")
    width = len(ascii_lines[0]) if ascii_lines else 0
    height = len(ascii_lines)

    print(f"Level dimensions: {width}x{height}")

    # Count different tile types
    ascii_str = level_data["ascii"]
    cube_count = ascii_str.count("C")
    cylinder_count = ascii_str.count("O")
    wall_count = ascii_str.count("#")
    spawn_count = ascii_str.count("S")

    print(f"Obstacles: {cube_count} cubes, {cylinder_count} cylinders")
    print(f"Walls: {wall_count}, Spawns: {spawn_count}")

    # Compile the level
    print("\nCompiling level...")
    compiled_level = compile_level(level_data)

    # Save to file
    filename = "obstacle_course_20x48.lvl"
    save_compiled_level(compiled_level, filename)
    print(f"✓ Saved level to {filename}")

    # Print compiled level info
    print("\nCompiled Level Info:")
    print(f"  Name: {compiled_level.level_name.decode('utf-8', errors='ignore')}")
    dims = f"{compiled_level.width}x{compiled_level.height}"
    scale = compiled_level.world_scale
    print(f"  Dimensions: {dims} (scale: {scale})")
    print(f"  Tiles: {compiled_level.num_tiles}")
    print(f"  Max entities: {compiled_level.max_entities}")
    print(f"  Spawn points: {compiled_level.num_spawns}")

    for i in range(compiled_level.num_spawns):
        x = compiled_level.spawn_x[i]
        y = compiled_level.spawn_y[i]
        facing_rad = compiled_level.spawn_facing[i]
        facing_deg = facing_rad * 180.0 / math.pi
        print(f"    Spawn {i}: ({x:.1f}, {y:.1f}) facing {facing_deg:.1f}°")

    print("\nWorld bounds:")
    print(f"  X: [{compiled_level.world_min_x:.1f}, {compiled_level.world_max_x:.1f}]")
    print(f"  Y: [{compiled_level.world_min_y:.1f}, {compiled_level.world_max_y:.1f}]")
    print(f"  Z: [{compiled_level.world_min_z:.1f}, {compiled_level.world_max_z:.1f}]")


if __name__ == "__main__":
    main()
