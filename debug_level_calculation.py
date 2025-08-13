#!/usr/bin/env python3
"""
Debug the level compiler calculation for our test level
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from madrona_escape_room.level_compiler import compile_level


def debug_level():
    level_ascii = """
#####
#S..#
#...#
#####
"""

    compiled = compile_level(level_ascii)

    print("Level analysis:")
    total_tiles = compiled["width"] * compiled["height"]
    print(f"Dimensions: {compiled['width']}x{compiled['height']} = {total_tiles} tiles")
    print("Entities with ObjectID that need BVH slots:")
    print(f"  - Level tiles (walls): {compiled['num_tiles']}")
    print("  - Persistent entities: 5 (1 floor + 1 agent + 3 origin markers)")
    print("  - Physics buffer: calculated based on level size")
    print("  - Calculation method: max_possible_tiles + persistent + buffer")
    print(f"  - Total max_entities: {compiled['max_entities']}")

    # Count actual walls
    wall_count = 0
    for i, tile_type in enumerate(compiled["tile_types"][: compiled["num_tiles"]]):
        if tile_type == 1:  # TILE_WALL
            wall_count += 1

    print(f"\nActual wall count: {wall_count}")
    print(f"Expected entities per world: {wall_count} walls + 5 persistent = {wall_count + 5}")
    print(f"With buffer: {wall_count + 5 + 50}")

    return compiled


if __name__ == "__main__":
    debug_level()
