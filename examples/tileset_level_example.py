#!/usr/bin/env python3
"""
Example demonstrating tileset-based level creation

This example shows how to create levels using custom tilesets that map
ASCII characters to asset names from the asset registry.
"""

import json

from madrona_escape_room import SimManager, madrona
from madrona_escape_room.level_compiler import compile_level_from_json


def example_simple_tileset():
    """Example using a simple tileset with basic assets"""

    # Define a level with custom tileset
    level_json = {
        "ascii": """
##########
#S.......#
#..CCC...#
#........#
##########
        """.strip(),
        "tileset": {
            "#": {"asset": "wall"},  # Wall blocks
            "C": {"asset": "cube"},  # Pushable cubes
            "S": {"asset": "spawn"},  # Agent spawn point
            ".": {"asset": "empty"},  # Empty space
        },
        "scale": 2.5,
        "name": "simple_tileset_level",
    }

    # Create simulation with the level (pass JSON as string)
    manager = SimManager(
        exec_mode=madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        level_ascii=json.dumps(level_json),
    )

    # Also compile it to get metadata
    compiled = compile_level_from_json(level_json)

    print(f"Created level '{compiled['level_name']}' with {compiled['num_tiles']} tiles")

    # Run a few steps
    for i in range(10):
        manager.step()

    return manager


def example_advanced_tileset():
    """Example using tileset with cylinder and other assets"""

    # Define a level with cylinders as obstacles
    level_json = {
        "ascii": """
############
#S.........#
#..O...O...#
#..........#
#..O...O...#
#....C.....#
############
        """.strip(),
        "tileset": {
            "#": {"asset": "wall"},  # Wall blocks
            "C": {"asset": "cube"},  # Pushable cube
            "O": {"asset": "cylinder"},  # Cylinder obstacles
            "S": {"asset": "spawn"},  # Agent spawn point
            ".": {"asset": "empty"},  # Empty space
        },
        "scale": 3.0,
        "name": "cylinder_obstacle_course",
        "agent_facing": [0.0],  # Agent faces forward
    }

    # Create simulation
    manager = SimManager(
        exec_mode=madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        level_ascii=json.dumps(level_json),
    )

    # Also compile it to get metadata
    compiled = compile_level_from_json(level_json)

    print(f"Created level '{compiled['level_name']}' with cylinders as obstacles")

    return manager


def example_auto_tileset():
    """Example showing automatic tileset usage for special characters"""

    # When using special characters like 'O' that aren't in the legacy CHAR_MAP,
    # the compiler automatically uses the DEFAULT_TILESET
    level_json = {
        "ascii": """
###O###
#S...C#
#######
        """.strip(),
        # No tileset specified - will use DEFAULT_TILESET automatically
        "scale": 2.0,
    }

    compiled = compile_level_from_json(level_json)

    print("Level compiled with automatic tileset for special character 'O' (cylinder)")

    return compiled


def example_custom_mapping():
    """Example with creative character mappings"""

    # You can map any character to any asset
    level_json = {
        "ascii": """
++++++++++
+@.......+
+..***...+
+........+
++++++++++
        """.strip(),
        "tileset": {
            "+": {"asset": "wall"},  # Use + for walls
            "*": {"asset": "cube"},  # Use * for cubes
            "@": {"asset": "spawn"},  # Use @ for spawn
            ".": {"asset": "empty"},  # Keep . for empty
        },
        "scale": 2.5,
        "name": "creative_characters",
    }

    compiled = compile_level_from_json(level_json)

    print("Level with creative character mapping: + for walls, * for cubes, @ for spawn")

    return compiled


def list_available_assets():
    """List all available assets that can be used in tilesets"""
    try:
        from madrona_escape_room.ctypes_bindings import (
            get_physics_assets_list,
            get_render_assets_list,
        )

        print("Available physics assets:")
        for asset in get_physics_assets_list():
            print(f"  - {asset}")

        print("\nAvailable render-only assets:")
        for asset in get_render_assets_list():
            print(f"  - {asset}")
    except ImportError:
        print("Asset listing requires compiled C API library")


if __name__ == "__main__":
    print("=== Tileset Level Examples ===\n")

    # List available assets
    print("Available assets for use in tilesets:")
    list_available_assets()
    print()

    # Run examples
    print("1. Simple tileset example:")
    example_simple_tileset()
    print()

    print("2. Advanced tileset with cylinders:")
    example_advanced_tileset()
    print()

    print("3. Automatic tileset for special characters:")
    example_auto_tileset()
    print()

    print("4. Creative character mapping:")
    example_custom_mapping()
    print()

    print("✅ All tileset examples completed successfully!")
