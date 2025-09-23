#!/usr/bin/env python3
"""
Progressive Size Level Generator for Madrona Escape Room

Generates levels with increasing size and object count:
- 12x12 with 0 objects
- 14x14 with 1 object
- 16x16 with 4 objects
- 18x18 with 9 objects
- 20x20 with 16 objects
- ...up to 30x30

Objects are placed in the center with randomization set to level max size.
"""

import argparse
import json
import math
from pathlib import Path


def create_level_grid(width, height, num_objects):
    """
    Create the ASCII grid for the level with objects placed in center.

    Args:
        width: Level width
        height: Level height
        num_objects: Number of objects to place in center

    Returns:
        List of strings representing the level grid
    """
    # Initialize grid with empty spaces
    grid = [["." for _ in range(width)] for _ in range(height)]

    # Add spawn point at bottom center
    spawn_x = width // 2
    spawn_y = height - 2  # Second from bottom
    grid[spawn_y][spawn_x] = "S"

    # Place objects in center if any
    if num_objects > 0:
        center_x = width // 2
        center_y = height // 2

        # Calculate grid size for objects (square root)
        grid_size = int(math.ceil(math.sqrt(num_objects)))

        # Calculate starting position for centered grid
        start_x = center_x - grid_size // 2
        start_y = center_y - grid_size // 2

        # Place objects in grid pattern
        obj_count = 0
        for dy in range(grid_size):
            for dx in range(grid_size):
                if obj_count >= num_objects:
                    break

                x = start_x + dx
                y = start_y + dy

                # Make sure we don't overwrite spawn and stay in bounds
                if (
                    0 <= x < width
                    and 0 <= y < height
                    and grid[y][x] == "."
                    and not (x == spawn_x and y == spawn_y)
                ):
                    grid[y][x] = "C"
                    obj_count += 1

            if obj_count >= num_objects:
                break

    # Convert to list of strings
    return ["".join(row) for row in grid]


def create_level_json(level_num, width, height, num_objects):
    """
    Create the complete level JSON structure.

    Args:
        level_num: Level number
        width: Level width
        height: Level height
        num_objects: Number of objects

    Returns:
        Dictionary representing the level JSON
    """
    grid = create_level_grid(width, height, num_objects)

    # Create level-specific tileset with appropriate randomization
    max_dimension = max(width, height)
    rand_value = float(max_dimension * 2)

    tileset = {
        "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0},
        ".": {"asset": "empty"},
    }

    # Add cube tile with level-specific randomization if objects exist
    if num_objects > 0:
        tileset["C"] = {
            "asset": "cube",
            "done_on_collision": True,
            "rand_x": rand_value,
            "rand_y": rand_value,
            "rand_z": 0.3,
            "rand_rot_z": 6.28318,  # Full rotation randomness
            "rand_scale": 0.4,  # Size randomness (±40%)
        }

    level_data = {
        "ascii": grid,
        "name": f"lvl{level_num}_{width}x{height}_{num_objects}obj",
        "tileset": tileset,
        "targets": [
            {
                "position": [0.0, 0.0, 1.0],
                "motion_type": "static",
                "params": {
                    "omega_x": 0.0,
                    "omega_y": 1.0,
                    "center": [0.0, 0.0, 1.0],
                    "mass": 1.0,
                    "phase_x": 0.0,
                    "phase_y": 0.0,
                },
            }
        ],
    }

    return level_data


def generate_progressive_levels(output_dir="levels", random_seed=42):
    """
    Generate all progressive size levels with increasing object counts.

    Args:
        output_dir: Output directory for level files
        random_seed: Base random seed for reproducibility
    """
    print(f"Generating progressive size levels in {output_dir}/")

    # Define level progression: (size, num_objects)
    levels = [
        (12, 0),  # 12x12 with 0 objects
        (14, 1),  # 14x14 with 1 object
        (16, 4),  # 16x16 with 4 objects (2x2)
        (18, 9),  # 18x18 with 9 objects (3x3)
        (20, 16),  # 20x20 with 16 objects (4x4)
        (22, 25),  # 22x22 with 25 objects (5x5)
        (24, 36),  # 24x24 with 36 objects (6x6)
        (26, 49),  # 26x26 with 49 objects (7x7)
        (28, 64),  # 28x28 with 64 objects (8x8)
        (30, 81),  # 30x30 with 81 objects (9x9)
    ]

    levels_data = []

    for level_num, (size, num_objects) in enumerate(levels, 1):
        print(f"Generating Level {level_num}: {size}x{size} with {num_objects} objects")

        # Create level data with level-specific tileset
        level_data = create_level_json(level_num, size, size, num_objects)
        levels_data.append(level_data)

    # Create multi-level JSON structure
    multi_level_data = {
        "levels": levels_data,
        "scale": 2.5,
        "spawn_random": True,
        "auto_boundary_walls": True,
        "boundary_wall_offset": 1.0,
        "name": "progressive_size_curriculum",
    }

    # Save to file
    output_path = Path(output_dir) / "progressive_size_curriculum.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(multi_level_data, f, indent=4)

    total_objects = sum(num_objects for _, num_objects in levels)
    print(
        f"Generated {output_path} with {len(levels_data)} levels and {total_objects} total objects"
    )
    print("Level progression:")
    for level_num, (size, num_objects) in enumerate(levels, 1):
        print(f"  Level {level_num}: {size}x{size} with {num_objects} objects")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate progressive size difficulty levels")
    parser.add_argument("--output-dir", default="levels", help="Output directory (default: levels)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    generate_progressive_levels(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
