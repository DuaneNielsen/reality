#!/usr/bin/env python3
"""
Empty Room Progressive Level Generator for Madrona Escape Room

Generates a progression of empty rooms with increasing sizes from 12x12 to 24x24.
These levels are useful for:
- Testing navigation in open spaces
- Benchmarking performance across different grid sizes
- Baseline difficulty assessment without obstacles
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def create_empty_room_grid(size, random_seed=None):
    """
    Create a size×size grid with boundary walls and empty interior.

    Args:
        size: Grid size (12-24)
        random_seed: Random seed for reproducible spawn placement

    Returns:
        List of strings representing the level grid
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize grid with empty spaces
    grid = [["." for _ in range(size)] for _ in range(size)]

    # Add boundary walls
    for i in range(size):
        grid[i][0] = "#"  # Left wall
        grid[i][size - 1] = "#"  # Right wall
    for j in range(size):
        grid[0][j] = "#"  # Top wall
        grid[size - 1][j] = "#"  # Bottom wall

    # Place spawn point randomly in interior (leaving 1-tile border)
    spawn_x = np.random.randint(2, size - 2)
    spawn_y = np.random.randint(2, size - 2)
    grid[spawn_y][spawn_x] = "S"

    # Convert to list of strings
    return ["".join(row) for row in grid]


def create_level_json(level_num, size, grid, spawn_random=True):
    """
    Create the complete level JSON structure.

    Args:
        level_num: Level number (1-13 for sizes 12-24)
        size: Grid size
        grid: ASCII grid as list of strings
        spawn_random: Whether to use random spawn positions

    Returns:
        Dictionary representing the level JSON
    """
    level_data = {
        "ascii": grid,
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": True},
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
        },
        "scale": 2.5,
        "auto_boundary_walls": False,  # We create our own walls
        "spawn_random": spawn_random,
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
        "name": f"empty_room_{size}x{size}_lvl{level_num}",
    }

    return level_data


def generate_multi_level_json(
    output_dir="levels",
    random_seed=42,
    spawn_random=True,
    min_size=12,
    max_size=24,
):
    """
    Generate a single multi-level JSON file containing all empty room progressions.

    Args:
        output_dir: Output directory for the multi-level file
        random_seed: Base random seed for reproducibility
        spawn_random: Whether to use random spawn positions
        min_size: Minimum room size (default: 12)
        max_size: Maximum room size (default: 24)

    Returns:
        Path to generated multi-level file
    """
    print(f"Generating empty room multi-level JSON file in {output_dir}/")

    level_name = f"empty_room_progression_{min_size}x{min_size}_to_{max_size}x{max_size}"
    if spawn_random:
        level_name += "_spawn_random"

    # Shared tileset for all levels
    shared_tileset = {
        "#": {"asset": "wall", "done_on_collision": True},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"},
    }

    levels_data = []

    # Generate each level's ASCII and metadata
    level_num = 1
    for size in range(min_size, max_size + 1):
        # Use different seed for each level
        level_seed = random_seed + level_num if random_seed else None

        # Create level grid
        grid = create_empty_room_grid(size, level_seed)

        # Add level to multi-level data
        level_entry = {
            "ascii": grid,
            "name": f"empty_room_{size}x{size}_lvl{level_num}",
        }
        levels_data.append(level_entry)
        level_num += 1

    # Create multi-level JSON structure
    multi_level_data = {
        "levels": levels_data,
        "tileset": shared_tileset,
        "scale": 2.5,
        "auto_boundary_walls": False,
        "spawn_random": spawn_random,
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
        "name": level_name,
    }

    # Save to file
    output_path = Path(output_dir) / f"{level_name}.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(multi_level_data, f, indent=4)

    print(f"Generated {output_path} with {len(levels_data)} levels")
    print(f"Room sizes: {min_size}×{min_size} to {max_size}×{max_size}")
    return output_path


def generate_single_level(level_size, output_dir="levels", random_seed=None, spawn_random=True):
    """
    Generate a single level file for testing.

    Args:
        level_size: Room size (12-24)
        output_dir: Output directory for level files
        random_seed: Random seed for reproducibility
        spawn_random: Whether to use random spawn positions

    Returns:
        Path to generated level file
    """
    grid = create_empty_room_grid(level_size, random_seed)
    level_data = create_level_json(1, level_size, grid, spawn_random)

    # Save to file
    output_path = Path(output_dir) / f"empty_room_{level_size}x{level_size}.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(level_data, f, indent=4)

    print(f"Generated {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate empty room progressive levels with increasing sizes"
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=range(12, 25),
        help="Generate single level with specific size (12-24)",
    )
    parser.add_argument("--output-dir", default="levels", help="Output directory (default: levels)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Generate single multi-level JSON file containing all levels",
    )
    parser.add_argument(
        "--spawn-random",
        action="store_true",
        default=True,
        help="Use random spawn positions (default: True)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=12,
        help="Minimum room size for multi-level file (default: 12)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=24,
        help="Maximum room size for multi-level file (default: 24)",
    )

    args = parser.parse_args()

    if args.size:
        # Generate single level with specific size
        if args.single:
            print("Error: --single and --size cannot be used together")
            sys.exit(1)
        generate_single_level(args.size, args.output_dir, args.seed, args.spawn_random)
    elif args.single or not args.size:
        # Generate multi-level file by default
        generate_multi_level_json(
            args.output_dir, args.seed, args.spawn_random, args.min_size, args.max_size
        )


if __name__ == "__main__":
    main()
