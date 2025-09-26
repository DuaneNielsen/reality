#!/usr/bin/env python3
"""
Four-Room Progressive Level Generator for Madrona Escape Room

Generates a progression of 32 levels with 4 rooms separated by gates of varying sizes.
The progression focuses on gate size difficulty - early levels have large gates,
later levels have smaller gates requiring more precise navigation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def create_gate(gate_size, wall_length, random_seed=None):
    """
    Create a gate of specified size at a random position in a wall.

    Args:
        gate_size: Size of the gate (1-4 tiles)
        wall_length: Length of the wall (excluding end walls)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (start_pos, end_pos) for the gate
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure gate fits within wall (leave at least 1 wall tile on each side)
    max_start = wall_length - gate_size - 1
    if max_start < 1:
        max_start = 1

    gate_start = np.random.randint(1, max_start + 1)
    gate_end = gate_start + gate_size - 1

    return gate_start, gate_end


def create_four_room_grid(level_num, random_seed=None):
    """
    Create a 20x20 grid with 4 rooms separated by walls with gates.

    Args:
        level_num: Level number (1-32) for determining gate sizes
        random_seed: Random seed for reproducibility

    Returns:
        List of strings representing the level grid
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize 20x20 grid with empty spaces
    grid = [["." for _ in range(20)] for _ in range(20)]

    # Add boundary walls
    for i in range(20):
        grid[i][0] = "#"  # Left wall
        grid[i][19] = "#"  # Right wall
    for j in range(20):
        grid[0][j] = "#"  # Top wall
        grid[19][j] = "#"  # Bottom wall

    # Calculate gate sizes based on level progression
    # Levels 1-8: Large gates (3-4)
    # Levels 9-16: Medium gates (2-3)
    # Levels 17-24: Small gates (2-2)
    # Levels 25-32: Minimum gates (2)
    if level_num <= 8:
        min_gate_size, max_gate_size = 3, 4
    elif level_num <= 16:
        min_gate_size, max_gate_size = 2, 3
    elif level_num <= 24:
        min_gate_size, max_gate_size = 2, 2
    else:
        min_gate_size, max_gate_size = 2, 2

    # Create vertical dividing wall (splitting into left/right halves)
    for i in range(1, 19):
        grid[i][10] = "#"

    # Create horizontal dividing wall (splitting into top/bottom halves)
    for j in range(1, 19):
        grid[10][j] = "#"

    # Create 4 gates - one in each wall segment between rooms
    # Use different seeds for each gate to ensure variety
    base_seed = random_seed if random_seed is not None else 0

    # Gate 1: Top vertical wall section (connects top-left and top-right)
    # This is in the vertical wall at column 10, rows 1-9
    np.random.seed(base_seed + 1)
    gate1_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate1_start, gate1_end = create_gate(
        gate1_size, 8, base_seed + 1
    )  # 8 available positions (rows 1-8)
    for i in range(gate1_start, gate1_end + 1):
        grid[i][10] = "."

    # Gate 2: Bottom vertical wall section (connects bottom-left and bottom-right)
    # This is in the vertical wall at column 10, rows 11-18
    np.random.seed(base_seed + 2)
    gate2_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate2_start, gate2_end = create_gate(
        gate2_size, 8, base_seed + 2
    )  # 8 available positions (rows 11-18 -> 0-7 offset)
    for i in range(gate2_start + 11, gate2_end + 11 + 1):  # Offset to rows 11-18
        grid[i][10] = "."

    # Gate 3: Left horizontal wall section (connects top-left and bottom-left)
    # This is in the horizontal wall at row 10, columns 1-9
    np.random.seed(base_seed + 3)
    gate3_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate3_start, gate3_end = create_gate(
        gate3_size, 8, base_seed + 3
    )  # 8 available positions (cols 1-8)
    for j in range(gate3_start, gate3_end + 1):
        grid[10][j] = "."

    # Gate 4: Right horizontal wall section (connects top-right and bottom-right)
    # This is in the horizontal wall at row 10, columns 11-18
    np.random.seed(base_seed + 4)
    gate4_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate4_start, gate4_end = create_gate(
        gate4_size, 8, base_seed + 4
    )  # 8 available positions (cols 11-18 -> 0-7 offset)
    for j in range(gate4_start + 11, gate4_end + 11 + 1):  # Offset to cols 11-18
        grid[10][j] = "."

    # Reset random state for spawn/target placement
    if random_seed is not None:
        np.random.seed(random_seed + 10)

    # Define room boundaries (excluding walls)
    rooms = [
        (1, 9, 1, 9),  # Top-left room
        (1, 9, 11, 18),  # Top-right room
        (11, 18, 1, 9),  # Bottom-left room
        (11, 18, 11, 18),  # Bottom-right room
    ]

    # Choose random rooms for spawn and target (must be different)
    spawn_room = np.random.randint(0, 4)
    target_room = np.random.randint(0, 4)
    while target_room == spawn_room:
        target_room = np.random.randint(0, 4)

    # Place spawn in random position within chosen room
    spawn_room_bounds = rooms[spawn_room]
    spawn_y = np.random.randint(spawn_room_bounds[0], spawn_room_bounds[1] + 1)
    spawn_x = np.random.randint(spawn_room_bounds[2], spawn_room_bounds[3] + 1)
    grid[spawn_y][spawn_x] = "S"

    # Don't place 'T' in the ASCII grid - the target will be handled by the targets array configuration

    # Convert to list of strings
    return ["".join(row) for row in grid]


def create_level_json(level_num, grid, spawn_random=True):
    """
    Create the complete level JSON structure.

    Args:
        level_num: Level number
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
        "name": f"four_room_20x20_lvl{level_num}",
    }

    return level_data


def generate_multi_level_json(
    output_dir="levels", random_seed=42, spawn_random=True, num_levels=32
):
    """
    Generate a single multi-level JSON file containing all 32 progressive levels.

    Args:
        output_dir: Output directory for the multi-level file
        random_seed: Base random seed for reproducibility
        spawn_random: Whether to use random spawn positions
        num_levels: Number of levels to generate (default: 32)

    Returns:
        Path to generated multi-level file
    """
    print(f"Generating four-room multi-level JSON file in {output_dir}/")

    level_name = f"four_room_progression_{num_levels}_levels"
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
    for level_num in range(1, num_levels + 1):
        # Use different seed for each level
        level_seed = random_seed + level_num if random_seed else None

        # Create level grid
        grid = create_four_room_grid(level_num, level_seed)

        # Add level to multi-level data
        level_entry = {
            "ascii": grid,
            "name": f"four_room_20x20_lvl{level_num}",
        }
        levels_data.append(level_entry)

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
    print(
        "Gate progression: Levels 1-8 (gates 3-4), 9-16 (gates 2-3), 17-24 (gates 2), 25-32 (gates 2)"
    )
    return output_path


def generate_single_level(level_num, output_dir="levels", random_seed=None, spawn_random=True):
    """
    Generate a single level file for testing.

    Args:
        level_num: Level number (1-32)
        output_dir: Output directory for level files
        random_seed: Random seed for reproducibility
        spawn_random: Whether to use random spawn positions

    Returns:
        Path to generated level file
    """
    grid = create_four_room_grid(level_num, random_seed)
    level_data = create_level_json(level_num, grid, spawn_random)

    # Save to file
    output_path = Path(output_dir) / f"four_room_20x20_lvl{level_num}.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(level_data, f, indent=4)

    print(f"Generated {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate four-room progressive levels with varying gate sizes"
    )
    parser.add_argument(
        "--level", type=int, choices=range(1, 33), help="Generate single level (1-32)"
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
        "--num-levels",
        type=int,
        default=32,
        help="Number of levels to generate for multi-level file (default: 32)",
    )

    args = parser.parse_args()

    if args.single:
        # Generate single multi-level JSON file
        if args.level:
            print("Error: --single and --level cannot be used together")
            sys.exit(1)
        generate_multi_level_json(args.output_dir, args.seed, args.spawn_random, args.num_levels)
    elif args.level:
        # Generate single level
        generate_single_level(args.level, args.output_dir, args.seed, args.spawn_random)
    else:
        # Generate multi-level file by default
        generate_multi_level_json(args.output_dir, args.seed, args.spawn_random, args.num_levels)


if __name__ == "__main__":
    main()
