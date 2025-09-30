#!/usr/bin/env python3
"""
Combined Progressive Level Generator for Madrona Escape Room

Generates levels combining:
- Progressive obstacles (weighted sampling without replacement)
- Four-room layouts with gates
- Various room sizes from 12x12 to 24x24
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def create_bias_weights(height, scale=0.5):
    """
    Create bias weights favoring lower y-coordinates using exponential decay.

    Args:
        height: Number of y-coordinates (excluding walls)
        scale: Scale parameter controlling bias strength

    Returns:
        Array of weights for each y-coordinate
    """
    y_coords = np.arange(1, height + 1)
    reversed_normalized_y = (height - y_coords + 1) / height
    weights = np.exp(reversed_normalized_y / scale)
    weights = weights / np.sum(weights)
    return weights


def create_cell_pool(width, height, exclude_gates=None):
    """
    Create pool of all valid cell positions with their weights.

    Args:
        width: Usable width (excluding walls)
        height: Usable height (excluding walls)
        exclude_gates: List of (x, y) positions to exclude (for gates)

    Returns:
        List of (x, y, weight) tuples for all valid cells
    """
    y_weights = create_bias_weights(height)
    cell_pool = []
    exclude_set = set(exclude_gates) if exclude_gates else set()

    spawn_x = (width + 2) // 2  # Center position
    spawn_y = height  # Bottom row (before walls)

    for y in range(1, height + 1):
        y_weight = y_weights[y - 1]
        for x in range(1, width + 1):
            # Skip spawn position
            if x == spawn_x and y == spawn_y:
                continue
            # Skip excluded positions (gates)
            if (x, y) in exclude_set:
                continue
            # Skip first 3 rows from bottom (clear spawn area)
            if y >= height - 2:
                continue
            cell_pool.append((x, y, y_weight))

    return cell_pool


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

    max_start = wall_length - gate_size - 1
    if max_start < 1:
        max_start = 1

    gate_start = np.random.randint(1, max_start + 1)
    gate_end = gate_start + gate_size - 1

    return gate_start, gate_end


def sample_obstacle_positions(cell_pool, num_obstacles, random_seed=None):
    """
    Sample obstacle positions using weighted sampling without replacement.

    Args:
        cell_pool: List of (x, y, weight) tuples
        num_obstacles: Number of obstacles to place
        random_seed: Random seed for reproducibility

    Returns:
        List of (x, y) positions for obstacles
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not cell_pool or num_obstacles == 0:
        return []

    # Extract y-coordinates and weights
    y_dict = {}
    for x, y, w in cell_pool:
        if y not in y_dict:
            y_dict[y] = w

    y_positions = list(y_dict.keys())
    y_weights = np.array(list(y_dict.values()))
    y_weights = y_weights / np.sum(y_weights)

    # Sample y-coordinates with replacement
    y_indices = np.random.choice(len(y_positions), size=num_obstacles, replace=True, p=y_weights)

    # Get x-coordinates from cell pool for sampled y values
    positions = []
    for i in y_indices:
        y = y_positions[i]
        available_x = [x for x, cy, w in cell_pool if cy == y]
        if available_x:
            x = np.random.choice(available_x)
            positions.append((x, y))

    return positions


def assign_obstacle_types(positions, cube_ratio=0.6):
    """
    Randomly assign obstacle types to positions.

    Args:
        positions: List of (x, y) positions
        cube_ratio: Proportion of obstacles that should be cubes

    Returns:
        List of (x, y, type) tuples where type is 'C' or 'O'
    """
    np.random.shuffle(positions)
    num_cubes = int(len(positions) * cube_ratio)

    obstacles = []
    for i, (x, y) in enumerate(positions):
        obstacle_type = "C" if i < num_cubes else "O"
        obstacles.append((x, y, obstacle_type))

    return obstacles


def create_combined_grid(size, level_num, random_seed=None):
    """
    Create a size×size grid with four rooms, gates, and obstacles.

    Args:
        size: Grid size (12-24)
        level_num: Level number (1-32) for determining obstacle density and gate sizes
        random_seed: Random seed for reproducibility

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

    # Create dividing walls (4 rooms)
    mid = size // 2
    for i in range(1, size - 1):
        grid[i][mid] = "#"  # Vertical divider
    for j in range(1, size - 1):
        grid[mid][j] = "#"  # Horizontal divider

    # Calculate gate sizes based on level progression (like four_room_progression)
    if level_num <= 8:
        min_gate_size, max_gate_size = 3, 4
    elif level_num <= 16:
        min_gate_size, max_gate_size = 2, 3
    elif level_num <= 24:
        min_gate_size, max_gate_size = 2, 2
    else:
        min_gate_size, max_gate_size = 2, 2

    base_seed = random_seed if random_seed is not None else 0
    gate_positions = []

    # Gate 1: Top vertical wall section
    np.random.seed(base_seed + 1)
    gate1_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate1_start, gate1_end = create_gate(gate1_size, mid - 2, base_seed + 1)
    for i in range(gate1_start, gate1_end + 1):
        grid[i][mid] = "."
        gate_positions.append((mid, i))

    # Gate 2: Bottom vertical wall section
    np.random.seed(base_seed + 2)
    gate2_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate2_start, gate2_end = create_gate(gate2_size, mid - 2, base_seed + 2)
    for i in range(gate2_start + mid + 1, gate2_end + mid + 2):
        grid[i][mid] = "."
        gate_positions.append((mid, i))

    # Gate 3: Left horizontal wall section
    np.random.seed(base_seed + 3)
    gate3_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate3_start, gate3_end = create_gate(gate3_size, mid - 2, base_seed + 3)
    for j in range(gate3_start, gate3_end + 1):
        grid[mid][j] = "."
        gate_positions.append((j, mid))

    # Gate 4: Right horizontal wall section
    np.random.seed(base_seed + 4)
    gate4_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate4_start, gate4_end = create_gate(gate4_size, mid - 2, base_seed + 4)
    for j in range(gate4_start + mid + 1, gate4_end + mid + 2):
        grid[mid][j] = "."
        gate_positions.append((j, mid))

    # Place obstacles (progressive density - like progressive_levels.py: 10, 20, 30... obstacles)
    num_obstacles = level_num * 10
    cell_pool = create_cell_pool(size - 2, size - 2, gate_positions)

    np.random.seed(random_seed + 100 if random_seed else None)
    positions = sample_obstacle_positions(
        cell_pool, num_obstacles, random_seed + 100 if random_seed else None
    )
    obstacles = assign_obstacle_types(positions)

    for x, y, obstacle_type in obstacles:
        grid[y][x] = obstacle_type

    # Place spawn in random room
    rooms = [
        (1, mid - 1, 1, mid - 1),  # Top-left
        (1, mid - 1, mid + 1, size - 2),  # Top-right
        (mid + 1, size - 2, 1, mid - 1),  # Bottom-left
        (mid + 1, size - 2, mid + 1, size - 2),  # Bottom-right
    ]

    np.random.seed(random_seed + 10 if random_seed else None)
    spawn_room = rooms[np.random.randint(0, 4)]
    spawn_y = np.random.randint(spawn_room[0], spawn_room[1] + 1)
    spawn_x = np.random.randint(spawn_room[2], spawn_room[3] + 1)
    grid[spawn_y][spawn_x] = "S"

    return ["".join(row) for row in grid]


def generate_multi_level_json(
    output_dir="levels",
    random_seed=42,
    spawn_random=True,
    num_levels=32,
):
    """
    Generate a single multi-level JSON file with combined features.
    Creates 32 levels (like four_room_progression) with progressive difficulty.

    Args:
        output_dir: Output directory for the multi-level file
        random_seed: Base random seed for reproducibility
        spawn_random: Whether to use random spawn positions
        num_levels: Number of levels to generate (default: 32)

    Returns:
        Path to generated multi-level file
    """
    print(f"Generating combined progression levels in {output_dir}/")

    level_name = f"combined_progression_{num_levels}_levels"
    if spawn_random:
        level_name += "_spawn_random"

    # Shared tileset
    shared_tileset = {
        "#": {"asset": "wall", "done_on_collision": True},
        "C": {
            "asset": "cube",
            "done_on_collision": True,
            "rand_x": 50.0,
            "rand_y": 1.2,
            "rand_z": 0.3,
            "rand_rot_z": 6.28318,
            "rand_scale": 0.4,
        },
        "O": {
            "asset": "cylinder",
            "done_on_collision": True,
            "rand_x": 50.0,
            "rand_y": 1.0,
            "rand_z": 0.2,
            "rand_rot_z": 6.28318,
            "rand_scale": 0.3,
        },
        "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0, "rand_rot_z": 0.0},
        ".": {"asset": "empty"},
    }

    levels_data = []

    # Generate each level with progressive size and difficulty
    for level_num in range(1, num_levels + 1):
        level_seed = random_seed + level_num if random_seed else None

        # Progressive room sizes: cycle through 12x12 to 24x24
        # Maps levels 1-32 across the size range multiple times
        size = 12 + ((level_num - 1) % 13)

        # Create combined grid with obstacles, gates, and room size
        grid = create_combined_grid(size, level_num, level_seed)

        level_entry = {
            "ascii": grid,
            "name": f"combined_{size}x{size}_lvl{level_num}",
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

    total_obstacles = sum(level_num * 10 for level_num in range(1, num_levels + 1))
    print(f"Generated {output_path} with {len(levels_data)} levels")
    print("Room sizes: 12×12 to 24×24 (cycling)")
    print(f"Obstacle progression: 10, 20, 30... {num_levels * 10}")
    print(f"Total obstacles: {total_obstacles}")
    print("Gate progression: Levels 1-8 (gates 3-4), 9-16 (gates 2-3), 17+ (gates 2)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined progressive levels with obstacles, gates, and room sizes"
    )
    parser.add_argument("--output-dir", default="levels", help="Output directory (default: levels)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
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
        help="Number of levels to generate (default: 32)",
    )

    args = parser.parse_args()

    generate_multi_level_json(args.output_dir, args.seed, args.spawn_random, args.num_levels)


if __name__ == "__main__":
    main()
