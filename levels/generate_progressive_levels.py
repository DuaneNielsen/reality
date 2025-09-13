#!/usr/bin/env python3
"""
Progressive Level Generator for Madrona Escape Room

Generates levels with increasing obstacle density using weighted sampling
without replacement. Higher y-coordinates (further from spawn) are more
likely to contain obstacles.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def create_bias_weights(height=46, scale=0.5):
    """
    Create bias weights favoring lower y-coordinates (closer to spawn) using exponential decay.

    Args:
        height: Number of y-coordinates (excluding walls)
        scale: Scale parameter controlling bias strength

    Returns:
        Array of weights for each y-coordinate
    """
    # Create y-coordinates from 1 to height (46)
    y_coords = np.arange(1, height + 1)

    # Lower y gets higher weight (exponential bias toward spawn/bottom)
    # Reverse so y=1 (bottom/spawn area) has high weight, y=46 (top) has low weight
    reversed_normalized_y = (height - y_coords + 1) / height
    weights = np.exp(reversed_normalized_y / scale)

    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)

    return weights


def create_cell_pool(width=17, height=46):
    """
    Create pool of all valid cell positions with their weights.
    Excludes the spawn position at the bottom center.

    Args:
        width: Usable width (excluding walls)
        height: Usable height (excluding walls and spawn area)

    Returns:
        List of (x, y, weight) tuples for all valid cells
    """
    y_weights = create_bias_weights(height)
    cell_pool = []

    # Spawn position calculation (same as in create_level_grid):
    # Total grid is 19x48, spawn at center: width//2 = 9, height-2 = 46
    total_width = 19
    total_height = 48
    spawn_x = total_width // 2
    spawn_y = total_height - 2

    for y in range(1, height + 1):  # y from 1 to 46
        y_weight = y_weights[y - 1]
        for x in range(1, width + 1):  # x from 1 to 17
            # Skip spawn position
            if x == spawn_x and y == spawn_y:
                continue
            # Skip first 3 rows from bottom (clear spawn area)
            if y >= height - 2:  # y >= 44 (rows 44, 45, 46 are clear)
                continue
            cell_pool.append((x, y, y_weight))

    return cell_pool


def sample_obstacle_positions(cell_pool, num_obstacles, random_seed=None):
    """
    Sample obstacle positions using weighted sampling without replacement.
    Places obstacles at segment x-centers with randomness for distribution.

    Args:
        cell_pool: List of (x, y, weight) tuples
        num_obstacles: Number of obstacles to place
        random_seed: Random seed for reproducibility

    Returns:
        List of (x, y) positions for obstacles (x at segment center)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Extract y-coordinates and weights (ignore x since we'll place at center)
    y_positions = []
    y_weights = []

    # Group by y-coordinate and get unique y values with their weights
    y_dict = {}
    for x, y, w in cell_pool:
        if y not in y_dict:
            y_dict[y] = w

    y_positions = list(y_dict.keys())
    y_weights = np.array(list(y_dict.values()))

    # Normalize weights for sampling
    y_weights = y_weights / np.sum(y_weights)

    # Sample y-coordinates with replacement (allows multiple obstacles per segment)
    y_indices = np.random.choice(len(y_positions), size=num_obstacles, replace=True, p=y_weights)

    # Place all obstacles at x-center of segments (x=9 for 19-wide grid)
    # The +2.5 offset will be handled by increased rand_x values during simulation
    segment_x_center = 9

    return [(segment_x_center, y_positions[i]) for i in y_indices]


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


def create_level_grid(obstacles, width=19, height=48):
    """
    Create the ASCII grid for the level.

    Args:
        obstacles: List of (x, y, type) tuples where x, y are center coordinates
        width: Total grid width including walls
        height: Total grid height including walls

    Returns:
        List of strings representing the level grid
    """
    # Initialize grid with empty spaces
    grid = [["." for _ in range(width)] for _ in range(height)]

    # Add walls around perimeter
    for i in range(height):
        grid[i][0] = "#"  # Left wall
        grid[i][width - 1] = "#"  # Right wall
    for j in range(width):
        grid[0][j] = "#"  # Top wall
        grid[height - 1][j] = "#"  # Bottom wall

    # Add spawn point at bottom center
    spawn_x = width // 2
    spawn_y = height - 2  # Second from bottom
    grid[spawn_y][spawn_x] = "S"

    # Place obstacles at grid centers (obstacles are already positioned at centers)
    for x, y, obstacle_type in obstacles:
        grid[y][x] = obstacle_type

    # Convert to list of strings
    return ["".join(row) for row in grid]


def create_level_json(level_num, obstacles, name_suffix="gen"):
    """
    Create the complete level JSON structure.

    Args:
        level_num: Level number
        obstacles: List of (x, y, type) tuples
        name_suffix: Suffix for level name

    Returns:
        Dictionary representing the level JSON
    """
    grid = create_level_grid(obstacles)

    level_data = {
        "ascii": grid,
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": True},
            "C": {
                "asset": "cube",
                "done_on_collision": True,
                "rand_x": 50.0,  # Very large x randomness for segment-wide distribution
                "rand_y": 1.2,  # Moderate y randomness
                "rand_z": 0.3,  # Vertical position variation
                "rand_rot_z": 6.28318,  # Full rotation randomness (2*pi)
                "rand_scale": 0.4,  # Size randomness (±40%)
            },
            "O": {
                "asset": "cylinder",
                "done_on_collision": True,
                "rand_x": 50.0,  # Very large x randomness for segment-wide distribution
                "rand_y": 1.0,  # Moderate y randomness
                "rand_z": 0.2,  # Vertical position variation
                "rand_rot_z": 6.28318,  # Full rotation randomness
                "rand_scale": 0.3,  # Size randomness (±30%)
            },
            "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0, "rand_rot_z": 0.0},
            ".": {"asset": "empty"},
        },
        "scale": 3.0,
        "agent_facing": [0],
        "name": f"lvl{level_num}_19x48_{name_suffix}",
    }

    return level_data


def generate_level(level_num, output_dir="levels", random_seed=None):
    """
    Generate a single level with progressive difficulty.

    Args:
        level_num: Level number (1-7)
        output_dir: Output directory for level files
        random_seed: Random seed for reproducibility

    Returns:
        Path to generated level file
    """
    num_obstacles = level_num * 10

    # Create cell pool and sample positions
    cell_pool = create_cell_pool()
    positions = sample_obstacle_positions(cell_pool, num_obstacles, random_seed)
    obstacles = assign_obstacle_types(positions)

    # Create level JSON
    level_data = create_level_json(level_num, obstacles)

    # Save to file
    output_path = Path(output_dir) / f"lvl{level_num}_19x48_gen.json"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(level_data, f, indent=4)

    print(f"Generated {output_path} with {len(obstacles)} obstacles")
    return output_path


def generate_all_levels(output_dir="levels", random_seed=42):
    """
    Generate all progressive levels (1-20).

    Args:
        output_dir: Output directory for level files
        random_seed: Base random seed for reproducibility
    """
    print(f"Generating progressive levels in {output_dir}/")

    for level_num in range(1, 21):
        # Use different seed for each level
        level_seed = random_seed + level_num if random_seed else None
        generate_level(level_num, output_dir, level_seed)

    print("All levels generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate progressive difficulty levels")
    parser.add_argument(
        "--level", type=int, choices=range(1, 21), help="Generate single level (1-20)"
    )
    parser.add_argument("--output-dir", default="levels", help="Output directory (default: levels)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    if args.level:
        generate_level(args.level, args.output_dir, args.seed)
    else:
        generate_all_levels(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
