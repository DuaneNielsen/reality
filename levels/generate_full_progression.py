#!/usr/bin/env python3
"""
Full Progressive Level Generator for Madrona Escape Room

Combines three separate progressions into one multi-level file:
1. Progressive size levels (50 levels with size+density progression)
2. Four-room with gates (96 levels = 32 levels × 3 obstacle variants: none/light/medium)
3. Empty rooms (13 levels from 12x12 to 24x24)

Total: 159 levels
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

# ========== Progressive Size Functions ==========


def create_progressive_size_level_grid(width, height, num_objects):
    """
    Create the ASCII grid for progressive size levels with objects placed in center.

    Args:
        width: Level width
        height: Level height
        num_objects: Number of objects to place in center

    Returns:
        List of strings representing the level grid
    """
    grid = [["." for _ in range(width)] for _ in range(height)]

    # Add spawn point at bottom center
    spawn_x = width // 2
    spawn_y = height - 2
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

    return ["".join(row) for row in grid]


# ========== Four-Room Functions ==========


def create_gate(gate_size, wall_length, random_seed=None):
    """Create a gate of specified size at a random position in a wall."""
    if random_seed is not None:
        np.random.seed(random_seed)

    max_start = wall_length - gate_size - 1
    if max_start < 1:
        max_start = 1

    gate_start = np.random.randint(1, max_start + 1)
    gate_end = gate_start + gate_size - 1

    return gate_start, gate_end


def create_four_room_grid(level_num, random_seed=None, num_obstacles=0):
    """
    Create a 20x20 grid with 4 rooms separated by walls with gates.

    Args:
        level_num: Level number for gate size progression
        random_seed: Random seed for reproducibility
        num_obstacles: Number of obstacles to place (0 for none, 5-10 for light, 15-25 for medium)

    Returns:
        List of strings representing the level grid
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    size = 20
    grid = [["." for _ in range(size)] for _ in range(size)]

    for i in range(size):
        grid[i][0] = "#"
        grid[i][size - 1] = "#"
    for j in range(size):
        grid[0][j] = "#"
        grid[size - 1][j] = "#"

    for i in range(1, size - 1):
        grid[i][10] = "#"
    for j in range(1, size - 1):
        grid[10][j] = "#"

    if level_num <= 8:
        min_gate_size, max_gate_size = 3, 4
    elif level_num <= 16:
        min_gate_size, max_gate_size = 2, 3
    elif level_num <= 24:
        min_gate_size, max_gate_size = 2, 2
    else:
        min_gate_size, max_gate_size = 2, 2

    base_seed = random_seed if random_seed is not None else 0

    np.random.seed(base_seed + 1)
    gate1_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate1_start, gate1_end = create_gate(gate1_size, 8, base_seed + 1)
    for i in range(gate1_start, gate1_end + 1):
        grid[i][10] = "."

    np.random.seed(base_seed + 2)
    gate2_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate2_start, gate2_end = create_gate(gate2_size, 8, base_seed + 2)
    for i in range(gate2_start + 11, gate2_end + 11 + 1):
        grid[i][10] = "."

    np.random.seed(base_seed + 3)
    gate3_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate3_start, gate3_end = create_gate(gate3_size, 8, base_seed + 3)
    for j in range(gate3_start, gate3_end + 1):
        grid[10][j] = "."

    np.random.seed(base_seed + 4)
    gate4_size = np.random.randint(min_gate_size, max_gate_size + 1)
    gate4_start, gate4_end = create_gate(gate4_size, 8, base_seed + 4)
    for j in range(gate4_start + 11, gate4_end + 11 + 1):
        grid[10][j] = "."

    # Define rooms for spawn and obstacles
    rooms = [
        (1, 9, 1, 9),
        (1, 9, 11, 18),
        (11, 18, 1, 9),
        (11, 18, 11, 18),
    ]

    # Place spawn
    if random_seed is not None:
        np.random.seed(random_seed + 10)

    spawn_room = np.random.randint(0, 4)
    target_room = np.random.randint(0, 4)
    while target_room == spawn_room:
        target_room = np.random.randint(0, 4)

    spawn_room_bounds = rooms[spawn_room]
    spawn_y = np.random.randint(spawn_room_bounds[0], spawn_room_bounds[1] + 1)
    spawn_x = np.random.randint(spawn_room_bounds[2], spawn_room_bounds[3] + 1)
    grid[spawn_y][spawn_x] = "S"

    # Place obstacles if requested
    if num_obstacles > 0:
        np.random.seed(random_seed + 20 if random_seed else None)
        placed_obstacles = 0
        attempts = 0
        max_attempts = num_obstacles * 10

        while placed_obstacles < num_obstacles and attempts < max_attempts:
            attempts += 1
            # Pick random room
            room_idx = np.random.randint(0, 4)
            room_bounds = rooms[room_idx]

            # Pick random position in room
            y = np.random.randint(room_bounds[0], room_bounds[1] + 1)
            x = np.random.randint(room_bounds[2], room_bounds[3] + 1)

            # Place obstacle if position is empty
            if grid[y][x] == ".":
                # 60% cubes, 40% cylinders
                grid[y][x] = "C" if np.random.random() < 0.6 else "O"
                placed_obstacles += 1

    return ["".join(row) for row in grid]


# ========== Empty Room Functions ==========


def create_empty_room_grid(size, random_seed=None):
    """Create a size×size grid with boundary walls and empty interior."""
    if random_seed is not None:
        np.random.seed(random_seed)

    grid = [["." for _ in range(size)] for _ in range(size)]

    for i in range(size):
        grid[i][0] = "#"
        grid[i][size - 1] = "#"
    for j in range(size):
        grid[0][j] = "#"
        grid[size - 1][j] = "#"

    spawn_x = np.random.randint(2, size - 2)
    spawn_y = np.random.randint(2, size - 2)
    grid[spawn_y][spawn_x] = "S"

    return ["".join(row) for row in grid]


# ========== Main Generation ==========


def generate_full_progression(
    output_dir="levels",
    random_seed=42,
    spawn_random=True,
):
    """
    Generate a single multi-level JSON file containing all three progressions.

    Args:
        output_dir: Output directory for the multi-level file
        random_seed: Base random seed for reproducibility
        spawn_random: Whether to use random spawn positions

    Returns:
        Path to generated multi-level file
    """
    print(f"Generating full progression multi-level JSON file in {output_dir}/")

    level_name = "full_progression_159_levels"
    if spawn_random:
        level_name += "_spawn_random"

    levels_data = []

    # Part 1: Progressive size levels (50 levels)
    print("Generating progressive size levels (1-50)...")
    base_sizes = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    max_objects_per_size = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    density_multipliers = [0.0, 0.25, 0.5, 0.75, 1.0]

    for size, max_objects in zip(base_sizes, max_objects_per_size):
        for density in density_multipliers:
            num_objects = math.ceil(max_objects * density) if max_objects > 0 else 0
            grid = create_progressive_size_level_grid(size, size, num_objects)

            # Create tileset with level-specific randomization
            max_dimension = size
            rand_value = float(max_dimension * 2)

            tileset = {
                "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0},
                ".": {"asset": "empty"},
            }

            if num_objects > 0:
                tileset["C"] = {
                    "asset": "cube",
                    "done_on_collision": True,
                    "rand_x": rand_value,
                    "rand_y": rand_value,
                    "rand_z": 0.3,
                    "rand_rot_z": 6.28318,
                    "rand_scale": 0.4,
                }

            level_entry = {
                "ascii": grid,
                "name": f"progressive_size_{size}x{size}_{num_objects}obj",
                "tileset": tileset,
                "auto_boundary_walls": True,
                "boundary_wall_offset": 1.0,
            }
            levels_data.append(level_entry)

    # Part 2: Four-room with gates (96 levels = 32 x 3 variants)
    print("Generating four-room gate levels (51-146)...")

    # Shared tileset for four-room levels (includes obstacles)
    four_room_tileset = {
        "#": {"asset": "wall", "done_on_collision": True},
        "C": {
            "asset": "cube",
            "done_on_collision": True,
            "rand_x": 1.5,
            "rand_y": 1.5,
            "rand_z": 0.3,
            "rand_rot_z": 6.28318,
            "rand_scale": 0.4,
        },
        "O": {
            "asset": "cylinder",
            "done_on_collision": True,
            "rand_x": 1.5,
            "rand_y": 1.5,
            "rand_z": 0.2,
            "rand_rot_z": 6.28318,
            "rand_scale": 0.3,
        },
        "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0, "rand_rot_z": 0.0},
        ".": {"asset": "empty"},
    }

    # Generate 3 variants for each of 32 levels: none, light, medium obstacles
    obstacle_variants = [
        ("none", 0),
        ("light", 4),
        ("medium", 8),
    ]

    for level_num in range(1, 33):
        for variant_name, num_obstacles in obstacle_variants:
            level_seed = (
                random_seed + 100 + level_num + (num_obstacles * 100) if random_seed else None
            )
            grid = create_four_room_grid(level_num, level_seed, num_obstacles)

            level_entry = {
                "ascii": grid,
                "name": f"four_room_20x20_lvl{level_num}_{variant_name}",
                "tileset": four_room_tileset,
                "auto_boundary_walls": False,  # Four-room levels create their own walls
            }
            levels_data.append(level_entry)

    # Part 3: Empty rooms (13 levels)
    print("Generating empty room levels (147-159)...")

    # Shared tileset for empty room levels
    empty_room_tileset = {
        "#": {"asset": "wall", "done_on_collision": True},
        "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0},
        ".": {"asset": "empty"},
    }

    for size in range(12, 25):
        level_seed = random_seed + 200 + size if random_seed else None
        grid = create_empty_room_grid(size, level_seed)

        level_entry = {
            "ascii": grid,
            "name": f"empty_room_{size}x{size}",
            "tileset": empty_room_tileset,
            "auto_boundary_walls": False,  # Empty rooms create their own walls
        }
        levels_data.append(level_entry)

    # Create multi-level JSON structure
    # Note: auto_boundary_walls is set per-level only, no global default
    multi_level_data = {
        "levels": levels_data,
        "scale": 2.5,
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

    print(f"\n✓ Generated {output_path} with {len(levels_data)} levels")
    print("  - Progressive size levels: 50 levels (12x12 to 30x30 with density progression)")
    print("  - Four-room gates: 96 levels (32 levels × 3 variants: none/light/medium obstacles)")
    print("  - Empty rooms: 13 levels (12x12 to 24x24)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate full progression with obstacles, gates, and empty rooms"
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

    args = parser.parse_args()

    generate_full_progression(args.output_dir, args.seed, args.spawn_random)


if __name__ == "__main__":
    main()
