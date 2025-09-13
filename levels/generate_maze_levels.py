#!/usr/bin/env python3
"""
Generate a series of maze levels with progressively thicker walls.
Creates balanced gate distribution on left and right sides.
"""

import json
import random
from typing import List, Tuple


def create_base_level_template() -> dict:
    """Create the base level template with tileset configuration."""
    return {
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": True},
            "C": {
                "asset": "cube",
                "done_on_collision": True,
                "rand_x": 0.8,
                "rand_y": 0.8,
                "rand_z": 0.0,
                "rand_rot_z": 3.14159,
            },
            "O": {
                "asset": "cylinder",
                "done_on_collision": True,
                "rand_x": 0.6,
                "rand_y": 0.6,
                "rand_z": 0.0,
                "rand_rot_z": 0.0,
            },
            "S": {"asset": "spawn", "rand_x": 0.5, "rand_y": 0.5, "rand_z": 0.0, "rand_rot_z": 0.0},
            ".": {"asset": "empty"},
        },
        "scale": 3.0,
        "agent_facing": [0],
    }


def generate_wall_positions(
    num_walls: int,
    width: int,
    height: int,
    spawn_row: int,
    min_distance_from_spawn: int = 5,
    min_gap_between_walls: int = 3,
) -> List[Tuple[int, int]]:
    """Generate balanced wall positions with proper spacing and spawn area avoidance."""
    positions = []

    # Avoid spawn area
    forbidden_rows = set(
        range(spawn_row - min_distance_from_spawn, spawn_row + min_distance_from_spawn + 1)
    )

    # Available rows for wall placement
    available_rows = [r for r in range(1, height - 1) if r not in forbidden_rows]

    # Sort available rows to ensure proper spacing
    available_rows.sort()

    # Track used rows to maintain minimum spacing
    used_rows = set()

    for _ in range(num_walls):
        if not available_rows:
            break

        # Find a row that maintains minimum spacing from existing walls
        valid_rows = []
        for row in available_rows:
            # Check if this row is far enough from all used rows
            too_close = any(abs(row - used_row) < min_gap_between_walls for used_row in used_rows)
            if not too_close:
                valid_rows.append(row)

        if not valid_rows:
            break  # No more valid positions

        # Choose a random valid row
        row = random.choice(valid_rows)

        # Generate balanced column position (avoid too close to edges)
        col = random.randint(3, width - 4)

        positions.append((row, col))
        used_rows.add(row)

        # Remove this row and nearby rows from available_rows
        available_rows = [r for r in available_rows if abs(r - row) >= min_gap_between_walls]

    return positions


def create_wall_segment(wall_thickness: int, col: int, width: int, wall_index: int = 0) -> str:
    """Create a wall segment that slides left and right uniformly."""
    # Calculate available space for positioning
    total_open_space = width - 2 - wall_thickness  # Subtract border walls

    # Ensure we have valid space
    if total_open_space < 2:
        # If wall is too thick, center it
        left_space = max(1, total_open_space // 2)
        right_space = total_open_space - left_space
    else:
        # Create sliding effect: alternate between left, center, right positions
        position_cycle = wall_index % 3

        if position_cycle == 0:  # Left position
            left_space = 1
            right_space = total_open_space - left_space
        elif position_cycle == 1:  # Center position
            left_space = total_open_space // 2
            right_space = total_open_space - left_space
        else:  # Right position
            right_space = 1
            left_space = total_open_space - right_space

    # Build the row
    row = ["#"]  # Left border

    # Add left open space
    row.extend(["."] * left_space)

    # Add wall
    row.extend(["#"] * wall_thickness)

    # Add right open space
    row.extend(["."] * right_space)

    row.append("#")  # Right border

    return "".join(row)


def generate_maze_level(level_num: int, wall_thickness: int, num_walls: int = 10) -> dict:
    """Generate a single maze level with specified wall thickness."""
    width = 20
    height = 50  # 48 + 2 for borders
    spawn_row = height - 3  # Near bottom

    # Create empty grid
    grid = []

    # Top border
    grid.append("#" * width)

    # Generate wall positions with proper spacing (at least 2 empty rows between walls)
    wall_positions = generate_wall_positions(
        num_walls, width, height, spawn_row, min_distance_from_spawn=5, min_gap_between_walls=3
    )
    wall_rows = {pos[0] for pos in wall_positions}

    # Track wall index for sliding effect
    wall_index = 0

    # Generate middle rows
    for row in range(1, height - 1):
        if row in wall_rows:
            # Create wall segment with sliding position
            grid.append(create_wall_segment(wall_thickness, 0, width, wall_index))
            wall_index += 1
        elif row == spawn_row:
            # Spawn row (20 total: # + 8 dots + S + 8 dots + #)
            spawn_line = "#" + "." * 8 + "S" + "." * 8 + "#"
            grid.append(spawn_line)
        else:
            # Empty row
            grid.append("#" + "." * (width - 2) + "#")

    # Bottom border
    grid.append("#" * width)

    # Create level data
    level = create_base_level_template()
    level["ascii"] = grid
    level["name"] = f"lvl{level_num}_maze_20x48"

    return level


def generate_all_maze_levels():
    """Generate the complete series of maze levels."""
    # Define the progression: (level_num, wall_thickness)
    level_configs = [
        (1, 1),  # Single wall tiles
        (2, 1),  # More single wall tiles
        (3, 2),  # Double wall tiles
        (4, 3),  # Triple wall tiles
        (5, 4),  # Quadruple wall tiles
        (6, 5),  # Five wall tiles
        (7, 8),  # Eight wall tiles
        (8, 10),  # Ten wall tiles
        (9, 12),  # Twelve wall tiles
    ]

    for level_num, wall_thickness in level_configs:
        # Reduce number of walls due to spacing requirements, but still increase with level
        num_walls = 6 + (level_num - 1) * 1

        level_data = generate_maze_level(level_num, wall_thickness, num_walls)

        filename = f"levels/lvl{level_num}_maze_20x48.json"

        with open(filename, "w") as f:
            json.dump(level_data, f, indent=4)

        print(
            f"Generated {filename} with {wall_thickness}-wide walls "
            f"and {num_walls} wall segments"
        )


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)

    print("Generating maze level series...")
    generate_all_maze_levels()
    print("Complete!")
