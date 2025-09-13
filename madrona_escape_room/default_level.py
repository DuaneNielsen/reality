"""
Default level generator for Madrona Escape Room
Creates a 16x16 room with walls and obstacles like default_level.cpp
Now produces two levels to match C++ implementation: full obstacles and cubes only
"""

from .dataclass_utils import create_compiled_level
from .generated_constants import AssetIDs, EntityType, ResponseType, consts, limits
from .generated_dataclasses import CompiledLevel


def create_base_level_template():
    """Helper function to initialize a CompiledLevel with common settings."""
    level = create_compiled_level()

    level.width = 16
    level.height = 16
    level.world_scale = 1.0
    level.max_entities = 150  # Enough for walls and obstacles

    # World boundaries using constants from consts.hpp
    level.world_min_x = -consts.worldWidth / 2.0  # -10.0
    level.world_max_x = consts.worldWidth / 2.0  # +10.0
    level.world_min_y = -consts.worldLength / 2.0  # -20.0
    level.world_max_y = consts.worldLength / 2.0  # +20.0
    level.world_min_z = 0.0  # Floor level
    level.world_max_z = 25.0  # Reasonable max height

    # Set spawn point at x=0, y=-17.0 (near southern wall)
    level.num_spawns = 1
    level.spawn_x[0] = 0.0
    level.spawn_y[0] = -17.0
    level.spawn_facing[0] = 0.0

    return level


def generate_walls(level, start_tile_index):
    """Generate border walls for a level."""
    tile_index = start_tile_index
    wall_tile_size = 2.5
    walls_per_side = 16  # 16 wall tiles per side
    room_size = walls_per_side * wall_tile_size  # 40 units
    half_room = room_size / 2.0  # 20.0

    # Calculate wall edge position (walls should be at the edge of the room)
    wall_edge = half_room - wall_tile_size * 0.5  # 18.75

    # Top and bottom walls
    for i in range(walls_per_side):
        x = -wall_edge + i * wall_tile_size  # Start from left edge and increment

        # Top wall
        level.object_ids[tile_index] = AssetIDs.WALL
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = wall_edge
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_done_on_collide[tile_index] = False  # Walls don't trigger episode end
        level.tile_entity_type[tile_index] = EntityType.Wall
        level.tile_response_type[tile_index] = ResponseType.Static
        tile_index += 1

        # Bottom wall
        level.object_ids[tile_index] = AssetIDs.WALL
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = -wall_edge
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_done_on_collide[tile_index] = False  # Walls don't trigger episode end
        level.tile_entity_type[tile_index] = EntityType.Wall
        level.tile_response_type[tile_index] = ResponseType.Static
        tile_index += 1

    # Left and right walls (skip corners to avoid overlaps)
    for i in range(1, walls_per_side - 1):
        y = -wall_edge + i * wall_tile_size

        # Left wall
        level.object_ids[tile_index] = AssetIDs.WALL
        level.tile_x[tile_index] = -wall_edge
        level.tile_y[tile_index] = y
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_done_on_collide[tile_index] = False  # Walls don't trigger episode end
        level.tile_entity_type[tile_index] = EntityType.Wall
        level.tile_response_type[tile_index] = ResponseType.Static
        tile_index += 1

        # Right wall
        level.object_ids[tile_index] = AssetIDs.WALL
        level.tile_x[tile_index] = wall_edge
        level.tile_y[tile_index] = y
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_done_on_collide[tile_index] = False  # Walls don't trigger episode end
        level.tile_entity_type[tile_index] = EntityType.Wall
        level.tile_response_type[tile_index] = ResponseType.Static
        tile_index += 1

    return tile_index


def generate_axis_marker(level, start_tile_index):
    """Add axis marker for visual reference."""
    tile_index = start_tile_index

    level.object_ids[tile_index] = AssetIDs.AXIS_X
    level.tile_x[tile_index] = 0.0
    level.tile_y[tile_index] = 12.5
    level.tile_persistent[tile_index] = True
    level.tile_render_only[tile_index] = True
    level.tile_done_on_collide[tile_index] = False  # Render-only, no collision
    level.tile_entity_type[tile_index] = EntityType.NoEntity
    level.tile_response_type[tile_index] = ResponseType.Static
    tile_index += 1

    return tile_index


def generate_cylinders(level, start_tile_index):
    """Generate cylinder obstacles."""
    tile_index = start_tile_index
    cylinder_z_offset = 2.55  # Adjusted for 1.7x scale cylinders
    variance_3m = 3.0  # 3-meter variance for XY positions

    # Cylinder positions (8 cylinders)
    cylinder_positions = [
        (-10.0, 10.0),  # Near top-left corner
        (8.0, 12.0),  # Near top-right corner
        (-12.0, -3.0),  # Left side
        (11.0, 3.0),  # Right side
        (3.0, -2.0),  # Near center but offset
        (-7.0, -10.0),  # Bottom-left area
        (9.0, -8.0),  # Bottom-right area
        (-5.0, 4.0),  # Mid-left
    ]

    for x, y in cylinder_positions:
        level.object_ids[tile_index] = AssetIDs.CYLINDER
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = y
        level.tile_z[tile_index] = cylinder_z_offset
        level.tile_scale_x[tile_index] = 1.7  # 1.7x base size
        level.tile_scale_y[tile_index] = 1.7
        level.tile_scale_z[tile_index] = 1.7
        level.tile_persistent[tile_index] = False
        level.tile_render_only[tile_index] = False
        level.tile_done_on_collide[tile_index] = True  # Obstacles trigger episode end
        level.tile_entity_type[tile_index] = EntityType.Cube  # Static obstacles
        level.tile_response_type[tile_index] = ResponseType.Static
        level.tile_rand_x[tile_index] = variance_3m  # 3m variance in X
        level.tile_rand_y[tile_index] = variance_3m  # 3m variance in Y
        level.tile_rand_scale_x[tile_index] = 1.5  # ±150% scale variation in X
        level.tile_rand_scale_y[tile_index] = 1.5  # ±150% scale variation in Y
        level.tile_rand_rot_z[tile_index] = 2.0 * consts.math.pi  # Full 360° rotation randomization
        tile_index += 1

    return tile_index


def generate_cubes(level, start_tile_index):
    """Generate cube obstacles."""
    tile_index = start_tile_index
    cube_z_offset = 0.75  # Half of scaled cube height (1.5 * 1.0 / 2)
    variance_3m = 3.0  # 3-meter variance for XY positions
    rotation_range = 2.0 * consts.math.pi  # Full rotation range (360 degrees)

    # Cube positions (5 cubes)
    cube_positions = [
        (-8.0, 6.0),  # Upper-left quadrant
        (6.0, 8.0),  # Upper-right quadrant
        (-10.0, -6.0),  # Lower-left quadrant
        (7.0, -5.0),  # Lower-right quadrant
        (-2.0, 1.0),  # Near center
    ]

    for x, y in cube_positions:
        level.object_ids[tile_index] = AssetIDs.CUBE
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = y
        level.tile_z[tile_index] = cube_z_offset
        level.tile_scale_x[tile_index] = 1.5  # 50% larger
        level.tile_scale_y[tile_index] = 1.5
        level.tile_scale_z[tile_index] = 1.5
        level.tile_persistent[tile_index] = False
        level.tile_render_only[tile_index] = False
        level.tile_done_on_collide[tile_index] = True  # Obstacles trigger episode end
        level.tile_entity_type[tile_index] = EntityType.Cube
        level.tile_response_type[tile_index] = ResponseType.Static
        level.tile_rand_x[tile_index] = variance_3m
        level.tile_rand_y[tile_index] = variance_3m
        level.tile_rand_rot_z[tile_index] = rotation_range
        level.tile_rand_scale_x[tile_index] = 0.4  # ±40% scale variation
        level.tile_rand_scale_y[tile_index] = 0.4
        tile_index += 1

    return tile_index


def create_default_levels():
    """Create two default levels: full obstacles and cubes only.

    Returns:
        list[CompiledLevel]: List containing both levels
    """
    levels = []

    # Create first level: Full obstacles (cubes + cylinders)
    level1 = create_base_level_template()
    level1.level_name = b"default_full_obstacles"

    tile_index = 0
    tile_index = generate_walls(level1, tile_index)
    tile_index = generate_axis_marker(level1, tile_index)
    tile_index = generate_cylinders(level1, tile_index)
    tile_index = generate_cubes(level1, tile_index)
    level1.num_tiles = tile_index
    levels.append(level1)

    # Create second level: Cubes only (no cylinders)
    level2 = create_base_level_template()
    level2.level_name = b"default_cubes_only"

    tile_index = 0
    tile_index = generate_walls(level2, tile_index)
    tile_index = generate_axis_marker(level2, tile_index)
    tile_index = generate_cubes(level2, tile_index)  # No cylinders
    level2.num_tiles = tile_index
    levels.append(level2)

    return levels


def create_default_level():
    """Create single default level for backward compatibility.

    Returns:
        CompiledLevel: The first level (full obstacles)
    """
    return create_default_levels()[0]


def main():
    """Command line interface for generating default level files."""
    import argparse
    import sys
    from pathlib import Path

    from .level_io import save_compiled_levels

    parser = argparse.ArgumentParser(
        description="Generate default level file for Madrona Escape Room",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default level file (2 levels: full obstacles + cubes only)
  python -m madrona_escape_room.default_level default.lvl

  # Generate and display info
  python -m madrona_escape_room.default_level default.lvl --info
        """,
    )

    parser.add_argument("output", help="Output binary .lvl file")
    parser.add_argument("--info", action="store_true", help="Display level info after generation")

    args = parser.parse_args()

    try:
        print("Generating default levels...")
        levels = create_default_levels()

        total_tiles = sum(level.num_tiles for level in levels)
        print(f"Generated {len(levels)} levels with {total_tiles} total tiles")
        print(f"Saving to '{args.output}'...")

        # Save the levels using unified format
        save_compiled_levels(levels, args.output)
        print("✓ Default levels saved successfully")

        if args.info:
            print(f"\nLevel Information ({len(levels)} levels):")
            for i, level in enumerate(levels, 1):
                print(f"\nLevel {i}/{len(levels)}:")
                print(f"  Name: {level.level_name.decode('utf-8', errors='ignore')}")
                print(f"  Dimensions: {level.width}x{level.height} (scale: {level.world_scale})")
                print(f"  Tiles: {level.num_tiles}")
                print(f"  Max entities: {level.max_entities}")
                print(f"  Spawn points: {level.num_spawns}")

                for j in range(level.num_spawns):
                    x = level.spawn_x[j]
                    y = level.spawn_y[j]
                    facing_rad = level.spawn_facing[j]
                    facing_deg = facing_rad * 180.0 / 3.14159
                    print(f"    Spawn {j}: ({x:.1f}, {y:.1f}) facing {facing_deg:.1f}°")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
