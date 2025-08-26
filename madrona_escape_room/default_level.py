"""
Default level generator for Madrona Escape Room
Creates a 16x16 room with walls and obstacles like default_level.cpp
"""

from .dataclass_structs import MAX_TILES, CompiledLevel


def create_default_level():
    """
    Create a default 16x16 room with border walls and obstacles.
    Python version of src/default_level.cpp
    """
    # Create empty level
    level = CompiledLevel()

    # Basic level configuration
    level.width = 16
    level.height = 16
    level.world_scale = 1.0
    level.done_on_collide = False
    level.max_entities = 150  # Enough for walls and objects
    level.level_name = b"default_16x16_room"

    # World boundaries for 16x16 room with 2.5 unit spacing per tile
    level.world_min_x = -20.0
    level.world_max_x = 20.0
    level.world_min_y = -20.0
    level.world_max_y = 20.0
    level.world_min_z = 0.0
    level.world_max_z = 25.0

    # Arrays are pre-initialized with proper sizes by dataclass!
    # Most values default to 0.0/False which is what we want
    # Only set non-zero defaults for the tiles we'll actually use

    # We'll only use about 100 tiles, so we don't need to set all 1024

    # Set spawn point at x=0, y=-17.0 (near southern wall)
    level.num_spawns = 1
    level.spawn_x[0] = 0.0
    level.spawn_y[0] = -17.0
    level.spawn_facing[0] = 0.0

    # Asset IDs from asset_ids.hpp
    WALL = 1
    CUBE = 0
    CYLINDER = 9
    AXIS_X = 5

    # Generate border walls
    tile_index = 0
    wall_tile_size = 2.5
    walls_per_side = 16
    wall_edge = 18.75  # 20.0 - 2.5 * 0.5

    # Top and bottom walls
    for i in range(walls_per_side):
        x = -wall_edge + i * wall_tile_size

        # Top wall
        level.object_ids[tile_index] = WALL
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = wall_edge
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        # tile_rotation is auto-initialized to identity quaternion (1,0,0,0) by factory
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        tile_index += 1

        # Bottom wall
        level.object_ids[tile_index] = WALL
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = -wall_edge
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        tile_index += 1

    # Left and right walls (skip corners to avoid overlaps)
    for i in range(1, walls_per_side - 1):
        y = -wall_edge + i * wall_tile_size

        # Left wall
        level.object_ids[tile_index] = WALL
        level.tile_x[tile_index] = -wall_edge
        level.tile_y[tile_index] = y
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        tile_index += 1

        # Right wall
        level.object_ids[tile_index] = WALL
        level.tile_x[tile_index] = wall_edge
        level.tile_y[tile_index] = y
        level.tile_scale_x[tile_index] = wall_tile_size
        level.tile_scale_y[tile_index] = wall_tile_size
        level.tile_scale_z[tile_index] = 1.0
        level.tile_persistent[tile_index] = True
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        tile_index += 1

    # Add an axis marker at x=0, y=12.5 for visual reference
    level.object_ids[tile_index] = AXIS_X
    level.tile_x[tile_index] = 0.0
    level.tile_y[tile_index] = 12.5
    level.tile_persistent[tile_index] = True
    level.tile_render_only[tile_index] = True
    level.tile_entity_type[tile_index] = 0  # EntityType::NoEntity
    level.tile_response_type[tile_index] = 2  # ResponseType::Static
    tile_index += 1

    # Add cylinders with randomization
    cylinder_z_offset = 2.55
    variance_3m = 3.0

    cylinder_positions = [
        (-10.0, 10.0),  # Near top-left
        (8.0, 12.0),  # Near top-right
        (-12.0, -3.0),  # Left side
        (11.0, 3.0),  # Right side
        (3.0, -2.0),  # Near center
        (-7.0, -10.0),  # Bottom-left
        (9.0, -8.0),  # Bottom-right
        (-5.0, 4.0),  # Mid-left
    ]

    for x, y in cylinder_positions:
        level.object_ids[tile_index] = CYLINDER
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = y
        level.tile_z[tile_index] = cylinder_z_offset
        level.tile_scale_x[tile_index] = 1.7
        level.tile_scale_y[tile_index] = 1.7
        level.tile_scale_z[tile_index] = 1.7
        level.tile_persistent[tile_index] = False
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 1  # EntityType::Cube (objects)
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        level.tile_rand_x[tile_index] = variance_3m
        level.tile_rand_y[tile_index] = variance_3m
        level.tile_rand_scale_x[tile_index] = 1.5
        level.tile_rand_scale_y[tile_index] = 1.5
        level.tile_rand_rot_z[tile_index] = 6.28318  # Full 360° rotation
        tile_index += 1

    # Add cubes with randomization
    cube_z_offset = 0.75
    rotation_range = 6.28318  # 2 * pi

    cube_positions = [
        (-8.0, 6.0),  # Upper-left
        (6.0, 8.0),  # Upper-right
        (-10.0, -6.0),  # Lower-left
        (7.0, -5.0),  # Lower-right
        (-2.0, 1.0),  # Near center
    ]

    for x, y in cube_positions:
        level.object_ids[tile_index] = CUBE
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = y
        level.tile_z[tile_index] = cube_z_offset
        level.tile_scale_x[tile_index] = 1.5
        level.tile_scale_y[tile_index] = 1.5
        level.tile_scale_z[tile_index] = 1.5
        level.tile_persistent[tile_index] = False
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 1  # EntityType::Cube
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        level.tile_rand_x[tile_index] = variance_3m
        level.tile_rand_y[tile_index] = variance_3m
        level.tile_rand_rot_z[tile_index] = rotation_range
        level.tile_rand_scale_x[tile_index] = 0.4
        level.tile_rand_scale_y[tile_index] = 0.4
        tile_index += 1

    # Set the actual number of tiles used
    level.num_tiles = tile_index

    return level
