# ruff: noqa
#!/usr/bin/env python3
"""Python version of default_level.cpp - creates a 16x16 room with walls and objects."""

import ctypes
import math
import os
import sys

# First, run the struct generator to get our types
exec(open("scratch/generate_structs_from_binary.py").read())


# Asset IDs from asset_ids.hpp
class AssetIDs:
    INVALID = 0
    CUBE = 1
    WALL = 2
    AGENT = 3
    PLANE = 4
    AXIS_X = 5
    AXIS_Y = 6
    AXIS_Z = 7
    CYLINDER = 8


def create_default_level():
    """Create a 16x16 room with border walls, matching default_level.cpp."""
    level = CompiledLevel()

    # Basic metadata
    level.width = 16
    level.height = 16
    level.world_scale = 1.0
    level.done_on_collide = False
    level.max_entities = 150  # Enough for walls (16*4 = 64) and other objects

    # Set level name
    level_name = b"default_16x16_room"
    for i in range(min(len(level_name), 64)):
        level.level_name[i] = level_name[i]
    if len(level_name) < 64:
        level.level_name[len(level_name)] = 0

    # World boundaries for 16x16 room with 2.5 unit spacing per tile
    # Room is 40x40 units total, centered at origin
    level.world_min_x = -20.0
    level.world_max_x = 20.0
    level.world_min_y = -20.0
    level.world_max_y = 20.0
    level.world_min_z = 0.0
    level.world_max_z = 25.0

    # Initialize all transform data to defaults
    for i in range(1024):  # MAX_TILES
        level.tile_z[i] = 0.0
        level.tile_scale_x[i] = 1.0
        level.tile_scale_y[i] = 1.0
        level.tile_scale_z[i] = 1.0
        level.tile_rot_w[i] = 1.0  # Identity quaternion
        level.tile_rot_x[i] = 0.0
        level.tile_rot_y[i] = 0.0
        level.tile_rot_z[i] = 0.0
        level.tile_response_type[i] = 2  # Default to Static

        # Initialize randomization arrays to 0 (no randomization)
        level.tile_rand_x[i] = 0.0
        level.tile_rand_y[i] = 0.0
        level.tile_rand_z[i] = 0.0
        level.tile_rand_rot_z[i] = 0.0
        level.tile_rand_scale_x[i] = 0.0
        level.tile_rand_scale_y[i] = 0.0
        level.tile_rand_scale_z[i] = 0.0

    # Set spawn point at x=0, y=-17.0 (near southern wall)
    level.num_spawns = 1
    level.spawn_x[0] = 0.0
    level.spawn_y[0] = -17.0
    level.spawn_facing[0] = 0.0

    # Generate border walls with 2.5 unit tile spacing
    tile_index = 0
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
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
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
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
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
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
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
        level.tile_entity_type[tile_index] = 2  # EntityType::Wall
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        tile_index += 1

    # Add an axis marker at x=0, y=12.5 for visual reference
    level.object_ids[tile_index] = AssetIDs.AXIS_X
    level.tile_x[tile_index] = 0.0
    level.tile_y[tile_index] = 12.5
    level.tile_persistent[tile_index] = True
    level.tile_render_only[tile_index] = True
    level.tile_entity_type[tile_index] = 0  # EntityType::None
    level.tile_response_type[tile_index] = 2  # ResponseType::Static (render-only)
    tile_index += 1

    # Add cylinders scattered around the level with 3m XY variance
    cylinder_z_offset = 2.55  # Adjusted for 1.7x scale cylinders
    variance_3m = 3.0  # 3-meter variance for XY positions

    # List of cylinder positions
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
        level.tile_entity_type[tile_index] = 1  # EntityType::Object (static obstacle)
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        level.tile_rand_x[tile_index] = variance_3m  # 3m variance in X
        level.tile_rand_y[tile_index] = variance_3m  # 3m variance in Y
        level.tile_rand_scale_x[tile_index] = 1.5  # ±150% scale variation in X
        level.tile_rand_scale_y[tile_index] = 1.5  # ±150% scale variation in Y
        level.tile_rand_rot_z[tile_index] = 6.28318  # Full 360° rotation randomization
        tile_index += 1

    # Add cubes with physics, XY variance, and random rotation
    cube_z_offset = 0.75  # Half of scaled cube height (1.5 * 1.0 / 2)
    rotation_range = 2.0 * math.pi  # Full rotation range (360 degrees)

    # List of cube positions
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
        level.tile_render_only[tile_index] = False  # Has physics
        level.tile_entity_type[tile_index] = 1  # EntityType::Cube
        level.tile_response_type[tile_index] = 2  # ResponseType::Static (immovable)
        level.tile_rand_x[tile_index] = variance_3m
        level.tile_rand_y[tile_index] = variance_3m
        level.tile_rand_rot_z[tile_index] = rotation_range  # Random Z-axis rotation
        level.tile_rand_scale_x[tile_index] = 0.4  # ±40% scale variation
        level.tile_rand_scale_y[tile_index] = 0.4
        # No Z randomization to keep cubes at consistent height
        tile_index += 1

    # Set the actual number of tiles used
    level.num_tiles = tile_index

    print(f"Created level with {tile_index} tiles:")
    print(f"  - {walls_per_side * 2 + (walls_per_side - 2) * 2} walls")
    print(f"  - {len(cylinder_positions)} cylinders")
    print(f"  - {len(cube_positions)} cubes")
    print("  - 1 axis marker")

    return level


def main():
    """Test the default level with SimManager."""
    print("Creating default 16x16 room level...")
    level = create_default_level()

    print("\nLevel details:")
    print(f"  Name: {bytes(level.level_name).split(b'\\0')[0].decode()}")
    print(f"  Size: {level.width}x{level.height}")
    print(f"  Tiles: {level.num_tiles}")
    print(
        f"  World bounds: X[{level.world_min_x}, {level.world_max_x}], Y[{level.world_min_y}, {level.world_max_y}]"
    )
    print(f"  Spawn: ({level.spawn_x[0]}, {level.spawn_y[0]})")

    # Load the C API library
    lib_path = os.path.join(os.getcwd(), "build/libmadrona_escape_room_c_api.so")
    if not os.path.exists(lib_path):
        print(f"Library not found at {lib_path}")
        return

    lib = ctypes.CDLL(lib_path)

    # Define Handle and Config structures
    class Handle(ctypes.Structure):
        _fields_ = [("ptr", ctypes.c_void_p)]

    class Config(ctypes.Structure):
        _fields_ = [
            ("execMode", ctypes.c_int),  # enum ExecMode at offset 0
            ("gpuID", ctypes.c_int),  # at offset 4
            ("numWorlds", ctypes.c_uint32),  # at offset 8
            ("randSeed", ctypes.c_uint32),  # at offset 12
            ("autoReset", ctypes.c_bool),  # at offset 16
            ("enableBatchRenderer", ctypes.c_bool),  # at offset 17
            ("_pad1", ctypes.c_byte * 2),  # 2-byte padding
            ("batchRenderViewWidth", ctypes.c_uint32),  # at offset 20
            ("batchRenderViewHeight", ctypes.c_uint32),  # at offset 24
            ("_pad2", ctypes.c_byte * 4),  # 4-byte padding
            ("extRenderAPI", ctypes.c_void_p),  # at offset 32
            ("extRenderDev", ctypes.c_void_p),  # at offset 40
            ("enableTrajectoryTracking", ctypes.c_bool),  # at offset 48
            ("_pad3", ctypes.c_byte * 7),  # 7-byte padding
        ]

    # Setup function signatures
    lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(CompiledLevel)]
    lib.mer_validate_compiled_level.restype = ctypes.c_int

    lib.mer_result_to_string.argtypes = [ctypes.c_int]
    lib.mer_result_to_string.restype = ctypes.c_char_p

    lib.mer_create_manager.argtypes = [
        ctypes.POINTER(Handle),
        ctypes.POINTER(Config),
        ctypes.POINTER(CompiledLevel),
        ctypes.c_uint32,
    ]
    lib.mer_create_manager.restype = ctypes.c_int

    lib.mer_step.argtypes = [ctypes.POINTER(Handle)]
    lib.mer_step.restype = None

    lib.mer_destroy_manager.argtypes = [ctypes.POINTER(Handle)]
    lib.mer_destroy_manager.restype = None

    # Validate the level
    print("\nValidating level...")
    validation_result = lib.mer_validate_compiled_level(ctypes.byref(level))
    if validation_result != 0:
        error_msg = lib.mer_result_to_string(validation_result)
        print(f"Level validation failed: {error_msg.decode() if error_msg else 'Unknown error'}")
        return
    print("Level validation passed!")

    # Create manager configuration
    config = Config()
    config.execMode = 0  # 0 = CPU, 1 = CUDA (swapped from what I thought!)
    config.gpuID = 0
    config.numWorlds = 1
    config.randSeed = 42
    config.autoReset = True
    config.enableBatchRenderer = False
    config.batchRenderViewWidth = 64
    config.batchRenderViewHeight = 64
    config.extRenderAPI = None
    config.extRenderDev = None
    config.enableTrajectoryTracking = False

    # Create the manager
    handle = Handle()
    print("\nCreating SimManager...")
    print(f"Config: execMode={config.execMode}, numWorlds={config.numWorlds}")

    try:
        result = lib.mer_create_manager(
            ctypes.byref(handle),
            ctypes.byref(config),
            ctypes.byref(level),
            1,  # num_levels = 1
        )
    except Exception as e:
        print(f"Exception during create_manager: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"mer_create_manager returned: {result}")

    if result != 0:
        print(f"Failed to create manager! Error code: {result}")
        error_msg = lib.mer_result_to_string(result)
        print(f"Error message: {error_msg.decode() if error_msg else 'Unknown'}")
        return

    print("SimManager created successfully!")

    # Run a few simulation steps
    print("\nRunning 10 simulation steps...")
    for i in range(10):
        lib.mer_step(ctypes.byref(handle))
        print(f"  Step {i+1} completed")

    # Clean up
    print("\nDestroying manager...")
    lib.mer_destroy_manager(ctypes.byref(handle))
    print("Done!")


if __name__ == "__main__":
    main()
