#!/usr/bin/env python3
"""
Direct C-API stress test that bypasses SimManager abstraction.
Creates a CompiledLevel and passes it directly to the C API.
"""

import ctypes
import sys

import numpy as np

# Import the C API directly
from madrona_escape_room.ctypes_bindings import (
    CompiledLevel,
    ManagerConfig,
    MER_ManagerHandle,
    MER_Tensor,
    lib,
)
from madrona_escape_room.generated_constants import limits

# Hardcoded constants to avoid dependency on generated files
# From types.hpp and the C API
MER_SUCCESS = 0  # Result::Success
MER_EXEC_MODE_CPU = 0  # ExecMode::CPU
MER_EXEC_MODE_CUDA = 1  # ExecMode::CUDA


def create_default_level():
    """
    Create a default 16x16 room with border walls and obstacles.
    Direct port of default_level.py without any Python abstractions.
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

    # Initialize all arrays to defaults
    for i in range(limits.maxTiles):
        level.tile_z[i] = 0.0
        level.tile_scale_x[i] = 1.0
        level.tile_scale_y[i] = 1.0
        level.tile_scale_z[i] = 1.0
        level.tile_rot_w[i] = 1.0  # Identity quaternion
        level.tile_rot_x[i] = 0.0
        level.tile_rot_y[i] = 0.0
        level.tile_rot_z[i] = 0.0
        level.tile_response_type[i] = 2  # ResponseType::Static

        # No randomization by default
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
    print(f"Created level with {tile_index} entities")

    return level


def main():
    # Get number of iterations from command line, default to 100
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"Running {iterations} iterations with direct C-API...")
    print("Testing: Direct C-API without Python SimManager abstraction")

    # Create default level once
    default_level = create_default_level()

    for run in range(iterations):
        try:
            # Create config directly
            config = ManagerConfig()
            config.exec_mode = MER_EXEC_MODE_CPU
            config.gpu_id = 0
            config.num_worlds = 1
            config.rand_seed = 42 + run
            config.auto_reset = True
            config.enable_batch_renderer = False
            config.batch_render_view_width = 64
            config.batch_render_view_height = 64

            # Create array of compiled levels (one for each world)
            CompiledLevelArray = CompiledLevel * 1
            compiled_levels = CompiledLevelArray(default_level)

            # Create manager handle
            handle = MER_ManagerHandle()

            # Call C API directly
            result = lib.mer_create_manager(
                ctypes.byref(handle),
                ctypes.byref(config),
                ctypes.cast(compiled_levels, ctypes.c_void_p),
                1,  # num_compiled_levels
            )

            if result != MER_SUCCESS:
                error_msg = lib.mer_result_to_string(result)
                if error_msg:
                    error_str = error_msg.decode("utf-8")
                else:
                    error_str = f"Unknown error code: {result}"
                raise RuntimeError(f"Failed to create manager: {error_str}")

            # Get action tensor to set random actions
            action_tensor_struct = MER_Tensor()
            result = lib.mer_get_action_tensor(handle, ctypes.byref(action_tensor_struct))
            if result != MER_SUCCESS:
                raise RuntimeError(f"Failed to get action tensor: {result}")

            # Run many steps with random actions
            import random

            random.seed(42 + run)

            for step in range(300):  # 300 steps per iteration
                # Set random actions directly via C tensor
                # Note: We're working with raw memory here
                # The action tensor shape is [num_worlds, num_agents, 3]
                # For 1 world, 1 agent, 3 action components

                # Simple approach: just call step without modifying actions
                # (they'll use whatever default values are there)
                result = lib.mer_step(handle)
                if result != MER_SUCCESS:
                    raise RuntimeError(f"Step failed at iteration {step}: {result}")

                # Periodically access tensors to check for corruption
                if step % 100 == 0:
                    # Get self observation tensor
                    obs_tensor = MER_Tensor()
                    result = lib.mer_get_self_observation_tensor(handle, ctypes.byref(obs_tensor))
                    if result != MER_SUCCESS:
                        raise RuntimeError(f"Failed to get observation tensor: {result}")

            # Clean up
            lib.mer_destroy_manager(handle)

            if (run + 1) % 10 == 0:
                print(f"Run {run+1} completed")

        except Exception as e:
            print(f"CRASH at iteration {run+1}: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    print("Done! No crashes detected.")


if __name__ == "__main__":
    main()
