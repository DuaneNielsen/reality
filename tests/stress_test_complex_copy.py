#!/usr/bin/env python3

import sys

import madrona_escape_room
from madrona_escape_room.ctypes_bindings import CompiledLevel
from madrona_escape_room.generated_constants import limits


def create_complex_level():
    """
    Create a complex level with MANY obstacles to stress test physics/collisions.
    Based on default_level.py but with way more obstacles.
    """
    # Create empty level
    level = CompiledLevel()

    # Make it bigger - 32x32
    level.width = 32
    level.height = 32
    level.world_scale = 1.0
    level.done_on_collide = False
    level.max_entities = 800  # LOTS of entities for stress testing
    level.level_name = b"complex_stress_test"

    # World boundaries for 32x32 room
    level.world_min_x = -40.0
    level.world_max_x = 40.0
    level.world_min_y = -40.0
    level.world_max_y = 40.0
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

    # Multiple spawn points to test spawn collision handling
    level.num_spawns = 4
    level.spawn_x[0] = -35.0
    level.spawn_y[0] = -35.0
    level.spawn_facing[0] = 0.0

    level.spawn_x[1] = 35.0
    level.spawn_y[1] = -35.0
    level.spawn_facing[1] = 3.14

    level.spawn_x[2] = -35.0
    level.spawn_y[2] = 35.0
    level.spawn_facing[2] = 1.57

    level.spawn_x[3] = 35.0
    level.spawn_y[3] = 35.0
    level.spawn_facing[3] = -1.57

    # Asset IDs from asset_ids.hpp
    WALL = 1
    CUBE = 0
    CYLINDER = 9

    # Generate border walls
    tile_index = 0
    wall_tile_size = 2.5
    walls_per_side = 32
    wall_edge = 38.75  # 40.0 - 2.5 * 0.5

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

    # Add a GRID of cylinders throughout the level for stress testing
    cylinder_z_offset = 2.55
    variance_3m = 1.0  # Less variance for more predictable collisions

    # Create a 6x6 grid of cylinders
    for grid_x in range(6):
        for grid_y in range(6):
            x = -30.0 + grid_x * 12.0  # Spread across the level
            y = -30.0 + grid_y * 12.0

            level.object_ids[tile_index] = CYLINDER
            level.tile_x[tile_index] = x
            level.tile_y[tile_index] = y
            level.tile_z[tile_index] = cylinder_z_offset
            level.tile_scale_x[tile_index] = 2.0  # Bigger for more collisions
            level.tile_scale_y[tile_index] = 2.0
            level.tile_scale_z[tile_index] = 2.0
            level.tile_persistent[tile_index] = False
            level.tile_render_only[tile_index] = False
            level.tile_entity_type[tile_index] = 1  # EntityType::Cube (objects)
            level.tile_response_type[tile_index] = 2  # ResponseType::Static
            level.tile_rand_x[tile_index] = variance_3m
            level.tile_rand_y[tile_index] = variance_3m
            level.tile_rand_scale_x[tile_index] = 0.5
            level.tile_rand_scale_y[tile_index] = 0.5
            level.tile_rand_rot_z[tile_index] = 6.28318  # Full 360° rotation
            tile_index += 1

    # Add MANY cubes scattered around for maximum collision testing
    cube_z_offset = 0.75
    rotation_range = 6.28318  # 2 * pi

    # Create a dense field of cubes
    import random

    random.seed(42)

    for _ in range(100):  # 100 randomly placed cubes!
        x = random.uniform(-35.0, 35.0)
        y = random.uniform(-35.0, 35.0)

        # Make sure we don't exceed limits.maxTiles
        if tile_index >= limits.maxTiles - 1:
            break

        level.object_ids[tile_index] = CUBE
        level.tile_x[tile_index] = x
        level.tile_y[tile_index] = y
        level.tile_z[tile_index] = cube_z_offset
        level.tile_scale_x[tile_index] = random.uniform(1.0, 2.5)
        level.tile_scale_y[tile_index] = random.uniform(1.0, 2.5)
        level.tile_scale_z[tile_index] = random.uniform(1.0, 2.5)
        level.tile_persistent[tile_index] = False
        level.tile_render_only[tile_index] = False
        level.tile_entity_type[tile_index] = 1  # EntityType::Cube
        level.tile_response_type[tile_index] = 2  # ResponseType::Static
        level.tile_rand_x[tile_index] = 0.5
        level.tile_rand_y[tile_index] = 0.5
        level.tile_rand_rot_z[tile_index] = rotation_range
        level.tile_rand_scale_x[tile_index] = 0.2
        level.tile_rand_scale_y[tile_index] = 0.2
        tile_index += 1

    # Set the actual number of tiles used
    level.num_tiles = tile_index
    print(f"Created level with {tile_index} entities")

    return level


def main():
    # Get number of iterations from command line, default to 100
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"Running {iterations} iterations with COMPLEX level...")
    print("Testing: Complex collision detection with hundreds of obstacles")

    # Create complex level once
    complex_level = create_complex_level()

    for run in range(iterations):
        try:
            mgr = madrona_escape_room.SimManager(
                exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
                gpu_id=0,
                num_worlds=4,  # Multiple worlds to stress test more
                rand_seed=42 + run,
                auto_reset=True,
                enable_batch_renderer=False,
                compiled_levels=[complex_level],
            )

            # Run many steps with random actions to trigger collisions
            import random

            random.seed(42 + run)

            for step in range(500):  # 500 steps per iteration
                # Set random actions to cause collisions
                action_tensor = mgr.action_tensor().to_numpy()
                # Check the shape first
                if len(action_tensor.shape) == 2:
                    # Shape is [num_worlds, action_dims]
                    for world in range(min(4, action_tensor.shape[0])):
                        # Random movement and rotation
                        action_tensor[world, 0] = random.randint(0, 3)  # move_amount
                        action_tensor[world, 1] = random.randint(0, 7)  # move_angle
                        action_tensor[world, 2] = random.randint(0, 4)  # rotate
                else:
                    # Shape is [num_worlds, num_agents, action_dims]
                    for world in range(min(4, action_tensor.shape[0])):
                        for agent in range(action_tensor.shape[1]):
                            # Random movement and rotation
                            action_tensor[world, agent, 0] = random.randint(0, 3)  # move_amount
                            action_tensor[world, agent, 1] = random.randint(0, 7)  # move_angle
                            action_tensor[world, agent, 2] = random.randint(0, 4)  # rotate

                mgr.step()

                # Periodically access tensors to check for memory corruption
                if step % 100 == 0:
                    _ = mgr.self_observation_tensor().to_numpy()
                    _ = mgr.done_tensor().to_numpy()
                    _ = mgr.reward_tensor().to_numpy()
                    _ = mgr.progress_tensor().to_numpy()

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
