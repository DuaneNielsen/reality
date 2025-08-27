#!/usr/bin/env python3

import sys

import madrona_escape_room


def create_complex_ascii_level():
    """
    Create a complex ASCII level with MANY obstacles to stress test physics/collisions.
    32x32 room with lots of cubes and cylinders.
    """
    # Create a 32x32 ASCII level
    # # = wall, C = cube, O = cylinder, S = spawn, . = empty
    level_lines = []

    # Top wall
    level_lines.append("#" * 32)

    # First spawn row
    level_lines.append("#S" + "." * 29 + "#")

    # Add rows with various obstacles
    for row in range(2, 31):
        if row == 31:
            # Bottom wall
            level_lines.append("#" * 32)
        else:
            line = "#"
            for col in range(1, 31):
                if row % 5 == 0 and col % 5 == 0:
                    # Grid of cylinders every 5 tiles
                    line += "O"
                elif (row * 7 + col * 3) % 11 == 0:
                    # Scattered cubes using prime-based pattern
                    line += "C"
                else:
                    line += "."
            line += "#"
            level_lines.append(line)

    # Bottom wall
    level_lines.append("#" * 32)

    ascii_level = "\n".join(level_lines)

    cube_count = ascii_level.count("C")
    cylinder_count = ascii_level.count("O")
    print(f"Created ASCII level with {cube_count} cubes and {cylinder_count} cylinders")
    return ascii_level


def main():
    # Get number of iterations from command line, default to 100
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"Running {iterations} iterations with COMPLEX ASCII level...")
    print("Testing: Complex collision detection with many obstacles")

    # Create complex ASCII level once
    complex_level = create_complex_ascii_level()

    for run in range(iterations):
        try:
            mgr = madrona_escape_room.SimManager(
                exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
                gpu_id=0,
                num_worlds=4,  # Multiple worlds to stress test more
                rand_seed=42 + run,
                auto_reset=True,
                enable_batch_renderer=False,
                level_ascii=complex_level,  # Pass ASCII level
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
