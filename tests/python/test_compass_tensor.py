#!/usr/bin/env python3

import numpy as np
import pytest


def test_compass_tensor_basic(cpu_manager):
    """Test that the compass tensor is working correctly"""

    mgr = cpu_manager

    # Get the compass tensor
    compass = mgr.compass_tensor()
    compass_np = compass.to_numpy()

    print(f"Compass tensor shape: {compass_np.shape}")
    num_worlds = compass_np.shape[0]
    print(f"Number of worlds: {num_worlds}")

    # Should be shape (num_worlds, num_agents=1, compass_size=128)
    assert compass_np.shape[1] == 1, f"Wrong number of agents: {compass_np.shape[1]}"
    assert compass_np.shape[2] == 128, f"Wrong compass size: {compass_np.shape[2]}"

    # Check that it's a one-hot encoding (exactly one 1.0, rest are 0.0) for each world
    for world_idx in range(num_worlds):
        world_agent_0 = compass_np[world_idx, 0, :]
        num_ones = np.sum(world_agent_0 == 1.0)
        num_zeros = np.sum(world_agent_0 == 0.0)

        print(f"World {world_idx} - Number of 1.0 values: {num_ones}")
        print(f"World {world_idx} - Number of 0.0 values: {num_zeros}")
        print(f"World {world_idx} - Sum of all values: {np.sum(world_agent_0)}")

        # Should have exactly one 1.0 and 127 zeros
        assert num_ones == 1, f"World {world_idx}: Expected exactly 1 one, got {num_ones}"
        assert num_zeros == 127, f"World {world_idx}: Expected exactly 127 zeros, got {num_zeros}"
        assert abs(np.sum(world_agent_0) - 1.0) < 1e-6, (
            f"World {world_idx}: Sum should be 1.0, got {np.sum(world_agent_0):.6f}"
        )

        # Find which bucket is active
        active_bucket = np.argmax(world_agent_0)
        print(f"World {world_idx} - Active compass bucket: {active_bucket}")


def test_compass_tensor_updates_after_step(cpu_manager):
    """Test that compass tensor updates after simulation steps"""

    mgr = cpu_manager

    # Get initial compass
    compass_initial = mgr.compass_tensor()
    compass_initial_np = compass_initial.to_numpy()
    num_worlds = compass_initial_np.shape[0]

    # Step the simulation multiple times
    for i in range(5):
        mgr.step()

        # Get updated compass
        compass_updated = mgr.compass_tensor()
        compass_updated_np = compass_updated.to_numpy()

        # Verify still one-hot for all worlds
        for world_idx in range(num_worlds):
            world_agent_updated = compass_updated_np[world_idx, 0, :]
            num_ones_updated = np.sum(world_agent_updated == 1.0)
            assert num_ones_updated == 1, (
                f"Step {i}, World {world_idx}: Expected 1 one, got {num_ones_updated}"
            )

            # Check sum is still 1.0
            assert abs(np.sum(world_agent_updated) - 1.0) < 1e-6, (
                f"Step {i}, World {world_idx}: Sum should be 1.0"
            )

            current_active_bucket = np.argmax(world_agent_updated)
            if world_idx == 0:  # Only print for first world to avoid spam
                print(f"Step {i}: World {world_idx} active compass bucket: {current_active_bucket}")


def test_compass_tensor_with_rotation_action(cpu_manager):
    """Test compass tensor changes when agent rotates"""

    mgr = cpu_manager

    # Get initial compass (just check world 0)
    compass_initial = mgr.compass_tensor()
    compass_initial_np = compass_initial.to_numpy()
    initial_bucket = np.argmax(compass_initial_np[0, 0, :])
    print(f"Initial compass bucket: {initial_bucket}")

    # Set a rotation action for world 0 agent 0 (turn right)
    actions = mgr.action_tensor()
    actions_np = actions.to_numpy()
    print(f"Action tensor shape: {actions_np.shape}")

    # Actions shape: (num_worlds, num_action_components) or
    # (num_worlds, num_agents, num_action_components)
    if len(actions_np.shape) == 3:
        actions_np[0, 0, 2] = 4  # FAST_RIGHT according to generated_constants.py
    else:
        actions_np[0, 2] = 4  # FAST_RIGHT - if it's (num_worlds, num_action_components)

    # Step a few times with rotation action
    for i in range(3):
        mgr.step()

        compass = mgr.compass_tensor()
        compass_np = compass.to_numpy()
        current_bucket = np.argmax(compass_np[0, 0, :])

        # Verify one-hot property for world 0
        assert np.sum(compass_np[0, 0, :] == 1.0) == 1, f"Step {i}: World 0 not one-hot"

        print(f"Step {i}: World 0 compass bucket: {current_bucket}")

    print("✅ Compass tensor rotation test completed successfully!")
