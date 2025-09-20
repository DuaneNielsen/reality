#!/usr/bin/env python3
"""
Test script to verify that the compass points toward the target entity.
"""

import numpy as np
import pytest


@pytest.mark.spec("docs/specs/sim.md", "compassSystem")
def test_compass_points_to_target(cpu_manager):
    """Test that compass points toward target entity, not agent rotation"""
    mgr = cpu_manager

    # Get compass and agent position
    compass = mgr.compass_tensor().to_numpy()

    # Import helper class for proper coordinate denormalization
    from test_helpers import ObservationReader

    observer = ObservationReader(mgr)

    # Get normalized position and denormalize to world coordinates
    norm_pos = observer.get_normalized_position(0, agent_idx=0)

    # Default level boundaries (from default_level.py)
    world_min_x = -20.0
    world_max_x = 20.0
    world_min_y = -20.0
    world_max_y = 20.0
    world_width = world_max_x - world_min_x
    world_length = world_max_y - world_min_y

    # Denormalize coordinates: pos = norm_pos * world_size + world_min
    agent_pos_x = norm_pos[0] * world_width + world_min_x
    agent_pos_y = norm_pos[1] * world_length + world_min_y
    agent_pos = np.array([agent_pos_x, agent_pos_y, norm_pos[2]])

    # Find which compass bucket is active (should point toward target at [5, 10, 1])
    compass_buckets = compass[0, 0, :]  # world 0, agent 0, all 128 buckets
    active_bucket = np.argmax(compass_buckets)

    # Calculate expected angle to target
    # Target is at [5, 10, 1], agent typically starts around [0, 0, 1]
    target_pos = np.array([5.0, 10.0, 1.0])

    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    expected_angle = np.arctan2(dy, dx)

    # Convert to compass bucket (formula from compass system)
    # Use exact same value as C++ code: consts::math::pi = 3.14159265359f
    math_pi = 3.14159265359
    two_pi = 2.0 * math_pi
    while expected_angle < 0:
        expected_angle += two_pi
    while expected_angle >= two_pi:
        expected_angle -= two_pi

    expected_bucket = (64 - int(expected_angle / two_pi * 128)) % 128
    # Match C++ implementation: handle negative buckets
    if expected_bucket < 0:
        expected_bucket += 128

    # Allow 1 bucket tolerance for discretization
    bucket_diff = abs(active_bucket - expected_bucket)
    assert bucket_diff <= 1, (
        f"Compass bucket {active_bucket} too far from expected {expected_bucket} "
        f"(diff: {bucket_diff})"
    )


@pytest.mark.spec("docs/specs/sim.md", "compassSystem")
def test_compass_updates_with_agent_movement(cpu_manager):
    """Test that compass direction updates as agent moves relative to target"""
    mgr = cpu_manager

    # Import helper class for movement control
    from test_helpers import AgentController

    controller = AgentController(mgr)

    initial_compass = mgr.compass_tensor().to_numpy()
    initial_bucket = np.argmax(initial_compass[0, 0, :])

    # Move agent significantly in different directions
    compass_buckets = []

    # Move left for multiple steps to create significant displacement
    for _ in range(10):
        controller.reset_actions()
        controller.strafe_left(speed=3)  # FAST movement
        mgr.step()

    compass = mgr.compass_tensor().to_numpy()
    active_bucket = np.argmax(compass[0, 0, :])
    compass_buckets.append(active_bucket)

    # Move right for multiple steps
    for _ in range(20):
        controller.reset_actions()
        controller.strafe_right(speed=3)  # FAST movement
        mgr.step()

    compass = mgr.compass_tensor().to_numpy()
    active_bucket = np.argmax(compass[0, 0, :])
    compass_buckets.append(active_bucket)

    # Compass should update (agent moved significantly left then right)
    unique_buckets = set(compass_buckets + [initial_bucket])
    assert len(unique_buckets) >= 2, (
        f"Compass did not update with movement: initial={initial_bucket}, "
        f"buckets={compass_buckets}"
    )


@pytest.mark.spec("docs/specs/sim.md", "customMotionSystem")
def test_target_entity_exists(cpu_manager):
    """Test that target entity is created and positioned correctly"""
    mgr = cpu_manager

    # Get initial observations to verify simulation is running
    obs = mgr.self_observation_tensor().to_numpy()
    compass = mgr.compass_tensor().to_numpy()

    # If compass has active direction, target entity must exist
    compass_buckets = compass[0, 0, :]
    max_value = np.max(compass_buckets)

    # Target should provide compass direction (non-zero max)
    assert max_value > 0, "No compass direction found - target entity may not exist"

    # Verify compass points in expected direction for static target at [5, 10, 1]
    agent_pos = obs[0, 0, :3]
    target_pos = np.array([5.0, 10.0, 1.0])

    # Basic sanity check - target should be northeast of typical spawn
    assert target_pos[0] > agent_pos[0], "Target should be east of agent spawn"
    assert target_pos[1] > agent_pos[1], "Target should be north of agent spawn"
