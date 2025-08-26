#!/usr/bin/env python3
"""Test that the custom_level marker works with compile_ascii_level"""

import numpy as np
import pytest


@pytest.mark.custom_level("""
#####
#S.C#
#####
""")
def test_custom_level_with_cube(cpu_manager):
    """Test that custom level with a cube is loaded correctly"""
    mgr = cpu_manager

    # Don't reset immediately - check initial state first
    obs_tensor = mgr.self_observation_tensor()
    obs = obs_tensor.to_numpy()

    print("Initial agent position (before any reset):")
    for world_idx in range(4):
        print(f"  World {world_idx}: {obs[world_idx, 0, :3]}")

    # Now reset to ensure clean state
    mgr.reset_tensor().to_numpy()[...] = 1
    mgr.step()

    # Get initial position
    obs = mgr.self_observation_tensor().to_numpy()
    initial_pos = obs[0, 0, :3]  # World 0, agent 0, position

    print(f"Agent spawned at position: {initial_pos}")
    print(f"Observation shape: {obs.shape}")

    # Basic sanity checks
    assert obs.shape[0] == 4, "Should have 4 worlds"
    assert obs.shape[1] == 1, "Should have 1 agent per world"

    # The agent should spawn at 'S' position
    # With a 5x3 grid and scale 2.5:
    # S is at grid position (1, 1)
    # World coordinates: ((1 - 2) * 2.5, -(1 - 1) * 2.5) = (-2.5, 0)
    # Note: Y is inverted in the compiler, so -(1 - 1) * 2.5 = 0
    expected_x = -2.5
    expected_y = -0.0  # Should be 0 or close to it

    print(f"Expected spawn: ({expected_x}, {expected_y})")
    print(f"Actual spawn: ({initial_pos[0]:.2f}, {initial_pos[1]:.2f})")

    # Allow some tolerance for floating point
    assert abs(initial_pos[0] - expected_x) < 0.5, f"X position {initial_pos[0]} != {expected_x}"
    assert abs(initial_pos[1] - expected_y) < 0.5, f"Y position {initial_pos[1]} != {expected_y}"

    print("✓ Custom level test passed!")
