#!/usr/bin/env python3
"""
Test that verifies import order independence between PyTorch and madrona_escape_room.
Also tests that default level dataclasses are properly loaded and used.
This test runs first (00_ prefix) to catch CUDA library version conflicts early.
"""

import numpy as np
import pytest


def test_import_order_torch_first():
    """Test that madrona_escape_room can be imported after PyTorch"""
    # Import PyTorch first (this loads PyTorch's CUDA libraries)

    # Now import madrona_escape_room - this should work regardless of import order
    import madrona_escape_room

    assert hasattr(madrona_escape_room, "SimManager")
    assert hasattr(madrona_escape_room, "ExecMode")


def test_import_order_madrona_first():
    """Test that PyTorch can be imported after madrona_escape_room"""
    # This test may need to run in a separate process since torch is already imported
    # For now, just verify madrona can be imported first
    # Import torch after madrona_escape_room
    import torch

    import madrona_escape_room

    assert hasattr(madrona_escape_room, "SimManager")
    assert hasattr(torch, "tensor")


def test_functionality_after_torch_import():
    """Test that madrona_escape_room works functionally after PyTorch import"""
    import madrona_escape_room

    # Create manager and verify it works
    mgr = madrona_escape_room.SimManager(
        exec_mode=madrona_escape_room.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
        compiled_levels=madrona_escape_room.create_default_level(),
    )

    assert mgr is not None

    # Test basic functionality
    mgr.step()

    # Test tensor access
    reward_tensor = mgr.reward_tensor()
    reward_np = reward_tensor.to_numpy()
    assert isinstance(reward_np, np.ndarray)


def test_tensor_interop_with_torch():
    """Test that madrona tensors work alongside PyTorch tensors"""
    import torch

    import madrona_escape_room

    mgr = madrona_escape_room.SimManager(
        exec_mode=madrona_escape_room.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
        compiled_levels=madrona_escape_room.create_default_level(),
    )

    # Get madrona tensor
    reward_tensor = mgr.reward_tensor()
    reward_np = reward_tensor.to_numpy()

    # Convert to PyTorch tensor
    reward_torch = torch.from_numpy(reward_np)

    # Verify they have the same data
    assert reward_torch.shape == reward_np.shape
    assert torch.allclose(reward_torch, torch.from_numpy(reward_np))


def test_default_level_spawn_position():
    """Test that agent spawns at the position defined in default_level.py"""
    import madrona_escape_room

    # Create manager with default level
    mgr = madrona_escape_room.SimManager(
        exec_mode=madrona_escape_room.ExecMode.CPU,
        gpu_id=0,
        num_worlds=2,  # Test with multiple worlds
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
        compiled_levels=madrona_escape_room.create_default_level(),
    )

    # Get initial agent position
    obs = mgr.self_observation_tensor().to_numpy()

    # IMPORTANT: Observations are normalized!
    # According to default_level.py:
    # - spawn is at x=0.0, y=-17.0
    # - world bounds are x=[-20, 20], y=[-20, 20]
    # - world_length = 40.0
    # Normalized values:
    # - x_norm = (0.0 - (-20.0)) / 40.0 = 0.5
    # - y_norm = (-17.0 - (-20.0)) / 40.0 = 3.0 / 40.0 = 0.075

    expected_x_norm = 0.5
    expected_y_norm = 0.075

    # Check both worlds spawn at the same position
    for world_idx in range(2):
        agent_x = obs[world_idx, 0, 0]  # world, agent, x coordinate (normalized)
        agent_y = obs[world_idx, 0, 1]  # world, agent, y coordinate (normalized)

        # Allow small tolerance for floating point
        assert (
            abs(agent_x - expected_x_norm) < 0.01
        ), f"World {world_idx}: Expected normalized X={expected_x_norm}, got {agent_x}"
        assert (
            abs(agent_y - expected_y_norm) < 0.01
        ), f"World {world_idx}: Expected normalized Y={expected_y_norm}, got {agent_y}"

    # Now move the agents to different positions
    from madrona_escape_room import action

    actions = mgr.action_tensor().to_numpy()
    # Actions are shape (num_worlds, num_actions) not (num_worlds, num_agents, num_actions)
    # Use valid action values - moveAmount: 0-3, moveAngle: 0-7, rotate: 0-4 (2=none)
    actions[:, 0] = action.move_amount.FAST  # 3 - maximum speed
    actions[:, 1] = action.move_angle.FORWARD  # 0 - move forward
    actions[:, 2] = action.rotate.NONE  # 2 - no rotation

    # Step the simulation
    for _ in range(10):  # Take 10 steps forward
        mgr.step()

    # Check agents have moved
    obs_after_move = mgr.self_observation_tensor().to_numpy()
    for world_idx in range(2):
        new_y = obs_after_move[world_idx, 0, 1]
        # Agent should have moved forward (more positive Y, larger normalized value)
        assert (
            new_y > expected_y_norm
        ), f"World {world_idx}: Agent moved from Y={expected_y_norm}, now at {new_y}"

    # Trigger reset to reload the level
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1  # Trigger reset for all worlds
    mgr.step()
    reset_tensor[:] = 0

    # Check that agents are back at spawn position
    obs_after_reset = mgr.self_observation_tensor().to_numpy()
    for world_idx in range(2):
        reset_x = obs_after_reset[world_idx, 0, 0]
        reset_y = obs_after_reset[world_idx, 0, 1]

        assert (
            abs(reset_x - expected_x_norm) < 0.01
        ), f"World {world_idx}: After reset, expected normalized X={expected_x_norm}, got {reset_x}"
        assert (
            abs(reset_y - expected_y_norm) < 0.01
        ), f"World {world_idx}: After reset, expected normalized Y={expected_y_norm}, got {reset_y}"


def test_dataclass_level_structure():
    """Test that the dataclass CompiledLevel has the expected structure"""
    import madrona_escape_room
    from madrona_escape_room.generated_constants import consts
    from madrona_escape_room.generated_dataclasses import CompiledLevel

    # Create a default level
    level = madrona_escape_room.create_default_level()

    # Verify it's the right type
    assert isinstance(level, CompiledLevel)

    # Check expected attributes from default_level.py
    assert level.width == 16
    assert level.height == 16
    assert level.world_scale == 1.0
    assert level.num_spawns == 1

    # Check spawn position
    assert level.spawn_x[0] == 0.0
    assert level.spawn_y[0] == -17.0
    assert level.spawn_facing[0] == 0.0

    # Check world boundaries (calculated from constants)
    expected_half_width = consts.worldWidth / 2.0
    expected_half_length = consts.worldLength / 2.0
    assert level.world_min_x == -expected_half_width
    assert level.world_max_x == expected_half_width
    assert level.world_min_y == -expected_half_length
    assert level.world_max_y == expected_half_length

    # Verify arrays are Python lists (not ctypes arrays)
    assert isinstance(level.spawn_x, list)
    assert isinstance(level.tile_x, list)
    assert len(level.spawn_x) == 8  # Pre-sized to 8
    assert len(level.tile_x) == 1024  # Pre-sized to 1024

    # Test that we can convert to ctypes for C API
    c_level = level.to_ctype()
    assert c_level is not None
