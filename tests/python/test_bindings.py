#!/usr/bin/env python3
"""
Test Python bindings for Madrona Escape Room using pytest.
"""

import numpy as np
import pytest
import torch

import madrona_escape_room

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def test_cpu_import():
    """Test basic CPU import"""
    from madrona_escape_room import SimManager

    assert SimManager is not None


def test_gpu_import():
    """Test GPU availability check"""
    from madrona_escape_room import SimManager

    # Just check that we can import and check CUDA
    assert SimManager is not None


def test_cpu_manager_exists(cpu_manager):
    """Test that CPU manager was created successfully"""
    assert cpu_manager is not None


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_manager_exists(gpu_manager):
    """Test that GPU manager was created successfully"""
    assert gpu_manager is not None


def test_tensor_shapes(cpu_manager):
    """Test tensor shapes and dtypes"""
    mgr = cpu_manager

    # Test action tensor
    actions = mgr.action_tensor().to_torch()
    assert actions.shape == (4, 3), f"Expected shape (4, 3), got {actions.shape}"
    assert actions.dtype == torch.int32

    # Test reward tensor - single-agent environment has shape [num_worlds, 1]
    rewards = mgr.reward_tensor().to_torch()
    assert rewards.shape == (4, 1), f"Expected shape (4, 1), got {rewards.shape}"
    assert rewards.dtype == torch.float32

    # Test done tensor - single-agent environment has shape [num_worlds, 1]
    dones = mgr.done_tensor().to_torch()
    assert dones.shape == (4, 1), f"Expected shape (4, 1), got {dones.shape}"
    assert dones.dtype == torch.int32

    # Test observation tensors
    self_obs = mgr.self_observation_tensor().to_torch()
    assert len(self_obs.shape) == 3  # [worlds, agents, features]

    # Test lidar tensor shape
    lidar = mgr.lidar_tensor().to_torch()
    assert lidar.shape == (4, 1, 256)  # 256-sample buffer (default config uses 128 active samples)

    # Room entity observations removed - no longer tracking room entities

    # Note: door_observation_tensor is not available in the current bindings
    # door_obs = mgr.door_observation_tensor().to_torch()
    # assert len(door_obs.shape) == 4  # [worlds, agents, doors, features]

    steps = mgr.steps_taken_tensor().to_torch()
    assert steps.shape == (4, 1, 1)


def test_simulation_step(cpu_manager):
    """Test running simulation steps"""
    mgr = cpu_manager

    # Get action tensor and set some actions
    actions = mgr.action_tensor().to_torch()
    actions[:] = 0  # Zero all actions
    actions[:, 0] = 1  # Set forward movement for all worlds

    # Get initial observations
    _rewards_before = mgr.reward_tensor().to_torch().clone()
    self_obs_before = mgr.self_observation_tensor().to_torch()[:, :, :3].clone()  # Position

    # Step simulation
    mgr.step()

    # Check if observations changed
    _rewards_after = mgr.reward_tensor().to_torch()
    self_obs_after = mgr.self_observation_tensor().to_torch()[:, :, :3]

    # At least some positions should have changed
    position_change = torch.abs(self_obs_after - self_obs_before).max().item()
    assert position_change > 0.0009, "Agents should have moved"


def test_reset_functionality(cpu_manager):
    """Test reset functionality via reset_tensor"""
    mgr = cpu_manager

    # First ensure all worlds are in a fresh state
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1  # Reset all worlds
    mgr.step()

    # Run some steps
    actions = mgr.action_tensor().to_torch()
    actions[:] = 0
    actions[:, 0] = 1  # Move forward

    for _ in range(10):
        mgr.step()

    # Get current steps for comparison
    _steps_before = mgr.steps_taken_tensor().to_torch().clone()

    # Reset only world 0
    reset_tensor[:] = 0  # Clear all reset flags
    reset_tensor[0] = 1  # Set reset flag for world 0
    mgr.step()

    # Check if episode was reset (steps taken should be back to 0)
    steps_after = mgr.steps_taken_tensor().to_torch()
    # World 0 should have 0 steps taken (just reset), while others have more
    assert steps_after[0, 0, 0].item() == 0, "World 0 should be reset to 0 steps taken"
    assert steps_after[1, 0, 0].item() > 0, "World 1 should have taken more steps"


def test_tensor_memory_layout(cpu_manager):
    """Test that tensors have correct memory layout for PyTorch"""
    mgr = cpu_manager

    tensors_to_check = [
        ("action", mgr.action_tensor().to_torch()),
        ("reward", mgr.reward_tensor().to_torch()),
        ("done", mgr.done_tensor().to_torch()),
        ("self_obs", mgr.self_observation_tensor().to_torch()),
    ]

    for name, tensor in tensors_to_check:
        assert tensor.is_contiguous(), f"{name} tensor is not contiguous"
        assert tensor.device.type == "cpu", f"{name} tensor is not on CPU"


def test_multiple_steps(cpu_manager):
    """Test running multiple simulation steps"""
    mgr = cpu_manager
    actions = mgr.action_tensor().to_torch()

    # Run 100 steps with random actions
    for i in range(100):
        actions[:, 0] = torch.randint(0, 3, (4,))  # Movement
        actions[:, 1] = torch.randint(0, 8, (4,))  # Angle
        actions[:, 2] = torch.randint(0, 5, (4,))  # Rotation
        mgr.step()

    # Check that simulation is still running
    dones = mgr.done_tensor().to_torch()
    # At least some episodes should still be running
    assert not dones.all(), "Some episodes should still be running"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_tensors(gpu_manager):
    """Test GPU tensor functionality"""
    mgr = gpu_manager

    # Get tensors
    actions = mgr.action_tensor().to_torch()
    rewards = mgr.reward_tensor().to_torch()

    # GPU tensors should be on CUDA device
    assert actions.device.type == "cuda"
    assert rewards.device.type == "cuda"

    # Should be able to run steps
    actions[:] = 0
    mgr.step()

    # Check rewards changed
    new_rewards = mgr.reward_tensor().to_torch()
    assert new_rewards.device.type == "cuda"


def test_observation_values(cpu_manager):
    """Test that observations have reasonable values"""
    mgr = cpu_manager

    # Step once to get initial observations
    mgr.step()

    # Check self observations
    self_obs = mgr.self_observation_tensor().to_torch()

    # Position should be reasonable (within world bounds)
    positions = self_obs[:, :, :3]
    assert positions.abs().max() < 100, "Positions should be within reasonable bounds"

    # Steps taken should be non-negative and within episode bounds
    steps = mgr.steps_taken_tensor().to_torch()
    assert steps.min() >= 0, "Steps taken should be non-negative"
    assert steps.max() < 200, "Steps taken should be less than episode length"

    # Lidar should have normalized values
    lidar = mgr.lidar_tensor().to_torch()
    assert lidar.min() >= 0, "Lidar values should be non-negative"
    assert lidar.max() <= 1, "Lidar values should be normalized"


# Additional test to verify state persistence across tests
def test_state_persistence(cpu_manager):
    """Test that manager maintains state across test functions"""
    mgr = cpu_manager

    # This test runs after others, so the manager should have been used
    # Check that we can still access tensors and run steps
    actions = mgr.action_tensor().to_torch()
    actions[:] = 0

    # Run a step - this should work even after previous tests
    mgr.step()

    # Verify tensors are still accessible
    rewards = mgr.reward_tensor().to_torch()
    assert rewards is not None
    assert rewards.shape == (4, 1)


@pytest.mark.xfail(reason="Progress tensor deprecated with target-based reward system")
def test_progress_tensor(cpu_manager):
    """Test that progress tensor is accessible and has correct shape."""
    mgr = cpu_manager

    # Get progress tensor
    progress = mgr.progress_tensor().to_torch()

    # Check shape
    expected_shape = (4, 1, 1)  # 4 worlds, 1 agent, 1 value
    assert (
        progress.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {progress.shape}"

    # Check initial values are reasonable (should be near spawn position)
    # Progress tracks maxY, so it starts at spawn Y (which can be negative)
    # For default level, spawn Y is -17.0
    assert (progress >= -20).all(), "Initial progress should be reasonable (near spawn position)"
    assert (progress < 40).all(), "Initial progress should be less than world length (40)"

    # Run some steps and verify progress updates
    initial_progress = progress.clone()

    # Move agents forward
    actions = mgr.action_tensor().to_torch()
    actions[:, 0] = 1  # Move forward
    actions[:, 1] = 0  # Forward angle
    actions[:, 2] = 2  # No rotation

    for _ in range(50):
        mgr.step()

    final_progress = mgr.progress_tensor().to_torch()

    # At least some agents should have made progress
    assert (final_progress > initial_progress).any(), "Some agents should have made progress"

    # Progress should never decrease (it tracks maximum Y)
    assert (final_progress >= initial_progress).all(), "Progress should never decrease"


def test_deterministic_actions(cpu_manager):
    """Test that fixed actions produce consistent results"""
    mgr = cpu_manager

    # Reset all worlds
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()

    actions = mgr.action_tensor().to_torch()
    initial_obs = mgr.self_observation_tensor().to_torch().clone()

    # Run specific action sequence with repetitions and pauses
    action_sequence = [
        (1, 0, 2),  # Move forward
        (1, 2, 2),  # Move right
        (1, 4, 2),  # Move back
        (1, 6, 2),  # Move left
        (0, 0, 2),  # Stop (no movement)
        (0, 0, 4),  # Rotate right only
        (0, 0, 0),  # Rotate left only
        (2, 0, 2),  # Move forward faster
    ]

    positions_history = []

    for move_amt, move_angle, rotate in action_sequence:
        # Repeat each movement 10 times
        for _ in range(10):
            actions[:, 0] = move_amt
            actions[:, 1] = move_angle
            actions[:, 2] = rotate

            mgr.step()

            pos = mgr.self_observation_tensor().to_torch()[:, :, :3].clone()
            positions_history.append(pos)

        # Pause for 5 steps between different movements
        for _ in range(5):
            actions[:, 0] = 0  # No movement
            actions[:, 1] = 0  # No angle
            actions[:, 2] = 2  # No rotation (center bucket)

            mgr.step()

            pos = mgr.self_observation_tensor().to_torch()[:, :, :3].clone()
            positions_history.append(pos)

    # Verify movement occurred
    final_obs = mgr.self_observation_tensor().to_torch()
    position_change = (final_obs[:, :, :3] - initial_obs[:, :, :3]).abs().sum()
    assert position_change > 0.05, "Agents should have moved from initial positions"

    # Verify different actions produced different results
    all_same = True
    for i in range(1, len(positions_history)):
        if not torch.allclose(positions_history[i], positions_history[i - 1]):
            all_same = False
            break
    assert not all_same, "Different actions should produce different positions"
