"""
Test SimInterface adapter that converts SimManager to training format
"""

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pytest
import torch
from madrona_escape_room_learn.sim_interface_adapter import (
    CompassIndex,
    DepthIndex,
    ObsIndex,
    SelfObsIndex,
)

import madrona_escape_room


@dataclass(frozen=True)
class SimInterface:
    """Training interface expected by PPO code"""

    step: Callable
    obs: List[torch.Tensor]
    actions: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor


def process_observations(
    self_obs: torch.Tensor, compass: torch.Tensor, depth: torch.Tensor
) -> torch.Tensor:
    """
    Extract only progress (maxY) from self_obs and combine with compass and depth.

    Args:
        self_obs: [batch, agents, 5] where 5 = [x, y, z, maxY, rotation]
        compass: [batch, agents, 128] one-hot compass encoding
        depth: [batch, agents, height, width] depth image

    Returns:
        Combined observation tensor [batch, agents, obs_dim] where:
        obs_dim = 1 (progress) + 128 (compass) + height*width (flattened depth)
    """
    batch_size = self_obs.shape[0]
    num_agents = self_obs.shape[1]

    # Extract progress (maxY) from self_obs using constants
    progress = self_obs[
        :, :, SelfObsIndex.PROGRESS : SelfObsIndex.PROGRESS + 1
    ]  # Shape: [batch, agents, 1]

    # Flatten depth image for each agent
    depth_flat = depth.view(batch_size, num_agents, -1)  # Shape: [batch, agents, height*width]

    # Concatenate: progress + compass + flattened depth
    combined_obs = torch.cat([progress, compass, depth_flat], dim=-1)

    return combined_obs


def create_sim_interface_adapter(
    manager: madrona_escape_room.SimManager, enable_depth: bool = True
) -> SimInterface:
    """
    Create a SimInterface adapter from a SimManager.

    Args:
        manager: The Madrona SimManager instance
        enable_depth: Whether to include depth sensor in observations

    Returns:
        SimInterface compatible with training code
    """
    # Get tensor references
    action_tensor = manager.action_tensor().to_torch()
    reward_tensor = manager.reward_tensor().to_torch()
    done_tensor = manager.done_tensor().to_torch()
    self_obs_tensor = manager.self_observation_tensor().to_torch()
    compass_tensor = manager.compass_tensor().to_torch()

    if enable_depth:
        depth_tensor = manager.depth_tensor().to_torch()
        obs_tensors = [self_obs_tensor, compass_tensor, depth_tensor]
    else:
        obs_tensors = [self_obs_tensor, compass_tensor]

    return SimInterface(
        step=manager.step,
        obs=obs_tensors,
        actions=action_tensor,
        dones=done_tensor,
        rewards=reward_tensor,
    )


@pytest.mark.lidar_128  # Use 128-beam horizontal lidar preset
def test_sim_interface_adapter_creation(cpu_manager):
    """Test that we can create a SimInterface adapter from SimManager"""
    # Use the cpu_manager fixture which properly handles lidar_128 configuration
    manager = cpu_manager

    # Create the interface adapter directly from the manager
    sim_interface = create_sim_interface_adapter(manager, enable_depth=True)

    # Verify interface structure
    assert hasattr(sim_interface, "step")
    assert hasattr(sim_interface, "obs")
    assert hasattr(sim_interface, "actions")
    assert hasattr(sim_interface, "dones")
    assert hasattr(sim_interface, "rewards")

    # Check tensor shapes - print actual shapes first
    print(f"Actions shape: {sim_interface.actions.shape}")
    print(f"Rewards shape: {sim_interface.rewards.shape}")
    print(f"Dones shape: {sim_interface.dones.shape}")

    self_obs = sim_interface.obs[sim_interface.ObsIndex.SELF_OBS]
    compass = sim_interface.obs[sim_interface.ObsIndex.COMPASS]
    depth = sim_interface.obs[sim_interface.ObsIndex.DEPTH]
    print(f"Self obs shape: {self_obs.shape}")
    print(f"Compass shape: {compass.shape}")
    print(f"Depth shape: {depth.shape}")

    # Update assertions based on actual shapes from fixture
    assert len(sim_interface.obs) == 3  # self_obs, compass, depth

    # Check observation tensor shapes using constants
    assert self_obs.shape == (4, 1, 5)  # [worlds, agents, 5] - position + progress + rotation
    assert compass.shape == (
        4,
        1,
        sim_interface.CompassIndex.bucket_count(),
    )  # [worlds, agents, 128] - one-hot compass
    assert depth.shape == (
        4,
        1,
        1,
        sim_interface.DepthIndex.beam_count(),
        1,
    )  # [worlds, agents, height, width, channels] - horizontal lidar

    # Action tensor could be [worlds, action_dim] or [worlds, agents, action_dim]
    assert len(sim_interface.actions.shape) in [
        2,
        3,
    ], f"Unexpected action tensor shape: {sim_interface.actions.shape}"
    if len(sim_interface.actions.shape) == 2:
        assert sim_interface.actions.shape == (4, 3)  # [worlds, action_dim]
    else:
        assert sim_interface.actions.shape == (4, 1, 3)  # [worlds, agents, action_dim]

    # Rewards and dones should have agent dimension
    assert sim_interface.rewards.shape == (4, 1, 1)  # [worlds, agents, 1]
    assert sim_interface.dones.shape == (4, 1, 1)  # [worlds, agents, 1]


def test_process_observations():
    """Test the observation preprocessing function"""
    batch_size = 2
    num_agents = 1

    # Create mock tensors
    self_obs = torch.randn(batch_size, num_agents, 5)
    self_obs[:, :, SelfObsIndex.PROGRESS] = torch.tensor([[0.25], [0.75]])  # Set progress values

    compass = torch.zeros(batch_size, num_agents, CompassIndex.bucket_count())
    compass[0, 0, CompassIndex.CENTER] = 1.0  # One-hot at center for first batch
    compass[1, 0, 32] = 1.0  # One-hot at index 32 for second batch

    depth_height, depth_width = 64, 64
    depth = torch.randn(batch_size, num_agents, depth_height, depth_width)

    # Process observations
    processed = process_observations(self_obs, compass, depth)

    # Check shape: 1 (progress) + 128 (compass) + 64*64 (depth) = 4225
    expected_dim = 1 + CompassIndex.bucket_count() + (depth_height * depth_width)
    assert processed.shape == (batch_size, num_agents, expected_dim)

    # Check that progress values are preserved
    assert torch.allclose(processed[0, 0, 0], torch.tensor(0.25))
    assert torch.allclose(processed[1, 0, 0], torch.tensor(0.75))

    # Check that compass values are preserved (compass starts at index 1 after progress)
    compass_start_idx = 1
    compass_end_idx = compass_start_idx + CompassIndex.bucket_count()
    assert torch.allclose(processed[0, 0, compass_start_idx:compass_end_idx], compass[0, 0])
    assert torch.allclose(processed[1, 0, compass_start_idx:compass_end_idx], compass[1, 0])

    # Check that depth values are flattened correctly
    depth_flat = depth.view(batch_size, num_agents, -1)
    depth_start_idx = compass_end_idx
    assert torch.allclose(processed[:, :, depth_start_idx:], depth_flat)


@pytest.mark.lidar_128  # Use 128-beam horizontal lidar preset
def test_sim_interface_step_integration(cpu_manager):
    """Test that the SimInterface adapter works with actual simulation stepping"""
    # Use the cpu_manager fixture which properly handles lidar_128 configuration
    manager = cpu_manager

    # Create the interface adapter directly from the manager
    sim_interface = create_sim_interface_adapter(manager, enable_depth=True)

    # Get initial observations using constants
    initial_self_obs = sim_interface.obs[sim_interface.ObsIndex.SELF_OBS].clone()

    # Set some actions - move forward at medium speed
    # Check action tensor shape first and set accordingly
    print(f"Action tensor shape: {sim_interface.actions.shape}")
    if len(sim_interface.actions.shape) == 2:
        # Shape is [worlds, action_dim]
        sim_interface.actions[:, 0] = 2  # moveAmount = MEDIUM
        sim_interface.actions[:, 1] = 0  # moveAngle = FORWARD
        sim_interface.actions[:, 2] = 2  # rotate = NONE
    else:
        # Shape might be [worlds, agents, action_dim]
        sim_interface.actions[:, :, 0] = 2  # moveAmount = MEDIUM
        sim_interface.actions[:, :, 1] = 0  # moveAngle = FORWARD
        sim_interface.actions[:, :, 2] = 2  # rotate = NONE

    # Step simulation
    sim_interface.step()

    # Check that observations changed using constants
    new_self_obs = sim_interface.obs[sim_interface.ObsIndex.SELF_OBS]

    # Position should have changed (agents moved forward)
    initial_position = initial_self_obs[
        :, :, sim_interface.SelfObsIndex.X : sim_interface.SelfObsIndex.Z + 1
    ]
    new_position = new_self_obs[
        :, :, sim_interface.SelfObsIndex.X : sim_interface.SelfObsIndex.Z + 1
    ]
    position_changed = not torch.allclose(initial_position, new_position)
    assert position_changed, "Agent positions should have changed after forward movement"

    # Progress (maxY) should have increased or stayed the same
    initial_progress = initial_self_obs[:, :, sim_interface.SelfObsIndex.PROGRESS]
    new_progress = new_self_obs[:, :, sim_interface.SelfObsIndex.PROGRESS]
    progress_increased = torch.all(new_progress >= initial_progress)
    assert progress_increased, "Progress should not decrease"


@pytest.mark.lidar_128  # Use 128-beam horizontal lidar preset
def test_compass_tensor_properties(cpu_manager):
    """Test that compass tensor has expected one-hot properties"""
    # Use the cpu_manager fixture which properly handles lidar_128 configuration
    manager = cpu_manager

    # Create the interface adapter directly from the manager
    sim_interface = create_sim_interface_adapter(manager, enable_depth=True)

    compass_tensor = sim_interface.obs[sim_interface.ObsIndex.COMPASS]  # compass tensor

    # Check shape - cpu_manager fixture uses 4 worlds
    assert compass_tensor.shape == (4, 1, sim_interface.CompassIndex.bucket_count())

    # Check one-hot property for each world
    compass_np = compass_tensor.numpy()
    for world_idx in range(4):
        for agent_idx in range(1):
            compass_slice = compass_np[world_idx, agent_idx, :]

            # Should have exactly one 1.0 and rest should be 0.0
            ones_count = np.sum(compass_slice == 1.0)
            zeros_count = np.sum(compass_slice == 0.0)

            assert (
                ones_count == 1
            ), f"World {world_idx}, Agent {agent_idx}: Expected exactly 1 one, got {ones_count}"
            assert (
                zeros_count == 127
            ), f"World {world_idx}, Agent {agent_idx}: Expected 127 zeros, got {zeros_count}"
            assert (
                np.sum(compass_slice) == 1.0
            ), f"World {world_idx}, Agent {agent_idx}: Sum should be 1.0"


@pytest.mark.lidar_128  # Use 128-beam horizontal lidar preset
def test_zero_copy_and_simulation_state_changes(cpu_manager):
    """
    Verify zero-copy tensor access and that simulation state changes are visible
    in the original tensor references after stepping.
    """
    manager = cpu_manager

    # Get original manager tensor references BEFORE creating interface
    original_action_tensor = manager.action_tensor().to_torch()
    original_reward_tensor = manager.reward_tensor().to_torch()
    original_done_tensor = manager.done_tensor().to_torch()
    original_self_obs_tensor = manager.self_observation_tensor().to_torch()
    original_compass_tensor = manager.compass_tensor().to_torch()
    original_depth_tensor = manager.depth_tensor().to_torch()

    # Record initial state from original tensors using constants
    initial_position = original_self_obs_tensor[
        0, 0, SelfObsIndex.X : SelfObsIndex.Z + 1
    ].clone()  # World 0, Agent 0, position [x,y,z]
    initial_progress = original_self_obs_tensor[
        0, 0, SelfObsIndex.PROGRESS
    ].clone()  # World 0, Agent 0, progress (maxY)
    initial_rotation = original_self_obs_tensor[
        0, 0, SelfObsIndex.ROTATION
    ].clone()  # World 0, Agent 0, rotation
    initial_compass = original_compass_tensor[0, 0, :].clone()  # World 0, Agent 0, compass
    initial_depth = original_depth_tensor[0, 0, 0, :, 0].clone()  # World 0, Agent 0, depth readings

    # Create interface adapter
    sim_interface = create_sim_interface_adapter(manager, enable_depth=True)

    # === ZERO-COPY VERIFICATION ===
    # Verify that interface tensors are the SAME memory objects as original tensors
    assert (
        sim_interface.actions.data_ptr() == original_action_tensor.data_ptr()
    ), "Action tensor should be zero-copy reference to original"
    assert (
        sim_interface.rewards.data_ptr() == original_reward_tensor.data_ptr()
    ), "Reward tensor should be zero-copy reference to original"
    assert (
        sim_interface.dones.data_ptr() == original_done_tensor.data_ptr()
    ), "Done tensor should be zero-copy reference to original"
    assert (
        sim_interface.obs[sim_interface.ObsIndex.SELF_OBS].data_ptr()
        == original_self_obs_tensor.data_ptr()
    ), "Self observation tensor should be zero-copy reference to original"
    assert (
        sim_interface.obs[sim_interface.ObsIndex.COMPASS].data_ptr()
        == original_compass_tensor.data_ptr()
    ), "Compass tensor should be zero-copy reference to original"
    assert (
        sim_interface.obs[sim_interface.ObsIndex.DEPTH].data_ptr()
        == original_depth_tensor.data_ptr()
    ), "Depth tensor should be zero-copy reference to original"

    print("✅ Zero-copy verification passed - all tensors share memory with originals")

    # === SIMULATION STATE CHANGE TEST ===
    # Set actions for forward movement with rotation using the interface
    # Actions: [moveAmount=MEDIUM, moveAngle=FORWARD, rotate=SLOW_RIGHT]
    sim_interface.actions[0, 0] = 2  # moveAmount = MEDIUM
    sim_interface.actions[0, 1] = 0  # moveAngle = FORWARD
    sim_interface.actions[0, 2] = 3  # rotate = SLOW_RIGHT

    # Verify action changes are immediately visible in original tensor (zero-copy proof)
    assert original_action_tensor[0, 0] == 2, "Action change should be visible in original tensor"
    assert original_action_tensor[0, 1] == 0, "Action change should be visible in original tensor"
    assert original_action_tensor[0, 2] == 3, "Action change should be visible in original tensor"

    print("✅ Action zero-copy verified - changes visible in original tensor")

    # Step simulation
    manager.step()

    # === VERIFY STATE CHANGES IN ORIGINAL TENSORS ===
    # After stepping, check that changes are visible in the original tensor references
    # (We don't create new tensors - we read from the same ones we got before)

    # 1. Position should have changed (forward movement)
    new_position = original_self_obs_tensor[0, 0, SelfObsIndex.X : SelfObsIndex.Z + 1]
    position_changed = not torch.allclose(initial_position, new_position, atol=1e-6)
    assert (
        position_changed
    ), f"Position should have changed. Initial: {initial_position}, New: {new_position}"

    # 2. Progress (maxY) should have increased (forward movement increases Y)
    new_progress = original_self_obs_tensor[0, 0, SelfObsIndex.PROGRESS]
    progress_increased = new_progress >= initial_progress
    assert (
        progress_increased
    ), f"Progress should have increased. Initial: {initial_progress}, New: {new_progress}"

    # 3. Rotation should have changed (SLOW_RIGHT rotation)
    new_rotation = original_self_obs_tensor[0, 0, SelfObsIndex.ROTATION]
    rotation_changed = not torch.allclose(initial_rotation, new_rotation, atol=1e-6)
    assert (
        rotation_changed
    ), f"Rotation should have changed. Initial: {initial_rotation}, New: {new_rotation}"

    # 4. Compass should have changed (rotation affects compass)
    new_compass = original_compass_tensor[0, 0, :]
    compass_changed = not torch.allclose(initial_compass, new_compass, atol=1e-6)
    assert compass_changed, "Compass should have changed due to rotation"

    # 5. Depth measurements should have changed (position/rotation change affects what agent sees)
    new_depth = original_depth_tensor[0, 0, 0, :, 0]
    depth_changed = not torch.allclose(initial_depth, new_depth, atol=1e-6)
    assert depth_changed, "Depth measurements should have changed due to movement/rotation"

    print("✅ Simulation state changes verified:")
    print(f"  Position changed: {torch.norm(new_position - initial_position):.4f} units")
    print(f"  Progress increased: {new_progress - initial_progress:.4f}")
    print(f"  Rotation changed: {new_rotation - initial_rotation:.4f} radians")
    print(f"  Compass changed: {torch.sum(torch.abs(new_compass - initial_compass)):.4f}")
    print(f"  Depth changed: {torch.sum(torch.abs(new_depth - initial_depth)):.4f}")

    # === VERIFY INTERFACE REFLECTS SAME CHANGES ===
    # The interface tensors should show the exact same values as original tensors
    assert torch.allclose(
        sim_interface.obs[sim_interface.ObsIndex.SELF_OBS][
            0, 0, SelfObsIndex.X : SelfObsIndex.Z + 1
        ],
        new_position,
    ), "Interface should show same position as original tensor"
    assert torch.allclose(
        sim_interface.obs[sim_interface.ObsIndex.SELF_OBS][
            0, 0, sim_interface.SelfObsIndex.PROGRESS
        ],
        new_progress,
    ), "Interface should show same progress as original tensor"
    assert torch.allclose(
        sim_interface.obs[sim_interface.ObsIndex.SELF_OBS][
            0, 0, sim_interface.SelfObsIndex.ROTATION
        ],
        new_rotation,
    ), "Interface should show same rotation as original tensor"
    assert torch.allclose(
        sim_interface.obs[sim_interface.ObsIndex.COMPASS][0, 0, :], new_compass
    ), "Interface should show same compass as original tensor"
    assert torch.allclose(
        sim_interface.obs[sim_interface.ObsIndex.DEPTH][0, 0, 0, :, 0], new_depth
    ), "Interface should show same depth as original tensor"

    print("✅ Interface consistency verified - shows same values as original tensors")
    print("✅ Zero-copy and simulation state test PASSED")


if __name__ == "__main__":
    # Run basic test
    test_sim_interface_adapter_creation()
    test_process_observations()
    test_sim_interface_step_integration()
    test_compass_tensor_properties()
    print("All tests passed!")
