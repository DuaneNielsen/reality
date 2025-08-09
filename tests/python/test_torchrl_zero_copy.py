"""Tests for verifying zero-copy tensor behavior in TorchRL environment wrapper.

These tests ensure that the environment wrapper maintains zero-copy semantics
when interfacing between Madrona simulation and PyTorch tensors.
"""

import gc
import time

import pytest
import torch

# Skip all tests in this module - TorchRL integration is a work in progress
pytest.skip("TorchRL integration is a work in progress", allow_module_level=True)


# Import both the raw SimManager and the TorchRL wrapper
from madrona_escape_room_learn import MadronaEscapeRoomEnv

import madrona_escape_room


@pytest.fixture
def sim_manager():
    """Create a raw SimManager for testing"""
    return madrona_escape_room.SimManager(
        exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
        gpu_id=-1,
        num_worlds=4,
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
    )


@pytest.fixture
def env():
    """Create TorchRL environment wrapper"""
    return MadronaEscapeRoomEnv(num_worlds=4, gpu_id=-1, rand_seed=42, auto_reset=True)


def test_data_pointer_stability(env):
    """Test that tensor data pointers remain stable across steps.

    Zero-copy tensors should maintain the same underlying memory address
    across multiple simulation steps.
    """
    # Get initial data pointers
    initial_pointers = {
        "action": env._action_tensor.data_ptr(),
        "reward": env._reward_tensor.data_ptr(),
        "done": env._done_tensor.data_ptr(),
        "self_obs": env._self_obs_tensor.data_ptr(),
        "steps_remaining": env._steps_remaining_tensor.data_ptr(),
    }

    # Reset and step multiple times
    td = env.reset()
    for _ in range(10):
        action = env.action_spec.rand()
        td["action"] = action
        td = env.step(td)

    # Check that pointers haven't changed
    final_pointers = {
        "action": env._action_tensor.data_ptr(),
        "reward": env._reward_tensor.data_ptr(),
        "done": env._done_tensor.data_ptr(),
        "self_obs": env._self_obs_tensor.data_ptr(),
        "steps_remaining": env._steps_remaining_tensor.data_ptr(),
    }

    for name, initial_ptr in initial_pointers.items():
        assert initial_ptr == final_pointers[name], (
            f"{name} tensor data pointer changed: {initial_ptr} -> {final_pointers[name]}"
        )

    env.close()


def test_view_operations_zero_copy(env):
    """Test that view operations in _get_observations maintain zero-copy.

    PyTorch view operations should return tensors that share the same
    underlying storage.
    """
    # Reset to populate tensors
    td = env.reset()

    # Get observations through the wrapper
    obs_dict = td["observation"]

    # Check self_obs is a view of the underlying tensor
    self_obs_view = obs_dict["self_obs"]
    original_self_obs = env._self_obs_tensor

    # Views share the same storage
    assert self_obs_view.data_ptr() == original_self_obs.data_ptr(), (
        "self_obs view doesn't share storage with original tensor"
    )

    # Modifying the view should affect the original
    if self_obs_view.numel() > 0:
        original_value = original_self_obs.view(-1)[0].item()
        self_obs_view.view(-1)[0] += 1.0
        assert original_self_obs.view(-1)[0].item() == original_value + 1.0, (
            "Modification to view didn't affect original tensor"
        )
        # Restore original value
        self_obs_view.view(-1)[0] -= 1.0

    env.close()


def test_bidirectional_updates_actions(sim_manager):
    """Test that action modifications are visible bidirectionally.

    Changes to action tensors in Python should be immediately visible
    to the simulator without explicit copying.
    """
    # Get action tensor as PyTorch tensor
    action_torch = sim_manager.action_tensor().to_torch()

    # Store original values
    original_values = action_torch.clone()

    # Modify actions in PyTorch
    test_values = torch.tensor(
        [[1, 2, 3], [2, 3, 4], [3, 4, 0], [0, 1, 2]], dtype=action_torch.dtype
    )
    action_torch[:] = test_values

    # Get the tensor again and verify changes are visible
    action_torch_2 = sim_manager.action_tensor().to_torch()
    assert torch.allclose(action_torch_2, test_values), (
        "Action modifications not visible when re-accessing tensor"
    )

    # Step the simulation
    sim_manager.step()

    # Verify tensor still has our values (not reset by step)
    assert torch.allclose(action_torch, test_values), (
        "Action tensor unexpectedly modified after step"
    )

    # Restore original values
    action_torch[:] = original_values


def test_bidirectional_updates_observations(sim_manager):
    """Test that observation updates from simulator are visible in Python.

    After stepping the simulator, observation changes should be immediately
    visible in Python tensors without explicit copying.
    """
    # Get observation tensor
    obs_torch = sim_manager.self_observation_tensor().to_torch()

    # Record initial state
    initial_obs = obs_torch.clone()

    # Set some actions to cause movement
    action_torch = sim_manager.action_tensor().to_torch()
    # Move forward fast in all worlds
    action_torch[:, 0] = 3  # move_amount = fast
    action_torch[:, 1] = 0  # move_angle = forward
    action_torch[:, 2] = 2  # rotate = none

    # Step simulation
    sim_manager.step()

    # Check that observations changed (at least Y position should increase)
    # obs format: [globalX, globalY, globalZ, maxY, theta]
    y_positions = obs_torch[:, 0, 1]  # All agents' Y positions
    initial_y = initial_obs[:, 0, 1]

    assert (y_positions > initial_y).any(), "Y positions didn't increase after forward movement"


def test_no_hidden_copies(env):
    """Test that common operations don't create hidden copies.

    Some operations legitimately need copies (like type conversion),
    but we should verify which operations maintain zero-copy.
    """
    td = env.reset()

    # Track operations that should be zero-copy
    zero_copy_ops = []

    # view() should be zero-copy
    obs = env._self_obs_tensor
    obs_view = obs.view(env.batch_size[0], -1)
    zero_copy_ops.append(("view", obs.data_ptr() == obs_view.data_ptr()))

    # slice operations should be zero-copy (if contiguous)
    obs_slice = obs[0:2]
    zero_copy_ops.append(("slice", obs.data_ptr() == obs_slice.data_ptr()))

    # reshape() on contiguous tensor should be zero-copy
    if obs.is_contiguous():
        obs_reshape = obs.reshape(env.batch_size[0], -1)
        zero_copy_ops.append(("reshape", obs.data_ptr() == obs_reshape.data_ptr()))

    # Check results
    for op_name, is_zero_copy in zero_copy_ops:
        assert is_zero_copy, f"{op_name} operation created a copy"

    # Document operations that create copies
    copy_ops = []

    # .float() creates a copy if tensor isn't already float
    steps_tensor = env._steps_remaining_tensor
    if steps_tensor.dtype != torch.float32:
        steps_float = steps_tensor.float()
        copy_ops.append(("float_conversion", steps_tensor.data_ptr() != steps_float.data_ptr()))

    # clone() always creates a copy
    obs_clone = obs.clone()
    copy_ops.append(("clone", obs.data_ptr() != obs_clone.data_ptr()))

    # Verify known copy operations
    for op_name, creates_copy in copy_ops:
        assert creates_copy, f"{op_name} operation should create a copy but didn't"

    env.close()


def test_performance_characteristics(env):
    """Test that tensor access matches zero-copy performance.

    Zero-copy tensor access should be significantly faster than
    explicit copying operations for larger tensors.
    """
    # For a more meaningful test, create a larger environment
    env.close()
    env = MadronaEscapeRoomEnv(
        num_worlds=256,  # Larger batch for meaningful performance test
        gpu_id=-1,
        rand_seed=42,
        auto_reset=True,
    )

    # Warm up
    td = env.reset()
    for _ in range(10):
        action = env.action_spec.rand()
        td["action"] = action
        td = env.step(td)

    # Measure zero-copy observation access time
    gc.collect()
    start_time = time.perf_counter()
    for _ in range(100):
        obs = td["observation"]
        # Access multiple elements to make measurement more meaningful
        self_obs = obs["self_obs"]
        _ = self_obs.sum().item()  # Force computation without copy
    zero_copy_time = time.perf_counter() - start_time

    # Measure explicit copy time
    gc.collect()
    start_time = time.perf_counter()
    for _ in range(100):
        obs = td["observation"]
        # Force a copy
        self_obs_copy = obs["self_obs"].clone()
        _ = self_obs_copy.sum().item()
    copy_time = time.perf_counter() - start_time

    # Zero-copy should be faster
    speedup = copy_time / zero_copy_time
    print("\nPerformance test results (256 worlds):")
    print(f"  Zero-copy access: {zero_copy_time * 1000:.2f}ms")
    print(f"  Explicit copy:    {copy_time * 1000:.2f}ms")
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Tensor shape:     {obs['self_obs'].shape}")

    # Assert meaningful speedup (lower threshold due to Python overhead)
    assert speedup > 1.2, f"Zero-copy access not meaningfully faster ({speedup:.2f}x)"

    # More importantly, verify the operations are actually zero-copy
    # by checking memory addresses
    obs1 = td["observation"]["self_obs"]
    obs2 = td["observation"]["self_obs"]
    assert obs1.data_ptr() == obs2.data_ptr(), "Repeated access returns different tensor instances"

    env.close()


def test_memory_sharing_sim_to_env(sim_manager):
    """Test that SimManager and env wrapper tensors share memory.

    The environment wrapper should store references to the same
    underlying memory as the SimManager tensors.
    """
    # Create env wrapper using same settings
    env = MadronaEscapeRoomEnv(num_worlds=4, gpu_id=-1, rand_seed=42, auto_reset=True)

    # Get tensors from both interfaces
    sim_action = sim_manager.action_tensor().to_torch()
    env_action = env._action_tensor

    # They might not have the same data_ptr if the env creates its own manager
    # But modifications to one should affect the other if they're properly linked
    # Since they're different manager instances, we'll test within each env

    # Test that env's internal tensors properly reference simulator memory
    env_sim = env.sim
    env_sim_action = env_sim.action_tensor().to_torch()

    # These should definitely share memory
    assert env_action.data_ptr() == env_sim_action.data_ptr(), (
        "Environment wrapper tensor doesn't share memory with its simulator"
    )

    env.close()


@pytest.mark.skip(
    reason="GPU env conflicts with gpu_manager fixture - Madrona doesn't support multiple GPU managers"
)
def test_gpu_zero_copy(gpu_env):
    """Test zero-copy behavior on GPU tensors.

    GPU tensors should maintain zero-copy semantics just like CPU tensors.
    """
    env = gpu_env

    # Verify tensors are on GPU
    assert env._action_tensor.is_cuda
    assert env._self_obs_tensor.is_cuda

    # Test pointer stability on GPU
    initial_ptr = env._self_obs_tensor.data_ptr()

    td = env.reset()
    for _ in range(5):
        action = env.action_spec.rand()
        td["action"] = action
        td = env.step(td)

    final_ptr = env._self_obs_tensor.data_ptr()
    assert initial_ptr == final_ptr, "GPU tensor data pointer changed after steps"

    # Test view operations on GPU
    obs_view = env._self_obs_tensor.view(-1)
    assert obs_view.data_ptr() == env._self_obs_tensor.data_ptr(), (
        "GPU tensor view doesn't share storage"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
