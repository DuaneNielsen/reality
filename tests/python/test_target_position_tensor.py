"""
Test target position tensor export functionality.
Tests that the new target position tensor is properly exported from the simulation
and accessible through the Python API.
"""

import numpy as np
import pytest

from madrona_escape_room.generated_constants import ExecMode, consts


@pytest.mark.spec("docs/specs/mgr.md", "targetPositionTensor")
class TestTargetPositionTensor:
    """Test suite for target position tensor functionality"""

    def test_target_position_tensor_cpu(self, cpu_manager):
        """Test target position tensor is accessible on CPU"""
        mgr = cpu_manager

        # Get the target position tensor
        target_tensor = mgr.target_position_tensor()

        # Verify tensor properties
        assert target_tensor is not None, "Target position tensor should not be None"

        # Convert to numpy for inspection
        data = target_tensor.to_numpy()

        # Check tensor shape - should be [num_worlds, max_targets * 3 + 1]
        # The +1 is for the count field in TargetPositionData
        expected_worlds = mgr._c_config.num_worlds
        assert (
            data.shape[0] == expected_worlds
        ), f"First dimension should be {expected_worlds} worlds"

        # The second dimension should be max_targets * 3 + 1
        # Max targets is defined in consts.limits.maxTargets (8), so 8 * 3 + 1 = 25
        expected_data_size = consts.limits.maxTargets * 3 + 1
        assert (
            data.shape[1] == expected_data_size
        ), f"Second dimension should be {expected_data_size}"

        print(f"✓ Target tensor shape: {data.shape}")
        print(f"✓ Sample data from first world: {data[0][:6]}...")

    @pytest.mark.skipif(
        not hasattr(pytest, "gpu_available") or not pytest.gpu_available, reason="GPU not available"
    )
    def test_target_position_tensor_gpu(self, gpu_manager):
        """Test target position tensor is accessible on GPU"""
        mgr = gpu_manager

        # Get the target position tensor
        target_tensor = mgr.target_position_tensor()

        # Verify tensor properties
        assert target_tensor is not None, "Target position tensor should not be None"

        # Convert to numpy for inspection
        data = target_tensor.to_numpy()

        # Check tensor shape
        expected_worlds = mgr._c_config.num_worlds
        assert (
            data.shape[0] == expected_worlds
        ), f"First dimension should be {expected_worlds} worlds"

        expected_data_size = consts.limits.maxTargets * 3 + 1
        assert (
            data.shape[1] == expected_data_size
        ), f"Second dimension should be {expected_data_size}"

        print(f"✓ GPU Target tensor shape: {data.shape}")

    def test_target_position_tensor_after_step(self, cpu_manager):
        """Test that target position tensor can be accessed after simulation steps"""
        mgr = cpu_manager

        # Get initial tensor
        target_tensor1 = mgr.target_position_tensor()
        data1 = target_tensor1.to_numpy()

        # Run a simulation step
        mgr.step()

        # Get tensor after step
        target_tensor2 = mgr.target_position_tensor()
        data2 = target_tensor2.to_numpy()

        # Both should have the same shape
        assert data1.shape == data2.shape, "Tensor shape should remain consistent after steps"

        print(f"✓ Tensor shape consistent after step: {data2.shape}")

    def test_target_position_tensor_multiple_worlds(self, cpu_manager):
        """Test target position tensor with multiple worlds"""
        mgr = cpu_manager

        # Get target tensor
        target_tensor = mgr.target_position_tensor()
        data = target_tensor.to_numpy()

        # Check that we have data for each world
        num_worlds = mgr._c_config.num_worlds
        assert data.shape[0] == num_worlds, f"Should have data for {num_worlds} worlds"

        # Check that each world's data is accessible
        for world_idx in range(num_worlds):
            world_data = data[world_idx]
            assert (
                len(world_data) == consts.limits.maxTargets * 3 + 1
            ), f"World {world_idx} should have correct data size"

        print(f"✓ All {num_worlds} worlds have target position data")

    def test_target_position_tensor_data_format(self, cpu_manager):
        """Test that target position tensor data is in expected format"""
        mgr = cpu_manager

        # Get target tensor
        target_tensor = mgr.target_position_tensor()
        data = target_tensor.to_numpy()

        # Data should be float32
        assert data.dtype == np.float32, f"Expected float32, got {data.dtype}"

        # Check that data is finite (no NaN or infinite values)
        assert np.all(np.isfinite(data)), "All tensor values should be finite"

        print(f"✓ Tensor data type: {data.dtype}")
        print(f"✓ All values finite: {np.all(np.isfinite(data))}")

    def test_target_position_tensor_integration_with_other_tensors(self, cpu_manager):
        """Test that target position tensor works alongside other tensor exports"""
        mgr = cpu_manager

        # Get multiple tensors to ensure no conflicts
        target_tensor = mgr.target_position_tensor()
        obs_tensor = mgr.self_observation_tensor()
        action_tensor = mgr.action_tensor()
        reward_tensor = mgr.reward_tensor()

        # All tensors should be accessible
        assert target_tensor is not None
        assert obs_tensor is not None
        assert action_tensor is not None
        assert reward_tensor is not None

        # All should have the same number of worlds in first dimension
        target_data = target_tensor.to_numpy()
        obs_data = obs_tensor.to_numpy()
        action_data = action_tensor.to_numpy()
        reward_data = reward_tensor.to_numpy()

        num_worlds = mgr._c_config.num_worlds
        assert target_data.shape[0] == num_worlds
        assert obs_data.shape[0] == num_worlds
        assert action_data.shape[0] == num_worlds
        assert reward_data.shape[0] == num_worlds

        print(f"✓ All tensors have consistent world dimension: {num_worlds}")
