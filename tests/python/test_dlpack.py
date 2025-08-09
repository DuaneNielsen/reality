#!/usr/bin/env python3
"""
Test DLPack functionality for zero-copy tensor sharing between Madrona and PyTorch.
"""

import pytest
import torch


def test_dlpack_protocol_exists(cpu_manager):
    """Test that tensors implement the DLPack protocol"""
    tensor = cpu_manager.action_tensor()

    # Check __dlpack__ method exists
    assert hasattr(tensor, "__dlpack__"), "Tensor should implement __dlpack__ method"
    assert hasattr(tensor, "__dlpack_device__"), "Tensor should implement __dlpack_device__ method"


@pytest.mark.skipif(
    not hasattr(torch, "from_dlpack"), reason="PyTorch version doesn't support DLPack"
)
def test_dlpack_to_torch_cpu(cpu_manager):
    """Test converting CPU tensors to PyTorch via DLPack"""
    # Get a tensor from the manager
    action_tensor = cpu_manager.action_tensor()

    try:
        # Convert to PyTorch using DLPack protocol
        torch_tensor = torch.from_dlpack(action_tensor)

        # Verify basic properties
        assert torch_tensor.device.type == "cpu", "Should be CPU tensor"
        # Action tensor is flattened to (num_worlds * num_agents, num_actions)
        assert len(torch_tensor.shape) == 2, "Should be 2D tensor"
        assert torch_tensor.shape[0] == 4, "Should have 4 worlds"
        assert torch_tensor.shape[1] == 3, "Should have 3 action dimensions"
        assert torch_tensor.dtype == torch.int32, "Should be int32 dtype"

    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension module not built")
        raise


@pytest.mark.skipif(
    not hasattr(torch, "from_dlpack"), reason="PyTorch version doesn't support DLPack"
)
def test_dlpack_zero_copy(cpu_manager):
    """Test that DLPack conversion is truly zero-copy"""
    # Get action tensor
    action_tensor = cpu_manager.action_tensor()

    try:
        # Convert to numpy first (this is zero-copy)
        numpy_view = action_tensor.to_numpy()
        original_value = numpy_view[0, 0].copy()

        # Convert to PyTorch via DLPack
        torch_tensor = torch.from_dlpack(action_tensor)

        # Modify PyTorch tensor
        new_value = (original_value + 1) % 4  # Keep in valid action range
        torch_tensor[0, 0] = new_value

        # Check that numpy view sees the change (proving zero-copy)
        assert numpy_view[0, 0] == new_value, "DLPack should provide zero-copy access"

        # Reset for cleanup
        torch_tensor[0, 0] = original_value

    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension module not built")
        raise


@pytest.mark.skipif(
    not hasattr(torch, "from_dlpack"), reason="PyTorch version doesn't support DLPack"
)
def test_dlpack_device_info(cpu_manager):
    """Test DLPack device information"""
    tensor = cpu_manager.action_tensor()

    # Get device info
    device_type, device_id = tensor.__dlpack_device__()

    # CPU device type is 1 in DLPack
    assert device_type == 1, "CPU device type should be 1"
    assert device_id == 0, "CPU device ID should be 0"


def test_dlpack_gpu_tensor(gpu_manager):
    """Test DLPack with GPU tensors"""
    # Get GPU tensor
    action_tensor = gpu_manager.action_tensor()

    # Check device info
    device_type, device_id = action_tensor.__dlpack_device__()

    # CUDA device type is 2 in DLPack
    assert device_type == 2, "CUDA device type should be 2"
    assert device_id == 0, "Should be on GPU 0"

    try:
        # Convert to PyTorch
        torch_tensor = torch.from_dlpack(action_tensor)

        # Verify it's on GPU
        assert torch_tensor.is_cuda, "Should be CUDA tensor"
        assert torch_tensor.device.index == 0, "Should be on GPU 0"
        assert torch_tensor.shape[0] == 4, "Should have 4 worlds"

    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension module not built")
        raise


def test_dlpack_multiple_tensors(cpu_manager):
    """Test DLPack with multiple tensor types"""
    try:
        # Test various tensor types
        tensors_to_test = [
            (cpu_manager.action_tensor(), torch.int32, "action"),
            (cpu_manager.reward_tensor(), torch.float32, "reward"),
            (cpu_manager.done_tensor(), torch.int32, "done"),
            (cpu_manager.self_observation_tensor(), torch.float32, "self_observation"),
        ]

        for madrona_tensor, expected_dtype, name in tensors_to_test:
            # Convert via DLPack
            torch_tensor = torch.from_dlpack(madrona_tensor)

            # Verify dtype
            assert torch_tensor.dtype == expected_dtype, (
                f"{name} tensor should have dtype {expected_dtype}"
            )

            # Verify it's a view (zero-copy)
            if hasattr(madrona_tensor, "to_numpy"):
                numpy_view = madrona_tensor.to_numpy()
                assert torch_tensor.data_ptr() == numpy_view.__array_interface__["data"][0], (
                    f"{name} tensor should be zero-copy"
                )

    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension module not built")
        raise


def test_dlpack_with_torch_operations(cpu_manager):
    """Test using DLPack tensors in PyTorch operations"""
    try:
        # Get observation tensor
        obs_tensor = cpu_manager.self_observation_tensor()
        torch_obs = torch.from_dlpack(obs_tensor)

        # Perform some PyTorch operations
        mean_obs = torch_obs.mean(dim=0)
        std_obs = torch_obs.std(dim=0)

        # Verify operations work
        assert mean_obs.shape == torch_obs.shape[1:], "Mean should reduce first dimension"
        assert std_obs.shape == torch_obs.shape[1:], "Std should reduce first dimension"

        # Test gradient computation (should work with float tensors)
        if torch_obs.dtype.is_floating_point:
            torch_obs_grad = torch_obs.clone().requires_grad_(True)
            loss = torch_obs_grad.sum()
            loss.backward()
            assert torch_obs_grad.grad is not None, "Should compute gradients"

    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension module not built")
        raise


def test_dlpack_memory_layout(cpu_manager):
    """Test that DLPack preserves memory layout"""
    try:
        # Get a tensor with known layout
        action_tensor = cpu_manager.action_tensor()
        numpy_view = action_tensor.to_numpy()

        # Convert via DLPack
        torch_tensor = torch.from_dlpack(action_tensor)

        # Check strides match
        numpy_strides = numpy_view.strides
        torch_strides = tuple(s * torch_tensor.element_size() for s in torch_tensor.stride())

        assert numpy_strides == torch_strides, "Memory layout should be preserved"

        # Check contiguity
        assert numpy_view.flags["C_CONTIGUOUS"] == torch_tensor.is_contiguous(), (
            "Contiguity should be preserved"
        )

    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension module not built")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
