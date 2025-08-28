#!/usr/bin/env python3
"""
Test DLPack implementation for Madrona Escape Room tensors
"""

import ctypes

import pytest


def test_dlpack_extension_import():
    """Test that the DLPack extension can be imported"""
    try:
        import _madrona_escape_room_dlpack as dlpack_ext

        assert hasattr(dlpack_ext, "create_dlpack_capsule")
        assert hasattr(dlpack_ext, "get_dlpack_device")
    except ImportError:
        pytest.skip("DLPack extension not built")


def test_dlpack_device_function():
    """Test the get_dlpack_device function"""
    try:
        import _madrona_escape_room_dlpack as dlpack_ext

        # Test CPU device
        device_info = dlpack_ext.get_dlpack_device(1, 0)  # CPU, device 0
        assert device_info == (1, 0)

        # Test CUDA device
        device_info = dlpack_ext.get_dlpack_device(2, 1)  # CUDA, device 1
        assert device_info == (2, 1)

    except ImportError:
        pytest.skip("DLPack extension not built")


def test_tensor_dlpack_methods_exist(cpu_manager):
    """Test that Tensor class has DLPack protocol methods"""
    # Get a tensor from the fixture
    action_tensor = cpu_manager.action_tensor()

    # Check that DLPack protocol methods exist
    assert hasattr(action_tensor, "__dlpack__")
    assert hasattr(action_tensor, "__dlpack_device__")
    assert callable(action_tensor.__dlpack__)
    assert callable(action_tensor.__dlpack_device__)


def test_dlpack_device_method(cpu_manager):
    """Test the __dlpack_device__ method"""
    action_tensor = cpu_manager.action_tensor()

    # Test __dlpack_device__ method
    device_info = action_tensor.__dlpack_device__()
    assert isinstance(device_info, tuple)
    assert len(device_info) == 2
    assert device_info[0] == 1  # CPU device type
    assert device_info[1] == 0  # CPU device id


def test_dlpack_capsule_creation(cpu_manager):
    """Test DLPack capsule creation"""
    action_tensor = cpu_manager.action_tensor()

    # Test __dlpack__ method
    capsule = action_tensor.__dlpack__()

    # Check that we got a PyCapsule
    assert capsule is not None
    assert str(type(capsule)) == "<class 'PyCapsule'>"

    # Check if it's a valid DLTensor capsule
    PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
    PyCapsule_IsValid.restype = ctypes.c_int
    PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]

    is_valid = PyCapsule_IsValid(capsule, b"dltensor")
    assert bool(is_valid), "Created capsule should be a valid DLTensor capsule"


@pytest.mark.skipif(False, reason="Test PyTorch integration")
def test_pytorch_from_dlpack(cpu_manager):
    """Test PyTorch integration with torch.from_dlpack()"""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    action_tensor = cpu_manager.action_tensor()

    # Test PyTorch integration
    try:
        torch_tensor = torch.from_dlpack(action_tensor)

        # Verify tensor properties
        assert torch_tensor.shape == action_tensor.shape
        assert torch_tensor.device.type == "cpu"

        # Test that it's zero-copy by modifying the torch tensor
        # and checking if the original tensor changes
        _original_value = action_tensor.to_numpy()[0, 0]
        torch_tensor[0, 0] = 999
        new_value = action_tensor.to_numpy()[0, 0]
        assert new_value == 999, "Should be zero-copy"

    except Exception as e:
        pytest.fail(f"torch.from_dlpack() failed: {e}")


def test_fallback_behavior(cpu_manager):
    """Test that DLPack extension is available (fallback behavior not needed)"""
    action_tensor = cpu_manager.action_tensor()

    # In the current environment, the DLPack extension is always available
    # So we test that __dlpack__ works normally
    try:
        dlpack_capsule = action_tensor.__dlpack__()
        assert dlpack_capsule is not None
        assert str(type(dlpack_capsule)) == "<class 'PyCapsule'>"
    except ImportError as e:
        if "DLPack extension module not found" in str(e):
            pytest.skip("DLPack extension not available - this is expected fallback behavior")
        raise
