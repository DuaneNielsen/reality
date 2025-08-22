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


def test_tensor_dlpack_methods_exist():
    """Test that Tensor class has DLPack protocol methods"""
    import madrona_escape_room

    # Create a manager to get tensors
    mgr = madrona_escape_room.SimManager(
        madrona_escape_room.madrona.ExecMode.CPU,
        0,  # gpu_id
        1,  # num_worlds
        42,  # rand_seed
        False,  # auto_reset
        False,  # enable_batch_renderer
    )

    # Get a tensor
    action_tensor = mgr.action_tensor()

    # Check that DLPack protocol methods exist
    assert hasattr(action_tensor, "__dlpack__")
    assert hasattr(action_tensor, "__dlpack_device__")
    assert callable(action_tensor.__dlpack__)
    assert callable(action_tensor.__dlpack_device__)


def test_dlpack_device_method():
    """Test the __dlpack_device__ method"""
    import madrona_escape_room

    # Create CPU manager
    mgr = madrona_escape_room.SimManager(
        madrona_escape_room.madrona.ExecMode.CPU,
        0,  # gpu_id
        1,  # num_worlds
        42,  # rand_seed
        False,  # auto_reset
        False,  # enable_batch_renderer
    )

    action_tensor = mgr.action_tensor()

    # Test __dlpack_device__ method
    device_info = action_tensor.__dlpack_device__()
    assert isinstance(device_info, tuple)
    assert len(device_info) == 2
    assert device_info[0] == 1  # CPU device type
    assert device_info[1] == 0  # CPU device id


def test_dlpack_capsule_creation():
    """Test DLPack capsule creation"""
    import madrona_escape_room

    # Create CPU manager
    mgr = madrona_escape_room.SimManager(
        madrona_escape_room.madrona.ExecMode.CPU,
        0,  # gpu_id
        1,  # num_worlds
        42,  # rand_seed
        False,  # auto_reset
        False,  # enable_batch_renderer
    )

    action_tensor = mgr.action_tensor()

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
def test_pytorch_from_dlpack():
    """Test PyTorch integration with torch.from_dlpack()"""
    import madrona_escape_room

    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    # Create CPU manager
    mgr = madrona_escape_room.SimManager(
        madrona_escape_room.madrona.ExecMode.CPU,
        0,  # gpu_id
        1,  # num_worlds
        42,  # rand_seed
        False,  # auto_reset
        False,  # enable_batch_renderer
    )

    action_tensor = mgr.action_tensor()

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


def test_fallback_behavior():
    """Test fallback behavior when DLPack extension is not available"""
    import warnings

    import madrona_escape_room

    # Create CPU manager
    mgr = madrona_escape_room.SimManager(
        madrona_escape_room.madrona.ExecMode.CPU,
        0,  # gpu_id
        1,  # num_worlds
        42,  # rand_seed
        False,  # auto_reset
        False,  # enable_batch_renderer
    )

    action_tensor = mgr.action_tensor()

    # Temporarily make the import fail by modifying sys.modules
    import sys

    original_module = sys.modules.get("_madrona_escape_room_dlpack")
    sys.modules["_madrona_escape_room_dlpack"] = None

    try:
        # This should raise an ImportError when DLPack extension is not available
        with pytest.raises(ImportError, match="DLPack extension module not found"):
            action_tensor.__dlpack__()

    finally:
        # Restore the original module state
        if original_module is not None:
            sys.modules["_madrona_escape_room_dlpack"] = original_module
        else:
            sys.modules.pop("_madrona_escape_room_dlpack", None)
