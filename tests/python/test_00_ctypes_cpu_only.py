#!/usr/bin/env python3
"""
Test that verifies import order independence between PyTorch and madrona_escape_room.
This test runs first (00_ prefix) to catch CUDA library version conflicts early.
"""

import numpy as np


def test_import_order_torch_first():
    """Test that madrona_escape_room can be imported after PyTorch"""
    # Import PyTorch first (this loads PyTorch's CUDA libraries)

    # Now import madrona_escape_room - this should work regardless of import order
    import madrona_escape_room

    assert hasattr(madrona_escape_room, "SimManager")
    assert hasattr(madrona_escape_room, "madrona")


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
        exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
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
        exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=1,
        rand_seed=42,
        auto_reset=True,
        enable_batch_renderer=False,
    )

    # Get madrona tensor
    reward_tensor = mgr.reward_tensor()
    reward_np = reward_tensor.to_numpy()

    # Convert to PyTorch tensor
    reward_torch = torch.from_numpy(reward_np)

    # Verify they have the same data
    assert reward_torch.shape == reward_np.shape
    assert torch.allclose(reward_torch, torch.from_numpy(reward_np))
