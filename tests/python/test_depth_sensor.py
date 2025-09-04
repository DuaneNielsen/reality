"""
Test depth sensor functionality in Madrona Escape Room
"""

import numpy as np
import pytest


@pytest.mark.depth_default
def test_depth_tensor_basic(cpu_manager):
    """Test that depth tensor is accessible with @depth_default marker"""
    mgr = cpu_manager

    # Step the simulation to get initial observations
    mgr.step()

    # Get depth tensor
    depth = mgr.depth_tensor()

    # Handle both CPU and GPU tensors
    try:
        if depth.isOnGPU():
            depth_torch = depth.to_torch()
            depth_np = depth_torch.cpu().numpy()
        else:
            depth_np = depth.to_numpy()
    except ImportError:
        # Skip if DLPack extension not available
        pytest.skip("DLPack extension not available")

    # Check tensor properties
    assert depth_np.shape == (4, 1, 64, 64, 1), f"Expected (4,1,64,64,1), got {depth_np.shape}"
    assert depth_np.dtype == np.float32, f"Expected float32, got {depth_np.dtype}"

    # Check that depth values are reasonable (should be positive distances)
    assert np.all(depth_np >= 0), "Depth values should be non-negative"
    assert np.any(depth_np > 0), "Should have some non-zero depth values"


def test_depth_tensor_always_enabled(cpu_manager_with_depth):
    """Test dedicated depth sensor fixture that always enables batch renderer"""
    mgr = cpu_manager_with_depth

    mgr.step()

    # Should work without @depth_sensor marker
    depth = mgr.depth_tensor()

    # Handle GPU tensors
    try:
        if depth.isOnGPU():
            depth_torch = depth.to_torch()
            depth_np = depth_torch.cpu().numpy()
        else:
            depth_np = depth.to_numpy()
    except ImportError:
        pytest.skip("DLPack extension not available")

    assert depth_np.shape == (4, 1, 64, 64, 1)
    assert np.all(depth_np >= 0)


@pytest.mark.rgbd_default
def test_depth_and_rgb_together(cpu_manager):
    """Test that both depth and RGB work together in RGBD mode"""
    mgr = cpu_manager

    mgr.step()

    # Get both tensors
    depth = mgr.depth_tensor()
    rgb = mgr.rgb_tensor()

    # Handle GPU tensors
    try:
        if depth.isOnGPU():
            depth_torch = depth.to_torch()
            depth_np = depth_torch.cpu().numpy()
        else:
            depth_np = depth.to_numpy()

        if rgb.isOnGPU():
            rgb_torch = rgb.to_torch()
            rgb_np = rgb_torch.cpu().numpy()
        else:
            rgb_np = rgb.to_numpy()
    except ImportError:
        pytest.skip("DLPack extension not available")

    # Check shapes match in spatial dimensions
    assert (
        depth_np.shape[:4] == rgb_np.shape[:4]
    ), "Depth and RGB should have matching spatial dimensions"
    assert depth_np.shape == (4, 1, 64, 64, 1), "Depth shape"
    assert rgb_np.shape == (4, 1, 64, 64, 4), "RGB shape (RGBA)"


@pytest.mark.skipif(True, reason="GPU test - enable manually")
@pytest.mark.depth_default
def test_depth_gpu(gpu_manager_with_depth):
    """Test depth sensor on GPU (manual enable)"""
    mgr = gpu_manager_with_depth

    mgr.step()

    depth = mgr.depth_tensor()
    # Convert to numpy - this will copy from GPU to CPU
    depth_np = depth.to_numpy()

    assert depth_np.shape == (4, 1, 64, 64, 1)
    assert np.all(depth_np >= 0)


@pytest.mark.skip(
    reason="Depth tensor access without renderer causes segfault - C API needs improvement"
)
def test_depth_tensor_without_renderer_fails(cpu_manager):
    """Test that depth tensor fails gracefully without batch renderer enabled"""
    mgr = cpu_manager  # No @depth_sensor marker, so no batch renderer

    mgr.step()

    # This currently segfaults - should be improved to raise proper exception
    # TODO: Improve C API to return error instead of segfault
    with pytest.raises(Exception):
        mgr.depth_tensor()
