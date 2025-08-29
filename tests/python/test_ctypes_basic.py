#!/usr/bin/env python3
"""
Basic test for ctypes bindings - using cpu_manager fixture
"""

import logging

import pytest

logger = logging.getLogger(__name__)

try:
    import torch

    import madrona_escape_room as mer

    logger.info("✓ Successfully imported madrona_escape_room")
except ImportError as e:
    logger.error(f"✗ Failed to import madrona_escape_room: {e}")
    pytest.skip(f"Failed to import madrona_escape_room: {e}")


def test_cpu_manager(cpu_manager):
    """Test CPU manager creation and basic operations"""
    logger.info("\n=== Testing CPU Manager ===")

    try:
        logger.info("✓ Using CPU manager fixture")

        # Get tensors
        logger.info("Getting tensors...")
        action_tensor = cpu_manager.action_tensor()
        logger.info(f"✓ Action tensor: shape={action_tensor.shape}, dtype={action_tensor.dtype}")

        obs_tensor = cpu_manager.self_observation_tensor()
        logger.info(f"✓ Observation tensor: shape={obs_tensor.shape}, dtype={obs_tensor.dtype}")

        reward_tensor = cpu_manager.reward_tensor()
        logger.info(f"✓ Reward tensor: shape={reward_tensor.shape}, dtype={reward_tensor.dtype}")

        # Run one step
        logger.info("Running simulation step...")
        cpu_manager.step()
        logger.info("✓ Simulation step completed")

        logger.info("✓ CPU manager test passed!")

    except Exception as e:
        logger.error(f"✗ CPU manager test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.slow
def test_gpu_manager(gpu_manager):
    """Test GPU manager operations using session fixture"""
    logger.info("\n=== Testing GPU Manager ===")

    try:
        # Use the session-scoped GPU manager
        mgr = gpu_manager
        logger.info("✓ Using session GPU manager")

        # Get tensors
        logger.info("Getting tensors...")
        action_tensor = mgr.action_tensor()
        logger.info(
            f"✓ Action tensor: shape={action_tensor.shape}, dtype={action_tensor.dtype}, "
            f"GPU={action_tensor.isOnGPU()}"
        )

        obs_tensor = mgr.self_observation_tensor()
        logger.info(
            f"✓ Observation tensor: shape={obs_tensor.shape}, dtype={obs_tensor.dtype}, "
            f"GPU={obs_tensor.isOnGPU()}"
        )

        reward_tensor = mgr.reward_tensor()
        logger.info(
            f"✓ Reward tensor: shape={reward_tensor.shape}, dtype={reward_tensor.dtype}, "
            f"GPU={reward_tensor.isOnGPU()}"
        )

        # Run one step
        logger.info("Running simulation step...")
        mgr.step()
        logger.info("✓ Simulation step completed")

        logger.info("✓ GPU manager test passed!")

    except Exception as e:
        logger.error(f"✗ GPU manager test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
