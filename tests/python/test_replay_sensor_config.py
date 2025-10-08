"""
Test replay sensor configuration preservation.

Verifies that replay recordings preserve and restore sensor configuration
(lidar beam count, FOV, noise parameters) correctly.
"""

import tempfile

import numpy as np
import pytest

from madrona_escape_room import SimManager
from madrona_escape_room.generated_madrona_constants import ExecMode
from madrona_escape_room.sensor_config import LidarConfig


def test_replay_default_sensor_config(cpu_manager):
    """Test replay with default sensor config from fixture."""
    mgr = cpu_manager

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Record 5 steps with default config
        mgr.start_recording(recording_path)

        for _ in range(5):
            mgr.step()

        mgr.stop_recording()

        # Get original lidar shape
        original_lidar = mgr.lidar_tensor().to_numpy()
        original_shape = original_lidar.shape
        original_beam_count = original_shape[2]

        # Load replay
        replay_mgr = SimManager.from_replay(recording_path, exec_mode=ExecMode.CPU)

        # Verify lidar tensor shape matches original exactly
        replay_lidar = replay_mgr.lidar_tensor().to_numpy()
        assert (
            replay_lidar.shape == original_shape
        ), f"Replay lidar shape {replay_lidar.shape} doesn't match original {original_shape}"

        # Verify beam count matches (not the hardcoded default)
        assert (
            replay_lidar.shape[2] == original_beam_count
        ), f"Expected {original_beam_count} lidar beams from replay, got {replay_lidar.shape[2]}"

    finally:
        import os

        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_custom_32_beam_180_fov():
    """Test replay with 32 beams, 180° FOV."""
    lidar_config = LidarConfig(lidar_num_samples=32, lidar_fov_degrees=180.0)

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=2,
        rand_seed=42,
        auto_reset=False,
        lidar_config=lidar_config,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Record 5 steps with 32 beam config
        mgr.start_recording(recording_path)

        for _ in range(5):
            mgr.step()

        mgr.stop_recording()

        # Get original lidar shape (tensor is always maxLidarSamples=256, config controls usage)
        original_lidar = mgr.lidar_tensor().to_numpy()
        original_shape = original_lidar.shape

        # Verify tensor has max size (256) but config specifies 32 beam usage
        assert (
            original_lidar.shape[2] == 256
        ), f"Lidar tensor should always be max size (256), got {original_lidar.shape[2]}"

        # Read metadata to verify sensor config was stored correctly
        metadata = SimManager.read_replay_metadata(recording_path)
        assert (
            metadata.sensor_config.lidar_num_samples == 32
        ), f"Expected 32 beams in config, got {metadata.sensor_config.lidar_num_samples}"
        assert (
            metadata.sensor_config.lidar_fov_degrees == 180.0
        ), f"Expected 180° FOV in config, got {metadata.sensor_config.lidar_fov_degrees}"

        # Load replay
        replay_mgr = SimManager.from_replay(recording_path, exec_mode=ExecMode.CPU)

        # Verify lidar tensor shape matches original (both use maxLidarSamples)
        replay_lidar = replay_mgr.lidar_tensor().to_numpy()
        assert (
            replay_lidar.shape == original_shape
        ), f"Replay lidar shape {replay_lidar.shape} doesn't match original {original_shape}"

        assert (
            replay_lidar.shape[2] == 256
        ), f"Lidar tensor should be max size (256), got {replay_lidar.shape[2]}"

    finally:
        import os

        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_custom_256_beam_360_fov():
    """Test replay with 256 beams, 360° FOV."""
    lidar_config = LidarConfig(lidar_num_samples=256, lidar_fov_degrees=360.0)

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=42,
        auto_reset=False,
        lidar_config=lidar_config,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Record 3 steps with 256 beam config
        mgr.start_recording(recording_path)

        for _ in range(3):
            mgr.step()

        mgr.stop_recording()

        # Get original lidar shape
        original_lidar = mgr.lidar_tensor().to_numpy()
        original_shape = original_lidar.shape

        # Verify original has 256 beams
        assert (
            original_lidar.shape[2] == 256
        ), f"Original should have 256 beams, got {original_lidar.shape[2]}"

        # Load replay
        replay_mgr = SimManager.from_replay(recording_path, exec_mode=ExecMode.CPU)

        # Verify lidar tensor shape matches original (256 beams, not default 128)
        replay_lidar = replay_mgr.lidar_tensor().to_numpy()
        assert (
            replay_lidar.shape == original_shape
        ), f"Replay lidar shape {replay_lidar.shape} doesn't match original {original_shape}"

        assert (
            replay_lidar.shape[2] == 256
        ), f"Expected 256 lidar beams from replay, got {replay_lidar.shape[2]}"

    finally:
        import os

        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_with_noise():
    """Test replay with noise enabled (64 beams, 120° FOV, noise)."""
    lidar_config = LidarConfig(
        lidar_num_samples=64,
        lidar_fov_degrees=120.0,
        lidar_noise_factor=0.005,
        lidar_base_sigma=0.02,
    )

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=42,
        auto_reset=False,
        lidar_config=lidar_config,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Record 5 steps with noise config
        mgr.start_recording(recording_path)

        for _ in range(5):
            mgr.step()

        mgr.stop_recording()

        # Get original lidar shape (tensor is always maxLidarSamples=256, config controls usage)
        original_lidar = mgr.lidar_tensor().to_numpy()
        original_shape = original_lidar.shape

        # Verify tensor has max size (256) but config specifies 64 beam usage
        assert (
            original_lidar.shape[2] == 256
        ), f"Lidar tensor should always be max size (256), got {original_lidar.shape[2]}"

        # Read metadata to verify sensor config was stored correctly
        metadata = SimManager.read_replay_metadata(recording_path)
        assert (
            metadata.sensor_config.lidar_num_samples == 64
        ), f"Expected 64 beams in config, got {metadata.sensor_config.lidar_num_samples}"
        assert metadata.sensor_config.lidar_fov_degrees == pytest.approx(
            120.0
        ), f"Expected 120° FOV in config, got {metadata.sensor_config.lidar_fov_degrees}"
        assert metadata.sensor_config.lidar_noise_factor == pytest.approx(
            0.005
        ), f"Expected 0.005 noise factor, got {metadata.sensor_config.lidar_noise_factor}"

        # Load replay
        replay_mgr = SimManager.from_replay(recording_path, exec_mode=ExecMode.CPU)

        # Verify lidar tensor shape matches original (both use maxLidarSamples)
        replay_lidar = replay_mgr.lidar_tensor().to_numpy()
        assert (
            replay_lidar.shape == original_shape
        ), f"Replay lidar shape {replay_lidar.shape} doesn't match original {original_shape}"

        assert (
            replay_lidar.shape[2] == 256
        ), f"Lidar tensor should be max size (256), got {replay_lidar.shape[2]}"

    finally:
        import os

        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_metadata_sensor_config():
    """Test that read_replay_metadata() includes sensor_config."""
    lidar_config = LidarConfig(
        lidar_num_samples=48, lidar_fov_degrees=90.0, lidar_noise_factor=0.01, lidar_base_sigma=0.05
    )

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=42,
        auto_reset=False,
        lidar_config=lidar_config,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Record 3 steps
        mgr.start_recording(recording_path)

        for _ in range(3):
            mgr.step()

        mgr.stop_recording()

        # Read metadata
        metadata = SimManager.read_replay_metadata(recording_path)

        assert metadata is not None, "Failed to read metadata"

        # Verify sensor_config field exists and has correct values
        assert hasattr(metadata, "sensor_config"), "Metadata missing sensor_config field"

        sensor_config = metadata.sensor_config
        assert (
            sensor_config.lidar_num_samples == 48
        ), f"Expected 48 beams in metadata, got {sensor_config.lidar_num_samples}"
        assert sensor_config.lidar_fov_degrees == pytest.approx(
            90.0
        ), f"Expected 90.0° FOV in metadata, got {sensor_config.lidar_fov_degrees}"
        assert sensor_config.lidar_noise_factor == pytest.approx(
            0.01
        ), f"Expected 0.01 noise factor in metadata, got {sensor_config.lidar_noise_factor}"
        assert sensor_config.lidar_base_sigma == pytest.approx(
            0.05
        ), f"Expected 0.05 base sigma in metadata, got {sensor_config.lidar_base_sigma}"

    finally:
        import os

        if os.path.exists(recording_path):
            os.unlink(recording_path)


def test_replay_determinism_with_custom_sensor():
    """Verify replay produces identical observations with custom sensor config."""
    lidar_config = LidarConfig(lidar_num_samples=96, lidar_fov_degrees=150.0)

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=42,
        auto_reset=False,
        lidar_config=lidar_config,
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Record 5 steps
        mgr.start_recording(recording_path)

        # Store observations from original run
        original_observations = []
        for _ in range(5):
            mgr.step()
            obs = mgr.self_observation_tensor().to_numpy().copy()
            original_observations.append(obs)

        mgr.stop_recording()

        # Load replay
        replay_mgr = SimManager.from_replay(recording_path, exec_mode=ExecMode.CPU)

        # Replay and verify observations match
        for step_idx in range(5):
            replay_mgr.replay_step()
            replay_mgr.step()

            replay_obs = replay_mgr.self_observation_tensor().to_numpy()

            # Observations should be identical
            np.testing.assert_array_almost_equal(
                replay_obs,
                original_observations[step_idx],
                decimal=5,
                err_msg=f"Observations differ at step {step_idx}",
            )

    finally:
        import os

        if os.path.exists(recording_path):
            os.unlink(recording_path)
