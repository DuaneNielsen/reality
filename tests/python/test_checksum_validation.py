"""
Test suite for episode checksum validation functionality.

Tests the checksum system that validates replay accuracy by computing
deterministic hashes of simulation state at episode boundaries.
"""

import numpy as np
import pytest
import torch
from test_helpers import AgentController

from madrona_escape_room import ExecMode, SimManager
from madrona_escape_room import generated_constants as consts


def set_random_actions(agent_controller):
    """Helper function to set random actions for all agents"""
    # Get random values for each action component
    # Actions shape is [num_worlds, action_components] where action_components = 3
    action_data = agent_controller.actions.cpu().numpy()

    for world_idx in range(agent_controller.num_worlds):
        # Random move amount (0-3)
        action_data[world_idx, 0] = np.random.randint(0, 4)
        # Random move angle (0-7)
        action_data[world_idx, 1] = np.random.randint(0, 8)
        # Random rotate (0-4)
        action_data[world_idx, 2] = np.random.randint(0, 5)


class TestChecksumValidation:
    """Test checksum validation API and functionality"""

    def test_checksum_validation_api_cpu(self, cpu_manager, tmp_path):
        """Test checksum validation control API with CPU manager"""
        mgr = cpu_manager

        # Test checksums enabled by default via start_recording
        recording_path_enabled = tmp_path / "test_enabled.rec"
        mgr.start_recording(str(recording_path_enabled), enable_checksums=True)
        mgr.stop_recording()

        # Verify checksums were enabled in metadata
        metadata = SimManager.read_replay_metadata(str(recording_path_enabled))
        assert metadata.enable_checksums == 1
        assert metadata.checksum_version == 1

        # Test checksums disabled via start_recording
        recording_path_disabled = tmp_path / "test_disabled.rec"
        mgr.start_recording(str(recording_path_disabled), enable_checksums=False)
        mgr.stop_recording()

        # Verify checksums were disabled in metadata
        metadata = SimManager.read_replay_metadata(str(recording_path_disabled))
        assert metadata.enable_checksums == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checksum_validation_api_gpu(self, gpu_manager, tmp_path):
        """Test checksum validation control API with GPU manager"""
        mgr = gpu_manager

        # Test checksums enabled by default via start_recording
        recording_path_enabled = tmp_path / "test_enabled_gpu.rec"
        mgr.start_recording(str(recording_path_enabled), enable_checksums=True)
        mgr.stop_recording()

        # Verify checksums were enabled in metadata
        metadata = SimManager.read_replay_metadata(str(recording_path_enabled))
        assert metadata.enable_checksums == 1
        assert metadata.checksum_version == 1

        # Test checksums disabled via start_recording
        recording_path_disabled = tmp_path / "test_disabled_gpu.rec"
        mgr.start_recording(str(recording_path_disabled), enable_checksums=False)
        mgr.stop_recording()

        # Verify checksums were disabled in metadata
        metadata = SimManager.read_replay_metadata(str(recording_path_disabled))
        assert metadata.enable_checksums == 0

    def test_recording_with_checksums(self, cpu_manager, tmp_path):
        """Test that recording includes checksums when enabled"""
        mgr = cpu_manager
        recording_path = tmp_path / "test_recording.rec"

        # Start recording with checksums enabled
        mgr.start_recording(str(recording_path), enable_checksums=True)

        # Simulate some steps to potentially generate episodes
        agent_controller = AgentController(mgr)

        for step in range(50):
            set_random_actions(agent_controller)
            mgr.step()

        mgr.stop_recording()

        # Verify metadata structure
        metadata = SimManager.read_replay_metadata(str(recording_path))
        assert metadata is not None
        assert hasattr(metadata, "enable_checksums")
        assert hasattr(metadata, "checksum_version")
        assert hasattr(metadata, "num_episode_checksums")

        assert metadata.enable_checksums == 1
        assert metadata.checksum_version == 1
        # num_episode_checksums should be >= 0 (may be 0 if no episodes completed)
        assert metadata.num_episode_checksums >= 0

    def test_replay_metadata_includes_checksum_fields(self, cpu_manager, tmp_path):
        """Test that replay metadata includes all checksum-related fields"""
        mgr = cpu_manager
        recording_path = tmp_path / "test_fields.rec"

        # Start recording with checksums enabled
        mgr.start_recording(str(recording_path), enable_checksums=True)

        # Take a few steps
        agent_controller = AgentController(mgr)
        for _ in range(10):
            set_random_actions(agent_controller)
            mgr.step()

        mgr.stop_recording()

        # Read replay metadata
        metadata = SimManager.read_replay_metadata(str(recording_path))
        assert metadata is not None

        # Verify version 4 fields are present
        assert hasattr(metadata, "version")
        assert metadata.version == 4  # Should be version 4 with checksum support

        assert hasattr(metadata, "checksum_version")
        assert metadata.checksum_version == 1  # Version 1 of checksum algorithm

        assert hasattr(metadata, "enable_checksums")
        assert metadata.enable_checksums == 1  # Should be enabled

        assert hasattr(metadata, "num_episode_checksums")
        assert metadata.num_episode_checksums >= 0  # Should have non-negative count

    def test_replay_with_checksums_enabled(self, cpu_manager, tmp_path):
        """Test replay functionality with checksums enabled"""
        mgr = cpu_manager
        recording_path = tmp_path / "test_replay.rec"

        # Start recording with checksums enabled
        mgr.start_recording(str(recording_path), enable_checksums=True)

        # Run some steps
        agent_controller = AgentController(mgr)
        for _ in range(30):
            set_random_actions(agent_controller)
            mgr.step()

        mgr.stop_recording()

        # Create new manager from replay
        mgr2 = SimManager.from_replay(str(recording_path), ExecMode.CPU, gpu_id=0)

        # Step through replay
        for _ in range(10):  # Replay some steps
            result = mgr2.replay_step()
            if not result:
                break  # End of replay

    def test_checksum_validation_disabled_during_replay(self, cpu_manager, tmp_path):
        """Test replay with checksum validation disabled"""
        mgr = cpu_manager
        recording_path = tmp_path / "test_disabled_replay.rec"

        # Start recording with checksums enabled
        mgr.start_recording(str(recording_path), enable_checksums=True)

        # Run some steps
        agent_controller = AgentController(mgr)
        for _ in range(20):
            set_random_actions(agent_controller)
            mgr.step()

        mgr.stop_recording()

        # Create new manager from replay (checksums are controlled at recording time)
        mgr2 = SimManager.from_replay(str(recording_path), ExecMode.CPU, gpu_id=0)

        # Step through replay
        for _ in range(10):
            result = mgr2.replay_step()
            if not result:
                break

    def test_empty_recording_metadata(self, cpu_manager, tmp_path):
        """Test metadata for empty recording"""
        mgr = cpu_manager
        recording_path = tmp_path / "test_empty.rec"

        # Start and immediately stop recording
        mgr.start_recording(str(recording_path), enable_checksums=True)
        mgr.stop_recording()

        # Read metadata
        metadata = SimManager.read_replay_metadata(str(recording_path))
        assert metadata is not None
        assert metadata.enable_checksums == 1
        assert metadata.checksum_version == 1
        assert metadata.num_episode_checksums == 0  # No episodes completed

    def test_default_enable_checksums_parameter(self, cpu_manager, tmp_path):
        """Test that start_recording defaults to enable_checksums=True"""
        mgr = cpu_manager
        recording_path = tmp_path / "test_default.rec"

        # Start recording without specifying enable_checksums (should default to True)
        mgr.start_recording(str(recording_path))
        mgr.stop_recording()

        # Verify checksums were enabled by default
        metadata = SimManager.read_replay_metadata(str(recording_path))
        assert metadata.enable_checksums == 1
        assert metadata.checksum_version == 1
