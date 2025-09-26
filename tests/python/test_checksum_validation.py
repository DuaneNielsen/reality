"""
Test suite for episode checksum validation functionality.

Tests the checksum system that validates replay accuracy by computing
deterministic hashes of simulation state at episode boundaries.
"""

import pytest
import torch
from test_helpers import AgentController

from madrona_escape_room import ExecMode, SimManager
from madrona_escape_room import generated_constants as consts


class TestChecksumValidation:
    """Test checksum validation API and functionality"""

    def test_checksum_validation_api_cpu(self, cpu_manager):
        """Test checksum validation control API with CPU manager"""
        mgr = cpu_manager

        # Test initial state (should be enabled by default)
        initial_enabled = mgr.is_checksum_validation_enabled()
        assert isinstance(initial_enabled, bool)
        assert initial_enabled  # Should be enabled by default

        # Test disabling checksum validation
        mgr.enable_checksum_validation(False)
        assert not mgr.is_checksum_validation_enabled()

        # Test re-enabling checksum validation
        mgr.enable_checksum_validation(True)
        assert mgr.is_checksum_validation_enabled()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checksum_validation_api_gpu(self, gpu_manager):
        """Test checksum validation control API with GPU manager"""
        mgr = gpu_manager

        # Test initial state (should be enabled by default)
        initial_enabled = mgr.is_checksum_validation_enabled()
        assert isinstance(initial_enabled, bool)
        assert initial_enabled  # Should be enabled by default

        # Test disabling checksum validation
        mgr.enable_checksum_validation(False)
        assert not mgr.is_checksum_validation_enabled()

        # Test re-enabling checksum validation
        mgr.enable_checksum_validation(True)
        assert mgr.is_checksum_validation_enabled()

    def test_recording_with_checksums(self, cpu_manager, tmp_path):
        """Test that recording includes checksums when enabled"""
        mgr = cpu_manager

        # Ensure checksum validation is enabled
        mgr.enable_checksum_validation(True)
        assert mgr.is_checksum_validation_enabled()

        # Start recording
        recording_path = tmp_path / "checksum_test.rec"
        mgr.start_recording(str(recording_path))
        assert mgr.is_recording()

        # Create agent controller for movement
        controller = AgentController(mgr)

        # Simulate some steps to trigger episodes
        for step in range(20):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check for episode completion
            done_tensor = mgr.done_tensor()
            done_data = done_tensor.to_torch()
            if done_data.any():
                break  # Episode completed

        # Stop recording
        mgr.stop_recording()
        assert not mgr.is_recording()

        # Verify recording file was created
        assert recording_path.exists()
        assert recording_path.stat().st_size > 0

    def test_replay_metadata_includes_checksum_fields(self, cpu_manager, tmp_path):
        """Test that replay metadata includes version 4 checksum fields"""
        mgr = cpu_manager

        # Enable checksum validation
        mgr.enable_checksum_validation(True)

        # Record a short episode
        recording_path = tmp_path / "metadata_test.rec"
        mgr.start_recording(str(recording_path))

        controller = AgentController(mgr)
        for _ in range(10):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.MEDIUM)
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

    def test_replay_with_checksum_validation(self, cpu_manager, tmp_path):
        """Test replay with checksum validation enabled"""
        mgr = cpu_manager

        # Enable checksum validation for recording
        mgr.enable_checksum_validation(True)

        # Record a deterministic episode
        recording_path = tmp_path / "replay_validation_test.rec"
        mgr.start_recording(str(recording_path))

        controller = AgentController(mgr)

        # Record deterministic movements
        for step in range(25):
            controller.reset_actions()
            if step < 10:
                controller.move_forward(speed=consts.action.move_amount.FAST)
            elif step < 20:
                controller.strafe_right(speed=consts.action.move_amount.MEDIUM)
            else:
                controller.move_backward(speed=consts.action.move_amount.SLOW)
            mgr.step()

        mgr.stop_recording()

        # Create new manager and load replay
        mgr2 = SimManager(
            exec_mode=ExecMode.CPU,
            num_worlds=1,
            rand_seed=5,  # Same seed for deterministic behavior
            auto_reset=True,
            enable_batch_renderer=False,
        )

        # Enable checksum validation for replay
        mgr2.enable_checksum_validation(True)

        # Load and play back the recording
        success = mgr2.load_replay(str(recording_path))
        assert success
        assert mgr2.has_replay()

        # Step through replay - checksums should validate automatically
        total_steps = mgr2.get_total_replay_steps()
        assert total_steps > 0

        for _ in range(total_steps):
            replay_success = mgr2.replay_step()
            if not replay_success:
                break  # End of replay

        # If we get here without assertion failures, checksum validation passed

    def test_checksum_validation_disabled_during_replay(self, cpu_manager, tmp_path):
        """Test that replay works when checksum validation is disabled"""
        mgr = cpu_manager

        # Record with checksums enabled
        mgr.enable_checksum_validation(True)
        recording_path = tmp_path / "disabled_validation_test.rec"
        mgr.start_recording(str(recording_path))

        controller = AgentController(mgr)
        for _ in range(15):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

        mgr.stop_recording()

        # Create new manager and disable checksum validation
        mgr2 = SimManager(
            exec_mode=ExecMode.CPU,
            num_worlds=1,
            rand_seed=5,
            auto_reset=True,
            enable_batch_renderer=False,
        )

        # Disable checksum validation for replay
        mgr2.enable_checksum_validation(False)
        assert not mgr2.is_checksum_validation_enabled()

        # Load and play back the recording (should work without validation)
        success = mgr2.load_replay(str(recording_path))
        assert success

        total_steps = mgr2.get_total_replay_steps()
        for _ in range(min(10, total_steps)):  # Test first 10 steps
            replay_success = mgr2.replay_step()
            assert replay_success

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_checksum_validation_gpu_recording(self, gpu_manager, tmp_path):
        """Test checksum validation with GPU recording and replay"""
        mgr = gpu_manager

        # Enable checksum validation
        mgr.enable_checksum_validation(True)

        # Record a short GPU episode
        recording_path = tmp_path / "gpu_checksum_test.rec"
        mgr.start_recording(str(recording_path))

        controller = AgentController(mgr)
        for _ in range(15):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.MEDIUM)
            mgr.step()

        mgr.stop_recording()

        # Verify recording was created with correct metadata
        metadata = SimManager.read_replay_metadata(str(recording_path))
        assert metadata is not None
        assert metadata.version == 4
        assert metadata.enable_checksums == 1

        # Note: Due to GPU manager constraints, we cannot create a second GPU manager
        # for replay testing. In a real scenario, the same GPU manager would be used
        # for both recording and replay.

    def test_empty_recording_metadata(self, cpu_manager, tmp_path):
        """Test metadata for empty recording (no episodes completed)"""
        mgr = cpu_manager

        # Enable checksum validation
        mgr.enable_checksum_validation(True)

        # Record very short session (no episodes completed)
        recording_path = tmp_path / "empty_recording_test.rec"
        mgr.start_recording(str(recording_path))

        # Take just a few steps (not enough for episode completion)
        controller = AgentController(mgr)
        for _ in range(3):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.SLOW)
            mgr.step()

        mgr.stop_recording()

        # Read metadata
        metadata = SimManager.read_replay_metadata(str(recording_path))
        assert metadata is not None
        assert metadata.version == 4
        assert metadata.enable_checksums == 1
        assert metadata.num_episode_checksums == 0  # No episodes completed

    def test_checksum_api_type_safety(self, cpu_manager):
        """Test that checksum API methods have correct type behavior"""
        mgr = cpu_manager

        # Test enable_checksum_validation accepts bool
        mgr.enable_checksum_validation(True)
        mgr.enable_checksum_validation(False)

        # Test return type of is_checksum_validation_enabled
        result = mgr.is_checksum_validation_enabled()
        assert isinstance(result, bool)

        # Test that enable_checksum_validation doesn't return anything
        return_value = mgr.enable_checksum_validation(True)
        assert return_value is None  # Should not return anything
