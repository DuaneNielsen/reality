#!/usr/bin/env python3
"""
Test native C++ recording/replay functionality on GPU.
"""

import os
import tempfile

import pytest
import torch


@pytest.mark.slow
@pytest.mark.spec("docs/specs/mgr.md", "replayStep")
def test_gpu_recording_and_replay_complete(request):
    """Test complete GPU recording and replay functionality in one test"""
    # Create a fresh GPU manager to ensure we start at step 0
    from madrona_escape_room import ExecMode, SimManager

    if request.config.getoption("--no-gpu"):
        pytest.skip("Skipping GPU fixture due to --no-gpu flag")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from conftest import _create_sim_manager

    mgr = _create_sim_manager(request, ExecMode.CUDA)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        recording_path = f.name

    try:
        # Phase 1: Test recording functionality
        assert not mgr.is_recording()
        mgr.start_recording(recording_path)
        assert mgr.is_recording()

        # Record varied actions and capture them for later verification
        action_tensor = mgr.action_tensor().to_torch()
        recorded_actions = []

        for step in range(3):
            action_tensor.fill_(0)
            action_tensor[:, 0] = (step % 3) + 1  # Varying move amounts (1, 2, 3)
            action_tensor[:, 1] = step % 8  # Varying move angles (0, 1, 2)
            action_tensor[:, 2] = 2  # No rotation

            recorded_actions.append(action_tensor.clone())
            mgr.step()

        mgr.stop_recording()
        assert not mgr.is_recording()

        # Verify recording file was created
        assert os.path.exists(recording_path)
        file_size = os.path.getsize(recording_path)
        assert file_size > 0
        print(f"✓ GPU recording successful: {file_size} bytes")

        # Phase 2: Test basic replay functionality
        replay_mgr = SimManager.from_replay(recording_path, ExecMode.CUDA, gpu_id=0)
        assert replay_mgr.has_replay()

        current, total = replay_mgr.get_replay_step_count()
        assert current == 0
        assert total == 3
        print(f"✓ GPU replay loaded: {total} steps")

        # Phase 3: Test replay accuracy (round-trip verification)
        for step in range(3):
            finished = replay_mgr.replay_step()

            # Get current action tensor after replay step
            current_actions = replay_mgr.action_tensor().to_torch()

            # Should match what we recorded exactly
            assert torch.equal(recorded_actions[step], current_actions), (
                f"Action mismatch at step {step}: "
                f"expected {recorded_actions[step]}, got {current_actions}"
            )

            # Check replay completion status
            if step < 2:
                assert not finished, f"Replay should not finish at step {step}"
            else:
                assert finished, f"Replay should finish at step {step}"

            replay_mgr.step()

        print("✓ GPU round-trip verification: all actions match exactly")

        # Clean up the replay manager
        del replay_mgr

    finally:
        if os.path.exists(recording_path):
            os.unlink(recording_path)
