"""
Pytest configuration for Madrona Escape Room tests.
Provides enhanced fixtures with optional action recording for visualization.
"""

import subprocess
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    """Add custom command-line options"""
    parser.addoption(
        "--record-actions",
        action="store_true",
        default=False,
        help="Record actions during test execution for viewer replay",
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Launch viewer after test with recorded actions",
    )
    parser.addoption(
        "--trace-trajectories",
        action="store_true",
        default=False,
        help="Enable trajectory tracing to file for all tests",
    )
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip all tests that require GPU",
    )


def pytest_runtest_setup(item):
    """Hook to skip tests with GPU fixtures when --no-gpu is passed"""
    if item.config.getoption("--no-gpu"):
        # Get all fixtures used by this test
        fixtures_used = item.fixturenames

        # List of GPU-related fixtures to skip
        gpu_fixtures = {"gpu_manager", "gpu_env"}  # Add any other GPU fixtures here

        # Check if test uses any GPU fixtures
        if any(fixture in gpu_fixtures for fixture in fixtures_used):
            pytest.skip("Skipping GPU test due to --no-gpu flag")


@pytest.fixture(scope="function")
def cpu_manager(request):
    """Create a CPU SimManager with optional native recording and trajectory tracing"""
    import madrona_escape_room
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    # Check if test has a custom_level marker
    marker = request.node.get_closest_marker("custom_level")
    if marker:
        # TODO: Convert ASCII level to CompiledLevel when needed
        # For now, use default level even with marker
        compiled_level = create_default_level()
    else:
        compiled_level = create_default_level()

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=False,  # Manual control for reward tests
        compiled_levels=compiled_level,  # Use compiled level
    )

    # Check for debug flags
    record_actions = request.config.getoption("--record-actions")
    trace_trajectories = request.config.getoption("--trace-trajectories")

    if record_actions or trace_trajectories:
        # Use same naming pattern as current system
        test_name = request.node.nodeid.replace("::", "__")
        test_filename = test_name.split("/")[-1] if "/" in test_name else test_name

        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)
        base_path = output_dir / f"{test_filename}_actions"  # No extension - DebugSession adds them

        # Use the master context manager
        with mgr.debug_session(
            base_path=base_path,
            enable_recording=record_actions,
            enable_tracing=trace_trajectories,
            seed=42,
        ):
            yield mgr

        # Print debug info after context exits
        print(f"\n{'=' * 60}")
        if record_actions:
            recording_path = base_path.with_suffix(".bin")
            print(f"Actions recorded: {recording_path}")
            print("Worlds: 4, Agents: 1")
            print(f"To visualize: ./build/viewer --num-worlds 4 --replay {recording_path}")

        if trace_trajectories:
            trajectory_path = base_path.with_name(f"{base_path.stem}_trajectory.txt")
            print(f"Trajectory logged: {trajectory_path}")

        print(f"{'=' * 60}\n")

        # Launch viewer if requested
        if request.config.getoption("--visualize") and record_actions:
            recording_path = base_path.with_suffix(".bin")
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print("Launching viewer...")
                subprocess.run(
                    [
                        str(viewer_path),
                        "--num-worlds",
                        "4",
                        "--replay",
                        str(recording_path),
                        "--pause",
                        "1",  # Start paused for 1 second to see initial state
                    ]
                )
            else:
                print(f"Viewer not found at {viewer_path}")
    else:
        yield mgr


@pytest.fixture(scope="function")
def log_and_verify_replay_cpu_manager(request):
    """Create a CPU SimManager that logs trajectory and automatically verifies replay matches.
    This fixture is independent of --record-actions flag and automatically:
    1. Records all actions during the test
    2. Logs trajectory traces during the test
    3. After test completes, replays the recording and verifies trajectory matches exactly
    """
    import os
    import tempfile

    import madrona_escape_room
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,
        compiled_levels=create_default_level(),
    )

    # Always create a debug session for this fixture
    test_name = request.node.nodeid.replace("::", "__")
    test_filename = test_name.split("/")[-1] if "/" in test_name else test_name

    output_dir = Path("test_recordings")
    output_dir.mkdir(exist_ok=True)
    base_path = output_dir / f"{test_filename}_debug"

    # Always enable both recording and tracing for debug manager
    with mgr.debug_session(
        base_path=base_path, enable_recording=True, enable_tracing=True, seed=42
    ) as debug_session:
        # Store debug session info on the manager for test access
        mgr._debug_recording_path = debug_session.recording_path
        mgr._debug_trajectory_path = debug_session.trajectory_path
        yield mgr

    # After context exits, recording and trajectory are finalized
    # Now automatically verify replay matches original trajectory
    print(f"\n{'=' * 60}")
    print("Debug session complete - verifying replay...")
    print(f"Recording: {debug_session.recording_path}")
    print(f"Original trajectory: {debug_session.trajectory_path}")

    try:
        # Create replay manager and trace file
        replay_mgr = SimManager.from_replay(str(debug_session.recording_path), ExecMode.CPU)

        with tempfile.NamedTemporaryFile(suffix="_replay_trace.txt", delete=False) as f:
            replay_trace_path = f.name

        # Enable trajectory logging for replay
        replay_mgr.enable_trajectory_logging(world_idx=0, agent_idx=0, filename=replay_trace_path)

        # Get total steps and replay them all
        current, total = replay_mgr.get_replay_step_count()
        print(f"Replaying {total} steps...")

        for step in range(total):
            finished = replay_mgr.replay_step()
            replay_mgr.step()  # Execute the simulation step

            if step == total - 1:
                assert finished, f"Expected replay to finish at step {step}"

        replay_mgr.disable_trajectory_logging()

        # Compare trajectory files
        with open(debug_session.trajectory_path, "r") as f:
            original_content = f.read().strip()

        with open(replay_trace_path, "r") as f:
            replay_content = f.read().strip()

        # Verify they match exactly
        if original_content == replay_content:
            print("✓ Replay verification PASSED - trajectories match exactly!")
        else:
            original_lines = original_content.split("\n")
            replay_lines = replay_content.split("\n")

            print("✗ Replay verification FAILED:")
            print(f"  Original: {len(original_lines)} lines, {len(original_content)} chars")
            print(f"  Replay:   {len(replay_lines)} lines, {len(replay_content)} chars")

            # Find first difference
            for i, (orig_line, replay_line) in enumerate(zip(original_lines, replay_lines)):
                if orig_line != replay_line:
                    print(f"  First difference at line {i + 1}:")
                    print(f"    Original: {orig_line}")
                    print(f"    Replay:   {replay_line}")
                    break

            # Clean up and fail
            if os.path.exists(replay_trace_path):
                os.unlink(replay_trace_path)
            raise AssertionError(
                "Replay trajectory verification failed - trajectories don't match!"
            )

        # Clean up replay trace file
        if os.path.exists(replay_trace_path):
            os.unlink(replay_trace_path)

    except Exception as e:
        print(f"✗ Replay verification ERROR: {e}")
        raise

    print(f"To visualize: ./build/viewer --num-worlds 4 --replay {debug_session.recording_path}")
    print(f"{'=' * 60}\n")

    # Launch viewer if --visualize was requested
    if request.config.getoption("--visualize"):
        viewer_path = Path("build/viewer")
        if viewer_path.exists() and debug_session.recording_path.exists():
            print("Launching viewer for debug session...")
            subprocess.run(
                [
                    str(viewer_path),
                    "--num-worlds",
                    "4",
                    "--replay",
                    str(debug_session.recording_path),
                    "--pause",
                    "1",  # Start paused for 1 second to see initial state
                ]
            )


@pytest.fixture(scope="session")
def gpu_manager(request):
    """Create a GPU SimManager for the entire test session"""
    # Check --no-gpu flag first
    if request.config.getoption("--no-gpu"):
        pytest.skip("Skipping GPU fixture due to --no-gpu flag")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import madrona_escape_room
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    mgr = SimManager(
        exec_mode=ExecMode.CUDA,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True,
        compiled_levels=create_default_level(),
    )

    # With session scope, we can't do per-test recording here
    # Just yield the manager
    yield mgr

    # Cleanup - ensure manager is properly destroyed at end of session
    del mgr


@pytest.fixture(scope="session")
def gpu_env(request):
    """Create a GPU MadronaEscapeRoomEnv for the entire test session"""
    # Check --no-gpu flag first
    if request.config.getoption("--no-gpu"):
        pytest.skip("Skipping GPU fixture due to --no-gpu flag")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from madrona_escape_room_learn import MadronaEscapeRoomEnv

    env = MadronaEscapeRoomEnv(num_worlds=4, gpu_id=0, rand_seed=42, auto_reset=False)

    yield env
    env.close()


@pytest.fixture(scope="module")
def _test_manager(request):
    """Private fixture: Create a test manager for reward system tests (module-scoped)

    Note: Use cpu_manager instead for individual tests that need custom levels.
    This fixture is module-scoped and doesn't support per-test custom levels.
    """
    import madrona_escape_room
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=False,  # Manual control over resets
        compiled_levels=create_default_level(),
    )

    yield mgr


@pytest.fixture(autouse=True)
def per_test_recording_handler(request):
    """Autouse fixture that handles per-test recording and tracing for module-scoped test_manager"""
    # Check for debug flags
    record_actions = request.config.getoption("--record-actions")
    trace_trajectories = request.config.getoption("--trace-trajectories")

    # Only run if we have either recording or tracing enabled
    if record_actions or trace_trajectories:
        # Only get _test_manager when we actually need it
        test_manager = request.getfixturevalue("_test_manager")
        # Create test-specific files
        test_name = request.node.nodeid.replace("::", "__")
        test_filename = test_name.split("/")[-1] if "/" in test_name else test_name

        # Create output directory
        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)
        base_path = output_dir / f"{test_filename}_actions"  # No extension - DebugSession adds them

        # Use the master context manager
        debug_session = test_manager.debug_session(
            base_path=base_path,
            enable_recording=record_actions,
            enable_tracing=trace_trajectories,
            seed=42,
        )
        with debug_session:
            yield

        # Print debug info after context exits using actual paths from DebugSession
        print(f"\n{'=' * 60}")
        if record_actions and debug_session.recording_path:
            print(f"Actions recorded: {debug_session.recording_path}")
            print("Worlds: 4, Agents: 1")
            print(
                f"To visualize: ./build/viewer --num-worlds 4 --replay "
                f"{debug_session.recording_path}"
            )

        if trace_trajectories and debug_session.trajectory_path:
            print(f"Trajectory logged: {debug_session.trajectory_path}")

        print(f"{'=' * 60}\n")

        # Launch viewer if requested
        if (
            request.config.getoption("--visualize")
            and record_actions
            and debug_session.recording_path
        ):
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print("Launching viewer...")
                subprocess.run(
                    [
                        str(viewer_path),
                        "--num-worlds",
                        "4",
                        "--replay",
                        str(debug_session.recording_path),
                        "--pause",
                        "1",  # Start paused for 1 second to see initial state
                    ]
                )
            else:
                print(f"Viewer not found at {viewer_path}")
    else:
        yield


@pytest.fixture(scope="function")
def test_manager_from_replay():
    """Factory fixture that creates a SimManager from a replay file.
    Returns a function that takes a replay path and returns the manager.
    """
    import madrona_escape_room
    from madrona_escape_room import ExecMode, SimManager

    def _create_manager(replay_path):
        """Create a manager from replay file"""
        return SimManager.from_replay(str(replay_path), ExecMode.CPU)

    return _create_manager
