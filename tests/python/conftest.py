"""
Pytest configuration for Madrona Escape Room tests.
Provides enhanced fixtures with optional action recording for visualization.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Add tests directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the spec plugin
import pytest
import torch
from conftest_spec import *  # noqa: F401, F403

logger = logging.getLogger(__name__)


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


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers",
        "depth_sensor: enable depth sensor (width, height, fov). Default: 64x64, 100° FOV",
    )
    config.addinivalue_line("markers", "ascii_level: mark test to use custom ASCII level layout")

    # Named sensor configuration markers
    config.addinivalue_line("markers", "sensor.rgb_default: Use default 64x64 RGB camera")
    config.addinivalue_line("markers", "sensor.rgb_high_res: Use 128x128 high-res RGB camera")
    config.addinivalue_line("markers", "sensor.depth_default: Use default 64x64 depth sensor")
    config.addinivalue_line("markers", "sensor.depth_high_res: Use 128x128 high-res depth sensor")
    config.addinivalue_line("markers", "sensor.lidar_128: Use 128-beam horizontal lidar (120° FOV)")
    config.addinivalue_line("markers", "sensor.lidar_64: Use 64-beam horizontal lidar (120° FOV)")
    config.addinivalue_line("markers", "sensor.lidar_256: Use 256-beam high-res lidar (120° FOV)")
    config.addinivalue_line("markers", "sensor.rgbd_default: Use default 64x64 RGBD sensor")
    config.addinivalue_line("markers", "json_level: mark test to use custom JSON level definition")
    config.addinivalue_line(
        "markers",
        "lidar_config: configure lidar sensor "
        "(lidar_num_samples, lidar_fov_degrees, lidar_noise_factor, lidar_base_sigma)",
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (e.g., GPU compilation tests)"
    )


def pytest_runtest_setup(item):
    """Hook to skip tests with GPU fixtures when --no-gpu is passed"""
    if item.config.getoption("--no-gpu"):
        # Get all fixtures used by this test
        fixtures_used = item.fixturenames

        # List of GPU-related fixtures to skip
        gpu_fixtures = {
            "gpu_manager",
            "gpu_env",
            "gpu_manager_with_depth",
        }  # Add any other GPU fixtures here

        # Check if test uses any GPU fixtures
        if any(fixture in gpu_fixtures for fixture in fixtures_used):
            pytest.skip("Skipping GPU test due to --no-gpu flag")


def _create_sim_manager(
    request,
    exec_mode,
    enable_depth_override=None,
    auto_reset=False,
    batch_render_width=64,
    batch_render_height=64,
    custom_vertical_fov=0.0,
    render_mode=None,
):
    """Helper function to create SimManager with pytest marker support

    This is a pytest-specific wrapper around create_sim_manager that handles
    pytest markers for level and sensor configuration.

    Args:
        request: pytest request object
        exec_mode: ExecMode.CPU or ExecMode.CUDA
        enable_depth_override: Optional bool to force enable/disable depth sensor
        auto_reset: Whether to enable automatic episode resets
        batch_render_width: Custom render view width (default 64)
        batch_render_height: Custom render view height (default 64)
        custom_vertical_fov: Custom vertical FOV in degrees (0 = use default)
        render_mode: RenderMode enum value (None = use default RGBD)
    """
    from madrona_escape_room import create_sim_manager
    from madrona_escape_room.sensor_config import LidarConfig, SensorConfig

    # Check for level markers
    ascii_marker = request.node.get_closest_marker("ascii_level")
    json_marker = request.node.get_closest_marker("json_level")
    depth_marker = request.node.get_closest_marker("depth_sensor")
    auto_reset_marker = request.node.get_closest_marker("auto_reset")
    lidar_config_marker = request.node.get_closest_marker("lidar_config")

    # Determine level data from markers
    level_data = None
    if ascii_marker:
        level_data = ascii_marker.args[0]
        logger.info(f"Using ASCII level:\n{level_data}")
    elif json_marker:
        json_dict = json_marker.args[0]
        # Convert dictionary to JSON string for proper processing
        import json

        level_data = json.dumps(json_dict)
        logger.info(f"Using JSON level:\n{level_data}")

    # Check for named sensor configuration markers
    sensor_config = None
    sensor_markers = {
        "sensor.rgb_default": SensorConfig.rgb_default,
        "sensor.rgb_high_res": SensorConfig.rgb_high_res,
        "sensor.depth_default": SensorConfig.depth_default,
        "sensor.depth_high_res": SensorConfig.depth_high_res,
        "sensor.lidar_128": SensorConfig.lidar_horizontal_128,
        "sensor.lidar_64": SensorConfig.lidar_horizontal_64,
        "sensor.lidar_256": SensorConfig.lidar_horizontal_256,
        "sensor.rgbd_default": SensorConfig.rgbd_default,
    }

    # Find which sensor marker is used (if any)
    for marker_name, config_factory in sensor_markers.items():
        # Extract the short name after "sensor."
        short_name = marker_name.split(".")[-1]
        if request.node.get_closest_marker(short_name):
            sensor_config = config_factory()
            logger.info(f"Using sensor config: {sensor_config}")
            break

    # Handle legacy depth_sensor marker if no modern sensor config is used
    if sensor_config is None and (depth_marker is not None or enable_depth_override is not None):
        # Create a sensor config from legacy depth marker
        width = batch_render_width
        height = batch_render_height
        fov = custom_vertical_fov if custom_vertical_fov > 0 else 100.0

        # Override with depth_sensor marker parameters
        if depth_marker:
            if len(depth_marker.args) >= 2:
                width = depth_marker.args[0]
                height = depth_marker.args[1]
            if len(depth_marker.args) >= 3:
                fov = depth_marker.args[2]

        # Determine if depth should be enabled
        enable_depth = (
            enable_depth_override if enable_depth_override is not None else depth_marker is not None
        )

        if enable_depth:
            from madrona_escape_room import RenderMode

            mode = render_mode if render_mode is not None else RenderMode.Depth
            sensor_config = SensorConfig.custom(
                width=width,
                height=height,
                vertical_fov=fov,
                render_mode=mode,
                name=f"Legacy Depth {width}x{height}",
            )

            exec_mode_name = "CPU" if exec_mode == 0 else "CUDA"
            logger.info(f"Legacy depth sensor for {exec_mode_name}: {sensor_config}")

    # Check for auto_reset marker override
    if auto_reset_marker:
        auto_reset = True
        logger.info("Auto-reset enabled via @pytest.mark.auto_reset marker")

    # Check for lidar_config marker
    lidar_config = None
    if lidar_config_marker:
        # Extract kwargs from marker
        kwargs = lidar_config_marker.kwargs
        lidar_config = LidarConfig(
            lidar_num_samples=kwargs.get("lidar_num_samples", 128),
            lidar_fov_degrees=kwargs.get("lidar_fov_degrees", 120.0),
            lidar_noise_factor=kwargs.get("lidar_noise_factor", 0.0),
            lidar_base_sigma=kwargs.get("lidar_base_sigma", 0.0),
        )
        logger.info(f"Using lidar config from marker: {lidar_config}")

    # Use the factory function
    return create_sim_manager(
        exec_mode=exec_mode,
        sensor_config=sensor_config,
        level_data=level_data,
        lidar_config=lidar_config,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        auto_reset=auto_reset,
        batch_render_width=batch_render_width,
        batch_render_height=batch_render_height,
        custom_vertical_fov=custom_vertical_fov,
        render_mode=render_mode,
    )


@pytest.fixture(scope="function")
def cpu_manager(request):
    """Create a CPU SimManager with optional native recording and trajectory tracing"""
    from madrona_escape_room import ExecMode

    mgr = _create_sim_manager(request, ExecMode.CPU)

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

        # Log debug info after context exits
        if record_actions:
            recording_path = base_path.with_suffix(".rec")
            logger.info(f"Actions recorded: {recording_path}")
            logger.info("Worlds: 4, Agents: 1")
            logger.info(f"To visualize: ./build/viewer --num-worlds 4 --replay {recording_path}")

        if trace_trajectories:
            trajectory_path = base_path.with_name(f"{base_path.stem}_trajectory.txt")
            logger.info(f"Trajectory logged: {trajectory_path}")

        # Launch viewer if requested
        if request.config.getoption("--visualize") and record_actions:
            recording_path = base_path.with_suffix(".rec")
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                logger.info("Launching viewer...")
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
                logger.warning(f"Viewer not found at {viewer_path}")
    else:
        yield mgr


@pytest.fixture(scope="function")
def log_and_verify_replay_cpu_manager(request):
    """Create a CPU SimManager that records actions and automatically verifies replay determinism.
    This fixture is independent of --record-actions flag and automatically:
    1. Records all actions during the test
    2. After test completes, replays the recording and verifies determinism using checksums
    """
    from madrona_escape_room import ExecMode

    mgr = _create_sim_manager(request, ExecMode.CPU, auto_reset=True)

    # Always create a debug session for this fixture
    test_name = request.node.nodeid.replace("::", "__")
    test_filename = test_name.split("/")[-1] if "/" in test_name else test_name

    output_dir = Path("test_recordings")
    output_dir.mkdir(exist_ok=True)
    base_path = output_dir / f"{test_filename}_debug"

    # Enable recording only (no trajectory logging needed with checksums)
    with mgr.debug_session(
        base_path=base_path, enable_recording=True, enable_tracing=False, seed=42
    ) as debug_session:
        # Store debug session info on the manager for test access
        mgr._debug_recording_path = debug_session.recording_path
        yield mgr

    # After context exits, recording is finalized
    # Now automatically verify replay determinism using checksums
    logger.info("Debug session complete - verifying replay determinism...")
    logger.info(f"Recording: {debug_session.recording_path}")

    try:
        # Create replay manager
        from madrona_escape_room import SimManager

        replay_mgr = SimManager.from_replay(str(debug_session.recording_path), ExecMode.CPU)

        # Get total steps and replay them all
        current, total = replay_mgr.get_replay_step_count()
        logger.info(f"Replaying {total} steps with checksum verification...")

        for step in range(total):
            finished = replay_mgr.replay_step()
            replay_mgr.step()  # Execute the simulation step

            if step == total - 1:
                assert finished, f"Expected replay to finish at step {step}"

        # Check if checksum verification passed
        checksum_failed = replay_mgr.has_checksum_failed()
        if not checksum_failed:
            logger.info("✓ Replay verification PASSED - checksums match (deterministic replay)!")
        else:
            logger.error("✗ Replay verification FAILED - checksum mismatch detected!")
            raise AssertionError(
                "Replay checksum verification failed - replay is not deterministic! "
                "Same actions produced different positions."
            )

    except Exception as e:
        logger.error(f"✗ Replay verification ERROR: {e}")
        raise

    logger.info(
        f"To visualize: ./build/viewer --num-worlds 4 --replay {debug_session.recording_path}"
    )

    # Launch viewer if --visualize was requested
    if request.config.getoption("--visualize"):
        viewer_path = Path("build/viewer")
        if viewer_path.exists() and debug_session.recording_path.exists():
            logger.info("Launching viewer for debug session...")
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

    from madrona_escape_room import ExecMode

    mgr = _create_sim_manager(request, ExecMode.CUDA, auto_reset=True)

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
    from madrona_escape_room import ExecMode

    mgr = _create_sim_manager(request, ExecMode.CPU, auto_reset=False)

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

        # Log debug info after context exits using actual paths from DebugSession
        if record_actions and debug_session.recording_path:
            logger.info(f"Actions recorded: {debug_session.recording_path}")
            logger.info("Worlds: 4, Agents: 1")
            logger.info(
                f"To visualize: ./build/viewer --num-worlds 4 --replay "
                f"{debug_session.recording_path}"
            )

        if trace_trajectories and debug_session.trajectory_path:
            logger.info(f"Trajectory logged: {debug_session.trajectory_path}")

        # Launch viewer if requested
        if (
            request.config.getoption("--visualize")
            and record_actions
            and debug_session.recording_path
        ):
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                logger.info("Launching viewer...")
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
                logger.warning(f"Viewer not found at {viewer_path}")
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


@pytest.fixture(scope="function")
def cpu_manager_with_depth(request):
    """Create a CPU SimManager with depth sensor (batch renderer) always enabled"""
    from madrona_escape_room import ExecMode

    return _create_sim_manager(request, ExecMode.CPU, enable_depth_override=True)


@pytest.fixture(scope="function")
def gpu_manager_with_depth(request):
    """Create a GPU SimManager with depth sensor (batch renderer) always enabled"""
    if request.config.getoption("--no-gpu"):
        pytest.skip("Skipping GPU fixture due to --no-gpu flag")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from madrona_escape_room import ExecMode

    return _create_sim_manager(request, ExecMode.CUDA, enable_depth_override=True)
