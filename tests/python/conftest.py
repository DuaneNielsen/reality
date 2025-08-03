"""
Pytest configuration for Madrona Escape Room tests.
Provides enhanced fixtures with optional action recording for visualization.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import subprocess
import sys


def pytest_addoption(parser):
    """Add custom command-line options"""
    parser.addoption(
        "--record-actions",
        action="store_true",
        default=False,
        help="Record actions during test execution for viewer replay"
    )
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Launch viewer after test with recorded actions"
    )
    parser.addoption(
        "--trace-trajectories",
        action="store_true",
        default=False,
        help="Enable trajectory tracing to file for all tests"
    )
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip all tests that require GPU"
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
    from madrona_escape_room import SimManager
    
    mgr = SimManager(
        exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True
    )
    
    # Check for debug flags
    record_actions = request.config.getoption("--record-actions")
    trace_trajectories = request.config.getoption("--trace-trajectories")
    
    if record_actions or trace_trajectories:
        # Use same naming pattern as current system
        test_name = request.node.nodeid.replace("::", "__")
        test_filename = test_name.split('/')[-1] if '/' in test_name else test_name
        
        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)
        base_path = output_dir / f"{test_filename}_actions"  # No extension - DebugSession adds them
        
        # Use the master context manager
        with mgr.debug_session(
            base_path=base_path,
            enable_recording=record_actions,
            enable_tracing=trace_trajectories,
            seed=42
        ):
            yield mgr
            
        # Print debug info after context exits
        print(f"\n{'='*60}")
        if record_actions:
            recording_path = base_path.with_suffix('.bin')
            print(f"Actions recorded: {recording_path}")
            print(f"Worlds: 4, Agents: 1")
            print(f"To visualize: ./build/viewer --num-worlds 4 --replay {recording_path}")
            
        if trace_trajectories:
            trajectory_path = base_path.with_name(f"{base_path.stem}_trajectory.txt")
            print(f"Trajectory logged: {trajectory_path}")
            
        print(f"{'='*60}\n")
        
        # Launch viewer if requested
        if request.config.getoption("--visualize") and record_actions:
            recording_path = base_path.with_suffix('.bin')
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print(f"Launching viewer...")
                subprocess.run([str(viewer_path), "--num-worlds", "4", "--replay", str(recording_path)])
            else:
                print(f"Viewer not found at {viewer_path}")
    else:
        yield mgr


@pytest.fixture(scope="session")
def gpu_manager(request):
    """Create a GPU SimManager for the entire test session"""
    # Check --no-gpu flag first
    if request.config.getoption("--no-gpu"):
        pytest.skip("Skipping GPU fixture due to --no-gpu flag")
        
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    import madrona_escape_room
    from madrona_escape_room import SimManager
    
    mgr = SimManager(
        exec_mode=madrona_escape_room.madrona.ExecMode.CUDA,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=True
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
    
    env = MadronaEscapeRoomEnv(
        num_worlds=4,
        gpu_id=0,
        rand_seed=42,
        auto_reset=False
    )
    
    yield env
    env.close()


@pytest.fixture(scope="module")
def test_manager(request):
    """Create a test manager for reward system tests"""
    import madrona_escape_room
    from madrona_escape_room import SimManager
    
    mgr = SimManager(
        exec_mode=madrona_escape_room.madrona.ExecMode.CPU,
        gpu_id=0,
        num_worlds=4,
        rand_seed=42,
        enable_batch_renderer=False,
        auto_reset=False  # Manual control over resets
    )
    
    yield mgr


@pytest.fixture(autouse=True)
def per_test_recording_handler(request, test_manager):
    """Autouse fixture that handles per-test recording and tracing for module-scoped test_manager"""
    # Check for debug flags
    record_actions = request.config.getoption("--record-actions")
    trace_trajectories = request.config.getoption("--trace-trajectories")
    
    # Only run if we have either recording or tracing enabled
    if record_actions or trace_trajectories:
        # Create test-specific files
        test_name = request.node.nodeid.replace("::", "__")
        test_filename = test_name.split('/')[-1] if '/' in test_name else test_name
        
        # Create output directory
        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)
        base_path = output_dir / f"{test_filename}_actions"  # No extension - DebugSession adds them
        
        
        # Use the master context manager
        debug_session = test_manager.debug_session(
            base_path=base_path,
            enable_recording=record_actions,
            enable_tracing=trace_trajectories,
            seed=42
        )
        with debug_session:
            yield
            
        # Print debug info after context exits using actual paths from DebugSession
        print(f"\n{'='*60}")
        if record_actions and debug_session.recording_path:
            print(f"Actions recorded: {debug_session.recording_path}")
            print(f"Worlds: 4, Agents: 1")
            print(f"To visualize: ./build/viewer --num-worlds 4 --replay {debug_session.recording_path}")
            
        if trace_trajectories and debug_session.trajectory_path:
            print(f"Trajectory logged: {debug_session.trajectory_path}")
            
        print(f"{'='*60}\n")
        
        # Launch viewer if requested
        if request.config.getoption("--visualize") and record_actions and debug_session.recording_path:
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print(f"Launching viewer...")
                subprocess.run([str(viewer_path), "--num-worlds", "4", "--replay", str(debug_session.recording_path)])
            else:
                print(f"Viewer not found at {viewer_path}")
    else:
        yield