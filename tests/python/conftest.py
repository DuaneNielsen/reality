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
    """Create a CPU SimManager with optional native recording"""
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
    
    # Check if recording is enabled
    if request.config.getoption("--record-actions"):
        # Use nodeid and replace :: with __ for filesystem compatibility
        test_name = request.node.nodeid.replace("::", "__")
        test_filename = test_name.split('/')[-1] if '/' in test_name else test_name
        
        # Create output directory
        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)
        action_path = output_dir / f"{test_filename}_actions.bin"
        
        # Start native recording
        mgr.start_recording(str(action_path), seed=42)
        
        yield mgr
        
        # Stop recording after test
        mgr.stop_recording()
        
        print(f"\n{'='*60}")
        print(f"Actions recorded: {action_path}")
        print(f"Worlds: 4, Agents: 1, Steps: {mgr.get_step_count() if hasattr(mgr, 'get_step_count') else 'N/A'}")
        print(f"To visualize: ./build/viewer 4 --cpu --replay {action_path}")
        print(f"{'='*60}\n")
        
        # Launch viewer if requested
        if request.config.getoption("--visualize"):
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print(f"Launching viewer...")
                subprocess.run([str(viewer_path), "4", "--cpu", "--replay", str(action_path)])
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
    """Autouse fixture that handles per-test native recording for module-scoped test_manager"""
    # Only run if we have recording enabled
    if request.config.getoption("--record-actions"):
        # Create test-specific recording file
        test_name = request.node.nodeid.replace("::", "__")
        test_filename = test_name.split('/')[-1] if '/' in test_name else test_name
        
        # Create output directory
        output_dir = Path("test_recordings")
        output_dir.mkdir(exist_ok=True)
        action_path = output_dir / f"{test_filename}_actions.bin"
        
        # Start native recording
        test_manager.start_recording(str(action_path), seed=42)
        
        yield
        
        # Stop recording after test completes
        test_manager.stop_recording()
        
        print(f"\n{'='*60}")
        print(f"Actions recorded: {action_path}")
        print(f"Worlds: 4, Agents: 1")
        print(f"To visualize: ./build/viewer 4 --cpu --replay {action_path}")
        print(f"{'='*60}\n")
        
        # Launch viewer if requested
        if request.config.getoption("--visualize"):
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print(f"Launching viewer...")
                subprocess.run([str(viewer_path), "4", "--cpu", "--replay", str(action_path)])
            else:
                print(f"Viewer not found at {viewer_path}")
    else:
        yield