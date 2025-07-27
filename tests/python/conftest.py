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


class RecordingWrapper:
    """Wrapper that adds action recording to a SimManager"""
    
    def __init__(self, manager, test_name, output_dir="test_recordings"):
        self.mgr = manager
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get manager info
        self.num_worlds = manager.action_tensor().to_torch().shape[0]
        self.num_agents = 1  # Single agent per world
        self.action_dims = manager.action_tensor().to_torch().shape[1]
        
        self.actions = []
        self.step_count = 0
        
        # Store original step method
        self._original_step = manager.step
        
        # Output paths - extract just the filename part from test_name
        # test_name might be like "tests/python/test_reward_system.py__test_forward_movement_reward"
        test_filename = test_name.split('/')[-1] if '/' in test_name else test_name
        self.output_path = self.output_dir / f"{test_filename}_actions.bin"
        
    def step(self):
        """Step simulation and record actions"""
        # Record current actions before stepping
        actions = self.mgr.action_tensor().to_torch()
        self.actions.append(actions.cpu().numpy().astype(np.int32))
        self.step_count += 1
        
        # Call original step
        self._original_step()
        
    def save_recording(self):
        """Save recorded actions in viewer format"""
        if self.step_count == 0:
            return None
            
        # Stack all actions: shape is [steps, worlds, 3]
        all_actions = np.stack(self.actions)
        
        # The viewer expects data in order: [step][world][3 components]
        # numpy.flatten() uses C-order (row-major) by default, which is correct
        # for this layout. Just ensure we're saving as int32.
        flat_actions = all_actions.astype(np.int32).flatten()
        
        # Save binary file
        flat_actions.tofile(str(self.output_path))
        
        print(f"\n{'='*60}")
        print(f"Actions recorded: {self.output_path}")
        print(f"Worlds: {self.num_worlds}, Agents: {self.num_agents}, Steps: {self.step_count}")
        print(f"To visualize: ./build/viewer {self.num_worlds} CPU {self.output_path}")
        print(f"{'='*60}\n")
        
        return str(self.output_path)
        
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped manager"""
        return getattr(self.mgr, name)


@pytest.fixture(scope="function")
def cpu_manager(request):
    """Create a CPU SimManager with optional recording"""
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
        wrapper = RecordingWrapper(mgr, test_name)
        yield wrapper
        
        # Save recording after test
        action_path = wrapper.save_recording()
        
        # Launch viewer if requested
        if request.config.getoption("--visualize") and action_path:
            viewer_path = Path("build/viewer")
            if viewer_path.exists():
                print(f"Launching viewer...")
                subprocess.run([str(viewer_path), str(wrapper.num_worlds), "CPU", action_path])
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
    """Create a test manager for reward system tests with optional recording"""
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
    
    # For module-scoped fixture with per-test recording
    if request.config.getoption("--record-actions"):
        # Create wrapper that we'll update per test
        wrapper = RecordingWrapper(mgr, "placeholder")
        wrapper._current_test_nodeid = None
        
        yield wrapper
    else:
        yield mgr


@pytest.fixture(autouse=True)
def per_test_recording_handler(request, test_manager):
    """Autouse fixture that handles per-test recording for module-scoped test_manager"""
    # Only run if we have recording enabled and test_manager is a RecordingWrapper
    if hasattr(test_manager, '_current_test_nodeid') and request.config.getoption("--record-actions"):
        # Set the current test name at test start
        test_manager._current_test_nodeid = request.node.nodeid
        
        # Update wrapper with correct test name
        current_test = request.node.nodeid.replace("::", "__")
        test_filename = current_test.split('/')[-1] if '/' in current_test else current_test
        test_manager.test_name = test_filename
        test_manager.output_path = test_manager.output_dir / f"{test_filename}_actions.bin"
        
        # Reset recording state for this test
        test_manager.actions = []
        test_manager.step_count = 0
        
        yield
        
        # Save recording after test completes
        if test_manager.step_count > 0:
            action_path = test_manager.save_recording()
            
            # Launch viewer if requested
            if request.config.getoption("--visualize") and action_path:
                viewer_path = Path("build/viewer")
                if viewer_path.exists():
                    print(f"Launching viewer...")
                    subprocess.run([str(viewer_path), str(test_manager.num_worlds), "CPU", action_path])
                else:
                    print(f"Viewer not found at {viewer_path}")
    else:
        yield