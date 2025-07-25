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


class RecordingWrapper:
    """Wrapper that adds action recording to a SimManager"""
    
    def __init__(self, manager, test_name, output_dir="test_recordings"):
        self.mgr = manager
        self.test_name = test_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get manager info
        self.num_worlds = manager.action_tensor().to_torch().shape[0]
        self.num_agents = manager.action_tensor().to_torch().shape[1]
        self.action_dims = manager.action_tensor().to_torch().shape[2]
        
        self.actions = []
        self.step_count = 0
        
        # Store original step method
        self._original_step = manager.step
        
        # Output paths
        self.output_path = self.output_dir / f"{test_name}_actions.bin"
        
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
            
        # Stack all actions: shape is [steps, worlds, agents, 3]
        all_actions = np.stack(self.actions)
        
        # The viewer expects data in order: [step][world][agent][3 components]
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
        test_name = request.node.name
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
    
    # Check if recording is enabled
    if request.config.getoption("--record-actions"):
        test_name = request.node.name
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