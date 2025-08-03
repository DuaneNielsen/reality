#!/usr/bin/env python3
"""
MCP Python REPL Initialization for Madrona Escape Room

This script sets up the Python environment for interactive development
through the Model Context Protocol (MCP) Python REPL server.

It provides helper functions for creating SimManager instances, testing,
benchmarking, and debugging the Madrona simulation environment.

Usage:
    execute_python('exec(open("scripts/mcp_python_repl_init.py").read())')

See: docs/development/tools/PYTHON_REPL_MCP_SETUP.md
"""

import sys
import os

# Add project to path (get parent directory of scripts/)
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

print(f"Added {project_path} to Python path")

# Imports
try:
    from madrona_escape_room import SimManager
    import numpy as np
    print("✓ Madrona imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you've built the project: make -C build -j$(nproc) -s")
    print("And installed Python package: uv pip install -e .")

# Helper functions
def create_cpu_manager(num_worlds=4, num_steps=1000, seed=5):
    """
    Create a CPU-based SimManager for interactive development.
    
    Args:
        num_worlds: Number of parallel worlds to simulate
        num_steps: Maximum steps per episode  
        seed: Random seed for reproducibility
    
    Returns:
        SimManager instance configured for CPU execution
    """
    return SimManager(
        num_worlds=num_worlds,
        num_steps=num_steps,
        gpu_id=-1,  # CPU mode
        seed=seed,
        debug_compile=False
    )

def create_gpu_manager(num_worlds=1024, num_steps=1000, gpu_id=0, seed=5):
    """
    Create a GPU-based SimManager for performance testing.
    
    Args:
        num_worlds: Number of parallel worlds to simulate
        num_steps: Maximum steps per episode
        gpu_id: CUDA device ID (0-based)
        seed: Random seed for reproducibility
        
    Returns:
        SimManager instance configured for GPU execution
    """
    return SimManager(
        num_worlds=num_worlds,
        num_steps=num_steps,
        gpu_id=gpu_id,
        seed=seed,
        debug_compile=False
    )

def quick_test(mgr):
    """
    Run a quick test of a SimManager instance.
    
    Args:
        mgr: SimManager instance to test
        
    Returns:
        The same manager instance for chaining
    """
    mgr.reset()
    print(f"Manager: {mgr.num_worlds} worlds, {mgr.num_steps} max steps")
    print(f"Observations shape: {mgr.observations.shape}")
    print(f"Actions shape: {mgr.actions.shape}")
    print(f"Rewards shape: {mgr.rewards.shape}")
    print(f"Done shape: {mgr.done.shape}")
    
    # Run one step with zero actions
    mgr.actions[:] = 0
    mgr.step()
    print(f"First step reward: {mgr.rewards[0, 0]:.4f}")
    print(f"Episodes done: {mgr.done.sum()}/{mgr.done.size}")
    
    return mgr

def inspect_world(mgr, world_id=0, agent_id=0):
    """
    Inspect the state of a specific world and agent.
    
    Args:
        mgr: SimManager instance
        world_id: World index to inspect
        agent_id: Agent index within the world
    """
    if world_id >= mgr.num_worlds:
        print(f"Error: world_id {world_id} >= num_worlds {mgr.num_worlds}")
        return
        
    print(f"\n=== World {world_id}, Agent {agent_id} ===")
    obs = mgr.observations[world_id, agent_id]
    print(f"Position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
    print(f"Max Y reached: {obs[3]:.2f}")  
    print(f"Rotation: {obs[4]:.2f} rad")
    print(f"Done: {mgr.done[world_id, agent_id]}")
    print(f"Reward: {mgr.rewards[world_id, agent_id]:.4f}")

def run_episode(mgr, max_steps=None, world_id=0, verbose=True):
    """
    Run a complete episode for a specific world.
    
    Args:
        mgr: SimManager instance
        max_steps: Maximum steps to run (None = run until done)
        world_id: World to focus on for reporting
        verbose: Print progress updates
        
    Returns:
        dict: Episode statistics
    """
    mgr.reset()
    
    step_count = 0
    total_reward = 0.0
    max_steps = max_steps or mgr.num_steps
    
    if verbose:
        print(f"Starting episode for world {world_id}...")
    
    while not mgr.done[world_id, 0] and step_count < max_steps:
        # Set random actions (you can customize this)
        mgr.actions[:] = np.random.uniform(-1, 1, mgr.actions.shape)
        mgr.step()
        
        step_count += 1
        total_reward += mgr.rewards[world_id, 0]
        
        if verbose and step_count % 50 == 0:
            print(f"Step {step_count}: reward = {mgr.rewards[world_id, 0]:.4f}, "
                  f"total = {total_reward:.4f}")
    
    stats = {
        'steps': step_count,
        'total_reward': total_reward,
        'final_position': mgr.observations[world_id, 0, :3].copy(),
        'max_y_reached': mgr.observations[world_id, 0, 3],
        'episode_done': bool(mgr.done[world_id, 0])
    }
    
    if verbose:
        print(f"Episode finished: {step_count} steps, {total_reward:.4f} total reward")
        print(f"Final position: {stats['final_position']}")
        print(f"Max Y reached: {stats['max_y_reached']:.2f}")
    
    return stats

def benchmark_performance(num_worlds=1024, num_steps=100, gpu_id=-1):
    """
    Benchmark simulation performance.
    
    Args:
        num_worlds: Number of worlds to simulate
        num_steps: Number of steps to run
        gpu_id: GPU device (-1 for CPU)
        
    Returns:
        dict: Performance statistics
    """
    import time
    
    print(f"Benchmarking {'GPU' if gpu_id >= 0 else 'CPU'} performance...")
    print(f"Worlds: {num_worlds}, Steps: {num_steps}")
    
    if gpu_id >= 0:
        mgr = create_gpu_manager(num_worlds=num_worlds, gpu_id=gpu_id)
    else:
        mgr = create_cpu_manager(num_worlds=num_worlds)
    
    # Warm up
    mgr.reset()
    mgr.step()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_steps):
        mgr.actions[:] = np.random.uniform(-1, 1, mgr.actions.shape)
        mgr.step()
    end_time = time.time()
    
    elapsed = end_time - start_time
    total_sims = num_worlds * num_steps
    sims_per_sec = total_sims / elapsed
    
    stats = {
        'elapsed_time': elapsed,
        'steps_per_second': num_steps / elapsed,
        'simulations_per_second': sims_per_sec,
        'worlds': num_worlds,
        'steps': num_steps
    }
    
    print(f"Results:")
    print(f"  Elapsed time: {elapsed:.3f}s")
    print(f"  Steps/sec: {stats['steps_per_second']:.1f}")
    print(f"  Simulations/sec: {sims_per_sec:.0f}")
    
    return stats

# Print available functions
print("\n" + "="*60)
print("Madrona Python REPL MCP Helper Functions:")
print("="*60)
print("Manager Creation:")
print("  create_cpu_manager(num_worlds=4, num_steps=1000, seed=5)")
print("  create_gpu_manager(num_worlds=1024, gpu_id=0, seed=5)")
print()
print("Testing & Inspection:")  
print("  quick_test(mgr)                    # Basic functionality test")
print("  inspect_world(mgr, world_id=0)     # Detailed world state")
print("  run_episode(mgr, max_steps=None)   # Complete episode with stats")
print("  benchmark_performance(num_worlds)  # Performance testing")
print()
print("Quick Start Example:")
print("  mgr = create_cpu_manager(num_worlds=2)")
print("  quick_test(mgr)")
print("  inspect_world(mgr, 0)")
print()
print("Documentation: docs/development/tools/PYTHON_REPL_MCP_SETUP.md")
print("="*60)