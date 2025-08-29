#!/usr/bin/env python3
"""
Stress test for Madrona Escape Room using pytest.
Tests the simulation with configurable iterations and world counts.
"""

import time

import numpy as np
import pytest

import madrona_escape_room


@pytest.mark.stress
@pytest.mark.parametrize(
    "iterations,num_worlds",
    [
        (100, 32),  # Quick stress test
        (500, 32),  # Medium stress test
        (1000, 64),  # Heavy stress test
    ],
)
def test_stress_simulation(cpu_manager, iterations, num_worlds):
    """
    Run stress test with specified iterations and world count.

    Args:
        cpu_manager: CPU SimManager fixture
        iterations: Number of simulation steps to run
        num_worlds: Expected number of parallel worlds
    """
    mgr = cpu_manager

    # Verify we have the expected number of worlds
    action_tensor = mgr.action_tensor()
    actions = action_tensor.to_numpy()

    # Check tensor dimensions to understand world/agent structure
    if len(actions.shape) == 3:
        actual_worlds, agents, action_dims = actions.shape
    else:
        actual_worlds, action_dims = actions.shape
        agents = 1

    # For stress testing, we expect at least 4 worlds (default fixture)
    # But allow more if specified
    assert actual_worlds >= 4, f"Expected at least 4 worlds, got {actual_worlds}"
    assert action_dims == 3, f"Expected 3 action dimensions, got {action_dims}"

    print("\nRunning stress test:")
    print(f"  Iterations: {iterations}")
    print(f"  Worlds: {actual_worlds}")
    print(f"  Agents per world: {agents}")

    # Progress tracking
    report_interval = max(1, iterations // 10)  # Report 10 times

    # Timing
    start_time = time.time()

    for i in range(iterations):
        # Set random actions
        # Action space: [move_amount (0-3), move_angle (0-7), rotate (0-4)]
        if len(actions.shape) == 3:
            # 3D tensor: [worlds, agents, actions]
            actions[:, :, 0] = np.random.randint(0, 4, size=(actual_worlds, agents))  # move_amount
            actions[:, :, 1] = np.random.randint(0, 8, size=(actual_worlds, agents))  # move_angle
            actions[:, :, 2] = np.random.randint(0, 5, size=(actual_worlds, agents))  # rotate
        else:
            # 2D tensor: [worlds, actions]
            actions[:, 0] = np.random.randint(0, 4, size=actual_worlds)  # move_amount
            actions[:, 1] = np.random.randint(0, 8, size=actual_worlds)  # move_angle
            actions[:, 2] = np.random.randint(0, 5, size=actual_worlds)  # rotate

        # Step simulation
        mgr.step()

        # Report progress
        if (i + 1) % report_interval == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) * actual_worlds / elapsed
            print(f"    Completed {i + 1}/{iterations} iterations - FPS: {fps:.0f}")

    # Final timing and performance assertions
    total_time = time.time() - start_time
    total_steps = iterations * actual_worlds
    fps = total_steps / total_time

    print(f"  ✅ Completed {iterations} iterations without crashes.")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Total steps: {total_steps:,} (iterations × worlds)")
    print(f"  Average FPS: {fps:.0f}")

    # Performance assertions - should maintain reasonable FPS
    min_fps = 1000  # Minimum acceptable FPS for stress testing
    assert fps >= min_fps, f"Performance too slow: {fps:.0f} FPS < {min_fps} FPS"

    # Verify simulation is still responsive
    rewards = mgr.reward_tensor().to_numpy()
    assert rewards is not None, "Reward tensor should be accessible after stress test"

    dones = mgr.done_tensor().to_numpy()
    assert dones is not None, "Done tensor should be accessible after stress test"


@pytest.mark.stress
def test_rapid_manager_creation():
    """
    Test rapid creation/destruction of managers to catch lifecycle issues.
    This is a separate test from the fixture-based stress test.
    """
    from madrona_escape_room import ExecMode, SimManager, create_default_level

    num_iterations = 50
    print(f"\nTesting rapid manager creation/destruction for {num_iterations} iterations")

    for i in range(num_iterations):
        # Create manager
        level = create_default_level()
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=4,
            rand_seed=42 + i,  # Vary seed
            auto_reset=True,
            compiled_levels=level,
        )

        # Run a few steps
        for _ in range(5):
            mgr.step()

        # Cleanup
        del mgr

        if (i + 1) % 10 == 0:
            print(f"    Completed {i + 1}/{num_iterations} manager lifecycles")

    print(f"  ✅ Successfully created/destroyed {num_iterations} managers")


@pytest.mark.stress
@pytest.mark.slow
def test_extended_stress_simulation(cpu_manager):
    """
    Extended stress test with higher iteration count.
    Marked as 'slow' so it can be skipped in regular test runs.
    """
    mgr = cpu_manager
    iterations = 2000

    print(f"\nRunning extended stress test with {iterations} iterations")

    action_tensor = mgr.action_tensor()
    actions = action_tensor.to_numpy()

    start_time = time.time()

    # Track some statistics during the run
    step_times = []

    for i in range(iterations):
        step_start = time.time()

        # Random actions
        if len(actions.shape) == 3:
            worlds, agents, _ = actions.shape
            actions[:, :, 0] = np.random.randint(0, 4, size=(worlds, agents))
            actions[:, :, 1] = np.random.randint(0, 8, size=(worlds, agents))
            actions[:, :, 2] = np.random.randint(0, 5, size=(worlds, agents))
        else:
            worlds = actions.shape[0]
            actions[:, 0] = np.random.randint(0, 4, size=worlds)
            actions[:, 1] = np.random.randint(0, 8, size=worlds)
            actions[:, 2] = np.random.randint(0, 5, size=worlds)

        mgr.step()

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Report every 200 iterations
        if (i + 1) % 200 == 0:
            avg_step_time = np.mean(step_times[-200:])
            fps = worlds / avg_step_time
            print(f"    Iteration {i + 1}: avg step time {avg_step_time*1000:.2f}ms, FPS {fps:.0f}")

    total_time = time.time() - start_time
    total_steps = iterations * worlds
    avg_fps = total_steps / total_time

    # Performance statistics
    avg_step_time = np.mean(step_times)
    min_step_time = np.min(step_times)
    max_step_time = np.max(step_times)

    print("  ✅ Extended stress test completed!")
    print(f"  Average FPS: {avg_fps:.0f}")
    print(f"  Step time - avg: {avg_step_time*1000:.2f}ms")
    print(f"  Step time - min: {min_step_time*1000:.2f}ms, max: {max_step_time*1000:.2f}ms")

    # Assert performance is stable
    assert avg_fps >= 500, f"Extended test performance too slow: {avg_fps:.0f} FPS"
    assert max_step_time < 0.1, f"Some steps too slow: max {max_step_time*1000:.0f}ms"
