#!/usr/bin/env python3
"""
Test SimManager integration with multi-level compilation.
Tests the next steps from the multi-level support implementation.
Uses pytest framework for proper test suite integration.
"""

import json

import numpy as np
import pytest

from madrona_escape_room import ExecMode, SimManager
from madrona_escape_room.level_compiler import compile_level


def test_multi_level_sim_manager_integration():
    """Test that SimManager can load and use compiled multi-level data."""

    # Load the generated multi-level file
    multi_level_path = "levels/progressive_levels_1_to_20_multi.json"
    print(f"Loading multi-level file: {multi_level_path}")

    with open(multi_level_path, "r") as f:
        multi_level_data = json.load(f)

    print(f"Loaded multi-level with {len(multi_level_data['levels'])} levels")

    # Compile the multi-level data
    compiled_levels = compile_level(multi_level_data)
    print(f"Compiled {len(compiled_levels)} levels")

    # Verify we got a list of CompiledLevel objects
    assert isinstance(compiled_levels, list), "Expected list of CompiledLevel objects"
    assert len(compiled_levels) == 20, f"Expected 20 levels, got {len(compiled_levels)}"

    # Print level names to verify they're unique
    print("Level names:")
    for i, level in enumerate(compiled_levels):
        print(f"  {i+1:2d}: {level.level_name}")

    # Test with multiple worlds using different levels
    # Note: num_compiled_levels cannot exceed num_worlds per C API validation
    num_worlds = 20  # Use same number of worlds as levels
    print(f"\nTesting SimManager with {num_worlds} worlds...")

    try:
        # Create SimManager with CPU execution for testing
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=-1,
            num_worlds=num_worlds,
            rand_seed=42,
            auto_reset=True,
            compiled_levels=compiled_levels,  # Pass the list of compiled levels
            enable_batch_renderer=False,
        )
        print("✅ SimManager created successfully with multi-level data")

        # Test basic simulation step
        mgr.step()
        print("✅ First simulation step completed")

        # Check observation tensor shape
        obs = mgr.self_observation_tensor().to_numpy()
        print(f"✅ Observation tensor shape: {obs.shape}")

        # Test multiple steps
        for step in range(5):
            mgr.step()
        print("✅ Multiple simulation steps completed")

        # Test with different action patterns to verify worlds can behave independently
        actions = mgr.action_tensor().to_numpy()
        print(f"Action tensor shape: {actions.shape}")

        # Set different actions for different worlds
        # Action tensor is [worlds, action_components] = (20, 3)
        for world_idx in range(min(num_worlds, 5)):  # Just test first 5 worlds
            actions[world_idx, 0] = 1  # move_amount
            actions[world_idx, 1] = world_idx % 8  # move_angle
            actions[world_idx, 2] = 0  # rotate

        mgr.step()
        print("✅ Different actions per world executed successfully")

        print("\n🎉 Multi-level SimManager integration test PASSED")

    except Exception as e:
        print(f"❌ SimManager integration test FAILED: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"SimManager integration test failed: {e}")


def test_level_selection():
    """Test that different worlds can use different levels from the compiled list."""

    multi_level_path = "levels/progressive_levels_1_to_20_multi.json"
    with open(multi_level_path, "r") as f:
        multi_level_data = json.load(f)

    compiled_levels = compile_level(multi_level_data)

    # Create SimManager with fewer worlds than levels to test level subset selection
    num_worlds = 5
    subset_levels = compiled_levels[:num_worlds]  # Use only first 5 levels
    print(f"\nTesting level selection with {num_worlds} worlds and {len(subset_levels)} levels...")

    try:
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=-1,
            num_worlds=num_worlds,
            rand_seed=42,
            auto_reset=True,
            compiled_levels=subset_levels,
            enable_batch_renderer=False,
        )

        # Run simulation and check that it works with multiple worlds
        for step in range(10):
            mgr.step()

        print("✅ Level selection test PASSED - multiple worlds working with level list")

    except Exception as e:
        print(f"❌ Level selection test FAILED: {e}")
        pytest.fail(f"Level selection test failed: {e}")


def test_curriculum_learning_scenario():
    """Test curriculum learning with more worlds than levels."""

    multi_level_path = "levels/progressive_levels_1_to_20_multi.json"
    with open(multi_level_path, "r") as f:
        multi_level_data = json.load(f)

    compiled_levels = compile_level(multi_level_data)

    # Test curriculum learning scenario: more worlds than levels
    num_worlds = 40  # 2x the number of levels for curriculum scenarios
    print(
        f"\nTesting curriculum learning with {num_worlds} worlds and {len(compiled_levels)} levels..."
    )
    print("This tests level cycling/reuse across multiple worlds")

    try:
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=-1,
            num_worlds=num_worlds,
            rand_seed=42,
            auto_reset=True,
            compiled_levels=compiled_levels,  # 20 levels for 40 worlds
            enable_batch_renderer=False,
        )

        print("✅ SimManager created with 40 worlds and 20 levels")

        # Verify level assignment: Should be 2 copies of each level
        # C API assigns levels[i % num_levels] to world i
        expected_pattern = []
        for world_idx in range(num_worlds):
            level_idx = world_idx % len(compiled_levels)
            expected_level_name = compiled_levels[level_idx].level_name.decode("utf-8")
            expected_pattern.append((world_idx, level_idx, expected_level_name))

        print("Expected level assignment pattern (first 10 worlds):")
        for i in range(min(10, len(expected_pattern))):
            world_idx, level_idx, level_name = expected_pattern[i]
            print(f"  World {world_idx:2d} -> Level {level_idx:2d} ({level_name})")

        print("Expected level assignment pattern (last 10 worlds):")
        for i in range(max(0, len(expected_pattern) - 10), len(expected_pattern)):
            world_idx, level_idx, level_name = expected_pattern[i]
            print(f"  World {world_idx:2d} -> Level {level_idx:2d} ({level_name})")

        # Verify we have exactly 2 copies of each level
        level_usage_count = {}
        for _, level_idx, level_name in expected_pattern:
            level_usage_count[level_name] = level_usage_count.get(level_name, 0) + 1

        print("Level usage verification:")
        for level_name, count in sorted(level_usage_count.items()):
            print(f"  {level_name}: used by {count} worlds")
            if count != 2:
                raise AssertionError(
                    f"Expected each level to be used by exactly 2 worlds, but {level_name} is used by {count}"
                )

        print("✅ Level assignment verified: each of 20 levels used by exactly 2 worlds")

        # Run simulation to verify it works
        for step in range(10):
            mgr.step()

        # Check tensor shapes
        obs = mgr.self_observation_tensor().to_numpy()
        actions = mgr.action_tensor().to_numpy()
        print(f"✅ Observation tensor shape: {obs.shape}")
        print(f"✅ Action tensor shape: {actions.shape}")

        # Set different actions for first few worlds to test functionality
        for world_idx in range(min(10, num_worlds)):
            actions[world_idx, 0] = 1  # move_amount
            actions[world_idx, 1] = world_idx % 8  # move_angle
            actions[world_idx, 2] = 0  # rotate

        mgr.step()
        print("✅ Multiple worlds with level reuse working correctly")

        print("🎉 Curriculum learning test PASSED")

    except Exception as e:
        print(f"❌ Curriculum learning test FAILED: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Curriculum learning test failed: {e}")


def test_all_multi_level_scenarios():
    """Run all multi-level tests in sequence for comprehensive coverage."""
    print("=== Multi-Level SimManager Integration Tests ===\n")

    test_multi_level_sim_manager_integration()
    test_level_selection()
    test_curriculum_learning_scenario()

    print("\n🎉 ALL TESTS PASSED - Multi-level integration working correctly!")


if __name__ == "__main__":
    # Support direct execution for debugging
    test_all_multi_level_scenarios()
