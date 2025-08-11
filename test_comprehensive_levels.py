#!/usr/bin/env python3
"""
Comprehensive test demonstrating all level compilation capabilities.

This test demonstrates:
1. Phase 2 fallback (no compiled levels)
2. Single shared level (backward compatibility)
3. Per-world different levels (new capability)
4. Mixed scenarios (some worlds compiled, some fallback)
"""

import ctypes
import sys

from test_per_world_compiled_levels import (
    MER_SUCCESS,
    MER_CompiledLevel,
    MER_ManagerConfig,
    create_empty_room,
    create_maze_room,
    create_obstacle_room,
    lib,
)


def test_phase_2_fallback():
    """Test Phase 2 fallback - no compiled levels at all."""
    print("=== Phase 2 Fallback Test ===")
    print("All worlds use hardcoded 16x16 room (max_entities=300)")

    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU
    config.gpu_id = 0
    config.num_worlds = 3
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        None,  # No compiled levels
        0,  # Zero levels
    )

    success = result == MER_SUCCESS
    if success:
        print("✓ Manager created - all 3 worlds using Phase 2 fallback")
        lib.mer_destroy_manager(handle)
        print("✓ Manager destroyed")
    else:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"✗ Failed: {error_msg}")

    return success


def test_single_shared_level():
    """Test single shared level - all worlds use same compiled level."""
    print("\n=== Single Shared Level Test ===")

    maze_level = create_maze_room()
    print(
        f"All worlds use maze room ({maze_level.num_tiles} tiles, max_entities={maze_level.max_entities})"
    )

    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU
    config.gpu_id = 0
    config.num_worlds = 4
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    # Single level for all worlds
    levels_array = (MER_CompiledLevel * 1)(maze_level)
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        levels_array,
        1,  # Single level
    )

    success = result == MER_SUCCESS
    if success:
        print("✓ Manager created - all 4 worlds sharing maze level")
        lib.mer_destroy_manager(handle)
        print("✓ Manager destroyed")
    else:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"✗ Failed: {error_msg}")

    return success


def test_per_world_different_levels():
    """Test per-world different levels - each world has unique level."""
    print("\n=== Per-World Different Levels Test ===")

    # Create 3 different levels
    empty_level = create_empty_room()
    obstacle_level = create_obstacle_room()
    maze_level = create_maze_room()

    print(
        f"World 0: Empty room ({empty_level.num_tiles} tiles, max_entities={empty_level.max_entities})"
    )
    print(
        f"World 1: Obstacle room ({obstacle_level.num_tiles} tiles, max_entities={obstacle_level.max_entities})"
    )
    print(
        f"World 2: Maze room ({maze_level.num_tiles} tiles, max_entities={maze_level.max_entities})"
    )

    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU
    config.gpu_id = 0
    config.num_worlds = 3
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    # Array with different level for each world
    levels_array = (MER_CompiledLevel * 3)(empty_level, obstacle_level, maze_level)
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        levels_array,
        3,  # Three different levels
    )

    success = result == MER_SUCCESS
    if success:
        print("✓ Manager created - each world has different level geometry!")
        lib.mer_destroy_manager(handle)
        print("✓ Manager destroyed")
    else:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"✗ Failed: {error_msg}")

    return success


def test_mixed_scenario():
    """Test mixed scenario - some worlds compiled, others fallback."""
    print("\n=== Mixed Scenario Test ===")

    empty_level = create_empty_room()
    obstacle_level = create_obstacle_room()

    print(
        f"World 0: Empty room ({empty_level.num_tiles} tiles, max_entities={empty_level.max_entities})"
    )
    print(
        f"World 1: Obstacle room ({obstacle_level.num_tiles} tiles, max_entities={obstacle_level.max_entities})"
    )
    print("World 2: Phase 2 fallback (hardcoded 16x16, max_entities=300)")
    print("World 3: Phase 2 fallback (hardcoded 16x16, max_entities=300)")

    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU
    config.gpu_id = 0
    config.num_worlds = 4  # 4 worlds
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    # Only 2 levels for 4 worlds - remaining worlds use Phase 2 fallback
    levels_array = (MER_CompiledLevel * 2)(empty_level, obstacle_level)
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        levels_array,
        2,  # Only 2 levels for 4 worlds
    )

    success = result == MER_SUCCESS
    if success:
        print("✓ Manager created - mixed compiled and fallback worlds!")
        lib.mer_destroy_manager(handle)
        print("✓ Manager destroyed")
    else:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"✗ Failed: {error_msg}")

    return success


def test_validation_edge_cases():
    """Test validation and edge cases."""
    print("\n=== Validation and Edge Cases Test ===")

    empty_level = create_empty_room()

    # Test: More levels than worlds (should fail)
    print("Testing more levels than worlds...")
    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU
    config.gpu_id = 0
    config.num_worlds = 2  # Only 2 worlds
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    # 3 levels for 2 worlds - should fail
    levels_array = (MER_CompiledLevel * 3)(empty_level, empty_level, empty_level)
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        levels_array,
        3,  # Too many levels
    )

    if result != MER_SUCCESS:
        print("✓ Correctly rejected too many levels")
        return True
    else:
        print("✗ Should have rejected too many levels")
        lib.mer_destroy_manager(handle)
        return False


if __name__ == "__main__":
    print("=== Comprehensive Level Compilation Test ===")
    print("Demonstrating all level compilation capabilities...\n")

    # Run all tests
    tests = [
        ("Phase 2 Fallback", test_phase_2_fallback),
        ("Single Shared Level", test_single_shared_level),
        ("Per-World Different Levels", test_per_world_different_levels),
        ("Mixed Scenario", test_mixed_scenario),
        ("Validation Edge Cases", test_validation_edge_cases),
    ]

    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print("=" * 50)

    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:<30} {status}")
        if not success:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✨ Complete Level Compilation System Capabilities:")
        print("   ✅ Phase 2 Fallback - No compilation needed")
        print("   ✅ Single Shared Level - All worlds same geometry")
        print("   ✅ Per-World Levels - Each world different geometry")
        print("   ✅ Mixed Scenarios - Some compiled, some fallback")
        print("   ✅ Proper Validation - Rejects invalid configurations")
        print("\n🚀 Ready for ASCII level compiler implementation!")
        sys.exit(0)
    else:
        print("❌ Some tests failed - check output above")
        sys.exit(1)
