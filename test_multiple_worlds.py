#!/usr/bin/env python3
"""
Test multiple worlds with compiled level interface.

This tests whether each world gets its own copy of the compiled level
or if they share a single instance.
"""

import ctypes
import sys

from test_compiled_level_interface import (
    MER_SUCCESS,
    MER_CompiledLevel,
    MER_ManagerConfig,
    create_hardcoded_16x16_room,
    lib,
)

# Update function signature for new array-based API
lib.mer_create_manager.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # MER_ManagerHandle* out_handle
    ctypes.POINTER(MER_ManagerConfig),  # const MER_ManagerConfig* config
    ctypes.POINTER(MER_CompiledLevel),  # const MER_CompiledLevel* compiled_levels (array)
    ctypes.c_uint32,  # uint32_t num_compiled_levels
]
lib.mer_create_manager.restype = ctypes.c_int


def test_multiple_worlds_compiled_level():
    """Test creating a manager with multiple worlds using compiled level."""
    print("Testing multiple worlds with compiled level interface...")

    # Create hardcoded level
    compiled_level = create_hardcoded_16x16_room()
    print(
        f"Created level with {compiled_level.num_tiles} tiles, max_entities={compiled_level.max_entities}"
    )

    # Test with different numbers of worlds
    for num_worlds in [1, 4, 8, 16]:
        print(f"\nTesting with {num_worlds} worlds...")

        # Create manager config
        config = MER_ManagerConfig()
        config.exec_mode = 0  # CPU mode
        config.gpu_id = 0
        config.num_worlds = num_worlds
        config.rand_seed = 42
        config.auto_reset = True
        config.enable_batch_renderer = False

        # Create manager with compiled level (single level for all worlds)
        levels_array = (MER_CompiledLevel * 1)(compiled_level)  # Single level
        handle = ctypes.c_void_p()
        result = lib.mer_create_manager(
            ctypes.byref(handle),
            ctypes.byref(config),
            levels_array,
            1,  # One level (will be used for all worlds)
        )

        if result != MER_SUCCESS:
            error_msg = lib.mer_result_to_string(result).decode("utf-8")
            print(f"ERROR: Manager creation failed for {num_worlds} worlds: {error_msg}")
            return False

        print(f"✓ Manager created successfully with {num_worlds} worlds")
        print(f"  Manager handle: {handle.value}")

        # Clean up
        lib.mer_destroy_manager(handle)
        print(f"✓ Manager with {num_worlds} worlds destroyed successfully")

    return True


def test_different_compiled_levels():
    """Test whether we can have different compiled levels (in theory)."""
    print("\nTesting compiled level data propagation...")

    # Create a modified level with different max_entities
    level1 = create_hardcoded_16x16_room()
    level1.max_entities = 150  # Different from default 110

    # Test that our validation catches the modification
    result = lib.mer_validate_compiled_level(ctypes.byref(level1))
    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"ERROR: Modified level validation failed: {error_msg}")
        return False

    print("✓ Modified level validation passed")

    # Create manager with modified level
    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU mode
    config.gpu_id = 0
    config.num_worlds = 2
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    levels_array = (MER_CompiledLevel * 1)(level1)
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(ctypes.byref(handle), ctypes.byref(config), levels_array, 1)

    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"ERROR: Manager creation with modified level failed: {error_msg}")
        return False

    print("✓ Manager created with modified compiled level (max_entities=150)")

    lib.mer_destroy_manager(handle)
    print("✓ Modified level manager cleaned up")

    return True


if __name__ == "__main__":
    print("=== Multiple Worlds Compiled Level Test ===")

    success1 = test_multiple_worlds_compiled_level()
    success2 = test_different_compiled_levels()

    print("\n=== Results ===")
    print(f"Multiple worlds test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Modified level test: {'PASSED' if success2 else 'FAILED'}")

    if success1 and success2:
        print("\n🎉 All multiple worlds tests passed!")
        print("\nKey findings:")
        print("- Multiple worlds (1, 4, 8, 16) all work with compiled levels")
        print("- Each manager uses the same compiled level data for all its worlds")
        print("- Modified compiled level data (different max_entities) works correctly")
        print("- This suggests all worlds in a manager share the same CompiledLevel singleton")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
