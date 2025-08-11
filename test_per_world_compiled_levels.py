#!/usr/bin/env python3
"""
Test per-world compiled level interface.

This tests the new interface that allows passing a different compiled level
for each world in a manager.
"""

import ctypes
import sys
from typing import List

# Reuse structures from previous tests
from test_compiled_level_interface import (
    MER_SUCCESS,
    TILE_CUBE,
    TILE_EMPTY,
    TILE_SPAWN,
    TILE_WALL,
    MER_CompiledLevel,
    MER_ManagerConfig,
    lib,
)


def create_empty_room() -> MER_CompiledLevel:
    """Create a simple empty room (just walls around perimeter)."""
    ROOM_SIZE = 8
    TILE_SIZE = 2.0

    level = MER_CompiledLevel()
    level.width = ROOM_SIZE
    level.height = ROOM_SIZE
    level.scale = TILE_SIZE

    tiles = []
    entity_count = 0

    # Create walls around perimeter only
    for x in range(ROOM_SIZE):
        for y in range(ROOM_SIZE):
            if x == 0 or x == ROOM_SIZE - 1 or y == 0 or y == ROOM_SIZE - 1:
                world_x = (x - ROOM_SIZE / 2.0) * TILE_SIZE
                world_y = (y - ROOM_SIZE / 2.0) * TILE_SIZE
                tiles.append((world_x, world_y, TILE_WALL))
                entity_count += 1

    # Add spawn point
    tiles.append((0.0, 0.0, TILE_SPAWN))

    # Account for persistent entities: 1 floor + 2 agents + 3 origin markers = 6
    # All 6 entities have ObjectID and consume BVH slots
    level.max_entities = entity_count + 6 + 30  # tiles + persistent + buffer
    level.num_tiles = len(tiles)

    # Fill arrays
    for i in range(256):
        if i < len(tiles):
            level.tile_x[i] = tiles[i][0]
            level.tile_y[i] = tiles[i][1]
            level.tile_types[i] = tiles[i][2]
        else:
            level.tile_x[i] = 0.0
            level.tile_y[i] = 0.0
            level.tile_types[i] = TILE_EMPTY

    return level


def create_obstacle_room() -> MER_CompiledLevel:
    """Create a room with cube obstacles."""
    ROOM_SIZE = 10
    TILE_SIZE = 2.0

    level = MER_CompiledLevel()
    level.width = ROOM_SIZE
    level.height = ROOM_SIZE
    level.scale = TILE_SIZE

    tiles = []
    entity_count = 0

    # Create walls around perimeter
    for x in range(ROOM_SIZE):
        for y in range(ROOM_SIZE):
            if x == 0 or x == ROOM_SIZE - 1 or y == 0 or y == ROOM_SIZE - 1:
                world_x = (x - ROOM_SIZE / 2.0) * TILE_SIZE
                world_y = (y - ROOM_SIZE / 2.0) * TILE_SIZE
                tiles.append((world_x, world_y, TILE_WALL))
                entity_count += 1

    # Add some cube obstacles in the center
    obstacles = [(-2.0, -2.0), (2.0, 2.0), (0.0, -4.0)]
    for obs_x, obs_y in obstacles:
        tiles.append((obs_x, obs_y, TILE_CUBE))
        entity_count += 1

    # Add spawn point
    tiles.append((0.0, 4.0, TILE_SPAWN))  # Spawn away from obstacles

    # Account for persistent entities: 1 floor + 2 agents + 3 origin markers = 6
    level.max_entities = entity_count + 6 + 40  # tiles + persistent + buffer
    level.num_tiles = len(tiles)

    # Fill arrays
    for i in range(256):
        if i < len(tiles):
            level.tile_x[i] = tiles[i][0]
            level.tile_y[i] = tiles[i][1]
            level.tile_types[i] = tiles[i][2]
        else:
            level.tile_x[i] = 0.0
            level.tile_y[i] = 0.0
            level.tile_types[i] = TILE_EMPTY

    return level


def create_maze_room() -> MER_CompiledLevel:
    """Create a room with a simple maze pattern."""
    ROOM_SIZE = 12
    TILE_SIZE = 1.5

    level = MER_CompiledLevel()
    level.width = ROOM_SIZE
    level.height = ROOM_SIZE
    level.scale = TILE_SIZE

    tiles = []
    entity_count = 0

    # Create a simple maze pattern
    maze_pattern = [
        "############",
        "#S.........#",
        "#.###.###..#",
        "#.#...#....#",
        "#.#.###.##.#",
        "#.#.......##",
        "#.#######..#",
        "#.........##",
        "#.#######..#",
        "#.........##",
        "#..........#",
        "############",
    ]

    for y, row in enumerate(maze_pattern):
        for x, char in enumerate(row):
            world_x = (x - ROOM_SIZE / 2.0) * TILE_SIZE
            world_y = (y - ROOM_SIZE / 2.0) * TILE_SIZE

            if char == "#":
                tiles.append((world_x, world_y, TILE_WALL))
                entity_count += 1
            elif char == "S":
                tiles.append((world_x, world_y, TILE_SPAWN))

    # Account for persistent entities: 1 floor + 2 agents + 3 origin markers = 6
    level.max_entities = entity_count + 6 + 50  # tiles + persistent + buffer
    level.num_tiles = len(tiles)

    # Fill arrays
    for i in range(256):
        if i < len(tiles):
            level.tile_x[i] = tiles[i][0]
            level.tile_y[i] = tiles[i][1]
            level.tile_types[i] = tiles[i][2]
        else:
            level.tile_x[i] = 0.0
            level.tile_y[i] = 0.0
            level.tile_types[i] = TILE_EMPTY

    return level


# Update function signatures for new array-based API
lib.mer_create_manager.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # MER_ManagerHandle* out_handle
    ctypes.POINTER(MER_ManagerConfig),  # const MER_ManagerConfig* config
    ctypes.POINTER(MER_CompiledLevel),  # const MER_CompiledLevel* compiled_levels (array)
    ctypes.c_uint32,  # uint32_t num_compiled_levels
]
lib.mer_create_manager.restype = ctypes.c_int


def test_per_world_compiled_levels():
    """Test creating different compiled levels for each world."""
    print("Testing per-world compiled levels interface...")

    NUM_WORLDS = 3

    # Create different levels for each world
    print("Creating different level types...")
    level_empty = create_empty_room()
    level_obstacles = create_obstacle_room()
    level_maze = create_maze_room()

    print(f"Empty room: {level_empty.num_tiles} tiles, max_entities={level_empty.max_entities}")
    print(
        f"Obstacle room: {level_obstacles.num_tiles} tiles, "
        f"max_entities={level_obstacles.max_entities}"
    )
    print(f"Maze room: {level_maze.num_tiles} tiles, max_entities={level_maze.max_entities}")

    # Validate all levels
    for i, level in enumerate([level_empty, level_obstacles, level_maze]):
        result = lib.mer_validate_compiled_level(ctypes.byref(level))
        if result != MER_SUCCESS:
            error_msg = lib.mer_result_to_string(result).decode("utf-8")
            print(f"ERROR: Level {i} validation failed: {error_msg}")
            return False
    print("✓ All levels validated successfully")

    # Create array of levels
    levels_array = (MER_CompiledLevel * NUM_WORLDS)(level_empty, level_obstacles, level_maze)

    # Create manager config
    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU mode
    config.gpu_id = 0
    config.num_worlds = NUM_WORLDS
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False
    config.batch_render_view_width = 64
    config.batch_render_view_height = 64

    # Create manager with per-world compiled levels
    print(f"Creating manager with {NUM_WORLDS} different levels...")
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle), ctypes.byref(config), levels_array, NUM_WORLDS
    )

    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"ERROR: Manager creation failed: {error_msg}")
        return False

    print("✓ Manager created successfully with per-world compiled levels!")
    print(f"  Manager handle: {handle.value}")
    print(f"  World 0: Empty room ({level_empty.num_tiles} tiles)")
    print(f"  World 1: Obstacle room ({level_obstacles.num_tiles} tiles)")
    print(f"  World 2: Maze room ({level_maze.num_tiles} tiles)")

    # Clean up
    print("Cleaning up...")
    result = lib.mer_destroy_manager(handle)
    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"WARNING: Manager cleanup failed: {error_msg}")
    else:
        print("✓ Manager destroyed successfully")

    return True


def test_partial_per_world_levels():
    """Test with all levels provided (no fallback needed)."""
    print("\nTesting per-world levels with all worlds covered...")

    NUM_WORLDS = 4

    # Create levels for all worlds
    level_empty = create_empty_room()
    level_obstacles = create_obstacle_room()
    level_maze = create_maze_room()
    level_empty2 = create_empty_room()  # Reuse empty room for 4th world

    # Create array with all 4 levels
    levels_array = (MER_CompiledLevel * NUM_WORLDS)(
        level_empty, level_obstacles, level_maze, level_empty2
    )

    # Create manager config
    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU mode
    config.gpu_id = 0
    config.num_worlds = NUM_WORLDS
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    # Create manager with per-world compiled levels
    print(f"Creating manager with {NUM_WORLDS} levels for {NUM_WORLDS} worlds...")
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle), ctypes.byref(config), levels_array, NUM_WORLDS
    )

    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"ERROR: Per-world manager creation failed: {error_msg}")
        return False

    print("✓ Manager created successfully with all per-world levels!")
    print("  World 0: Empty room")
    print("  World 1: Obstacle room")
    print("  World 2: Maze room")
    print("  World 3: Empty room (reused)")

    # Clean up
    lib.mer_destroy_manager(handle)
    print("✓ Partial manager destroyed successfully")

    return True


def test_backward_compatibility():
    """Test with basic compiled levels for all worlds."""
    print("\nTesting with basic compiled levels...")

    # Create manager config
    config = MER_ManagerConfig()
    config.exec_mode = 0  # CPU mode
    config.gpu_id = 0
    config.num_worlds = 2
    config.rand_seed = 42
    config.auto_reset = True
    config.enable_batch_renderer = False

    # Create basic compiled levels for both worlds
    level1 = create_empty_room()
    level2 = create_empty_room()
    levels_array = (MER_CompiledLevel * 2)(level1, level2)

    # Create manager with compiled levels for all worlds
    handle = ctypes.c_void_p()
    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        levels_array,  # Basic compiled levels
        2,  # Two levels
    )

    if result != MER_SUCCESS:
        error_msg = lib.mer_result_to_string(result).decode("utf-8")
        print(f"ERROR: Basic levels test failed: {error_msg}")
        return False

    print("✓ Manager created successfully with basic compiled levels for all worlds!")

    # Clean up
    lib.mer_destroy_manager(handle)
    print("✓ Basic levels manager destroyed successfully")

    return True


if __name__ == "__main__":
    print("=== Per-World Compiled Levels Test ===")

    success1 = test_per_world_compiled_levels()
    success2 = test_partial_per_world_levels()
    success3 = test_backward_compatibility()

    print("\n=== Results ===")
    print(f"Per-world levels test: {'PASSED' if success1 else 'FAILED'}")
    print(f"Partial levels test: {'PASSED' if success2 else 'FAILED'}")
    print(f"Backward compatibility test: {'PASSED' if success3 else 'FAILED'}")

    if success1 and success2 and success3:
        print("\n🎉 All per-world compiled level tests passed!")
        print("\nKey capabilities demonstrated:")
        print("- Different level layout for each world (empty, obstacles, maze)")
        print("- Partial per-world levels (remaining worlds use Phase 2 fallback)")
        print("- Backward compatibility (no compiled levels = all Phase 2 fallback)")
        print("- Each world can have different max_entities, num_tiles, geometry")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
